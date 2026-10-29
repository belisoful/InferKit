//
//  NFKMLXStableFast3D.swift
//  InferKitMLX
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN

// Stable Fast 3D (Stability AI, 2024) reconstructs a textured object from one image in one forward. A
// DINOv2 ViT-L/14 whose every LayerNorm is modulated by an embedding of the camera tokenizes the image; a
// two-stream transformer turns a learned bank of 3×96×96 triplane tokens into three feature planes by
// trading attention with a set of latent tokens (the image tokens and 1792 learned ones) that also attend
// to the image; four 3×3 convolutions and a pixel shuffle raise the planes to 384×384 at 40 channels; and a
// set of small MLP heads reads density, a vertex offset, an albedo, and a perturbed normal from the three
// planes' bilinear samples at any point. An OpenCLIP ViT-B/32 reads the image again for the surface's
// roughness and metalness, and a convolutional head reads the planes for spherical-Gaussian light
// amplitudes. Marching tetrahedra over a deformed tetrahedral grid gives the mesh, a box projection
// unwraps it into a texture atlas, and the atlas is baked from the heads (`NFKMLXStableFast3DGeometry.swift`).
// The parameter names mirror the release checkpoint (`model.safetensors`, `stabilityai/stable-fast-3d`).
// Tensors flow NHWC.

/// The network geometry. Defaults are the released Stable Fast 3D.
public struct NFKMLXStableFast3DConfiguration: Sendable {
    /// The side of the conditioning image the tokenizer reads, in pixels.
    public var imageSize: Int = 512
    public var patchSize: Int = 14
    /// The side of the patch grid the released position embeddings were trained at (518 / 14).
    public var pretrainedPositionGrid: Int = 37
    public var dinoWidth: Int = 1024
    public var dinoDepth: Int = 24
    public var dinoHeads: Int = 16
    public var dinoHidden: Int = 4096
    /// The width of the camera embedding that modulates the tokenizer's LayerNorms.
    public var modulationWidth: Int = 768
    /// The camera condition's length: the 4×4 camera-to-world matrix and the normalized 3×3 intrinsics.
    public var cameraInputs: Int = 25
    /// The side of each triplane in tokens.
    public var planeSize: Int = 96
    public var tokenWidth: Int = 1024
    public var latentCount: Int = 1792
    public var backboneBlocks: Int = 4
    public var basicBlocks: Int = 3
    public var heads: Int = 16
    public var headDimensions: Int = 64
    public var normGroups: Int = 32
    public var upsampleLayers: Int = 4
    public var upsampleFactor: Int = 4
    /// Each upsampled plane's channel count; the heads read three planes' worth.
    public var planeChannels: Int = 40
    public var decoderNeurons: Int = 64
    /// The material estimator's image tower (the OpenCLIP ViT-B/32).
    public var clip: NFKMLXCLIPConfiguration = .base
    public var estimatorWidth: Int = 512
    public var estimatorHiddenLayers: Int = 3
    public var illuminationWidth: Int = 512
    public var illuminationLayers: Int = 2
    public var illuminationHiddenLayers: Int = 3
    /// The number of spherical-Gaussian lobes whose amplitudes the illumination head estimates.
    public var illuminationLobes: Int = 24
    /// Half the side of the cube the object lives in, in world units.
    public var radius: Float = 0.87
    /// The resolution the tetrahedral grid was generated at.
    public var gridResolution: Int = 160
    /// The density the surface is drawn at.
    public var threshold: Float = 10
    public var fieldOfViewDegrees: Double = 40
    public var cameraDistance: Float = 1.6
    /// The gray the conditioning image is composited on.
    public var background: Float = 0.5

    public init() {}

    /// The released geometry.
    public static var released: NFKMLXStableFast3DConfiguration { NFKMLXStableFast3DConfiguration() }

    var patchGrid: Int { imageSize / patchSize }
    var headInputs: Int { 3 * planeChannels }
    var innerWidth: Int { heads * headDimensions }

    /// The camera embedder's input: the default conditioning camera-to-world matrix and the intrinsics
    /// normalized by the side, 25 floats.
    var cameraCondition: [Float] {
        let d = cameraDistance
        let cameraToWorld: [Float] = [0, 0, 1, d, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        // `get_intrinsic_from_fov` computes the focal length in float64 and stores it in a float32 matrix,
        // which `create_intrinsic_from_fov_deg` then divides by the side in float32.
        let side = Float(imageSize)
        let fieldOfView = fieldOfViewDegrees * (Double.pi / 180)                             // `np.deg2rad`
        let focal = Float(0.5 * Double(imageSize) / tan(0.5 * fieldOfView))
        let center = side / 2
        return cameraToWorld + [focal / side, 0, center / side, 0, focal / side, center / side, 0, 0, 1]
    }
}

// MARK: - The modulated DINOv2 tokenizer (transformers `Dinov2Model` layout)

/// `normN_modulation`: a scale and a shift from the camera embedding, applied after a LayerNorm as
/// `x · (1 + scale) + shift`. The release's single-layer form has no first projection.
final class NFKSF3DModulation: Module {
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(width: Int, conditionWidth: Int) {
        _linear2.wrappedValue = Linear(conditionWidth, width * 2)
    }

    /// - Parameters:
    ///   - x: `[N, tokens, width]`.
    ///   - condition: `[N, conditionWidth]`.
    func callAsFunction(_ x: MLXArray, condition: MLXArray) -> MLXArray {
        let parts = linear2(silu(condition)).split(parts: 2, axis: -1)
        return x * (1 + parts[0].expandedDimensions(axis: 1)) + parts[1].expandedDimensions(axis: 1)
    }
}

final class NFKSF3DDinoSelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    let heads: Int

    init(width: Int, heads: Int) {
        self.heads = heads
        _query.wrappedValue = Linear(width, width)
        _key.wrappedValue = Linear(width, width)
        _value.wrappedValue = Linear(width, width)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, tokens, width) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = width / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([batch, tokens, heads, headDim]).transposed(0, 2, 1, 3) }
        let attended = MLXFast.scaledDotProductAttention(queries: split(query(x)), keys: split(key(x)), values: split(value(x)),
                                                         scale: 1 / sqrtf(Float(headDim)), mask: nil)
        return attended.transposed(0, 2, 1, 3).reshaped([batch, tokens, width])
    }
}

final class NFKSF3DDinoAttention: Module {
    @ModuleInfo(key: "attention") var attention: NFKSF3DDinoSelfAttention
    @ModuleInfo(key: "output") var output: NFKTripoViTSelfOutput
    init(width: Int, heads: Int) {
        _attention.wrappedValue = NFKSF3DDinoSelfAttention(width: width, heads: heads)
        _output.wrappedValue = NFKTripoViTSelfOutput(width: width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(attention(x)) }
}

/// `layer_scaleN.lambda1`: DINOv2's per-channel learned scale.
final class NFKSF3DLayerScale: Module {
    @ModuleInfo(key: "lambda1") var lambda1: MLXArray
    init(width: Int) { _lambda1.wrappedValue = MLXArray.ones([width]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * lambda1 }
}

final class NFKSF3DDinoMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(width: Int, hidden: Int) {
        _fc1.wrappedValue = Linear(width, hidden)
        _fc2.wrappedValue = Linear(hidden, width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(NFKReferenceRounding.gelu(fc1(x))) }
}

/// One DINOv2 layer with both LayerNorms modulated by the camera.
final class NFKSF3DDinoLayer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm1_modulation") var norm1Modulation: NFKSF3DModulation
    @ModuleInfo(key: "attention") var attention: NFKSF3DDinoAttention
    @ModuleInfo(key: "layer_scale1") var layerScale1: NFKSF3DLayerScale
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm2_modulation") var norm2Modulation: NFKSF3DModulation
    @ModuleInfo(key: "mlp") var mlp: NFKSF3DDinoMLP
    @ModuleInfo(key: "layer_scale2") var layerScale2: NFKSF3DLayerScale

    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        let width = configuration.dinoWidth
        // DINOv2 builds every LayerNorm with eps 1e-6.
        _norm1.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6)
        _norm1Modulation.wrappedValue = NFKSF3DModulation(width: width, conditionWidth: configuration.modulationWidth)
        _attention.wrappedValue = NFKSF3DDinoAttention(width: width, heads: configuration.dinoHeads)
        _layerScale1.wrappedValue = NFKSF3DLayerScale(width: width)
        _norm2.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6)
        _norm2Modulation.wrappedValue = NFKSF3DModulation(width: width, conditionWidth: configuration.modulationWidth)
        _mlp.wrappedValue = NFKSF3DDinoMLP(width: width, hidden: configuration.dinoHidden)
        _layerScale2.wrappedValue = NFKSF3DLayerScale(width: width)
    }

    func callAsFunction(_ x: MLXArray, condition: MLXArray) -> MLXArray {
        let attended = layerScale1(attention(norm1Modulation(norm1(x), condition: condition))) + x
        return layerScale2(mlp(norm2Modulation(norm2(attended), condition: condition))) + attended
    }
}

/// The class token, the position embeddings at the conditioning size, and the patch projection.
final class NFKSF3DDinoEmbeddings: Module {
    @ModuleInfo(key: "cls_token") var clsToken: MLXArray
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: MLXArray
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: NFKTripoViTPatchEmbeddings

    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        let width = configuration.dinoWidth
        let grid = configuration.patchGrid
        _clsToken.wrappedValue = MLXArray.zeros([1, 1, width])
        _positionEmbeddings.wrappedValue = MLXArray.zeros([1, grid * grid + 1, width])
        _patchEmbeddings.wrappedValue = NFKTripoViTPatchEmbeddings(patchSize: configuration.patchSize, width: width)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let patches = patchEmbeddings(x)                                   // [N, grid, grid, width]
        let (batch, grid, width) = (patches.dim(0), patches.dim(1), patches.dim(3))
        let tokens = patches.reshaped([batch, grid * grid, width])
        return concatenated([broadcast(clsToken, to: [batch, 1, width]), tokens], axis: 1) + positionEmbeddings
    }
}

final class NFKSF3DDinoEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [NFKSF3DDinoLayer]
    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        _layer.wrappedValue = (0 ..< configuration.dinoDepth).map { _ in NFKSF3DDinoLayer(configuration) }
    }
    func callAsFunction(_ x: MLXArray, condition: MLXArray) -> MLXArray {
        layer.reduce(x) { $1($0, condition: condition) }
    }
}

/// `image_tokenizer.model`: the modulated `Dinov2Model`, whose `last_hidden_state` (every token, class
/// token first) is what the backbone reads.
final class NFKSF3DDino: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKSF3DDinoEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKSF3DDinoEncoder
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm

    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        _embeddings.wrappedValue = NFKSF3DDinoEmbeddings(configuration)
        _encoder.wrappedValue = NFKSF3DDinoEncoder(configuration)
        _layernorm.wrappedValue = LayerNorm(dimensions: configuration.dinoWidth, eps: 1e-6)
    }

    /// - Parameters:
    ///   - images: `[N, imageSize, imageSize, 3]` in 0...1.
    ///   - condition: the camera embedding `[N, modulationWidth]`.
    /// - Returns: `[N, grid² + 1, dinoWidth]`.
    func callAsFunction(_ images: MLXArray, condition: MLXArray) -> MLXArray {
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225])
        return layernorm(encoder(embeddings((images - mean) / standardDeviation), condition: condition))
    }
}

final class NFKSF3DImageTokenizer: Module {
    @ModuleInfo(key: "model") var model: NFKSF3DDino
    init(_ configuration: NFKMLXStableFast3DConfiguration) { _model.wrappedValue = NFKSF3DDino(configuration) }
}

/// `camera_embedder.linear`: the camera condition to the modulation width.
final class NFKSF3DCameraEmbedder: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        _linear.wrappedValue = Linear(configuration.cameraInputs, configuration.modulationWidth)
    }
    func callAsFunction(_ condition: MLXArray) -> MLXArray { linear(condition) }
}

// MARK: - The two-stream backbone

/// `attn`, `attn1`, `attn2`: query, key, and value projections without bias, an output projection with one.
final class NFKSF3DAttention: Module {
    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    let heads: Int

    init(width: Int, contextWidth: Int, heads: Int) {
        self.heads = heads
        _wq.wrappedValue = Linear(width, width, bias: false)
        _wk.wrappedValue = Linear(contextWidth, width, bias: false)
        _wv.wrappedValue = Linear(contextWidth, width, bias: false)
        _proj.wrappedValue = Linear(width, width)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let (batch, queries, width) = (x.dim(0), x.dim(1), x.dim(2))
        let keys = context.dim(1)
        let headDim = width / heads
        let q = wq(x).reshaped([batch, queries, heads, headDim]).transposed(0, 2, 1, 3)
        let k = wk(context).reshaped([batch, keys, heads, headDim]).transposed(0, 2, 1, 3)
        let v = wv(context).reshaped([batch, keys, heads, headDim]).transposed(0, 2, 1, 3)
        // The fused kernel holds no queries × keys score matrix: the triplane stream's 27,648 queries over
        // 3,089 latents would take 5.5 GB at float32.
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / sqrtf(Float(headDim)), mask: nil)
        return proj(attended.transposed(0, 2, 1, 3).reshaped([batch, queries, width]))
    }
}

/// `BasicBlock`: self-attention, cross-attention to the image tokens, and the GEGLU feed-forward, each
/// after a LayerNorm and added back.
final class NFKSF3DBasicBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn1") var attn1: NFKSF3DAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn2") var attn2: NFKSF3DAttention
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "ff") var ff: NFKTripoFeedForward

    init(width: Int, contextWidth: Int, heads: Int) {
        _norm1.wrappedValue = LayerNorm(dimensions: width)
        _attn1.wrappedValue = NFKSF3DAttention(width: width, contextWidth: width, heads: heads)
        _norm2.wrappedValue = LayerNorm(dimensions: width)
        _attn2.wrappedValue = NFKSF3DAttention(width: width, contextWidth: contextWidth, heads: heads)
        _norm3.wrappedValue = LayerNorm(dimensions: width)
        _ff.wrappedValue = NFKTripoFeedForward(width: width)
    }

    func callAsFunction(_ z: MLXArray, context: MLXArray) -> MLXArray {
        let normed = norm1(z)
        var h = z + attn1(normed, context: normed)
        h = h + attn2(norm2(h), context: context)
        return h + ff(norm3(h))
    }
}

/// `FuseBlock`: one stream attends to the other, then the feed-forward. The release does not normalize the
/// attended stream (`norm_x_input: false`).
final class NFKSF3DFuseBlock: Module {
    @ModuleInfo(key: "attn") var attn: NFKSF3DAttention
    @ModuleInfo(key: "norm_z1") var normZ1: LayerNorm
    @ModuleInfo(key: "norm_z2") var normZ2: LayerNorm
    @ModuleInfo(key: "ff") var ff: NFKTripoFeedForward

    init(width: Int, contextWidth: Int, heads: Int) {
        _attn.wrappedValue = NFKSF3DAttention(width: width, contextWidth: contextWidth, heads: heads)
        _normZ1.wrappedValue = LayerNorm(dimensions: width)
        _normZ2.wrappedValue = LayerNorm(dimensions: width)
        _ff.wrappedValue = NFKTripoFeedForward(width: width)
    }

    func callAsFunction(_ z: MLXArray, context: MLXArray) -> MLXArray {
        let h = z + attn(normZ1(z), context: context)
        return h + ff(normZ2(h))
    }
}

/// `TwoStreamBlock`: the triplane tokens fused into the latents, the latents' basic blocks over the image
/// tokens, and the latents fused back into the triplane tokens.
final class NFKSF3DTwoStreamBlock: Module {
    @ModuleInfo(key: "fuse_block_in") var fuseIn: NFKSF3DFuseBlock
    @ModuleInfo(key: "transformer_block") var blocks: [NFKSF3DBasicBlock]
    @ModuleInfo(key: "fuse_block_out") var fuseOut: NFKSF3DFuseBlock

    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        let width = configuration.innerWidth
        _fuseIn.wrappedValue = NFKSF3DFuseBlock(width: width, contextWidth: configuration.tokenWidth, heads: configuration.heads)
        _blocks.wrappedValue = (0 ..< configuration.basicBlocks).map { _ in
            NFKSF3DBasicBlock(width: width, contextWidth: configuration.dinoWidth, heads: configuration.heads)
        }
        _fuseOut.wrappedValue = NFKSF3DFuseBlock(width: configuration.tokenWidth, contextWidth: width, heads: configuration.heads)
    }

    func callAsFunction(latents: MLXArray, triplane: MLXArray, image: MLXArray) -> (latents: MLXArray, triplane: MLXArray) {
        var h = fuseIn(latents, context: triplane)
        for block in blocks {
            h = block(h, context: image)
        }
        return (h, fuseOut(triplane, context: h))
    }
}

/// `backbone`: the `TwoStreamInterleaveTransformer`. The triplane tokens are group-normalized and
/// projected, the latents are the projected image tokens followed by the projected learned latents, the
/// blocks run, and the triplane stream is projected back and added to its input.
final class NFKSF3DBackbone: Module {
    @ModuleInfo(key: "norm_triplane") var normTriplane: GroupNorm
    @ModuleInfo(key: "proj_triplane") var projTriplane: Linear
    @ModuleInfo(key: "norm_image") var normImage: LayerNorm
    @ModuleInfo(key: "proj_image") var projImage: Linear
    @ModuleInfo(key: "norm_latent") var normLatent: LayerNorm
    @ModuleInfo(key: "proj_latent") var projLatent: Linear
    @ModuleInfo(key: "latent_init") var latentInit: MLXArray
    @ModuleInfo(key: "main_blocks") var blocks: [NFKSF3DTwoStreamBlock]
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        let width = configuration.tokenWidth
        let inner = configuration.innerWidth
        _normTriplane.wrappedValue = GroupNorm(groupCount: configuration.normGroups, dimensions: width, eps: 1e-6,
                                               affine: true, pytorchCompatible: true)
        _projTriplane.wrappedValue = Linear(width, width)
        _normImage.wrappedValue = LayerNorm(dimensions: configuration.dinoWidth)
        _projImage.wrappedValue = Linear(configuration.dinoWidth, inner)
        _normLatent.wrappedValue = LayerNorm(dimensions: inner)
        _projLatent.wrappedValue = Linear(inner, inner)
        _latentInit.wrappedValue = MLXArray.zeros([1, configuration.latentCount, inner])
        _blocks.wrappedValue = (0 ..< configuration.backboneBlocks).map { _ in NFKSF3DTwoStreamBlock(configuration) }
        _projOut.wrappedValue = Linear(width, width)
    }

    /// - Parameters:
    ///   - tokens: the triplane tokens `[N, 3 · planeSize², tokenWidth]`.
    ///   - image: the image tokens `[N, grid² + 1, dinoWidth]`.
    func callAsFunction(_ tokens: MLXArray, image: MLXArray) -> MLXArray {
        let batch = tokens.dim(0)
        // The reference normalizes [N, C, tokens] with GroupNorm over C; channels-last here, the groups
        // still run along the channel axis and the statistics over the group's channels and every token.
        var triplane = projTriplane(normTriplane(tokens))
        let initial = projLatent(normLatent(latentInit))
        var latents = concatenated([projImage(normImage(image)),
                                    broadcast(initial, to: [batch, initial.dim(1), initial.dim(2)])], axis: 1)
        for block in blocks {
            (latents, triplane) = block(latents: latents, triplane: triplane, image: image)
        }
        return projOut(triplane) + tokens
    }
}

/// `tokenizer.embeddings`: the learned triplane tokens, stored as the checkpoint stores them,
/// `[3, tokenWidth, planeSize, planeSize]`.
final class NFKSF3DTriplaneTokens: Module {
    @ModuleInfo(key: "embeddings") var embeddings: MLXArray
    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        _embeddings.wrappedValue = MLXArray.zeros([3, configuration.tokenWidth, configuration.planeSize, configuration.planeSize])
    }
    /// `[batch, 3 · planeSize², tokenWidth]`, plane-major then row-major, as the reference flattens them.
    func tokens(batch: Int) -> MLXArray {
        let (planes, width, side) = (embeddings.dim(0), embeddings.dim(1), embeddings.dim(2))
        let flat = embeddings.transposed(0, 2, 3, 1).reshaped([1, planes * side * side, width])
        return broadcast(flat, to: [batch, planes * side * side, width])
    }
}

/// `post_processor.upsample`: 3×3 convolutions with ReLU between them, the last to
/// `planeChannels · factor²`, then a pixel shuffle by `factor`.
final class NFKSF3DUpsampler: Module {
    @ModuleInfo(key: "upsample") var upsample: [Module]
    let factor: Int
    let channels: Int

    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        factor = configuration.upsampleFactor
        channels = configuration.planeChannels
        var layers = [Module]()
        let width = configuration.tokenWidth
        for index in 0 ..< configuration.upsampleLayers {
            let last = index == configuration.upsampleLayers - 1
            layers.append(Conv2d(inputChannels: width, outputChannels: last ? channels * factor * factor : width,
                                 kernelSize: 3, padding: 1))
            layers.append(NFKTripoPlaceholder())
        }
        _upsample.wrappedValue = layers
    }

    /// - Parameter planes: `[P, side, side, tokenWidth]`.
    /// - Returns: `[P, side · factor, side · factor, planeChannels]`.
    func callAsFunction(_ planes: MLXArray) -> MLXArray {
        let convolutions = upsample.compactMap { $0 as? Conv2d }
        var h = planes
        for (index, convolution) in convolutions.enumerated() {
            h = convolution(h)
            if index + 1 < convolutions.count {
                h = relu(h)
            }
        }
        // PixelShuffle: output channel c at sub-position (i, j) is input channel c·factor² + i·factor + j.
        let (count, side) = (h.dim(0), h.dim(1))
        return h.reshaped([count, side, side, channels, factor, factor])
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped([count, side * factor, side * factor, channels])
    }
}

// MARK: - The heads

/// The checkpoint's `nn.Sequential` heads: linear layers with parameter-free slots between them, so the
/// indices match the release's keys.
enum NFKSF3DLayers {

    /// `hidden` linear layers of `neurons` from `inputs`, a slot after each, then a projection to `outputs`.
    static func stack(inputs: Int, neurons: Int, hidden: Int, outputs: Int) -> [Module] {
        var sequence = [Module]()
        for index in 0 ..< hidden {
            sequence.append(Linear(index == 0 ? inputs : neurons, neurons))
            sequence.append(NFKTripoPlaceholder())
        }
        sequence.append(Linear(hidden == 0 ? inputs : neurons, outputs))
        return sequence
    }

    /// Applies each linear layer, `activation` after every one a slot follows.
    static func run(_ layers: [Module], _ x: MLXArray, activation: (MLXArray) -> MLXArray) -> MLXArray {
        var h = x
        for (index, module) in layers.enumerated() {
            guard let linear = module as? Linear else { continue }
            h = linear(h)
            if index + 1 < layers.count {
                h = activation(h)
            }
        }
        return h
    }
}

/// `decoder.heads`: density, the vertex offset, the albedo (`features`), and the perturbed normal, each
/// a SiLU stack.
final class NFKSF3DDecoderHeads: Module {
    @ModuleInfo(key: "density") var density: [Module]
    @ModuleInfo(key: "features") var features: [Module]
    @ModuleInfo(key: "perturb_normal") var perturbNormal: [Module]
    @ModuleInfo(key: "vertex_offset") var vertexOffset: [Module]

    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        let (inputs, neurons) = (configuration.headInputs, configuration.decoderNeurons)
        _density.wrappedValue = NFKSF3DLayers.stack(inputs: inputs, neurons: neurons, hidden: 2, outputs: 1)
        _features.wrappedValue = NFKSF3DLayers.stack(inputs: inputs, neurons: neurons, hidden: 3, outputs: 3)
        _perturbNormal.wrappedValue = NFKSF3DLayers.stack(inputs: inputs, neurons: neurons, hidden: 3, outputs: 3)
        _vertexOffset.wrappedValue = NFKSF3DLayers.stack(inputs: inputs, neurons: neurons, hidden: 2, outputs: 3)
    }
}

final class NFKSF3DDecoder: Module {
    @ModuleInfo(key: "heads") var heads: NFKSF3DDecoderHeads
    init(_ configuration: NFKMLXStableFast3DConfiguration) { _heads.wrappedValue = NFKSF3DDecoderHeads(configuration) }
}

/// One `ClipBasedHeadEstimator` head: a shared ReLU stack, then two branches that each give one
/// concentration of a beta distribution. The checkpoint holds the three as a `ModuleList` of
/// `nn.Sequential`s (`heads.roughness.0.4.weight`); a list nested in a list is a container the module
/// replacement that quantization and adapters go through cannot address, so they are held as `shared`,
/// `alpha`, and `beta`, and the loader renames the checkpoint's `.0.`, `.1.`, and `.2.` to them.
final class NFKSF3DBetaHead: Module {
    @ModuleInfo(key: "shared") var shared: [Module]
    @ModuleInfo(key: "alpha") var alpha: [Module]
    @ModuleInfo(key: "beta") var beta: [Module]

    init(width: Int, hidden: Int) {
        _shared.wrappedValue = (0 ..< hidden).flatMap { _ in [Linear(width, width), NFKTripoPlaceholder()] as [Module] }
        _alpha.wrappedValue = [Linear(width, width), NFKTripoPlaceholder(), Linear(width, 1)]
        _beta.wrappedValue = [Linear(width, width), NFKTripoPlaceholder(), Linear(width, 1)]
    }

    /// The beta distribution's mode `[N]` for features `[N, width]`: torch's `Beta.mode` with each
    /// concentration `softplus(branch + bias)`.
    func mode(_ features: MLXArray, bias: Float) -> MLXArray {
        // The shared stack is ReLU after every layer, its last one included.
        var h = features
        for module in shared {
            if let linear = module as? Linear {
                h = relu(linear(h))
            }
        }
        let a = NFKMLXStableFast3D.softplus(NFKSF3DLayers.run(alpha, h, activation: relu)[0..., 0] + bias)
        let b = NFKMLXStableFast3D.softplus(NFKSF3DLayers.run(beta, h, activation: relu)[0..., 0] + bias)
        return NFKMLXStableFast3D.betaMode(alpha: a, beta: b)
    }

    /// The module's key for a checkpoint key under a head, `0.N` → `shared.N`, `1.N` → `alpha.N`,
    /// `2.N` → `beta.N`, or the key itself.
    static func moduleKey(_ key: String) -> String {
        for material in ["roughness", "metallic"] {
            let prefix = "image_estimator.heads.\(material)."
            guard key.hasPrefix(prefix) else { continue }
            let rest = key.dropFirst(prefix.count)
            for (index, name) in ["shared", "alpha", "beta"].enumerated() where rest.hasPrefix("\(index).") {
                return prefix + name + "." + rest.dropFirst(2)
            }
        }
        return key
    }
}

final class NFKSF3DBetaHeads: Module {
    @ModuleInfo(key: "roughness") var roughness: NFKSF3DBetaHead
    @ModuleInfo(key: "metallic") var metallic: NFKSF3DBetaHead

    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        _roughness.wrappedValue = NFKSF3DBetaHead(width: configuration.estimatorWidth, hidden: configuration.estimatorHiddenLayers)
        _metallic.wrappedValue = NFKSF3DBetaHead(width: configuration.estimatorWidth, hidden: configuration.estimatorHiddenLayers)
    }
}

/// `image_estimator`: the OpenCLIP ViT-B/32 image tower (`model.visual`) and the roughness and metalness
/// heads. The checkpoint's text tower is never read and is not held.
final class NFKSF3DImageEstimatorModel: Module {
    @ModuleInfo(key: "visual") var visual: NFKCLIPVisionTransformer
    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        _visual.wrappedValue = NFKCLIPVisionTransformer(configuration.clip, activation: .gelu)
    }
}

final class NFKSF3DImageEstimator: Module {
    @ModuleInfo(key: "model") var model: NFKSF3DImageEstimatorModel
    @ModuleInfo(key: "heads") var heads: NFKSF3DBetaHeads
    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        _model.wrappedValue = NFKSF3DImageEstimatorModel(configuration)
        _heads.wrappedValue = NFKSF3DBetaHeads(configuration)
    }
}

final class NFKSF3DIlluminationHeads: Module {
    @ModuleInfo(key: "sg_amplitudes") var amplitudes: [Module]
    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        let width = configuration.illuminationWidth
        _amplitudes.wrappedValue = NFKSF3DLayers.stack(inputs: width, neurons: width, hidden: configuration.illuminationHiddenLayers,
                                                       outputs: configuration.illuminationLobes)
    }
}

/// `global_estimator`: stride-2 3×3 convolutions without padding over the three planes stacked along the
/// channels, a max over the plane, and the amplitude head.
final class NFKSF3DIlluminationEstimator: Module {
    @ModuleInfo(key: "layers") var layers: [Module]
    @ModuleInfo(key: "heads") var heads: NFKSF3DIlluminationHeads

    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        var sequence = [Module]()
        var inputs = configuration.tokenWidth * 3
        for _ in 0 ..< configuration.illuminationLayers {
            sequence.append(Conv2d(inputChannels: inputs, outputChannels: configuration.illuminationWidth, kernelSize: 3, stride: 2))
            sequence.append(NFKTripoPlaceholder())
            inputs = configuration.illuminationWidth
        }
        _layers.wrappedValue = sequence
        _heads.wrappedValue = NFKSF3DIlluminationHeads(configuration)
    }
}

// MARK: - The whole network

/// Stable Fast 3D: the camera embedder, the tokenizer, the triplane token bank, the backbone, the
/// upsampler, the heads, and the two estimators. Build one with
/// `NFKMLXStableFast3D.network(weightsURL:configuration:)`. Introduced in InferKit 0.4.0.
public final class NFKMLXStableFast3DNet: Module {
    @ModuleInfo(key: "camera_embedder") var cameraEmbedder: NFKSF3DCameraEmbedder
    @ModuleInfo(key: "image_tokenizer") var imageTokenizer: NFKSF3DImageTokenizer
    @ModuleInfo(key: "tokenizer") var tokenizer: NFKSF3DTriplaneTokens
    @ModuleInfo(key: "backbone") var backbone: NFKSF3DBackbone
    @ModuleInfo(key: "post_processor") var postProcessor: NFKSF3DUpsampler
    @ModuleInfo(key: "decoder") var decoder: NFKSF3DDecoder
    @ModuleInfo(key: "image_estimator") var imageEstimator: NFKSF3DImageEstimator
    @ModuleInfo(key: "global_estimator") var illuminationEstimator: NFKSF3DIlluminationEstimator

    public let configuration: NFKMLXStableFast3DConfiguration

    init(_ configuration: NFKMLXStableFast3DConfiguration) {
        self.configuration = configuration
        _cameraEmbedder.wrappedValue = NFKSF3DCameraEmbedder(configuration)
        _imageTokenizer.wrappedValue = NFKSF3DImageTokenizer(configuration)
        _tokenizer.wrappedValue = NFKSF3DTriplaneTokens(configuration)
        _backbone.wrappedValue = NFKSF3DBackbone(configuration)
        _postProcessor.wrappedValue = NFKSF3DUpsampler(configuration)
        _decoder.wrappedValue = NFKSF3DDecoder(configuration)
        _imageEstimator.wrappedValue = NFKSF3DImageEstimator(configuration)
        _illuminationEstimator.wrappedValue = NFKSF3DIlluminationEstimator(configuration)
    }

    /// The camera condition the release reconstructs from: the default camera-to-world matrix at
    /// `cameraDistance` along +x, looking back at the origin with +z up, flattened row-major, followed by
    /// the intrinsics of a `fieldOfViewDegrees` pinhole normalized by the image side, `[1, cameraInputs]`.
    public func cameraCondition() -> MLXArray {
        MLXArray(configuration.cameraCondition).reshaped([1, configuration.cameraInputs])
    }

    /// The camera embedding `[N, modulationWidth]` that modulates the tokenizer.
    public func cameraEmbedding(batch: Int = 1) -> MLXArray {
        let embedding = cameraEmbedder(cameraCondition())
        return broadcast(embedding, to: [batch, embedding.dim(1)])
    }

    /// The image tokens `[N, grid² + 1, dinoWidth]` from conditioning images `[N, imageSize, imageSize, 3]`
    /// in 0...1.
    public func imageTokens(_ images: MLXArray) -> MLXArray {
        imageTokenizer.model(images, condition: cameraEmbedding(batch: images.dim(0)))
    }

    /// The backbone's triplane tokens `[N, 3 · planeSize², tokenWidth]` from image tokens.
    public func triplaneTokens(_ imageTokens: MLXArray) -> MLXArray {
        backbone(tokenizer.tokens(batch: imageTokens.dim(0)), image: imageTokens)
    }

    /// The scene codes `[N, 3, planeSize · factor, planeSize · factor, planeChannels]` from triplane tokens.
    public func sceneCodes(_ triplaneTokens: MLXArray) -> MLXArray {
        let batch = triplaneTokens.dim(0)
        let side = configuration.planeSize
        let planes = triplaneTokens.reshaped([batch * 3, side, side, configuration.tokenWidth])   // plane-major token order
        let upsampled = postProcessor(planes)
        return upsampled.reshaped([batch, 3, upsampled.dim(1), upsampled.dim(2), upsampled.dim(3)])
    }

    /// The reference's `get_scene_codes`: the scene codes for conditioning images `[N, imageSize, imageSize, 3]`
    /// in 0...1.
    public func callAsFunction(_ images: MLXArray) -> MLXArray {
        sceneCodes(triplaneTokens(imageTokens(images)))
    }

    /// The three planes' bilinear samples at `positions` `[M, 3]` in world units from one scene code
    /// `[3, side, side, planeChannels]`, concatenated, `[M, 3 · planeChannels]`. The reference's
    /// `query_triplane`: each plane is read at the pair of coordinates it spans (xy, xz, yz) with
    /// `grid_sample(align_corners: true)` and zero padding.
    public func planeFeatures(_ positions: MLXArray, sceneCode: MLXArray) -> MLXArray {
        let radius = configuration.radius
        // `scale_tensor(positions, (-radius, radius), (-1, 1))`.
        let normalized = (positions - (-radius)) / (radius - (-radius)) * Float(2) + Float(-1)
        let side = sceneCode.dim(1)
        let channels = sceneCode.dim(3)
        let flat = sceneCode.reshaped([3, side * side, channels])
        let pairs: [(Int, Int)] = [(0, 1), (0, 2), (1, 2)]                               // (width axis, height axis)
        var samples = [MLXArray]()
        for (plane, pair) in pairs.enumerated() {
            let x = (normalized[0..., pair.0] + 1) / 2 * Float(side - 1)
            let y = (normalized[0..., pair.1] + 1) / 2 * Float(side - 1)
            samples.append(NFKMLXTripoSR.bilinearSample(flat[plane], sampleY: y, sampleX: x, height: side, width: side))
        }
        return concatenated(samples, axis: -1)
    }

    /// The density (after its exponential) and the vertex offset at `positions` `[M, 3]`.
    public func geometry(_ positions: MLXArray, sceneCode: MLXArray) -> (density: MLXArray, vertexOffset: MLXArray) {
        let features = planeFeatures(positions, sceneCode: sceneCode)
        let heads = decoder.heads
        // `trunc_exp` is an exponential in the forward; the truncation is its gradient's.
        let density = exp(NFKSF3DLayers.run(heads.density, features, activation: silu)[0..., 0] + Float(-1))
        return (density, NFKSF3DLayers.run(heads.vertexOffset, features, activation: silu))
    }

    /// The albedo (after its sigmoid) and the unit perturbed normal at `positions` `[M, 3]`.
    public func surface(_ positions: MLXArray, sceneCode: MLXArray) -> (albedo: MLXArray, normal: MLXArray) {
        let features = planeFeatures(positions, sceneCode: sceneCode)
        let heads = decoder.heads
        let albedo = sigmoid(NFKSF3DLayers.run(heads.features, features, activation: silu))
        let normal = NFKSF3DLayers.run(heads.perturbNormal, features, activation: silu)
        return (albedo, NFKMLXStableFast3D.normalized(normal, epsilon: 1e-7))
    }

    /// The roughness and metalness `[N]` the material estimator reads from the masked conditioning images
    /// `[N, imageSize, imageSize, 3]`: a bilinear resize to the tower's resolution, OpenAI's normalization,
    /// the image tower's projected class token, and each head's beta mode.
    public func materials(_ maskedImages: MLXArray) -> (roughness: MLXArray, metallic: MLXArray) {
        let tower = imageEstimator.model.visual
        let side = configuration.clip.imageResolution
        let rows = NFKMLXStableFast3D.bilinearMatrix(from: maskedImages.dim(1), to: side)
        let columns = NFKMLXStableFast3D.bilinearMatrix(from: maskedImages.dim(2), to: side)
        let resized = matmul(matmul(rows, maskedImages.transposed(0, 3, 1, 2)), columns.transposed(1, 0))
            .transposed(0, 2, 3, 1)                                                      // [N, side, side, 3]
        let mean = MLXArray([Float(0.48145466), 0.4578275, 0.40821073])
        let standardDeviation = MLXArray([Float(0.26862954), 0.26130258, 0.27577711])
        let features = tower.features((resized - mean) / standardDeviation)               // [N, embed]
        let heads = imageEstimator.heads
        return (heads.roughness.mode(features, bias: 1), heads.metallic.mode(features, bias: 1))
    }

    /// The spherical-Gaussian light amplitudes `[N, illuminationLobes]` the illumination head reads from the
    /// planes before the upsampler, `[N · 3, planeSize, planeSize, tokenWidth]` as `sceneCodes` reshapes them.
    public func illumination(_ triplaneTokens: MLXArray) -> MLXArray {
        let batch = triplaneTokens.dim(0)
        let side = configuration.planeSize
        let width = configuration.tokenWidth
        // `[N, 3, C, H, W] → [N, 3·C, H, W]`: channel index plane · C + c.
        var h = triplaneTokens.reshaped([batch, 3, side, side, width]).transposed(0, 2, 3, 1, 4)
            .reshaped([batch, side, side, 3 * width])
        for module in illuminationEstimator.layers {
            if let convolution = module as? Conv2d {
                h = relu(convolution(h))
            }
        }
        let pooled = h.max(axes: [1, 2])
        let amplitudes = NFKSF3DLayers.run(illuminationEstimator.heads.amplitudes, pooled, activation: relu)
        return NFKMLXStableFast3D.softplus(amplitudes + Float(1))
    }
}

extension NFKCLIPVisionTransformer {

    /// The image tower's projected class token for normalized images `[N, resolution, resolution, 3]`:
    /// OpenCLIP's `encode_image` without the normalization of the result.
    func features(_ normalized: MLXArray) -> MLXArray {
        let patches = conv1(normalized.asType(NFKReferenceRounding.parameterType(of: conv1)))
        let (batch, width) = (patches.dim(0), patches.dim(3))
        var x = patches.reshaped([batch, patches.dim(1) * patches.dim(2), width])
        x = concatenated([broadcast(classEmbedding.reshaped([1, 1, width]), to: [batch, 1, width]), x], axis: 1)
        x = lnPre(x + positionalEmbedding)
        x = transformer(x, mask: nil)
        return lnPost(x[0..., 0]).matmul(proj)
    }
}
