//
//  NFKMLXTripoSR.swift
//  InferKitMLX
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN

// TripoSR (Stability AI and Tripo, 2024) reconstructs an object from one image in a single forward: a
// DINO ViT-B/16 tokenizes the image; a 16-block transformer with cross-attention to those tokens turns a
// learned bank of 3×32×32 triplane tokens into three feature planes; a transposed convolution doubles
// them to 64×64 at 40 channels; and a 9-layer MLP reads density and color from the three planes'
// bilinear samples at any point. Marching cubes over a density grid gives the mesh, and the MLP's color
// at each vertex gives its vertex color. The parameter names mirror the release checkpoint (`model.ckpt`,
// `stabilityai/TripoSR`): `image_tokenizer.model.*` (transformers' `ViTModel` layout), `tokenizer.embeddings`,
// `backbone.*`, `post_processor.upsample`, and `decoder.layers.*`. Tensors flow NHWC.

/// The network geometry. Defaults are the released TripoSR.
public struct NFKMLXTripoSRConfiguration: Sendable {
    /// The side of the conditioning image the ViT reads, in pixels.
    public var imageSize: Int = 512
    public var patchSize: Int = 16
    /// The side of the patch grid the released position embeddings were trained at (224 / 16).
    public var pretrainedPositionGrid: Int = 14
    public var vitWidth: Int = 768
    public var vitDepth: Int = 12
    public var vitHeads: Int = 12
    public var vitHidden: Int = 3072
    /// The side of each triplane in tokens.
    public var planeSize: Int = 32
    public var tokenWidth: Int = 1024
    public var backboneDepth: Int = 16
    public var backboneHeads: Int = 16
    public var headDimensions: Int = 64
    public var normGroups: Int = 32
    /// Each upsampled plane's channel count; the decoder reads three planes' worth.
    public var planeChannels: Int = 40
    public var decoderNeurons: Int = 64
    public var decoderHiddenLayers: Int = 9
    /// Half the side of the cube the object lives in, in world units.
    public var radius: Float = 0.87
    public var densityBias: Float = -1

    public init() {}

    /// The released geometry.
    public static var released: NFKMLXTripoSRConfiguration { NFKMLXTripoSRConfiguration() }

    var patchGrid: Int { imageSize / patchSize }
    var decoderInputs: Int { 3 * planeChannels }
}

// MARK: - The DINO ViT-B/16 image tokenizer (transformers `ViTModel` layout)

final class NFKTripoViTSelfAttention: Module {
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
        let q = split(query(x)), k = split(key(x)), v = split(value(x))
        let scores = softmax(matmul(q, k.transposed(0, 1, 3, 2)) * (1 / sqrtf(Float(headDim))), axis: -1)
        return matmul(scores, v).transposed(0, 2, 1, 3).reshaped([batch, tokens, width])
    }
}

final class NFKTripoViTSelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    init(width: Int) { _dense.wrappedValue = Linear(width, width) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { dense(x) }
}

final class NFKTripoViTAttention: Module {
    @ModuleInfo(key: "attention") var attention: NFKTripoViTSelfAttention
    @ModuleInfo(key: "output") var output: NFKTripoViTSelfOutput
    init(width: Int, heads: Int) {
        _attention.wrappedValue = NFKTripoViTSelfAttention(width: width, heads: heads)
        _output.wrappedValue = NFKTripoViTSelfOutput(width: width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(attention(x)) }
}

final class NFKTripoViTDense: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    init(_ inputs: Int, _ outputs: Int) { _dense.wrappedValue = Linear(inputs, outputs) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { dense(x) }
}

final class NFKTripoViTLayer: Module {
    @ModuleInfo(key: "attention") var attention: NFKTripoViTAttention
    @ModuleInfo(key: "intermediate") var intermediate: NFKTripoViTDense
    @ModuleInfo(key: "output") var output: NFKTripoViTDense
    @ModuleInfo(key: "layernorm_before") var layernormBefore: LayerNorm
    @ModuleInfo(key: "layernorm_after") var layernormAfter: LayerNorm

    init(_ configuration: NFKMLXTripoSRConfiguration) {
        let width = configuration.vitWidth
        _attention.wrappedValue = NFKTripoViTAttention(width: width, heads: configuration.vitHeads)
        _intermediate.wrappedValue = NFKTripoViTDense(width, configuration.vitHidden)
        _output.wrappedValue = NFKTripoViTDense(configuration.vitHidden, width)
        // transformers' ViT builds its LayerNorms with the BERT epsilon, 1e-12.
        _layernormBefore.wrappedValue = LayerNorm(dimensions: width, eps: 1e-12)
        _layernormAfter.wrappedValue = LayerNorm(dimensions: width, eps: 1e-12)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attended = x + attention(layernormBefore(x))
        return attended + output(NFKReferenceRounding.gelu(intermediate(layernormAfter(attended))))
    }
}

final class NFKTripoViTPatchEmbeddings: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d
    init(patchSize: Int, width: Int) {
        _projection.wrappedValue = Conv2d(inputChannels: 3, outputChannels: width,
                                          kernelSize: IntOrPair(patchSize), stride: IntOrPair(patchSize))
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(x) }
}

/// The class token, the position embeddings at the conditioning size, and the patch projection.
final class NFKTripoViTEmbeddings: Module {
    @ModuleInfo(key: "cls_token") var clsToken: MLXArray
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: MLXArray
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: NFKTripoViTPatchEmbeddings

    init(_ configuration: NFKMLXTripoSRConfiguration) {
        let width = configuration.vitWidth
        let grid = configuration.patchGrid
        _clsToken.wrappedValue = MLXArray.zeros([1, 1, width])
        _positionEmbeddings.wrappedValue = MLXArray.zeros([1, grid * grid + 1, width])
        _patchEmbeddings.wrappedValue = NFKTripoViTPatchEmbeddings(patchSize: configuration.patchSize, width: width)
    }

    /// - Parameter x: `[N, imageSize, imageSize, 3]`, normalized.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let patches = patchEmbeddings(x)                                   // [N, grid, grid, width]
        let (batch, grid, width) = (patches.dim(0), patches.dim(1), patches.dim(3))
        let tokens = patches.reshaped([batch, grid * grid, width])
        return concatenated([broadcast(clsToken, to: [batch, 1, width]), tokens], axis: 1) + positionEmbeddings
    }
}

final class NFKTripoViTEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [NFKTripoViTLayer]
    init(_ configuration: NFKMLXTripoSRConfiguration) {
        _layer.wrappedValue = (0 ..< configuration.vitDepth).map { _ in NFKTripoViTLayer(configuration) }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { layer.reduce(x) { $1($0) } }
}

/// `image_tokenizer.model`: a `ViTModel` whose `last_hidden_state` (every token, class token first) is
/// the conditioning the backbone attends to. The pooler is not part of the forward and is not held.
final class NFKTripoImageTokenizer: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKTripoViTEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKTripoViTEncoder
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm

    init(_ configuration: NFKMLXTripoSRConfiguration) {
        _embeddings.wrappedValue = NFKTripoViTEmbeddings(configuration)
        _encoder.wrappedValue = NFKTripoViTEncoder(configuration)
        _layernorm.wrappedValue = LayerNorm(dimensions: configuration.vitWidth, eps: 1e-12)
    }

    /// - Parameter images: `[N, imageSize, imageSize, 3]` in 0...1.
    /// - Returns: `[N, grid² + 1, vitWidth]`.
    func callAsFunction(_ images: MLXArray) -> MLXArray {
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225])
        return layernorm(encoder(embeddings((images - mean) / standardDeviation)))
    }
}

// MARK: - The triplane backbone (diffusers-style blocks)

/// `attn1` / `attn2`: projections without bias, an output projection with one, scaled dot-product attention.
final class NFKTripoAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    let heads: Int
    let headDimensions: Int

    init(width: Int, contextWidth: Int, heads: Int, headDimensions: Int) {
        self.heads = heads
        self.headDimensions = headDimensions
        let inner = heads * headDimensions
        _toQ.wrappedValue = Linear(width, inner, bias: false)
        _toK.wrappedValue = Linear(contextWidth, inner, bias: false)
        _toV.wrappedValue = Linear(contextWidth, inner, bias: false)
        _toOut.wrappedValue = [Linear(inner, width)]
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil) -> MLXArray {
        let source = context ?? x
        let (batch, queries) = (x.dim(0), x.dim(1))
        let keys = source.dim(1)
        let q = toQ(x).reshaped([batch, queries, heads, headDimensions]).transposed(0, 2, 1, 3)
        let k = toK(source).reshaped([batch, keys, heads, headDimensions]).transposed(0, 2, 1, 3)
        let v = toV(source).reshaped([batch, keys, heads, headDimensions]).transposed(0, 2, 1, 3)
        let scores = softmax(matmul(q, k.transposed(0, 1, 3, 2)) * (1 / sqrtf(Float(headDimensions))), axis: -1)
        let merged = matmul(scores, v).transposed(0, 2, 1, 3).reshaped([batch, queries, heads * headDimensions])
        return toOut[0](merged)
    }
}

/// `ff.net.0`: one projection to twice the hidden width, half gating the other through an exact GELU.
final class NFKTripoGEGLU: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    init(width: Int, hidden: Int) { _proj.wrappedValue = Linear(width, hidden * 2) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = proj(x).split(parts: 2, axis: -1)
        return parts[0] * NFKReferenceRounding.gelu(parts[1])
    }
}

/// A parameter-free slot in a `nn.Sequential`, so the surrounding indices match the checkpoint's.
final class NFKTripoPlaceholder: Module {}

/// `ff`: GEGLU, (dropout), projection back. Held as `net` so the keys read `ff.net.0.proj` and `ff.net.2`.
final class NFKTripoFeedForward: Module {
    @ModuleInfo(key: "net") var net: [Module]
    init(width: Int) {
        _net.wrappedValue = [NFKTripoGEGLU(width: width, hidden: width * 4), NFKTripoPlaceholder(), Linear(width * 4, width)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gated = (net[0] as! NFKTripoGEGLU)(x)
        return (net[2] as! Linear)(gated)
    }
}

final class NFKTripoTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn1") var attn1: NFKTripoAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn2") var attn2: NFKTripoAttention
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "ff") var ff: NFKTripoFeedForward

    init(_ configuration: NFKMLXTripoSRConfiguration) {
        let width = configuration.tokenWidth
        _norm1.wrappedValue = LayerNorm(dimensions: width)
        _attn1.wrappedValue = NFKTripoAttention(width: width, contextWidth: width, heads: configuration.backboneHeads,
                                                headDimensions: configuration.headDimensions)
        _norm2.wrappedValue = LayerNorm(dimensions: width)
        _attn2.wrappedValue = NFKTripoAttention(width: width, contextWidth: configuration.vitWidth,
                                                heads: configuration.backboneHeads, headDimensions: configuration.headDimensions)
        _norm3.wrappedValue = LayerNorm(dimensions: width)
        _ff.wrappedValue = NFKTripoFeedForward(width: width)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        var h = x + attn1(norm1(x))
        h = h + attn2(norm2(h), context: context)
        return h + ff(norm3(h))
    }
}

/// `backbone`: GroupNorm over the token channels, a projection in, the blocks, a projection out, and the
/// residual from before the norm.
final class NFKTripoBackbone: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "transformer_blocks") var blocks: [NFKTripoTransformerBlock]
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(_ configuration: NFKMLXTripoSRConfiguration) {
        let width = configuration.tokenWidth
        _norm.wrappedValue = GroupNorm(groupCount: configuration.normGroups, dimensions: width, eps: 1e-6,
                                       affine: true, pytorchCompatible: true)
        _projIn.wrappedValue = Linear(width, width)
        _blocks.wrappedValue = (0 ..< configuration.backboneDepth).map { _ in NFKTripoTransformerBlock(configuration) }
        _projOut.wrappedValue = Linear(width, width)
    }

    /// - Parameters:
    ///   - tokens: `[N, 3 · planeSize², tokenWidth]`.
    ///   - context: the image tokens `[N, grid² + 1, vitWidth]`.
    func callAsFunction(_ tokens: MLXArray, context: MLXArray) -> MLXArray {
        // The reference normalizes [N, C, tokens] with GroupNorm over C; channels-last here, the groups
        // still run along the channel axis and the statistics over the group's channels and every token.
        var h = projIn(norm(tokens))
        for block in blocks {
            h = block(h, context: context)
        }
        return projOut(h) + tokens
    }
}

/// `post_processor.upsample`: a stride-2 transposed convolution from the token width to the plane channels.
final class NFKTripoUpsampler: Module {
    @ModuleInfo(key: "upsample") var upsample: ConvTransposed2d
    init(_ configuration: NFKMLXTripoSRConfiguration) {
        _upsample.wrappedValue = ConvTransposed2d(inputChannels: configuration.tokenWidth,
                                                  outputChannels: configuration.planeChannels, kernelSize: 2, stride: 2)
    }
    func callAsFunction(_ planes: MLXArray) -> MLXArray { upsample(planes) }
}

/// `decoder`: the NeRF MLP as the checkpoint's `layers` sequence, SiLU after every hidden projection, four
/// outputs (density, then three color features).
final class NFKTripoDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    init(_ configuration: NFKMLXTripoSRConfiguration) {
        var sequence: [Module] = [Linear(configuration.decoderInputs, configuration.decoderNeurons), NFKTripoPlaceholder()]
        for _ in 0 ..< (configuration.decoderHiddenLayers - 1) {
            sequence.append(Linear(configuration.decoderNeurons, configuration.decoderNeurons))
            sequence.append(NFKTripoPlaceholder())
        }
        sequence.append(Linear(configuration.decoderNeurons, 4))
        _layers.wrappedValue = sequence
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var h = features
        for (index, module) in layers.enumerated() {
            guard let linear = module as? Linear else { continue }
            h = linear(h)
            if index + 1 < layers.count {
                h = silu(h)
            }
        }
        return h
    }
}

// MARK: - The whole network

/// TripoSR: the image tokenizer, the triplane token bank, the backbone, the upsampler, and the decoder.
/// Build one with `NFKMLXTripoSR.network(weightsURL:configuration:)`. Introduced in InferKit 0.4.0.
public final class NFKMLXTripoSRNet: Module {
    @ModuleInfo(key: "image_tokenizer") var imageTokenizer: NFKTripoImageTokenizerHolder
    @ModuleInfo(key: "tokenizer") var tokenizer: NFKTripoTriplaneTokens
    @ModuleInfo(key: "backbone") var backbone: NFKTripoBackbone
    @ModuleInfo(key: "post_processor") var postProcessor: NFKTripoUpsampler
    @ModuleInfo(key: "decoder") var decoder: NFKTripoDecoder

    public let configuration: NFKMLXTripoSRConfiguration

    init(_ configuration: NFKMLXTripoSRConfiguration) {
        self.configuration = configuration
        _imageTokenizer.wrappedValue = NFKTripoImageTokenizerHolder(configuration)
        _tokenizer.wrappedValue = NFKTripoTriplaneTokens(configuration)
        _backbone.wrappedValue = NFKTripoBackbone(configuration)
        _postProcessor.wrappedValue = NFKTripoUpsampler(configuration)
        _decoder.wrappedValue = NFKTripoDecoder(configuration)
    }

    /// The image tokens the backbone attends to, `[N, grid² + 1, vitWidth]`, from images
    /// `[N, imageSize, imageSize, 3]` in 0...1.
    public func imageTokens(_ images: MLXArray) -> MLXArray {
        imageTokenizer.model(images)
    }

    /// The reference's `forward`: the upsampled triplanes `[N, 3, 2 · planeSize, 2 · planeSize, planeChannels]`
    /// (the scene codes) for images `[N, imageSize, imageSize, 3]` in 0...1.
    public func callAsFunction(_ images: MLXArray) -> MLXArray {
        sceneCodes(imageTokens(images))
    }

    /// The scene codes from image tokens, so a caller can hold the tokens as a seam.
    public func sceneCodes(_ imageTokens: MLXArray) -> MLXArray {
        let batch = imageTokens.dim(0)
        let tokens = backbone(tokenizer.tokens(batch: batch), context: imageTokens)   // [N, 3·S², C]
        let side = configuration.planeSize
        let planes = tokens.reshaped([batch * 3, side, side, configuration.tokenWidth])  // plane-major token order
        let upsampled = postProcessor(planes)                                            // [N·3, 2S, 2S, planeChannels]
        return upsampled.reshaped([batch, 3, upsampled.dim(1), upsampled.dim(2), upsampled.dim(3)])
    }

    /// The density (after its exponential) and the color (after its sigmoid) at `positions` `[M, 3]` in
    /// world units, read from one scene code `[3, side, side, planeChannels]`. The reference's
    /// `query_triplane`: each plane is sampled bilinearly with half-pixel centers and zero padding at the
    /// pair of coordinates it spans (xy, xz, yz), the three samples are concatenated, and the decoder reads
    /// them.
    public func query(_ positions: MLXArray, sceneCode: MLXArray) -> (density: MLXArray, color: MLXArray) {
        let normalized = positions / configuration.radius                                // (-1, 1)
        let side = sceneCode.dim(1)
        let channels = sceneCode.dim(3)
        let flat = sceneCode.reshaped([3, side * side, channels])
        let pairs: [(Int, Int)] = [(0, 1), (0, 2), (1, 2)]                               // (width axis, height axis)
        var samples = [MLXArray]()
        for (plane, pair) in pairs.enumerated() {
            let x = (normalized[0..., pair.0] + 1) * Float(side) / 2 - 0.5
            let y = (normalized[0..., pair.1] + 1) * Float(side) / 2 - 0.5
            samples.append(NFKMLXTripoSR.bilinearSample(flat[plane], sampleY: y, sampleX: x, height: side, width: side))
        }
        let out = decoder(concatenated(samples, axis: -1))                                // [M, 4]
        let density = exp(out[0..., 0] + configuration.densityBias)
        let color = sigmoid(out[0..., 1 ..< 4])
        return (density, color)
    }
}

/// `image_tokenizer`: the module that holds the ViT as `model`, so the keys read `image_tokenizer.model.*`.
final class NFKTripoImageTokenizerHolder: Module {
    @ModuleInfo(key: "model") var model: NFKTripoImageTokenizer
    init(_ configuration: NFKMLXTripoSRConfiguration) { _model.wrappedValue = NFKTripoImageTokenizer(configuration) }
}

/// `tokenizer.embeddings`: the learned triplane tokens, stored as the checkpoint stores them,
/// `[3, tokenWidth, planeSize, planeSize]`.
final class NFKTripoTriplaneTokens: Module {
    @ModuleInfo(key: "embeddings") var embeddings: MLXArray
    init(_ configuration: NFKMLXTripoSRConfiguration) {
        _embeddings.wrappedValue = MLXArray.zeros([3, configuration.tokenWidth, configuration.planeSize, configuration.planeSize])
    }
    /// `[batch, 3 · planeSize², tokenWidth]`, plane-major then row-major, as the reference flattens them.
    func tokens(batch: Int) -> MLXArray {
        let (planes, width, side) = (embeddings.dim(0), embeddings.dim(1), embeddings.dim(2))
        let flat = embeddings.transposed(0, 2, 3, 1).reshaped([1, planes * side * side, width])
        return broadcast(flat, to: [batch, planes * side * side, width])
    }
}

// MARK: - Request options

/// The request-parameter keys the TripoSR backend reads. Introduced in InferKit 0.4.0.
@objc(NFKMLXTripoSRParameterKey)
public final class NFKMLXTripoSRParameterKey: NSObject {
    /// The side of the marching-cubes grid (NSNumber, default 256). Memory and time grow with its cube.
    @objc public static let resolution = "NFKMLXParameterMeshResolution"
    /// The density the surface is drawn at (NSNumber, default 25).
    @objc public static let threshold = "NFKMLXParameterDensityThreshold"
    /// The share of the image's side the object is scaled to fill when the image carries alpha (NSNumber in
    /// 0...1, default 0.85). An image without alpha is used as it is.
    @objc public static let foregroundRatio = "NFKMLXParameterForegroundRatio"
    /// Whether each vertex is colored by the decoder (NSNumber boolean, default YES).
    @objc public static let vertexColors = "NFKMLXParameterVertexColors"
}

/// The options a run takes.
public struct NFKMLXTripoSROptions: Sendable {
    public var resolution: Int = 256
    public var threshold: Float = 25
    public var foregroundRatio: Float = 0.85
    public var vertexColors: Bool = true
    public init() {}
    public static let `default` = NFKMLXTripoSROptions()
}

// MARK: - Backend

private final class NFKTripoHolder: @unchecked Sendable {
    let net: NFKMLXTripoSRNet
    init(_ net: NFKMLXTripoSRNet) { self.net = net }
}

/// TripoSR as an InferKit backend: an image under `NFKInputImage` in, an `NFKMesh` under `NFKOutputMesh` out.
@objc(NFKMLXTripoSRBackend)
public final class NFKMLXTripoSRBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKTripoHolder
    private let identifier: String

    init(net: NFKMLXTripoSRNet, identifier: String) {
        holder = NFKTripoHolder(net)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc public var supportedParameterKeys: Set<String> {
        [NFKMLXTripoSRParameterKey.resolution, NFKMLXTripoSRParameterKey.threshold,
         NFKMLXTripoSRParameterKey.foregroundRatio, NFKMLXTripoSRParameterKey.vertexColors]
    }
    @objc public var supportedInputKeys: Set<String> { [NFKInputImage] }
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputMesh] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let value = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedInput
        }
        let options = Self.options(from: request)
        let rgba = try NFKMLXImageBridge.tensor(from: value, channels: 4, colorSpace: CGColorSpaceCreateDeviceRGB())
        let mesh = try holder.net.reconstruct(rgba: rgba, options: options)
        return NFKInferenceResult(outputs: [NFKOutputMesh: mesh])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
            do {
                job.finish(with: try self.runInference(for: request))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    static func options(from request: NFKInferenceRequest) -> NFKMLXTripoSROptions {
        var options = NFKMLXTripoSROptions.default
        func number(_ key: String) -> NSNumber? { request.parameter(forKey: key) as? NSNumber }
        if let value = number(NFKMLXTripoSRParameterKey.resolution) { options.resolution = max(2, value.intValue) }
        if let value = number(NFKMLXTripoSRParameterKey.threshold) { options.threshold = value.floatValue }
        if let value = number(NFKMLXTripoSRParameterKey.foregroundRatio) { options.foregroundRatio = min(1, max(0.05, value.floatValue)) }
        if let value = number(NFKMLXTripoSRParameterKey.vertexColors) { options.vertexColors = value.boolValue }
        return options
    }
}

extension NFKMLXTripoSRNet {

    /// The whole run from a bridged image `[H, W, 4]` in 0...1: the reference demo's preparation (an image
    /// with alpha is cropped to its foreground, padded to `foregroundRatio` of a square, and composited on
    /// mid-gray), the antialiased bilinear resize to the conditioning size, the forward, the density grid,
    /// marching cubes, and the vertex colors.
    func reconstruct(rgba: MLXArray, options: NFKMLXTripoSROptions) throws -> NFKMesh {
        let prepared = NFKMLXTripoSR.prepare(rgba: rgba, foregroundRatio: options.foregroundRatio)
        let conditioning = NFKMLXTripoSR.resizeAntialiased(prepared, side: configuration.imageSize)
        let sceneCode = self(conditioning.reshaped([1] + conditioning.shape))[0]
        eval(sceneCode)
        return try mesh(sceneCode: sceneCode, options: options)
    }

    /// The mesh of one scene code: density on a `resolution³` grid over the object's cube, marching cubes at
    /// `threshold`, and the decoder's color at each vertex when asked.
    func mesh(sceneCode: MLXArray, options: NFKMLXTripoSROptions) throws -> NFKMesh {
        let resolution = options.resolution
        let density = densityGrid(sceneCode: sceneCode, resolution: resolution)
        let surface = NFKMarchingCubes.surface(of: density, resolution: resolution, threshold: options.threshold)
        let radius = configuration.radius
        var positions = surface.vertices
        let span = 2 * radius / Float(resolution - 1)
        for index in positions.indices {
            positions[index] = positions[index] * span - radius
        }
        var colors: Data? = nil
        if options.vertexColors && !positions.isEmpty {
            let points = MLXArray(positions).reshaped([positions.count / 3, 3])
            let values = NFKMLXTripoSR.chunked(points, size: 1 << 16) { chunk in self.query(chunk, sceneCode: sceneCode).color }
            colors = values.asArray(Float.self).withUnsafeBufferPointer { Data(buffer: $0) }
        }
        let positionData = positions.withUnsafeBufferPointer { Data(buffer: $0) }
        let indexData = surface.indices.withUnsafeBufferPointer { Data(buffer: $0) }
        return try NFKMesh(positions: positionData, normals: nil, colors: colors, textureCoordinates: nil, indices: indexData)
    }

    /// The decoder's density at every vertex of a `resolution³` grid over `[-radius, radius]³`, flattened
    /// x-major (x, then y, then z), as the reference's `meshgrid(indexing: "ij")` lays it out.
    func densityGrid(sceneCode: MLXArray, resolution: Int) -> [Float] {
        let radius = configuration.radius
        let axis = (0 ..< resolution).map { -radius + 2 * radius * Float($0) / Float(resolution - 1) }
        let x = MLXArray(axis).reshaped([resolution, 1, 1])
        let y = MLXArray(axis).reshaped([1, resolution, 1])
        let z = MLXArray(axis).reshaped([1, 1, resolution])
        let shape = [resolution, resolution, resolution]
        let points = stacked([broadcast(x, to: shape), broadcast(y, to: shape), broadcast(z, to: shape)], axis: -1)
            .reshaped([resolution * resolution * resolution, 3])
        let density = NFKMLXTripoSR.chunked(points, size: 1 << 18) { chunk in self.query(chunk, sceneCode: sceneCode).density }
        return density.asArray(Float.self)
    }
}

// MARK: - Factories and loading

/// TripoSR as an InferKit backend, and its registration for the Objective-C path.
@objc(NFKMLXTripoSR)
public final class NFKMLXTripoSR: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "triposr"

    static func makeNet(_ configuration: NFKMLXTripoSRConfiguration = .released) -> NFKMLXTripoSRNet {
        NFKMLXTripoSRNet(configuration)
    }

    /// The network at `configuration`, with the checkpoint at `weightsURL` loaded. Nil weights is the random
    /// initialization. A `dtype` converts every floating tensor as it loads (the release is float32;
    /// `.bfloat16` halves it); `quantizationBits` then packs every linear layer whose input width is a
    /// multiple of `groupSize` (the decoder's first layer, 120 wide, stays as loaded). Introduced in InferKit 0.4.0.
    public static func network(weightsURL: URL?, configuration: NFKMLXTripoSRConfiguration = .released,
                               castingTo dtype: DType? = nil, quantizationBits: Int? = nil,
                               groupSize: Int = NFKMLXLoadOptions.defaultQuantizationGroupSize) throws -> NFKMLXTripoSRNet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL, castingTo: dtype)
        } else if let dtype {
            net.update(parameters: net.parameters().mapValues { $0.asType(dtype) })
        }
        if let bits = quantizationBits {
            guard (2 ... 8).contains(bits) else {
                throw NFKMLXError.unsupportedConfiguration("a packed load takes 2 to 8 bits, not \(bits)")
            }
            // MLX's quantizer aborts the process on any other group size.
            guard [32, 64, 128].contains(groupSize) else {
                throw NFKMLXError.unsupportedConfiguration("a packed load takes a group size of 32, 64, or 128, not \(groupSize)")
            }
            try NFKMLXQuantization.quantize(module: net, bits: bits, groupSize: groupSize)
        }
        return net
    }

    /// Builds a TripoSR backend from optional local weights (the release's `model.ckpt`, or a safetensors
    /// file). A nil `weightsURL` builds random weights (`isReady` is true). Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        NFKMLXTripoSRBackend(net: try network(weightsURL: weightsURL), identifier: modelName)
    }

    /// Downloads the checkpoint from Hugging Face, then builds. Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
            return try backend(weightsURL: url)
        }
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers TripoSR (`triposr`) with `NFKMLXModelRegistry`. `NFKModelLoadOptionPrecision` `bfloat16`
    /// converts the float32 release as it loads and `checkpoint` keeps it; `NFKModelLoadOptionQuantizationBits`
    /// packs the linear layers.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName, consumes: [NFKModelLoadOptionPrecision, NFKModelLoadOptionQuantizationBits,
                                                                NFKModelLoadOptionQuantizationGroupSize]) { weightsURL, options in
            let net = try network(weightsURL: weightsURL, castingTo: try options.bfloat16Cast(of: modelName),
                                  quantizationBits: options.quantizationBits, groupSize: options.quantizationGroupSize)
            return NFKMLXTripoSRBackend(net: net, identifier: modelName)
        }
    }

    /// Loads the release checkpoint or a safetensors file into `net`.
    ///
    /// The release stores the ViT's position embeddings at its 224-pixel training grid (14 × 14 patches
    /// plus the class token); the forward reads the conditioning size, so they are resized to its grid
    /// here with the bicubic interpolation `ViTModel` applies at run time (`interpolate_pos_encoding`). The
    /// pooler, which the forward never reaches, is dropped. Convolution weights move from PyTorch's layout
    /// to MLX's; the upsampler is a transposed convolution, whose PyTorch weight is `[in, out, kH, kW]`.
    public static func loadWeights(into net: NFKMLXTripoSRNet, from url: URL, castingTo dtype: DType? = nil) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        var arrays = mapped(checkpoint.arrays, needsConvTranspose: checkpoint.needsConvTranspose, for: net)
        if let dtype {
            arrays = arrays.map { key, value in
                ([.float32, .float16, .bfloat16].contains(value.dtype) ? (key, value.asType(dtype)) : (key, value))
            }
        }
        try NFKMLXWeights.apply(arrays, to: net)
    }

    /// Checkpoint arrays under the module's keys and layout.
    static func mapped(_ arrays: [String: MLXArray], needsConvTranspose: Bool, for net: NFKMLXTripoSRNet) -> [(String, MLXArray)] {
        let positionKey = "image_tokenizer.model.embeddings.position_embeddings"
        let expectedPositions = net.configuration.patchGrid * net.configuration.patchGrid + 1
        return arrays.compactMap { key, value -> (String, MLXArray)? in
            if key.contains(".pooler.") {
                return nil
            }
            if key == positionKey && value.dim(1) != expectedPositions {
                return (key, resizedPositionEmbeddings(value, grid: net.configuration.patchGrid))
            }
            guard needsConvTranspose && value.ndim == 4 else { return (key, value) }
            if key == "post_processor.upsample.weight" {
                return (key, value.transposed(1, 2, 3, 0))
            }
            if key.hasSuffix("projection.weight") {
                return (key, value.transposed(0, 2, 3, 1))
            }
            return (key, value)
        }
    }

    /// `ViTEmbeddings.interpolate_pos_encoding`: the patch positions as a square grid, bicubic to `grid × grid`
    /// (half-pixel centers, no antialiasing), the class position kept first.
    static func resizedPositionEmbeddings(_ embeddings: MLXArray, grid: Int) -> MLXArray {
        let width = embeddings.dim(2)
        let count = embeddings.dim(1) - 1
        let side = Int(Double(count).squareRoot().rounded())
        let classPosition = embeddings[0..., 0 ..< 1, 0...]
        let patches = embeddings[0..., 1..., 0...].reshaped([side, side, width])
        let resized = NFKMLXPhi4MMImageProcessor.bicubicResized(patches, height: grid, width: grid)
        return concatenated([classPosition, resized.reshaped([1, grid * grid, width])], axis: 1)
    }

    // MARK: Sampling and preparation

    /// Bilinear sample of the flattened `[H·W, C]` plane at per-point `(sampleY, sampleX)` pixel coordinates,
    /// a corner outside the plane contributing nothing (`grid_sample(align_corners: false, padding_mode: "zeros")`).
    static func bilinearSample(_ plane: MLXArray, sampleY: MLXArray, sampleX: MLXArray, height: Int, width: Int) -> MLXArray {
        let y0 = floor(sampleY), x0 = floor(sampleX)
        let y1 = y0 + 1, x1 = x0 + 1
        let weightX = (sampleX - x0).reshaped([-1, 1])
        let weightY = (sampleY - y0).reshaped([-1, 1])
        func inside(_ value: MLXArray, _ size: Int) -> MLXArray {
            let low: MLXArray = (value .>= MLXArray(Float(0))).asType(.float32)
            let high: MLXArray = (value .<= MLXArray(Float(size - 1))).asType(.float32)
            return low * high
        }
        func gather(_ yy: MLXArray, _ xx: MLXArray) -> MLXArray {
            let rows: MLXArray = clip(yy, min: 0, max: Float(height - 1)).asType(.int32)
            let columns: MLXArray = clip(xx, min: 0, max: Float(width - 1)).asType(.int32)
            let valid: MLXArray = (inside(yy, height) * inside(xx, width)).reshaped([-1, 1])
            return plane.take(rows * Int32(width) + columns, axis: 0) * valid
        }
        let top = gather(y0, x0) * (1 - weightX) + gather(y0, x1) * weightX
        let bottom = gather(y1, x0) * (1 - weightX) + gather(y1, x1) * weightX
        return top * (1 - weightY) + bottom * weightY
    }

    /// Applies `body` to `points` `[M, 3]` in slices of `size` rows and concatenates the results.
    static func chunked(_ points: MLXArray, size: Int, _ body: (MLXArray) -> MLXArray) -> MLXArray {
        let count = points.dim(0)
        var pieces = [MLXArray]()
        var start = 0
        while start < count {
            let end = min(start + size, count)
            let piece = body(points[start ..< end])
            eval(piece)
            pieces.append(piece)
            start = end
        }
        return pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0)
    }

    /// The reference demo's preparation of an image `[H, W, 4]` in 0...1. An image whose alpha is all one
    /// is returned as RGB. Otherwise the foreground (alpha above zero) is cropped to its box, padded to a
    /// square, padded again so the object fills `foregroundRatio` of the side, and composited on mid-gray.
    static func prepare(rgba: MLXArray, foregroundRatio: Float) -> MLXArray {
        let (height, width) = (rgba.dim(0), rgba.dim(1))
        let alpha = rgba[0..., 0..., 3]
        eval(alpha)
        let alphaValues = alpha.asArray(Float.self)
        var minY = height, maxY = -1, minX = width, maxX = -1
        for y in 0 ..< height {
            for x in 0 ..< width where alphaValues[y * width + x] > 0 {
                minY = min(minY, y); maxY = max(maxY, y); minX = min(minX, x); maxX = max(maxX, x)
            }
        }
        let opaque = alphaValues.allSatisfy { $0 >= 1 }
        if opaque || maxY < minY || maxX < minX {
            return rgba[0..., 0..., 0 ..< 3]
        }
        // `resize_foreground`: `image[y1:y2, x1:x2]` is half open, so the last row and column are dropped.
        let cropHeight = max(1, maxY - minY), cropWidth = max(1, maxX - minX)
        let foreground = rgba[minY ..< (minY + cropHeight), minX ..< (minX + cropWidth), 0...]
        let square = max(cropHeight, cropWidth)
        let padTop = (square - cropHeight) / 2, padLeft = (square - cropWidth) / 2
        let newSize = Int(Float(square) / foregroundRatio)
        let outerTop = (newSize - square) / 2, outerLeft = (newSize - square) / 2
        let top = padTop + outerTop, left = padLeft + outerLeft
        let padded = padded(foreground, widths: [IntOrPair((top, newSize - cropHeight - top)),
                                                 IntOrPair((left, newSize - cropWidth - left)),
                                                 IntOrPair(0)])
        let color = padded[0..., 0..., 0 ..< 3]
        let coverage = padded[0..., 0..., 3 ..< 4]
        return color * coverage + (1 - coverage) * 0.5
    }

    /// `F.interpolate(mode: "bilinear", align_corners: false, antialias: true)` of `[H, W, C]` to `side × side`.
    static func resizeAntialiased(_ image: MLXArray, side: Int) -> MLXArray {
        let rows = antialiasBilinearMatrix(from: image.dim(0), to: side)                  // [side, H]
        let columns = antialiasBilinearMatrix(from: image.dim(1), to: side)               // [side, W]
        let vertical = matmul(rows, image.transposed(2, 0, 1))                             // [C, side, W]
        return matmul(vertical, columns.transposed(1, 0)).transposed(1, 2, 0)               // [side, side, C]
    }

    /// The 1-D operator `[outN, inN]` of torch's antialiased bilinear resize: a triangle filter whose support
    /// widens with the scale when shrinking, half-pixel centers, and per-output normalization over the
    /// taps inside the image. At a scale of one or an enlargement it is the ordinary bilinear resize.
    static func antialiasBilinearMatrix(from inN: Int, to outN: Int) -> MLXArray {
        let scale = Float(inN) / Float(outN)
        let support: Float = scale > 1 ? scale : 1
        let invscale: Float = scale > 1 ? 1 / scale : 1
        var rows = [Float](repeating: 0, count: outN * inN)
        for i in 0 ..< outN {
            let center = scale * (Float(i) + 0.5)
            let start = max(Int(center - support + 0.5), 0)
            let end = min(Int(center + support + 0.5), inN)
            var weights = [Float](), total: Float = 0
            for j in start ..< end {
                let weight = max(0, 1 - abs((Float(j) - center + 0.5) * invscale))
                weights.append(weight)
                total += weight
            }
            for (offset, j) in (start ..< end).enumerated() where total > 0 {
                rows[i * inN + j] = weights[offset] / total
            }
        }
        return MLXArray(rows).reshaped([outN, inN])
    }
}

// MARK: - Marching cubes

/// Marching cubes over a scalar grid, on the CPU. The case table is derived at first use from the cube's
/// topology instead of being typed in: for each of the 256 corner sign patterns the crossed edges are
/// paired around each face (an ambiguous face, two diagonal corners inside, keeps its inside corners
/// apart), the pairs are followed into closed loops, and each loop is fanned into triangles wound so the
/// surface faces away from the values above the threshold.
enum NFKMarchingCubes {

    struct Surface {
        /// Three floats a vertex, in grid units (a vertex on the edge between grid points `i` and `i + 1`
        /// lies at `i + t`).
        var vertices: [Float]
        var indices: [UInt32]
    }

    /// Corner `c` of a cube is at `(x + cornerOffsets[c].0, y + .1, z + .2)`.
    private static let cornerOffsets: [(Int, Int, Int)] = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                                           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    private static let edgeCorners: [(Int, Int)] = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                                                    (0, 4), (1, 5), (2, 6), (3, 7)]
    /// Each face as its four corners in cyclic order; the face's edges join consecutive corners.
    private static let faceCorners: [[Int]] = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]]

    /// The triangles of every case, as triples of edge indices.
    private static let cases: [[Int]] = (0 ..< 256).map(triangles(forCase:))

    private static func edge(between a: Int, _ b: Int) -> Int {
        edgeCorners.firstIndex { ($0.0 == a && $0.1 == b) || ($0.0 == b && $0.1 == a) }!
    }

    private static func triangles(forCase mask: Int) -> [Int] {
        func inside(_ corner: Int) -> Bool { mask & (1 << corner) != 0 }
        var crossed = Set<Int>()
        for (index, corners) in edgeCorners.enumerated() where inside(corners.0) != inside(corners.1) {
            crossed.insert(index)
        }
        if crossed.isEmpty {
            return []
        }
        // Two connections per crossed edge, one through each adjacent face.
        var links = [Int: [Int]]()
        func link(_ a: Int, _ b: Int) {
            links[a, default: []].append(b)
            links[b, default: []].append(a)
        }
        for corners in faceCorners {
            let faceEdges = (0 ..< 4).map { edge(between: corners[$0], corners[($0 + 1) % 4]) }.filter { crossed.contains($0) }
            if faceEdges.count == 2 {
                link(faceEdges[0], faceEdges[1])
            } else if faceEdges.count == 4 {
                // Diagonal pattern: join the two edges around each inside corner, keeping the corners apart.
                for (position, corner) in corners.enumerated() where inside(corner) {
                    let previous = corners[(position + 3) % 4], next = corners[(position + 1) % 4]
                    link(edge(between: previous, corner), edge(between: corner, next))
                }
            }
        }
        // Follow the links into loops.
        var remaining = crossed
        var result = [Int]()
        while let start = remaining.min() {
            var loop = [start]
            remaining.remove(start)
            var previous = -1, current = start
            while true {
                let next = links[current]!.first { $0 != previous } ?? links[current]![0]
                if next == start { break }
                loop.append(next)
                remaining.remove(next)
                previous = current
                current = next
            }
            result.append(contentsOf: fanned(loop, mask: mask))
        }
        return result
    }

    /// The loop as a fan of triangles, wound to face the outside.
    private static func fanned(_ loop: [Int], mask: Int) -> [Int] {
        func midpoint(_ edge: Int) -> SIMD3<Float> {
            let (a, b) = edgeCorners[edge]
            let p = cornerOffsets[a], q = cornerOffsets[b]
            return SIMD3<Float>(Float(p.0 + q.0), Float(p.1 + q.1), Float(p.2 + q.2)) * 0.5
        }
        let points = loop.map(midpoint)
        var normal = SIMD3<Float>(repeating: 0)
        for index in points.indices {
            let current = points[index], next = points[(index + 1) % points.count]
            normal += SIMD3<Float>((current.y - next.y) * (current.z + next.z),
                                   (current.z - next.z) * (current.x + next.x),
                                   (current.x - next.x) * (current.y + next.y))
        }
        let centroid = points.reduce(SIMD3<Float>(repeating: 0), +) / Float(points.count)
        // The indicator (1 inside) interpolated trilinearly a little along the normal decides the winding.
        func indicator(_ p: SIMD3<Float>) -> Float {
            var sum: Float = 0
            for corner in 0 ..< 8 where mask & (1 << corner) != 0 {
                let o = cornerOffsets[corner]
                let wx = o.0 == 1 ? p.x : 1 - p.x, wy = o.1 == 1 ? p.y : 1 - p.y, wz = o.2 == 1 ? p.z : 1 - p.z
                sum += wx * wy * wz
            }
            return sum
        }
        let step = normal / max(1e-6, (normal * normal).sum().squareRoot()) * 0.05
        let outwardLoop = indicator(centroid + step) <= indicator(centroid - step) ? loop : loop.reversed()
        var triangles = [Int]()
        for index in 1 ..< (outwardLoop.count - 1) {
            triangles.append(contentsOf: [outwardLoop[0], outwardLoop[index], outwardLoop[index + 1]])
        }
        return triangles
    }

    /// The surface where `values` (x-major, `resolution³`) crosses `threshold`, values above it inside.
    static func surface(of values: [Float], resolution: Int, threshold: Float) -> Surface {
        var vertices = [Float]()
        var indices = [UInt32]()
        var vertexForEdge = [Int: UInt32]()
        let stride = (resolution * resolution, resolution, 1)
        func value(_ x: Int, _ y: Int, _ z: Int) -> Float { values[x * stride.0 + y * stride.1 + z] }

        for x in 0 ..< (resolution - 1) {
            for y in 0 ..< (resolution - 1) {
                for z in 0 ..< (resolution - 1) {
                    var mask = 0
                    var corner = [Float](repeating: 0, count: 8)
                    for c in 0 ..< 8 {
                        let o = cornerOffsets[c]
                        corner[c] = value(x + o.0, y + o.1, z + o.2)
                        if corner[c] > threshold { mask |= 1 << c }
                    }
                    if mask == 0 || mask == 255 { continue }
                    for edgeIndex in cases[mask] {
                        let (a, b) = edgeCorners[edgeIndex]
                        let pa = cornerOffsets[a], pb = cornerOffsets[b]
                        // The edge's grid key: its lower grid point and its axis.
                        let lower = (x + min(pa.0, pb.0), y + min(pa.1, pb.1), z + min(pa.2, pb.2))
                        let axis = pa.0 != pb.0 ? 0 : pa.1 != pb.1 ? 1 : 2
                        let key = ((lower.0 * resolution + lower.1) * resolution + lower.2) * 3 + axis
                        if let existing = vertexForEdge[key] {
                            indices.append(existing)
                            continue
                        }
                        let va = corner[a], vb = corner[b]
                        let t = vb == va ? Float(0.5) : (threshold - va) / (vb - va)
                        vertices.append(Float(x + pa.0) + Float(pb.0 - pa.0) * t)
                        vertices.append(Float(y + pa.1) + Float(pb.1 - pa.1) * t)
                        vertices.append(Float(z + pa.2) + Float(pb.2 - pa.2) * t)
                        let index = UInt32(vertexForEdge.count)
                        vertexForEdge[key] = index
                        indices.append(index)
                    }
                }
            }
        }
        return Surface(vertices: vertices, indices: indices)
    }
}
