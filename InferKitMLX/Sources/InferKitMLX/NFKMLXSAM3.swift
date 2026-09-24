//
//  NFKMLXSAM3.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// SAM 3's vision encoder (`Sam3VisionModel`, Meta's Segment Anything 3): a plain-resolution ViT with
// 2-D rotary attention under a windowed schedule, and an FPN neck that reads the single final feature
// map at four scales. SAM 3 segments every instance a text prompt names, where SAM and SAM 2 segment
// what a click points at; this file is the stage its detector and mask decoder read.
//
// The ViT differs from the SAM 1 ViT it succeeds in three ways. Position is rotary rather than a
// learned relative bias, applied to adjacent channel pairs over an (x, y) grid. The learned absolute
// position grid is TILED to the input size rather than interpolated, so a 336-pixel pretraining grid
// covers a 1008-pixel input by repeating. And the layer normalization that usually ends a ViT runs
// BEFORE the block stack.
//
// The windowing is what makes the rotary grid per layer rather than per model. A windowed layer
// attends inside a 24 x 24 window and rotates over a 24 x 24 grid at unit scale; a global layer
// attends over the whole 72 x 72 map and rotates over it at `window / 72`, so a position means the
// same distance in both. A port that shares one table across the stack is right for one kind of layer.

/// SAM 3 vision-encoder geometry. Defaults are the released `facebook/sam3` model.
public struct NFKMLXSAM3Configuration: Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var layers: Int
    public var heads: Int
    public var imageSize: Int
    public var patchSize: Int
    /// The grid the position embedding was trained at, which the input's grid is tiled from.
    public var pretrainImageSize: Int
    public var windowSize: Int
    /// Layers that attend over the whole map instead of inside a window.
    public var globalAttentionLayers: [Int]
    public var layerNormEpsilon: Float
    public var ropeTheta: Float
    public var fpnHiddenSize: Int
    /// Scales the neck reads the final feature map at, coarse to fine as the reference lists them.
    public var scaleFactors: [Double]

    public init(hiddenSize: Int = 1024, intermediateSize: Int = 4736, layers: Int = 32, heads: Int = 16,
                imageSize: Int = 1008, patchSize: Int = 14, pretrainImageSize: Int = 336,
                windowSize: Int = 24, globalAttentionLayers: [Int] = [7, 15, 23, 31],
                layerNormEpsilon: Float = 1e-6, ropeTheta: Float = 10000, fpnHiddenSize: Int = 256,
                scaleFactors: [Double] = [4, 2, 1, 0.5]) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.layers = layers
        self.heads = heads
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.pretrainImageSize = pretrainImageSize
        self.windowSize = windowSize
        self.globalAttentionLayers = globalAttentionLayers
        self.layerNormEpsilon = layerNormEpsilon
        self.ropeTheta = ropeTheta
        self.fpnHiddenSize = fpnHiddenSize
        self.scaleFactors = scaleFactors
    }

    /// The released `facebook/sam3` and `facebook/sam3.1` geometry, which the defaults spell out. The
    /// two releases ship a byte-identical configuration and differ only in their weights.
    public static let base = NFKMLXSAM3Configuration()

    public static let tiny = NFKMLXSAM3Configuration(
        hiddenSize: 32, intermediateSize: 48, layers: 4, heads: 2, imageSize: 56, patchSize: 14,
        pretrainImageSize: 28, windowSize: 2, globalAttentionLayers: [1, 3], fpnHiddenSize: 8)

    var headDim: Int { hiddenSize / heads }
    var grid: Int { imageSize / patchSize }
    var pretrainGrid: Int { pretrainImageSize / patchSize }
}

/// The 2-D rotary table one layer rotates over: `dim/4` frequencies per axis, each held for two
/// channels, the x half leading.
struct NFKSAM3Rotary {
    let cos: MLXArray                                                      // [endX·endY, headDim]
    let sin: MLXArray

    init(headDim: Int, endX: Int, endY: Int, scale: Float, theta: Float) {
        let pairs = headDim / 4
        var angles = [Float](repeating: 0, count: endX * endY * headDim)
        for token in 0 ..< endX * endY {
            let x = Float(token % endX) * scale, y = Float(token / endX) * scale
            for pair in 0 ..< pairs {
                let frequency = powf(theta, -(Float(4 * pair) / Float(headDim)))
                // `repeat_interleave(2)` after the x/y concatenation: each frequency covers the two
                // channels of one rotated pair.
                angles[token * headDim + 2 * pair] = x * frequency
                angles[token * headDim + 2 * pair + 1] = x * frequency
                angles[token * headDim + 2 * (pairs + pair)] = y * frequency
                angles[token * headDim + 2 * (pairs + pair) + 1] = y * frequency
            }
        }
        let table = MLXArray(angles, [endX * endY, headDim])
        self.cos = MLX.cos(table)
        self.sin = MLX.sin(table)
    }

    /// `x` `[batch, heads, tokens, headDim]`, rotated in adjacent channel pairs.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let split = x.reshaped([shape[0], shape[1], shape[2], shape[3] / 2, 2])
        let even = split[0..., 0..., 0..., 0..., 0]
        let odd = split[0..., 0..., 0..., 0..., 1]
        let rotated = stacked([-odd, even], axis: -1).reshaped(shape)
        return x * cos + rotated * sin
    }
}

/// Self-attention with the layer's own 2-D rotary table.
final class NFKSAM3Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let heads: Int
    let headDim: Int

    init(_ configuration: NFKMLXSAM3Configuration) {
        self.heads = configuration.heads
        self.headDim = configuration.headDim
        let width = configuration.hiddenSize
        _qProj.wrappedValue = Linear(width, width)
        _kProj.wrappedValue = Linear(width, width)
        _vProj.wrappedValue = Linear(width, width)
        _oProj.wrappedValue = Linear(width, width)
    }

    /// `x` `[batch, height, width, hidden]` → the same shape. A windowed layer arrives with its
    /// windows in the batch axis.
    func callAsFunction(_ x: MLXArray, rotary: NFKSAM3Rotary) -> MLXArray {
        let batch = x.dim(0), height = x.dim(1), width = x.dim(2)
        let tokens = height * width
        func split(_ projection: Linear) -> MLXArray {
            projection(x).reshaped([batch, tokens, heads, headDim]).transposed(0, 2, 1, 3)
        }
        let queries = rotary(split(qProj))
        let keys = rotary(split(kProj))
        let values = split(vProj)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1 / sqrt(Float(headDim)), mask: .none)
        return oProj(attended.transposed(0, 2, 1, 3).reshaped([batch, height, width, heads * headDim]))
    }
}

/// One ViT layer: rotary attention, then a two-layer MLP, each pre-normalized and residual.
final class NFKSAM3Layer: Module {
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: NFKSAM3Attention
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKSAM3MLP

    /// Zero for a layer that attends over the whole map.
    let windowSize: Int
    let rotary: NFKSAM3Rotary

    init(_ configuration: NFKMLXSAM3Configuration, windowSize: Int) {
        self.windowSize = windowSize
        let side = windowSize == 0 ? configuration.grid : windowSize
        self.rotary = NFKSAM3Rotary(headDim: configuration.headDim, endX: side, endY: side,
                                    scale: Float(configuration.windowSize) / Float(side),
                                    theta: configuration.ropeTheta)
        _norm1.wrappedValue = LayerNorm(dimensions: configuration.hiddenSize,
                                        eps: configuration.layerNormEpsilon)
        _attention.wrappedValue = NFKSAM3Attention(configuration)
        _norm2.wrappedValue = LayerNorm(dimensions: configuration.hiddenSize,
                                        eps: configuration.layerNormEpsilon)
        _mlp.wrappedValue = NFKSAM3MLP(configuration)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var normalized = norm1(x)
        let (height, width) = (x.dim(1), x.dim(2))
        var attended: MLXArray
        if windowSize > 0 {
            let (windows, padded) = NFKSAM3Windows.partition(normalized, size: windowSize)
            attended = NFKSAM3Windows.join(attention(windows, rotary: rotary), size: windowSize,
                                           padded: padded, original: (height, width))
        } else {
            attended = attention(normalized, rotary: rotary)
        }
        let residual = x + attended
        normalized = norm2(residual)
        return residual + mlp(normalized)
    }
}

/// The ViT's feed-forward: one widening projection, the exact gelu, one narrowing projection.
final class NFKSAM3MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ configuration: NFKMLXSAM3Configuration) {
        _fc1.wrappedValue = Linear(configuration.hiddenSize, configuration.intermediateSize)
        _fc2.wrappedValue = Linear(configuration.intermediateSize, configuration.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// Splitting a feature map into attention windows and putting it back, padding to a whole number of
/// windows and cropping the padding away afterwards.
enum NFKSAM3Windows {
    static func partition(_ x: MLXArray, size: Int) -> (MLXArray, (height: Int, width: Int)) {
        let batch = x.dim(0), height = x.dim(1), width = x.dim(2), channels = x.dim(3)
        let padHeight = (size - height % size) % size
        let padWidth = (size - width % size) % size
        var padded = x
        if padHeight > 0 || padWidth > 0 {
            padded = MLX.padded(x, widths: [.init((0, 0)), .init((0, padHeight)),
                                            .init((0, padWidth)), .init((0, 0))])
        }
        let (paddedHeight, paddedWidth) = (height + padHeight, width + padWidth)
        let windows = padded
            .reshaped([batch, paddedHeight / size, size, paddedWidth / size, size, channels])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([-1, size, size, channels])
        return (windows, (paddedHeight, paddedWidth))
    }

    static func join(_ windows: MLXArray, size: Int, padded: (height: Int, width: Int),
                     original: (height: Int, width: Int)) -> MLXArray {
        let channels = windows.dim(3)
        let batch = windows.dim(0) / (padded.height * padded.width / size / size)
        var joined = windows
            .reshaped([batch, padded.height / size, padded.width / size, size, size, channels])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([batch, padded.height, padded.width, channels])
        if padded.height != original.height || padded.width != original.width {
            joined = joined[0..., 0 ..< original.height, 0 ..< original.width, 0...]
        }
        return joined
    }
}

/// The patch embedding and the tiled absolute position grid.
final class NFKSAM3Embeddings: Module {
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: NFKSAM3PatchEmbedding
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray

    let configuration: NFKMLXSAM3Configuration

    init(_ configuration: NFKMLXSAM3Configuration) {
        self.configuration = configuration
        _patchEmbeddings.wrappedValue = NFKSAM3PatchEmbedding(configuration)
        let patches = configuration.pretrainGrid * configuration.pretrainGrid
        _positionEmbeddings.wrappedValue = MLXArray.zeros([1, patches, configuration.hiddenSize])
    }

    /// `image` `[1, H, W, 3]` → `[1, H/patch, W/patch, hidden]`.
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        let embedded = patchEmbeddings(image)                              // [1, h, w, hidden]
        let (height, width) = (embedded.dim(1), embedded.dim(2))
        return embedded + tiled(height: height, width: width)
    }

    /// The pretraining grid REPEATED to cover `height × width` and cropped, which is what the
    /// reference does instead of interpolating.
    func tiled(height: Int, width: Int) -> MLXArray {
        let side = configuration.pretrainGrid
        let hidden = configuration.hiddenSize
        let grid = positionEmbeddings.reshaped([1, side, side, hidden])
        if side == height && side == width {
            return grid
        }
        let repeats = [1, height / side + 1, width / side + 1, 1]
        return MLX.tiled(grid, repetitions: repeats)[0..., 0 ..< height, 0 ..< width, 0...]
    }
}

/// The patch embedding: one strided convolution, no bias.
final class NFKSAM3PatchEmbedding: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(_ configuration: NFKMLXSAM3Configuration) {
        _projection.wrappedValue = Conv2d(inputChannels: 3, outputChannels: configuration.hiddenSize,
                                          kernelSize: IntOrPair(configuration.patchSize),
                                          stride: IntOrPair(configuration.patchSize), bias: false)
    }

    func callAsFunction(_ image: MLXArray) -> MLXArray { projection(image) }
}

/// The SAM 3 ViT.
public final class NFKMLXSAM3BackboneNet: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKSAM3Embeddings
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [NFKSAM3Layer]

    public init(_ configuration: NFKMLXSAM3Configuration) {
        _embeddings.wrappedValue = NFKSAM3Embeddings(configuration)
        _layerNorm.wrappedValue = LayerNorm(dimensions: configuration.hiddenSize,
                                            eps: configuration.layerNormEpsilon)
        let global = Set(configuration.globalAttentionLayers)
        _layers.wrappedValue = (0 ..< configuration.layers).map {
            NFKSAM3Layer(configuration, windowSize: global.contains($0) ? 0 : configuration.windowSize)
        }
    }

    /// `image` `[1, H, W, 3]` → the final feature map `[1, H/patch, W/patch, hidden]`.
    ///
    /// The normalization runs before the stack, not after it, which is the reverse of the usual ViT.
    public func callAsFunction(_ image: MLXArray) -> MLXArray {
        var hidden = layerNorm(embeddings(image))
        for layer in layers {
            hidden = layer(hidden)
        }
        return hidden
    }
}

/// One FPN level: the scale change its factor calls for, then a 1x1 projection to the neck's width
/// and a 3x3 that mixes it.
final class NFKSAM3FPNLayer: Module {
    @ModuleInfo(key: "scale_layers") var scaleLayers: [Module]
    @ModuleInfo(key: "proj1") var proj1: Conv2d
    @ModuleInfo(key: "proj2") var proj2: Conv2d

    let scaleFactor: Double

    init(inChannels: Int, fpnDimensions: Int, scaleFactor: Double) {
        self.scaleFactor = scaleFactor
        var scaled = [Module]()
        var intermediate = inChannels
        switch scaleFactor {
        case 4:
            // The activation between the two transposed convolutions carries nothing, so it holds a
            // slot in the reference's `Sequential` and a marker here.
            scaled = [ConvTransposed2d(inputChannels: inChannels, outputChannels: inChannels / 2,
                                       kernelSize: 2, stride: 2),
                      Module(),
                      ConvTransposed2d(inputChannels: inChannels / 2, outputChannels: inChannels / 4,
                                       kernelSize: 2, stride: 2)]
            intermediate = inChannels / 4
        case 2:
            scaled = [ConvTransposed2d(inputChannels: inChannels, outputChannels: inChannels / 2,
                                       kernelSize: 2, stride: 2)]
            intermediate = inChannels / 2
        default:
            // Scale 1 passes through and scale 0.5 pools, neither carrying parameters.
            break
        }
        _scaleLayers.wrappedValue = scaled
        _proj1.wrappedValue = Conv2d(inputChannels: intermediate, outputChannels: fpnDimensions,
                                     kernelSize: 1)
        _proj2.wrappedValue = Conv2d(inputChannels: fpnDimensions, outputChannels: fpnDimensions,
                                     kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        switch scaleFactor {
        case 4:
            out = gelu((scaleLayers[0] as! ConvTransposed2d)(out))
            out = (scaleLayers[2] as! ConvTransposed2d)(out)
        case 2:
            out = (scaleLayers[0] as! ConvTransposed2d)(out)
        case 0.5:
            out = NFKMLXResample.maxPooled(out, kernel: 2, stride: 2)
        default:
            break
        }
        return proj2(proj1(out))
    }
}

/// The SAM 3 vision encoder: the ViT and the FPN neck that reads its one output map at four scales.
public final class NFKMLXSAM3VisionNet: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKMLXSAM3BackboneNet
    @ModuleInfo(key: "neck") var neck: NFKSAM3Neck

    public let configuration: NFKMLXSAM3Configuration

    public init(_ configuration: NFKMLXSAM3Configuration = .base) {
        self.configuration = configuration
        _backbone.wrappedValue = NFKMLXSAM3BackboneNet(configuration)
        _neck.wrappedValue = NFKSAM3Neck(configuration)
    }

    /// `image` `[1, H, W, 3]`, normalized → the FPN levels in the reference's order, finest first.
    public func callAsFunction(_ image: MLXArray) -> [MLXArray] {
        neck(backbone(image))
    }
}

/// The FPN neck: one level per scale factor, each reading the same final feature map.
final class NFKSAM3Neck: Module {
    @ModuleInfo(key: "fpn_layers") var fpnLayers: [NFKSAM3FPNLayer]

    init(_ configuration: NFKMLXSAM3Configuration) {
        _fpnLayers.wrappedValue = configuration.scaleFactors.map {
            NFKSAM3FPNLayer(inChannels: configuration.hiddenSize,
                            fpnDimensions: configuration.fpnHiddenSize, scaleFactor: $0)
        }
    }

    func callAsFunction(_ hidden: MLXArray) -> [MLXArray] { fpnLayers.map { $0(hidden) } }
}

/// Building the SAM 3 vision encoder, and naming what its released checkpoint holds.
@objc(NFKMLXSAM3)
public final class NFKMLXSAM3: NSObject {

    /// The vision encoder at a chosen geometry.
    public static func makeVisionNet(_ configuration: NFKMLXSAM3Configuration = .base) -> NFKMLXSAM3VisionNet {
        NFKMLXSAM3VisionNet(configuration)
    }

    /// Reads a released `config.json`, which nests the vision encoder under the detector.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXSAM3Configuration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let detector = json["detector_config"] as? [String: Any],
              let vision = detector["vision_config"] as? [String: Any],
              let backbone = vision["backbone_config"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a SAM 3 configuration")
        }
        func integer(_ source: [String: Any], _ key: String, _ fallback: Int) -> Int {
            (source[key] as? NSNumber)?.intValue ?? fallback
        }
        return NFKMLXSAM3Configuration(
            hiddenSize: integer(backbone, "hidden_size", 1024),
            intermediateSize: integer(backbone, "intermediate_size", 4736),
            layers: integer(backbone, "num_hidden_layers", 32),
            heads: integer(backbone, "num_attention_heads", 16),
            imageSize: integer(backbone, "image_size", 1008),
            patchSize: integer(backbone, "patch_size", 14),
            pretrainImageSize: integer(backbone, "pretrain_image_size", 336),
            windowSize: integer(backbone, "window_size", 24),
            globalAttentionLayers: (backbone["global_attn_indexes"] as? [Int]) ?? [7, 15, 23, 31],
            layerNormEpsilon: (backbone["layer_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6,
            ropeTheta: (backbone["rope_theta"] as? NSNumber)?.floatValue ?? 10000,
            fpnHiddenSize: integer(vision, "fpn_hidden_size", 256),
            scaleFactors: (vision["scale_factors"] as? [Double]) ?? [4, 2, 1, 0.5])
    }

    /// The prefix the released checkpoint keeps its vision encoder under. Its 1797 tensors also carry
    /// the text tower, the detector, and the video tracker, none of which this module reads.
    static let visionPrefix = "detector_model.vision_encoder."

    /// The module key a released tensor name maps to, or nil for a tensor outside the vision encoder.
    /// Inside the prefix the release's names ARE the module's.
    static func remapVisionKey(_ key: String) -> String? {
        key.hasPrefix(visionPrefix) ? String(key.dropFirst(visionPrefix.count)) : nil
    }

    /// Loads the vision encoder out of a released checkpoint, ignoring everything else it holds.
    public static func loadVisionWeights(into net: NFKMLXSAM3VisionNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            guard let name = remapVisionKey(key) else { return nil }
            guard checkpoint.needsConvTranspose, value.ndim == 4 else { return (name, value) }
            // The neck's scale layers are transposed convolutions, stored `[in, out, kH, kW]`; the
            // patch embedding and the two projections are forward ones, stored `[out, in, kH, kW]`.
            if name.contains("scale_layers") { return (name, value.transposed(1, 2, 3, 0)) }
            return (name, value.transposed(0, 2, 3, 1))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}


// MARK: - Text encoder
//
// SAM 3 names its targets in words, so the prompt side is a CLIP text tower: a 24-layer causal
// transformer 1024 wide over a 49408-token vocabulary, with a 32-position context. Two projections
// leave it. The tower's own 512-wide `text_projection` is CLIP's contrastive head and SAM 3 never
// reads it; the detector's 1024 -> 256 projection is the one that reaches the decoder, and it is
// applied to EVERY token rather than to the pooled end-of-text one.

/// One CLIP text layer: causal self-attention, then an MLP, each pre-normalized and residual.
final class NFKSAM3TextLayer: Module {
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: NFKSAM3TextAttention
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKSAM3TextMLP

    init(width: Int, heads: Int, intermediate: Int, epsilon: Float) {
        _norm1.wrappedValue = LayerNorm(dimensions: width, eps: epsilon)
        _attention.wrappedValue = NFKSAM3TextAttention(width: width, heads: heads)
        _norm2.wrappedValue = LayerNorm(dimensions: width, eps: epsilon)
        _mlp.wrappedValue = NFKSAM3TextMLP(width: width, intermediate: intermediate)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let attended = x + attention(norm1(x), mask: mask)
        return attended + mlp(norm2(attended))
    }
}

/// CLIP text attention: separate projections, and an output named for the reference's `out_proj`.
final class NFKSAM3TextAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let heads: Int
    let headDim: Int

    init(width: Int, heads: Int) {
        self.heads = heads
        self.headDim = width / heads
        _qProj.wrappedValue = Linear(width, width)
        _kProj.wrappedValue = Linear(width, width)
        _vProj.wrappedValue = Linear(width, width)
        _outProj.wrappedValue = Linear(width, width)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let batch = x.dim(0), tokens = x.dim(1)
        func split(_ projection: Linear) -> MLXArray {
            projection(x).reshaped([batch, tokens, heads, headDim]).transposed(0, 2, 1, 3)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(qProj), keys: split(kProj), values: split(vProj),
            scale: 1 / sqrt(Float(headDim)), mask: mask.map { .array($0) } ?? .none)
        return outProj(attended.transposed(0, 2, 1, 3).reshaped([batch, tokens, heads * headDim]))
    }
}

/// The text tower's feed-forward. The released configuration names the exact gelu, not the quick
/// approximation CLIP's own releases use.
final class NFKSAM3TextMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(width: Int, intermediate: Int) {
        _fc1.wrappedValue = Linear(width, intermediate)
        _fc2.wrappedValue = Linear(intermediate, width)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// The CLIP text transformer.
final class NFKSAM3TextTransformer: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKSAM3TextEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKSAM3TextEncoderStack
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(_ configuration: NFKMLXSAM3TextConfiguration) {
        _embeddings.wrappedValue = NFKSAM3TextEmbeddings(configuration)
        _encoder.wrappedValue = NFKSAM3TextEncoderStack(configuration)
        _finalNorm.wrappedValue = LayerNorm(dimensions: configuration.hiddenSize,
                                            eps: configuration.layerNormEpsilon)
    }

    func callAsFunction(_ tokens: MLXArray, mask: MLXArray?) -> MLXArray {
        finalNorm(encoder(embeddings(tokens), mask: mask))
    }
}

/// The layer stack, under the reference's `encoder.layers`.
final class NFKSAM3TextEncoderStack: Module {
    @ModuleInfo(key: "layers") var layers: [NFKSAM3TextLayer]

    init(_ configuration: NFKMLXSAM3TextConfiguration) {
        _layers.wrappedValue = (0 ..< configuration.layers).map { _ in
            NFKSAM3TextLayer(width: configuration.hiddenSize, heads: configuration.heads,
                             intermediate: configuration.intermediateSize,
                             epsilon: configuration.layerNormEpsilon)
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var hidden = x
        for layer in layers {
            hidden = layer(hidden, mask: mask)
        }
        return hidden
    }
}

/// Token and absolute position embeddings.
final class NFKSAM3TextEmbeddings: Module {
    @ModuleInfo(key: "token_embedding") var tokens: Embedding
    @ModuleInfo(key: "position_embedding") var positions: Embedding

    init(_ configuration: NFKMLXSAM3TextConfiguration) {
        _tokens.wrappedValue = Embedding(embeddingCount: configuration.vocabularySize,
                                         dimensions: configuration.hiddenSize)
        _positions.wrappedValue = Embedding(embeddingCount: configuration.contextLength,
                                            dimensions: configuration.hiddenSize)
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let length = ids.dim(1)
        return tokens(ids) + positions.weight[0 ..< length].expandedDimensions(axis: 0)
    }
}

/// SAM 3 text-tower geometry. Defaults are the released `facebook/sam3` text encoder.
public struct NFKMLXSAM3TextConfiguration: Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var layers: Int
    public var heads: Int
    public var vocabularySize: Int
    public var contextLength: Int
    public var layerNormEpsilon: Float
    /// The contrastive head CLIP ships and SAM 3 does not read.
    public var projectionSize: Int
    /// The width the detector reads the prompt at.
    public var promptSize: Int

    public init(hiddenSize: Int = 1024, intermediateSize: Int = 4096, layers: Int = 24, heads: Int = 16,
                vocabularySize: Int = 49408, contextLength: Int = 32, layerNormEpsilon: Float = 1e-5,
                projectionSize: Int = 512, promptSize: Int = 256) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.layers = layers
        self.heads = heads
        self.vocabularySize = vocabularySize
        self.contextLength = contextLength
        self.layerNormEpsilon = layerNormEpsilon
        self.projectionSize = projectionSize
        self.promptSize = promptSize
    }

    public static let base = NFKMLXSAM3TextConfiguration()

    public static let tiny = NFKMLXSAM3TextConfiguration(
        hiddenSize: 32, intermediateSize: 48, layers: 2, heads: 2, vocabularySize: 64,
        contextLength: 16, projectionSize: 16, promptSize: 8)
}

/// The CLIP text tower and its contrastive projection, under the release's own names.
final class NFKSAM3TextEncoder: Module {
    @ModuleInfo(key: "text_model") var textModel: NFKSAM3TextTransformer
    @ModuleInfo(key: "text_projection") var projection: Linear

    init(_ configuration: NFKMLXSAM3TextConfiguration) {
        _textModel.wrappedValue = NFKSAM3TextTransformer(configuration)
        _projection.wrappedValue = Linear(configuration.hiddenSize, configuration.projectionSize,
                                          bias: false)
    }
}

/// SAM 3's prompt side: the CLIP text tower and the projection that carries its tokens to the
/// detector's width.
public final class NFKMLXSAM3TextNet: Module {
    @ModuleInfo(key: "text_encoder") var encoder: NFKSAM3TextEncoder
    @ModuleInfo(key: "text_projection") var promptProjection: Linear

    public let configuration: NFKMLXSAM3TextConfiguration

    public init(_ configuration: NFKMLXSAM3TextConfiguration = .base) {
        self.configuration = configuration
        _encoder.wrappedValue = NFKSAM3TextEncoder(configuration)
        _promptProjection.wrappedValue = Linear(configuration.hiddenSize, configuration.promptSize)
    }

    /// `ids` `[1, L]` → the tower's token features `[1, L, hiddenSize]`.
    ///
    /// The mask is causal, as CLIP's text tower is trained. A caller that pads its prompt passes the
    /// padding positions in `padding` so they score nothing.
    public func features(_ ids: MLXArray, padding: MLXArray? = nil) -> MLXArray {
        encoder.textModel(ids, mask: mask(length: ids.dim(1), padding: padding))
    }

    /// The prompt as the detector reads it: every token at the detector's width.
    public func prompt(_ ids: MLXArray, padding: MLXArray? = nil) -> MLXArray {
        promptProjection(features(ids, padding: padding))
    }

    /// The additive attention mask: causal, plus any padding the caller marks.
    func mask(length: Int, padding: MLXArray?) -> MLXArray {
        var values = [Float](repeating: 0, count: length * length)
        for query in 0 ..< length {
            for key in (query + 1) ..< length {
                values[query * length + key] = -Float.greatestFiniteMagnitude
            }
        }
        var mask = MLXArray(values, [1, 1, length, length])
        if let padding {
            mask = mask + padding.reshaped([1, 1, 1, length])
        }
        return mask
    }
}

extension NFKMLXSAM3 {

    /// The text encoder at a chosen geometry.
    public static func makeTextNet(_ configuration: NFKMLXSAM3TextConfiguration = .base) -> NFKMLXSAM3TextNet {
        NFKMLXSAM3TextNet(configuration)
    }

    /// Reads the text tower's geometry from a released `config.json`.
    public static func textConfiguration(fromHuggingFace url: URL) throws -> NFKMLXSAM3TextConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let detector = json["detector_config"] as? [String: Any],
              let text = detector["text_config"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a SAM 3 configuration")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (text[key] as? NSNumber)?.intValue ?? fallback }
        let encoder = (detector["detr_encoder_config"] as? [String: Any]) ?? [:]
        return NFKMLXSAM3TextConfiguration(
            hiddenSize: integer("hidden_size", 1024),
            intermediateSize: integer("intermediate_size", 4096),
            layers: integer("num_hidden_layers", 24),
            heads: integer("num_attention_heads", 16),
            vocabularySize: integer("vocab_size", 49408),
            contextLength: integer("max_position_embeddings", 32),
            layerNormEpsilon: (text["layer_norm_eps"] as? NSNumber)?.floatValue ?? 1e-5,
            projectionSize: integer("projection_dim", 512),
            promptSize: (encoder["hidden_size"] as? NSNumber)?.intValue ?? 256)
    }

    /// Loads the text tower and the prompt projection out of a released checkpoint.
    public static func loadTextWeights(into net: NFKMLXSAM3TextNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let prefix = "detector_model."
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix(prefix) else { return nil }
            let name = String(key.dropFirst(prefix.count))
            guard name.hasPrefix("text_encoder.") || name.hasPrefix("text_projection.") else { return nil }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
