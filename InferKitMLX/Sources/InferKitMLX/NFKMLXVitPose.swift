//
//  NFKMLXVitPose.swift
//  InferKitMLX
//
//  ViTPose top-down pose estimation (`VitPoseForPoseEstimation`, ViTAE-Transformer / usyd-community),
//  a plain ViT backbone under a small decoding head, and the modern counterpart to the SimpleBaseline
//  ResNet port. Reference parity is measured against transformers' own implementation. Tensors flow
//  NHWC.
//

import Foundation
import CoreGraphics
import MLX
import MLXNN
import InferKit

// MARK: - Configuration

/// How a ViTPose release turns the backbone's feature map into heatmaps.
///
/// - Since: InferKit 0.4.0
@objc(NFKMLXVitPoseDecoder)
public enum NFKMLXVitPoseDecoder: Int, Sendable {
    /// `use_simple_decoder: true` — a ReLU, a bilinear upsample, and one 3×3 convolution.
    case simple
    /// `use_simple_decoder: false` — two transposed-convolution blocks and a 1×1 convolution, which is
    /// the head SimpleBaseline uses.
    case classic
}

/// The ViTPose geometry. Defaults are the base releases (ViT-B at 256×192).
public struct NFKMLXVitPoseConfiguration: Sendable {
    public var hiddenSize: Int = 768
    public var layerCount: Int = 12
    public var headCount: Int = 12
    public var mlpRatio: Int = 4
    /// A person crop is taller than it is wide, so the trained geometry is not square.
    public var inputHeight: Int = 256
    public var inputWidth: Int = 192
    public var patchSize: Int = 16
    /// The patch convolution's padding, which the reference hardcodes at two.
    public var patchPadding: Int = 2
    /// The reference's `layer_norm_eps`, which is 1e-12 rather than a ViT's usual 1e-6.
    public var layerNormEps: Float = 1e-12
    public var keypointCount: Int = 17
    /// How far the simple decoder upsamples before its convolution (`scale_factor`).
    public var scaleFactor: Int = 4
    public var decoder: NFKMLXVitPoseDecoder = .simple
    /// The classic decoder's transposed-convolution width.
    public var decoderChannels: Int = 256

    public init() {}

    /// `usyd-community/vitpose-base-simple`.
    public static var baseSimple: NFKMLXVitPoseConfiguration { NFKMLXVitPoseConfiguration() }

    /// `usyd-community/vitpose-base`, the same backbone under the classic decoder.
    public static var base: NFKMLXVitPoseConfiguration {
        var configuration = NFKMLXVitPoseConfiguration()
        configuration.decoder = .classic
        return configuration
    }

    /// A small configuration for weight-free tests.
    public static var tiny: NFKMLXVitPoseConfiguration {
        var configuration = NFKMLXVitPoseConfiguration()
        configuration.hiddenSize = 32
        configuration.layerCount = 2
        configuration.headCount = 2
        configuration.inputHeight = 64
        configuration.inputWidth = 48
        configuration.keypointCount = 4
        configuration.decoderChannels = 16
        return configuration
    }

    var gridHeight: Int { (inputHeight + 2 * patchPadding - patchSize) / patchSize + 1 }
    var gridWidth: Int { (inputWidth + 2 * patchPadding - patchSize) / patchSize + 1 }
    var patchCount: Int { gridHeight * gridWidth }
    var headDimensions: Int { hiddenSize / headCount }

    /// Reads a release's `config.json`.
    ///
    /// A `vitpose-plus-*` release routes its feed-forward through per-dataset experts
    /// (`num_experts` above one, with `part_features` splitting the hidden width) and needs a dataset
    /// index at inference. That is refused rather than loaded into a dense stack.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXVitPoseConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        let kind = (json["model_type"] as? String) ?? ""
        guard kind == "vitpose" else {
            throw NFKMLXError.unsupportedConfiguration("this reads a ViTPose config, not \(kind)")
        }
        let backbone = (json["backbone_config"] as? [String: Any]) ?? [:]
        func integer(_ source: [String: Any], _ key: String, _ fallback: Int) -> Int {
            (source[key] as? NSNumber)?.intValue ?? fallback
        }
        if integer(backbone, "num_experts", 1) > 1 {
            throw NFKMLXError.unsupportedConfiguration(
                "a vitpose-plus release routes its feed-forward through per-dataset experts, which this does not build")
        }
        var configuration = NFKMLXVitPoseConfiguration()
        configuration.hiddenSize = integer(backbone, "hidden_size", 768)
        configuration.layerCount = integer(backbone, "num_hidden_layers", 12)
        configuration.headCount = integer(backbone, "num_attention_heads", 12)
        configuration.mlpRatio = integer(backbone, "mlp_ratio", 4)
        if let size = backbone["image_size"] as? [NSNumber], size.count == 2 {
            configuration.inputHeight = size[0].intValue
            configuration.inputWidth = size[1].intValue
        }
        if let patch = backbone["patch_size"] as? [NSNumber], patch.count == 2 {
            configuration.patchSize = patch[0].intValue
        }
        if let eps = backbone["layer_norm_eps"] as? NSNumber { configuration.layerNormEps = eps.floatValue }
        configuration.keypointCount = (json["id2label"] as? [String: Any])?.count ?? 17
        configuration.scaleFactor = integer(json, "scale_factor", 4)
        configuration.decoder = (json["use_simple_decoder"] as? Bool ?? false) ? .simple : .classic
        return configuration
    }
}

// MARK: - Backbone

/// The patch embedding (`embeddings.patch_embeddings.projection`), a stride-`patch` convolution that
/// **pads by two**.
///
/// The padding leaves the patch count alone at the trained geometry (256×192 at patch 16 gives 16×12
/// either way), so a plain non-overlapping patchify loads cleanly and samples every window two pixels
/// early. That scored the backbone at 0.998 against the reference's 0.9999999999.
final class NFKVitPosePatchEmbeddings: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(_ c: NFKMLXVitPoseConfiguration) {
        _projection.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c.hiddenSize,
                                          kernelSize: IntOrPair(c.patchSize), stride: IntOrPair(c.patchSize),
                                          padding: IntOrPair(c.patchPadding))
    }

    /// - Parameter image: `[1, H, W, 3]`.
    /// - Returns: `[1, patches, hidden]`.
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        let patches = projection(image)                                 // [1, gh, gw, hidden]
        return patches.reshaped([patches.shape[0], patches.shape[1] * patches.shape[2], patches.shape[3]])
    }
}

/// The embeddings (`embeddings`). The position table carries a class-token row the sequence does not:
/// the reference adds the patch rows AND that row to every patch (`pos[:, 1:] + pos[:, :1]`), so the
/// class position acts as a constant bias rather than a token.
final class NFKVitPoseEmbeddings: Module {
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: NFKVitPosePatchEmbeddings
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: MLXArray

    init(_ c: NFKMLXVitPoseConfiguration) {
        _patchEmbeddings.wrappedValue = NFKVitPosePatchEmbeddings(c)
        _positionEmbeddings.wrappedValue = MLXArray.zeros([1, c.patchCount + 1, c.hiddenSize])
    }

    func callAsFunction(_ image: MLXArray) -> MLXArray {
        let patches = patchEmbeddings(image)
        return patches + positionEmbeddings[0..., 1..., 0...] + positionEmbeddings[0..., 0 ..< 1, 0...]
    }
}

/// The self-attention (`attention.attention.{query,key,value}` and `attention.output.dense`), kept in
/// the reference's separate-projection layout rather than a fused one.
final class NFKVitPoseAttention: Module {
    @ModuleInfo(key: "attention") var attention: NFKVitPoseSelfAttention
    @ModuleInfo(key: "output") var output: NFKVitPoseSelfOutput

    init(_ c: NFKMLXVitPoseConfiguration) {
        _attention.wrappedValue = NFKVitPoseSelfAttention(c)
        _output.wrappedValue = NFKVitPoseSelfOutput(c)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { output(attention(x)) }
}

final class NFKVitPoseSelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    let heads: Int
    let headDimensions: Int

    init(_ c: NFKMLXVitPoseConfiguration) {
        heads = c.headCount
        headDimensions = c.headDimensions
        _query.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _key.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _value.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, tokens) = (x.shape[0], x.shape[1])
        func split(_ projection: MLXArray) -> MLXArray {
            projection.reshaped([batch, tokens, heads, headDimensions]).transposed(0, 2, 1, 3)
        }
        let q = split(query(x)), k = split(key(x)), v = split(value(x))
        let scale = 1.0 / sqrtf(Float(headDimensions))
        let scores = softmax((q * scale).matmul(k.transposed(0, 1, 3, 2)), axis: -1)
        return scores.matmul(v).transposed(0, 2, 1, 3).reshaped([batch, tokens, heads * headDimensions])
    }
}

final class NFKVitPoseSelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    init(_ c: NFKMLXVitPoseConfiguration) { _dense.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { dense(x) }
}

/// The block feed-forward (`mlp.fc1`, `mlp.fc2`) at the reference's exact error-function GELU.
final class NFKVitPoseMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ c: NFKMLXVitPoseConfiguration) {
        _fc1.wrappedValue = Linear(c.hiddenSize, c.hiddenSize * c.mlpRatio, bias: true)
        _fc2.wrappedValue = Linear(c.hiddenSize * c.mlpRatio, c.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// One pre-norm transformer block.
final class NFKVitPoseLayer: Module {
    @ModuleInfo(key: "attention") var attention: NFKVitPoseAttention
    @ModuleInfo(key: "mlp") var mlp: NFKVitPoseMLP
    @ModuleInfo(key: "layernorm_before") var normBefore: LayerNorm
    @ModuleInfo(key: "layernorm_after") var normAfter: LayerNorm

    init(_ c: NFKMLXVitPoseConfiguration) {
        _attention.wrappedValue = NFKVitPoseAttention(c)
        _mlp.wrappedValue = NFKVitPoseMLP(c)
        _normBefore.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _normAfter.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attended = attention(normBefore(x)) + x
        return mlp(normAfter(attended)) + attended
    }
}

final class NFKVitPoseEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [NFKVitPoseLayer]

    init(_ c: NFKMLXVitPoseConfiguration) {
        _layer.wrappedValue = (0 ..< c.layerCount).map { _ in NFKVitPoseLayer(c) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var tokens = x
        for block in layer { tokens = block(tokens) }
        return tokens
    }
}

/// The ViT backbone (`backbone`): embeddings, blocks, a final LayerNorm, and the token sequence
/// reshaped back to a feature map.
final class NFKVitPoseBackbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKVitPoseEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKVitPoseEncoder
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm

    private let configuration: NFKMLXVitPoseConfiguration

    init(_ c: NFKMLXVitPoseConfiguration) {
        configuration = c
        _embeddings.wrappedValue = NFKVitPoseEmbeddings(c)
        _encoder.wrappedValue = NFKVitPoseEncoder(c)
        _layernorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
    }

    /// - Parameter image: `[1, inputHeight, inputWidth, 3]`.
    /// - Returns: `[1, gridHeight, gridWidth, hidden]`.
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        let tokens = layernorm(encoder(embeddings(image)))
        return tokens.reshaped([tokens.shape[0], configuration.gridHeight, configuration.gridWidth,
                                configuration.hiddenSize])
    }
}

// MARK: - Heads

/// `VitPoseSimpleDecoder`: a ReLU, a bilinear upsample, and one 3×3 convolution.
final class NFKVitPoseSimpleDecoder: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    private let scaleFactor: Int

    init(_ c: NFKMLXVitPoseConfiguration) {
        scaleFactor = c.scaleFactor
        _conv.wrappedValue = Conv2d(inputChannels: c.hiddenSize, outputChannels: c.keypointCount,
                                    kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activated = relu(x)
        let resized = NFKMLXResample.resizeBilinear(activated,
                                                    height: activated.shape[1] * scaleFactor,
                                                    width: activated.shape[2] * scaleFactor)
        return conv(resized)
    }
}

/// `VitPoseClassicDecoder`: two transposed-convolution blocks and a 1×1 convolution, the head
/// SimpleBaseline uses. The transposed convolutions carry no bias; the BatchNorms do.
final class NFKVitPoseClassicDecoder: Module {
    @ModuleInfo(key: "deconv1") var deconv1: ConvTransposed2d
    @ModuleInfo(key: "batchnorm1") var batchnorm1: BatchNorm
    @ModuleInfo(key: "deconv2") var deconv2: ConvTransposed2d
    @ModuleInfo(key: "batchnorm2") var batchnorm2: BatchNorm
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(_ c: NFKMLXVitPoseConfiguration) {
        _deconv1.wrappedValue = ConvTransposed2d(inputChannels: c.hiddenSize, outputChannels: c.decoderChannels,
                                                 kernelSize: 4, stride: 2, padding: 1, bias: false)
        _batchnorm1.wrappedValue = BatchNorm(featureCount: c.decoderChannels)
        _deconv2.wrappedValue = ConvTransposed2d(inputChannels: c.decoderChannels, outputChannels: c.decoderChannels,
                                                 kernelSize: 4, stride: 2, padding: 1, bias: false)
        _batchnorm2.wrappedValue = BatchNorm(featureCount: c.decoderChannels)
        _conv.wrappedValue = Conv2d(inputChannels: c.decoderChannels, outputChannels: c.keypointCount, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(batchnorm1(deconv1(x)))
        out = relu(batchnorm2(deconv2(out)))
        return conv(out)
    }
}

// MARK: - Model

/// The whole model: `backbone` + `head`.
final class NFKMLXVitPoseNet: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKVitPoseBackbone
    @ModuleInfo(key: "head") var head: Module

    let configuration: NFKMLXVitPoseConfiguration

    init(_ c: NFKMLXVitPoseConfiguration) {
        configuration = c
        _backbone.wrappedValue = NFKVitPoseBackbone(c)
        _head.wrappedValue = c.decoder == .simple ? NFKVitPoseSimpleDecoder(c) : NFKVitPoseClassicDecoder(c)
    }

    /// The heatmaps for a prepared `[1, inputHeight, inputWidth, 3]` image: `[1, h, w, keypoints]`.
    func heatmaps(_ image: MLXArray) -> MLXArray {
        let features = backbone(image)
        if let simple = head as? NFKVitPoseSimpleDecoder { return simple(features) }
        return (head as! NFKVitPoseClassicDecoder)(features)
    }

    /// The image processor's normalization: rescale to `0...1`, then ImageNet statistics.
    ///
    /// The bridged image already arrives in `0...1`, so only the statistics are applied here.
    static func normalized(_ image: MLXArray) -> MLXArray {
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225])
        return (image - mean) / standardDeviation
    }

    /// Resizes a bridged `[H, W, 3]` image to the trained crop and normalizes it.
    ///
    /// The reference warps a person's box through an affine transform; a whole-image caller has no
    /// box, so this resizes, which is what the SimpleBaseline backend here also does.
    func prepared(_ image: MLXArray) -> MLXArray {
        let batched = image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]])
        let resized = NFKMLXResample.resizeBilinear(batched, height: configuration.inputHeight,
                                                    width: configuration.inputWidth)
        return NFKMLXVitPoseNet.normalized(resized)
    }

    /// Estimates the pose in a bridged `[H, W, 3]` image, positions normalized `0...1` from the top left.
    ///
    /// The Newton refinement is faithful to the reference, which lets a flat neighbourhood move a peak
    /// past its own heatmap. `NFKKeypoint` promises a normalized position, so the result is clamped
    /// here rather than in the decode the parity test measures.
    func estimate(_ image: MLXArray, jointNames: [String]?) -> [NFKKeypoint] {
        let maps = heatmaps(prepared(image))
        return NFKVitPoseDecoding.keypoints(from: maps, jointNames: jointNames).map { keypoint in
            let x = Swift.min(Swift.max(keypoint.position.x, 0), 1)
            let y = Swift.min(Swift.max(keypoint.position.y, 0), 1)
            guard x != keypoint.position.x || y != keypoint.position.y else { return keypoint }
            return NFKKeypoint(name: keypoint.name, index: keypoint.index,
                               position: CGPoint(x: x, y: y), confidence: keypoint.confidence)
        }
    }
}

// MARK: - Decoding

/// The DARK decode ViTPose ships with (`post_dark_unbiased_data_processing`).
///
/// SimpleBaseline nudges the peak a quarter of a cell toward its larger neighbour. ViTPose instead
/// blurs the heatmap, takes its logarithm, and refines the peak by one Newton step against the local
/// derivative and Hessian, which places a keypoint between cells rather than on a quarter grid.
///
/// Papers: Zhang et al., *Distribution-Aware Coordinate Representation for Human Pose Estimation*, and
/// Huang et al., *The Devil is in the Details*, both CVPR 2020.
enum NFKVitPoseDecoding {
    /// The Gaussian radius the release's own decode uses.
    ///
    /// `VitPoseImageProcessor.keypoints_from_heatmaps` defaults its `kernel` to **11**, which is
    /// `radius = (kernel - 1) / 2 = 5`. `post_dark_unbiased_data_processing`'s own default is 3, so
    /// reading that signature instead of the caller's gives a three-tap blur and a decode that is
    /// close but wrong.
    static let blurRadius = 5

    /// scipy's `_gaussian_kernel1d(sigma: 0.8, radius:)`: `exp(-x² / 2σ²)` over `-radius...radius`,
    /// normalized over exactly those taps.
    static let blurWeights: [Float] = {
        let sigma: Float = 0.8
        let raw = (-blurRadius ... blurRadius).map { expf(-0.5 * Float($0 * $0) / (sigma * sigma)) }
        let total = raw.reduce(0, +)
        return raw.map { $0 / total }
    }()

    /// Blurs each heatmap separably, clamps it away from zero, and takes its logarithm, which is what
    /// the Newton step differentiates.
    static func modulated(_ maps: MLXArray) -> MLXArray {
        let weights = MLXArray(blurWeights)
        var blurred = separable(maps, weights: weights, axis: 1)
        blurred = separable(blurred, weights: weights, axis: 2)
        return log(clip(blurred, min: MLXArray(Float(0.001)), max: MLXArray(Float(50))))
    }

    /// One separable pass along `axis` (1 for rows, 2 for columns) over `[1, h, w, K]`.
    ///
    /// scipy's default `reflect` mode is SYMMETRIC — `(d c b a | a b c d | d c b a)`, which repeats the
    /// edge sample — where its `mirror` mode is the reflection that does not. MLX pads with a constant
    /// or the edge value only, so the border is built by gathering reflected indices.
    private static func separable(_ x: MLXArray, weights: MLXArray, axis: Int) -> MLXArray {
        let size = x.shape[axis]
        let radius = blurRadius
        var indices = [Int32]()
        indices.reserveCapacity(size + 2 * radius)
        for offset in 0 ..< (size + 2 * radius) {
            var index = offset - radius
            while index < 0 || index >= size {
                if index < 0 { index = -index - 1 }
                if index >= size { index = 2 * size - index - 1 }
            }
            indices.append(Int32(index))
        }
        let gathered = take(x, MLXArray(indices), axis: axis)
        var total: MLXArray?
        for offset in 0 ..< (2 * radius + 1) {
            let slice = axis == 1
                ? gathered[0..., offset ..< (offset + size), 0..., 0...]
                : gathered[0..., 0..., offset ..< (offset + size), 0...]
            let term = slice * weights[offset]
            total = total.map { $0 + term } ?? term
        }
        return total!
    }

    /// The integer peak of each heatmap, which the Newton step refines from.
    static func peaks(from maps: MLXArray) -> [(row: Int, column: Int, value: Float)] {
        eval(maps)
        let (height, width, count) = (maps.shape[1], maps.shape[2], maps.shape[3])
        let raw = maps.reshaped([-1]).asArray(Float.self)
        return (0 ..< count).map { joint in
            var bestRow = 0, bestColumn = 0
            var best = -Float.greatestFiniteMagnitude
            for row in 0 ..< height {
                for column in 0 ..< width {
                    let value = raw[(row * width + column) * count + joint]
                    if value > best { best = value; bestRow = row; bestColumn = column }
                }
            }
            return (bestRow, bestColumn, best)
        }
    }

    /// Reads the keypoints out of `[1, h, w, keypoints]` heatmaps.
    static func keypoints(from maps: MLXArray, jointNames: [String]?) -> [NFKKeypoint] {
        let modulatedMaps = modulated(maps)
        eval(maps, modulatedMaps)
        let (height, width) = (maps.shape[1], maps.shape[2])
        let count = maps.shape[3]
        let refined = modulatedMaps.reshaped([-1]).asArray(Float.self)

        // The Newton step reads a 3×3 neighbourhood, and the reference pads the modulated map by one
        // with the edge sample, so an out-of-range read clamps rather than wrapping.
        func logValue(_ row: Int, _ column: Int, _ joint: Int) -> Float {
            let r = Swift.min(Swift.max(row, 0), height - 1)
            let c = Swift.min(Swift.max(column, 0), width - 1)
            return refined[(r * width + c) * count + joint]
        }

        var keypoints = [NFKKeypoint]()
        keypoints.reserveCapacity(count)
        for (joint, peak) in peaks(from: maps).enumerated() {
            let (bestRow, bestColumn, best) = peak

            let centre = logValue(bestRow, bestColumn, joint)
            let right = logValue(bestRow, bestColumn + 1, joint)
            let left = logValue(bestRow, bestColumn - 1, joint)
            let below = logValue(bestRow + 1, bestColumn, joint)
            let above = logValue(bestRow - 1, bestColumn, joint)
            let belowRight = logValue(bestRow + 1, bestColumn + 1, joint)
            let aboveLeft = logValue(bestRow - 1, bestColumn - 1, joint)

            let dx = 0.5 * (right - left)
            let dy = 0.5 * (below - above)
            // The reference adds a float32 epsilon to the diagonal before inverting.
            let epsilon = Float.ulpOfOne
            let dxx = right - 2 * centre + left + epsilon
            let dyy = below - 2 * centre + above + epsilon
            let dxy = 0.5 * (belowRight - right - below + centre + centre - left - above + aboveLeft)

            var shiftX: Float = 0, shiftY: Float = 0
            let determinant = dxx * dyy - dxy * dxy
            if determinant != 0 {
                // The 2×2 inverse in closed form, then `coords -= H⁻¹ d`.
                shiftX = -(dyy * dx - dxy * dy) / determinant
                shiftY = -(-dxy * dx + dxx * dy) / determinant
            }
            // A flat neighbourhood can shift a peak by more than a cell, and the reference lets it:
            // clamping the magnitude to one cell disagreed with the reference by 1.04 cells on the
            // classic-decoder release. Only a non-finite result is refused, because a NaN position is
            // not something a caller can use.
            if !shiftX.isFinite || !shiftY.isFinite {
                shiftX = 0
                shiftY = 0
            }

            let position = CGPoint(x: (CGFloat(bestColumn) + CGFloat(shiftX)) / CGFloat(width),
                                   y: (CGFloat(bestRow) + CGFloat(shiftY)) / CGFloat(height))
            let name = jointNames.flatMap { joint < $0.count ? $0[joint] : nil }
            keypoints.append(NFKKeypoint(name: name, index: joint, position: position,
                                         confidence: Double(Swift.min(Swift.max(best, 0), 1))))
        }
        return keypoints
    }
}

// MARK: - Backend

private final class NFKVitPoseHolder: @unchecked Sendable {
    let net: NFKMLXVitPoseNet
    let jointNames: [String]?
    init(net: NFKMLXVitPoseNet, jointNames: [String]?) {
        self.net = net
        self.jointNames = jointNames
    }
}

/// Runs ViTPose through the InferKit contract: `NFKInputImage` in, `NFKOutputPose` out.
@objc(NFKMLXVitPoseBackend)
public final class NFKMLXVitPoseBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKVitPoseHolder
    private let identifier: String

    init(net: NFKMLXVitPoseNet, identifier: String, jointNames: [String]?) {
        holder = NFKVitPoseHolder(net: net, jointNames: jointNames)
        self.identifier = identifier
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result { return result }
        if let error = job.error { throw error }
        throw NFKMLXError.noOutput
    }

    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let holder = holder
        Task.detached(priority: .userInitiated) {
            do {
                guard let value = request.input(forKey: NFKInputImage) else {
                    throw NFKMLXError.unsupportedInput
                }
                let image = try NFKMLXImageBridge.tensor(from: value, channels: 3,
                                                         colorSpace: CGColorSpaceCreateDeviceRGB())
                let pose = holder.net.estimate(image, jointNames: holder.jointNames)
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputPose: pose]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

// MARK: - Public surface

/// The size a ViTPose release ships at.
///
/// - Since: InferKit 0.4.0
@objc(NFKMLXVitPoseVariant)
public enum NFKMLXVitPoseVariant: Int {
    /// `usyd-community/vitpose-base-simple`.
    case baseSimple
    /// `usyd-community/vitpose-base`, the same backbone under the classic decoder.
    case base
}

/// ViTPose: a plain ViT backbone under a small decoding head, at reference parity against
/// transformers' own `VitPoseForPoseEstimation`.
///
/// - Since: InferKit 0.4.0
@objc(NFKMLXVitPose)
public final class NFKMLXVitPose: NSObject {
    /// The registry name the base-simple release builds under.
    @objc public static let modelName = "vitpose-base-simple"

    static func specs(for variant: NFKMLXVitPoseVariant) -> (name: String, configuration: NFKMLXVitPoseConfiguration) {
        switch variant {
        case .baseSimple: return (modelName, .baseSimple)
        case .base: return ("vitpose-base", .base)
        }
    }

    static func makeNet(_ configuration: NFKMLXVitPoseConfiguration = .baseSimple) -> NFKMLXVitPoseNet {
        let net = NFKMLXVitPoseNet(configuration)
        net.train(false)                                                // BatchNorm running statistics
        return net
    }

    /// Builds a pose backend from optional local weights — no registry required. A nil `weightsURL`
    /// builds random weights (`isReady` is true). Run inference off the render thread.
    @objc(backendWithWeightsURL:jointNames:error:)
    public static func backend(weightsURL: URL?, jointNames: [String]?) throws -> any NFKInferenceBackend {
        try backend(variant: .baseSimple, weightsURL: weightsURL, jointNames: jointNames)
    }

    /// Builds one of the released sizes from optional local weights.
    @objc(backendWithVariant:weightsURL:jointNames:error:)
    public static func backend(variant: NFKMLXVitPoseVariant, weightsURL: URL?,
                               jointNames: [String]?) throws -> any NFKInferenceBackend {
        let spec = specs(for: variant)
        let net = makeNet(spec.configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
            net.train(false)
        }
        return NFKMLXVitPoseBackend(net: net, identifier: spec.name, jointNames: jointNames)
    }

    /// Builds from a release directory, reading its own `config.json` for the geometry.
    @objc(backendWithDirectoryURL:jointNames:error:)
    public static func backend(directoryURL: URL, jointNames: [String]?) throws -> any NFKInferenceBackend {
        let configuration = try NFKMLXVitPoseConfiguration.configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        let net = makeNet(configuration)
        try loadWeights(into: net, from: directoryURL.appendingPathComponent("model.safetensors"))
        net.train(false)
        return NFKMLXVitPoseBackend(net: net, identifier: modelName, jointNames: jointNames)
    }

    /// Downloads the checkpoint from Hugging Face, then builds. Blocking on the network; run off the
    /// render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:jointNames:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, jointNames: [String]?) throws -> any NFKInferenceBackend {
        try backend(variant: .baseSimple, repo: repo, weightsPath: weightsPath, revision: revision,
                    cacheDirectoryURL: cacheDirectoryURL, jointNames: jointNames)
    }

    /// The download factory at a chosen size.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:jointNames:error:)
    public static func backend(variant: NFKMLXVitPoseVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               jointNames: [String]?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url, jointNames: jointNames)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:jointNames:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, jointNames: [String]?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        backend(variant: .baseSimple, repo: repo, weightsPath: weightsPath, revision: revision,
                cacheDirectoryURL: cacheDirectoryURL, jointNames: jointNames,
                completionHandler: completionHandler)
    }

    /// The asynchronous download factory at a chosen size.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:jointNames:completionHandler:)
    public static func backend(variant: NFKMLXVitPoseVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?, jointNames: [String]?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0, jointNames: jointNames) },
                               completionHandler: completionHandler)
    }

    /// Registers the released sizes with `NFKMLXModelRegistry`.
    @objc public static func register() {
        for variant in [NFKMLXVitPoseVariant.baseSimple, .base] {
            NFKMLXModelRegistry.register(name: specs(for: variant).name) { weightsURL in
                try backend(variant: variant, weightsURL: weightsURL, jointNames: nil)
            }
        }
    }

    /// Loads a released checkpoint. The module keys are the checkpoint's, so only the convolution
    /// layouts move: a forward convolution to `[out, kH, kW, in]` and a transposed one to
    /// `[out, kH, kW, in]` from PyTorch's `[in, out, kH, kW]`, a different axis order.
    static func loadWeights(into net: NFKMLXVitPoseNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var mapped = [(String, MLXArray)]()
        for (key, value) in checkpoint.arrays {
            guard !key.hasSuffix("num_batches_tracked") else { continue }
            let array: MLXArray
            if checkpoint.needsConvTranspose && value.ndim == 4 {
                array = key.contains("deconv") ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1)
            } else {
                array = value
            }
            mapped.append((key, array))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
