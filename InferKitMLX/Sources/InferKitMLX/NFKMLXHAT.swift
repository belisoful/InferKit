//
//  NFKMLXHAT.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// HAT (Hybrid Attention Transformer) is SwinIR's successor for image super-resolution. It keeps the
// window self-attention and the residual group structure and adds two things: a channel-attention
// convolution branch inside every block, and an overlapping cross-attention block at the end of every
// group whose keys and values come from a window wider than the query's. The window attention, the
// window partitioning, and the shifted-window mask are SwinIR's, so `NFKSwinOps` and
// `NFKSwinWindowAttention` carry over unchanged. Tensors flow NHWC.

/// The geometry of a released HAT, which the presets name.
public struct NFKMLXHATConfiguration: Sendable {
    public var dimensions: Int
    public var groups: Int
    public var blocksPerGroup: Int
    public var heads: Int
    public var windowSize: Int
    public var compressRatio: Int
    public var squeezeFactor: Int
    public var convScale: Float
    public var overlapRatio: Float
    public var mlpRatio: Float
    public var upscale: Int
    public var features: Int

    public init(dimensions: Int = 180, groups: Int = 6, blocksPerGroup: Int = 6, heads: Int = 6,
                windowSize: Int = 16, compressRatio: Int = 3, squeezeFactor: Int = 30,
                convScale: Float = 0.01, overlapRatio: Float = 0.5, mlpRatio: Float = 2,
                upscale: Int = 4, features: Int = 64) {
        self.dimensions = dimensions
        self.groups = groups
        self.blocksPerGroup = blocksPerGroup
        self.heads = heads
        self.windowSize = windowSize
        self.compressRatio = compressRatio
        self.squeezeFactor = squeezeFactor
        self.convScale = convScale
        self.overlapRatio = overlapRatio
        self.mlpRatio = mlpRatio
        self.upscale = upscale
        self.features = features
    }

    /// The window a key/value reaches over, which is wider than the query's by the overlap ratio.
    public var overlapWindowSize: Int { Int(Float(windowSize) * overlapRatio) + windowSize }

    /// HAT: the base release, 6 groups of 6 blocks at 180 channels.
    public static let base = NFKMLXHATConfiguration()

    /// HAT-L: the large release, 12 groups of 6 blocks at 180 channels.
    public static let large = NFKMLXHATConfiguration(groups: 12)

    /// Real-HAT-GAN: the base geometry trained for real-world degradation.
    public static let realWorld = NFKMLXHATConfiguration()
}

/// The relative-position index tables HAT precomputes. The reference registers both as buffers; they
/// are pure functions of the window geometry, so they are recomputed rather than loaded.
enum NFKHATIndices {

    /// The overlapping-attention index: `[ws·ws · wse·wse]` into a `(ws + wse − 1)²` bias table.
    ///
    /// The reference's own arithmetic leaves negative entries in this table, and PyTorch reads a
    /// negative index from the end of the tensor. The span of the raw values is exactly the table
    /// size, so taking the index modulo that size reproduces the wraparound the reference relies on.
    static func overlapPositionIndex(windowSize ws: Int, overlapWindowSize wse: Int) -> [Int32] {
        let (queries, keys) = (ws * ws, wse * wse)
        let stride = ws + wse - 1
        let table = stride * stride
        var index = [Int32](repeating: 0, count: queries * keys)
        for a in 0 ..< queries {
            let (ai, aj) = (a / ws, a % ws)
            for b in 0 ..< keys {
                let (bi, bj) = (b / wse, b % wse)
                let dh = bi - ai + ws - wse + 1
                let dw = bj - aj + ws - wse + 1
                let raw = dh * stride + dw
                index[a * keys + b] = Int32(((raw % table) + table) % table)
            }
        }
        return index
    }
}

/// RCAN's channel attention: pool to one value per channel, squeeze, excite, and gate.
final class NFKHATChannelAttention: Module {
    @ModuleInfo(key: "attention") var attention: [Module]

    init(dimensions: Int, squeezeFactor: Int) {
        let narrow = dimensions / squeezeFactor
        // Slots 0, 2 and 4 are the pool, the ReLU and the sigmoid, which hold their Sequential index.
        _attention.wrappedValue = [
            Module(),
            Conv2d(inputChannels: dimensions, outputChannels: narrow, kernelSize: 1),
            Module(),
            Conv2d(inputChannels: narrow, outputChannels: dimensions, kernelSize: 1),
            Module(),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let squeeze = attention[1] as? Conv2d, let excite = attention[3] as? Conv2d else { return x }
        let pooled = x.mean(axes: [1, 2], keepDims: true)
        return x * sigmoid(excite(relu(squeeze(pooled))))
    }
}

/// The convolution branch HAT adds beside window attention: two 3×3 convolutions through a narrow
/// waist, then channel attention.
final class NFKHATConvBlock: Module {
    @ModuleInfo(key: "cab") var cab: [Module]

    init(dimensions: Int, compressRatio: Int, squeezeFactor: Int) {
        let narrow = dimensions / compressRatio
        _cab.wrappedValue = [
            Conv2d(inputChannels: dimensions, outputChannels: narrow, kernelSize: 3, padding: 1),
            Module(),                                           // the GELU, which holds index 1
            Conv2d(inputChannels: narrow, outputChannels: dimensions, kernelSize: 3, padding: 1),
            NFKHATChannelAttention(dimensions: dimensions, squeezeFactor: squeezeFactor),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let compress = cab[0] as? Conv2d, let expand = cab[2] as? Conv2d,
              let attention = cab[3] as? NFKHATChannelAttention else { return x }
        return attention(expand(gelu(compress(x))))
    }
}

/// The two-layer MLP every HAT block ends with.
final class NFKHATMlp: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(dimensions: Int, hidden: Int) {
        _fc1.wrappedValue = Linear(dimensions, hidden)
        _fc2.wrappedValue = Linear(hidden, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// A Hybrid Attention Block: window self-attention and the channel-attention convolution branch,
/// summed into the residual, then the MLP.
final class NFKHATBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NFKSwinWindowAttention
    @ModuleInfo(key: "conv_block") var convBlock: NFKHATConvBlock
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKHATMlp

    private let windowSize: Int
    private let shift: Int
    private let convScale: Float

    init(_ configuration: NFKMLXHATConfiguration, shifted: Bool) {
        windowSize = configuration.windowSize
        shift = shifted ? configuration.windowSize / 2 : 0
        convScale = configuration.convScale
        let dimensions = configuration.dimensions
        _norm1.wrappedValue = LayerNorm(dimensions: dimensions)
        _attn.wrappedValue = NFKSwinWindowAttention(dimensions: dimensions, heads: configuration.heads,
                                                    windowSize: configuration.windowSize)
        _convBlock.wrappedValue = NFKHATConvBlock(dimensions: dimensions,
                                                  compressRatio: configuration.compressRatio,
                                                  squeezeFactor: configuration.squeezeFactor)
        _norm2.wrappedValue = LayerNorm(dimensions: dimensions)
        _mlp.wrappedValue = NFKHATMlp(dimensions: dimensions,
                                      hidden: Int(Float(dimensions) * configuration.mlpRatio))
    }

    /// `x`: `[1, H·W, C]`.
    func callAsFunction(_ x: MLXArray, height: Int, width: Int, mask: MLXArray?) -> MLXArray {
        let dimensions = x.shape[2]
        let normed = norm1(x).reshaped([1, height, width, dimensions])
        let convolved = convBlock(normed).reshaped([1, height * width, dimensions])

        let shifted = NFKSwinOps.roll(normed, shift: shift)
        let windows = NFKSwinOps.partition(shifted, windowSize: windowSize)
        let attended = attn(windows, mask: shift > 0 ? mask : nil)
        let merged = NFKSwinOps.reverse(attended, windowSize: windowSize, height: height, width: width)
        let restored = NFKSwinOps.unroll(merged, shift: shift).reshaped([1, height * width, dimensions])

        let residual = x + restored + convolved * convScale
        return residual + mlp(norm2(residual))
    }
}

/// The overlapping cross-attention block that closes every group: queries from a window, keys and
/// values from a wider window centered on it.
final class NFKHATOverlapBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "relative_position_bias_table") var biasTable: MLXArray
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKHATMlp

    private let heads: Int
    private let windowSize: Int
    private let overlapWindowSize: Int
    private let indexBox: NFKSwinIndexBox

    /// The relative-position index, held indirectly so MLX's parameter reflection does not report it
    /// as a missing weight (a real checkpoint carries it as a buffer, not a parameter).
    var index: MLXArray { indexBox.value }

    init(_ configuration: NFKMLXHATConfiguration) {
        heads = configuration.heads
        windowSize = configuration.windowSize
        overlapWindowSize = configuration.overlapWindowSize
        let dimensions = configuration.dimensions
        let stride = windowSize + overlapWindowSize - 1
        _norm1.wrappedValue = LayerNorm(dimensions: dimensions)
        _qkv.wrappedValue = Linear(dimensions, dimensions * 3)
        _biasTable.wrappedValue = NFKCodeFormerOps.parameter([stride * stride, configuration.heads])
        _proj.wrappedValue = Linear(dimensions, dimensions)
        _norm2.wrappedValue = LayerNorm(dimensions: dimensions)
        _mlp.wrappedValue = NFKHATMlp(dimensions: dimensions,
                                      hidden: Int(Float(dimensions) * configuration.mlpRatio))
        indexBox = NFKSwinIndexBox(MLXArray(NFKHATIndices.overlapPositionIndex(
            windowSize: windowSize, overlapWindowSize: overlapWindowSize)))
    }

    /// The reference's `nn.Unfold(kernel_size: wse, stride: ws, padding: (wse − ws) / 2)` over a
    /// `[1, H, W, C]` map, returning `[numWindows, wse·wse, C]` in the unfold's own window order.
    private func overlappingWindows(_ x: MLXArray) -> MLXArray {
        let pad = (overlapWindowSize - windowSize) / 2
        let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(pad), IntOrPair(pad), IntOrPair(0)],
                                mode: .constant, value: MLXArray(Float(0)))
        let (rows, columns) = (x.shape[1] / windowSize, x.shape[2] / windowSize)
        func offsets(_ count: Int) -> MLXArray {
            var values = [Int32]()
            for window in 0 ..< count {
                for step in 0 ..< overlapWindowSize {
                    values.append(Int32(window * windowSize + step))
                }
            }
            return MLXArray(values, [count, overlapWindowSize])
        }
        // The first gather replaces the height axis with two, which moves the width axis to 3.
        let gathered = padded.take(offsets(rows), axis: 1).take(offsets(columns), axis: 3)
        // [1, rows, wse, columns, wse, C] -> [rows·columns, wse·wse, C]
        return gathered.transposed(0, 1, 3, 2, 4, 5)
            .reshaped([rows * columns, overlapWindowSize * overlapWindowSize, x.shape[3]])
    }

    /// `x`: `[1, H·W, C]`.
    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let dimensions = x.shape[2]
        let headDimensions = dimensions / heads
        let scale = 1.0 / sqrtf(Float(headDimensions))
        let normed = norm1(x).reshaped([1, height, width, dimensions])

        // The reference reads the projection as `(b, h, w, 3, c)`, so the three parts are consecutive
        // spans of the channel axis.
        let projected = qkv(normed)
        let queries = NFKSwinOps.partition(projected[0..., 0..., 0..., 0 ..< dimensions],
                                           windowSize: windowSize)
        let keys = overlappingWindows(projected[0..., 0..., 0..., dimensions ..< (2 * dimensions)])
        let values = overlappingWindows(projected[0..., 0..., 0..., (2 * dimensions) ..< (3 * dimensions)])

        let windows = queries.shape[0]
        let (queryCount, keyCount) = (queries.shape[1], keys.shape[1])
        let q = queries.reshaped([windows, queryCount, heads, headDimensions]).transposed(0, 2, 1, 3)
        let k = keys.reshaped([windows, keyCount, heads, headDimensions]).transposed(0, 2, 1, 3)
        let v = values.reshaped([windows, keyCount, heads, headDimensions]).transposed(0, 2, 1, 3)

        var attn = q.matmul(k.transposed(0, 1, 3, 2)) * scale
        let bias = biasTable[index].reshaped([queryCount, keyCount, heads]).transposed(2, 0, 1)
        attn = softmax(attn + bias, axis: -1)
        let attended = attn.matmul(v).transposed(0, 2, 1, 3).reshaped([windows, queryCount, dimensions])

        let merged = NFKSwinOps.reverse(attended, windowSize: windowSize, height: height, width: width)
            .reshaped([1, height * width, dimensions])
        let residual = proj(merged) + x
        return residual + mlp(norm2(residual))
    }
}

/// The blocks of one group plus its overlapping cross-attention block.
final class NFKHATAttentionBlocks: Module {
    @ModuleInfo(key: "blocks") var blocks: [NFKHATBlock]
    @ModuleInfo(key: "overlap_attn") var overlapAttention: NFKHATOverlapBlock

    init(_ configuration: NFKMLXHATConfiguration) {
        _blocks.wrappedValue = (0 ..< configuration.blocksPerGroup).map {
            NFKHATBlock(configuration, shifted: $0 % 2 == 1)
        }
        _overlapAttention.wrappedValue = NFKHATOverlapBlock(configuration)
    }

    func callAsFunction(_ x: MLXArray, height: Int, width: Int, mask: MLXArray?) -> MLXArray {
        var out = x
        for block in blocks {
            out = block(out, height: height, width: width, mask: mask)
        }
        return overlapAttention(out, height: height, width: width)
    }
}

/// A Residual Hybrid Attention Group: the blocks, a 3×3 convolution, and the group's own residual.
final class NFKHATResidualGroup: Module {
    @ModuleInfo(key: "residual_group") var residualGroup: NFKHATAttentionBlocks
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(_ configuration: NFKMLXHATConfiguration) {
        _residualGroup.wrappedValue = NFKHATAttentionBlocks(configuration)
        _conv.wrappedValue = Conv2d(inputChannels: configuration.dimensions,
                                    outputChannels: configuration.dimensions, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray, height: Int, width: Int, mask: MLXArray?) -> MLXArray {
        let dimensions = x.shape[2]
        let attended = residualGroup(x, height: height, width: width, mask: mask)
        let convolved = conv(attended.reshaped([1, height, width, dimensions]))
        return convolved.reshaped([1, height * width, dimensions]) + x
    }
}

/// The HAT super-resolution network. Input `[1, H, W, 3]` in `0...1` → `[1, H·s, W·s, 3]`.
final class NFKMLXHATNet: Module {
    @ModuleInfo(key: "conv_first") var convFirst: Conv2d
    @ModuleInfo(key: "patch_embed") var patchEmbed: NFKHATPatchNorm
    @ModuleInfo(key: "layers") var layers: [NFKHATResidualGroup]
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "conv_after_body") var convAfterBody: Conv2d
    @ModuleInfo(key: "conv_before_upsample") var convBeforeUpsample: [Module]
    @ModuleInfo(key: "upsample") var upsample: [Module]
    @ModuleInfo(key: "conv_last") var convLast: Conv2d

    let configuration: NFKMLXHATConfiguration

    init(_ configuration: NFKMLXHATConfiguration) {
        self.configuration = configuration
        let dimensions = configuration.dimensions
        _convFirst.wrappedValue = Conv2d(inputChannels: 3, outputChannels: dimensions, kernelSize: 3, padding: 1)
        _patchEmbed.wrappedValue = NFKHATPatchNorm(dimensions: dimensions)
        _layers.wrappedValue = (0 ..< configuration.groups).map { _ in NFKHATResidualGroup(configuration) }
        _norm.wrappedValue = LayerNorm(dimensions: dimensions)
        _convAfterBody.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimensions,
                                             kernelSize: 3, padding: 1)
        _convBeforeUpsample.wrappedValue = [
            Conv2d(inputChannels: dimensions, outputChannels: configuration.features, kernelSize: 3, padding: 1),
            Module(),                                           // the leaky ReLU, which holds index 1
        ]
        var stages = [Module]()
        var remaining = configuration.upscale
        while remaining > 1 {
            let factor = remaining % 3 == 0 && remaining == 3 ? 3 : 2
            stages.append(Conv2d(inputChannels: configuration.features,
                                 outputChannels: configuration.features * factor * factor,
                                 kernelSize: 3, padding: 1))
            stages.append(Module())                             // the pixel shuffle
            remaining /= factor
        }
        _upsample.wrappedValue = stages
        _convLast.wrappedValue = Conv2d(inputChannels: configuration.features, outputChannels: 3,
                                        kernelSize: 3, padding: 1)
    }

    /// The deep-feature trunk, `[1, H, W, C]` in and out. Separated so a parity test can compare it on
    /// its own.
    func features(_ x: MLXArray) -> MLXArray {
        let (height, width) = (x.shape[1], x.shape[2])
        let dimensions = x.shape[3]
        let mask = NFKSwinOps.shiftMask(height: height, width: width,
                                        windowSize: configuration.windowSize,
                                        shift: configuration.windowSize / 2)
        var tokens = patchEmbed(x.reshaped([1, height * width, dimensions]))
        for layer in layers {
            tokens = layer(tokens, height: height, width: width, mask: mask)
        }
        return norm(tokens).reshaped([1, height, width, dimensions])
    }

    func upscale(_ input: MLXArray) -> MLXArray {
        // The per-channel mean the reference subtracts before the network and adds back after. Built
        // per call: an `MLXArray` property would register as a parameter the checkpoint cannot cover.
        let mean = MLXArray([Float(0.4488), 0.4371, 0.4040])
        let normalized = input - mean
        let shallow = convFirst(normalized)
        var out = convAfterBody(features(shallow)) + shallow
        if let conv = convBeforeUpsample[0] as? Conv2d {
            out = leakyRelu(conv(out), negativeSlope: 0.01)
        }
        var index = 0
        while index < upsample.count {
            guard let conv = upsample[index] as? Conv2d else { break }
            let widened = conv(out)
            let factor = Int((Float(widened.shape[3]) / Float(out.shape[3])).squareRoot().rounded())
            out = NFKMLXPixelShuffle.apply(widened, factor: factor)
            index += 2
        }
        return convLast(out) + mean
    }
}

/// The patch embedding, which in HAT is a LayerNorm over the already-flattened features.
final class NFKHATPatchNorm: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(dimensions: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(x) }
}

/// The released HAT geometry, for the Objective-C factory.
@objc(NFKMLXHATVariant)
public enum NFKMLXHATVariant: Int {
    /// HAT ×4.
    case base
    /// HAT-L ×4.
    case large
    /// Real-HAT-GAN ×4, the real-world degradation release.
    case realWorld
}

/// HAT super-resolution as an InferKit backend, and its registration for the Objective-C path.
@objc(NFKMLXHAT)
public final class NFKMLXHAT: NSObject {

    @objc public static let modelName = "hat-x4"
    @objc public static let largeModelName = "hat-l-x4"
    @objc public static let realWorldModelName = "real-hat-gan-x4"

    static func makeNet(_ configuration: NFKMLXHATConfiguration = .large) -> NFKMLXHATNet {
        NFKMLXHATNet(configuration)
    }

    static func specs(for variant: NFKMLXHATVariant) -> (name: String, configuration: NFKMLXHATConfiguration) {
        switch variant {
        case .base: return (modelName, .base)
        case .large: return (largeModelName, .large)
        case .realWorld: return (realWorldModelName, .realWorld)
        }
    }

    /// Builds a HAT super-resolution backend directly from optional local weights — no registry
    /// required. A nil `weightsURL` builds random weights (`isReady` is true).
    /// The input's height and width must be multiples of the window size.
    /// Run inference off the render thread.
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXHATVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let spec = specs(for: variant)
        let net = NFKMLXHATNet(spec.configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXHATHolder(net)
        return NFKMLXModuleBackend(identifier: spec.name, isReady: true) { image in
            let (height, width) = (image.shape[0], image.shape[1])
            let batched = image.reshaped([1, height, width, 3])
            return clip(holder.net.upscale(batched)[0], min: 0, max: 1)
        }
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXHATVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXHATVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers every released HAT geometry with `NFKMLXModelRegistry`.
    @objc public static func register() {
        for variant in [NFKMLXHATVariant.base, .large, .realWorld] {
            NFKMLXModelRegistry.register(name: specs(for: variant).name) { weightsURL in
                try backend(variant: variant, weightsURL: weightsURL)
            }
        }
    }

    /// Loads a released checkpoint, whose tensors live under `params_ema` or `params`. The two
    /// relative-position index buffers are recomputed from the window geometry, so they are dropped.
    static func loadWeights(into net: NFKMLXHATNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            guard !key.hasSuffix("relative_position_index_SA"),
                  !key.hasSuffix("relative_position_index_OCA") else { return nil }
            return (key, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

private final class NFKMLXHATHolder: @unchecked Sendable {
    let net: NFKMLXHATNet
    init(_ net: NFKMLXHATNet) { self.net = net }
}
