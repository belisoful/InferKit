//
//  NFKMLXSwinIR.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// SwinIR super-resolves an image with a Swin Transformer. Shallow features from a convolution feed a
// stack of residual Swin Transformer blocks (RSTB); each block runs window attention, alternating
// regular and shifted windows so information crosses window boundaries, with a relative-position bias
// inside each window. A pixel-shuffle upsampler and a final convolution reconstruct the high-resolution
// image.
//
// This implements the real window attention: window partition/reverse, cyclic shift with the standard
// attention mask, and the relative-position bias table gathered by a precomputed index. Module names
// are grouped cleanly (`layers.N.blocks.M.*`) rather than the reference's
// `layers.N.residual_group.blocks.M.*`, so the exact key remap is a validation-sweep item, as is
// non-power-of-two scaling. Tensors flow in NHWC; the input side must be a multiple of the window size.

/// SwinIR dimensions. Defaults size the lightweight SR model; `tiny` keeps tests fast.
/// The reconstruction tail a SwinIR release was trained with.
public enum NFKMLXSwinIRUpsampler: Sendable {
    /// A convolution, ×2 pixel-shuffle stages, then a final convolution. The classical releases.
    case classical
    /// One convolution to `3·scale²` channels and a single shuffle. The lightweight releases.
    case direct
}

public struct NFKMLXSwinIRConfiguration: Sendable {
    public var embedDimensions: Int
    public var depths: [Int]
    public var heads: Int
    public var windowSize: Int
    public var mlpRatio: Int
    public var scale: Int

    /// Which upsampler the release was trained with.
    ///
    /// The classical models reconstruct through a convolution, one or more ×2 pixel-shuffle stages,
    /// and a final convolution. The lightweight models use the reference's `pixelshuffledirect`: ONE
    /// convolution straight to `3·scale²` channels and a single shuffle, with no surrounding
    /// convolutions at all. A checkpoint carries weights for one or the other, never both.
    public var upsampler: NFKMLXSwinIRUpsampler = .classical

    public init(embedDimensions: Int = 60, depths: [Int] = [6, 6, 6, 6], heads: Int = 6,
                windowSize: Int = 8, mlpRatio: Int = 2, scale: Int = 4) {
        self.embedDimensions = embedDimensions
        self.depths = depths
        self.heads = heads
        self.windowSize = windowSize
        self.mlpRatio = mlpRatio
        self.scale = scale
    }

    /// The pixel-shuffle factor each upsampling stage applies.
    ///
    /// A power-of-two scale is reached by repeated ×2 stages; a scale of three is one ×3 stage, since
    /// the reference's `PixelShuffle(3)` packs channels in an order no pair of ×2 shuffles reproduces.
    /// These are the scales the reference `Upsample` accepts.
    var shuffleFactor: Int { scale == 3 ? 3 : 2 }

    /// How many upsampling stages the scale needs.
    var upsampleStages: Int { scale == 3 ? 1 : Int(log2(Double(scale))) }

    /// Whether the reference builds an upsampler for this scale at all.
    public var isSupportedScale: Bool {
        scale == 3 || (scale > 1 && scale & (scale - 1) == 0)
    }

    /// The released `001_classicalSR_DIV2K_s48w8_SwinIR-M_x8` geometry: the same network as x4 with a
    /// third ×2 pixel-shuffle stage.
    public static let classicalSRx8 = NFKMLXSwinIRConfiguration(embedDimensions: 180,
                                                                depths: [6, 6, 6, 6, 6, 6], heads: 6,
                                                                windowSize: 8, mlpRatio: 2, scale: 8)

    /// The released `002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2` geometry: a narrower, shallower
    /// network with the direct upsampler.
    public static let lightweightSRx2: NFKMLXSwinIRConfiguration = {
        var configuration = NFKMLXSwinIRConfiguration(embedDimensions: 60, depths: [6, 6, 6, 6],
                                                      heads: 6, windowSize: 8, mlpRatio: 2, scale: 2)
        configuration.upsampler = .direct
        return configuration
    }()

    public static let lightweight = NFKMLXSwinIRConfiguration()

    /// The released `001_classicalSR_DIV2K_s48w8_SwinIR-M_x3` geometry. Its upsampler is a single ×3
    /// stage where the ×4 release runs two ×2 stages, so a checkpoint fits only the scale it was
    /// trained for.
    public static let classicalSRx3 = NFKMLXSwinIRConfiguration(embedDimensions: 180,
                                                                depths: [6, 6, 6, 6, 6, 6], heads: 6,
                                                                windowSize: 8, mlpRatio: 2, scale: 3)

    /// The released `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4` geometry.
    public static let classicalSRx4 = NFKMLXSwinIRConfiguration(embedDimensions: 180,
                                                                depths: [6, 6, 6, 6, 6, 6], heads: 6,
                                                                windowSize: 8, mlpRatio: 2, scale: 4)

    // Depth 2 so the second layer runs the shifted-window path (shift + attention mask).
    public static let tiny = NFKMLXSwinIRConfiguration(embedDimensions: 16, depths: [2], heads: 2,
                                                       windowSize: 4, mlpRatio: 2, scale: 2)
}

/// Window operations, cyclic shift, and the shifted-window attention mask, shared by the layers.
enum NFKSwinOps {

    /// Partitions `[1, H, W, C]` into `[numWindows, ws·ws, C]`.
    static func partition(_ x: MLXArray, windowSize ws: Int) -> MLXArray {
        let (h, w, c) = (x.shape[1], x.shape[2], x.shape[3])
        return x.reshaped([1, h / ws, ws, w / ws, ws, c])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([(h / ws) * (w / ws), ws * ws, c])
    }

    /// Reverses `partition`, back to `[1, H, W, C]`.
    static func reverse(_ windows: MLXArray, windowSize ws: Int, height h: Int, width w: Int) -> MLXArray {
        let c = windows.shape[2]
        return windows.reshaped([1, h / ws, w / ws, ws, ws, c])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([1, h, w, c])
    }

    /// Cyclically shifts `[1, H, W, C]` by `shift` along H and W (negative roll, as in the reference).
    static func roll(_ x: MLXArray, shift: Int) -> MLXArray {
        if shift == 0 {
            return x
        }
        let (h, w) = (x.shape[1], x.shape[2])
        let rolledH = concatenated([x[0..., shift ..< h], x[0..., 0 ..< shift]], axis: 1)
        return concatenated([rolledH[0..., 0..., shift ..< w], rolledH[0..., 0..., 0 ..< shift]], axis: 2)
    }

    /// Undoes `roll`.
    static func unroll(_ x: MLXArray, shift: Int) -> MLXArray {
        roll(x, shift: shift == 0 ? 0 : (x.shape[1] - shift))
    }

    /// The relative-position index for a window: `[ws·ws · ws·ws]` into a `(2ws−1)²` bias table.
    static func relativePositionIndex(windowSize ws: Int) -> [Int32] {
        var index = [Int32](repeating: 0, count: ws * ws * ws * ws)
        let n = ws * ws
        for a in 0 ..< n {
            let (ai, aj) = (a / ws, a % ws)
            for b in 0 ..< n {
                let (bi, bj) = (b / ws, b % ws)
                let dh = ai - bi + ws - 1
                let dw = aj - bj + ws - 1
                index[a * n + b] = Int32(dh * (2 * ws - 1) + dw)
            }
        }
        return index
    }

    /// The additive attention mask for shifted windows: `[numWindows, N, N]`, 0 where two positions
    /// share a region and −100 otherwise.
    static func shiftMask(height h: Int, width w: Int, windowSize ws: Int, shift: Int) -> MLXArray {
        let n = ws * ws
        let (nwh, nww) = (h / ws, w / ws)
        func region(_ coord: Int, _ size: Int) -> Int {
            if coord < size - ws { return 0 }
            if coord < size - shift { return 1 }
            return 2
        }
        var values = [Float](repeating: 0, count: nwh * nww * n * n)
        for wy in 0 ..< nwh {
            for wx in 0 ..< nww {
                let window = wy * nww + wx
                var labels = [Int](repeating: 0, count: n)
                for p in 0 ..< n {
                    let gy = wy * ws + p / ws
                    let gx = wx * ws + p % ws
                    labels[p] = region(gy, h) * 3 + region(gx, w)
                }
                for a in 0 ..< n {
                    for b in 0 ..< n {
                        values[(window * n + a) * n + b] = labels[a] == labels[b] ? 0 : -100
                    }
                }
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [nwh * nww, n, n]) }
    }
}

/// Holds a constant tensor out of MLX's parameter reflection (see `NFKSwinWindowAttention.index`).
final class NFKSwinIndexBox {
    let value: MLXArray
    init(_ value: MLXArray) { self.value = value }
}

/// Window multi-head self-attention with a relative-position bias.
final class NFKSwinWindowAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "relative_position_bias_table") var biasTable: MLXArray
    private let heads: Int
    private let indexBox: NFKSwinIndexBox

    /// The relative-position index is a precomputed constant, not a learned weight. Held indirectly
    /// because MLX's module reflection registers every `MLXArray` property as a parameter, which would
    /// make a real checkpoint look incomplete (it carries no such tensor).
    var index: MLXArray { indexBox.value }

    init(dimensions: Int, heads: Int, windowSize ws: Int) {
        self.heads = heads
        _qkv.wrappedValue = Linear(dimensions, dimensions * 3)
        _proj.wrappedValue = Linear(dimensions, dimensions)
        _biasTable.wrappedValue = NFKCodeFormerOps.parameter([(2 * ws - 1) * (2 * ws - 1), heads])
        indexBox = NFKSwinIndexBox(MLXArray(NFKSwinOps.relativePositionIndex(windowSize: ws)))
    }

    /// `x`: `[numWindows, N, C]`. `mask`: optional `[numWindows, N, N]`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (windows, n, dimensions) = (x.shape[0], x.shape[1], x.shape[2])
        let headDim = dimensions / heads
        let scale = 1.0 / sqrtf(Float(headDim))

        let triple = qkv(x).reshaped([windows, n, 3, heads, headDim]).transposed(2, 0, 3, 1, 4)
        let q = triple[0]                                       // [windows, heads, N, headDim]
        let k = triple[1]
        let v = triple[2]

        var attn = q.matmul(k.transposed(0, 1, 3, 2)) * scale   // [windows, heads, N, N]
        let bias = biasTable[index].reshaped([n, n, heads]).transposed(2, 0, 1)   // [heads, N, N]
        attn = attn + bias
        if let mask {
            attn = attn + mask.reshaped([windows, 1, n, n])
        }
        attn = softmax(attn, axis: -1)
        let out = attn.matmul(v).transposed(0, 2, 1, 3).reshaped([windows, n, dimensions])
        return proj(out)
    }
}

/// One Swin Transformer layer: pre-norm window attention (regular or shifted) and a pre-norm MLP.
final class NFKSwinLayer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NFKSwinWindowAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp_fc1") var fc1: Linear
    @ModuleInfo(key: "mlp_fc2") var fc2: Linear

    private let windowSize: Int
    private let shift: Int

    init(dimensions: Int, heads: Int, windowSize: Int, shift: Int, mlpRatio: Int) {
        self.windowSize = windowSize
        self.shift = shift
        _norm1.wrappedValue = LayerNorm(dimensions: dimensions)
        _attn.wrappedValue = NFKSwinWindowAttention(dimensions: dimensions, heads: heads, windowSize: windowSize)
        _norm2.wrappedValue = LayerNorm(dimensions: dimensions)
        _fc1.wrappedValue = Linear(dimensions, dimensions * mlpRatio)
        _fc2.wrappedValue = Linear(dimensions * mlpRatio, dimensions)
    }

    /// `x`: `[1, H·W, C]`.
    func callAsFunction(_ x: MLXArray, height h: Int, width w: Int) -> MLXArray {
        let dimensions = x.shape[2]
        let shortcut = x
        var feature = norm1(x).reshaped([1, h, w, dimensions])
        feature = NFKSwinOps.roll(feature, shift: shift)

        let windows = NFKSwinOps.partition(feature, windowSize: windowSize)
        let mask = shift > 0 ? NFKSwinOps.shiftMask(height: h, width: w, windowSize: windowSize, shift: shift) : nil
        let attended = attn(windows, mask: mask)

        var merged = NFKSwinOps.reverse(attended, windowSize: windowSize, height: h, width: w)
        merged = NFKSwinOps.unroll(merged, shift: shift)
        let afterAttention = shortcut + merged.reshaped([1, h * w, dimensions])
        return afterAttention + fc2(gelu(fc1(norm2(afterAttention))))
    }
}

/// A residual Swin Transformer block: a run of Swin layers, a convolution, and a residual connection.
final class NFKSwinRSTB: Module {
    @ModuleInfo(key: "blocks") var blocks: [NFKSwinLayer]
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(dimensions: Int, depth: Int, heads: Int, windowSize: Int, mlpRatio: Int) {
        _blocks.wrappedValue = (0 ..< depth).map { i in
            NFKSwinLayer(dimensions: dimensions, heads: heads, windowSize: windowSize,
                         shift: i % 2 == 0 ? 0 : windowSize / 2, mlpRatio: mlpRatio)
        }
        _conv.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimensions, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray, height h: Int, width w: Int) -> MLXArray {
        let dimensions = x.shape[2]
        var tokens = x
        for block in blocks {
            tokens = block(tokens, height: h, width: w)
        }
        let spatial = conv(tokens.reshaped([1, h, w, dimensions]))
        return x + spatial.reshaped([1, h * w, dimensions])
    }
}

/// The SwinIR network: shallow feature convolution, RSTB body, and a pixel-shuffle upsampler.
final class NFKMLXSwinIRNet: Module {
    @ModuleInfo(key: "conv_first") var convFirst: Conv2d
    @ModuleInfo(key: "layers") var layers: [NFKSwinRSTB]
    @ModuleInfo(key: "conv_after_body") var convAfterBody: Conv2d
    @ModuleInfo(key: "conv_before_upsample") var convBeforeUpsample: Conv2d?
    @ModuleInfo(key: "upsample") var upsample: [Conv2d]
    @ModuleInfo(key: "conv_last") var convLast: Conv2d?
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "patch_embed_norm") var patchEmbedNorm: LayerNorm

    let configuration: NFKMLXSwinIRConfiguration

    init(_ c: NFKMLXSwinIRConfiguration) {
        configuration = c
        let dim = c.embedDimensions
        _convFirst.wrappedValue = Conv2d(inputChannels: 3, outputChannels: dim, kernelSize: 3, padding: 1)
        _layers.wrappedValue = c.depths.map { NFKSwinRSTB(dimensions: dim, depth: $0, heads: c.heads, windowSize: c.windowSize, mlpRatio: c.mlpRatio) }
        // `forward_features` normalizes the token sequence after the residual groups, and `PatchEmbed`
        // normalizes it before them (`patch_norm=True`).
        _norm.wrappedValue = LayerNorm(dimensions: dim)
        _patchEmbedNorm.wrappedValue = LayerNorm(dimensions: dim)
        _convAfterBody.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, padding: 1)
        let direct = c.upsampler == .direct
        // A direct upsampler has neither surrounding convolution; building them would leave weights
        // the checkpoint does not carry, which a strict load would then reject.
        _convBeforeUpsample.wrappedValue = direct
            ? nil : Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, padding: 1)
        // The reference `Upsample` builds a power-of-two scale as repeated ×2 stages and a scale of
        // three as ONE ×3 stage, because a factor-3 shuffle cannot be composed from factor-2 ones.
        // Those are the only scales it accepts, and a released checkpoint exists for each.
        _upsample.wrappedValue = direct
            ? [Conv2d(inputChannels: dim, outputChannels: 3 * c.scale * c.scale, kernelSize: 3, padding: 1)]
            : (0 ..< c.upsampleStages).map { _ in
                Conv2d(inputChannels: dim, outputChannels: dim * c.shuffleFactor * c.shuffleFactor,
                       kernelSize: 3, padding: 1)
            }
        _convLast.wrappedValue = direct
            ? nil : Conv2d(inputChannels: dim, outputChannels: 3, kernelSize: 3, padding: 1)
    }

    /// Upscales a bridged image `[H, W, 3]` (`0...1`) to `[scale·H, scale·W, 3]`.
    func upscale(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        // SwinIR centers the plate on a fixed RGB mean and restores it at the end
        // (`x = (x - self.mean) * self.img_range`, then `x / self.img_range + self.mean`).
        let mean = MLXArray([Float(0.4488), 0.4371, 0.4040])
        let centered = image.reshaped([1, height, width, 3]) - mean

        let first = convFirst(centered)
        var tokens = patchEmbedNorm(first.reshaped([1, height * width, configuration.embedDimensions]))
        for layer in layers {
            tokens = layer(tokens, height: height, width: width)
        }
        tokens = norm(tokens)
        let body = convAfterBody(tokens.reshaped([1, height, width, configuration.embedDimensions])) + first

        var reconstructed: MLXArray
        if let convBeforeUpsample, let convLast {
            var feature = relu(convBeforeUpsample(body))
            for stage in upsample {
                feature = NFKMLXPixelShuffle.apply(stage(feature), factor: configuration.shuffleFactor)
            }
            reconstructed = convLast(feature)
        } else {
            // The direct upsampler reconstructs in one step: the shuffle's output already has three
            // channels, so there is nothing to project afterwards.
            reconstructed = NFKMLXPixelShuffle.apply(upsample[0](body), factor: configuration.scale)
        }
        // Restore the centering before clamping, so the result is back in the plate's own range.
        let output = clip(reconstructed + mean, min: 0, max: 1)
        return output.reshaped([output.shape[1], output.shape[2], 3])
    }
}

/// Holds the network for capture in the backend's `@Sendable` forward closure.
private final class NFKSwinIRHolder: @unchecked Sendable {
    let net: NFKMLXSwinIRNet
    init(_ net: NFKMLXSwinIRNet) { self.net = net }
}

/// SwinIR super-resolution as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXSwinIRNet` is the real Swin Transformer SR model. Random weights run (proving the pipeline);
/// a trained checkpoint upscales sharply. The input side must be a multiple of the window size. Load a
/// **safetensors** checkpoint; the loader transposes 4-D convolution weights `[out, in, kH, kW]` to
/// MLX's `[out, kH, kW, in]`.
/// The SwinIR geometry to build, for the Objective-C factory.
///
/// The upsampler differs between them, not only the scale factor: `classicalX4` runs two ×2
/// pixel-shuffle stages where `classicalX3` runs one ×3 stage, so a checkpoint fits only the variant
/// it was trained as.
@objc(NFKMLXSwinIRVariant)
public enum NFKMLXSwinIRVariant: Int {
    case lightweightX2      // the small default geometry
    case classicalX3        // 001_classicalSR_DIV2K_s48w8_SwinIR-M_x3
    case classicalX4        // 001_classicalSR_DIV2K_s48w8_SwinIR-M_x4
    case classicalX8        // 001_classicalSR_DIV2K_s48w8_SwinIR-M_x8
    case lightweightSRX2    // 002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2
}

@objc(NFKMLXSwinIR)
public final class NFKMLXSwinIR: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "swinir-x4"

    static func configuration(for variant: NFKMLXSwinIRVariant) -> NFKMLXSwinIRConfiguration {
        switch variant {
        case .lightweightX2: return .lightweight
        case .classicalX3:   return .classicalSRx3
        case .classicalX4:   return .classicalSRx4
        case .classicalX8:   return .classicalSRx8
        case .lightweightSRX2: return .lightweightSRx2
        }
    }

    /// Builds a super-resolution backend directly from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true). Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .lightweightX2, weightsURL: weightsURL)
    }

    /// Builds a super-resolution backend at a chosen geometry.
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXSwinIRVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = try makeNet(configuration(for: variant))
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKSwinIRHolder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in holder.net.upscale(image) }
    }

    /// Builds the network, rejecting a scale the reference upsampler has no architecture for.
    static func makeNet(_ configuration: NFKMLXSwinIRConfiguration) throws -> NFKMLXSwinIRNet {
        guard configuration.isSupportedScale else {
            throw NFKMLXError.unsupportedConfiguration(
                "SwinIR upsamples by a power of two or by three; \(configuration.scale) is neither")
        }
        return NFKMLXSwinIRNet(configuration)
    }

    /// Downloads the checkpoint from Hugging Face, then builds the backend — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
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

    /// The download factory at a chosen scale, so x2/x3/x8 and the lightweight release are reachable
    /// over the network, not only from a local file. Blocking; run off the render thread.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXSwinIRVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url)
    }

    /// The asynchronous form of the variant download factory.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXSwinIRVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers SwinIR (`swinir-x4`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Loads a safetensors checkpoint into `net`, transposing 4-D convolution weights from PyTorch's
    /// `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKMLXSwinIRNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (remapReferenceKey(key), checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Maps a released SwinIR checkpoint's names onto this module's. The reference wraps each residual
    /// group in an extra `residual_group` module, nests the block MLP, and keeps `conv_before_upsample`
    /// and `upsample` as `nn.Sequential`s — so their layers are positional, and the upsampler's
    /// convolutions sit at the even indices with `PixelShuffle` between them.
    static func remapReferenceKey(_ key: String) -> String {
        var name = key.replacingOccurrences(of: "patch_embed.norm.", with: "patch_embed_norm.")
        name = name.replacingOccurrences(of: ".residual_group.", with: ".")
        name = name.replacingOccurrences(of: ".mlp.fc1.", with: ".mlp_fc1.")
        name = name.replacingOccurrences(of: ".mlp.fc2.", with: ".mlp_fc2.")
        name = name.replacingOccurrences(of: "conv_before_upsample.0.", with: "conv_before_upsample.")
        if name.hasPrefix("upsample.") {
            for stage in 0 ..< 8 {
                name = name.replacingOccurrences(of: "upsample.\(stage * 2).", with: "upsample.\(stage).")
            }
        }
        return name
    }
}
