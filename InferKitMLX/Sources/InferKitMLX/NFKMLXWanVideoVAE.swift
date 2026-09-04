//
//  NFKMLXWanVideoVAE.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The Wan 3D causal VAE (`AutoencoderKLWan`, Alibaba Wan), the autoencoder the Wan text-to-video model
// encodes into, and the last stage of the Wan pipeline. It is a CAUSAL 3D autoencoder that compresses a
// video 4× in time and 16× in space (an 8× spatial convolution stack plus a 2× pixel patchify).
//
// Unlike the LTX VAE (a clean full-clip causal forward), the Wan VAE runs a STATEFUL streaming loop: the
// encoder consumes the frames in chunks (1, then 4 at a time) and the decoder produces them one latent
// frame at a time, threading a per-convolution feature cache (`feat_cache`) that supplies each causal
// convolution's temporal context across chunk boundaries. The temporal up/downsampling happens ONLY on
// that cache path — a one-shot forward would skip it — so the streaming loop is reproduced faithfully.
//
// This is the Wan 2.2 residual VAE (`is_residual`, patch size 2): its down/up blocks carry an AvgDown3D /
// DupUp3D reshape shortcut, and the non-residual Wan 2.1 path is deliberately not implemented.

private let wanCacheFrames = 2

/// Wan VAE geometry. Defaults are the released Wan 2.2 TI2V-5B autoencoder.
public struct NFKMLXWanVAEConfiguration: Sendable {
    public var baseDim: Int
    public var decoderBaseDim: Int
    public var zDim: Int
    public var dimMult: [Int]
    public var numResBlocks: Int
    public var temporalDownsample: [Bool]
    public var patchSize: Int
    public var inChannels: Int
    /// Wan 2.2 uses residual down/up blocks (AvgDown3D / DupUp3D shortcuts); Wan 2.1 uses a flat
    /// down-block list and a halving upsampler.
    public var isResidual: Bool

    public init(baseDim: Int = 160, decoderBaseDim: Int = 256, zDim: Int = 48,
                dimMult: [Int] = [1, 2, 4, 4], numResBlocks: Int = 2,
                temporalDownsample: [Bool] = [false, true, true], patchSize: Int = 2, inChannels: Int = 3,
                isResidual: Bool = true) {
        self.baseDim = baseDim
        self.decoderBaseDim = decoderBaseDim
        self.zDim = zDim
        self.dimMult = dimMult
        self.numResBlocks = numResBlocks
        self.temporalDownsample = temporalDownsample
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.isResidual = isResidual
    }

    public static let wan22 = NFKMLXWanVAEConfiguration()

    /// The Wan 2.1 autoencoder: 16 latent channels, no patchify, non-residual blocks.
    public static let wan21 = NFKMLXWanVAEConfiguration(
        baseDim: 96, decoderBaseDim: 96, zDim: 16, dimMult: [1, 2, 4, 4], numResBlocks: 2,
        temporalDownsample: [false, true, true], patchSize: 1, inChannels: 3, isResidual: false)

    public static let tiny = NFKMLXWanVAEConfiguration(
        baseDim: 8, decoderBaseDim: 8, zDim: 4, dimMult: [2, 2], numResBlocks: 1,
        temporalDownsample: [true], patchSize: 2, inChannels: 3)

    public static let tiny21 = NFKMLXWanVAEConfiguration(
        baseDim: 8, decoderBaseDim: 8, zDim: 4, dimMult: [1, 2], numResBlocks: 1,
        temporalDownsample: [true], patchSize: 1, inChannels: 3, isResidual: false)

    var temporalUpsample: [Bool] { temporalDownsample.reversed() }
    var patchedInChannels: Int { inChannels * patchSize * patchSize }
}

// MARK: Streaming cache

/// A per-convolution feature cache. Each causal convolution has a slot holding the last frames of its
/// previous chunk's input; the slots persist across chunks while the index resets per chunk.
final class NFKWanCache {
    enum Entry { case empty, rep, frames(MLXArray) }
    var slots: [Int: Entry] = [:]
    var index = 0

    func resetIndex() { index = 0 }
    func next() -> Int { let i = index; index += 1; return i }
    func slot(_ i: Int) -> Entry { slots[i] ?? .empty }
}

/// Slices the last `n` frames off the time axis (axis 1) of an NDHWC tensor.
private func wanTail(_ x: MLXArray, _ n: Int) -> MLXArray {
    let t = x.dim(1)
    return x[0..., max(0, t - n)..., 0..., 0..., 0...]
}

// MARK: Layers

/// A causal 3D convolution (an `nn.Conv3d` subclass in the reference, so the weight is at `.weight`). The
/// temporal axis is left-padded (causal); the spatial axes are symmetric. The functional `conv3d` keeps
/// the weight under the module's own name, matching the checkpoint with no remap.
final class NFKWanCausalConv3d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    let causalPad: Int                                                     // total left time-pad (2·padT)
    let spatialPad: Int
    let strideT: Int

    init(_ inChannels: Int, _ outChannels: Int, kernel: (Int, Int, Int), stride: Int = 1, padTime: Int = 0, padSpatial: Int = 0) {
        _weight.wrappedValue = MLXArray.zeros([outChannels, kernel.0, kernel.1, kernel.2, inChannels])
        _bias.wrappedValue = MLXArray.zeros([outChannels])
        self.causalPad = 2 * padTime
        self.spatialPad = padSpatial
        self.strideT = stride
    }

    /// `x` NDHWC, `cache` the previous chunk's stored frames (or none). Prepends the cache and left-pads
    /// the remainder, so the temporal receptive field crosses the chunk boundary causally.
    func callAsFunction(_ x: MLXArray, cache: NFKWanCache.Entry) -> MLXArray {
        var h = x
        var left = causalPad
        if case .frames(let c) = cache, left > 0 {
            h = concatenated([c, h], axis: 1)
            left -= c.dim(1)
        }
        h = padded(h, widths: [IntOrPair(0), IntOrPair((left, 0)), IntOrPair((spatialPad, spatialPad)),
                               IntOrPair((spatialPad, spatialPad)), IntOrPair(0)])
        let out = conv3d(h, weight, stride: IntOrTriple((strideT, 1, 1)), padding: 0)
        return out + bias.reshaped([1, 1, 1, 1, -1])
    }

    /// A plain application (no caching), for the temporal resample convolutions the reference calls
    /// directly on an already-concatenated input.
    func plain(_ x: MLXArray) -> MLXArray {
        var h = x
        if causalPad > 0 || spatialPad > 0 {
            h = padded(h, widths: [IntOrPair(0), IntOrPair((causalPad, 0)), IntOrPair((spatialPad, spatialPad)),
                                   IntOrPair((spatialPad, spatialPad)), IntOrPair(0)])
        }
        return conv3d(h, weight, stride: IntOrTriple((strideT, 1, 1)), padding: 0) + bias.reshaped([1, 1, 1, 1, -1])
    }
}

/// Runs a causal convolution through its cache slot: reads the previous chunk's frames, applies the
/// convolution, and stores this chunk's trailing frames.
private func wanCausal(_ conv: NFKWanCausalConv3d, _ x: MLXArray, _ cache: NFKWanCache) -> MLXArray {
    let i = cache.next()
    var cacheX = wanTail(x, wanCacheFrames)
    if cacheX.dim(1) < 2, case .frames(let prev) = cache.slot(i) {
        cacheX = concatenated([wanTail(prev, 1), cacheX], axis: 1)
    }
    let out = conv(x, cache: cache.slot(i))
    cache.slots[i] = .frames(cacheX)
    return out
}

/// Wan's channel RMS normalization: `normalize(x)·√C·gamma`, an RMS over the channel axis.
final class NFKWanRMSNorm: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    let scale: Float

    init(_ dim: Int) {
        _gamma.wrappedValue = MLXArray.ones([dim])
        self.scale = sqrt(Float(dim))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = x * rsqrt(sum(x * x, axis: -1, keepDims: true) + 1e-12)
        return normed * scale * gamma
    }
}

/// A residual block: `norm1 → SiLU → conv1 → norm2 → SiLU → conv2`, plus a 1×1 shortcut when the width
/// changes.
final class NFKWanResidualBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: NFKWanRMSNorm
    @ModuleInfo(key: "conv1") var conv1: NFKWanCausalConv3d
    @ModuleInfo(key: "norm2") var norm2: NFKWanRMSNorm
    @ModuleInfo(key: "conv2") var conv2: NFKWanCausalConv3d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: NFKWanCausalConv3d?

    init(_ inDim: Int, _ outDim: Int) {
        _norm1.wrappedValue = NFKWanRMSNorm(inDim)
        _conv1.wrappedValue = NFKWanCausalConv3d(inDim, outDim, kernel: (3, 3, 3), padTime: 1, padSpatial: 1)
        _norm2.wrappedValue = NFKWanRMSNorm(outDim)
        _conv2.wrappedValue = NFKWanCausalConv3d(outDim, outDim, kernel: (3, 3, 3), padTime: 1, padSpatial: 1)
        _convShortcut.wrappedValue = inDim != outDim
            ? NFKWanCausalConv3d(inDim, outDim, kernel: (1, 1, 1)) : nil
    }

    func callAsFunction(_ x: MLXArray, _ cache: NFKWanCache) -> MLXArray {
        let shortcut = convShortcut.map { wanCausal($0, x, cache) } ?? x
        var h = wanCausal(conv1, silu(norm1(x)), cache)
        h = wanCausal(conv2, silu(norm2(h)), cache)
        return h + shortcut
    }
}

/// The single-head causal self-attention over each frame's spatial map, used in the middle block.
final class NFKWanAttentionBlock: Module {
    @ModuleInfo(key: "norm") var norm: NFKWanRMSNorm
    @ModuleInfo(key: "to_qkv") var toQKV: Conv2d
    @ModuleInfo(key: "proj") var proj: Conv2d
    let dim: Int

    init(_ dim: Int) {
        self.dim = dim
        _norm.wrappedValue = NFKWanRMSNorm(dim)
        _toQKV.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1)
        _proj.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), h = x.dim(2), w = x.dim(3)
        let perFrame = x.reshaped([b * t, h, w, dim])
        let normed = norm(perFrame)
        let qkv = toQKV(normed).reshaped([b * t, h * w, 3 * dim])
        let q = qkv[0..., 0..., 0 ..< dim].expandedDimensions(axis: 1)
        let k = qkv[0..., 0..., dim ..< 2 * dim].expandedDimensions(axis: 1)
        let v = qkv[0..., 0..., 2 * dim ..< 3 * dim].expandedDimensions(axis: 1)
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                         scale: 1.0 / sqrt(Float(dim)), mask: .none)
        let out = proj(attended[0..., 0].reshaped([b * t, h, w, dim]))
        return out.reshaped([b, t, h, w, dim]) + x
    }
}

/// The middle block: a residual block, one attention block, and a second residual block.
final class NFKWanMidBlock: Module {
    @ModuleInfo(key: "attentions") var attentions: [NFKWanAttentionBlock]
    @ModuleInfo(key: "resnets") var resnets: [NFKWanResidualBlock]

    init(_ dim: Int) {
        _attentions.wrappedValue = [NFKWanAttentionBlock(dim)]
        _resnets.wrappedValue = [NFKWanResidualBlock(dim, dim), NFKWanResidualBlock(dim, dim)]
    }

    func callAsFunction(_ x: MLXArray, _ cache: NFKWanCache) -> MLXArray {
        var h = resnets[0](x, cache)
        h = attentions[0](h)
        h = resnets[1](h, cache)
        return h
    }
}

/// The cacheless average-pooling downsample shortcut (folds `factor_t × factor_s²` into the channel and
/// averages the groups).
private func wanAvgDown(_ x: MLXArray, factorT: Int, factorS: Int, outChannels: Int) -> MLXArray {
    let b = x.dim(0)
    var h = x
    let padT = (factorT - h.dim(1) % factorT) % factorT
    if padT > 0 {
        h = padded(h, widths: [IntOrPair(0), IntOrPair((padT, 0)), IntOrPair(0), IntOrPair(0), IntOrPair(0)])
    }
    let t = h.dim(1), height = h.dim(2), w = h.dim(3), c = h.dim(4)
    let tt = t / factorT, hh = height / factorS, ww = w / factorS
    // [B, tt, ft, hh, fs, ww, fs, C] -> [B, tt, hh, ww, C, ft, fs, fs]
    let grouped = h.reshaped([b, tt, factorT, hh, factorS, ww, factorS, c])
        .transposed(0, 1, 3, 5, 7, 2, 4, 6)
        .reshaped([b, tt, hh, ww, c * factorT * factorS * factorS])
    let groupSize = c * factorT * factorS * factorS / outChannels
    return mean(grouped.reshaped([b, tt, hh, ww, outChannels, groupSize]), axis: -1)
}

/// The cacheless duplicate-and-shuffle upsample shortcut (repeats the channel and interleaves it into the
/// frame and spatial axes).
private func wanDupUp(_ x: MLXArray, factorT: Int, factorS: Int, outChannels: Int, firstChunk: Bool) -> MLXArray {
    let b = x.dim(0), t = x.dim(1), h = x.dim(2), w = x.dim(3), c = x.dim(4)
    let repeats = outChannels * factorT * factorS * factorS / c
    let expanded = broadcast(x.expandedDimensions(axis: -1), to: [b, t, h, w, c, repeats])
        .reshaped([b, t, h, w, c * repeats])
    // [B, T, H, W, out, ft, fs, fs] -> [B, T, ft, H, fs, W, fs, out]
    var out = expanded.reshaped([b, t, h, w, outChannels, factorT, factorS, factorS])
        .transposed(0, 1, 5, 2, 6, 3, 7, 4)
        .reshaped([b, t * factorT, h * factorS, w * factorS, outChannels])
    if firstChunk {
        out = out[0..., (factorT - 1)..., 0..., 0..., 0...]
    }
    return out
}

/// The temporal/spatial resample with its cache logic. Spatial down is an asymmetric-pad stride-2
/// convolution; spatial up is a nearest interpolate + convolution; the temporal variants add a cached
/// `time_conv`.
final class NFKWanResample: Module {
    @ModuleInfo(key: "resample") var resample: [Module]                   // [pad/marker, Conv2d]
    @ModuleInfo(key: "time_conv") var timeConv: NFKWanCausalConv3d?
    let mode: String
    let dim: Int

    init(dim: Int, mode: String, upsampleOutDim: Int? = nil) {
        self.mode = mode
        self.dim = dim
        let outDim = upsampleOutDim ?? dim / 2
        switch mode {
        case "upsample2d", "upsample3d":
            _resample.wrappedValue = [Module(), Conv2d(inputChannels: dim, outputChannels: outDim, kernelSize: 3, padding: 1)]
            _timeConv.wrappedValue = mode == "upsample3d"
                ? NFKWanCausalConv3d(dim, dim * 2, kernel: (3, 1, 1), padTime: 1) : nil
        case "downsample2d", "downsample3d":
            _resample.wrappedValue = [Module(), Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2)]
            _timeConv.wrappedValue = mode == "downsample3d"
                ? NFKWanCausalConv3d(dim, dim, kernel: (3, 1, 1), stride: 2) : nil
        default:
            _resample.wrappedValue = []
            _timeConv.wrappedValue = nil
        }
    }

    /// Applies the per-frame spatial resample convolution.
    private func spatial(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), c = x.dim(4)
        var frames = x.reshaped([b * t, x.dim(2), x.dim(3), c])
        if mode.hasPrefix("downsample") {
            frames = padded(frames, widths: [IntOrPair(0), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair(0)])
        } else {
            frames = NFKRealESRGANNet.upsampleNearest2x(frames)
        }
        let out = (resample[1] as! Conv2d)(frames)
        return out.reshaped([b, t, out.dim(1), out.dim(2), out.dim(3)])
    }

    func callAsFunction(_ x: MLXArray, _ cache: NFKWanCache) -> MLXArray {
        var h = x
        if mode == "upsample3d", let timeConv {
            let i = cache.next()
            if case .empty = cache.slot(i) {
                cache.slots[i] = .rep
            } else {
                let b = h.dim(0), c = h.dim(4), t = h.dim(1), height = h.dim(2), w = h.dim(3)
                // The trailing frames stored for the NEXT chunk: borrow the previous chunk's last frame
                // (or zeros in the Rep case) when this chunk has fewer than two.
                var cacheX = wanTail(h, wanCacheFrames)
                if cacheX.dim(1) < 2 {
                    switch cache.slot(i) {
                    case .frames(let prev): cacheX = concatenated([wanTail(prev, 1), cacheX], axis: 1)
                    case .rep: cacheX = concatenated([MLXArray.zeros(cacheX.shape), cacheX], axis: 1)
                    case .empty: break
                    }
                }
                let conv: MLXArray
                if case .frames(let prev) = cache.slot(i) {
                    conv = timeConv(h, cache: .frames(prev))
                } else {
                    conv = timeConv.plain(h)                               // Rep: no cache prepend
                }
                cache.slots[i] = .frames(cacheX)
                // [B, T, H, W, 2C] -> split the doubled channel and interleave it as a new frame sub-axis.
                h = conv.reshaped([b, t, height, w, 2, c]).transposed(0, 1, 4, 2, 3, 5)
                    .reshaped([b, t * 2, height, w, c])
            }
        }
        h = spatial(h)
        if mode == "downsample3d", let timeConv {
            let i = cache.next()
            if case .empty = cache.slot(i) {
                cache.slots[i] = .frames(h)
            } else if case .frames(let prev) = cache.slot(i) {
                let cacheX = wanTail(h, 1)
                h = timeConv.plain(concatenated([wanTail(prev, 1), h], axis: 1))
                cache.slots[i] = .frames(cacheX)
            }
        }
        return h
    }
}

/// A residual down block: residual blocks and an optional downsampler, plus the AvgDown3D shortcut.
final class NFKWanResidualDownBlock: Module {
    @ModuleInfo(key: "avg_shortcut") var avgShortcut: NFKWanAvgShortcut
    @ModuleInfo(key: "resnets") var resnets: [NFKWanResidualBlock]
    @ModuleInfo(key: "downsampler") var downsampler: NFKWanResample?

    let factorT: Int
    let factorS: Int
    let outDim: Int

    init(_ inDim: Int, _ outDim: Int, numResBlocks: Int, temporalDownsample: Bool, downFlag: Bool) {
        self.factorT = temporalDownsample ? 2 : 1
        self.factorS = downFlag ? 2 : 1
        self.outDim = outDim
        _avgShortcut.wrappedValue = NFKWanAvgShortcut()
        var blocks: [NFKWanResidualBlock] = []
        var dim = inDim
        for _ in 0 ..< numResBlocks { blocks.append(NFKWanResidualBlock(dim, outDim)); dim = outDim }
        _resnets.wrappedValue = blocks
        if downFlag {
            _downsampler.wrappedValue = NFKWanResample(dim: outDim, mode: temporalDownsample ? "downsample3d" : "downsample2d")
        } else {
            _downsampler.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray, _ cache: NFKWanCache) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h, cache) }
        if let downsampler { h = downsampler(h, cache) }
        return h + wanAvgDown(x, factorT: factorT, factorS: factorS, outChannels: outDim)
    }
}

/// A residual up block: an optional upsampler then residual blocks, plus the DupUp3D shortcut.
final class NFKWanResidualUpBlock: Module {
    @ModuleInfo(key: "avg_shortcut") var avgShortcut: NFKWanAvgShortcut?
    @ModuleInfo(key: "resnets") var resnets: [NFKWanResidualBlock]
    @ModuleInfo(key: "upsampler") var upsampler: NFKWanResample?

    let factorT: Int
    let factorS: Int
    let inDim: Int
    let outDim: Int

    init(_ inDim: Int, _ outDim: Int, numResBlocks: Int, temporalUpsample: Bool, upFlag: Bool) {
        self.factorT = temporalUpsample ? 2 : 1
        self.factorS = upFlag ? 2 : 1
        self.inDim = inDim
        self.outDim = outDim
        _avgShortcut.wrappedValue = upFlag ? NFKWanAvgShortcut() : nil
        var blocks: [NFKWanResidualBlock] = []
        var dim = inDim
        for _ in 0 ..< (numResBlocks + 1) { blocks.append(NFKWanResidualBlock(dim, outDim)); dim = outDim }
        _resnets.wrappedValue = blocks
        if upFlag {
            _upsampler.wrappedValue = NFKWanResample(dim: outDim, mode: temporalUpsample ? "upsample3d" : "upsample2d", upsampleOutDim: outDim)
        } else {
            _upsampler.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray, _ cache: NFKWanCache, firstChunk: Bool) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h, cache) }
        if let upsampler { h = upsampler(h, cache) }
        if avgShortcut != nil {
            h = h + wanDupUp(x, factorT: factorT, factorS: factorS, outChannels: outDim, firstChunk: firstChunk)
        }
        return h
    }
}

/// A marker for the parameter-free AvgDown3D / DupUp3D shortcuts (they carry no weights).
final class NFKWanAvgShortcut: Module {}

/// A non-residual up block (Wan 2.1): residual blocks then an optional halving upsampler.
final class NFKWanUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [NFKWanResidualBlock]
    @ModuleInfo(key: "upsamplers") var upsamplers: [NFKWanResample]?

    init(_ inDim: Int, _ outDim: Int, numResBlocks: Int, temporalUpsample: Bool, upFlag: Bool) {
        var blocks: [NFKWanResidualBlock] = []
        var dim = inDim
        for _ in 0 ..< (numResBlocks + 1) { blocks.append(NFKWanResidualBlock(dim, outDim)); dim = outDim }
        _resnets.wrappedValue = blocks
        _upsamplers.wrappedValue = upFlag
            ? [NFKWanResample(dim: outDim, mode: temporalUpsample ? "upsample3d" : "upsample2d")] : nil
    }

    func callAsFunction(_ x: MLXArray, _ cache: NFKWanCache) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h, cache) }
        if let upsamplers { h = upsamplers[0](h, cache) }
        return h
    }
}

/// Runs one encoder down-block (residual stage, residual block, or resample).
private func wanEncBlock(_ m: Module, _ x: MLXArray, _ cache: NFKWanCache) -> MLXArray {
    if let b = m as? NFKWanResidualDownBlock { return b(x, cache) }
    if let b = m as? NFKWanResidualBlock { return b(x, cache) }
    if let b = m as? NFKWanResample { return b(x, cache) }
    return x
}

/// The Wan encoder (both the 2.2 residual and 2.1 non-residual paths).
final class NFKWanEncoder3d: Module {
    @ModuleInfo(key: "conv_in") var convIn: NFKWanCausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [Module]
    @ModuleInfo(key: "mid_block") var midBlock: NFKWanMidBlock
    @ModuleInfo(key: "norm_out") var normOut: NFKWanRMSNorm
    @ModuleInfo(key: "conv_out") var convOut: NFKWanCausalConv3d

    init(_ config: NFKMLXWanVAEConfiguration, zDim: Int) {
        let dims = [1] + config.dimMult
        let base = config.baseDim
        _convIn.wrappedValue = NFKWanCausalConv3d(config.patchedInChannels, base * dims[0], kernel: (3, 3, 3), padTime: 1, padSpatial: 1)
        var blocks: [Module] = []
        let last = config.dimMult.count - 1
        for i in 0 ..< config.dimMult.count {
            if config.isResidual {
                blocks.append(NFKWanResidualDownBlock(
                    base * dims[i], base * dims[i + 1], numResBlocks: config.numResBlocks,
                    temporalDownsample: i != last ? config.temporalDownsample[i] : false, downFlag: i != last))
            } else {
                var inDim = base * dims[i]
                for _ in 0 ..< config.numResBlocks {
                    blocks.append(NFKWanResidualBlock(inDim, base * dims[i + 1]))
                    inDim = base * dims[i + 1]
                }
                if i != last {
                    blocks.append(NFKWanResample(dim: base * dims[i + 1],
                                                 mode: config.temporalDownsample[i] ? "downsample3d" : "downsample2d"))
                }
            }
        }
        _downBlocks.wrappedValue = blocks
        let outDim = base * dims[config.dimMult.count]
        _midBlock.wrappedValue = NFKWanMidBlock(outDim)
        _normOut.wrappedValue = NFKWanRMSNorm(outDim)
        _convOut.wrappedValue = NFKWanCausalConv3d(outDim, zDim, kernel: (3, 3, 3), padTime: 1, padSpatial: 1)
    }

    func callAsFunction(_ x: MLXArray, _ cache: NFKWanCache) -> MLXArray {
        var h = wanCausal(convIn, x, cache)
        for block in downBlocks { h = wanEncBlock(block, h, cache) }
        h = midBlock(h, cache)
        h = silu(normOut(h))
        return wanCausal(convOut, h, cache)
    }
}

/// The Wan decoder (both the 2.2 residual and 2.1 non-residual paths).
final class NFKWanDecoder3d: Module {
    @ModuleInfo(key: "conv_in") var convIn: NFKWanCausalConv3d
    @ModuleInfo(key: "mid_block") var midBlock: NFKWanMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [Module]
    @ModuleInfo(key: "norm_out") var normOut: NFKWanRMSNorm
    @ModuleInfo(key: "conv_out") var convOut: NFKWanCausalConv3d

    init(_ config: NFKMLXWanVAEConfiguration, zDim: Int) {
        let base = config.decoderBaseDim
        let dims = [config.dimMult.last!] + config.dimMult.reversed()
        _convIn.wrappedValue = NFKWanCausalConv3d(zDim, base * dims[0], kernel: (3, 3, 3), padTime: 1, padSpatial: 1)
        _midBlock.wrappedValue = NFKWanMidBlock(base * dims[0])
        var blocks: [Module] = []
        let up = config.temporalUpsample
        let last = config.dimMult.count - 1
        for i in 0 ..< config.dimMult.count {
            let upFlag = i != last
            if config.isResidual {
                blocks.append(NFKWanResidualUpBlock(
                    base * dims[i], base * dims[i + 1], numResBlocks: config.numResBlocks,
                    temporalUpsample: upFlag ? up[i] : false, upFlag: upFlag))
            } else {
                // The non-residual upsampler halves the channels, so an inner stage's input is halved.
                let inDim = i > 0 ? base * dims[i] / 2 : base * dims[i]
                blocks.append(NFKWanUpBlock(
                    inDim, base * dims[i + 1], numResBlocks: config.numResBlocks,
                    temporalUpsample: upFlag ? up[i] : false, upFlag: upFlag))
            }
        }
        _upBlocks.wrappedValue = blocks
        let outDim = base * dims[config.dimMult.count]
        _normOut.wrappedValue = NFKWanRMSNorm(outDim)
        _convOut.wrappedValue = NFKWanCausalConv3d(outDim, config.patchedInChannels, kernel: (3, 3, 3), padTime: 1, padSpatial: 1)
    }

    func callAsFunction(_ x: MLXArray, _ cache: NFKWanCache, firstChunk: Bool) -> MLXArray {
        var h = wanCausal(convIn, x, cache)
        h = midBlock(h, cache)
        for block in upBlocks {
            if let b = block as? NFKWanResidualUpBlock { h = b(h, cache, firstChunk: firstChunk) }
            else if let b = block as? NFKWanUpBlock { h = b(h, cache) }
        }
        h = silu(normOut(h))
        return wanCausal(convOut, h, cache)
    }
}

/// The Wan 3D causal VAE.
public final class NFKMLXWanVideoVAENet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKWanEncoder3d
    @ModuleInfo(key: "quant_conv") var quantConv: NFKWanCausalConv3d
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: NFKWanCausalConv3d
    @ModuleInfo(key: "decoder") var decoder: NFKWanDecoder3d

    public let configuration: NFKMLXWanVAEConfiguration
    public init(_ configuration: NFKMLXWanVAEConfiguration) {
        self.configuration = configuration
        _encoder.wrappedValue = NFKWanEncoder3d(configuration, zDim: configuration.zDim * 2)
        _quantConv.wrappedValue = NFKWanCausalConv3d(configuration.zDim * 2, configuration.zDim * 2, kernel: (1, 1, 1))
        _postQuantConv.wrappedValue = NFKWanCausalConv3d(configuration.zDim, configuration.zDim, kernel: (1, 1, 1))
        _decoder.wrappedValue = NFKWanDecoder3d(configuration, zDim: configuration.zDim)
    }

    /// Patchify an NDHWC video `[1, T, H, W, C]` by the spatial patch size into the channel axis.
    private func patchify(_ x: MLXArray) -> MLXArray {
        let p = configuration.patchSize
        guard p > 1 else { return x }
        let b = x.dim(0), t = x.dim(1), h = x.dim(2), w = x.dim(3), c = x.dim(4)
        // [B, T, h/p, ph, w/p, pw, C] -> [B, T, h/p, w/p, C·p·p], channel order (C, pw, ph) per the reference.
        return x.reshaped([b, t, h / p, p, w / p, p, c])
            .transposed(0, 1, 2, 4, 6, 5, 3)
            .reshaped([b, t, h / p, w / p, c * p * p])
    }

    private func unpatchify(_ x: MLXArray) -> MLXArray {
        let p = configuration.patchSize
        guard p > 1 else { return x }
        let b = x.dim(0), t = x.dim(1), h = x.dim(2), w = x.dim(3)
        let c = x.dim(4) / (p * p)
        // Channel (C, pw, ph): ph pairs with height, pw with width, matching the reference's unpatchify.
        return x.reshaped([b, t, h, w, c, p, p])
            .transposed(0, 1, 2, 6, 3, 5, 4)
            .reshaped([b, t, h * p, w * p, c])
    }

    /// Encodes a video `[1, T, H, W, inChannels]` to the deterministic latent mean `[1, T', H', W', zDim]`.
    public func encode(_ video: MLXArray) -> MLXArray {
        let x = patchify(video)
        let frames = x.dim(1)
        let cache = NFKWanCache()
        let iterations = 1 + (frames - 1) / 4
        var out: MLXArray?
        for i in 0 ..< iterations {
            cache.resetIndex()
            let chunk = i == 0 ? x[0..., 0 ..< 1, 0..., 0..., 0...]
                               : x[0..., (1 + 4 * (i - 1)) ..< min(1 + 4 * i, frames), 0..., 0..., 0...]
            let encoded = encoder(chunk, cache)
            out = out.map { concatenated([$0, encoded], axis: 1) } ?? encoded
        }
        return quantConv(out!, cache: .empty)[0..., 0..., 0..., 0..., 0 ..< configuration.zDim]
    }

    /// The full pre-split moments (both mean and log-variance), for validation.
    public func encodeMoments(_ video: MLXArray) -> MLXArray {
        let x = patchify(video)
        let frames = x.dim(1)
        let cache = NFKWanCache()
        let iterations = 1 + (frames - 1) / 4
        var out: MLXArray?
        for i in 0 ..< iterations {
            cache.resetIndex()
            let chunk = i == 0 ? x[0..., 0 ..< 1, 0..., 0..., 0...]
                               : x[0..., (1 + 4 * (i - 1)) ..< min(1 + 4 * i, frames), 0..., 0..., 0...]
            let encoded = encoder(chunk, cache)
            out = out.map { concatenated([$0, encoded], axis: 1) } ?? encoded
        }
        return quantConv(out!, cache: .empty)
    }

    /// The decoder's first-frame stage outputs (conv_in, mid, up-block 0), for validation.
    public func decodeStages(_ latentFrame: MLXArray) -> (convIn: MLXArray, mid: MLXArray, up0: MLXArray) {
        let cache = NFKWanCache()
        let x = postQuantConv(latentFrame, cache: .empty)
        let convInOut = wanCausal(decoder.convIn, x, cache)
        let midOut = decoder.midBlock(convInOut, cache)
        let up0 = (decoder.upBlocks[0] as! NFKWanResidualUpBlock)(midOut, cache, firstChunk: true)
        return (convInOut, midOut, up0)
    }

    /// Decodes a latent `[1, T', H', W', zDim]` to a video `[1, T, H, W, inChannels]`.
    public func decode(_ latent: MLXArray) -> MLXArray {
        let frames = latent.dim(1)
        let cache = NFKWanCache()
        let x = postQuantConv(latent, cache: .empty)
        var out: MLXArray?
        for i in 0 ..< frames {
            cache.resetIndex()
            let frame = x[0..., i ..< (i + 1), 0..., 0..., 0...]
            let decoded = decoder(frame, cache, firstChunk: i == 0)
            out = out.map { concatenated([$0, decoded], axis: 1) } ?? decoded
        }
        return clip(unpatchify(out!), min: -1, max: 1)
    }
}
