//
//  NFKMLXDCAutoencoder.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// The Deep-Compression Autoencoder (`AutoencoderDC`, NVIDIA), the autoencoder SANA encodes into. It
// compresses an image 32× spatially (against the 8× of a Stable Diffusion VAE), which is what keeps
// SANA's latent-token count low enough to run at high resolution. It is a DETERMINISTIC autoencoder —
// `encode` returns one latent, not a Gaussian — built from two block families: `ResBlock`s at the
// shallow stages and `EfficientViTBlock`s (a multiscale ReLU linear attention plus a GLUMBConv gated
// depthwise convolution) at the deep stages. Down/up sampling is pixel-unshuffle / pixel-shuffle with a
// channel-averaging or channel-repeating shortcut.

/// DC-AE geometry. Defaults are the released SANA autoencoder (`dc-ae-f32c32`).
public struct NFKMLXDCAEConfiguration: Sendable {
    public var inChannels: Int
    public var latentChannels: Int
    public var blockTypes: [Bool]                                          // true = EfficientViTBlock
    public var blockOutChannels: [Int]
    public var encoderLayers: [Int]
    public var decoderLayers: [Int]
    public var attentionHeadDim: Int
    public var multiscaleKernels: [Int]                                    // the qkv_multiscales kernel sizes
    public var scaleFactor: Float

    public init(inChannels: Int = 3, latentChannels: Int = 32,
                blockTypes: [Bool] = [false, false, false, true, true, true],
                blockOutChannels: [Int] = [128, 256, 512, 512, 1024, 1024],
                encoderLayers: [Int] = [2, 2, 2, 3, 3, 3],
                decoderLayers: [Int] = [3, 3, 3, 3, 3, 3], attentionHeadDim: Int = 32,
                multiscaleKernels: [Int] = [5], scaleFactor: Float = 0.41407) {
        self.inChannels = inChannels
        self.latentChannels = latentChannels
        self.blockTypes = blockTypes
        self.blockOutChannels = blockOutChannels
        self.encoderLayers = encoderLayers
        self.decoderLayers = decoderLayers
        self.attentionHeadDim = attentionHeadDim
        self.multiscaleKernels = multiscaleKernels
        self.scaleFactor = scaleFactor
    }

    public static let sana = NFKMLXDCAEConfiguration()

    public static let tiny = NFKMLXDCAEConfiguration(
        latentChannels: 8, blockTypes: [false, true], blockOutChannels: [16, 32],
        encoderLayers: [1, 1], decoderLayers: [1, 1], attentionHeadDim: 4, multiscaleKernels: [3])
}

/// RMS normalization over the last (channel) axis, with a weight and a bias (diffusers' `RMSNorm`).
final class NFKDCRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let eps: Float

    init(_ dim: Int, eps: Float = 1e-5) {
        _weight.wrappedValue = MLXArray.ones([dim])
        _bias.wrappedValue = MLXArray.zeros([dim])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps)
        return normed * weight + bias
    }
}

/// Repeat-interleave the last (channel) axis: each channel repeated `count` times consecutively.
private func channelRepeatInterleave(_ x: MLXArray, _ count: Int) -> MLXArray {
    let s = x.shape
    let expanded = broadcast(x.expandedDimensions(axis: -1), to: s + [count])
    return expanded.reshaped(Array(s.dropLast()) + [s[s.count - 1] * count])
}

/// A residual convolution block: `conv1 → SiLU → conv2 → RMSNorm`, added to the input.
final class NFKDCResBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "norm") var norm: NFKDCRMSNorm

    init(_ channels: Int) {
        _conv1.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1, bias: false)
        _norm.wrappedValue = NFKDCRMSNorm(channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(conv2(silu(conv1(x)))) + x
    }
}

/// One multiscale depthwise + grouped-pointwise projection of the fused q/k/v.
final class NFKDCMultiscaleProjection: Module {
    @ModuleInfo(key: "proj_in") var projIn: Conv2d
    @ModuleInfo(key: "proj_out") var projOut: Conv2d

    init(inner: Int, heads: Int, kernel: Int) {
        let channels = 3 * inner
        _projIn.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels,
                                      kernelSize: IntOrPair(kernel), padding: IntOrPair(kernel / 2),
                                      groups: channels, bias: false)
        _projOut.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels,
                                       kernelSize: 1, groups: 3 * heads, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { projOut(projIn(x)) }
}

/// SANA's multiscale ReLU linear attention.
final class NFKDCMultiscaleAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_qkv_multiscale") var multiscale: [NFKDCMultiscaleProjection]
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_out") var normOut: NFKDCRMSNorm

    let heads: Int
    let headDim: Int
    let eps: Float = 1e-15

    init(_ channels: Int, headDim: Int, kernels: [Int]) {
        let headCount = channels / headDim
        self.headDim = headDim
        self.heads = headCount
        let inner = headCount * headDim
        _toQ.wrappedValue = Linear(channels, inner, bias: false)
        _toK.wrappedValue = Linear(channels, inner, bias: false)
        _toV.wrappedValue = Linear(channels, inner, bias: false)
        _multiscale.wrappedValue = kernels.map { NFKDCMultiscaleProjection(inner: inner, heads: headCount, kernel: $0) }
        _toOut.wrappedValue = Linear(inner * (1 + kernels.count), channels, bias: false)
        _normOut.wrappedValue = NFKDCRMSNorm(channels)
    }

    /// `x` `[1, H, W, C]` → `[1, H, W, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x.dim(1), w = x.dim(2), hw = h * w
        let qkv = concatenated([toQ(x), toK(x), toV(x)], axis: 3)          // [1, H, W, 3·inner]
        var scales = [qkv]
        for proj in multiscale { scales.append(proj(qkv)) }
        let cat = concatenated(scales, axis: 3)                            // [1, H, W, 3·inner·(1+K)]
        let groups = heads * multiscale.count + heads                      // heads·(1+K)

        // [groups, 3·headDim, HW]
        let t = cat.reshaped([hw, groups, 3 * headDim]).transposed(1, 2, 0)
        let q = relu(t[0..., 0 ..< headDim, 0...])
        let k = relu(t[0..., headDim ..< 2 * headDim, 0...])
        let v = t[0..., 2 * headDim ..< 3 * headDim, 0...]
        let vPadded = concatenated([v, MLXArray.ones([groups, 1, hw])], axis: 1) // [groups, headDim+1, HW]
        let scores = matmul(vPadded, k.transposed(0, 2, 1))                // [groups, headDim+1, headDim]
        let attended = matmul(scores, q)                                   // [groups, headDim+1, HW]
        let numerator = attended[0..., 0 ..< headDim, 0...]
        let denominator = attended[0..., headDim ..< (headDim + 1), 0...] + eps
        let attn = numerator / denominator                                 // [groups, headDim, HW]
        let merged = attn.reshaped([groups * headDim, hw]).transposed(1, 0).reshaped([1, h, w, groups * headDim])
        return normOut(toOut(merged)) + x
    }
}

/// The GLUMBConv gated depthwise feed-forward, with an RMS norm and a residual (the DC-AE variant).
final class NFKDCGLUMBConv: Module {
    @ModuleInfo(key: "conv_inverted") var convInverted: Conv2d
    @ModuleInfo(key: "conv_depth") var convDepth: Conv2d
    @ModuleInfo(key: "conv_point") var convPoint: Conv2d
    @ModuleInfo(key: "norm") var norm: NFKDCRMSNorm

    init(_ channels: Int, expandRatio: Float = 4) {
        let hidden = Int(expandRatio * Float(channels))
        _convInverted.wrappedValue = Conv2d(inputChannels: channels, outputChannels: hidden * 2, kernelSize: 1)
        _convDepth.wrappedValue = Conv2d(inputChannels: hidden * 2, outputChannels: hidden * 2,
                                         kernelSize: 3, padding: 1, groups: hidden * 2)
        _convPoint.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: channels, kernelSize: 1, bias: false)
        _norm.wrappedValue = NFKDCRMSNorm(channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = silu(convInverted(x))
        h = convDepth(h)
        let half = h.dim(3) / 2
        let value = h[0..., 0..., 0..., 0 ..< half]
        let gate = h[0..., 0..., 0..., half ..< (2 * half)]
        return norm(convPoint(value * silu(gate))) + x
    }
}

/// An efficient ViT block: multiscale linear attention followed by the gated feed-forward.
final class NFKDCEfficientViTBlock: Module {
    @ModuleInfo(key: "attn") var attn: NFKDCMultiscaleAttention
    @ModuleInfo(key: "conv_out") var convOut: NFKDCGLUMBConv

    init(_ channels: Int, headDim: Int, kernels: [Int]) {
        _attn.wrappedValue = NFKDCMultiscaleAttention(channels, headDim: headDim, kernels: kernels)
        _convOut.wrappedValue = NFKDCGLUMBConv(channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { convOut(attn(x)) }
}

/// Stride-2 downsample: a strided convolution added to a pixel-unshuffled, channel-averaged shortcut.
final class NFKDCDownBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    let groupSize: Int

    init(inChannels: Int, outChannels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: 3, stride: 2, padding: 1)
        self.groupSize = inChannels * 4 / outChannels
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = conv(x)
        let unshuffled = NFKRealESRGANNet.pixelUnshuffle(x, scale: 2)       // [1, H/2, W/2, C·4]
        let s = unshuffled.shape
        let shortcut = mean(unshuffled.reshaped([s[0], s[1], s[2], s[3] / groupSize, groupSize]), axis: -1)
        return out + shortcut
    }
}

/// ×2 upsample: nearest interpolate + convolution, added to a channel-repeated, pixel-shuffled shortcut.
final class NFKDCUpBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    let repeats: Int

    init(inChannels: Int, outChannels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self.repeats = outChannels * 4 / inChannels
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let up = NFKRealESRGANNet.upsampleNearest2x(x)
        let out = conv(up)
        let shortcut = NFKMLXPixelShuffle.apply(channelRepeatInterleave(x, repeats), factor: 2)
        return out + shortcut
    }
}

/// A stage: an array of blocks and an optional resampler, mirroring the reference's `nn.Sequential`.
final class NFKDCStage: Module {
    @ModuleInfo(key: "block") var block: [Module]
    init(_ modules: [Module]) { _block.wrappedValue = modules }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for m in block { h = (m as! NFKDCForward).callAsForward(h) }
        return h
    }
}

/// A uniform forward over the DC-AE block families (so a stage can hold a mixed array).
protocol NFKDCForward { func callAsForward(_ x: MLXArray) -> MLXArray }
extension NFKDCResBlock: NFKDCForward { func callAsForward(_ x: MLXArray) -> MLXArray { self(x) } }
extension NFKDCEfficientViTBlock: NFKDCForward { func callAsForward(_ x: MLXArray) -> MLXArray { self(x) } }
extension NFKDCDownBlock: NFKDCForward { func callAsForward(_ x: MLXArray) -> MLXArray { self(x) } }
extension NFKDCUpBlock: NFKDCForward { func callAsForward(_ x: MLXArray) -> MLXArray { self(x) } }

/// The DC-AE encoder.
final class NFKDCEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [NFKDCStage]
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    let outGroupSize: Int

    init(_ config: NFKMLXDCAEConfiguration) {
        _convIn.wrappedValue = Conv2d(inputChannels: config.inChannels,
                                      outputChannels: config.blockOutChannels[0], kernelSize: 3, padding: 1)
        var stages: [NFKDCStage] = []
        let n = config.blockOutChannels.count
        for i in 0 ..< n {
            var modules: [Module] = []
            for _ in 0 ..< config.encoderLayers[i] {
                modules.append(NFKDCEncoder.makeBlock(config, index: i))
            }
            if i < n - 1 {
                modules.append(NFKDCDownBlock(inChannels: config.blockOutChannels[i],
                                              outChannels: config.blockOutChannels[i + 1]))
            }
            stages.append(NFKDCStage(modules))
        }
        _downBlocks.wrappedValue = stages
        _convOut.wrappedValue = Conv2d(inputChannels: config.blockOutChannels[n - 1],
                                       outputChannels: config.latentChannels, kernelSize: 3, padding: 1)
        self.outGroupSize = config.blockOutChannels[n - 1] / config.latentChannels
    }

    static func makeBlock(_ config: NFKMLXDCAEConfiguration, index: Int) -> Module {
        let channels = config.blockOutChannels[index]
        return config.blockTypes[index]
            ? NFKDCEfficientViTBlock(channels, headDim: config.attentionHeadDim, kernels: config.multiscaleKernels)
            : NFKDCResBlock(channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for stage in downBlocks { h = stage(h) }
        let s = h.shape
        let shortcut = mean(h.reshaped([s[0], s[1], s[2], s[3] / outGroupSize, outGroupSize]), axis: -1)
        return convOut(h) + shortcut
    }
}

/// The DC-AE decoder.
final class NFKDCDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "up_blocks") var upBlocks: [NFKDCStage]
    @ModuleInfo(key: "norm_out") var normOut: NFKDCRMSNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    let inRepeats: Int

    init(_ config: NFKMLXDCAEConfiguration) {
        let n = config.blockOutChannels.count
        _convIn.wrappedValue = Conv2d(inputChannels: config.latentChannels,
                                      outputChannels: config.blockOutChannels[n - 1], kernelSize: 3, padding: 1)
        self.inRepeats = config.blockOutChannels[n - 1] / config.latentChannels
        var stages: [NFKDCStage] = []
        for i in 0 ..< n {
            var modules: [Module] = []
            if i < n - 1 {
                modules.append(NFKDCUpBlock(inChannels: config.blockOutChannels[i + 1],
                                            outChannels: config.blockOutChannels[i]))
            }
            for _ in 0 ..< config.decoderLayers[i] {
                modules.append(NFKDCEncoder.makeBlock(config, index: i))
            }
            stages.append(NFKDCStage(modules))
        }
        _upBlocks.wrappedValue = stages
        _normOut.wrappedValue = NFKDCRMSNorm(config.blockOutChannels[0])
        _convOut.wrappedValue = Conv2d(inputChannels: config.blockOutChannels[0],
                                       outputChannels: config.inChannels, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = convIn(z) + channelRepeatInterleave(z, inRepeats)
        for stage in upBlocks.reversed() { h = stage(h) }
        h = relu(normOut(h))
        return convOut(h)
    }
}

/// The Deep-Compression Autoencoder.
public final class NFKMLXDCAutoencoderNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKDCEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKDCDecoder

    public let configuration: NFKMLXDCAEConfiguration

    public init(_ configuration: NFKMLXDCAEConfiguration) {
        self.configuration = configuration
        _encoder.wrappedValue = NFKDCEncoder(configuration)
        _decoder.wrappedValue = NFKDCDecoder(configuration)
    }

    /// `image` `[1, H, W, inChannels]` → latent `[1, H/32, W/32, latentChannels]`.
    public func encode(_ image: MLXArray) -> MLXArray { encoder(image) }

    /// `latent` `[1, h, w, latentChannels]` → `[1, 32h, 32w, inChannels]`.
    public func decode(_ latent: MLXArray) -> MLXArray { decoder(latent) }

    /// Loads a diffusers `AutoencoderDC` checkpoint. The reference's `<blocks>.<i>.<j>` `nn.Sequential`
    /// indices map onto the module's `<i>.block.<j>`; the convolution weights transpose to NHWC.
    public static func loadWeights(into net: NFKMLXDCAutoencoderNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let stageIndex = try! NSRegularExpression(pattern: "(down_blocks|up_blocks)\\.([0-9]+)\\.([0-9]+)\\.")
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            let name = stageIndex.stringByReplacingMatches(in: key, range: NSRange(key.startIndex..., in: key),
                                                           withTemplate: "$1.$2.block.$3.")
            if checkpoint.needsConvTranspose, value.ndim == 4 { return (name, value.transposed(0, 2, 3, 1)) }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
