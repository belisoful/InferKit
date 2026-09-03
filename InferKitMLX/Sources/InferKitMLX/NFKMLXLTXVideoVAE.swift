//
//  NFKMLXLTXVideoVAE.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// The LTX-Video VAE (`AutoencoderKLLTXVideo`): a causal 3D autoencoder that compresses a video to a small
// spatiotemporal latent and back. It is the first piece of the toolkit's video-generation path (the DiT
// and the T5 text encoder follow); on its own it is a self-contained codec, tested by encode→decode
// reconstruction against diffusers.
//
// The encoder is causal in time (a frame sees only past frames): each temporal convolution left-pads by
// repeating the first frame. The decoder is non-causal. Downsampling is a stride-2 causal convolution;
// upsampling is a convolution followed by a 3D pixel shuffle. Resnet blocks normalize with a
// (parameter-free) RMS norm over channels. Tensors flow as `[batch, frames, height, width, channels]`
// (MLX's NDHWC), where the reference is `[B, C, T, H, W]`.

/// LTX-Video VAE geometry. Defaults are the released base model.
public struct NFKMLXLTXVAEConfiguration: Sendable {
    public var inChannels: Int
    public var latentChannels: Int
    public var blockOutChannels: [Int]
    public var layersPerBlock: [Int]
    public var spatioTemporalScaling: [Bool]
    public var patchSize: Int
    public var patchSizeT: Int

    public init(inChannels: Int = 3, latentChannels: Int = 128,
                blockOutChannels: [Int] = [128, 256, 512, 512], layersPerBlock: [Int] = [4, 3, 3, 3, 4],
                spatioTemporalScaling: [Bool] = [true, true, true, false], patchSize: Int = 4, patchSizeT: Int = 1) {
        self.inChannels = inChannels
        self.latentChannels = latentChannels
        self.blockOutChannels = blockOutChannels
        self.layersPerBlock = layersPerBlock
        self.spatioTemporalScaling = spatioTemporalScaling
        self.patchSize = patchSize
        self.patchSizeT = patchSizeT
    }

    public static let base = NFKMLXLTXVAEConfiguration()

    /// A small configuration for weight-free tests.
    public static let tiny = NFKMLXLTXVAEConfiguration(
        latentChannels: 8, blockOutChannels: [8, 16, 16, 16], layersPerBlock: [1, 1, 1, 1, 1])
}

/// Channel-wise RMS normalization over the last axis (NDHWC), parameter-free, the reference's affine-free
/// `RMSNorm(eps=1e-8)`.
private func ltxRMSNorm(_ x: MLXArray) -> MLXArray {
    x * rsqrt(x.square().mean(axis: -1, keepDims: true) + 1e-8)
}

/// A causal (or symmetric) 3D convolution: the temporal axis is padded before a spatially-padded Conv3d.
/// Causal padding repeats the first frame `kernel - 1` times on the left, so a frame depends only on the
/// past; non-causal padding repeats both ends by half.
final class NFKLTXCausalConv3d: Module {
    @ModuleInfo(key: "conv") var conv: Conv3d
    let timeKernel: Int
    let isCausal: Bool

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int = 3, stride: Int = 1, isCausal: Bool = true) {
        timeKernel = kernel
        self.isCausal = isCausal
        let spatialPad = kernel / 2
        _conv.wrappedValue = Conv3d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrTriple(kernel), stride: IntOrTriple(stride),
                                    padding: IntOrTriple((0, spatialPad, spatialPad)))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard timeKernel > 1 else { return conv(x) }
        let frames = x.shape[1]
        let first = x[0..., 0 ..< 1, 0..., 0..., 0...]                    // [B, 1, H, W, C]
        var padded: MLXArray
        if isCausal {
            let left = repeatedFrame(first, timeKernel - 1)
            padded = concatenated([left, x], axis: 1)
        } else {
            let half = (timeKernel - 1) / 2
            let last = x[0..., (frames - 1) ..< frames, 0..., 0..., 0...]
            padded = concatenated([repeatedFrame(first, half), x, repeatedFrame(last, half)], axis: 1)
        }
        return conv(padded)
    }

    private func repeatedFrame(_ frame: MLXArray, _ count: Int) -> MLXArray {
        let s = frame.shape
        return broadcast(frame, to: [s[0], count, s[2], s[3], s[4]])
    }
}

/// One LTX resnet block: RMS-normed, SiLU-activated causal convolutions, added back to the input (a
/// LayerNorm + 1×1×1 convolution on the shortcut when the width changes).
final class NFKLTXResnet: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKLTXCausalConv3d
    @ModuleInfo(key: "conv2") var conv2: NFKLTXCausalConv3d
    @ModuleInfo(key: "norm3") var norm3: LayerNorm?
    @ModuleInfo(key: "conv_shortcut") var convShortcut: NFKLTXCausalConv3d?

    init(_ inChannels: Int, _ outChannels: Int, isCausal: Bool) {
        _conv1.wrappedValue = NFKLTXCausalConv3d(inChannels, outChannels, kernel: 3, isCausal: isCausal)
        _conv2.wrappedValue = NFKLTXCausalConv3d(outChannels, outChannels, kernel: 3, isCausal: isCausal)
        if inChannels != outChannels {
            _norm3.wrappedValue = LayerNorm(dimensions: inChannels, eps: 1e-6)
            _convShortcut.wrappedValue = NFKLTXCausalConv3d(inChannels, outChannels, kernel: 1, isCausal: isCausal)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(ltxRMSNorm(x)))
        h = conv2(silu(ltxRMSNorm(h)))
        var residual = x
        if let norm3 { residual = norm3(residual) }
        if let convShortcut { residual = convShortcut(residual) }
        return h + residual
    }
}

/// A 3D pixel-shuffle upsampler: a convolution widens the channels by `2·2·2`, then the extra channels
/// interleave into the frame, height, and width axes (×2 each), and the first frame is dropped.
final class NFKLTXUpsampler: Module {
    @ModuleInfo(key: "conv") var conv: NFKLTXCausalConv3d

    init(_ channels: Int, isCausal: Bool) {
        _conv.wrappedValue = NFKLTXCausalConv3d(channels, channels * 8, kernel: 3, isCausal: isCausal)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = conv(x)                                                 // [B, T, H, W, C·8]
        let (b, t, h, w) = (out.shape[0], out.shape[1], out.shape[2], out.shape[3])
        let c = out.shape[4] / 8
        // Channels split (C, dt, dh, dw); interleave each into its spatial axis.
        let shuffled = out.reshaped([b, t, h, w, c, 2, 2, 2])
            .transposed(0, 1, 5, 2, 6, 3, 7, 4)                          // [B, T, dt, H, dh, W, dw, C]
            .reshaped([b, t * 2, h * 2, w * 2, c])
        return shuffled[0..., 1..., 0..., 0..., 0...]                    // drop the first frame
    }
}

/// An encoder down block: resnets, an optional stride-2 causal downsample, and a widening resnet.
final class NFKLTXDownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [NFKLTXResnet]
    @ModuleInfo(key: "downsamplers") var downsamplers: [NFKLTXCausalConv3d]?
    @ModuleInfo(key: "conv_out") var convOut: NFKLTXResnet?

    init(_ inChannels: Int, _ outChannels: Int, layers: Int, scale: Bool, isCausal: Bool) {
        _resnets.wrappedValue = (0 ..< layers).map { _ in NFKLTXResnet(inChannels, inChannels, isCausal: isCausal) }
        _downsamplers.wrappedValue = scale
            ? [NFKLTXCausalConv3d(inChannels, inChannels, kernel: 3, stride: 2, isCausal: isCausal)] : nil
        _convOut.wrappedValue = inChannels != outChannels ? NFKLTXResnet(inChannels, outChannels, isCausal: isCausal) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h) }
        if let downsamplers { h = downsamplers[0](h) }
        if let convOut { h = convOut(h) }
        return h
    }
}

/// A decoder up block: an optional widening resnet, an optional upsampler, then resnets.
final class NFKLTXUpBlock: Module {
    @ModuleInfo(key: "conv_in") var convIn: NFKLTXResnet?
    @ModuleInfo(key: "upsamplers") var upsamplers: [NFKLTXUpsampler]?
    @ModuleInfo(key: "resnets") var resnets: [NFKLTXResnet]

    init(_ inChannels: Int, _ outChannels: Int, layers: Int, scale: Bool, isCausal: Bool) {
        _convIn.wrappedValue = inChannels != outChannels ? NFKLTXResnet(inChannels, outChannels, isCausal: isCausal) : nil
        _upsamplers.wrappedValue = scale ? [NFKLTXUpsampler(outChannels, isCausal: isCausal)] : nil
        _resnets.wrappedValue = (0 ..< layers).map { _ in NFKLTXResnet(outChannels, outChannels, isCausal: isCausal) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if let convIn { h = convIn(h) }
        if let upsamplers { h = upsamplers[0](h) }
        for resnet in resnets { h = resnet(h) }
        return h
    }
}

/// A middle block: a stack of resnets at a fixed width.
final class NFKLTXMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [NFKLTXResnet]

    init(_ channels: Int, layers: Int, isCausal: Bool) {
        _resnets.wrappedValue = (0 ..< layers).map { _ in NFKLTXResnet(channels, channels, isCausal: isCausal) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h) }
        return h
    }
}

/// The causal encoder: patchify, a wide convolution, the down blocks, the middle block, and a projection
/// to the latent (plus one channel the posterior uses for its log-variance).
final class NFKLTXEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: NFKLTXCausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [NFKLTXDownBlock]
    @ModuleInfo(key: "mid_block") var midBlock: NFKLTXMidBlock
    @ModuleInfo(key: "conv_out") var convOut: NFKLTXCausalConv3d

    let configuration: NFKMLXLTXVAEConfiguration

    init(_ c: NFKMLXLTXVAEConfiguration) {
        configuration = c
        let patchedChannels = c.inChannels * c.patchSize * c.patchSize
        _convIn.wrappedValue = NFKLTXCausalConv3d(patchedChannels, c.blockOutChannels[0], kernel: 3, isCausal: true)
        var channel = c.blockOutChannels[0]
        let count = c.blockOutChannels.count
        _downBlocks.wrappedValue = (0 ..< count).map { i -> NFKLTXDownBlock in
            let input = channel
            channel = (i + 1 < count) ? c.blockOutChannels[i + 1] : c.blockOutChannels[i]
            return NFKLTXDownBlock(input, channel, layers: c.layersPerBlock[i], scale: c.spatioTemporalScaling[i], isCausal: true)
        }
        _midBlock.wrappedValue = NFKLTXMidBlock(channel, layers: c.layersPerBlock[count], isCausal: true)
        _convOut.wrappedValue = NFKLTXCausalConv3d(channel, c.latentChannels + 1, kernel: 3, isCausal: true)
    }

    /// A video `[B, T, H, W, 3]` → the latent mean `[B, t, h, w, latentChannels]`.
    func callAsFunction(_ video: MLXArray) -> MLXArray {
        var h = patchify(video)
        h = convIn(h)
        for block in downBlocks { h = block(h) }
        h = midBlock(h)
        h = convOut(silu(ltxRMSNorm(h)))
        return h[0..., 0..., 0..., 0..., 0 ..< configuration.latentChannels]   // the latent mean
    }

    /// Folds each `patchSizeT × patchSize × patchSize` block into the channel axis, in the reference's
    /// order (`channel, temporal, width, height`).
    func patchify(_ x: MLXArray) -> MLXArray {
        let (p, pt) = (configuration.patchSize, configuration.patchSizeT)
        let (b, t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        return x.reshaped([b, t / pt, pt, h / p, p, w / p, p, c])              // [B, T', pt, H', ph, W', pw, C]
            .transposed(0, 1, 3, 5, 7, 2, 6, 4)                                // [B, T', H', W', C, pt, pw, ph]
            .reshaped([b, t / pt, h / p, w / p, c * pt * p * p])
    }
}

/// The non-causal decoder: a wide convolution, the middle block, the up blocks, and a projection back to
/// pixels, unpatchified.
final class NFKLTXDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: NFKLTXCausalConv3d
    @ModuleInfo(key: "mid_block") var midBlock: NFKLTXMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [NFKLTXUpBlock]
    @ModuleInfo(key: "conv_out") var convOut: NFKLTXCausalConv3d

    let configuration: NFKMLXLTXVAEConfiguration

    init(_ c: NFKMLXLTXVAEConfiguration) {
        configuration = c
        let channels = Array(c.blockOutChannels.reversed())
        let scaling = Array(c.spatioTemporalScaling.reversed())
        let layers = Array(c.layersPerBlock.reversed())
        var channel = channels[0]
        _convIn.wrappedValue = NFKLTXCausalConv3d(c.latentChannels, channel, kernel: 3, isCausal: false)
        _midBlock.wrappedValue = NFKLTXMidBlock(channel, layers: layers[0], isCausal: false)
        _upBlocks.wrappedValue = (0 ..< channels.count).map { i -> NFKLTXUpBlock in
            let input = channel
            channel = channels[i]
            return NFKLTXUpBlock(input, channel, layers: layers[i + 1], scale: scaling[i], isCausal: false)
        }
        _convOut.wrappedValue = NFKLTXCausalConv3d(channel, c.inChannels * c.patchSize * c.patchSize, kernel: 3, isCausal: false)
    }

    /// A latent `[B, t, h, w, latentChannels]` → a video `[B, T, H, W, 3]`.
    func callAsFunction(_ latent: MLXArray) -> MLXArray {
        var h = convIn(latent)
        h = midBlock(h)
        for block in upBlocks { h = block(h) }
        h = convOut(silu(ltxRMSNorm(h)))
        return unpatchify(h)
    }

    /// The inverse of the encoder's patchify: unfolds the channel axis back into the frame, height, and
    /// width dimensions.
    func unpatchify(_ x: MLXArray) -> MLXArray {
        let (p, pt) = (configuration.patchSize, configuration.patchSizeT)
        let (b, t, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let out = configuration.inChannels
        return x.reshaped([b, t, h, w, out, pt, p, p])                         // [B, T, H, W, out, pt, pw, ph]
            .transposed(0, 1, 5, 2, 7, 3, 6, 4)                                // [B, T, pt, H, ph, W, pw, out]
            .reshaped([b, t * pt, h * p, w * p, out])
    }
}

/// The LTX-Video VAE.
final class NFKMLXLTXVideoVAENet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKLTXEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKLTXDecoder
    @ParameterInfo(key: "latents_mean") var latentsMean: MLXArray
    @ParameterInfo(key: "latents_std") var latentsStd: MLXArray

    let configuration: NFKMLXLTXVAEConfiguration

    init(_ c: NFKMLXLTXVAEConfiguration) {
        configuration = c
        _encoder.wrappedValue = NFKLTXEncoder(c)
        _decoder.wrappedValue = NFKLTXDecoder(c)
        _latentsMean.wrappedValue = MLXArray.zeros([c.latentChannels])
        _latentsStd.wrappedValue = MLXArray.ones([c.latentChannels])
    }

    /// A video `[B, T, H, W, 3]` → the deterministic latent (the posterior mean).
    func encode(_ video: MLXArray) -> MLXArray { encoder(video) }

    /// A latent → the reconstructed video `[B, T, H, W, 3]`.
    func decode(_ latent: MLXArray) -> MLXArray { decoder(latent) }
}

/// Holds the network for capture across an isolation boundary.
private final class NFKLTXVAEHolder: @unchecked Sendable {
    let net: NFKMLXLTXVideoVAENet
    init(_ net: NFKMLXLTXVideoVAENet) { self.net = net }
}

/// The LTX-Video VAE: encode a video to a spatiotemporal latent and decode it back. The first stage of
/// the toolkit's video-generation path; the DiT and text encoder are later milestones.
@objc(NFKMLXLTXVideoVAE)
public final class NFKMLXLTXVideoVAE: NSObject {

    private let holder: NFKLTXVAEHolder

    init(net: NFKMLXLTXVideoVAENet) { holder = NFKLTXVAEHolder(net); super.init() }

    /// A video `[B, T, H, W, 3]` (or `[T, H, W, 3]`) in `-1…1` → the deterministic latent.
    public func encode(_ video: MLXArray) -> MLXArray {
        let batched = video.ndim == 4 ? video.expandedDimensions(axis: 0) : video
        let latent = holder.net.encode(batched)
        eval(latent)
        return latent
    }

    /// A latent → the reconstructed video `[B, T, H, W, 3]`.
    public func decode(_ latent: MLXArray) -> MLXArray {
        let batched = latent.ndim == 4 ? latent.expandedDimensions(axis: 0) : latent
        let video = holder.net.decode(batched)
        eval(video)
        return video
    }

    /// Builds the VAE from optional local weights (the released diffusers safetensors loads directly).
    public static func vae(configuration: NFKMLXLTXVAEConfiguration = .base, weightsURL: URL?) throws -> NFKMLXLTXVideoVAE {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXLTXVideoVAE(net: net)
    }

    static func makeNet(_ configuration: NFKMLXLTXVAEConfiguration = .base) -> NFKMLXLTXVideoVAENet {
        NFKMLXLTXVideoVAENet(configuration)
    }

    /// Loads a checkpoint, transposing 5-D Conv3d weights `[out, in, kT, kH, kW]` → MLX's
    /// `[out, kT, kH, kW, in]`. The causal-conv wrapper keeps the reference's `.conv` key, so the names
    /// match with no remap.
    static func loadWeights(into net: NFKMLXLTXVideoVAENet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let transpose = checkpoint.needsConvTranspose
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            (transpose && value.ndim == 5) ? (key, value.transposed(0, 2, 3, 4, 1)) : (key, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
