//
//  NFKMLXLTX2AudioVAE.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

// The LTX-2 audio autoencoder's decoder (`AutoencoderKLLTX2Audio`, Lightricks). A latent spectrogram
// decodes to a stereo log-mel spectrogram through an image-style decoder whose height axis is TIME and
// whose width axis is the mel bins: a convolution in, two resnets, then one stage per resolution of
// three resnets each, all but the last ending in a 2x nearest-neighbour upsample and a convolution.
// Every convolution is causal in time: it pads the whole `kernel − 1` on the past side and splits the
// mel padding evenly, and an upsample drops the first time row after its convolution. Normalization is
// an affine-free per-pixel RMS norm over the channels (eps 1e-6). `T` latent frames decode to
// `4 T − 3` mel frames, which the decoder crops or zero-pads to exactly that length and to the
// configured mel bins.
//
// Tensors flow as `[batch, time, mel, channels]` (MLX's NHWC); the reference is `[B, C, T, M]`.

/// The geometry of an LTX-2 audio decoder, read from a release's `audio_vae/config.json`.
public struct NFKMLXLTX2AudioVAEConfiguration: Sendable {
    public var baseChannels: Int
    public var outputChannels: Int
    public var channelMultipliers: [Int]
    public var resnetBlocks: Int
    public var latentChannels: Int
    public var melBins: Int
    public var sampleRate: Int
    public var melHopLength: Int
    /// The encoder's input channels, the spectrogram's two stereo channels.
    public var inChannels: Int

    public init(baseChannels: Int = 128, outputChannels: Int = 2, channelMultipliers: [Int] = [1, 2, 4],
                resnetBlocks: Int = 2, latentChannels: Int = 8, melBins: Int = 64, sampleRate: Int = 16000,
                melHopLength: Int = 160, inChannels: Int = 2) {
        self.inChannels = inChannels
        self.baseChannels = baseChannels
        self.outputChannels = outputChannels
        self.channelMultipliers = channelMultipliers
        self.resnetBlocks = resnetBlocks
        self.latentChannels = latentChannels
        self.melBins = melBins
        self.sampleRate = sampleRate
        self.melHopLength = melHopLength
    }

    /// The factor the latent compresses time and the mel axis by.
    public static let compression = 4

    /// The latent's mel bins, `melBins / 4`.
    public var latentMelBins: Int { melBins / Self.compression }

    /// The LTX-2.3 release's decoder (`diffusers/LTX-2.3-Diffusers`, `audio_vae/config.json`).
    public static let ltx23 = NFKMLXLTX2AudioVAEConfiguration()

    /// The release's arrangement at a sixteenth of its width, for weight-free tests.
    public static let tiny = NFKMLXLTX2AudioVAEConfiguration(baseChannels: 8, latentChannels: 2, melBins: 16)
}

/// A 2-D convolution causal in time: the past side of the time axis takes the whole `kernel − 1` of
/// zero padding, and the mel axis splits it.
final class NFKLTX2AudioCausalConv2d: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    let kernel: Int

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int) {
        self.kernel = kernel
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrPair(kernel))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pad = kernel - 1
        guard pad > 0 else { return conv(x) }
        return conv(padded(x, widths: [IntOrPair(0), IntOrPair((pad, 0)), IntOrPair((pad / 2, pad - pad / 2)),
                                       IntOrPair(0)]))
    }
}

/// An audio resnet: pixel norm, SiLU and a causal convolution, twice, added to the input, whose width a
/// causal 1×1 convolution changes where the block widens or narrows.
final class NFKLTX2AudioResnet: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKLTX2AudioCausalConv2d
    @ModuleInfo(key: "conv2") var conv2: NFKLTX2AudioCausalConv2d
    @ModuleInfo(key: "nin_shortcut") var shortcut: NFKLTX2AudioCausalConv2d?

    init(_ inChannels: Int, _ outChannels: Int) {
        _conv1.wrappedValue = NFKLTX2AudioCausalConv2d(inChannels, outChannels, kernel: 3)
        _conv2.wrappedValue = NFKLTX2AudioCausalConv2d(outChannels, outChannels, kernel: 3)
        _shortcut.wrappedValue = inChannels != outChannels
            ? NFKLTX2AudioCausalConv2d(inChannels, outChannels, kernel: 1) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(NFKReferenceRounding.silu(ltx2PixelNorm(x, eps: 1e-6)))
        h = conv2(NFKReferenceRounding.silu(ltx2PixelNorm(h, eps: 1e-6)))
        return (shortcut.map { $0(x) } ?? x) + h
    }
}

/// A 2x nearest-neighbour upsample of both axes, a causal convolution, and the first time row dropped.
final class NFKLTX2AudioUpsample: Module {
    @ModuleInfo(key: "conv") var conv: NFKLTX2AudioCausalConv2d

    init(_ channels: Int) {
        _conv.wrappedValue = NFKLTX2AudioCausalConv2d(channels, channels, kernel: 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, m, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let doubled = broadcast(x.reshaped([b, t, 1, m, 1, c]), to: [b, t, 2, m, 2, c]).reshaped([b, 2 * t, 2 * m, c])
        return conv(doubled)[0..., 1...]
    }
}

/// One decoder resolution: its resnets and, above the finest resolution, the upsample.
final class NFKLTX2AudioStage: Module {
    @ModuleInfo(key: "block") var block: [NFKLTX2AudioResnet]
    @ModuleInfo(key: "upsample") var upsample: NFKLTX2AudioUpsample?

    init(blocks: [NFKLTX2AudioResnet], upsample: NFKLTX2AudioUpsample?) {
        _block.wrappedValue = blocks
        _upsample.wrappedValue = upsample
    }
}

/// The two resnets of the decoder's mid block; the release carries no attention between them.
final class NFKLTX2AudioMid: Module {
    @ModuleInfo(key: "block_1") var block1: NFKLTX2AudioResnet
    @ModuleInfo(key: "block_2") var block2: NFKLTX2AudioResnet

    init(_ channels: Int) {
        _block1.wrappedValue = NFKLTX2AudioResnet(channels, channels)
        _block2.wrappedValue = NFKLTX2AudioResnet(channels, channels)
    }
}

/// The LTX-2 audio decoder: a latent `[B, T, latentMelBins, latentChannels]` → a log-mel spectrogram
/// `[B, 4 T − 3, melBins, outputChannels]`.
public final class NFKMLXLTX2AudioDecoderNet: Module {
    @ModuleInfo(key: "conv_in") var convIn: NFKLTX2AudioCausalConv2d
    @ModuleInfo(key: "mid") var mid: NFKLTX2AudioMid
    @ModuleInfo(key: "up") var up: [NFKLTX2AudioStage]
    @ModuleInfo(key: "conv_out") var convOut: NFKLTX2AudioCausalConv2d

    public let configuration: NFKMLXLTX2AudioVAEConfiguration

    public init(_ c: NFKMLXLTX2AudioVAEConfiguration) {
        configuration = c
        let top = c.baseChannels * c.channelMultipliers[c.channelMultipliers.count - 1]
        _convIn.wrappedValue = NFKLTX2AudioCausalConv2d(c.latentChannels, top, kernel: 3)
        _mid.wrappedValue = NFKLTX2AudioMid(top)
        // The reference builds the stages coarsest first and inserts each at the front, so `up.0` is the
        // finest resolution and the decoder runs them from the last index down.
        var stages = [NFKLTX2AudioStage]()
        var width = top
        for level in c.channelMultipliers.indices.reversed() {
            let out = c.baseChannels * c.channelMultipliers[level]
            var blocks = [NFKLTX2AudioResnet]()
            for _ in 0 ... c.resnetBlocks {
                blocks.append(NFKLTX2AudioResnet(width, out))
                width = out
            }
            stages.insert(NFKLTX2AudioStage(blocks: blocks, upsample: level != 0 ? NFKLTX2AudioUpsample(width) : nil),
                          at: 0)
        }
        _up.wrappedValue = stages
        _convOut.wrappedValue = NFKLTX2AudioCausalConv2d(width, c.outputChannels, kernel: 3)
    }

    public func callAsFunction(_ latent: MLXArray) -> MLXArray {
        let frames = latent.dim(1)
        var h = mid.block2(mid.block1(convIn(latent)))
        for stage in up.reversed() {
            for block in stage.block { h = block(h) }
            if let upsample = stage.upsample { h = upsample(h) }
        }
        h = convOut(NFKReferenceRounding.silu(ltx2PixelNorm(h, eps: 1e-6)))
        let targetFrames = max(frames * NFKMLXLTX2AudioVAEConfiguration.compression
                                   - (NFKMLXLTX2AudioVAEConfiguration.compression - 1), 1)
        let targetBins = configuration.melBins
        h = h[0..., 0 ..< min(h.dim(1), targetFrames), 0 ..< min(h.dim(2), targetBins)]
        let missingFrames = targetFrames - h.dim(1), missingBins = targetBins - h.dim(2)
        if missingFrames > 0 || missingBins > 0 {
            h = padded(h, widths: [IntOrPair(0), IntOrPair((0, max(missingFrames, 0))),
                                   IntOrPair((0, max(missingBins, 0))), IntOrPair(0)])
        }
        return h
    }
}

/// Reading, building, and loading the LTX-2 audio decoder.
public enum NFKMLXLTX2AudioVAE {

    /// The decoder geometry an `audio_vae/config.json` describes. Refuses by name what this decoder does
    /// not build: group normalization, attention, and causality along any axis but time.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXLTX2AudioVAEConfiguration {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "AutoencoderKLLTX2Audio" {
            throw NFKMLXError.unsupportedConfiguration("this reads an LTX-2 audio autoencoder, not \(kind)")
        }
        if let norm = json["norm_type"] as? String, norm != "pixel" {
            throw NFKMLXError.unsupportedConfiguration(
                "this decoder builds the per-pixel RMS norm; the release asks for \(norm) normalization")
        }
        if let axis = json["causality_axis"] as? String, axis != "height" {
            throw NFKMLXError.unsupportedConfiguration(
                "this decoder is causal along time (the height axis); the release asks for \(axis)")
        }
        if (json["mid_block_add_attention"] as? NSNumber)?.boolValue == true
            || !((json["attn_resolutions"] as? [Any])?.isEmpty ?? true) {
            throw NFKMLXError.unsupportedConfiguration(
                "this decoder carries no attention blocks; the release asks for them")
        }
        let base = NFKMLXLTX2AudioVAEConfiguration.ltx23
        return NFKMLXLTX2AudioVAEConfiguration(
            baseChannels: (json["base_channels"] as? NSNumber)?.intValue ?? base.baseChannels,
            outputChannels: (json["output_channels"] as? NSNumber)?.intValue ?? base.outputChannels,
            channelMultipliers: (json["ch_mult"] as? [NSNumber])?.map(\.intValue) ?? base.channelMultipliers,
            resnetBlocks: (json["num_res_blocks"] as? NSNumber)?.intValue ?? base.resnetBlocks,
            latentChannels: (json["latent_channels"] as? NSNumber)?.intValue ?? base.latentChannels,
            melBins: (json["mel_bins"] as? NSNumber)?.intValue ?? base.melBins,
            sampleRate: (json["sample_rate"] as? NSNumber)?.intValue ?? base.sampleRate,
            melHopLength: (json["mel_hop_length"] as? NSNumber)?.intValue ?? base.melHopLength,
            inChannels: (json["in_channels"] as? NSNumber)?.intValue ?? base.inChannels)
    }

    /// Loads a release's `audio_vae/` directory into `decoder`: the `decoder.` tensors, 4-D convolutions
    /// transposed to MLX's `[out, kH, kW, in]`; the encoder is not read. Returns the packed latents'
    /// mean and standard deviation, which the pipeline denormalizes by before unpacking.
    @discardableResult
    public static func loadWeights(into decoder: NFKMLXLTX2AudioDecoderNet, from directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws
        -> (mean: MLXArray, std: MLXArray) {
        let arrays = try NFKMLXReleaseWeights.materializedArrays(
            inDirectory: directory, precision: precision,
            remap: { $0.hasPrefix("decoder.") || $0.hasPrefix("latents_") ? $0 : nil },
            transform: { key, value in (key, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value) })
        return try apply(arrays, to: decoder)
    }

    /// Applies `decoder.`-prefixed tensors (already in MLX's layout) and returns the latent statistics.
    @discardableResult
    static func apply(_ arrays: [(String, MLXArray)], to decoder: NFKMLXLTX2AudioDecoderNet) throws
        -> (mean: MLXArray, std: MLXArray) {
        var statistics = [String: MLXArray]()
        var mapped = [(String, MLXArray)]()
        for (key, value) in arrays {
            if key == "latents_mean" || key == "latents_std" {
                statistics[key] = value.asType(.float32)
            } else if key.hasPrefix("decoder.") {
                mapped.append((String(key.dropFirst("decoder.".count)), value))
            }
        }
        try NFKMLXWeights.apply(mapped, to: decoder)
        let width = decoder.configuration.latentChannels * decoder.configuration.latentMelBins
        return (statistics["latents_mean"] ?? MLXArray.zeros([width]),
                statistics["latents_std"] ?? MLXArray.ones([width]))
    }
}
