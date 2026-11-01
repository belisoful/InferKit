//
//  NFKMLXLTX2AudioEncoder.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

// The LTX-2 audio autoencoder's encoder (`LTX2AudioEncoder`, Lightricks) and the log-mel front end
// Lightricks' trainer feeds it (`ltx_core`'s `AudioProcessor`). A stereo waveform is resampled to the
// encoder's rate with a Hann-windowed sinc (torchaudio's `resample`: rolloff 0.99, six zero crossings),
// framed by a centered, reflect-padded 1024-point STFT at the mel hop, and its magnitude is projected onto
// a Slaney-scale, Slaney-normalized mel bank and logged after clamping at 1e-5. The encoder mirrors the
// decoder: a causal convolution in, one stage per resolution of resnets, all but the last ending in a
// stride-2 convolution padded two rows in the past and one mel bin above, two mid resnets, the per-pixel
// norm, SiLU, and a causal convolution to the posterior's means and log variances. The trainer keeps the
// means, packs them `[T, channels · melBins]`, and normalizes them by the per-channel statistics at that
// packed width.
//
// Tensors flow as `[batch, time, mel, channels]`; the reference is `[B, C, T, M]`.

/// A stride-2 convolution padded two time rows before and one mel bin after, the reference's
/// `LTX2AudioDownsample` with `causality_axis="height"`.
final class NFKLTX2AudioDownsample: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(_ channels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: IntOrPair(3),
                                    stride: IntOrPair(2))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(padded(x, widths: [IntOrPair(0), IntOrPair((2, 0)), IntOrPair((0, 1)), IntOrPair(0)]))
    }
}

/// One encoder resolution: its resnets and, below the coarsest resolution, the downsample.
final class NFKLTX2AudioDownStage: Module {
    @ModuleInfo(key: "block") var block: [NFKLTX2AudioResnet]
    @ModuleInfo(key: "downsample") var downsample: NFKLTX2AudioDownsample?

    init(blocks: [NFKLTX2AudioResnet], downsample: NFKLTX2AudioDownsample?) {
        _block.wrappedValue = blocks
        _downsample.wrappedValue = downsample
    }
}

/// The LTX-2 audio encoder: a log-mel spectrogram `[B, T, melBins, inChannels]` → the posterior's means and
/// log variances `[B, ⌈T / 4⌉, melBins / 4, 2 · latentChannels]`.
public final class NFKMLXLTX2AudioEncoderNet: Module {
    @ModuleInfo(key: "conv_in") var convIn: NFKLTX2AudioCausalConv2d
    @ModuleInfo(key: "down") var down: [NFKLTX2AudioDownStage]
    @ModuleInfo(key: "mid") var mid: NFKLTX2AudioMid
    @ModuleInfo(key: "conv_out") var convOut: NFKLTX2AudioCausalConv2d

    public let configuration: NFKMLXLTX2AudioVAEConfiguration

    public init(_ c: NFKMLXLTX2AudioVAEConfiguration) {
        configuration = c
        _convIn.wrappedValue = NFKLTX2AudioCausalConv2d(c.inChannels, c.baseChannels, kernel: 3)
        var width = c.baseChannels
        var stages = [NFKLTX2AudioDownStage]()
        for (level, multiplier) in c.channelMultipliers.enumerated() {
            let out = c.baseChannels * multiplier
            var blocks = [NFKLTX2AudioResnet]()
            for _ in 0 ..< c.resnetBlocks {
                blocks.append(NFKLTX2AudioResnet(width, out))
                width = out
            }
            stages.append(NFKLTX2AudioDownStage(
                blocks: blocks,
                downsample: level != c.channelMultipliers.count - 1 ? NFKLTX2AudioDownsample(width) : nil))
        }
        _down.wrappedValue = stages
        _mid.wrappedValue = NFKLTX2AudioMid(width)
        _convOut.wrappedValue = NFKLTX2AudioCausalConv2d(width, 2 * c.latentChannels, kernel: 3)
    }

    /// The posterior's means then log variances for a log-mel spectrogram.
    public func callAsFunction(_ spectrogram: MLXArray) -> MLXArray {
        var h = convIn(spectrogram)
        for stage in down {
            for block in stage.block { h = block(h) }
            if let downsample = stage.downsample { h = downsample(h) }
        }
        h = mid.block2(mid.block1(h))
        return convOut(NFKReferenceRounding.silu(ltx2PixelNorm(h, eps: 1e-6)))
    }

    /// The packed, normalized latents `[B, T', latentChannels · melBins']` of a log-mel spectrogram: the
    /// means, channel-major in each frame, less the per-channel mean over the per-channel standard deviation.
    public func latents(_ spectrogram: MLXArray, statistics: (mean: MLXArray, std: MLXArray)) -> MLXArray {
        let means = self(spectrogram)[.ellipsis, 0 ..< configuration.latentChannels]
        let (b, t, m, c) = (means.dim(0), means.dim(1), means.dim(2), means.dim(3))
        let packed = means.transposed(0, 1, 3, 2).reshaped([b, t, c * m])
        return (packed - statistics.mean.asType(packed.dtype)) / statistics.std.asType(packed.dtype)
    }
}

/// The trainer's audio front end (`ltx_core`'s `AudioProcessor`): the resample to the encoder's rate and the
/// log-mel spectrogram it encodes.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXLTX2MelFrontEnd: Sendable {
    public var sampleRate: Int
    public var melBins: Int
    public var hopLength: Int
    public var fftSize: Int

    public init(sampleRate: Int = 16000, melBins: Int = 64, hopLength: Int = 160, fftSize: Int = 1024) {
        self.sampleRate = sampleRate
        self.melBins = melBins
        self.hopLength = hopLength
        self.fftSize = fftSize
    }

    /// The front end an audio autoencoder's configuration names.
    public init(_ configuration: NFKMLXLTX2AudioVAEConfiguration) {
        self.init(sampleRate: configuration.sampleRate, melBins: configuration.melBins,
                  hopLength: configuration.melHopLength)
    }

    /// `torchaudio.functional.resample` with its defaults (`sinc_interp_hann`, six zero crossings, rolloff
    /// 0.99): each channel of `waveform` `[channels, samples]` from `rate` to ``sampleRate``.
    public func resampled(_ waveform: MLXArray, from rate: Int) -> MLXArray {
        guard rate != sampleRate else { return waveform }
        func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
        let divisor = gcd(rate, sampleRate)
        let original = rate / divisor, target = sampleRate / divisor
        let crossings = 6.0, base = Double(min(original, target)) * 0.99
        let width = Int((crossings * Double(original) / base).rounded(.up))
        let taps = 2 * width + original
        var kernel = [Float](repeating: 0, count: target * taps)
        for phase in 0 ..< target {
            for tap in 0 ..< taps {
                var t = (-Double(phase) / Double(target) + Double(tap - width) / Double(original)) * base
                t = min(max(t, -crossings), crossings)
                let window = pow(cos(t * .pi / crossings / 2), 2)
                t *= .pi
                let sinc = t == 0 ? 1 : sin(t) / t
                kernel[phase * taps + tap] = Float(sinc * window * base / Double(original))
            }
        }
        let (channels, length) = (waveform.dim(0), waveform.dim(1))
        let padded = MLX.padded(waveform, widths: [IntOrPair(0), IntOrPair((width, width + original))])
        let frames = (padded.dim(1) - taps) / original + 1
        let starts = MLXArray((0 ..< frames).map { Int32($0 * original) }).reshaped([frames, 1])
        let windows = padded.take(starts + MLXArray((0 ..< taps).map { Int32($0) }).reshaped([1, taps]), axis: 1)
        let weights = MLXArray(kernel, [target, taps]).asType(waveform.dtype)
        let output = matmul(windows, weights.transposed()).reshaped([channels, frames * target])
        let kept = Int((Double(target) * Double(length) / Double(original)).rounded(.up))
        return output[0..., 0 ..< kept]
    }

    /// The log-mel spectrogram `[1, frames, melBins, channels]` of `waveform` `[channels, samples]` at
    /// ``sampleRate``: a centered, reflect-padded periodic-Hann STFT's magnitude through a Slaney mel bank,
    /// logged after clamping at 1e-5.
    public func logMel(_ waveform: MLXArray) -> MLXArray {
        let stft = NFKMLXComplexSTFT(nFFT: fftSize, hop: hopLength)
        let bank = NFKMLXMel.melFilters(sampleRate: sampleRate, bins: fftSize / 2 + 1, nMels: melBins)
        let channels = (0 ..< waveform.dim(0)).map { channel -> MLXArray in
            let magnitude = stft.transform(waveform[channel ..< (channel + 1)].asType(.float32)).magnitude[0]
            return log(maximum(matmul(magnitude.transposed(), bank), MLXArray(Float(1e-5))))
        }
        return MLX.stacked(channels, axis: -1).expandedDimensions(axis: 0)
    }
}

extension NFKMLXLTX2AudioVAE {

    /// Loads a release's `audio_vae/` directory into `encoder`: the `encoder.` tensors, 4-D convolutions
    /// transposed to MLX's layout. Returns the packed latents' per-channel mean and standard deviation, which
    /// ``NFKMLXLTX2AudioEncoderNet/latents(_:statistics:)`` normalizes by.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func loadEncoderWeights(into encoder: NFKMLXLTX2AudioEncoderNet, from directory: URL,
                                          precision: NFKMLXWeightPrecision = .float32) throws
        -> (mean: MLXArray, std: MLXArray) {
        let arrays = try NFKMLXReleaseWeights.materializedArrays(
            inDirectory: directory, precision: precision,
            remap: { $0.hasPrefix("encoder.") || $0.hasPrefix("latents_") ? $0 : nil },
            transform: { key, value in (key, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value) })
        var statistics = [String: MLXArray]()
        var mapped = [(String, MLXArray)]()
        for (key, value) in arrays {
            if key == "latents_mean" || key == "latents_std" {
                statistics[key] = value.asType(.float32)
            } else {
                mapped.append((String(key.dropFirst("encoder.".count)), value))
            }
        }
        try NFKMLXWeights.apply(mapped, to: encoder, verifyShapes: true)
        let width = encoder.configuration.latentChannels * encoder.configuration.latentMelBins
        return (statistics["latents_mean"] ?? MLXArray.zeros([width]),
                statistics["latents_std"] ?? MLXArray.ones([width]))
    }
}
