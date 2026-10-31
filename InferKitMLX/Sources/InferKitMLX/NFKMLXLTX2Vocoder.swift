//
//  NFKMLXLTX2Vocoder.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

// The LTX-2.3 vocoder with bandwidth extension (`LTX2VocoderWithBWE`, Lightricks). Two BigVGAN-style
// generators run in sequence. The first turns the stereo log-mel spectrogram the audio decoder emits
// into a 16 kHz waveform. A causal STFT (left padding only, `window − hop` samples, its windowed DFT
// basis read from the checkpoint) and the checkpoint's mel filterbank turn that waveform back into a
// log-mel, which the second generator turns into a 48 kHz residual. A fixed Hann-windowed sinc
// resampler lifts the 16 kHz waveform by 3x for the skip, and the sum is clamped to [−1, 1].
//
// A generator is a convolution in, then per upsample stage a transposed convolution and the MEAN of
// three dilated resnets, then an activation and a convolution out. Every activation is snake-beta,
// `x + sin²(αx) / (β + 1e-9)` with α and β stored as logarithms, wrapped in a 2x anti-aliased
// up-activate-down step: a replicate-padded transposed convolution with a Kaiser-windowed sinc, the
// activation, and a replicate-padded strided convolution with the same kind of filter. Both filters are
// stored in the checkpoint and read from it.
//
// Tensors flow as `[batch, samples, channels]` (MLX's NLC); the reference is `[B, C, L]`.

/// One generator's geometry.
public struct NFKMLXLTX2GeneratorConfiguration: Sendable {
    public var inChannels: Int
    public var hiddenChannels: Int
    public var outChannels: Int
    public var upsampleKernels: [Int]
    public var upsampleFactors: [Int]
    public var resnetKernels: [Int]
    public var resnetDilations: [[Int]]
    public var antialiasKernel: Int
    public var finalBias: Bool

    public init(inChannels: Int, hiddenChannels: Int, outChannels: Int, upsampleKernels: [Int],
                upsampleFactors: [Int], resnetKernels: [Int] = [3, 7, 11],
                resnetDilations: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]], antialiasKernel: Int = 12,
                finalBias: Bool = false) {
        self.inChannels = inChannels
        self.hiddenChannels = hiddenChannels
        self.outChannels = outChannels
        self.upsampleKernels = upsampleKernels
        self.upsampleFactors = upsampleFactors
        self.resnetKernels = resnetKernels
        self.resnetDilations = resnetDilations
        self.antialiasKernel = antialiasKernel
        self.finalBias = finalBias
    }

    /// The samples one input frame becomes.
    public var totalUpsampling: Int { upsampleFactors.reduce(1, *) }
}

/// The geometry of an LTX-2 vocoder with bandwidth extension, read from a release's
/// `vocoder/config.json`.
public struct NFKMLXLTX2VocoderConfiguration: Sendable {
    public var vocoder: NFKMLXLTX2GeneratorConfiguration
    public var bandwidthExtension: NFKMLXLTX2GeneratorConfiguration
    public var filterLength: Int
    public var hopLength: Int
    public var windowLength: Int
    public var melChannels: Int
    public var inputSampleRate: Int
    public var outputSampleRate: Int

    public init(vocoder: NFKMLXLTX2GeneratorConfiguration, bandwidthExtension: NFKMLXLTX2GeneratorConfiguration,
                filterLength: Int = 512, hopLength: Int = 80, windowLength: Int = 512, melChannels: Int = 64,
                inputSampleRate: Int = 16000, outputSampleRate: Int = 48000) {
        self.vocoder = vocoder
        self.bandwidthExtension = bandwidthExtension
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.windowLength = windowLength
        self.melChannels = melChannels
        self.inputSampleRate = inputSampleRate
        self.outputSampleRate = outputSampleRate
    }

    /// The LTX-2.3 release's vocoder (`diffusers/LTX-2.3-Diffusers`, `vocoder/config.json`).
    public static let ltx23 = NFKMLXLTX2VocoderConfiguration(
        vocoder: NFKMLXLTX2GeneratorConfiguration(inChannels: 128, hiddenChannels: 1536, outChannels: 2,
                                                  upsampleKernels: [11, 4, 4, 4, 4, 4],
                                                  upsampleFactors: [5, 2, 2, 2, 2, 2]),
        bandwidthExtension: NFKMLXLTX2GeneratorConfiguration(inChannels: 128, hiddenChannels: 512, outChannels: 2,
                                                    upsampleKernels: [12, 11, 4, 4, 4],
                                                    upsampleFactors: [6, 5, 2, 2, 2]))

    /// The release's arrangement at a small width, its STFT scaled with it, for weight-free tests.
    public static let tiny = NFKMLXLTX2VocoderConfiguration(
        vocoder: NFKMLXLTX2GeneratorConfiguration(inChannels: 16, hiddenChannels: 32, outChannels: 2,
                                                  upsampleKernels: [11, 4], upsampleFactors: [5, 2]),
        bandwidthExtension: NFKMLXLTX2GeneratorConfiguration(inChannels: 16, hiddenChannels: 16, outChannels: 2,
                                                    upsampleKernels: [12, 11], upsampleFactors: [6, 5]),
        filterLength: 32, hopLength: 10, windowLength: 32, melChannels: 8)
}

/// Snake-beta, `x + sin²(e^α x) / (e^β + 1e-9)`, its parameters per channel and stored as logarithms.
final class NFKLTX2SnakeBeta: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(_ channels: Int) {
        _alpha.wrappedValue = MLXArray.zeros([channels])
        _beta.wrappedValue = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + (1 / (exp(beta) + 1e-9)) * sin(x * exp(alpha)).square()
    }
}

/// Filters every channel of `x` `[B, L, C]` with the same single-channel kernel by folding the channels
/// into the batch: `body` takes and returns `[B · C, L, 1]`.
func ltx2PerChannel(_ x: MLXArray, _ body: (MLXArray) -> MLXArray) -> MLXArray {
    let (b, l, c) = (x.dim(0), x.dim(1), x.dim(2))
    let folded = body(x.transposed(0, 2, 1).reshaped([b * c, l, 1]))
    return folded.reshaped([b, c, folded.dim(1)]).transposed(0, 2, 1)
}

/// A replicate-padded transposed convolution of every channel with `filter` at stride `ratio`, scaled by
/// `ratio` and cropped: the reference's `UpSample1d`.
func ltx2Upsample(_ x: MLXArray, filter: MLXArray, ratio: Int, pad: Int, cropLeft: Int, cropRight: Int) -> MLXArray {
    let kernel = filter.reshaped([1, -1, 1]).asType(x.dtype)
    return ltx2PerChannel(x) { folded in
        let widened = padded(folded, widths: [IntOrPair(0), IntOrPair(pad), IntOrPair(0)], mode: .edge)
        let up = Float(ratio) * convTransposed1d(widened, kernel, stride: ratio)
        return up[0..., cropLeft ..< (up.dim(1) - cropRight)]
    }
}

/// The 2x anti-aliased activation: upsample, snake-beta, downsample, with the Kaiser filters the
/// checkpoint stores.
final class NFKLTX2AntiAliasActivation: Module {
    @ModuleInfo(key: "upsample") var upsample: NFKLTX2Filter
    @ModuleInfo(key: "act") var act: NFKLTX2SnakeBeta
    @ModuleInfo(key: "downsample") var downsample: NFKLTX2Filter
    let kernel: Int

    init(_ channels: Int, kernel: Int) {
        self.kernel = kernel
        _upsample.wrappedValue = NFKLTX2Filter(kernel)
        _act.wrappedValue = NFKLTX2SnakeBeta(channels)
        _downsample.wrappedValue = NFKLTX2Filter(kernel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let ratio = 2
        let pad = kernel / ratio - 1
        let up = ltx2Upsample(x, filter: upsample.filter, ratio: ratio, pad: pad,
                              cropLeft: pad * ratio + (kernel - ratio) / 2,
                              cropRight: pad * ratio + (kernel - ratio + 1) / 2)
        let activated = act(up)
        let downKernel = downsample.filter.reshaped([1, -1, 1]).asType(x.dtype)
        let left = kernel / 2 + kernel % 2 - 1, right = kernel / 2
        return ltx2PerChannel(activated) { folded in
            conv1d(padded(folded, widths: [IntOrPair(0), IntOrPair((left, right)), IntOrPair(0)], mode: .edge),
                   downKernel, stride: ratio)
        }
    }
}

/// A stored low-pass filter, `[1, 1, kernel]`.
final class NFKLTX2Filter: Module {
    @ParameterInfo(key: "filter") var filter: MLXArray

    init(_ kernel: Int) { _filter.wrappedValue = MLXArray.zeros([1, 1, kernel]) }
}

/// A dilated resnet: per dilation, an anti-aliased activation, a dilated convolution, another
/// activation and an undilated convolution, added back.
final class NFKLTX2VocoderResnet: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "acts1") var acts1: [NFKLTX2AntiAliasActivation]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "acts2") var acts2: [NFKLTX2AntiAliasActivation]

    init(channels: Int, kernel: Int, dilations: [Int], antialiasKernel: Int) {
        _convs1.wrappedValue = dilations.map {
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel,
                   padding: $0 * (kernel - 1) / 2, dilation: $0)
        }
        _acts1.wrappedValue = dilations.map { _ in NFKLTX2AntiAliasActivation(channels, kernel: antialiasKernel) }
        _convs2.wrappedValue = dilations.map { _ in
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: (kernel - 1) / 2)
        }
        _acts2.wrappedValue = dilations.map { _ in NFKLTX2AntiAliasActivation(channels, kernel: antialiasKernel) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for index in convs1.indices {
            h = h + convs2[index](acts2[index](convs1[index](acts1[index](h))))
        }
        return h
    }
}

/// One BigVGAN-style generator: `[B, frames, inChannels]` → `[B, frames · totalUpsampling, outChannels]`.
final class NFKLTX2Generator: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv1d
    @ModuleInfo(key: "upsamplers") var upsamplers: [ConvTransposed1d]
    @ModuleInfo(key: "resnets") var resnets: [NFKLTX2VocoderResnet]
    @ModuleInfo(key: "act_out") var actOut: NFKLTX2AntiAliasActivation
    @ModuleInfo(key: "conv_out") var convOut: Conv1d

    let resnetsPerUpsample: Int

    init(_ c: NFKMLXLTX2GeneratorConfiguration) {
        resnetsPerUpsample = c.resnetKernels.count
        _convIn.wrappedValue = Conv1d(inputChannels: c.inChannels, outputChannels: c.hiddenChannels, kernelSize: 7,
                                      padding: 3)
        var ups = [ConvTransposed1d](), blocks = [NFKLTX2VocoderResnet]()
        var width = c.hiddenChannels
        for (factor, kernel) in zip(c.upsampleFactors, c.upsampleKernels) {
            ups.append(ConvTransposed1d(inputChannels: width, outputChannels: width / 2, kernelSize: kernel,
                                        stride: factor, padding: (kernel - factor) / 2))
            width /= 2
            for (resnetKernel, dilations) in zip(c.resnetKernels, c.resnetDilations) {
                blocks.append(NFKLTX2VocoderResnet(channels: width, kernel: resnetKernel, dilations: dilations,
                                                   antialiasKernel: c.antialiasKernel))
            }
        }
        _upsamplers.wrappedValue = ups
        _resnets.wrappedValue = blocks
        _actOut.wrappedValue = NFKLTX2AntiAliasActivation(width, kernel: c.antialiasKernel)
        _convOut.wrappedValue = Conv1d(inputChannels: width, outputChannels: c.outChannels, kernelSize: 7, padding: 3,
                                       bias: c.finalBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for (index, upsampler) in upsamplers.enumerated() {
            h = upsampler(h)
            let stage = resnets[(index * resnetsPerUpsample) ..< ((index + 1) * resnetsPerUpsample)]
            h = stage.map { $0(h) }.reduce(MLXArray(Float(0)).asType(h.dtype), +) / Float(resnetsPerUpsample)
        }
        return convOut(actOut(h))
    }
}

/// The causal STFT's stored basis.
final class NFKLTX2STFT: Module {
    @ParameterInfo(key: "forward_basis") var forwardBasis: MLXArray

    init(filterLength: Int) {
        _forwardBasis.wrappedValue = MLXArray.zeros([2 * (filterLength / 2 + 1), filterLength, 1])
    }
}

/// The log-mel front end of the bandwidth extension: the causal STFT and the stored mel filterbank.
final class NFKLTX2MelSTFT: Module {
    @ModuleInfo(key: "stft_fn") var stft: NFKLTX2STFT
    @ParameterInfo(key: "mel_basis") var melBasis: MLXArray

    init(filterLength: Int, melChannels: Int) {
        _stft.wrappedValue = NFKLTX2STFT(filterLength: filterLength)
        _melBasis.wrappedValue = MLXArray.zeros([melChannels, filterLength / 2 + 1])
    }

    /// `waveform` `[N, samples]` → the log-mel `[N, frames, melChannels]`.
    func logMel(_ waveform: MLXArray, hop: Int, window: Int) -> MLXArray {
        let input = padded(waveform.expandedDimensions(axis: -1),
                           widths: [IntOrPair(0), IntOrPair((max(0, window - hop), 0)), IntOrPair(0)])
        let spectrum = conv1d(input, stft.forwardBasis.asType(input.dtype), stride: hop)
        let bins = spectrum.dim(2) / 2
        let real = spectrum[0..., 0..., 0 ..< bins], imaginary = spectrum[0..., 0..., bins...]
        let magnitude = sqrt(real.square() + imaginary.square())
        let mel = matmul(magnitude, melBasis.asType(magnitude.dtype).transposed())
        return log(maximum(mel, MLXArray(Float(1e-5)).asType(mel.dtype)))
    }
}

/// The LTX-2 vocoder with bandwidth extension: a stereo log-mel spectrogram
/// `[B, frames, melBins, 2]` (the audio decoder's layout) → a waveform `[B, samples, 2]` at
/// ``NFKMLXLTX2VocoderConfiguration/outputSampleRate``.
public final class NFKMLXLTX2VocoderNet: Module {
    @ModuleInfo(key: "vocoder") var vocoder: NFKLTX2Generator
    @ModuleInfo(key: "bwe_generator") var extensionGenerator: NFKLTX2Generator
    @ModuleInfo(key: "mel_stft") var melSTFT: NFKLTX2MelSTFT

    public let configuration: NFKMLXLTX2VocoderConfiguration
    /// The fixed 3x resampler's Hann-windowed sinc, which the release does not store. Held as plain
    /// values so it is not a parameter a checkpoint must supply.
    let resamplerTaps: [Float]

    /// ``resamplerTaps`` as an array.
    var resamplerFilter: MLXArray { MLXArray(resamplerTaps) }

    public init(_ c: NFKMLXLTX2VocoderConfiguration) {
        configuration = c
        _vocoder.wrappedValue = NFKLTX2Generator(c.vocoder)
        _extensionGenerator.wrappedValue = NFKLTX2Generator(c.bandwidthExtension)
        _melSTFT.wrappedValue = NFKLTX2MelSTFT(filterLength: c.filterLength, melChannels: c.melChannels)
        resamplerTaps = Self.hannSinc(ratio: c.outputSampleRate / c.inputSampleRate)
    }

    /// The reference's Hann-windowed sinc for an integer `ratio`: rolloff 0.99, six zero crossings, in
    /// float32 as the reference computes it.
    static func hannSinc(ratio: Int) -> [Float] {
        let rolloff: Float = 0.99, zeroCrossings: Float = 6
        let width = Int((zeroCrossings / rolloff).rounded(.up))
        let kernel = 2 * width * ratio + 1
        return (0 ..< kernel).map { index in
            let time = (Float(index) / Float(ratio) - Float(width)) * rolloff
            let clamped = min(max(time, -zeroCrossings), zeroCrossings)
            let window = pow(cos(clamped * Float.pi / zeroCrossings / 2), 2)
            let sinc = time == 0 ? 1 : sin(Float.pi * time) / (Float.pi * time)
            return sinc * window * rolloff / Float(ratio)
        }
    }

    /// The first generator's waveform `[B, samples, 2]` at the input rate.
    func baseWaveform(_ mel: MLXArray) -> MLXArray {
        vocoder(Self.flattenedMel(mel))
    }

    /// `[B, frames, bins, channels]` → `[B, frames, channels · bins]`, channel-major as the reference
    /// flattens `[B, C, bins, frames]`.
    static func flattenedMel(_ mel: MLXArray) -> MLXArray {
        mel.transposed(0, 1, 3, 2).reshaped([mel.dim(0), mel.dim(1), mel.dim(2) * mel.dim(3)])
    }

    /// The log-mel the extension generator reads, `[B, frames, melChannels, 2]`, from a waveform.
    func extensionMel(_ waveform: MLXArray) -> MLXArray {
        let (b, samples, channels) = (waveform.dim(0), waveform.dim(1), waveform.dim(2))
        let mono = waveform.transposed(0, 2, 1).reshaped([b * channels, samples])
        let mel = melSTFT.logMel(mono, hop: configuration.hopLength, window: configuration.windowLength)
        return mel.reshaped([b, channels, mel.dim(1), mel.dim(2)]).transposed(0, 2, 3, 1)
    }

    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var waveform = baseWaveform(mel)
        let samples = waveform.dim(1)
        let remainder = samples % configuration.hopLength
        if remainder != 0 {
            waveform = padded(waveform, widths: [IntOrPair(0), IntOrPair((0, configuration.hopLength - remainder)),
                                                 IntOrPair(0)])
        }
        let residual = extensionGenerator(Self.flattenedMel(extensionMel(waveform)))
        let ratio = configuration.outputSampleRate / configuration.inputSampleRate
        let width = (resamplerTaps.count - 1) / (2 * ratio)
        let skip = ltx2Upsample(waveform, filter: resamplerFilter, ratio: ratio, pad: width,
                                cropLeft: 2 * width * ratio, cropRight: resamplerTaps.count - ratio)
        let output = samples * configuration.outputSampleRate / configuration.inputSampleRate
        return clip(residual + skip, min: -1, max: 1)[0..., 0 ..< output]
    }
}

/// Reading, building, and loading the LTX-2 vocoder.
public enum NFKMLXLTX2Vocoder {

    /// The geometry a `vocoder/config.json` describes. Refuses by name what this port does not build:
    /// the LTX-2.0 vocoder without bandwidth extension, and any activation but anti-aliased snake-beta.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXLTX2VocoderConfiguration {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "LTX2VocoderWithBWE" {
            throw NFKMLXError.unsupportedConfiguration(
                "this reads the LTX-2 vocoder with bandwidth extension, not \(kind)")
        }
        for prefix in ["", "bwe_"] {
            if let activation = json["\(prefix)act_fn"] as? String, activation != "snakebeta" {
                throw NFKMLXError.unsupportedConfiguration(
                    "this vocoder builds snake-beta activations; the release asks for \(activation)")
            }
            if (json["\(prefix)antialias"] as? NSNumber)?.boolValue == false {
                throw NFKMLXError.unsupportedConfiguration(
                    "this vocoder builds anti-aliased activations; the release turns them off")
            }
            if let ratio = (json["\(prefix)antialias_ratio"] as? NSNumber)?.intValue, ratio != 2 {
                throw NFKMLXError.unsupportedConfiguration(
                    "this vocoder anti-aliases at 2x; the release asks for \(ratio)x")
            }
            if json["\(prefix)final_act_fn"] is String {
                throw NFKMLXError.unsupportedConfiguration(
                    "this vocoder applies no final activation; the release asks for one")
            }
        }
        let base = NFKMLXLTX2VocoderConfiguration.ltx23
        func generator(_ prefix: String, _ fallback: NFKMLXLTX2GeneratorConfiguration) -> NFKMLXLTX2GeneratorConfiguration {
            func integer(_ key: String, _ value: Int) -> Int { (json[prefix + key] as? NSNumber)?.intValue ?? value }
            func integers(_ key: String, _ value: [Int]) -> [Int] {
                (json[prefix + key] as? [NSNumber])?.map(\.intValue) ?? value
            }
            return NFKMLXLTX2GeneratorConfiguration(
                inChannels: integer("in_channels", fallback.inChannels),
                hiddenChannels: integer("hidden_channels", fallback.hiddenChannels),
                outChannels: integer("out_channels", fallback.outChannels),
                upsampleKernels: integers("upsample_kernel_sizes", fallback.upsampleKernels),
                upsampleFactors: integers("upsample_factors", fallback.upsampleFactors),
                resnetKernels: integers("resnet_kernel_sizes", fallback.resnetKernels),
                resnetDilations: (json[prefix + "resnet_dilations"] as? [[NSNumber]])?.map { $0.map(\.intValue) }
                    ?? fallback.resnetDilations,
                antialiasKernel: integer("antialias_kernel_size", fallback.antialiasKernel),
                finalBias: (json[prefix + "final_bias"] as? NSNumber)?.boolValue ?? fallback.finalBias)
        }
        func integer(_ key: String, _ value: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? value }
        let configuration = NFKMLXLTX2VocoderConfiguration(
            vocoder: generator("", base.vocoder), bandwidthExtension: generator("bwe_", base.bandwidthExtension),
            filterLength: integer("filter_length", base.filterLength),
            hopLength: integer("hop_length", base.hopLength),
            windowLength: integer("window_length", base.windowLength),
            melChannels: integer("num_mel_channels", base.melChannels),
            inputSampleRate: integer("input_sampling_rate", base.inputSampleRate),
            outputSampleRate: integer("output_sampling_rate", base.outputSampleRate))
        guard configuration.outputSampleRate % configuration.inputSampleRate == 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "this vocoder resamples by an integer factor; the release asks for "
                + "\(configuration.inputSampleRate) → \(configuration.outputSampleRate) Hz")
        }
        return configuration
    }

    /// The MLX layout of a stored vocoder tensor, or nil for the inverse STFT basis, which generation
    /// never reads.
    static func mapped(_ key: String, _ value: MLXArray) -> (String, MLXArray)? {
        if key.hasSuffix("inverse_basis") { return nil }
        guard value.ndim == 3, !key.hasSuffix(".filter") else { return (key, value) }
        return (key, key.contains(".upsamplers.") ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1))
    }

    /// Loads a release's `vocoder/` directory into `net`.
    public static func loadWeights(into net: NFKMLXLTX2VocoderNet, from directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.materializedArrays(
            inDirectory: directory, precision: precision, transform: mapped)
        try NFKMLXWeights.apply(arrays, to: net)
    }
}
