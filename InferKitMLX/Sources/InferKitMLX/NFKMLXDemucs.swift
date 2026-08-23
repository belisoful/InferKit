//
//  NFKMLXDemucs.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Demucs separates a music mix into stems (drums, bass, other, vocals). This is the time-domain U-Net:
// a 1-D convolutional encoder (stride-4, GLU) down to a recurrent bottleneck and a mirrored
// transposed-conv decoder with skip connections, ending in `stems × channels`. It runs as an audio →
// audio backend (`NFKMLXDemucsBackend`): audio under `NFKInputAudio`, four stems under their own result
// keys. Tensors flow NLC (channels last), the layout MLX's Conv1d uses.
//
// The same network is the speech denoiser with `stems == 1` (see `NFKMLXDenoiser`), so the configuration
// carries everything the two released families differ in: the music model is stereo, six blocks deep,
// mixes decoder channels with a kernel-3 convolution, runs a bidirectional bottleneck, and resamples ×2
// with a polyphase filter; the speech models are mono, five deep, kernel 1, causal (one direction), and
// resample ×4 with a half-sample-shift filter. The 1-D transposed convolution is a ConvTranspose2d over
// a singleton width.

/// How the waveform is resampled around the encoder/decoder stack.
public enum NFKMLXDemucsResampler: Sendable {
    /// The half-sample-shift sinc filter of the reference `upsample2`/`downsample2`, applied in ×2 steps.
    case halfSampleShift
    /// `julius.resample_frac`: one windowed-sinc kernel per output phase, convolved with a stride of the
    /// input rate. The music model's path.
    case fractional
}

/// Demucs sizing. The defaults are the released music model (Défossez et al., "Music Source Separation
/// in the Waveform Domain"); `NFKMLXDenoiser` overrides them for the speech models.
public struct NFKMLXDemucsConfiguration: Sendable {
    public var audioChannels: Int = 2
    public var baseChannels: Int = 64
    public var depth: Int = 6
    public var stems: Int = 4
    /// Layers in the recurrent bottleneck between the encoder and decoder stacks (the reference uses two).
    public var lstmLayers: Int = 2
    /// True when the bottleneck runs the sequence in both directions and projects the concatenation back
    /// through a linear. The released speech models are causal, so they run one direction and carry no
    /// projection.
    public var bidirectional: Bool = true
    /// Kernel size of the decoder's channel-mixing convolution — the reference `context`, which gives the
    /// transposed convolution a view of neighboring time steps.
    public var context: Int = 3
    /// The factor the waveform is resampled by before encoding (and back after); 1 disables it.
    public var resample: Int = 2
    public var resampler: NFKMLXDemucsResampler = .fractional
    /// True when the result is trimmed to the input length from the middle. The music model's decoder
    /// shrinks symmetrically around the mix, because its `context` is wider than one sample; the speech
    /// models are length-preserving, so their reference drops the trailing padding instead.
    public var centersOutput: Bool = true
    /// True when the network divides the input by its standard deviation and rescales the output by it.
    public var normalize: Bool = false
    /// Added to the standard deviation before dividing, so a silent input does not blow up.
    public var normalizationFloor: Float = 1e-5
    public init() {}
    var stemNames: [String] { ["drums", "bass", "other", "vocals"] }
}

/// A 1-D transposed convolution, implemented as a 2-D transposed conv over a singleton width.
final class NFKDemucsConvT1d: Module {
    @ModuleInfo(key: "conv") var conv: ConvTransposed2d

    /// - Parameter bias: false for a filterbank-style decoder, which carries only a weight matrix.
    init(_ inChannels: Int, _ outChannels: Int, kernel: Int, stride: Int, padding: Int, bias: Bool = true) {
        _conv.wrappedValue = ConvTransposed2d(inputChannels: inChannels, outputChannels: outChannels,
                                              kernelSize: IntOrPair((kernel, 1)), stride: IntOrPair((stride, 1)),
                                              padding: IntOrPair((padding, 0)), bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l, c) = (x.shape[0], x.shape[1], x.shape[2])
        let out = conv(x.reshaped([b, l, 1, c]))
        return out.reshaped([out.shape[0], out.shape[1], out.shape[3]])
    }
}

/// An encoder block: strided conv → ReLU → 1×1 conv → GLU (halves the channels back).
final class NFKDemucsEncoder: Module {
    let conv1: Conv1d
    let conv2: Conv1d

    init(_ inChannels: Int, _ outChannels: Int) {
        // The reference convolves without padding (`nn.Conv1d(chin, hidden, 8, 4)`).
        conv1 = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 8, stride: 4)
        conv2 = Conv1d(inputChannels: outChannels, outputChannels: outChannels * 2, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        glu(conv2(relu(conv1(x))), axis: -1)
    }
}

/// A decoder block: 1×1 conv → GLU → transposed conv (upsample ×4), with a ReLU on all but the last.
final class NFKDemucsDecoder: Module {
    let conv1: Conv1d
    @ModuleInfo(key: "convt") var convt: NFKDemucsConvT1d

    init(_ inChannels: Int, _ outChannels: Int, context: Int) {
        // The reference mixes channels with a `context`-wide convolution and upsamples, both without
        // padding (`nn.Conv1d(hidden, 2 * hidden, context)`, `nn.ConvTranspose1d(hidden, chout, 8, 4)`).
        conv1 = Conv1d(inputChannels: inChannels, outputChannels: inChannels * 2, kernelSize: context)
        _convt.wrappedValue = NFKDemucsConvT1d(inChannels, outChannels, kernel: 8, stride: 4, padding: 0)
    }

    func callAsFunction(_ x: MLXArray, last: Bool) -> MLXArray {
        let y = convt(glu(conv1(x), axis: -1))
        return last ? y : relu(y)
    }
}

/// The Demucs time-domain separation network. `[1, L, audioChannels]` → `[1, L, stems·audioChannels]`.
/// Band-limited ×2 resampling, as the reference applies around the encoder/decoder stack. Both directions
/// convolve the signal with a windowed-sinc kernel and interleave (or average) the result, so the model
/// sees the sample rate it was trained at. Naive duplication or dropping would alias.
enum NFKDemucsResample {
    private static let zeros = 56

    /// The windowed-sinc half-sample-shift kernel, shaped `[1, taps, 1]` for `Conv1d` over `[N, L, 1]`.
    private static func kernel() -> MLXArray {
        let taps = 2 * zeros
        let windowLength = 4 * zeros + 1
        var values = [Float](repeating: 0, count: taps)
        for i in 0 ..< taps {
            // The odd samples of a symmetric Hann window, matching `hann_window(4*zeros+1)[1::2]`.
            let windowIndex = 2 * i + 1
            let window = 0.5 - 0.5 * cosf(2 * .pi * Float(windowIndex) / Float(windowLength - 1))
            let t = (-Float(zeros) + 0.5 + Float(i)) * .pi
            values[i] = (t == 0 ? 1 : sinf(t) / t) * window
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [1, taps, 1]) }
    }

    /// Applies the kernel to every channel independently — the decoder side carries one channel per stem.
    private static func filter(_ x: MLXArray) -> MLXArray {
        let channels = x.shape[2]
        let single = kernel()
        let weight = channels == 1 ? single : repeated(single, count: channels, axis: 0)
        return conv1d(x, weight, padding: zeros, groups: channels)
    }

    /// `[N, L, C]` → `[N, 2L, C]`: the original samples interleaved with half-shifted ones.
    static func upsample2(_ x: MLXArray) -> MLXArray {
        let (n, length, channels) = (x.shape[0], x.shape[1], x.shape[2])
        let shifted = filter(x)[0..., 1..., 0...][0..., 0 ..< length, 0...]
        return stacked([x, shifted], axis: 2).reshaped([n, 2 * length, channels])
    }

    /// `[N, L, C]` → `[N, L/2, C]`: the even samples averaged with the filtered odd ones.
    static func downsample2(_ x: MLXArray) -> MLXArray {
        var x = x
        if x.shape[1] % 2 != 0 {
            x = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 1)), IntOrPair((0, 0))], mode: .constant)
        }
        let even = x[0..., .stride(by: 2), 0...]
        let odd = x[0..., 1..., 0...][0..., .stride(by: 2), 0...]
        let length = odd.shape[1]
        let filtered = filter(odd)[0..., ..<(-1), 0...]
        return (even + filtered[0..., 0 ..< length, 0...]) * 0.5
    }
}

/// Rate matching for the models whose front end is tied to one sample rate. The mel front ends assume
/// their model's rate, so a clip recorded at another one has to be resampled before it reaches them —
/// otherwise every frequency lands in the wrong mel bin and the analysis is quietly wrong rather than
/// obviously broken.
enum NFKMLXAudioRate {
    private static func greatestCommonDivisor(_ a: Int, _ b: Int) -> Int {
        var (a, b) = (abs(a), abs(b))
        while b != 0 {
            (a, b) = (b, a % b)
        }
        return max(a, 1)
    }

    /// Resamples a mono clip, returning it untouched when the rates already agree. The ratio is
    /// reduced first: at 44100 → 16000 the polyphase filter would otherwise build 16000 kernels
    /// instead of 160.
    static func matched(_ samples: [Float], from old: Int, to new: Int) -> [Float] {
        guard old > 0, new > 0, old != new, !samples.isEmpty else { return samples }
        let divisor = greatestCommonDivisor(old, new)
        let tensor = samples.withUnsafeBufferPointer { MLXArray($0, [1, samples.count, 1]) }
        let resampled = NFKDemucsFractionalResample.resample(tensor, from: old / divisor, to: new / divisor)
        eval(resampled)
        return resampled.reshaped([-1]).asArray(Float.self)
    }
}

/// `julius.resample_frac`, the polyphase resampler the music model runs around its encoder/decoder
/// stack. Each output phase has its own windowed-sinc kernel; convolving with a stride of the input rate
/// produces all phases at once, and interleaving them gives the resampled signal.
enum NFKDemucsFractionalResample {
    private static let zeros = 24
    private static let rolloff: Float = 0.945

    /// One kernel per output phase, shaped `[new, taps, 1]`, with the edge padding they need per side.
    private static func filters(from old: Int, to new: Int) -> (weight: MLXArray, width: Int) {
        // Rolling the cutoff below Nyquist suppresses the edge artifacts a plain sinc leaves behind.
        let rate = Float(min(old, new)) * rolloff
        let width = Int(ceil(Float(zeros * old) / rate))
        let taps = 2 * width + old
        var values = [Float](repeating: 0, count: new * taps)
        for phase in 0 ..< new {
            var sum: Float = 0
            for tap in 0 ..< taps {
                var t = (-Float(phase) / Float(new) + Float(tap - width) / Float(old)) * rate
                t = .pi * max(-Float(zeros), min(Float(zeros), t))
                let window = powf(cosf(t / Float(zeros) / 2), 2)
                let value = (t == 0 ? 1 : sinf(t) / t) * window
                values[phase * taps + tap] = value
                sum += value
            }
            // Renormalized so a constant signal comes through unchanged.
            for tap in 0 ..< taps {
                values[phase * taps + tap] /= sum
            }
        }
        return (values.withUnsafeBufferPointer { MLXArray($0, [new, taps, 1]) }, width)
    }

    /// `[N, L, C]` → `[N, new·L/old, C]`.
    static func resample(_ x: MLXArray, from old: Int, to new: Int) -> MLXArray {
        guard old != new else { return x }
        let (batch, length, channels) = (x.shape[0], x.shape[1], x.shape[2])
        let (weight, width) = filters(from: old, to: new)
        // Every channel filters independently, so they ride along as batch rows.
        var signal = x.transposed(0, 2, 1).reshaped([batch * channels, length, 1])
        signal = MLX.padded(signal, widths: [IntOrPair((0, 0)), IntOrPair((width, width + old)), IntOrPair((0, 0))],
                            mode: .edge)
        let phases = conv1d(signal, weight, stride: old)                 // [N·C, steps, new]
        let outputLength = new * length / old
        let stream = phases.reshaped([batch * channels, phases.shape[1] * new])[0..., 0 ..< outputLength]
        return stream.reshaped([batch, channels, outputLength]).transposed(0, 2, 1)
    }
}

/// The recurrent bottleneck the reference runs between the encoder and decoder stacks. The music model
/// is bidirectional: every layer runs the sequence forward and reversed, the two directions concatenate,
/// and a linear projects the result back to the bottleneck width. The released DNS speech checkpoints
/// are causal, so they carry one direction and no projection.
final class NFKDemucsBLSTM: Module {
    @ModuleInfo(key: "lstm") var layers: [LSTM]
    @ModuleInfo(key: "reverse") var reverse: [LSTM]
    @ModuleInfo(key: "linear") var projection: Linear?

    init(dimensions: Int, layers count: Int, bidirectional: Bool) {
        // A bidirectional layer hands the next one both directions, so every layer after the first takes
        // twice the width.
        func inputSize(_ index: Int) -> Int { bidirectional && index > 0 ? 2 * dimensions : dimensions }
        _layers.wrappedValue = (0 ..< count).map { LSTM(inputSize: inputSize($0), hiddenSize: dimensions) }
        _reverse.wrappedValue = bidirectional
            ? (0 ..< count).map { LSTM(inputSize: inputSize($0), hiddenSize: dimensions) }
            : []
        _projection.wrappedValue = bidirectional ? Linear(2 * dimensions, dimensions) : nil
    }

    /// `[N, L, C]` → `[N, L, C]`, running the sequence along the time axis.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        let backwards = MLXArray((0 ..< x.shape[1]).reversed().map { Int32($0) })
        for (index, layer) in layers.enumerated() {
            let forward = layer(h).0
            guard index < reverse.count else {
                h = forward
                continue
            }
            let backward = reverse[index](take(h, backwards, axis: 1)).0
            h = concatenated([forward, take(backward, backwards, axis: 1)], axis: -1)
        }
        if let projection {
            h = projection(h)
        }
        return h
    }
}

final class NFKMLXDemucsNet: Module {
    @ModuleInfo(key: "encoder") var encoder: [NFKDemucsEncoder]
    @ModuleInfo(key: "lstm") var lstm: NFKDemucsBLSTM
    @ModuleInfo(key: "decoder") var decoder: [NFKDemucsDecoder]

    let configuration: NFKMLXDemucsConfiguration

    init(_ configuration: NFKMLXDemucsConfiguration) {
        self.configuration = configuration
        var channels = configuration.audioChannels
        var width = configuration.baseChannels
        var encoders = [NFKDemucsEncoder]()
        var widths = [Int]()
        for _ in 0 ..< configuration.depth {
            encoders.append(NFKDemucsEncoder(channels, width))
            widths.append(width)
            channels = width
            width *= 2
        }
        _encoder.wrappedValue = encoders
        // The bottleneck runs at the deepest encoder block's output width.
        _lstm.wrappedValue = NFKDemucsBLSTM(dimensions: widths[configuration.depth - 1],
                                            layers: configuration.lstmLayers,
                                            bidirectional: configuration.bidirectional)

        var decoders = [NFKDemucsDecoder]()
        for i in 0 ..< configuration.depth {
            let inChannels = widths[configuration.depth - 1 - i]
            let outChannels = i == configuration.depth - 1
                ? configuration.stems * configuration.audioChannels
                : widths[configuration.depth - 2 - i]
            decoders.append(NFKDemucsDecoder(inChannels, outChannels, context: configuration.context))
        }
        _decoder.wrappedValue = decoders
    }

    /// The input length the encoder/decoder stack round-trips exactly, mirroring the reference's
    /// `valid_length`: each encoder block takes `ceil((length - kernel) / stride) + 1` and gains the
    /// `context - 1` samples its decoder convolution will consume, each decoder block grows the result
    /// back by `(length - 1) * stride + kernel`.
    func validLength(_ length: Int) -> Int {
        var value = length * configuration.resample
        for _ in 0 ..< configuration.depth {
            value = max(Int(ceil(Double(value - 8) / 4)) + 1, 1)
            value += configuration.context - 1
        }
        for _ in 0 ..< configuration.depth {
            value = (value - 1) * 4 + 8
        }
        return Int(ceil(Double(value) / Double(configuration.resample)))
    }

    /// Separates `[L]` mono or `[L, channels]` samples into `[1, L, stems·channels]`, the stems' channels
    /// interleaved as the reference's `[stems, channels]` view flattens them.
    func separate(_ samples: MLXArray) -> MLXArray {
        let length = samples.shape[0]
        // A mono clip feeds every input channel, so a stereo model still runs on one.
        var x = samples.ndim == 1
            ? repeated(samples.reshaped([1, length, 1]), count: configuration.audioChannels, axis: 2)
            : samples.reshaped([1, length, samples.shape[1]])

        // The reference divides by `floor + std` and rescales by `std`, with `std` fixed at 1 when
        // normalization is off — so the input is still scaled by 1/(1 + floor) and never scaled back.
        // Reproduced as written; the music model trains with normalization off.
        let deviation = configuration.normalize ? Self.standardDeviation(of: x) : MLXArray(Float(1))
        x = x / (configuration.normalizationFloor + deviation)

        let target = validLength(length)
        if target > length {
            x = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, target - length)), IntOrPair((0, 0))],
                           mode: .constant)
        }
        x = resampled(x, from: 1, to: configuration.resample)

        var skips = [MLXArray]()
        for block in encoder {
            x = block(x)
            skips.append(x)
        }
        x = lstm(x)                                             // the recurrent bottleneck
        for (i, block) in decoder.enumerated() {
            let skip = Self.centerTrimmed(skips[configuration.depth - 1 - i], to: x.shape[1])
            let common = min(x.shape[1], skip.shape[1])
            x = x[0..., 0 ..< common, 0...] + skip[0..., 0 ..< common, 0...]
            x = block(x, last: i == configuration.depth - 1)
        }
        x = resampled(x, from: configuration.resample, to: 1)
        if x.shape[1] < length {
            x = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, length - x.shape[1])), IntOrPair((0, 0))],
                           mode: .edge)
        }
        let trimmed = configuration.centersOutput ? Self.centerTrimmed(x, to: length) : x
        return trimmed[0..., 0 ..< length, 0...] * deviation
    }

    /// The deviation the reference measures: over the mono mix rather than per channel, and unbiased.
    private static func standardDeviation(of x: MLXArray) -> MLXArray {
        let mono = mean(x, axis: 2, keepDims: true)
        let centered = mono - mean(mono, axis: 1, keepDims: true)
        return sqrt(sum(square(centered), axis: 1, keepDims: true) / Float(mono.shape[1] - 1))
    }

    /// The reference's `center_trim`: a `context` wider than one sample leaves the decoder shorter than
    /// the skip it adds, so the skip is trimmed to the middle rather than the head. With a `context` of
    /// one the two lengths already agree and this is an identity.
    private static func centerTrimmed(_ tensor: MLXArray, to length: Int) -> MLXArray {
        let delta = tensor.shape[1] - length
        guard delta > 0 else { return tensor }
        return tensor[0..., (delta / 2) ..< (tensor.shape[1] - (delta - delta / 2)), 0...]
    }

    private func resampled(_ x: MLXArray, from old: Int, to new: Int) -> MLXArray {
        guard old != new else { return x }
        switch configuration.resampler {
        case .halfSampleShift:
            var y = x
            for _ in 0 ..< (max(old, new) / 2) {
                y = new > old ? NFKDemucsResample.upsample2(y) : NFKDemucsResample.downsample2(y)
            }
            return y
        case .fractional:
            return NFKDemucsFractionalResample.resample(x, from: old, to: new)
        }
    }
}

/// Demucs stem separation as an InferKit backend. Reads `NFKInputAudio`; returns one `NFKAudioAsset`
/// per stem under its name ("drums", "bass", "other", "vocals").
@objc(NFKMLXDemucsBackend)
public final class NFKMLXDemucsBackend: NSObject, NFKInferenceBackend {

    private let net: NFKMLXDemucsNet
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXDemucsNet, identifier: String = "demucs", outputDirectory: URL = FileManager.default.temporaryDirectory) {
        self.net = net
        self.identifier = identifier
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else {
            throw NFKMLXError.unsupportedInput
        }
        let input = samples.withUnsafeBufferPointer { MLXArray($0, [samples.count]) }
        let stems = net.separate(input)
        eval(stems)
        let stemValues = stems[0].asType(Float.self)                // [L, stems]

        var outputs: [String: Any] = [:]
        let length = stemValues.shape[0]
        let channels = net.configuration.audioChannels
        for (index, name) in net.configuration.stemNames.enumerated() where index < net.configuration.stems {
            // Each stem owns `channels` consecutive output channels, already interleaved for the file.
            let stem = stemValues[0..., (index * channels) ..< ((index + 1) * channels)].asArray(Float.self)
            let url = outputDirectory.appendingPathComponent("demucs-\(name)-\(UUID().uuidString).wav")
            try NFKMLXWaveFile.write(samples: stem, sampleRate: sampleRate, channels: channels, to: url)
            outputs[name] = NFKAudioAsset(fileURL: url, durationSeconds: Double(length) / Double(sampleRate),
                                          sampleRate: Double(sampleRate), channelCount: channels)
        }
        return NFKInferenceResult(outputs: outputs)
    }

    private static func audio(from request: NFKInferenceRequest) -> (samples: [Float], sampleRate: Int)? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.read(data)
        }
        if let data = value as? Data { return NFKMLXWaveFile.read(data) }
        return nil
    }
}

/// Registration and weight loading for Demucs.
@objc(NFKMLXDemucs)
public final class NFKMLXDemucs: NSObject {

    @objc public static let modelName = "demucs"

    static func makeNet(_ configuration: NFKMLXDemucsConfiguration = NFKMLXDemucsConfiguration()) -> NFKMLXDemucsNet {
        NFKMLXDemucsNet(configuration)
    }

    /// Builds a Demucs stem-separation backend directly from optional local weights — no registry
    /// required. A nil `weightsURL` builds random weights (`isReady` is true). Reads audio under
    /// `NFKInputAudio`, returns a stem `NFKAudioAsset` under each of "drums"/"bass"/"other"/"vocals".
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXDemucsNet(NFKMLXDemucsConfiguration())
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXDemucsHolder(net)
        return NFKMLXDemucsBackend(net: holder.net)
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
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

    /// Registers Demucs (`demucs`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// The reference stores each block as an `nn.Sequential`, so its checkpoint keys are positional:
    /// `encoder.N.0` is the strided convolution and `encoder.N.2` the 1×1 before the GLU, while
    /// `decoder.N.0` is the channel mixer and `decoder.N.2` the transposed convolution. Map those onto
    /// the module's names; a checkpoint already in the module's layout passes through unchanged.
    static func remapReferenceKey(_ key: String) -> String {
        let parts = key.split(separator: ".").map(String.init)
        guard parts.count >= 4, parts[0] == "encoder" || parts[0] == "decoder", Int(parts[1]) != nil else {
            return key
        }
        let tail = parts[3...].joined(separator: ".")
        switch (parts[0], parts[2]) {
        case ("encoder", "0"): return "encoder.\(parts[1]).conv1.\(tail)"
        case ("encoder", "2"): return "encoder.\(parts[1]).conv2.\(tail)"
        case ("decoder", "0"): return "decoder.\(parts[1]).conv1.\(tail)"
        case ("decoder", "2"): return "decoder.\(parts[1]).convt.conv.\(tail)"
        default: return key
        }
    }

    /// Loads a safetensors checkpoint — the reference layout or the module's own. Block keys are
    /// Sequential indices (see ``remapReferenceKey(_:)``); convolution weights transpose from PyTorch's
    /// layout (3-D Conv1d `[out,in,k]` → `[out,k,in]`, 4-D `[out,in,kH,kW]` → `[out,kH,kW,in]`); and the
    /// decoder's transposed convolutions need their own fix, because PyTorch stores `ConvTranspose1d` as
    /// `[in, out, kernel]` while the module's `ConvTransposed2d` over a singleton width expects
    /// `[out, kernel, 1, in]`. The plain Conv1d transpose would silently mis-shape those.
    static func loadWeights(into net: NFKMLXDemucsNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        var mapped = raw.compactMap { key, value -> (String, MLXArray)? in
            if isBottleneckKey(key) { return nil }              // handled below: two biases fold into one
            let name = remapReferenceKey(key)
            if checkpoint.needsConvTranspose, name.hasSuffix("convt.conv.weight"), value.ndim == 3 {
                return (name, value.transposed(1, 2, 0).expandedDimensions(axis: 2))
            }
            if checkpoint.needsConvTranspose, value.ndim == 4 { return (name, value.transposed(0, 2, 3, 1)) }
            if checkpoint.needsConvTranspose, value.ndim == 3 { return (name, value.transposed(0, 2, 1)) }
            return (name, value)
        }
        mapped += bottleneckWeights(from: raw)
        try NFKMLXWeights.apply(mapped, to: net)
    }

    private static func isBottleneckKey(_ key: String) -> Bool {
        key.hasPrefix("lstm.lstm.weight_") || key.hasPrefix("lstm.lstm.bias_")
    }

    /// Maps the reference's `lstm.lstm.*_l<N>[_reverse]` tensors onto the module's stacked `LSTM` layers.
    /// PyTorch keeps the input and hidden biases separate and adds both each step, while MLX carries a
    /// single bias, so the two fold together. The gate order is `i, f, g, o` in both, so the matrices
    /// transfer unchanged.
    private static func bottleneckWeights(from raw: [String: MLXArray]) -> [(String, MLXArray)] {
        var result = [(String, MLXArray)]()
        for layer in 0 ..< 32 {
            guard raw["lstm.lstm.weight_ih_l\(layer)"] != nil else { break }
            for (suffix, module) in [("", "lstm.lstm"), ("_reverse", "lstm.reverse")] {
                guard let inputWeight = raw["lstm.lstm.weight_ih_l\(layer)\(suffix)"] else { continue }
                result.append(("\(module).\(layer).Wx", inputWeight))
                if let hiddenWeight = raw["lstm.lstm.weight_hh_l\(layer)\(suffix)"] {
                    result.append(("\(module).\(layer).Wh", hiddenWeight))
                }
                var bias: MLXArray?
                for name in ["bias_ih", "bias_hh"] {
                    guard let value = raw["lstm.lstm.\(name)_l\(layer)\(suffix)"] else { continue }
                    bias = bias.map { $0 + value } ?? value
                }
                if let bias {
                    result.append(("\(module).\(layer).bias", bias))
                }
            }
        }
        return result
    }
}

private final class NFKMLXDemucsHolder: @unchecked Sendable {
    let net: NFKMLXDemucsNet
    init(_ net: NFKMLXDemucsNet) { self.net = net }
}
