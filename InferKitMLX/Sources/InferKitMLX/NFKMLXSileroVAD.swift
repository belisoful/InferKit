//
//  NFKMLXSileroVAD.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Voice activity detection marks where speech occurs in an audio clip. This is Silero VAD v6: a learned
// STFT front end, a four-layer convolutional encoder, and a single-layer LSTM decoder that scores one
// speech probability per 512-sample chunk (32 ms at 16 kHz). Consecutive above-threshold chunks merge
// into spans. The result is an NSArray<NFKAudioSegment *> under NFKOutputSegments.
//
// The model streams: each chunk carries a 64-sample look-back context from the previous chunk (so the
// STFT has proper support at the chunk boundary), and the LSTM state threads across chunks. A whole clip
// runs as one pass, the chunks laid out along the LSTM's sequence axis, which carries the state exactly
// as a chunk-by-chunk stream would. Tensors flow as `[chunks, length, channels]` (MLX's NLC).
//
// v6 differs from v5 in the STFT padding alone: v5 pads the 576-sample input symmetrically by 128 and
// then drops the first transform frame; v6 pads the right by 64, which lands four frames directly with no
// drop. The rest of the architecture — encoder widths, LSTM, decoder — is shared.
//
// The reference (snakers4/silero-vad) keeps its weights under `_model.*`; this port names them, and
// `NFKMLXSileroVAD.remapReferenceKey` maps between the two.

/// Silero VAD dimensions and thresholds. The defaults are the released 16 kHz v6 model.
public struct NFKMLXSileroVADConfiguration: Sendable {
    public var sampleRate: Int
    /// The STFT window length, `sampleRate / 62.5`. 256 at 16 kHz.
    public var filterLength: Int
    /// The STFT hop, `filterLength / 2`.
    public var hopLength: Int
    /// The chunk the model scores, `filterLength * 2`. 512 at 16 kHz (32 ms).
    public var numSamples: Int
    /// The look-back prepended to each chunk, `filterLength / 4`. 64 at 16 kHz.
    public var contextSamples: Int
    /// The right reflection pad the v6 STFT applies to each `context + chunk` input. Equal to
    /// `contextSamples` for the released model, held separately because it is the v5/v6 difference.
    public var stftPadRight: Int
    public var hidden: Int
    public var threshold: Float

    public init(sampleRate: Int = 16000, filterLength: Int = 256, hopLength: Int = 128,
                numSamples: Int = 512, contextSamples: Int = 64, stftPadRight: Int = 64,
                hidden: Int = 128, threshold: Float = 0.5) {
        self.sampleRate = sampleRate
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.numSamples = numSamples
        self.contextSamples = contextSamples
        self.stftPadRight = stftPadRight
        self.hidden = hidden
        self.threshold = threshold
    }

    /// The released 16 kHz Silero VAD v6.
    public static let v6 = NFKMLXSileroVADConfiguration()

    /// `[chunks]` size the STFT magnitude carries per chunk: `(context + chunk + padRight - filter) / hop + 1`.
    var framesPerChunk: Int {
        (contextSamples + numSamples + stftPadRight - filterLength) / hopLength + 1
    }

    /// The number of features the STFT magnitude produces, `hopLength + 1`.
    var featureSize: Int { hopLength + 1 }
    /// The real/imaginary split point of the transform's `filterLength + 2` channels.
    var cutoff: Int { filterLength / 2 + 1 }
    /// The seconds one scored chunk spans.
    var chunkSeconds: Double { Double(numSamples) / Double(sampleRate) }
}

/// The learned STFT and convolutional encoder. The STFT basis is a stored checkpoint tensor, so it is a
/// (non-trained) convolution here rather than a computed constant.
final class NFKSileroEncoder: Module {
    @ModuleInfo(key: "stft") var stft: Conv1d
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "conv3") var conv3: Conv1d
    @ModuleInfo(key: "conv4") var conv4: Conv1d

    let cutoff: Int

    init(_ c: NFKMLXSileroVADConfiguration) {
        cutoff = c.cutoff
        _stft.wrappedValue = Conv1d(inputChannels: 1, outputChannels: c.filterLength + 2,
                                    kernelSize: c.filterLength, stride: c.hopLength, bias: false)
        _conv1.wrappedValue = Conv1d(inputChannels: c.featureSize, outputChannels: 128, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: 128, outputChannels: 64, kernelSize: 3, stride: 2, padding: 1)
        _conv3.wrappedValue = Conv1d(inputChannels: 64, outputChannels: 64, kernelSize: 3, stride: 2, padding: 1)
        _conv4.wrappedValue = Conv1d(inputChannels: 64, outputChannels: 128, kernelSize: 3, padding: 1)
    }

    /// `[chunks, samples, 1]` → per-chunk encoder features `[chunks, 1, 128]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let transform = stft(x)                                  // [chunks, frames, filter + 2]
        let real = transform[0..., 0..., 0 ..< cutoff]           // [chunks, frames, cutoff]
        let imag = transform[0..., 0..., cutoff ..< (2 * cutoff)]
        var f = sqrt(real * real + imag * imag)                  // magnitude, [chunks, frames, features]
        f = relu(conv1(f))
        f = relu(conv2(f))
        f = relu(conv3(f))
        f = relu(conv4(f))
        return f                                                 // [chunks, 1, 128]
    }
}

/// The LSTM decoder scoring one speech probability per chunk. Dropout is identity in evaluation, so it is
/// omitted rather than modeled.
final class NFKSileroDecoder: Module {
    @ModuleInfo(key: "rnn") var rnn: LSTM
    @ModuleInfo(key: "final") var final: Conv1d

    init(_ c: NFKMLXSileroVADConfiguration) {
        _rnn.wrappedValue = LSTM(inputSize: c.hidden, hiddenSize: c.hidden)
        _final.wrappedValue = Conv1d(inputChannels: c.hidden, outputChannels: 1, kernelSize: 1)
    }

    /// Encoder features `[chunks, 1, hidden]` → speech probability per chunk `[chunks]`. The chunks are
    /// the LSTM's sequence axis (batch one), which threads the recurrent state across them.
    func callAsFunction(_ feats: MLXArray) -> MLXArray {
        let sequence = feats.reshaped([1, feats.shape[0], feats.shape[2]])   // [1, chunks, hidden]
        let hidden = rnn(sequence).0                                          // [1, chunks, hidden]
        let logits = final(relu(hidden))                                      // [1, chunks, 1]
        return sigmoid(logits).reshaped([-1])                                 // [chunks]
    }
}

/// The Silero VAD network: the STFT encoder and the LSTM decoder, plus the chunking and span extraction.
final class NFKMLXSileroVADNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKSileroEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKSileroDecoder

    let configuration: NFKMLXSileroVADConfiguration

    init(_ c: NFKMLXSileroVADConfiguration) {
        configuration = c
        _encoder.wrappedValue = NFKSileroEncoder(c)
        _decoder.wrappedValue = NFKSileroDecoder(c)
    }

    /// Speech probability per 512-sample chunk for a mono waveform.
    func speechProbabilities(_ samples: [Float], sampleRate: Int) -> [Float] {
        let matched = NFKMLXAudioRate.matched(samples, from: sampleRate, to: configuration.sampleRate)
        guard !matched.isEmpty else { return [] }
        let input = Self.chunkedInput(matched, configuration)
        let probabilities = decoder(encoder(input))
        eval(probabilities)
        return probabilities.asArray(Float.self)
    }

    /// Builds the model input `[chunks, context + chunk + padRight, 1]` from a mono clip: the clip is
    /// padded to a whole number of chunks, each chunk is prefixed with the previous chunk's final
    /// `contextSamples` (the first chunk with zeros), and each `context + chunk` row is right-reflected by
    /// `stftPadRight`, reproducing the reference's per-row `ReflectionPad1d`.
    static func chunkedInput(_ samples: [Float], _ c: NFKMLXSileroVADConfiguration) -> MLXArray {
        let num = c.numSamples, ctx = c.contextSamples, pad = c.stftPadRight
        var signal = samples
        let remainder = signal.count % num
        if remainder != 0 {
            signal += [Float](repeating: 0, count: num - remainder)
        }
        let chunks = signal.count / num
        let rowIn = ctx + num                                   // context + chunk, before the STFT pad
        let rowOut = rowIn + pad

        var data = [Float](repeating: 0, count: chunks * rowOut)
        for chunk in 0 ..< chunks {
            let base = chunk * rowOut
            // The look-back context: the previous chunk's final `ctx` samples, zeros for the first chunk.
            if chunk > 0 {
                let prevTail = (chunk - 1) * num + (num - ctx)
                for i in 0 ..< ctx {
                    data[base + i] = signal[prevTail + i]
                }
            }
            // The chunk itself.
            let chunkStart = chunk * num
            for i in 0 ..< num {
                data[base + ctx + i] = signal[chunkStart + i]
            }
            // Right reflection pad over the `rowIn` row: mirror inward without repeating the edge sample.
            for i in 0 ..< pad {
                data[base + rowIn + i] = data[base + rowIn - 2 - i]
            }
        }
        return data.withUnsafeBufferPointer { MLXArray($0, [chunks, rowOut, 1]) }
    }

    /// Detects speech spans, merging consecutive above-threshold chunks. Each chunk spans
    /// `numSamples / sampleRate` seconds; resampling preserves duration, so the seconds are the caller's.
    func detect(_ samples: [Float], sampleRate: Int) -> [NFKAudioSegment] {
        let probabilities = speechProbabilities(samples, sampleRate: sampleRate)
        let chunkSeconds = configuration.chunkSeconds

        var segments: [NFKAudioSegment] = []
        var runStart: Int? = nil
        var runSum: Float = 0
        func close(_ end: Int) {
            guard let start = runStart else { return }
            let confidence = Double(runSum) / Double(end - start)
            segments.append(NFKAudioSegment(startSeconds: Double(start) * chunkSeconds,
                                            endSeconds: Double(end) * chunkSeconds,
                                            label: nil, confidence: min(max(confidence, 0), 1)))
            runStart = nil
            runSum = 0
        }
        for (chunk, probability) in probabilities.enumerated() {
            if probability >= configuration.threshold {
                if runStart == nil { runStart = chunk }
                runSum += probability
            } else {
                close(chunk)
            }
        }
        close(probabilities.count)
        return segments
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKSileroVADHolder: @unchecked Sendable {
    let net: NFKMLXSileroVADNet
    init(_ net: NFKMLXSileroVADNet) { self.net = net }
}

/// Silero voice activity detection as an InferKit backend. Reads `NFKInputAudio`; returns speech spans
/// under `NFKOutputSegments`.
@objc(NFKMLXSileroVADBackend)
public final class NFKMLXSileroVADBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKSileroVADHolder
    private let identifier: String

    init(net: NFKMLXSileroVADNet, identifier: String) {
        holder = NFKSileroVADHolder(net)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else {
            throw NFKMLXError.unsupportedInput
        }
        let segments = holder.net.detect(samples, sampleRate: sampleRate)
        return NFKInferenceResult(outputs: [NFKOutputSegments: segments])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
            do {
                job.finish(with: try self.runInference(for: request))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
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

/// Registration and weight loading for Silero voice activity detection.
@objc(NFKMLXSileroVAD)
public final class NFKMLXSileroVAD: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "silero-vad"

    static func makeNet(_ configuration: NFKMLXSileroVADConfiguration = .v6) -> NFKMLXSileroVADNet {
        NFKMLXSileroVADNet(configuration)
    }

    /// Builds a Silero VAD backend directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run inference off the render thread.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXSileroVADBackend(net: net, identifier: modelName)
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required. Blocking on the
    /// network; run off the render thread.
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

    /// Registers Silero VAD (`silero-vad`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Maps a reference `_model.*` key onto the module's names. The encoder convolutions are a numbered
    /// `reparam_conv` list; the decoder's output convolution is the third entry of a `Sequential`.
    static func remapReferenceKey(_ key: String) -> String {
        if key == "_model.stft.forward_basis_buffer" {
            return "encoder.stft.weight"
        }
        let parts = key.split(separator: ".").map(String.init)
        if parts.count == 5, parts[0] == "_model", parts[1] == "encoder", let index = Int(parts[2]),
           parts[3] == "reparam_conv" {
            return "encoder.conv\(index + 1).\(parts[4])"
        }
        if key.hasPrefix("_model.decoder.decoder.2.") {
            return "decoder.final." + key.dropFirst("_model.decoder.decoder.2.".count)
        }
        return key
    }

    /// Loads a checkpoint, transposing Conv1d weights `[out, in, k]` → MLX's `[out, k, in]`. The LSTM is
    /// handled apart: PyTorch keeps the input and hidden biases separate and adds both each step, while
    /// MLX carries one bias, so the two fold together; the gate order is `i, f, g, o` in both, so the
    /// matrices transfer unchanged (the same treatment `NFKMLXDemucs` gives its bottleneck LSTM).
    static func loadWeights(into net: NFKMLXSileroVADNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays

        var mapped = [(String, MLXArray)]()
        if let inputWeight = raw["_model.decoder.rnn.weight_ih"] {
            mapped.append(("decoder.rnn.Wx", inputWeight))
        }
        if let hiddenWeight = raw["_model.decoder.rnn.weight_hh"] {
            mapped.append(("decoder.rnn.Wh", hiddenWeight))
        }
        var bias: MLXArray?
        for name in ["bias_ih", "bias_hh"] {
            guard let value = raw["_model.decoder.rnn.\(name)"] else { continue }
            bias = bias.map { $0 + value } ?? value
        }
        if let bias {
            mapped.append(("decoder.rnn.bias", bias))
        }

        for (key, value) in raw {
            // The released checkpoint carries an 8 kHz branch (`_model_8k.*`) this port does not run,
            // and the LSTM is applied above.
            if key.hasPrefix("_model_8k") || key.hasPrefix("_model.decoder.rnn.") { continue }
            let name = remapReferenceKey(key)
            let transposed = checkpoint.needsConvTranspose && value.ndim == 3
            mapped.append((name, transposed ? value.transposed(0, 2, 1) : value))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
