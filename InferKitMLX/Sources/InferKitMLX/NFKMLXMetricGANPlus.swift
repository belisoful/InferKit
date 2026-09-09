//
//  NFKMLXMetricGANPlus.swift
//  InferKitMLX
//
//  MetricGAN+ (speechbrain, Apache-2.0): a spectral-mask speech enhancer whose generator is a
//  two-layer bidirectional LSTM over log-magnitude frames, the smallest member of the restoration
//  family and the plumbing check for the rest of it.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The released MetricGAN+ geometry (speechbrain/metricgan-plus-voicebank).
public struct NFKMLXMetricGANPlusConfiguration: Sendable {
    public var sampleRate = 16_000
    public var fftSize = 512
    public var hopSize = 256
    public var hiddenSize = 200
    public var layerCount = 2
    public var projectionSize = 300
    public var negativeSlope: Float = 0.3
    public var bins: Int { fftSize / 2 + 1 }
    public init() {}
}

/// One bidirectional layer: a forward cell over the sequence and a reverse cell over its reversal,
/// their states concatenated.
final class NFKMetricGANBiLSTM: Module {
    @ModuleInfo(key: "forward") var forward: LSTM
    @ModuleInfo(key: "reverse") var reverse: LSTM

    init(inputSize: Int, hiddenSize: Int) {
        _forward.wrappedValue = LSTM(inputSize: inputSize, hiddenSize: hiddenSize)
        _reverse.wrappedValue = LSTM(inputSize: inputSize, hiddenSize: hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (ahead, _) = forward(x)
        let (behind, _) = reverse(x[0..., .stride(by: -1), 0...])
        return concatenated([ahead, behind[0..., .stride(by: -1), 0...]], axis: -1)
    }
}

/// `1.2 · sigmoid(slope · x)`, one learned slope per bin.
final class NFKMetricGANLearnableSigmoid: Module {
    @ParameterInfo(key: "slope") var slope: MLXArray

    init(bins: Int) {
        _slope.wrappedValue = MLXArray.ones([bins])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { 1.2 * sigmoid(slope * x) }
}

/// The MetricGAN+ generator: `log1p(|X|)` frames → BLSTM stack → Linear → LeakyReLU(0.3) → Linear →
/// learnable sigmoid → a mask over the bins.
public final class NFKMLXMetricGANPlusNet: Module {
    @ModuleInfo(key: "blstm") var blstm: [NFKMetricGANBiLSTM]
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "Learnable_sigmoid") var learnableSigmoid: NFKMetricGANLearnableSigmoid
    let configuration: NFKMLXMetricGANPlusConfiguration

    init(_ c: NFKMLXMetricGANPlusConfiguration) {
        configuration = c
        _blstm.wrappedValue = (0 ..< c.layerCount).map {
            NFKMetricGANBiLSTM(inputSize: $0 == 0 ? c.bins : 2 * c.hiddenSize, hiddenSize: c.hiddenSize)
        }
        _linear1.wrappedValue = Linear(2 * c.hiddenSize, c.projectionSize)
        _linear2.wrappedValue = Linear(c.projectionSize, c.bins)
        _learnableSigmoid.wrappedValue = NFKMetricGANLearnableSigmoid(bins: c.bins)
        super.init()
    }

    /// `[batch, frames, bins]` features → the mask, the same shape.
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = features
        for layer in blstm { x = layer(x) }
        x = leakyRelu(linear1(x), negativeSlope: configuration.negativeSlope)
        return learnableSigmoid(linear2(x))
    }
}

/// Holds the network for capture in a `@Sendable` body.
private final class NFKMetricGANHolder: @unchecked Sendable {
    let net: NFKMLXMetricGANPlusNet
    init(_ net: NFKMLXMetricGANPlusNet) { self.net = net }
}

/// MetricGAN+ as an InferKit backend: `NFKInputAudio` (any rate; resampled to 16 kHz) → the enhanced
/// clip under `NFKOutputAudio`.
@objc(NFKMLXMetricGANPlusBackend)
public final class NFKMLXMetricGANPlusBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKMetricGANHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXMetricGANPlusNet, identifier: String,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKMetricGANHolder(net)
        self.identifier = identifier
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, rate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let configuration = holder.net.configuration
        let matched = NFKMLXAudioRate.matched(samples, from: rate, to: configuration.sampleRate)
        let enhanced = Self.enhance(matched, net: holder.net)
        eval(enhanced)
        let stream = enhanced.reshaped([enhanced.shape.last!]).asArray(Float.self)
        let url = outputDirectory.appendingPathComponent("metricgan-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: stream, sampleRate: configuration.sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(stream.count) / Double(configuration.sampleRate),
                                  sampleRate: Double(configuration.sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    /// speechbrain's STFT: a 512-sample periodic Hamming window at hop 256, centered with ZERO padding.
    static func stft(_ configuration: NFKMLXMetricGANPlusConfiguration) -> NFKMLXComplexSTFT {
        let n = configuration.fftSize
        let hamming = (0 ..< n).map { 0.54 - 0.46 * cosf(2 * Float.pi * Float($0) / Float(n)) }
        return NFKMLXComplexSTFT(nFFT: n, hop: configuration.hopSize, window: MLXArray(hamming), zeroPadded: true)
    }

    /// `log1p(|X|)` frames `[1, frames, bins]`, the generator's input.
    static func features(_ samples: [Float], configuration: NFKMLXMetricGANPlusConfiguration) -> (MLXArray, phase: MLXArray) {
        let signal = samples.withUnsafeBufferPointer { MLXArray($0, [1, samples.count]) }
        let (magnitude, phase) = stft(configuration).transform(signal)
        return (log1p(magnitude).transposed(0, 2, 1), phase)
    }

    /// The reference `enhance_batch`: the mask times the features, `expm1` back to a magnitude, the
    /// noisy phase, the inverse STFT at the input's length (speechbrain's `resynthesize` passes
    /// `sig_length`), and a peak normalization.
    static func enhance(_ samples: [Float], net: NFKMLXMetricGANPlusNet) -> MLXArray {
        let configuration = net.configuration
        let (features, phase) = features(samples, configuration: configuration)
        let mask = net(features)
        let magnitude = expm1(mask * features).transposed(0, 2, 1)
        let waveform = stft(configuration).inverse(magnitude: magnitude, phase: phase, length: samples.count)
        let peak = abs(waveform).max()
        return waveform / (peak + 1e-14)
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
            do { job.finish(with: try self.runInference(for: request)) }
            catch { job.finish(withError: error as NSError) }
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

/// Registration and weight loading for MetricGAN+.
@objc(NFKMLXMetricGANPlus)
public final class NFKMLXMetricGANPlus: NSObject {
    @objc public static let modelName = "metricgan-plus"

    static func makeNet(_ configuration: NFKMLXMetricGANPlusConfiguration = .init()) -> NFKMLXMetricGANPlusNet {
        NFKMLXMetricGANPlusNet(configuration)
    }

    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL { try loadWeights(into: net, from: weightsURL) }
        return NFKMLXMetricGANPlusBackend(net: net, identifier: modelName)
    }

    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) }, completionHandler: completionHandler)
    }

    /// Registers `metricgan-plus` with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads the released `enhance_model.ckpt` (a plain state dict the native torch reader opens). The
    /// linears and the sigmoid slope match by name; each PyTorch LSTM layer's separate input/hidden
    /// weights and biases fold into MLX's `Wx`/`Wh`/`bias` under `blstm.N.forward` / `.reverse` (the
    /// gate order i, f, g, o is shared, so the matrices transfer as they are).
    static func loadWeights(into net: NFKMLXMetricGANPlusNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var mapped = [(String, MLXArray)]()
        for (key, value) in checkpoint.arrays where !key.hasPrefix("blstm.rnn.") {
            mapped.append((key, value))
        }
        for layer in 0 ..< net.configuration.layerCount {
            for (direction, tag) in [("forward", ""), ("reverse", "_reverse")] {
                let base = "blstm.rnn."
                guard let wx = checkpoint.arrays["\(base)weight_ih_l\(layer)\(tag)"],
                      let wh = checkpoint.arrays["\(base)weight_hh_l\(layer)\(tag)"],
                      let biasIH = checkpoint.arrays["\(base)bias_ih_l\(layer)\(tag)"],
                      let biasHH = checkpoint.arrays["\(base)bias_hh_l\(layer)\(tag)"] else {
                    throw NFKMLXError.weightsMismatch("MetricGAN+ layer \(layer) \(direction) LSTM weights are missing")
                }
                mapped.append(("blstm.\(layer).\(direction).Wx", wx))
                mapped.append(("blstm.\(layer).\(direction).Wh", wh))
                mapped.append(("blstm.\(layer).\(direction).bias", biasIH + biasHH))
            }
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }
}
