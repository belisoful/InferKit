//
//  NFKMLXAudioTagger.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

// Audio tagging labels the sounds present in a clip (speech, music, a dog, a siren). This is PANNs
// Cnn14: a log-mel spectrogram, normalized across its mel bands, feeds a stack of VGG-style blocks —
// two 3×3 convolutions and an average pooling each — and the result pools over time into one score per
// class. The scores are independent (a sigmoid per class), so a clip carries several tags at once. The
// result is an NSArray<NFKClassification *> under NFKOutputClassifications, most-confident first.
//
// The reference numbers its blocks from one (`conv_block1`) and stores its mel filterbank in the
// checkpoint, so the filterbank loads rather than being recomputed. Tensors flow in NHWC, where the mel
// band is the width axis. Resampling an input that is not already at the model's rate is a
// validation-sweep item.

/// Audio-tagger dimensions. Defaults are the released Cnn14 over AudioSet's 527 classes; `tiny` keeps
/// tests fast.
public struct NFKMLXAudioTaggerConfiguration: Sendable {
    public var mels: Int
    public var sampleRate: Int
    public var fftSize: Int
    public var hopSamples: Int
    /// The band the filterbank spans. Cnn14 is trained on 50 Hz to 14 kHz, not the whole spectrum.
    public var melMinimumHz: Float
    public var melMaximumHz: Float
    /// The first block's width; each block doubles it.
    public var baseChannels: Int
    public var blocks: Int
    /// The width of the embedding the classifier reads.
    public var embedding: Int
    public var classCount: Int
    public var topK: Int

    public init(mels: Int = 64, sampleRate: Int = 32000, fftSize: Int = 1024, hopSamples: Int = 320,
                melMinimumHz: Float = 50, melMaximumHz: Float = 14000, baseChannels: Int = 64,
                blocks: Int = 6, embedding: Int = 2048, classCount: Int = 527, topK: Int = 5) {
        self.mels = mels
        self.sampleRate = sampleRate
        self.fftSize = fftSize
        self.hopSamples = hopSamples
        self.melMinimumHz = melMinimumHz
        self.melMaximumHz = melMaximumHz
        self.baseChannels = baseChannels
        self.blocks = blocks
        self.embedding = embedding
        self.classCount = classCount
        self.topK = topK
    }

    public static let panns = NFKMLXAudioTaggerConfiguration()

    public static let tiny = NFKMLXAudioTaggerConfiguration(
        mels: 16, sampleRate: 16000, fftSize: 128, hopSamples: 40, melMinimumHz: 0, melMaximumHz: 8000,
        baseChannels: 4, blocks: 2, embedding: 8, classCount: 6, topK: 3)

    /// The width the last block emits.
    var outputChannels: Int { baseChannels * (1 << (blocks - 1)) }
}

/// The mel spectrogram the network is trained on. The reference stores its filterbank in the
/// checkpoint, so it loads rather than being recomputed; the default reproduces it for a randomly
/// initialized network.
///
/// Held outside the module graph deliberately. MLX reflects every `MLXArray` property into
/// `parameters()`, so a constant declared on a `Module` makes a real checkpoint look short.
final class NFKAudioTaggerFrontEnd {
    private let configuration: NFKMLXAudioTaggerConfiguration
    /// `[bins, mels]`.
    private(set) var filterbank: MLXArray

    init(_ configuration: NFKMLXAudioTaggerConfiguration) {
        self.configuration = configuration
        filterbank = NFKMLXMel.melFilters(sampleRate: configuration.sampleRate,
                                          bins: configuration.fftSize / 2 + 1, nMels: configuration.mels,
                                          fMinimum: configuration.melMinimumHz,
                                          fMaximum: configuration.melMaximumHz)
    }

    func load(filterbank raw: MLXArray) {
        filterbank = raw
    }

    /// `[1, frames, mels]` — a centered short-time transform, the power spectrum through the mel
    /// filterbank, and the reference's decibel scale (`10·log₁₀`, floored at `1e-10`).
    func logMel(_ samples: [Float]) -> MLXArray {
        let power = NFKMLXMel.powerSpectrogram(samples, nFFT: configuration.fftSize,
                                               hop: configuration.hopSamples, dropsFinalFrame: false)
        let mel = power.matmul(filterbank)
        let decibels = 10 * log(maximum(mel, MLXArray(Float(1e-10)))) / logf(10)
        return decibels.reshaped([1, decibels.shape[0], configuration.mels])
    }
}

/// One Cnn14 block: two 3×3 convolutions, each normalized and activated, then an average pooling that
/// halves both axes. The reference gives its last block a pooling window of one, so that block passes
/// its result through at full resolution.
final class NFKPANNsConvBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm

    private let pools: Bool

    init(_ inChannels: Int, _ outChannels: Int, pools: Bool = true) {
        self.pools = pools
        _conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                     kernelSize: 3, padding: 1, bias: false)
        _bn1.wrappedValue = BatchNorm(featureCount: outChannels)
        _conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels,
                                     kernelSize: 3, padding: 1, bias: false)
        _bn2.wrappedValue = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activated = relu(bn2(conv2(relu(bn1(conv1(x))))))
        return pools ? NFKMLXResample.averagePooled(activated, kernel: 2, stride: 2) : activated
    }
}

/// The PANNs Cnn14 tagging network: a spectrogram normalization, a stack of convolution blocks, and a
/// classifier over the pooled embedding.
final class NFKMLXAudioTaggerNet: Module {
    @ModuleInfo(key: "bn0") var bn0: BatchNorm
    @ModuleInfo(key: "conv_block") var blocks: [NFKPANNsConvBlock]
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc_audioset") var classifier: Linear

    let configuration: NFKMLXAudioTaggerConfiguration
    let frontEnd: NFKAudioTaggerFrontEnd

    init(_ c: NFKMLXAudioTaggerConfiguration) {
        configuration = c
        frontEnd = NFKAudioTaggerFrontEnd(c)
        // The reference normalizes the spectrogram across its mel bands, so the count of bands is what
        // this layer is sized by.
        _bn0.wrappedValue = BatchNorm(featureCount: c.mels)
        var channels = 1
        var stack: [NFKPANNsConvBlock] = []
        for block in 0 ..< c.blocks {
            let out = c.baseChannels * (1 << block)
            stack.append(NFKPANNsConvBlock(channels, out, pools: block < c.blocks - 1))
            channels = out
        }
        _blocks.wrappedValue = stack
        _fc1.wrappedValue = Linear(channels, c.embedding)
        _classifier.wrappedValue = Linear(c.embedding, c.classCount)
    }

    /// The clip embedding `[1, embedding]` the classifier reads, from a log-mel spectrogram
    /// `[1, frames, mels]`.
    func embedding(_ mel: MLXArray) -> MLXArray {
        var x = bn0(mel).reshaped([1, mel.shape[1], mel.shape[2], 1])  // single-channel spectrogram image
        for block in blocks {
            x = block(x)
        }
        let overBands = mean(x, axis: 2)                               // [1, frames, channels]
        // The reference sums the two pools rather than choosing between them.
        let pooled = overBands.max(axis: 1) + mean(overBands, axis: 1)
        return relu(fc1(pooled))
    }

    /// Per-class logits `[1, classCount]` from a log-mel spectrogram `[1, frames, mels]`.
    func logits(_ mel: MLXArray) -> MLXArray {
        classifier(embedding(mel))
    }

    /// Tags a mono waveform: the highest-scoring classes, most-confident first. `labels` names classes
    /// when available. The front end's filterbank is built for one rate, so a clip recorded at another
    /// is resampled to it first — feeding it directly would put every frequency in the wrong mel bin.
    func tag(_ samples: [Float], sampleRate: Int, labels: [String]?) -> [NFKClassification] {
        let matched = NFKMLXAudioRate.matched(samples, from: sampleRate, to: configuration.sampleRate)
        let scores = sigmoid(logits(frontEnd.logMel(matched)))
        eval(scores)
        let values = scores.reshaped([-1]).asArray(Float.self)

        let ranked = values.enumerated().sorted { $0.element > $1.element }.prefix(configuration.topK)
        return ranked.map { index, score in
            let label = labels.flatMap { index < $0.count ? $0[index] : nil }
            return NFKClassification(label: label, classIndex: index, confidence: Double(score))
        }
    }
}

/// Holds the network and class labels for capture in the backend's `@Sendable` closure.
private final class NFKAudioTaggerHolder: @unchecked Sendable {
    let net: NFKMLXAudioTaggerNet
    let labels: [String]?
    init(_ net: NFKMLXAudioTaggerNet, labels: [String]?) {
        self.net = net
        self.labels = labels
    }
}

/// Audio tagging as an InferKit backend. Reads `NFKInputAudio`; returns tags under
/// `NFKOutputClassifications`.
@objc(NFKMLXAudioTaggerBackend)
public final class NFKMLXAudioTaggerBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKAudioTaggerHolder
    private let identifier: String

    init(net: NFKMLXAudioTaggerNet, identifier: String, labels: [String]?) {
        holder = NFKAudioTaggerHolder(net, labels: labels)
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
        let tags = holder.net.tag(samples, sampleRate: sampleRate, labels: holder.labels)
        return NFKInferenceResult(outputs: [NFKOutputClassifications: tags])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached {
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

/// Registration and weight loading for audio tagging.
@objc(NFKMLXAudioTagger)
public final class NFKMLXAudioTagger: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "audio-tagger-panns"

    static func makeNet(_ configuration: NFKMLXAudioTaggerConfiguration = .panns) -> NFKMLXAudioTaggerNet {
        let net = NFKMLXAudioTaggerNet(configuration)
        net.train(false)                                       // BatchNorm running statistics
        return net
    }

    /// Builds an audio-tagging backend directly from optional local weights — no registry required. A
    /// nil `weightsURL` builds random weights (`isReady` is true). `labels` names classes when
    /// available. Run inference off the render thread.
    @objc(backendWithWeightsURL:labels:error:)
    public static func backend(weightsURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXAudioTaggerBackend(net: net, identifier: modelName, labels: labels)
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:labels:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url, labels: labels)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:labels:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, labels: [String]?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0, labels: labels) },
                               completionHandler: completionHandler)
    }

    /// Registers audio tagging (`audio-tagger-panns`) with `NFKMLXModelRegistry`, delegating to
    /// `backend(weightsURL:labels:)`. The registered backend has no class labels; a caller that wants
    /// names builds through the factory.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL, labels: nil) }
    }

    /// The reference numbers its blocks from one (`conv_block1`), where the module holds an array.
    static func remapReferenceKey(_ key: String) -> String {
        let prefix = "conv_block"
        guard key.hasPrefix(prefix) else { return key }
        let rest = key.dropFirst(prefix.count)
        guard let separator = rest.firstIndex(of: "."), let number = Int(rest[..<separator]) else { return key }
        return "conv_block.\(number - 1)" + rest[separator...]
    }

    /// Loads a safetensors checkpoint, transposing 4-D convolution weights `[out, in, kH, kW]` → MLX's
    /// `[out, kH, kW, in]`. The checkpoint carries the mel filterbank the reference front end built, so
    /// it loads instead of being recomputed; its short-time transform is stored as convolution weights,
    /// which this port computes directly and ignores.
    static func loadWeights(into net: NFKMLXAudioTaggerNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        if let filterbank = raw["logmel_extractor.melW"] {
            net.frontEnd.load(filterbank: filterbank)
        }
        let mapped = raw.compactMap { key, value -> (String, MLXArray)? in
            if key.hasPrefix("spectrogram_extractor.") || key.hasPrefix("logmel_extractor.") { return nil }
            let name = remapReferenceKey(key)
            return checkpoint.needsConvTranspose && value.ndim == 4 ? (name, value.transposed(0, 2, 3, 1)) : (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
