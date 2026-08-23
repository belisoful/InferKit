//
//  NFKMLXVAD.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

// Voice activity detection marks where speech occurs in an audio clip. This is MarbleNet: a mel
// spectrogram front end feeding a stack of QuartzNet-style blocks, each a run of time-channel-separable
// convolutions (a depthwise convolution over time, a pointwise channel mix, batch normalization) with an
// optional projected residual. A two-class head scores every frame, and consecutive above-threshold
// frames merge into spans. The result is an NSArray<NFKAudioSegment *> under NFKOutputSegments.
//
// The reference (NVIDIA NeMo) keeps each block's convolutions in one flat list addressed positionally;
// this port names them, and `NFKMLXVAD.remapReferenceKey` maps between the two. Tensors flow as
// `[batch, frames, channels]`. Resampling an input that is not already at the model's rate is a
// validation-sweep item.

/// One MarbleNet block: `repeats` convolutions of the same width, and a residual projection when the
/// block carries one.
public struct NFKMLXVADBlockConfiguration: Sendable {
    public var filters: Int
    public var repeats: Int
    public var kernel: Int
    public var stride: Int
    public var dilation: Int
    public var residual: Bool
    public var separable: Bool

    public init(filters: Int, repeats: Int, kernel: Int, stride: Int = 1, dilation: Int = 1,
                residual: Bool = false, separable: Bool = true) {
        self.filters = filters
        self.repeats = repeats
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.residual = residual
        self.separable = separable
    }
}

/// VAD dimensions and thresholds. The front-end values are the reference preprocessor's: a 25 ms window
/// every 10 ms, 80 mel bands from a 512-point transform.
public struct NFKMLXVADConfiguration: Sendable {
    public var mels: Int
    public var sampleRate: Int
    public var windowSamples: Int
    public var hopSamples: Int
    public var fftSize: Int
    public var preemphasis: Float
    public var classes: Int
    public var threshold: Float
    public var blocks: [NFKMLXVADBlockConfiguration]

    public init(mels: Int = 80, sampleRate: Int = 16000, windowSamples: Int = 400, hopSamples: Int = 160,
                fftSize: Int = 512, preemphasis: Float = 0.97, classes: Int = 2, threshold: Float = 0.5,
                blocks: [NFKMLXVADBlockConfiguration]) {
        self.mels = mels
        self.sampleRate = sampleRate
        self.windowSamples = windowSamples
        self.hopSamples = hopSamples
        self.fftSize = fftSize
        self.preemphasis = preemphasis
        self.classes = classes
        self.threshold = threshold
        self.blocks = blocks
    }

    /// The released frame-level MarbleNet: six blocks, the first halving the frame rate to 20 ms.
    public static let marbleNet = NFKMLXVADConfiguration(blocks: [
        NFKMLXVADBlockConfiguration(filters: 128, repeats: 1, kernel: 11, stride: 2),
        NFKMLXVADBlockConfiguration(filters: 64, repeats: 2, kernel: 13, residual: true),
        NFKMLXVADBlockConfiguration(filters: 64, repeats: 2, kernel: 15, residual: true),
        NFKMLXVADBlockConfiguration(filters: 64, repeats: 2, kernel: 17, residual: true),
        NFKMLXVADBlockConfiguration(filters: 128, repeats: 1, kernel: 29, dilation: 2),
        NFKMLXVADBlockConfiguration(filters: 128, repeats: 1, kernel: 1, separable: false),
    ])

    public static let tiny = NFKMLXVADConfiguration(mels: 16, windowSamples: 100, hopSamples: 40,
                                                    fftSize: 128, blocks: [
        NFKMLXVADBlockConfiguration(filters: 16, repeats: 1, kernel: 3, stride: 2),
        NFKMLXVADBlockConfiguration(filters: 16, repeats: 2, kernel: 3, residual: true),
    ])

    /// The factor the block stack decimates the frame rate by.
    var totalStride: Int { blocks.reduce(1) { $0 * $1.stride } }
}

/// The mel spectrogram the encoder is trained on. The reference stores its analysis window and mel
/// filterbank in the checkpoint, so both load rather than being recomputed; the defaults reproduce them
/// for a randomly initialized network.
///
/// Held outside the module graph deliberately. MLX reflects every `MLXArray` property into
/// `parameters()`, so a constant declared on a `Module` makes a real checkpoint look short.
final class NFKVADFrontEnd {
    private let configuration: NFKMLXVADConfiguration
    /// `[fftSize]`. The reference stores a `windowSamples` window and lets the transform zero-pad it to
    /// the transform length, so that padding is folded in here.
    private(set) var window: MLXArray
    /// `[mels, bins]`.
    private(set) var filterbank: MLXArray

    init(_ configuration: NFKMLXVADConfiguration) {
        self.configuration = configuration
        let bins = configuration.fftSize / 2 + 1
        // A symmetric Hann window, the reference's `periodic=False`.
        let count = configuration.windowSamples
        let raw = (0 ..< count).map { 0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(count - 1)) }
        window = MLXArray(raw)
        filterbank = NFKMLXMel.melFilters(sampleRate: configuration.sampleRate, bins: bins,
                                          nMels: configuration.mels).transposed(1, 0)
        window = Self.centered(window, in: configuration.fftSize)
    }

    func load(window raw: MLXArray) {
        window = Self.centered(raw, in: configuration.fftSize)
    }

    func load(filterbank raw: MLXArray) {
        filterbank = raw.ndim == 3 ? raw[0] : raw
    }

    /// The transform centers a window shorter than its length, zero-padding both sides.
    private static func centered(_ raw: MLXArray, in size: Int) -> MLXArray {
        let total = size - raw.shape[0]
        guard total > 0 else { return raw }
        return MLX.padded(raw, widths: [IntOrPair((total / 2, total - total / 2))], mode: .constant)
    }

    /// `[1, frames, mels]` — preemphasis, a centered short-time transform, the power spectrum through
    /// the mel filterbank, and a natural log. The frame count is padded to a multiple of two, as the
    /// reference pads for efficiency.
    func logMel(_ samples: [Float]) -> MLXArray {
        let size = configuration.fftSize, hop = configuration.hopSamples
        var signal = samples
        if configuration.preemphasis != 0 {
            for index in stride(from: signal.count - 1, to: 0, by: -1) {
                signal[index] -= configuration.preemphasis * signal[index - 1]
            }
        }

        let pad = size / 2
        var padded: [Float]
        if signal.count >= pad + 1 {
            padded = (1 ... pad).reversed().map { signal[$0] } + signal
                   + (0 ..< pad).map { signal[signal.count - 2 - $0] }
        } else {
            padded = [Float](repeating: 0, count: pad) + signal + [Float](repeating: 0, count: pad)
        }
        if padded.count < size {
            padded += [Float](repeating: 0, count: size - padded.count)
        }
        let frames = 1 + (padded.count - size) / hop

        var frameData = [Float](repeating: 0, count: frames * size)
        for frame in 0 ..< frames {
            for offset in 0 ..< size {
                frameData[frame * size + offset] = padded[frame * hop + offset]
            }
        }
        let windowed = frameData.withUnsafeBufferPointer { MLXArray($0, [frames, size]) } * window
        let spectrum = rfft(windowed, axis: 1)
        let power = spectrum.realPart() * spectrum.realPart() + spectrum.imaginaryPart() * spectrum.imaginaryPart()

        // The reference's zero guard is added inside the log, not clamped outside it.
        var mel = log(power.matmul(filterbank.transposed(1, 0)) + MLXArray(powf(2, -24)))
        if frames % 2 != 0 {
            mel = MLX.padded(mel, widths: [IntOrPair((0, 1)), IntOrPair((0, 0))], mode: .constant)
        }
        return mel.reshaped([1, mel.shape[0], configuration.mels])
    }
}

/// One convolution of a MarbleNet block: depthwise over time and pointwise across channels when the
/// block is separable, a single convolution when it is not, then batch normalization.
final class NFKVADSeparableConv: Module {
    @ModuleInfo(key: "depthwise") var depthwise: Conv1d?
    @ModuleInfo(key: "pointwise") var pointwise: Conv1d
    @ModuleInfo(key: "norm") var norm: BatchNorm

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int, stride: Int, dilation: Int, separable: Bool) {
        // The reference pads to keep the frame count (before striding): `dilation * (kernel - 1) / 2`.
        let padding = dilation * (kernel - 1) / 2
        if separable {
            _depthwise.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: inChannels,
                                             kernelSize: kernel, stride: stride, padding: padding,
                                             dilation: dilation, groups: inChannels, bias: false)
            _pointwise.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels,
                                             kernelSize: 1, bias: false)
        } else {
            _pointwise.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels,
                                             kernelSize: kernel, stride: stride, padding: padding,
                                             dilation: dilation, bias: false)
        }
        // The reference normalizes with an epsilon of 1e-3, two orders above the framework default.
        // Its running variances are small enough that the default visibly rescales every block, and
        // eight of them compound into saturated logits.
        _norm.wrappedValue = BatchNorm(featureCount: outChannels, eps: 1e-3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(pointwise(depthwise.map { $0(x) } ?? x))
    }
}

/// A MarbleNet block: `repeats` convolutions, then the residual projection, then the activation. The
/// reference holds the last activation back until after the residual is added, so only the convolutions
/// between are activated.
final class NFKVADBlock: Module {
    @ModuleInfo(key: "convs") var convs: [NFKVADSeparableConv]
    @ModuleInfo(key: "residual") var residual: NFKVADSeparableConv?

    init(_ inChannels: Int, _ configuration: NFKMLXVADBlockConfiguration) {
        _convs.wrappedValue = (0 ..< configuration.repeats).map { index in
            NFKVADSeparableConv(index == 0 ? inChannels : configuration.filters, configuration.filters,
                                kernel: configuration.kernel,
                                stride: index == 0 ? configuration.stride : 1,
                                dilation: configuration.dilation, separable: configuration.separable)
        }
        _residual.wrappedValue = configuration.residual
            ? NFKVADSeparableConv(inChannels, configuration.filters, kernel: 1, stride: 1, dilation: 1,
                                  separable: false)
            : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for (index, conv) in convs.enumerated() {
            out = conv(out)
            if index < convs.count - 1 {
                out = relu(out)
            }
        }
        if let residual {
            out = out + residual(x)
        }
        return relu(out)
    }
}

/// The MarbleNet VAD network: a mel front end, the block stack, and a per-frame class head.
final class NFKMLXVADNet: Module {
    @ModuleInfo(key: "blocks") var blocks: [NFKVADBlock]
    @ModuleInfo(key: "head") var head: Linear

    let configuration: NFKMLXVADConfiguration
    let frontEnd: NFKVADFrontEnd

    init(_ c: NFKMLXVADConfiguration) {
        configuration = c
        frontEnd = NFKVADFrontEnd(c)
        var channels = c.mels
        _blocks.wrappedValue = c.blocks.map { block in
            defer { channels = block.filters }
            return NFKVADBlock(channels, block)
        }
        _head.wrappedValue = Linear(c.blocks.last?.filters ?? c.mels, c.classes)
    }

    /// Per-frame class logits `[1, frames, classes]` from a log-mel spectrogram `[1, frames, mels]`.
    func logits(_ mel: MLXArray) -> MLXArray {
        var x = mel
        for block in blocks {
            x = block(x)
        }
        return head(x)
    }

    /// Speech probability per frame for a mono waveform. The reference labels non-speech 0 and speech 1,
    /// so the second class is the one that matters.
    func speechProbabilities(_ samples: [Float], sampleRate: Int) -> [Float] {
        let matched = NFKMLXAudioRate.matched(samples, from: sampleRate, to: configuration.sampleRate)
        let probabilities = softmax(logits(frontEnd.logMel(matched)), axis: -1)[0..., 0..., 1]
        eval(probabilities)
        return probabilities.reshaped([-1]).asArray(Float.self)
    }

    /// Detects speech spans in a mono waveform, merging consecutive above-threshold frames.
    func detect(_ samples: [Float], sampleRate: Int) -> [NFKAudioSegment] {
        let probabilities = speechProbabilities(samples, sampleRate: sampleRate)
        // The frames are produced from the resampled clip, so they are timed by the model's rate.
        // Resampling preserves duration, so the seconds this yields are the caller's own.
        let frameSeconds = Double(configuration.hopSamples * configuration.totalStride) / Double(configuration.sampleRate)

        var segments: [NFKAudioSegment] = []
        var runStart: Int? = nil
        var runSum: Float = 0
        func close(_ end: Int) {
            guard let start = runStart else { return }
            let confidence = Double(runSum) / Double(end - start)
            segments.append(NFKAudioSegment(startSeconds: Double(start) * frameSeconds,
                                            endSeconds: Double(end) * frameSeconds,
                                            label: nil, confidence: min(max(confidence, 0), 1)))
            runStart = nil
            runSum = 0
        }
        for (frame, probability) in probabilities.enumerated() {
            if probability >= configuration.threshold {
                if runStart == nil { runStart = frame }
                runSum += probability
            } else {
                close(frame)
            }
        }
        close(probabilities.count)
        return segments
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKVADHolder: @unchecked Sendable {
    let net: NFKMLXVADNet
    init(_ net: NFKMLXVADNet) { self.net = net }
}

/// Voice activity detection as an InferKit backend. Reads `NFKInputAudio`; returns speech spans under
/// `NFKOutputSegments`.
@objc(NFKMLXVADBackend)
public final class NFKMLXVADBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKVADHolder
    private let identifier: String

    init(net: NFKMLXVADNet, identifier: String) {
        holder = NFKVADHolder(net)
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

/// Registration and weight loading for voice activity detection.
@objc(NFKMLXVAD)
public final class NFKMLXVAD: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "vad-marblenet"

    static func makeNet(_ configuration: NFKMLXVADConfiguration = .marbleNet) -> NFKMLXVADNet {
        let net = NFKMLXVADNet(configuration)
        net.train(false)                            // batch normalization uses the checkpoint's statistics
        return net
    }

    /// Builds a VAD backend directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXVADBackend(net: net, identifier: modelName)
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

    /// Registers VAD (`vad-marblenet`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// The reference is a NeMo `ConvASREncoder`. Each block keeps its convolutions in one flat `mconv`
    /// list whose indices step over the activation and dropout layers sitting between sub-blocks — five
    /// entries per separable convolution, four per plain one — holds its residual projection under
    /// `res.0`, and wraps every convolution in a masking layer, one `conv` level deeper. The classifier
    /// is a one-layer perceptron, `decoder.layer0`.
    static func remapReferenceKey(_ key: String, blocks: [NFKMLXVADBlockConfiguration]) -> String {
        if key.hasPrefix("decoder.layer0.") {
            return "head." + key.dropFirst("decoder.layer0.".count)
        }
        let parts = key.split(separator: ".").map(String.init)
        guard parts.count >= 6, parts[0] == "encoder", parts[1] == "encoder",
              let block = Int(parts[2]), block < blocks.count else { return key }

        if parts[3] == "res", let slot = Int(parts[5]) {
            let name = slot == 0 ? "pointwise" : "norm"
            return "blocks.\(block).residual.\(name).\(unmasked(parts[6...].joined(separator: ".")))"
        }
        guard parts[3] == "mconv", let index = Int(parts[4]) else { return key }
        let separable = blocks[block].separable
        let span = separable ? 5 : 4
        let name: String?
        switch (separable, index % span) {
        case (true, 0): name = "depthwise"
        case (true, 1): name = "pointwise"
        case (true, 2): name = "norm"
        case (false, 0): name = "pointwise"
        case (false, 1): name = "norm"
        default: name = nil
        }
        guard let name else { return key }
        return "blocks.\(block).convs.\(index / span).\(name).\(unmasked(parts[5...].joined(separator: ".")))"
    }

    private static func unmasked(_ tail: String) -> String {
        tail.hasPrefix("conv.") ? String(tail.dropFirst("conv.".count)) : tail
    }

    /// Loads a safetensors checkpoint, transposing Conv1d weights `[out, in, k]` → MLX's `[out, k, in]`.
    /// The reference stores its analysis window and mel filterbank alongside the weights; those are front
    /// end constants rather than parameters, so they load onto the front end directly.
    static func loadWeights(into net: NFKMLXVADNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        if let window = raw["preprocessor.featurizer.window"] {
            net.frontEnd.load(window: window)
        }
        if let filterbank = raw["preprocessor.featurizer.fb"] {
            net.frontEnd.load(filterbank: filterbank)
        }
        let mapped = raw.compactMap { key, value -> (String, MLXArray)? in
            if key.hasPrefix("preprocessor.") { return nil }
            let name = remapReferenceKey(key, blocks: net.configuration.blocks)
            return checkpoint.needsConvTranspose && value.ndim == 3 ? (name, value.transposed(0, 2, 1)) : (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
