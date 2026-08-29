//
//  NFKMLXDenoiser.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX

// Speech noise suppression with the time-domain Demucs U-Net (Défossez et al., "Real Time Speech
// Enhancement in the Waveform Domain"). The architecture is the same encoder/decoder-with-skips network
// as `NFKMLXDemucs`, configured for a single output channel: instead of splitting a music mix into
// stems, it maps a noisy waveform to a clean one.
//
// Because the network is identical to Demucs with `stems == 1`, `NFKMLXDenoiser` reuses `NFKMLXDemucsNet`
// and its weight loader; the released DNS models differ from the music model only in the configuration
// below. Tensors flow as `[batch, time, channels]`.

/// Builds the denoiser's Demucs configuration. The released DNS models are mono, causal (so the
/// bottleneck runs one direction and carries no projection), mix decoder channels with a 1×1
/// convolution, normalize by the input's standard deviation, and are trained at four times the input
/// rate with the half-sample-shift resampler.
private func denoiserConfiguration(baseChannels: Int, depth: Int) -> NFKMLXDemucsConfiguration {
    var configuration = NFKMLXDemucsConfiguration()
    configuration.audioChannels = 1
    configuration.stems = 1
    configuration.baseChannels = baseChannels
    configuration.depth = depth
    configuration.bidirectional = false
    configuration.context = 1
    configuration.resample = 4
    configuration.resampler = .halfSampleShift
    configuration.centersOutput = false
    configuration.normalize = true
    configuration.normalizationFloor = 1e-3
    return configuration
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKDenoiserHolder: @unchecked Sendable {
    let net: NFKMLXDemucsNet
    init(_ net: NFKMLXDemucsNet) { self.net = net }
}

/// Speech denoising as an InferKit backend. Reads `NFKInputAudio`; returns the cleaned clip as a single
/// `NFKAudioAsset` under `NFKOutputAudio`.
@objc(NFKMLXDenoiserBackend)
public final class NFKMLXDenoiserBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKDenoiserHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXDemucsNet, identifier: String, outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKDenoiserHolder(net)
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
        let cleaned = holder.net.separate(input)                         // [1, L, 1]
        eval(cleaned)
        let length = cleaned.shape[1]
        let stream = cleaned.reshaped([length]).asArray(Float.self)

        let url = outputDirectory.appendingPathComponent("denoiser-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: stream, sampleRate: sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(length) / Double(sampleRate),
                                  sampleRate: Double(sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
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

/// Registration and weight loading for the speech denoiser.
@objc(NFKMLXDenoiser)
public final class NFKMLXDenoiser: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "denoiser"

    /// Builds a denoiser net (Demucs U-Net with a single output channel). The reference
    /// (facebookresearch/denoiser) always stacks five encoder/decoder blocks; `dns48` and `dns64` differ
    /// only in the base channel count.
    static func makeNet(baseChannels: Int = 48, depth: Int = 5) -> NFKMLXDemucsNet {
        NFKMLXDemucsNet(denoiserConfiguration(baseChannels: baseChannels, depth: depth))
    }

    /// Builds a speech-denoising backend directly from optional local weights — no registry required. A
    /// nil `weightsURL` builds random weights (`isReady` is true). Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL {
            try NFKMLXDemucs.loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXDenoiserBackend(net: net, identifier: modelName)
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

    /// Registers the denoiser (`denoiser`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }
}
