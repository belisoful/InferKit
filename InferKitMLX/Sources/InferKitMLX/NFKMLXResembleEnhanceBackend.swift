// The Resemble Enhance speech-restoration backend: a degraded clip in, a restored 44.1 kHz clip out. The
// factory loads the released `enhancer_stage2` checkpoint (`ds/G/default/mp_rank_00_model_states.pt`).

import Foundation
import InferKit
import MLX

private final class NFKResembleHolder: @unchecked Sendable {
    let net: NFKMLXResembleEnhance
    init(net: NFKMLXResembleEnhance) { self.net = net }
}

/// A Resemble Enhance backend: `NFKInputAudio` (a degraded clip, resampled to 44.1 kHz) → the restored
/// `NFKAudioAsset` under `NFKOutputAudio`. `lambd` is the denoiser strength, `tau` the prior temperature,
/// `nfe` the CFM function-evaluation budget (the `enhance()` defaults 0.5 / 0.5 / 32).
@objc(NFKMLXResembleEnhanceBackend)
public final class NFKMLXResembleEnhanceBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKResembleHolder
    private let identifier: String
    private let outputDirectory: URL
    private let lambd: Float
    private let tau: Float
    private let nfe: Int

    init(net: NFKMLXResembleEnhance, identifier: String, lambd: Float, tau: Float, nfe: Int,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKResembleHolder(net: net)
        self.identifier = identifier
        self.lambd = lambd
        self.tau = tau
        self.nfe = nfe
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { [] }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let input = sampleRate == 44100 ? samples : NFKMLXAudioRate.matched(samples, from: sampleRate, to: 44100)
        let wav = holder.net.enhance(input, lambd: lambd, tau: tau, nfe: nfe)   // [1, samples, 1]
        eval(wav)
        var stream = wav[0][0..., 0].asArray(Float.self)
        if stream.count > input.count { stream = Array(stream[0 ..< input.count]) }   // trim npad, match length
        let url = outputDirectory.appendingPathComponent("resemble-enhance-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: stream, sampleRate: 44100, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(stream.count) / 44100,
                                  sampleRate: 44100, channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
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

// MARK: - Backend factory

extension NFKMLXResembleEnhanceFactory {
    /// Builds a runnable backend from the `enhancer_stage2` model-states checkpoint.
    @objc(backendWithCheckpointURL:lambd:tau:nfe:error:)
    public static func backend(checkpointURL: URL, lambd: Float, tau: Float, nfe: Int) throws -> any NFKInferenceBackend {
        let net = makeNet()
        try loadWeights(into: net, from: checkpointURL)
        return NFKMLXResembleEnhanceBackend(net: net, identifier: modelName, lambd: lambd, tau: tau, nfe: nfe)
    }

    /// Builds the backend from an `enhancer_stage2` directory (holding `ds/G/default/mp_rank_00_model_states.pt`).
    /// A nil directory builds random-weight networks.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let directoryURL {
            let ckpt = directoryURL.appendingPathComponent("ds/G/default/mp_rank_00_model_states.pt")
            let path = FileManager.default.fileExists(atPath: ckpt.path) ? ckpt : directoryURL
            try loadWeights(into: net, from: path)
        }
        return NFKMLXResembleEnhanceBackend(net: net, identifier: modelName, lambd: 0.5, tau: 0.5, nfe: 32)
    }

    /// The released checkpoint's path in the repo. The build reads it alone; the hyperparameters are
    /// the release's `hparams.yaml` values, compiled in.
    static let checkpointFile = "enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt"

    /// Downloads the `enhancer_stage2` checkpoint (`ResembleAI/resemble-enhance`, public) and builds
    /// the backend.
    ///
    /// @discussion The download is the one DeepSpeed model-states file, about 713 MB. A file already
    /// in the cache is not fetched again. The call blocks on the network, so run it off the main and
    /// render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let checkpoint = try NFKMLXReleaseDownload.hub(cacheDirectoryURL: cacheDirectoryURL)
            .downloadRepo(repo, revision: revision, path: checkpointFile, sha256: nil)
        let stageDirectory = checkpointFile.split(separator: "/").dropFirst()
            .reduce(checkpoint) { folder, _ in folder.deletingLastPathComponent() }
        return try backend(directoryURL: stageDirectory)
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Registers the backend under `resemble-enhance` for the by-name registry.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { url in try backend(directoryURL: url) }
    }
}
