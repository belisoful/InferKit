//
//  NFKMLXVoxtralBackend.swift
//  InferKitMLX
//
//  A speech-to-text backend for Voxtral-Mini. Raw audio becomes the Whisper 128-band log-mel features
//  the audio encoder reads (reusing `NFKMLXMel.logMel`), the encoder and projector turn them into
//  audio embeddings, and the Llama decoder generates the transcription with the audio embeddings
//  scattered into the transcription prompt. Prefill-only, re-decoding the growing sequence each step.
//
//  The prompt is Voxtral's transcription request, read from mistral-common:
//  `<s> [INST] [BEGIN_AUDIO] [AUDIO]×N [/INST] lang:<code> [TRANSCRIBE]`, where the audio placeholder is
//  repeated once per projected audio embedding. The tokenizer is the release's tekken.
//

import Foundation
import InferKit
import MLX

/// Holds the model and tokenizer across the async job boundary.
final class NFKVoxtralBackendHolder: @unchecked Sendable {
    let net: NFKMLXVoxtralNet
    let tokenizer: NFKMLXTekkenTokenizer
    init(net: NFKMLXVoxtralNet, tokenizer: NFKMLXTekkenTokenizer) {
        self.net = net
        self.tokenizer = tokenizer
    }
}

/// A speech-to-text backend over Voxtral-Mini: audio under `NFKInputAudio` is transcribed; the language
/// code comes from `NFKInputPrompt` (default `en`).
@objc(NFKMLXVoxtralBackend)
public final class NFKMLXVoxtralBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKVoxtralBackendHolder
    private let identifier: String
    private let stopToken: Int

    /// A 30-second window at 16 kHz, the fixed clip length the Whisper encoder reads.
    private let windowSamples = 30 * 16000

    init(net: NFKMLXVoxtralNet, tokenizer: NFKMLXTekkenTokenizer, identifier: String) {
        holder = NFKVoxtralBackendHolder(net: net, tokenizer: tokenizer)
        self.identifier = identifier
        stopToken = tokenizer.eosTokenId
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio, NFKInputPrompt] }
    @objc public var supportedParameterKeys: Set<String> { [NFKParameterMaxTokens] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result { return result }
        if let error = job.error { throw error }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let holder = self.holder
        let stopToken = self.stopToken, windowSamples = self.windowSamples
        let language = (request.prompt?.isEmpty == false) ? request.prompt! : "en"
        var maximumTokens = 200
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber { maximumTokens = value.intValue }
        Task.detached(priority: .userInitiated) {
            do {
                guard let (samples, rate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
                var matched = NFKMLXAudioRate.matched(samples, from: rate, to: 16000)
                if matched.count < windowSamples { matched += [Float](repeating: 0, count: windowSamples - matched.count) }
                else if matched.count > windowSamples { matched = Array(matched[0 ..< windowSamples]) }
                let mel = NFKMLXMel.logMel(matched, sampleRate: 16000, nMels: holder.net.config.audioMels)
                let audio = holder.net.audioEmbeddings(mel)
                let audioCount = audio.dim(0)

                let tokenizer = holder.tokenizer
                func special(_ name: String) -> Int { tokenizer.specialTokenId(name) ?? -1 }
                var tokens = [tokenizer.bosTokenId, special("[INST]"), special("[BEGIN_AUDIO]")]
                tokens += Array(repeating: holder.net.config.audioTokenId, count: audioCount)
                tokens += [special("[/INST]")]
                tokens += tokenizer.encode("lang:\(language)").map(\.intValue)
                tokens += [special("[TRANSCRIBE]")]

                var produced = [Int]()
                for _ in 0 ..< max(maximumTokens, 0) {
                    let input = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
                    let next = holder.net.logits(tokens: input, audioEmbeddings: audio)[0, tokens.count - 1]
                        .argMax(axis: -1).item(Int.self)
                    if next == stopToken { break }
                    produced.append(next)
                    tokens.append(next)
                }
                let text = tokenizer.decode(produced.map { NSNumber(value: $0) })
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputText: text]))
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

/// Building a speech-to-text backend from a Voxtral-Mini release directory.
public extension NFKMLXVoxtral {

    /// Builds a Voxtral transcription backend from a released directory, reading its `config.json`,
    /// weights, and `tekken.json`. Run inference off the render thread; the decoder re-runs the growing
    /// sequence each step.
    static func backend(directoryURL: URL,
                        precision: NFKMLXWeightPrecision = .checkpoint) throws -> any NFKInferenceBackend {
        guard let tokenizer = NFKMLXTekkenTokenizer(tekkenURL: directoryURL.appendingPathComponent("tekken.json")) else {
            throw NFKMLXError.unsupportedConfiguration("the Voxtral release has no readable tekken.json")
        }
        let net = try net(fromDirectory: directoryURL)
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
        return NFKMLXVoxtralBackend(net: net, tokenizer: tokenizer, identifier: voxtralModelName)
    }

    /// The Objective-C entry: builds a Voxtral transcription backend from a release directory.
    @objc(voxtralBackendWithDirectoryURL:error:)
    static func voxtralBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL)
    }

    /// The registry name a Voxtral backend reports.
    @objc static let voxtralModelName = "voxtral-mini-3b"

    internal static let requiredFiles = ["config.json", "tekken.json"]
    internal static let optionalFiles = [String]()
    // The transformers-named weights; the release's `consolidated.safetensors` uses Mistral's own names.
    internal static let weightFiles = ["model.safetensors", "model.safetensors.index.json"]

    /// Downloads a Voxtral release into the hub cache and builds the transcription backend.
    ///
    /// @discussion The download fetches `config.json`, `tekken.json`, and the weight shards the index
    /// names. A file already in the cache is not fetched again. The call blocks on the network; run it
    /// off the render thread. The public release is `mistralai/Voxtral-Mini-3B-2507`.
    @objc(voxtralBackendWithRepo:revision:cacheDirectoryURL:error:)
    static func voxtralBackend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``voxtralBackend(repo:revision:cacheDirectoryURL:)``.
    @objc(voxtralBackendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    static func voxtralBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try voxtralBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Registers `voxtral-mini-3b` with `NFKMLXModelRegistry`; the registry's URL is the release directory.
    @objc static func register() {
        NFKMLXModelRegistry.register(name: voxtralModelName) { url in
            guard let url else {
                throw NFKMLXError.unsupportedConfiguration("voxtral-mini-3b builds from a release directory, not without weights")
            }
            return try voxtralBackend(directoryURL: url)
        }
    }
}
