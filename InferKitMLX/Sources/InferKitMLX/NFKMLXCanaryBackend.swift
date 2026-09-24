//
//  NFKMLXCanaryBackend.swift
//  InferKitMLX
//
//  A speech-to-text backend for Canary-1B-v2. Raw audio becomes the FastConformer's normalized log-mel
//  features (the Parakeet front end, reused), the encoder turns them into frames, and the Transformer
//  decoder generates the transcription from a task prompt built out of control tokens. Greedy,
//  re-decoding the growing sequence each step (the clip is short).
//
//  The prompt is Canary's ASR turn: `<|startofcontext|><|startoftranscript|><|emo:undefined|>
//  <|src|><|tgt|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>`. A source language equal to the target
//  transcribes; a different target translates. Both default to `en`, read from `NFKInputPrompt` as a
//  language code or a `src>tgt` pair.
//

import Foundation
import InferKit
import MLX

/// Holds the model and tokenizer across the async job boundary.
final class NFKCanaryBackendHolder: @unchecked Sendable {
    let net: NFKMLXCanaryNet
    let tokenizer: NFKMLXCanaryTokenizer
    init(net: NFKMLXCanaryNet, tokenizer: NFKMLXCanaryTokenizer) {
        self.net = net
        self.tokenizer = tokenizer
    }
}

/// A speech-to-text backend over Canary-1B-v2: audio under `NFKInputAudio` is transcribed; the language
/// (or a `src>tgt` translation pair) comes from `NFKInputPrompt` (default `en`).
@objc(NFKMLXCanaryBackend)
public final class NFKMLXCanaryBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKCanaryBackendHolder
    private let identifier: String

    init(net: NFKMLXCanaryNet, tokenizer: NFKMLXCanaryTokenizer, identifier: String) {
        holder = NFKCanaryBackendHolder(net: net, tokenizer: tokenizer)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio, NFKInputPrompt] }
    @objc public var supportedParameterKeys: Set<String> { [] }

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
        let (source, target) = Self.languages(from: request)
        Task.detached(priority: .userInitiated) {
            do {
                guard let (samples, rate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
                let matched = NFKMLXAudioRate.matched(samples, from: rate, to: holder.net.configuration.sampleRate)
                let prompt = holder.tokenizer.transcriptionPrompt(source: source, target: target, punctuation: true)
                let ids = holder.net.recognize(matched, prompt: prompt)
                let text = holder.tokenizer.text(for: ids)
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputText: text]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    /// `NFKInputPrompt` is a language code (`en`, transcribe) or a `src>tgt` pair (`en>de`, translate).
    private static func languages(from request: NFKInferenceRequest) -> (source: String, target: String) {
        guard let prompt = request.input(forKey: NFKInputPrompt) as? String, !prompt.isEmpty else {
            return ("en", "en")
        }
        let parts = prompt.split(separator: ">", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
        if parts.count == 2 { return (parts[0], parts[1]) }
        return (parts[0], parts[0])
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

extension NFKMLXCanary {
    /// Builds the recognizer from a release directory holding `model_weights.ckpt` (the unpacked `.nemo`)
    /// and `tokenizer.json`. Blocking on the load; run off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        let tokenizer = try NFKMLXCanaryTokenizer(tokenizerURL: directoryURL.appendingPathComponent("tokenizer.json"))
        let net = NFKMLXCanaryNet(.v2)
        try loadWeights(into: net, from: directoryURL.appendingPathComponent("model_weights.ckpt"))
        return NFKMLXCanaryBackend(net: net, tokenizer: tokenizer, identifier: modelName)
    }

    // The repo's `model.safetensors` carries transformers names this loader does not read, so the
    // weights come from the `.nemo` archive the parity measurement read.
    static let requiredFiles = ["tokenizer.json"]
    static let optionalFiles = [String]()
    static let weightFiles = ["canary-1b-v2.nemo"]

    /// Downloads the Canary-1B-v2 release into the hub cache and builds the recognizer.
    ///
    /// @discussion The download fetches `tokenizer.json` and the release's `.nemo` archive, and the
    /// build reads `model_weights.ckpt` from inside the archive without unpacking it. A file already in
    /// the cache is not fetched again. The call blocks on the network; run it off the render thread. The
    /// public release is `nvidia/canary-1b-v2`.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let directory = try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles)
        let tokenizer = try NFKMLXCanaryTokenizer(tokenizerURL: directory.appendingPathComponent("tokenizer.json"))
        let archive = try NFKMLXNemoArchive(url: directory.appendingPathComponent(weightFiles[0]))
        let net = NFKMLXCanaryNet(.v2)
        try loadWeights(into: net, checkpoint: try archive.checkpoint(named: "model_weights.ckpt"))
        return NFKMLXCanaryBackend(net: net, tokenizer: tokenizer, identifier: modelName)
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) { try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL) }
    }

    /// Registers `canary-1b-v2` with `NFKMLXModelRegistry`; the registry's URL is the release directory.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { url in
            guard let url else {
                throw NFKMLXError.unsupportedConfiguration("canary-1b-v2 builds from a release directory, not without weights")
            }
            return try backend(directoryURL: url)
        }
    }

    /// A random-weights recognizer at the released geometry (or `configuration`), for shape checks.
    public static func backend(configuration: NFKMLXCanaryConfiguration = .v2) -> any NFKInferenceBackend {
        let net = NFKMLXCanaryNet(configuration)
        net.train(false)
        let empty = NFKMLXCanaryTokenizer.empty
        return NFKMLXCanaryBackend(net: net, tokenizer: empty, identifier: modelName)
    }
}
