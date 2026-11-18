//
//  NFKMLXVoxtralBackend.swift
//  InferKitMLX
//
//  A speech-to-text backend for Voxtral-Mini. Raw audio becomes the Whisper 128-band log-mel features
//  the audio encoder reads (reusing `NFKMLXMel.logMel`), the encoder and projector turn them into
//  audio embeddings, and the Llama decoder generates the transcription with the audio embeddings
//  scattered into the transcription prompt. The prompt prefills the decoder's key-value cache and each new token
//  runs alone; a conversation's cache is kept between requests, its clip keyed by a digest of its features.
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

    /// One generation at a time: a run moves a conversation's cache through the shared decoder.
    private let generationLock = NSLock()
    /// The conversations' caches, read and written under `generationLock`.
    private let conversations = NFKMLXConversationKeeper<NFKMLXPromptCache>(byteBudget: 0)
    /// The release directory, which names the default spill directory, or nil for a model built without one.
    private let releaseDirectoryURL: URL?
    private let modelInfoCache = NFKMLXModelInfoCache()
    /// The model holds no adapters.
    private var adapterStrengths: [Float] { [] }
    /// The shared dense decoder.
    private var decoder: NFKMLXGraniteTextNet { holder.net.languageModel }

    /// The model's parameter count, weight bytes, precision, and quantization, with the bytes its release
    /// directory occupies on disk and the keys and values a position takes (`NFKModelInfo*` keys).
    /// Introduced in InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            var info = NFKMLXModelDescription.info(of: [holder.net], releaseDirectoryURL: releaseDirectoryURL)
            let c = decoder.config
            let elementBytes = decoder.model.embedTokens.weight.dtype.size
            info[NFKModelInfoKeyValueBytesPerToken] = 2 * c.layerCount * c.keyValueHeadCount * c.headDimensions * elementBytes
            return info
        }
    }

    /// The bytes a run's keys and values take over the positions a transcription prompt and its reply reach.
    private var runBytes: Int { NFKMLXConversationBudget.runBytes(info: modelInfo, positions: 4096) }

    init(net: NFKMLXVoxtralNet, tokenizer: NFKMLXTekkenTokenizer, identifier: String,
         releaseDirectoryURL: URL? = nil) {
        self.releaseDirectoryURL = releaseDirectoryURL
        holder = NFKVoxtralBackendHolder(net: net, tokenizer: tokenizer)
        self.identifier = identifier
        stopToken = tokenizer.eosTokenId
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio, NFKInputPrompt] }
    @objc public var supportedParameterKeys: Set<String> { [NFKParameterMaxTokens, NFKParameterConversationKey] }

    /// The result keys the backend writes. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputText, NFKOutputUsage] }

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
        Task.detached(priority: .userInitiated) { [self] in
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

                let (produced, cached) = try self.generate(tokens, audio: audio, digest: NFKMLXVisionImage.digest(of: mel),
                                                           request: request, limit: max(maximumTokens, 0), stop: stopToken)
                let text = tokenizer.decode(produced.map { NSNumber(value: $0) })
                job.finish(with: NFKInferenceResult(outputs: [
                    NFKOutputText: text,
                    NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: tokens.count, cachedTokens: cached,
                                                        outputTokens: produced.count, reasoningTokens: nil),
                ]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    /// Generation through the conversation the request names or its prompt continues, under the generation lock.
    private func generate(_ tokens: [Int], audio: MLXArray, digest: UInt64, request: NFKInferenceRequest, limit: Int,
                          stop: Int) throws -> (produced: [Int], cached: Int) {
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
        return try NFKMLXAudioConversation.run(decoder: decoder, tokens: tokens, audio: audio, audioToken: holder.net.config.audioTokenId,
                                               digest: digest, conversations: conversations, request: request,
                                               limit: limit, stop: stop) {
            NFKMLXConversationBudget.afforded(info: modelInfo, runBytes: runBytes)
        }
    }

    private static func audio(from request: NFKInferenceRequest) -> (samples: [Float], sampleRate: Int)? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.read(data)
        }
        if let data = value as? Data { return NFKMLXWaveFile.read(data) }
        return nil
    }

    // MARK: Conversation caches

    /// The bytes the conversations' caches may hold together.
    ///
    /// @discussion A request continues the conversation its `NFKParameterConversationKey` names, or, naming
    /// none, the kept conversation its prompt continues furthest, so a follow-up turn prefills only what it
    /// adds. When the caches outgrow the budget, the least recently used conversations are written to
    /// ``conversationSpillDirectoryURL`` and dropped, never the one a request just ran. Until it is set, the
    /// budget is what the memory planned beside the model affords after one run's cache at the context
    /// length, recomputed each run; setting it fixes it, and lowering it drops down to it at once. Introduced
    /// in InferKit 0.4.0.
    @objc public var conversationCacheByteBudget: Int {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            return conversations.states.byteBudget
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.fixBudget(newValue)
        }
    }

    /// How many conversations hold a cache. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheCount: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return conversations.states.count
    }

    /// The bytes the conversations' caches occupy. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheBytes: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return conversations.states.heldBytes
    }

    /// Drops one named conversation's cache, in memory and on disk. Introduced in InferKit 0.4.0.
    @objc(resetPromptCacheForConversation:)
    public func resetPromptCache(forConversation conversation: String) {
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.states.remove(conversation)
        conversations.spill?.remove(named: conversation)
        conversations.refresh()
    }

    /// Drops every conversation's cache in memory; spilled ones stay on disk. Introduced in InferKit 0.4.0.
    @objc public func resetPromptCache() {
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.states.removeAll()
        conversations.refresh()
    }

    /// Writes every conversation's cache to the spill directory and drops it from memory when no run is in
    /// progress, and returns the bytes they held; returns 0 at once while a run holds them.
    /// `NFKInferenceServer` calls it to relieve memory, and the reloadable backend before it unloads the model.
    /// Introduced in InferKit 0.4.0.
    @objc public func releaseIdleConversationCaches() -> Int {
        guard generationLock.try() else { return 0 }
        defer { generationLock.unlock() }
        return conversations.releaseAll(spills: adapterStrengths.isEmpty)
    }

    /// The directory conversations the memory budget lets go are written to, so a later request continues one
    /// without its prefill, after an unload or a restart too; nil writes none. A backend over a release
    /// directory defaults to a directory under the user's caches that only the same release and weights read.
    /// A model that holds adapters writes none. Introduced in InferKit 0.4.0.
    @objc public var conversationSpillDirectoryURL: URL? {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
            return conversations.spill?.directory
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.setSpillDirectory(newValue)
        }
    }

    /// The bytes the spilled conversations may take on disk; the least recently written go first past it.
    /// Defaults to 8 GiB. Introduced in InferKit 0.4.0.
    @objc public var conversationSpillByteBudget: Int {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
            return conversations.spill?.byteBudget ?? NFKMLXConversationSpill.defaultByteBudget
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
            conversations.spill?.byteBudget = newValue
        }
    }

    /// The conversation caches as of the last run or change: `conversation_caches`,
    /// `conversation_cache_bytes`, and `conversation_cache_byte_budget`. `NFKInferenceServer` serves it under
    /// the model's "status". It never waits on a run in progress. Introduced in InferKit 0.4.0.
    @objc public var backendStatus: [String: Any] { conversations.status }
}

/// Building a speech-to-text backend from a Voxtral-Mini release directory.
public extension NFKMLXVoxtral {

    /// Builds a Voxtral transcription backend from a released directory, reading its `config.json`,
    /// weights, and `tekken.json`. Run inference off the render thread; the prompt prefills the decoder's cache
    /// and each new token runs alone.
    static func backend(directoryURL: URL,
                        precision: NFKMLXWeightPrecision = .checkpoint) throws -> any NFKInferenceBackend {
        guard let tokenizer = NFKMLXTekkenTokenizer(tekkenURL: directoryURL.appendingPathComponent("tekken.json")) else {
            throw NFKMLXError.unsupportedConfiguration("the Voxtral release has no readable tekken.json")
        }
        let net = try net(fromDirectory: directoryURL)
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
        return NFKMLXVoxtralBackend(net: net, tokenizer: tokenizer, identifier: voxtralModelName,
                                    releaseDirectoryURL: directoryURL)
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
