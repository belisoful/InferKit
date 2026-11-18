//
//  NFKMLXGraniteSpeechBackend.swift
//  InferKitMLX
//
//  A speech-to-text backend for Granite Speech 3.3-2B. Raw audio becomes the stacked log-mel features
//  the Conformer encoder reads, the encoder and Q-former projector turn them into audio embeddings, and
//  the dense Granite decoder generates the transcription with the audio embeddings scattered into the
//  prompt at the audio-token positions. The prompt prefills the decoder's key-value cache and each new
//  token runs alone; a conversation's cache is kept between requests, its clip keyed by a digest of its
//  features.
//
//  The audio front-end matches Granite Speech's feature extractor: a torchaudio-style Mel spectrogram
//  (n_fft 512, 400-sample Hann window, hop 160, 80 HTK mel bands, no area normalization, power 2), a
//  `log10` with the reference's `max(x, max − 8) / 4 + 1` normalization, and frame-pair stacking that
//  concatenates each two adjacent 80-band frames into one 160-wide feature.
//

import Foundation
import InferKit
import MLX
import MLXRandom

/// The stacked log-mel feature extractor Granite Speech reads. Produces `[frames, 160]` from mono 16 kHz
/// samples.
public struct NFKMLXGraniteSpeechFeatures {
    public var sampleRate: Int
    public var fftSize: Int
    public var windowLength: Int
    public var hop: Int
    public var melCount: Int

    private let filterbank: MLXArray                                          // [bins, mels]
    private let window: MLXArray                                              // [fftSize]

    public init(sampleRate: Int = 16000, fftSize: Int = 512, windowLength: Int = 400,
                hop: Int = 160, melCount: Int = 80) {
        self.sampleRate = sampleRate
        self.fftSize = fftSize
        self.windowLength = windowLength
        self.hop = hop
        self.melCount = melCount
        filterbank = NFKMLXGraniteSpeechFeatures.htkMelFilters(
            sampleRate: sampleRate, bins: fftSize / 2 + 1, melCount: melCount)
        // A periodic Hann window of `windowLength`, centered in the `fftSize` frame (zeros on both sides).
        var hann = [Float](repeating: 0, count: fftSize)
        let offset = (fftSize - windowLength) / 2
        for i in 0 ..< windowLength {
            hann[offset + i] = 0.5 - 0.5 * cosf(2 * .pi * Float(i) / Float(windowLength))
        }
        window = MLXArray(hann, [fftSize])
    }

    /// The HTK mel filterbank (`2595·log10(1 + f/700)`), no area normalization — torchaudio's default.
    static func htkMelFilters(sampleRate: Int, bins: Int, melCount: Int) -> MLXArray {
        func hzToMel(_ f: Float) -> Float { 2595 * log10f(1 + f / 700) }
        func melToHz(_ m: Float) -> Float { 700 * (powf(10, m / 2595) - 1) }
        let nyquist = Float(sampleRate) / 2
        let melMin = hzToMel(0), melMax = hzToMel(nyquist)
        let points = (0 ..< melCount + 2).map { melToHz(melMin + (melMax - melMin) * Float($0) / Float(melCount + 1)) }
        let binHz = (0 ..< bins).map { Float($0) * Float(sampleRate) / Float((bins - 1) * 2) }
        var filters = [Float](repeating: 0, count: bins * melCount)
        for m in 0 ..< melCount {
            let lower = points[m], center = points[m + 1], upper = points[m + 2]
            for k in 0 ..< bins {
                let f = binHz[k]
                var weight: Float = 0
                if f >= lower && f <= center { weight = (f - lower) / (center - lower) }
                else if f > center && f <= upper { weight = (upper - f) / (upper - center) }
                filters[k * melCount + m] = max(0, weight)
            }
        }
        return MLXArray(filters, [bins, melCount])
    }

    /// `[frames, 160]` stacked log-mel features for mono `samples` at `sampleRate`.
    public func callAsFunction(_ samples: [Float]) -> MLXArray {
        // torchaudio MelSpectrogram uses center = true with reflect padding.
        let pad = fftSize / 2
        var padded = [Float](repeating: 0, count: samples.count + 2 * pad)
        for i in 0 ..< samples.count { padded[pad + i] = samples[i] }
        for i in 0 ..< pad {
            padded[pad - 1 - i] = samples[min(i + 1, samples.count - 1)]      // reflect (no edge repeat)
            padded[pad + samples.count + i] = samples[max(samples.count - 2 - i, 0)]
        }
        let frames = 1 + (padded.count - fftSize) / hop
        var frameData = [Float](repeating: 0, count: frames * fftSize)
        for frame in 0 ..< frames {
            for offset in 0 ..< fftSize { frameData[frame * fftSize + offset] = padded[frame * hop + offset] }
        }
        let windowed = frameData.withUnsafeBufferPointer { MLXArray($0, [frames, fftSize]) } * window
        let spectrum = rfft(windowed, axis: 1)
        let power = spectrum.realPart() * spectrum.realPart() + spectrum.imaginaryPart() * spectrum.imaginaryPart()
        var logmel = log10(maximum(power.matmul(filterbank), MLXArray(Float(1e-10))))   // [frames, mels]
        let maximumValue = logmel.max()
        logmel = maximum(logmel, maximumValue - 8) / 4 + 1
        var count = logmel.dim(0)
        if count % 2 == 1 { logmel = logmel[0 ..< count - 1]; count -= 1 }
        // Frame-pair stacking: each two adjacent 80-band frames form one 160-wide feature.
        return logmel.reshaped([count / 2, 2 * melCount])
    }
}

/// Holds the model, tokenizer, and feature extractor across the async job boundary. None are
/// `Sendable`, so the crossing is made explicit, as the other backends do.
final class NFKGraniteSpeechBackendHolder: @unchecked Sendable {
    let net: NFKMLXGraniteSpeechNet
    let tokenizer: NFKTokenizer
    let features: NFKMLXGraniteSpeechFeatures
    init(net: NFKMLXGraniteSpeechNet, tokenizer: NFKTokenizer, features: NFKMLXGraniteSpeechFeatures) {
        self.net = net
        self.tokenizer = tokenizer
        self.features = features
    }
}

/// A speech-to-text backend over Granite Speech 3.3-2B: audio under `NFKInputAudio` is transcribed
/// following an instruction under `NFKInputPrompt`. The prompt prefills the decoder's key-value cache and each
/// new token runs alone. A request continues the conversation its `NFKParameterConversationKey` names, or, naming
/// none, the kept conversation its prompt continues furthest; the clip's positions are kept under a digest of its
/// features, so another instruction about the same clip prefills only what it adds.
@objc(NFKMLXGraniteSpeechBackend)
public final class NFKMLXGraniteSpeechBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKGraniteSpeechBackendHolder
    private let identifier: String
    private let stopToken: Int
    private let audioToken: Int

    /// Granite's role markers, fixed in its tokenizer.
    private let startOfRole = 49152, endOfRole = 49153
    private let systemPrompt = "Knowledge Cutoff Date: April 2024.\nYou are Granite, developed by IBM. "
        + "You are a helpful AI assistant."
    private let defaultInstruction = "can you transcribe the speech into a written format?"

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

    init(net: NFKMLXGraniteSpeechNet, tokenizer: NFKTokenizer, identifier: String,
         releaseDirectoryURL: URL? = nil) {
        self.releaseDirectoryURL = releaseDirectoryURL
        holder = NFKGraniteSpeechBackendHolder(
            net: net, tokenizer: tokenizer, features: NFKMLXGraniteSpeechFeatures())
        self.identifier = identifier
        stopToken = tokenizer.eosTokenId
        audioToken = net.audioTokenId
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
        let audioToken = self.audioToken, stopToken = self.stopToken
        let startOfRole = self.startOfRole, endOfRole = self.endOfRole
        let systemPrompt = self.systemPrompt
        let instruction = request.prompt ?? defaultInstruction
        var maximumTokens = 200
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber { maximumTokens = value.intValue }
        Task.detached(priority: .userInitiated) { [self] in
            do {
                guard let (samples, rate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
                let matched = NFKMLXAudioRate.matched(samples, from: rate, to: holder.features.sampleRate)
                let mel = holder.features(matched)
                let features = mel.reshaped([1, mel.dim(0), mel.dim(1)])
                let audio = holder.net.audioEmbeddings(features)
                let audioCount = audio.dim(1)

                func encode(_ text: String) -> [Int] { holder.tokenizer.encode(text).map(\.intValue) }
                var tokens = [startOfRole] + encode("system") + [endOfRole] + encode(systemPrompt) + [stopToken]
                tokens += [startOfRole] + encode("user") + [endOfRole]
                tokens += Array(repeating: audioToken, count: audioCount) + encode(instruction) + [stopToken]
                tokens += [startOfRole] + encode("assistant") + [endOfRole]

                let (produced, cached) = try self.generate(tokens, audio: audio, digest: NFKMLXVisionImage.digest(of: mel),
                                                           request: request, limit: max(maximumTokens, 0), stop: stopToken)
                let text = holder.tokenizer.decode(produced.map { NSNumber(value: $0) })
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
        return try NFKMLXAudioConversation.run(decoder: decoder, tokens: tokens, audio: audio, audioToken: audioToken,
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

/// Building a speech-to-text backend from a Granite Speech release directory.
public extension NFKMLXGraniteSpeech {

    /// Builds a Granite Speech transcription backend from a released directory, reading its nested
    /// `config.json`, weights (with the audio adapter folded in), and `tokenizer.json`. Run inference off
    /// the render thread; the prompt prefills the decoder's cache and each new token runs alone.
    static func backend(directoryURL: URL,
                        precision: NFKMLXWeightPrecision = .checkpoint) throws -> any NFKInferenceBackend {
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Granite Speech release has no readable tokenizer")
        }
        let net = try net(fromDirectory: directoryURL)
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
        return NFKMLXGraniteSpeechBackend(net: net, tokenizer: tokenizer, identifier: graniteSpeechModelName,
                                          releaseDirectoryURL: directoryURL)
    }

    /// The Objective-C entry: builds a Granite Speech transcription backend from a release directory.
    @objc(graniteSpeechBackendWithDirectoryURL:error:)
    static func graniteSpeechBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL)
    }

    /// The registry name a Granite Speech backend reports.
    @objc static let graniteSpeechModelName = "granite-speech-3.3-2b"

    // The audio adapter is listed as required: without it the build succeeds and transcribes with the
    // text-only decoder. The vocabulary and merges are read from tokenizer.json, the files the parity
    // measurement read, so the separate vocab.json and merges.txt are left out.
    internal static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json",
                                         "adapter_config.json", "adapter_model.safetensors"]
    internal static let optionalFiles = ["added_tokens.json"]
    internal static let weightFiles = ["model.safetensors", "model.safetensors.index.json"]

    /// Downloads a Granite Speech release into the hub cache and builds the transcription backend.
    ///
    /// @discussion The download fetches `config.json`, the tokenizer files, the audio LoRA adapter, and
    /// the weight shards the index names. A file already in the cache is not fetched again. The call
    /// blocks on the network; run it off the render thread. The public release is
    /// `ibm-granite/granite-speech-3.3-2b`.
    @objc(graniteSpeechBackendWithRepo:revision:cacheDirectoryURL:error:)
    static func graniteSpeechBackend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``graniteSpeechBackend(repo:revision:cacheDirectoryURL:)``.
    @objc(graniteSpeechBackendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    static func graniteSpeechBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                     completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try graniteSpeechBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Registers `granite-speech-3.3-2b` with `NFKMLXModelRegistry`; the registry's URL is the release directory.
    @objc static func register() {
        NFKMLXModelRegistry.register(name: graniteSpeechModelName) { url in
            guard let url else {
                throw NFKMLXError.unsupportedConfiguration("granite-speech-3.3-2b builds from a release directory, not without weights")
            }
            return try graniteSpeechBackend(directoryURL: url)
        }
    }
}
