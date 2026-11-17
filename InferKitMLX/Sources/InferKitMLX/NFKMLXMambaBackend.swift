//
//  NFKMLXMambaBackend.swift
//  InferKitMLX
//
//  A text-generation backend for the Mamba-2 decoder (Codestral-Mamba and the other pure-Mamba
//  releases). Generation runs through the decoder's recurrent cache: the prompt fills each layer's
//  convolution inputs and scan state, and each new token then runs alone; a conversation's cache is kept
//  between requests.
//
//  The tokenizer is the Mistral byte-fallback BPE (`NFKMLXMistralTokenizer`), read from the release's
//  own `tokenizer.json`. A raw prompt is encoded after the begin-of-sequence marker `<s>`; generation
//  stops at `</s>`.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

/// Holds the decoder's forward and the tokenizer across the async job boundary. `MLXArray` and the
/// tokenizer are not `Sendable`, so the crossing is made explicit, as the other language backends do.
final class NFKMambaBackendHolder: @unchecked Sendable {
    let logits: (MLXArray) -> MLXArray
    let tokenizer: NFKMLXMistralTokenizer
    /// Generation through the decoder's cache; nil for a decoder built without one.
    let cached: NFKRecurrentCachedGeneration?
    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKMLXMistralTokenizer,
         cached: NFKRecurrentCachedGeneration? = nil) {
        self.logits = logits
        self.tokenizer = tokenizer
        self.cached = cached
    }
}

/// A text-generation backend over a Mamba-2 decoder.
@objc(NFKMLXMambaBackend)
public final class NFKMLXMambaBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKMambaBackendHolder
    private let identifier: String
    private let beginOfSequence: Int?
    private let stopTokens: Set<Int>
    private let describedModules: [Module]
    private let releaseDirectoryURL: URL?
    /// The sampling the release's generation config sets: step 3 of ``NFKMLXGenerationOptions``' sampling order.
    private let releaseSampling: NFKMLXReleaseSampling?
    private let modelInfoCache = NFKMLXModelInfoCache()
    private let layerStream: NFKMLXLayerStream?
    /// One cached generation at a time: a run moves a conversation's cache through the shared decoder.
    private let generationLock = NSLock()
    /// The conversations' caches, read and written under `generationLock`.
    private let conversations = NFKMLXConversationKeeper<NFKMLXRecurrentConversation>(byteBudget: 0)

    /// How many decoder blocks the model leaves in the release, read in their turn on every pass; 0
    /// where every block is held. Introduced in InferKit 0.4.0.
    @objc public var streamedLayerCount: Int { layerStream?.layers.count ?? 0 }

    /// The bytes each pass reads from the release for the streamed blocks; 0 where every block is held.
    /// Introduced in InferKit 0.4.0.
    @objc public var streamedBytesPerPass: Int { layerStream?.bytesPerPass ?? 0 }

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKMLXMistralTokenizer, identifier: String,
         modules: [Module] = [], releaseDirectoryURL: URL? = nil, layerStream: NFKMLXLayerStream? = nil,
         cached: NFKRecurrentCachedGeneration? = nil) {
        self.holder = NFKMambaBackendHolder(logits: logits, tokenizer: tokenizer, cached: cached)
        self.identifier = identifier
        self.layerStream = layerStream
        self.describedModules = modules
        self.releaseDirectoryURL = releaseDirectoryURL
        releaseSampling = NFKMLXReleaseSampling(directoryURL: releaseDirectoryURL)
        beginOfSequence = tokenizer.id(forToken: "<s>")
        stopTokens = Set([tokenizer.id(forToken: "</s>")].compactMap { $0 })
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    /// The positions the release's configuration states, which bound a prompt and its reply.
    private var contextLength: Int? { modelInfo[NFKModelInfoContextLength] as? Int }

    /// The decoder's parameter count, weight bytes, precision, and quantization, with the bytes its
    /// release directory occupies on disk, the `model_type` and `max_position_embeddings` its
    /// `config.json` states, and the residency, `streamed` for a decoder that streams its blocks
    /// (`NFKModelInfo*` keys). Introduced in InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            var info = NFKMLXModelDescription.info(of: describedModules, releaseDirectoryURL: releaseDirectoryURL)
            info[NFKModelInfoResidency] = layerStream != nil ? NFKModelResidencyStreamed : NFKModelResidencyResident
            return info
        }
    }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        var keys: Set<String> = [NFKParameterTemperature, NFKParameterTopP, NFKParameterTopK, NFKParameterMaxTokens,
                                 NFKParameterSeed]
        if holder.cached != nil {
            keys.insert(NFKParameterConversationKey)
        }
        return adapterStrengths.isEmpty ? keys : keys.union([NFKParameterAdapterStrengths])
    }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputPrompt, NFKInputMessages] }

    /// The result keys the backend writes. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputText, NFKOutputUsage, NFKOutputFinishReason] }

    /// The strengths the held adapters were loaded at, in load order; empty for a model that holds none.
    var adapterStrengths: [Float] = []

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        // The held adapters read the run's strengths as each pass builds its graph.
        try NFKMLXAdapterRun.$strengths.withValue(try NFKMLXAdapterRun.strengths(for: request, loaded: adapterStrengths)) {
            try run(request)
        }
    }

    private func run(_ request: NFKInferenceRequest) throws -> NFKInferenceResult {
        var sampling = NFKMLXStepSampling(release: releaseSampling)
        sampling.read(request)
        var maximumTokens: Int?
        var seed: UInt64?
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber {
            maximumTokens = value.intValue
        }
        if let value = request.parameter(forKey: NFKParameterSeed) as? NSNumber {
            seed = value.uint64Value
        }

        let promptTokens = try prompt(for: request)
        let (produced, limit, cachedTokens) = try generate(promptTokens, sampling: sampling, maxTokens: maximumTokens,
                                                           seed: seed, request: request)
        return NFKInferenceResult(outputs: [
            NFKOutputText: holder.tokenizer.decode(produced),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: cachedTokens,
                                                outputTokens: produced.count, reasoningTokens: nil),
            NFKOutputFinishReason: NFKMLXContextRoom.finishReason(produced: produced.count, limit: limit),
        ])
    }

    /// The prompt as token ids: the begin-of-sequence marker, then the encoded text. Codestral-Mamba is
    /// a base completion model with no chat template, so a message list is joined into plain text.
    private func tokens(for request: NFKInferenceRequest) throws -> [Int] {
        var ids = [Int]()
        if let bos = beginOfSequence { ids.append(bos) }
        if request.messages == nil, let prompt = request.prompt {
            ids += holder.tokenizer.encode(prompt)
            return ids
        }
        guard let messages = request.messages else { throw NFKMLXError.unsupportedInput }
        let joined = messages.compactMap { $0["content"] as? String }.joined(separator: "\n\n")
        ids += holder.tokenizer.encode(joined)
        return ids
    }

    /// The input tokens a run of the request costs, tokenized as a run tokenizes it. Runs no model.
    /// Introduced in InferKit 0.4.0.
    @objc(tokenCountForRequest:error:)
    public func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber {
        NSNumber(value: try prompt(for: request).count)
    }

    /// A run's prompt as token ids. A reasoning effort is refused, since the model has no way to apply it.
    private func prompt(for request: NFKInferenceRequest) throws -> [Int] {
        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        return try tokens(for: request)
    }

    /// Generation. A cached decoder continues the conversation the request names or its prompt continues and
    /// runs each new token alone (``NFKMLXRecurrentGeneration``); a decoder without a cache runs the whole
    /// growing sequence each step. Answers the tokens produced, the limit, and how many of the prompt's tokens a
    /// kept cache already held.
    private func generate(_ promptTokens: [Int], sampling: NFKMLXStepSampling, maxTokens: Int?, seed: UInt64?,
                          request: NFKInferenceRequest) throws -> (tokens: [Int], limit: Int, cached: Int) {
        var tokens = promptTokens
        var produced = [Int]()
        let limit = try NFKMLXContextRoom.bounded(maxTokens, promptTokens: promptTokens.count,
                                                  contextLength: contextLength, source: identifier)
        if let cached = holder.cached {
            guard limit > 0 else { return ([], limit, 0) }
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
            let (produced, start) = try NFKMLXRecurrentGeneration.run(
                promptTokens, turnStart: nil, cached: cached, conversations: conversations, request: request,
                sampling: sampling, limit: limit, seed: seed, stops: stopTokens,
                afforded: { NFKMLXConversationBudget.afforded(info: modelInfo, runBytes: conversationBytes(cached)) },
                spills: adapterStrengths.isEmpty, verify: verifyStream)
            return (produced, limit, start)
        }
        for step in 0 ..< max(limit, 0) {
            let input = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
            let logits = holder.logits(input)[0, tokens.count - 1]
            let next = sampling.next(logits, seed: seed, step: step)
            try verifyStream()
            if stopTokens.contains(next) { break }
            produced.append(next)
            tokens.append(next)
        }
        return (produced, limit, 0)
    }

    /// The bytes one run's cache takes: the Mamba layers' state for the live cache and the kept snapshots.
    private func conversationBytes(_ cached: NFKRecurrentCachedGeneration) -> Int {
        guard let net = describedModules.first as? NFKMLXMamba2Net else { return 0 }
        return NFKMLXRecurrentGeneration.bytes(positions: 0, attentionLayers: 0, keyValueHeads: 0, headDimensions: 0,
                                               mamba: net.config, mambaLayers: cached.mambaLayers.count,
                                               elementBytes: net.backbone.embeddings.weight.dtype.size)
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


    /// Throws a streamed block's failed read and clears it, so the next request reads the release again.
    private func verifyStream() throws {
        if let failure = layerStream?.readFailure {
            layerStream?.clearFailure()
            throw failure
        }
    }
}

/// Building a text-generation backend from a Mamba-2 release directory.
public extension NFKMLXMamba {
    /// Generation through the decoder's recurrent cache, every layer a Mamba layer.
    internal static func cachedGeneration(_ net: NFKMLXMamba2Net) -> NFKRecurrentCachedGeneration {
        NFKRecurrentCachedGeneration(mambaLayers: Array(repeating: true, count: net.config.layerCount), attentionLayers: [],
                                     attentionHeads: 0) { tokens, cache in net.lastLogits(tokens, cache: cache) }
    }

    /// Builds a text-generation backend from a released Mamba-2 directory, reading its `config.json`,
    /// weights, and `tokenizer.json`. Run inference off the render thread; the prompt fills the decoder's
    /// recurrent cache, and each new token runs alone.
    static func backend(directoryURL: URL,
                        precision: NFKMLXWeightPrecision = .checkpoint) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: .automatic)
    }

    /// ``backend(directoryURL:precision:)`` holding the decoder as `residency` says. Under
    /// ``NFKMLXResidency/automatic`` a release that fits the working set loads whole, and one that does
    /// not, such as Codestral Mamba at float32 on a 32 GB machine, holds the first blocks that fit and reads
    /// the others from the release on every pass (``NFKMLXMambaBackend/streamedLayerCount``);
    /// ``NFKMLXResidency/streamed`` streams wherever the decoder does not fit whole. A failed read fails
    /// the request, and the next request reads every block again. Introduced in InferKit 0.4.0.
    static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision,
                        residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: residency,
                    budget: NFKMLXResidencyBudget.current())
    }

    internal static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                                 budget: Int, adapters: [(url: URL, strength: Float)] = []) throws -> NFKMLXMambaBackend {
        guard let tokenizer = NFKMLXMistralTokenizer(directoryURL: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Mamba release has no readable tokenizer.json")
        }
        let net = makeNet(try configuration(fromDirectory: directoryURL))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision, residency: residency, budget: budget)
        let backend = NFKMLXMambaBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: modelName,
                                         modules: [net], releaseDirectoryURL: directoryURL, layerStream: net.layerStream,
                                         cached: cachedGeneration(net))
        if !adapters.isEmpty {
            guard net.layerStream == nil else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(modelName) holds adapters beside layers held in memory; load it with a residency that holds every layer")
            }
            backend.adapterStrengths = try NFKMLXDecoderAdapters.hold(adapters, in: net)
        }
        return backend
    }

    /// The Objective-C entry: builds a Mamba text-generation backend from a release directory.
    @objc(mambaBackendWithDirectoryURL:error:)
    static func mambaBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL)
    }

    /// The Objective-C entry for ``backend(directoryURL:precision:residency:)``. Introduced in InferKit 0.4.0.
    @objc(mambaBackendWithDirectoryURL:precision:residency:error:)
    static func mambaBackend(directoryURL: URL, precision: NFKMLXWeightPrecision,
                             residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: residency)
    }

    internal static let requiredFiles = ["config.json", "tokenizer.json"]
    internal static let optionalFiles = [String]()
    internal static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// Downloads a Mamba-2 release and builds its text-generation backend.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer.json`, and every shard the
    /// release's `model.safetensors.index.json` names into the hub cache under `cacheDirectoryURL`,
    /// or the default cache when nil; the repo's duplicate `consolidated.safetensors` is not fetched.
    /// A cached file is not fetched again. The call blocks on the network; call it off the render
    /// thread. It serves `mistralai/Mamba-Codestral-7B-v0.1`, which is not gated. Introduced in
    /// InferKit 0.4.0.
    @objc(mambaBackendWithRepo:revision:cacheDirectoryURL:error:)
    static func mambaBackend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``mambaBackend(repo:revision:cacheDirectoryURL:)``. Introduced in
    /// InferKit 0.4.0.
    @objc(mambaBackendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    static func mambaBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                             completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try mambaBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Registers `codestral-mamba` with `NFKMLXModelRegistry`; the registry's URL is the release directory.
    @objc static func register() {
        NFKMLXModelRegistry.register(name: mambaModelName) { url in
            guard let url else {
                throw NFKMLXError.unsupportedConfiguration("codestral-mamba builds from a release directory, not without weights")
            }
            return try mambaBackend(directoryURL: url)
        }
    }

    /// The registry name a Mamba backend reports.
    @objc static let mambaModelName = "codestral-mamba"

    private static var modelName: String { mambaModelName }
}
