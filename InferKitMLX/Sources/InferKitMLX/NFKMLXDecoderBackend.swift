//
//  NFKMLXDecoderBackend.swift
//  InferKitMLX
//
//  A text-generation backend over a decoder that carries no key-value cache: generation re-runs the
//  growing sequence each step. Qwen4-Exp and the Qwen3.5 hybrid decoder run through it. A message list
//  renders through the release's own chat template where the release ships one, and a raw prompt is
//  encoded as given.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

/// Holds the decoder's forward and the tokenizer across the async job boundary. `MLXArray` and the
/// tokenizer are not `Sendable`, so the crossing is made explicit, as the other language backends do.
final class NFKDecoderBackendHolder: @unchecked Sendable {
    let logits: (MLXArray) -> MLXArray
    let tokenizer: NFKTokenizer
    let draft: ((MLXArray) -> MLXArray)?
    let cached: NFKDecoderCachedGeneration?
    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKTokenizer, draft: ((MLXArray) -> MLXArray)? = nil,
         cached: NFKDecoderCachedGeneration? = nil) {
        self.logits = logits
        self.tokenizer = tokenizer
        self.draft = draft
        self.cached = cached
    }
}

/// Generation over a hybrid decoder that keeps its state between passes: `prefill` runs tokens through a
/// cache, pausing at each of a list of positions for a callback, and returns the logits `[vocabulary]` after
/// the last; `step` runs one more token, a scalar that may be unevaluated, and returns the logits after it.
/// `elementBytes` is the size of an activation, which the cache stores keys and values at.
struct NFKDecoderCachedGeneration {
    let configuration: NFKMLXHybridConfiguration
    let elementBytes: Int
    let prefill: (_ tokens: [Int], _ cache: NFKMLXHybridCache, _ stops: [Int],
                  _ atStop: (NFKMLXHybridCache) -> Void) -> MLXArray
    let step: (_ token: MLXArray, _ cache: NFKMLXHybridCache) -> MLXArray
}

/// What a run asks of the kept conversations: whether it keeps one at all, the conversation its request
/// names, and the position its generation prompt begins at, which the conversation snapshots before.
struct NFKDecoderConversationRun {
    let reuses: Bool
    let name: String?
    let turnStart: Int?
}

/// A prefill-only text-generation backend over a decoder whose forward maps token ids to logits.
///
/// @discussion Reads a prompt under `NFKInputPrompt` or a message list under `NFKInputMessages`, and
/// returns the reply under `NFKOutputText` with its token counts under `NFKOutputUsage`. A message
/// list renders through the release's chat template, which is what an instruct release was trained
/// on; a release without one has its messages joined as plain text. The chain a reasoning release
/// shows comes back under `NFKOutputReasoning`, split at the markers its template writes. Tools under
/// `NFKParameterTools` render into the template, and the calls the model writes come back under
/// `NFKOutputToolCalls`, read in the syntax the template shows (``NFKMLXToolCallFormat``). Generation stops at the
/// tokenizer's end-of-sequence marker or at any id the release's `generation_config.json` lists. A decoder
/// without a cache re-runs the whole sequence each step, so a reply's cost grows with its length; a smaller
/// release of the same vocabulary drafting for it proposes several tokens a pass (``hasDraftModel``). A
/// cached decoder (``isCached``) keeps each conversation's cache between requests, so a follow-up turn
/// prefills only what it adds (``conversationCacheByteBudget``). Run inference off the render thread.
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXDecoderBackend)
public final class NFKMLXDecoderBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKDecoderBackendHolder
    private let identifier: String
    private let chatTemplate: String?
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
    private let conversations = NFKMLXConversationKeeper<NFKMLXHybridConversation>(byteBudget: 0)
    /// The budget a caller set, or nil for the budget the memory left beside the model affords.
    private var setConversationByteBudget: Int?
    /// Where conversations the budget lets go are written, read and written under `generationLock`.
    private var spill: NFKMLXConversationSpill?

    /// How many decoder layers the model leaves in the release, read in their turn on every pass; 0
    /// where every layer is held. Introduced in InferKit 0.4.0.
    @objc public var streamedLayerCount: Int { layerStream?.layers.count ?? 0 }

    /// The bytes each pass reads from the release for the streamed layers; 0 where every layer is held.
    /// Introduced in InferKit 0.4.0.
    @objc public var streamedBytesPerPass: Int { layerStream?.bytesPerPass ?? 0 }

    /// Whether generation runs each new token alone through the decoder's cache, where the whole sequence would
    /// otherwise run again. Introduced in InferKit 0.4.0.
    @objc public var isCached: Bool { holder.cached != nil }

    /// The context window and key-value cache quantization a run takes where its request sets none.
    var cacheDefaults = NFKMLXGenerationOptions()
    /// The strengths the held adapters were loaded at, in load order; empty for a model that holds none.
    var adapterStrengths: [Float] = []

    /// Whether generation drafts with a smaller release of the same vocabulary
    /// (``NFKMLXHybridLanguage/backend(directoryURL:draftDirectoryURL:precision:residency:draftBits:)``).
    /// Introduced in InferKit 0.4.0.
    @objc public var hasDraftModel: Bool { holder.draft != nil }

    /// The rounds, proposals, and kept proposals of the last drafted generation. Introduced in InferKit 0.4.0.
    public private(set) var lastSpeculativeReport = NFKMLXSpeculativeReport()

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKTokenizer, identifier: String,
         chatTemplate: String?, stopTokens: Set<Int>, modules: [Module] = [], releaseDirectoryURL: URL? = nil,
         layerStream: NFKMLXLayerStream? = nil, draft: ((MLXArray) -> MLXArray)? = nil,
         cached: NFKDecoderCachedGeneration? = nil) {
        self.holder = NFKDecoderBackendHolder(logits: logits, tokenizer: tokenizer, draft: draft, cached: cached)
        self.identifier = identifier
        self.layerStream = layerStream
        self.describedModules = modules
        self.releaseDirectoryURL = releaseDirectoryURL
        releaseSampling = NFKMLXReleaseSampling(directoryURL: releaseDirectoryURL)
        self.chatTemplate = chatTemplate
        beginOfSequence = tokenizer.bosTokenId >= 0 ? tokenizer.bosTokenId : nil
        self.stopTokens = stopTokens
        super.init()
        if cached != nil, let releaseDirectoryURL {
            spill = NFKMLXConversationSpill(
                directory: NFKMLXConversationSpill.defaultDirectory(
                    release: releaseDirectoryURL, identifier: identifier,
                    weights: NFKMLXConversationSpill.weightsDescription(modelInfo)),
                byteBudget: NFKMLXConversationSpill.defaultByteBudget)
        }
    }

    /// The decoder, tokenizer, chat template, and stop ids a release directory supplies; `modules`
    /// are the networks `logits` runs, which ``modelInfo`` describes.
    static func release(directoryURL: URL, identifier: String, modules: [Module] = [],
                        layerStream: NFKMLXLayerStream? = nil, draft: ((MLXArray) -> MLXArray)? = nil,
                        cached: NFKDecoderCachedGeneration? = nil,
                        logits: @escaping (MLXArray) -> MLXArray) throws -> NFKMLXDecoderBackend {
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("\(directoryURL.lastPathComponent) has no readable tokenizer")
        }
        return NFKMLXDecoderBackend(logits: logits, tokenizer: tokenizer, identifier: identifier,
                                    chatTemplate: NFKMLXLanguage.chatTemplate(inDirectory: directoryURL),
                                    stopTokens: stopTokens(tokenizer: tokenizer, directoryURL: directoryURL),
                                    modules: modules, releaseDirectoryURL: directoryURL, layerStream: layerStream,
                                    draft: draft, cached: cached)
    }

    /// The tokenizer's end-of-sequence id and every id `generation_config.json` names as one; an
    /// instruct release ends a turn on its own marker, which only the generation config lists.
    static func stopTokens(tokenizer: NFKTokenizer, directoryURL: URL) -> Set<Int> {
        var stops = NFKMLXToolCalling.generationConfigStops(directoryURL: directoryURL)
        if tokenizer.eosTokenId >= 0 { stops.insert(tokenizer.eosTokenId) }
        return stops
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    /// The decoder's parameter count, weight bytes, precision, and quantization, with the bytes its
    /// release directory occupies on disk and the `model_type` and `max_position_embeddings` its
    /// `config.json` states (`NFKModelInfo*` keys). A paged decoder's routed experts count toward its
    /// parameters; its weight bytes are what memory holds, which leaves out experts read from the
    /// release. A decoder that streams its layers reports the `streamed` residency. Introduced in
    /// InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            var info = NFKMLXModelDescription.info(of: describedModules, releaseDirectoryURL: releaseDirectoryURL)
            if layerStream != nil {
                info[NFKModelInfoResidency] = NFKModelResidencyStreamed
            }
            return info
        }
    }

    /// The positions the release's configuration states, which bound a prompt and its reply.
    private var contextLength: Int? { modelInfo[NFKModelInfoContextLength] as? Int }

    /// The request parameters the backend reads, with `NFKMLXGenerationParameterKey.draftTokens` where
    /// a draft model proposes tokens and `NFKParameterAdapterStrengths` where the model holds adapters.
    /// Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        var keys: Set<String> = [NFKParameterTemperature, NFKParameterTopP, NFKParameterTopK, NFKParameterMaxTokens, NFKParameterSeed, NFKParameterTools,
                                 NFKMLXGenerationParameterKey.toolCallFormat]
        if templateReadsReasoningEffort { keys.insert(NFKParameterReasoningEffort) }
        if hasDraftModel {
            keys.insert(NFKMLXGenerationParameterKey.draftTokens)
        }
        if !adapterStrengths.isEmpty {
            keys.insert(NFKParameterAdapterStrengths)
        }
        if isCached {
            keys.formUnion([NFKMLXGenerationParameterKey.contextWindow, NFKMLXGenerationParameterKey.cacheQuantizationBits,
                            NFKMLXGenerationParameterKey.cacheQuantizationGroupSize,
                            NFKMLXGenerationParameterKey.cacheQuantizationPerChannelKeys,
                            NFKParameterConversationKey, NFKMLXGenerationParameterKey.reusesPromptCache])
        }
        return keys
    }

    /// Whether the chat template reads a thinking switch or level, which a reasoning effort binds:
    /// Qwen3.5 thinks only when `enable_thinking` is true, and Qwen4-Exp reads `reasoning_effort`.
    private var templateReadsReasoningEffort: Bool {
        guard let chatTemplate else { return false }
        return chatTemplate.contains("enable_thinking") || chatTemplate.contains("reasoning_effort")
    }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputPrompt, NFKInputMessages] }

    /// The result keys the backend writes: the reply, the tool calls and the assistant turn, the reasoning
    /// when the model shows any, and the usage. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> {
        NFKMLXToolCalling.outputKeys.union([NFKOutputReasoning, NFKOutputUsage, NFKOutputFinishReason])
    }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try run(request)
    }

    /// The shortest time between two partial results a job reports.
    var partialResultInterval: TimeInterval = 0.05

    /// Runs the request as a job that reports the reply as it generates.
    ///
    /// @discussion Each partial result holds the answer so far under `NFKOutputText` and a reasoning
    /// model's chain so far under `NFKOutputReasoning`, split at the markers the finished result is split
    /// at, and the progress is the share of the token limit produced. Cancelling the job ends generation
    /// at the next token. Introduced in InferKit 0.4.0.
    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        NFKMLXGenerationJob.submit(partialResultInterval: partialResultInterval) { [self] job in
            try run(request, job: job)
        }
    }

    private func run(_ request: NFKInferenceRequest, job: NFKMLXGenerationJob? = nil) throws -> NFKInferenceResult {
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
        var draftTokens = hasDraftModel ? NFKMLXGenerationOptions().draftTokens : 0
        if hasDraftModel, let value = request.parameter(forKey: NFKMLXGenerationParameterKey.draftTokens) as? NSNumber {
            draftTokens = value.intValue
        }

        var cacheOptions = cacheDefaults
        if isCached {
            NFKMLXLanguageBackend.applyMLXParameters(from: request, to: &cacheOptions)
        }

        let (promptTokens, format, turnStart) = try prompt(for: request, findsTurnStart: isCached)
        let reuses = (request.parameter(forKey: NFKMLXGenerationParameterKey.reusesPromptCache) as? NSNumber)?.boolValue ?? true
        let continued = NFKDecoderConversationRun(reuses: reuses, name: NFKMLXConversation.name(of: request),
                                                   turnStart: turnStart)
        let decode: ([Int]) -> String = { [holder] in holder.tokenizer.decode($0.map { NSNumber(value: $0) }) }
        let reasoningFormat = request.messages != nil ? chatTemplate.flatMap(NFKMLXReasoningFormat.detected(inChatTemplate:)) : nil
        let strengths = try NFKMLXAdapterRun.strengths(for: request, loaded: adapterStrengths)
        // The held adapters read the run's strengths as each pass builds its graph.
        let (produced, limit, cachedTokens) = try NFKMLXAdapterRun.$strengths.withValue(strengths) {
            try generate(promptTokens, sampling: sampling, maxTokens: maximumTokens, seed: seed,
                         draftTokens: draftTokens, cacheOptions: cacheOptions, strengths: strengths,
                         conversation: continued) { limit in
                job.map { job in
                    NFKMLXLanguageBackend.partialReporter(job, prompt: promptTokens, limit: limit, format: reasoningFormat,
                                                          callMarkers: NFKMLXToolCalling.templateTools(for: request) != nil ? format?.markers ?? [] : [],
                                                          decode: decode)
                }
            }
        }
        let (reasoning, answer, reasoningTokens) = NFKMLXUsage.split(decode(produced), format: reasoningFormat,
                                                                     produced: produced, decode: decode)
        var outputs = NFKMLXToolCalling.outputs(answer: answer, reasoning: reasoning, request: request,
                                                format: format, source: identifier)
        if !reasoning.isEmpty {
            outputs[NFKOutputReasoning] = reasoning
        }
        outputs[NFKOutputFinishReason] = NFKMLXContextRoom.finishReason(produced: produced.count, limit: limit, outputs: outputs)
        outputs[NFKOutputUsage] = NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: cachedTokens,
                                                      outputTokens: produced.count, reasoningTokens: reasoningTokens)
        return NFKInferenceResult(outputs: outputs)
    }

    /// The input tokens a run of the request costs: its prompt rendered through the chat template,
    /// tools included, and tokenized as a run tokenizes it. Runs no model. Introduced in InferKit 0.4.0.
    @objc(tokenCountForRequest:error:)
    public func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber {
        NSNumber(value: try prompt(for: request, findsTurnStart: false).tokens.count)
    }

    /// A run's prompt as token ids, the syntax its calls are read in, and, where `findsTurnStart` asks and
    /// the prompt renders a message list through the chat template, the position its generation prompt
    /// begins at: where the tokens part from the same messages rendered without one.
    private func prompt(for request: NFKInferenceRequest, findsTurnStart: Bool) throws
        -> (tokens: [Int], format: NFKMLXToolCallFormat?, turnStart: Int?) {
        var variables: [String: Any] = [:]
        if let chatTemplate, templateReadsReasoningEffort {
            variables = try NFKMLXLanguageBackend.templateVariables(for: request, template: .jinja(template: chatTemplate))
        } else {
            try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        }
        let format = toolCallFormat(for: request)
        let tokens = try tokens(for: request, toolCallFormat: format, variables: variables)
        guard findsTurnStart, let opening = try openingTokens(for: request, toolCallFormat: format, variables: variables) else {
            return (tokens, format, nil)
        }
        return (tokens, format, zip(tokens, opening).prefix { $0.0 == $0.1 }.count)
    }

    /// The syntax the model's calls are read in: the request's, or the one the chat template shows.
    private func toolCallFormat(for request: NFKInferenceRequest) -> NFKMLXToolCallFormat? {
        NFKMLXToolCalling.format(for: request, template: chatTemplate)
    }

    /// A message list rendered through the chat template without the generation prompt, as token ids; nil
    /// for a request without one or a release without a template.
    private func openingTokens(for request: NFKInferenceRequest, toolCallFormat: NFKMLXToolCallFormat?,
                               variables: [String: Any]) throws -> [Int]? {
        guard let chatTemplate, let original = request.messages else { return nil }
        let messages = NFKMLXToolCalling.templateMessages(original, format: toolCallFormat, source: identifier)
        let rendered = try NFKMLXChatTemplateRenderer.render(chatTemplate, messages: messages, addGenerationPrompt: false,
                                                             tools: NFKMLXToolCalling.templateTools(for: request),
                                                             variables: variables)
        return holder.tokenizer.encode(rendered).map(\.intValue)
    }

    /// The prompt as token ids. A message list renders through the chat template, which places its own
    /// markers. A request without one has its raw prompt follow the begin-of-sequence marker where the
    /// tokenizer defines one.
    private func tokens(for request: NFKInferenceRequest, toolCallFormat: NFKMLXToolCallFormat?,
                        variables: [String: Any]) throws -> [Int] {
        let prefix = beginOfSequence.map { [$0] } ?? []
        if request.messages == nil, let prompt = request.prompt {
            return prefix + holder.tokenizer.encode(prompt).map(\.intValue)
        }
        guard let original = request.messages else { throw NFKMLXError.unsupportedInput }
        let messages = NFKMLXToolCalling.templateMessages(original, format: toolCallFormat, source: identifier)
        guard let chatTemplate else {
            let joined = messages.compactMap { $0["content"] as? String }.joined(separator: "\n\n")
            return prefix + holder.tokenizer.encode(joined).map(\.intValue)
        }
        let rendered = try NFKMLXChatTemplateRenderer.render(chatTemplate, messages: messages,
                                                             tools: NFKMLXToolCalling.templateTools(for: request),
                                                             variables: variables)
        return holder.tokenizer.encode(rendered).map(\.intValue)
    }

    /// Generation. With a draft and `draftTokens` above 0, each round of proposals runs the whole sequence
    /// through the decoder (``NFKMLXPrefillDrafting``). Otherwise a cached decoder continues the conversation
    /// `conversation` names or the prompt's prefix finds (``cachedGenerate``), and a decoder without a cache
    /// runs the whole growing sequence each step. `onToken`, made for the run's token limit, sees each token,
    /// and answering false ends the run. Answers the tokens produced, the limit, and how many of the prompt's
    /// tokens a kept cache already held.
    private func generate(_ promptTokens: [Int], sampling: NFKMLXStepSampling, maxTokens: Int?, seed: UInt64?,
                          draftTokens: Int, cacheOptions: NFKMLXGenerationOptions, strengths: [Float]?,
                          conversation: NFKDecoderConversationRun,
                          onToken makeOnToken: (Int) -> ((Int) -> Bool)?) throws -> (tokens: [Int], limit: Int, cached: Int) {
        var tokens = promptTokens
        var produced = [Int]()
        let limit = try NFKMLXContextRoom.bounded(maxTokens, promptTokens: promptTokens.count,
                                                  contextLength: contextLength,
                                                  window: isCached ? cacheOptions.contextWindow : nil, source: identifier)
        let onToken = makeOnToken(limit)
        if let cached = holder.cached, holder.draft == nil || draftTokens <= 0 {
            guard limit > 0 else { return ([], limit, 0) }
            return try cachedGenerate(promptTokens, cached: cached, sampling: sampling, limit: limit, seed: seed,
                                      options: cacheOptions, strengths: strengths, conversation: conversation,
                                      onToken: onToken)
        }
        if let draft = holder.draft, draftTokens > 0 {
            var report = NFKMLXSpeculativeReport()
            defer { lastSpeculativeReport = report }
            return (try NFKMLXPrefillDrafting.generate(prompt: promptTokens, draft: draft, target: holder.logits,
                                                       verify: verifyStream, proposals: draftTokens, sampling: sampling,
                                                       limit: max(limit, 0), seed: seed, stops: stopTokens,
                                                       report: &report, onToken: onToken),
                    limit, 0)
        }
        for step in 0 ..< max(limit, 0) {
            let input = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
            let logits = holder.logits(input)[0, tokens.count - 1]
            let next = sampling.next(logits, seed: seed, step: step)
            try verifyStream()
            if stopTokens.contains(next) { break }
            produced.append(next)
            tokens.append(next)
            if let onToken, !onToken(next) { break }
        }
        return (produced, limit, 0)
    }

    /// Generation through the decoder's cache: the prompt continues a kept conversation from the furthest
    /// position it can, prefills the rest, and then runs each new token alone, under the window and cache
    /// quantization `options` set. The conversation keeps the positions the run wrote, and snapshots its
    /// recurrent state where ``NFKMLXHybridConversation`` places them. A run that fails discards it.
    private func cachedGenerate(_ promptTokens: [Int], cached: NFKDecoderCachedGeneration, sampling: NFKMLXStepSampling,
                                limit: Int, seed: UInt64?, options: NFKMLXGenerationOptions, strengths: [Float]?,
                                conversation: NFKDecoderConversationRun,
                                onToken: ((Int) -> Bool)?) throws -> (tokens: [Int], limit: Int, cached: Int) {
        generationLock.lock(); defer { generationLock.unlock() }
        let (name, state) = conversationState(for: conversation, prompt: promptTokens, cached: cached,
                                              options: options, strengths: strengths)
        var finished = false
        defer {
            if let name {
                if !finished {
                    conversations.states.remove(name)
                }
                conversations.states.byteBudget = setConversationByteBudget ?? affordedConversationBytes(cached, options: options)
                conversations.states.evict(keeping: name, releasing: spillConversation)
            }
            conversations.refresh()
        }
        let start = state.resume(for: promptTokens)
        let stops = name == nil ? [] : NFKMLXHybridConversation.snapshotPositions(
            start: start, end: promptTokens.count, turnStart: conversation.turnStart)
        let logits = cached.prefill(Array(promptTokens[start...]), state.cache, stops) { _ in state.snapshot() }
        state.record(Array(promptTokens[start...]))
        try verifyStream()
        var produced = [Int]()
        var fed = [Int]()
        // Each pass is built on the previous token before that token is read, so the graph of one pass is
        // built while the GPU runs the one before it. A run that stops leaves one pass unread.
        var token = sampling.token(logits, seed: seed, step: 0)
        asyncEval(token)
        for index in 0 ..< limit {
            var following: MLXArray?
            if index + 1 < limit {
                following = sampling.token(cached.step(token, state.cache), seed: seed, step: index + 1)
                asyncEval(following!)
            }
            let next = token.item(Int.self)
            if following != nil {
                fed.append(next)
            }
            try verifyStream()
            if stopTokens.contains(next) || NFKMLXErrorScope.hasFailed { break }
            produced.append(next)
            if let onToken, !onToken(next) { break }
            guard let following else { break }
            token = following
        }
        guard !NFKMLXErrorScope.hasFailed else { return (produced, limit, start) }
        state.record(fed)
        if name != nil {
            state.snapshot()
        }
        finished = true
        return (produced, limit, start)
    }

    /// The conversation a run continues and the name it is kept under, or a conversation kept under no name
    /// where the request turns reuse off. A named request continues its own conversation, from memory or
    /// from the spill directory. A request that names none continues the unnamed conversation, in memory
    /// or on disk, its prompt resumes furthest into, where the resumed prefix is at least as long as what
    /// the rollback discards, so one client's prompt never discards most of another's conversation;
    /// otherwise it starts one. A conversation built for another window, cache quantization, or adapter
    /// strengths starts over. The weights' precision and quantization are part of the spill directory's
    /// identity, so a spilled conversation always comes from these weights.
    private func conversationState(for run: NFKDecoderConversationRun, prompt: [Int], cached: NFKDecoderCachedGeneration,
                                   options: NFKMLXGenerationOptions,
                                   strengths: [Float]?) -> (name: String?, state: NFKMLXHybridConversation) {
        let make = {
            NFKMLXHybridConversation(configuration: cached.configuration, window: options.contextWindow,
                                     quantization: options.cacheQuantization, adapterStrengths: strengths)
        }
        let usable = { (state: NFKMLXHybridConversation) in
            state.matches(window: options.contextWindow, quantization: options.cacheQuantization, adapterStrengths: strengths)
        }
        let spilled = { [spill] (name: String) in
            spill?.take(name) { url, record in
                try NFKMLXHybridConversation.load(from: url, tokens: record.tokens, configuration: cached.configuration,
                                                  window: options.contextWindow, quantization: options.cacheQuantization,
                                                  adapterStrengths: strengths)
            }
        }
        guard run.reuses else { return (nil, make()) }
        if let name = run.name {
            let held = conversations.states.held.contains { $0.name == name }
            return (name, conversations.states.state(for: name, isUsable: usable) { (held ? nil : spilled(name)) ?? make() })
        }
        let inMemory = conversations.states.held
            .filter { $0.name.hasPrefix(Self.unnamedPrefix) && usable($0.state) }
            .map { (name: $0.name, kept: $0.state.resumablePosition(for: prompt), held: $0.state.tokens.count) }
            .filter { $0.kept > 0 && $0.kept >= $0.held - $0.kept }
            .max { $0.kept < $1.kept }
        let signature = NFKMLXHybridConversation.optionsSignature(window: options.contextWindow,
                                                                  quantization: options.cacheQuantization,
                                                                  adapterStrengths: strengths)
        let onDisk = spill?.resumable(for: prompt, prefix: Self.unnamedPrefix, options: signature)
        if let onDisk, onDisk.kept > inMemory?.kept ?? 0, let state = spilled(onDisk.name) {
            return (onDisk.name, conversations.states.state(for: onDisk.name, isUsable: usable) { state })
        }
        let name = inMemory?.name ?? Self.unnamedPrefix + UUID().uuidString
        return (name, conversations.states.state(for: name, isUsable: usable, make: make))
    }

    /// Writes a conversation the memory lets go to the spill directory. A conversation that cannot be written
    /// is let go all the same.
    /// A model that holds adapters writes none, since the directory cannot tell one set of adapters from
    /// another at the same strengths.
    private func spillConversation(_ name: String, _ state: NFKMLXHybridConversation) {
        guard adapterStrengths.isEmpty else { return }
        try? spill?.write(state, name: name)
    }

    private static let unnamedPrefix = NFKMLXConversation.unnamedPrefix

    /// The bytes the conversations may hold where no budget is set: the planned working set less the weights
    /// the model holds, the reserve beside them, and the cache one run takes at the window, or at the
    /// release's context length without one. The conversation a run just used is kept beyond it.
    private func affordedConversationBytes(_ cached: NFKDecoderCachedGeneration, options: NFKMLXGenerationOptions) -> Int {
        let budget = NFKMLXResidencyBudget.current()
        let weights = modelInfo[NFKModelInfoWeightBytes] as? Int ?? 0
        let run = NFKMLXHybridConversation.bytes(positions: options.contextWindow ?? contextLength ?? 0,
                                                 configuration: cached.configuration,
                                                 quantization: options.cacheQuantization, elementBytes: cached.elementBytes)
        return max(0, budget - weights - NFKMLXResidencyBudget.reserve(forBudget: budget) - run)
    }

    // MARK: Conversation caches

    /// The bytes the conversations' caches may hold together.
    ///
    /// @discussion A request continues the conversation its `NFKParameterConversationKey` names, or, naming
    /// none, the kept conversation its prompt shares the longest prefix with, so a follow-up turn prefills
    /// only what it adds. When the caches outgrow the budget, the least recently used conversations are
    /// dropped, never the one a request just ran. Until it is set, the budget is what the memory planned
    /// beside the model affords after one run's cache at its window, recomputed each run; setting it fixes
    /// it, and lowering it drops down to it at once. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheByteBudget: Int {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            return conversations.states.byteBudget
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            setConversationByteBudget = newValue
            conversations.states.byteBudget = newValue
            conversations.states.evict(releasing: spillConversation)
            conversations.refresh()
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
        spill?.remove(named: conversation)
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
    /// `NFKInferenceServer` calls it to relieve memory, and the reloadable backend before it unloads the
    /// model. Introduced in InferKit 0.4.0.
    @objc public func releaseIdleConversationCaches() -> Int {
        guard generationLock.try() else { return 0 }
        defer { generationLock.unlock() }
        let held = conversations.states.heldBytes
        for (name, state) in conversations.states.held {
            spillConversation(name, state)
        }
        conversations.states.removeAll()
        conversations.refresh()
        return held
    }

    /// The directory conversations the memory budget lets go are written to, so a later request continues
    /// one without its prefill, after an unload or a restart too; nil writes none. A release-directory
    /// backend defaults to a directory under the user's caches that only the same release and weights read.
    /// A model that holds adapters writes none. Introduced in InferKit 0.4.0.
    @objc public var conversationSpillDirectoryURL: URL? {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            return spill?.directory
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            spill = newValue.map { NFKMLXConversationSpill(directory: $0, byteBudget: spill?.byteBudget ?? NFKMLXConversationSpill.defaultByteBudget) }
        }
    }

    /// The bytes the spilled conversations may take on disk; the least recently written go first past it.
    /// Defaults to 8 GiB. Introduced in InferKit 0.4.0.
    @objc public var conversationSpillByteBudget: Int {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            return spill?.byteBudget ?? NFKMLXConversationSpill.defaultByteBudget
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            spill?.byteBudget = newValue
        }
    }

    /// The conversation caches as of the last run or change: `conversation_caches`,
    /// `conversation_cache_bytes`, and `conversation_cache_byte_budget`. `NFKInferenceServer` serves it under
    /// the model's "status". It never waits on a run in progress. Introduced in InferKit 0.4.0.
    @objc public var backendStatus: [String: Any] { conversations.status }

    /// Throws a streamed layer's failed read and clears it, so the next request reads the release again.
    private func verifyStream() throws {
        if let failure = layerStream?.readFailure {
            layerStream?.clearFailure()
            throw failure
        }
    }
}
