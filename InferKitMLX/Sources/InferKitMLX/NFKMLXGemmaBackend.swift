//
//  NFKMLXGemmaBackend.swift
//  InferKitMLX
//
//  A text-generation backend for the Gemma 4 decoders (E2B/E4B, the 26B-A4B mixture, and the 12B
//  unified), so a consumer runs them through the InferKit contract like any other backend. The Gemma
//  decoders run prefill-only — they carry no key-value cache — so generation re-runs the growing
//  sequence each step; this is quadratic and suits the short outputs an on-device assistant produces.
//
//  Gemma's tokenizer is a byte-fallback BPE the core reader cannot produce (its `tokenizer.model`
//  scores are merge ranks, not unigram log-probabilities), so this backend uses `NFKMLXGemmaTokenizer`.
//  A conversation renders through the release's chat template, Gemma 4's `<|turn>` format; a tokenizer
//  that carries `<start_of_turn>` takes its turns from those special-token ids instead.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

/// Holds the decoder's forward and the tokenizer across the async job boundary. `MLXArray` and the
/// modules are not `Sendable`; the backend runs generation on a background queue, so the pieces are
/// carried through an unchecked holder, as the core language backend does.
final class NFKGemmaBackendHolder: @unchecked Sendable {
    let logits: (MLXArray) -> MLXArray
    let tokenizer: NFKMLXGemmaTokenizer
    /// The draft decoder's logits, which propose tokens for `logits` to verify; nil where nothing drafts.
    let draft: ((MLXArray) -> MLXArray)?
    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKMLXGemmaTokenizer,
         draft: ((MLXArray) -> MLXArray)? = nil) {
        self.logits = logits
        self.tokenizer = tokenizer
        self.draft = draft
    }
}

/// A prefill-only text-generation backend over a Gemma 4 decoder.
@objc(NFKMLXGemmaBackend)
public final class NFKMLXGemmaBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKGemmaBackendHolder
    private let identifier: String
    private let beginOfSequence: Int?
    private let startOfTurn: Int?
    private let endOfTurn: Int?
    private let stopTokens: Set<Int>
    /// The release's chat template, which a tool request and a Gemma 4 conversation render through.
    private let chatTemplate: String?
    /// Whether the tokenizer carries Gemma 4's turn markers, which a conversation without a template
    /// is built from.
    private let carriesTurnMarkers: Bool
    /// What ends a turn the template opened: the hand-built turn's stops, Gemma 4's `<turn|>`, the
    /// `<|tool_response>` a model writes when it waits on a call, and every id the generation config
    /// lists.
    private let templateStopTokens: Set<Int>
    private let describedModules: [Module]
    private let releaseDirectoryURL: URL?
    /// The sampling the release's generation config sets: step 3 of ``NFKMLXGenerationOptions``' sampling order.
    private let releaseSampling: NFKMLXReleaseSampling?
    private let modelInfoCache = NFKMLXModelInfoCache()

    /// The store a paged mixture reads its routed experts from, or nil where they are resident. Its
    /// cache budget can be changed between requests. Introduced in InferKit 0.4.0.
    @objc public let expertStore: NFKMLXExpertStore?

    /// Whether the 26B-A4B mixture's routed experts are paged: left in the release and read as the
    /// router reaches them. Introduced in InferKit 0.4.0.
    @objc public var pagesExperts: Bool { expertStore != nil }

    /// The decoder layers a streamed load leaves in the release, read in their turn on every pass; nil
    /// where every layer is held.
    private let layerStream: NFKMLXLayerStream?

    /// How many decoder layers the model leaves in the release, read in their turn on every pass; 0
    /// where every layer is held. Introduced in InferKit 0.4.0.
    @objc public var streamedLayerCount: Int { layerStream?.layers.count ?? 0 }

    /// The bytes each pass reads from the release for the streamed layers; 0 where every layer is held.
    /// Introduced in InferKit 0.4.0.
    @objc public var streamedBytesPerPass: Int { layerStream?.bytesPerPass ?? 0 }

    /// Whether generation drafts with a smaller Gemma 4 release of the same vocabulary
    /// (``NFKMLXGemmaLanguage/backend(directoryURL:draftDirectoryURL:precision:residency:draftBits:)``).
    /// Introduced in InferKit 0.4.0.
    @objc public var hasDraftModel: Bool { holder.draft != nil }

    /// What the last request that drafted did: its verifying passes and its kept proposals.
    /// Introduced in InferKit 0.4.0.
    public private(set) var lastSpeculativeReport = NFKMLXSpeculativeReport()

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKMLXGemmaTokenizer, identifier: String,
         expertStore: NFKMLXExpertStore? = nil, layerStream: NFKMLXLayerStream? = nil,
         draft: ((MLXArray) -> MLXArray)? = nil, modules: [Module] = [], releaseDirectoryURL: URL? = nil) {
        self.holder = NFKGemmaBackendHolder(logits: logits, tokenizer: tokenizer, draft: draft)
        self.identifier = identifier
        self.describedModules = modules
        self.releaseDirectoryURL = releaseDirectoryURL
        releaseSampling = NFKMLXReleaseSampling(directoryURL: releaseDirectoryURL)
        self.expertStore = expertStore
        self.layerStream = layerStream
        beginOfSequence = tokenizer.id(forToken: "<bos>")
        startOfTurn = tokenizer.id(forToken: "<start_of_turn>")
        endOfTurn = tokenizer.id(forToken: "<end_of_turn>")
        stopTokens = Set([tokenizer.id(forToken: "<eos>"), tokenizer.id(forToken: "<end_of_turn>")]
            .compactMap { $0 })
        chatTemplate = releaseDirectoryURL.flatMap(NFKMLXLanguage.chatTemplate(inDirectory:))
        carriesTurnMarkers = tokenizer.id(forToken: "<|turn>") != nil && tokenizer.id(forToken: "<turn|>") != nil
        templateStopTokens = stopTokens
            .union([tokenizer.id(forToken: "<turn|>"), tokenizer.id(forToken: "<|tool_response>")].compactMap { $0 })
            .union(NFKMLXToolCalling.generationConfigStops(directoryURL: releaseDirectoryURL))
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    /// The decoder's parameter count, weight bytes, precision, and quantization, with the bytes its
    /// release directory occupies on disk and the `model_type` and `max_position_embeddings` its
    /// `config.json` states, and the residency, `paged` or `resident` (`NFKModelInfo*` keys). The 26B-A4B
    /// mixture's paged experts count toward its parameters; its weight bytes are what memory holds, which
    /// leaves out experts read from the release, and a streamed decoder reports `streamed`. Introduced
    /// in InferKit 0.4.0.
    /// The positions the release's configuration states, which bound a prompt and its reply.
    private var contextLength: Int? { modelInfo[NFKModelInfoContextLength] as? Int }

    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            var info = NFKMLXModelDescription.info(of: describedModules, releaseDirectoryURL: releaseDirectoryURL)
            info[NFKModelInfoResidency] = pagesExperts ? NFKModelResidencyPaged
                : layerStream != nil ? NFKModelResidencyStreamed : NFKModelResidencyResident
            return info
        }
    }

    /// The expert cache now, where the model pages its experts: `expert_cache_bytes` and
    /// `expert_cache_byte_budget`; empty where it holds them resident. `NFKInferenceServer` serves it under
    /// the model's "status". Introduced in InferKit 0.4.0.
    @objc public var backendStatus: [String: Any] { expertStore?.statusFigures ?? [:] }

    /// Releases up to `byteCount` of the routed experts the model caches, keeping `keepingByteCount`, and
    /// returns the bytes freed; 0 where the model holds its experts resident. The cache's budget is
    /// unchanged. Introduced in InferKit 0.4.0.
    @objc public func releaseExpertCacheByteCount(_ byteCount: Int, keepingByteCount: Int) -> Int {
        expertStore?.releaseCachedBytes(byteCount, keeping: keepingByteCount) ?? 0
    }

    /// The request parameters the backend reads, with `NFKMLXGenerationParameterKey.draftTokens` where
    /// a draft model proposes tokens. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        var keys: Set<String> = [NFKParameterTemperature, NFKParameterTopP, NFKParameterTopK, NFKParameterMaxTokens,
                                 NFKParameterSeed, NFKParameterTools, NFKMLXGenerationParameterKey.toolCallFormat]
        if hasDraftModel {
            keys.insert(NFKMLXGenerationParameterKey.draftTokens)
        }
        return keys
    }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputPrompt, NFKInputMessages] }

    /// The result keys the backend writes: the reply, the tool calls and the assistant turn of a tool
    /// request, and the usage. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { NFKMLXToolCalling.outputKeys.union([NFKOutputUsage, NFKOutputFinishReason]) }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
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

        if let (promptTokens, format) = try toolPrompt(for: request) {
            let (produced, limit) = try generate(promptTokens, sampling: sampling, maxTokens: maximumTokens, seed: seed,
                                                 stops: templateStopTokens, draftTokens: draftTokens)
            var outputs = NFKMLXToolCalling.outputs(answer: holder.tokenizer.decode(produced), reasoning: "",
                                                    request: request, format: format, source: identifier)
            outputs[NFKOutputFinishReason] = NFKMLXContextRoom.finishReason(produced: produced.count, limit: limit, outputs: outputs)
            outputs[NFKOutputUsage] = NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                          outputTokens: produced.count, reasoningTokens: nil)
            return NFKInferenceResult(outputs: outputs)
        }
        let promptTokens = try tokens(for: request)
        let (produced, limit) = try generate(promptTokens, sampling: sampling, maxTokens: maximumTokens, seed: seed,
                                             stops: rendersConversation(request) ? templateStopTokens : stopTokens,
                                             draftTokens: draftTokens)
        return NFKInferenceResult(outputs: [
            NFKOutputText: holder.tokenizer.decode(produced),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                outputTokens: produced.count, reasoningTokens: nil),
            NFKOutputFinishReason: NFKMLXContextRoom.finishReason(produced: produced.count, limit: limit),
        ])
    }

    /// The input tokens a run of the request costs: a tool request's prompt rendered through the
    /// release's template, or the conversation or prompt tokenized as a run tokenizes it. Runs no
    /// model. Introduced in InferKit 0.4.0.
    @objc(tokenCountForRequest:error:)
    public func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber {
        if let (promptTokens, _) = try toolPrompt(for: request) {
            return NSNumber(value: promptTokens.count)
        }
        return NSNumber(value: try tokens(for: request).count)
    }

    /// A tool request's prompt as token ids and the syntax its calls are read in, or nil for a request
    /// that offers no tools. A tool request renders the release's own template, which writes Gemma 4's
    /// tool syntax. A reasoning effort is refused here, since the backend has no way to apply it.
    private func toolPrompt(for request: NFKInferenceRequest) throws -> ([Int], NFKMLXToolCallFormat?)? {
        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        guard let (text, format) = try NFKMLXToolCalling.toolPrompt(for: request, template: chatTemplate, source: identifier,
                                                                   bosToken: "<bos>", eosToken: "<eos>") else {
            return nil
        }
        return (holder.tokenizer.encode(text), format)
    }

    /// Whether a request's messages render as a Gemma 4 conversation: a message list on a tokenizer
    /// that has no `<start_of_turn>` turns.
    private func rendersConversation(_ request: NFKInferenceRequest) -> Bool {
        request.messages != nil && (startOfTurn == nil || endOfTurn == nil)
    }

    /// The prompt as token ids. A raw prompt is encoded after the begin-of-sequence marker. A message
    /// list on a tokenizer with `<start_of_turn>` is built into those turns, with the special markers
    /// taken as ids. Otherwise it renders as a Gemma 4 conversation through
    /// `NFKMLXGemmaLanguage.conversationText(_:template:mediaPrefix:)`, which needs the release's
    /// chat template or the tokenizer's `<|turn>` markers.
    func tokens(for request: NFKInferenceRequest) throws -> [Int] {
        if rendersConversation(request), let messages = request.messages {
            guard chatTemplate != nil || carriesTurnMarkers else { throw NFKMLXError.unsupportedInput }
            return holder.tokenizer.encode(try NFKMLXGemmaLanguage.conversationText(messages, template: chatTemplate))
        }
        var ids = [Int]()
        if let bos = beginOfSequence { ids.append(bos) }
        if request.messages == nil, let prompt = request.prompt {
            ids += holder.tokenizer.encode(prompt)
            return ids
        }
        guard let messages = request.messages, let startOfTurn, let endOfTurn else {
            throw NFKMLXError.unsupportedInput
        }
        for message in messages {
            let role = (message["role"] as? String) == "assistant" ? "model" : (message["role"] as? String ?? "user")
            let content = (message["content"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            ids.append(startOfTurn)
            ids += holder.tokenizer.encode(role + "\n" + content)
            ids.append(endOfTurn)
            ids += holder.tokenizer.encode("\n")
        }
        ids.append(startOfTurn)
        ids += holder.tokenizer.encode("model\n")
        return ids
    }

    /// Prefill-only generation: the whole growing sequence runs through the decoder each step, or each
    /// round where a draft proposes `draftTokens` tokens. A step whose streamed layer could not be read
    /// throws, and the next request reads the release again.
    func generate(_ promptTokens: [Int], sampling: NFKMLXStepSampling, maxTokens: Int?, seed: UInt64?,
                  stops: Set<Int>, draftTokens: Int) throws -> (tokens: [Int], limit: Int) {
        var tokens = promptTokens
        var produced = [Int]()
        let limit = try NFKMLXContextRoom.bounded(maxTokens, promptTokens: promptTokens.count,
                                                  contextLength: contextLength, source: identifier)
        if let draft = holder.draft, draftTokens > 0 {
            return (try speculate(promptTokens, draft: draft, proposals: draftTokens, sampling: sampling,
                                  limit: limit, seed: seed, stops: stops), limit)
        }
        for step in 0 ..< max(limit, 0) {
            let input = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
            let logits = holder.logits(input)[0, tokens.count - 1]
            let next = sampling.next(logits, seed: seed, step: step)
            try verifyStream()
            if stops.contains(next) { break }
            produced.append(next)
            tokens.append(next)
        }
        return (produced, limit)
    }

    /// Throws a streamed layer's failed read and clears it, so the next request reads the release again.
    private func verifyStream() throws {
        if let failure = layerStream?.readFailure {
            layerStream?.clearFailure()
            throw failure
        }
    }

    /// Drafted prefill-only generation, sampling as `sampling` says. Each round the draft proposes up to `proposals` tokens, one pass of
    /// its own over the growing sequence each, and one pass of the decoder over the sequence and every
    /// proposal scores them all: the leading proposals it agrees with are kept, with the decoder's own
    /// token after them, so a round yields up to `proposals` + 1 tokens for one pass of the decoder.
    ///
    /// @discussion At temperature 0 each kept token is the decoder's argmax given the tokens before it, so
    /// the reply is the greedy reply. Above 0 the proposals pass a rejection test that leaves the reply
    /// distributed as the decoder alone samples it (``NFKMLXLanguageNet/verifyBySampling(rows:proposals:proposalDistributions:options:)``).
    private func speculate(_ promptTokens: [Int], draft: (MLXArray) -> MLXArray, proposals count: Int,
                           sampling: NFKMLXStepSampling, limit: Int, seed: UInt64?, stops: Set<Int>) throws -> [Int] {
        var options = NFKMLXGenerationOptions()
        options.temperature = sampling.temperature
        options.topP = sampling.topP
        options.topK = sampling.topK
        let temperature = sampling.temperature
        if let seed { MLXRandom.seed(seed) }
        func input(_ sequence: [Int]) -> MLXArray { MLXArray(sequence.map(Int32.init)).reshaped([1, sequence.count]) }

        var report = NFKMLXSpeculativeReport()
        defer { lastSpeculativeReport = report }
        var tokens = promptTokens
        var produced = [Int]()
        while produced.count < limit {
            var proposals = [Int]()
            var distributions = [MLXArray]()
            var drafted = tokens
            // The round's last token is the decoder's own, so the draft proposes no more than the reply has room for.
            while proposals.count < Swift.min(count, limit - produced.count - 1) {
                let row = draft(input(drafted))[0, drafted.count - 1]
                let token: Int
                if temperature > 0 {
                    let distribution = NFKMLXLanguageNet.probabilities(of: row, options: options)
                    distributions.append(distribution)
                    token = MLXRandom.categorical(log(distribution)).item(Int.self)
                } else {
                    token = row.argMax().item(Int.self)
                }
                proposals.append(token)
                drafted.append(token)
                if stops.contains(token) { break }
            }
            // Row j predicts what follows the j-th proposal's predecessor, so row j judges proposal j and
            // the last row gives the token after the final proposal.
            let rows = holder.logits(input(tokens + proposals))[0, (tokens.count - 1)...]
            eval(rows)
            try verifyStream()
            let (accepted, following) = temperature > 0
                ? NFKMLXLanguageNet.verifyBySampling(rows: rows, proposals: proposals,
                                                     proposalDistributions: distributions, options: options)
                : NFKMLXLanguageNet.verifyGreedily(rows: rows, proposals: proposals)
            report.rounds += 1
            report.proposed += proposals.count
            report.accepted += accepted
            for token in proposals.prefix(accepted) + [following] {
                guard produced.count < limit, !stops.contains(token) else { return produced }
                produced.append(token)
                tokens.append(token)
            }
        }
        return produced
    }
}

/// Building a text-generation backend from a Gemma 4 release directory.
public extension NFKMLXGemmaLanguage {

    /// Builds a text-generation backend from a released Gemma directory, reading its `config.json`,
    /// weights, and tokenizer. The Gemma 4 E-series, the 26B-A4B mixture, and the 12B unified decoder
    /// are all dispatched from the config's model type, and a Gemma 3 release goes to
    /// ``NFKMLXGemma3``'s backend. Run inference off the render thread.
    static func backend(directoryURL: URL,
                        precision: NFKMLXWeightPrecision = .float32) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: .automatic)
    }

    /// ``backend(directoryURL:precision:)`` with the 26B-A4B mixture's routed experts held as
    /// `residency` plans them: ``NFKMLXResidency/paged`` leaves them in the release and reads each as
    /// the router reaches it, and ``NFKMLXResidency/automatic`` pages only where the release is known
    /// not to fit whole. The dense decoders (the E-series, the 31B, and the 12B unified) have nothing
    /// to page: ``NFKMLXResidency/automatic`` holds them whole where the release fits and streams the
    /// layers it cannot hold otherwise, as ``NFKMLXResidency/streamed`` does wherever the release does
    /// not fit. The release directory has to stay in place for the life of a paged or streamed backend.
    /// Introduced in InferKit 0.4.0.
    static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision = .float32,
                        residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        let configURL = directoryURL.appendingPathComponent("config.json")
        // A Gemma 3 release (`gemma3` / `gemma3_text`) has its own decoder and a cached generation loop.
        if NFKMLXGemma3.isGemma3(configURL: configURL) {
            return try NFKMLXGemma3.backend(directoryURL: directoryURL, precision: precision)
        }
        guard let tokenizer = NFKMLXGemmaTokenizer(directoryURL: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Gemma release has no readable tokenizer.json")
        }
        let decoder = try textDecoder(directoryURL: directoryURL, precision: precision, residency: residency, streams: true)
        return NFKMLXGemmaBackend(logits: decoder.logits, tokenizer: tokenizer, identifier: modelName,
                                  expertStore: decoder.expertStore, layerStream: decoder.layerStream,
                                  modules: [decoder.module], releaseDirectoryURL: directoryURL)
    }

    /// Builds a text-generation backend from a Gemma 4 release with a smaller Gemma 4 release of the same
    /// vocabulary drafting for it: the 31B streamed with the E2B held beside it, for instance.
    ///
    /// @discussion The decoders run prefill-only, so each token of a plain run is one pass of the release,
    /// and a streamed one reads every layer it does not hold on each. With a draft, a round is one pass of
    /// the draft per proposal and one pass of the release over every proposal
    /// (`NFKMLXGenerationParameterKey.draftTokens` a round, 4 by default, 0 to decode without it). At
    /// temperature 0 the reply is the release's greedy reply. The draft loads held at its release's own
    /// precision and is quantized to `draftBits` (4 by default; nil keeps it at that precision), its
    /// embeddings included, since a draft's precision moves how many proposals are kept and never the
    /// reply: the E2B's per-layer embedding alone is 4.7 GB at bfloat16. The release is held as
    /// `residency` plans it against what the draft leaves of the working set. Introduced in InferKit 0.4.0.
    static func backend(directoryURL: URL, draftDirectoryURL: URL, precision: NFKMLXWeightPrecision = .float32,
                        residency: NFKMLXResidency = .automatic, draftBits: Int? = 4) throws -> NFKMLXGemmaBackend {
        let configURL = directoryURL.appendingPathComponent("config.json")
        guard !NFKMLXGemma3.isGemma3(configURL: configURL),
              !NFKMLXGemma3.isGemma3(configURL: draftDirectoryURL.appendingPathComponent("config.json")) else {
            throw NFKMLXError.unsupportedConfiguration("a Gemma 3 release drafts through NFKMLXGemma3's backend")
        }
        guard let tokenizer = NFKMLXGemmaTokenizer(directoryURL: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Gemma release has no readable tokenizer.json")
        }
        let draft = try textDecoder(directoryURL: draftDirectoryURL, precision: draftBits == nil ? precision : .checkpoint,
                                    residency: .resident, streams: false)
        if let draftBits {
            try NFKMLXQuantization.quantize(module: draft.module, bits: draftBits, includeEmbeddings: true)
            // The weights the quantized ones replace go before the release is planned, which they would
            // otherwise crowd.
            eval(draft.module)
            NFKMLXGPU.clearCache()
        }
        let draftBytes = NFKMLXLanguage.heldBytes(of: draft.module)
        let budget = NFKMLXResidencyBudget.current()
        let decoder = try textDecoder(directoryURL: directoryURL, precision: precision, residency: residency, streams: true,
                                      budget: budget > 0 ? Swift.max(budget - draftBytes, 1) : 0, reserving: draftBytes)
        guard draft.vocabularySize == decoder.vocabularySize else {
            throw NFKMLXError.unsupportedConfiguration(
                "the draft's vocabulary has \(draft.vocabularySize) entries where the model's has "
                + "\(decoder.vocabularySize); a draft must share the model's vocabulary")
        }
        return NFKMLXGemmaBackend(logits: decoder.logits, tokenizer: tokenizer, identifier: modelName,
                                  expertStore: decoder.expertStore, layerStream: decoder.layerStream, draft: draft.logits,
                                  modules: [decoder.module], releaseDirectoryURL: directoryURL)
    }

    /// The Objective-C entry for a release with a smaller release drafting for it, the draft at 4 bits;
    /// see ``backend(directoryURL:draftDirectoryURL:precision:residency:draftBits:)``. Introduced in
    /// InferKit 0.4.0.
    @objc(gemmaBackendWithDirectoryURL:draftDirectoryURL:precision:residency:error:)
    static func gemmaBackend(directoryURL: URL, draftDirectoryURL: URL, precision: NFKMLXWeightPrecision,
                             residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, draftDirectoryURL: draftDirectoryURL, precision: precision,
                    residency: residency)
    }

    /// A Gemma 4 text decoder from a release directory, the E-series, the 31B, the mixture, or the 12B
    /// unified as its config's model type says, with its logits, held as `residency` plans it and streamed
    /// where `streams` allows.
    internal static func textDecoder(directoryURL: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                                     streams: Bool, budget: Int = NFKMLXResidencyBudget.current(), reserving reserved: Int = 0)
        throws -> (logits: (MLXArray) -> MLXArray, module: Module, layerStream: NFKMLXLayerStream?,
                   expertStore: NFKMLXExpertStore?, vocabularySize: Int) {
        let configURL = directoryURL.appendingPathComponent("config.json")
        if try isUnified(configURL) {
            let net = makeUnifiedNet(try unifiedConfiguration(fromHuggingFace: configURL))
            try loadUnifiedWeights(into: net, fromDirectory: directoryURL, precision: precision, residency: residency,
                                   streams: streams, budget: budget, reserving: reserved)
            return ({ net($0) }, net, net.layerStream, nil, net.configuration.vocabularySize)
        }
        let net = makeNet(try configuration(fromHuggingFace: configURL))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision, residency: residency,
                        streams: streams, budget: budget, reserving: reserved)
        return ({ net($0) }, net, net.layerStream, net.expertStore, net.configuration.vocabularySize)
    }

    /// The Objective-C entry: builds a Gemma text-generation backend from a release directory.
    @objc(gemmaBackendWithDirectoryURL:error:)
    static func gemmaBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL)
    }

    /// The Objective-C entry holding the mixture's routed experts as `residency` says; see
    /// ``backend(directoryURL:precision:residency:)``. Introduced in InferKit 0.4.0.
    @objc(gemmaBackendWithDirectoryURL:residency:error:)
    static func gemmaBackend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, residency: residency)
    }

    /// The Objective-C entry loading the decoder at `precision` and holding it as `residency` says; see
    /// ``backend(directoryURL:precision:residency:)``. A release larger than memory loads at its own
    /// precision (``NFKMLXWeightPrecision/checkpoint``) and streams what it cannot hold. Introduced in
    /// InferKit 0.4.0.
    @objc(gemmaBackendWithDirectoryURL:precision:residency:error:)
    static func gemmaBackend(directoryURL: URL, precision: NFKMLXWeightPrecision,
                             residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: residency)
    }

    // The optional files are the chat template a Gemma 3 release carries, which the dispatch to
    // ``NFKMLXGemma3`` reads.
    internal static let requiredFiles = ["config.json", "tokenizer.json"]
    internal static let optionalFiles = ["chat_template.jinja", "tokenizer_config.json"]
    internal static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// Downloads a Gemma release and builds its text-generation backend.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer.json`, the chat template where the
    /// repo serves one, and the weights (a single `model.safetensors` or every shard a
    /// `model.safetensors.index.json` names) into the hub cache under `cacheDirectoryURL`, or the
    /// default cache when nil. A cached file is not fetched again. The call blocks on the network;
    /// call it off the render thread. It serves the Gemma 4 releases, such as `google/gemma-4-E2B-it`
    /// and `google/gemma-4-E4B-it`, which are not gated, and the Gemma 3 releases the directory
    /// factory dispatches to ``NFKMLXGemma3``. The `google/gemma-3-*` repos are gated: the caller
    /// sets `NFKHFHub.defaultAccessToken` before the first download, or names the ungated
    /// `unsloth/gemma-3-*` mirrors. Introduced in InferKit 0.4.0.
    @objc(gemmaBackendWithRepo:revision:cacheDirectoryURL:error:)
    static func gemmaBackend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``gemmaBackend(repo:revision:cacheDirectoryURL:)``. Introduced in
    /// InferKit 0.4.0.
    @objc(gemmaBackendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    static func gemmaBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                             completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try gemmaBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// ``gemmaBackend(repo:revision:cacheDirectoryURL:)`` holding the mixture's routed experts as
    /// `residency` says. Introduced in InferKit 0.4.0.
    @objc(gemmaBackendWithRepo:revision:cacheDirectoryURL:residency:error:)
    static func gemmaBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                             residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles), residency: residency)
    }

    /// The asynchronous form of ``gemmaBackend(repo:revision:cacheDirectoryURL:residency:)``.
    /// Introduced in InferKit 0.4.0.
    @objc(gemmaBackendWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    static func gemmaBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                             residency: NFKMLXResidency,
                             completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try gemmaBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                             residency: residency)
        }
    }

    /// The registry name a Gemma backend reports.
    @objc static let gemmaModelName = "gemma4"

    private static var modelName: String { gemmaModelName }

    /// A conversation in Gemma 4's turn format: the release's chat template when `template` is set,
    /// and otherwise the same text built here for plain-text turns.
    ///
    /// @discussion The release's template writes `<bos>`, a `<|turn>system\n…<turn|>\n` turn from a leading
    /// system or developer message, a `<|turn>user\n…<turn|>\n` or `<|turn>model\n…<turn|>\n` turn per
    /// message (a user turn trimmed, a model turn without its thinking channel), and the generation prompt
    /// `<|turn>model\n<|channel>thought\n<channel|>`, which opens the reply with an empty thinking
    /// channel. A non-empty `mediaPrefix` opens the first user turn, where the release's processor places
    /// an image or a clip; messages with no user turn then throw.
    internal static func conversationText(_ messages: [[String: Any]], template: String?,
                                 mediaPrefix: String = "") throws -> String {
        var messages = messages
        if !mediaPrefix.isEmpty {
            guard let index = messages.firstIndex(where: { $0["role"] as? String == "user" }) else {
                throw NFKMLXError.unsupportedConfiguration("the messages carry no user turn to place the media in")
            }
            let content = ((messages[index]["content"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            messages[index]["content"] = mediaPrefix + content
        }
        if let template {
            return try NFKMLXChatTemplateRenderer.render(template, messages: messages, bosToken: "<bos>", eosToken: "<eos>")
        }
        var text = "<bos>"
        if let first = messages.first, let role = first["role"] as? String, role == "system" || role == "developer" {
            text += "<|turn>system\n" + turnContent(first, ofModel: false) + "<turn|>\n"
            messages.removeFirst()
        }
        for message in messages where message["role"] as? String != "tool" {
            let role = message["role"] as? String ?? "user"
            let turn = role == "assistant" ? "model" : role
            text += "<|turn>\(turn)\n" + turnContent(message, ofModel: turn == "model") + "<turn|>\n"
        }
        return text + "<|turn>model\n<|channel>thought\n<channel|>"
    }

    /// A message's text as the template writes it: trimmed, and for a model turn without the
    /// `<|channel>…<channel|>` thinking it carried.
    private static func turnContent(_ message: [String: Any], ofModel: Bool) -> String {
        var content = (message["content"] as? String) ?? ""
        if ofModel {
            content = content.components(separatedBy: "<channel|>").map { part in
                part.range(of: "<|channel>").map { String(part[..<$0.lowerBound]) } ?? part
            }.joined()
        }
        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Whether a config describes the unified decoder rather than the E-series or the mixture.
    private static func isUnified(_ configURL: URL) throws -> Bool {
        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        let text = (json["text_config"] as? [String: Any]) ?? json
        let kind = (text["model_type"] as? String) ?? (json["model_type"] as? String) ?? ""
        return kind.hasPrefix("gemma4_unified")
    }
}
