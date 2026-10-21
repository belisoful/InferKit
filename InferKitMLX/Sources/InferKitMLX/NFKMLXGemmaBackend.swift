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
    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKMLXGemmaTokenizer) {
        self.logits = logits
        self.tokenizer = tokenizer
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
    private let modelInfoCache = NFKMLXModelInfoCache()

    /// The store a paged mixture reads its routed experts from, or nil where they are resident. Its
    /// cache budget can be changed between requests. Introduced in InferKit 0.4.0.
    @objc public let expertStore: NFKMLXExpertStore?

    /// Whether the 26B-A4B mixture's routed experts are paged: left in the release and read as the
    /// router reaches them. Introduced in InferKit 0.4.0.
    @objc public var pagesExperts: Bool { expertStore != nil }

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKMLXGemmaTokenizer, identifier: String,
         expertStore: NFKMLXExpertStore? = nil, modules: [Module] = [], releaseDirectoryURL: URL? = nil) {
        self.holder = NFKGemmaBackendHolder(logits: logits, tokenizer: tokenizer)
        self.identifier = identifier
        self.describedModules = modules
        self.releaseDirectoryURL = releaseDirectoryURL
        self.expertStore = expertStore
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
    /// leaves out experts read from the release. Introduced in InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            var info = NFKMLXModelDescription.info(of: describedModules, releaseDirectoryURL: releaseDirectoryURL)
            info[NFKModelInfoResidency] = pagesExperts ? NFKModelResidencyPaged : NFKModelResidencyResident
            return info
        }
    }

    /// The expert cache now, where the model pages its experts: `expert_cache_bytes` and
    /// `expert_cache_byte_budget`; empty where it holds them resident. `NFKInferenceServer` serves it under
    /// the model's "status". Introduced in InferKit 0.4.0.
    @objc public var backendStatus: [String: Any] { expertStore?.statusFigures ?? [:] }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterTemperature, NFKParameterMaxTokens, NFKParameterSeed, NFKParameterTools,
         NFKMLXGenerationParameterKey.toolCallFormat]
    }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputPrompt, NFKInputMessages] }

    /// The result keys the backend writes: the reply, the tool calls and the assistant turn of a tool
    /// request, and the usage. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { NFKMLXToolCalling.outputKeys.union([NFKOutputUsage]) }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        var temperature: Float = 0
        var maximumTokens = 256
        var seed: UInt64?
        if let value = request.parameter(forKey: NFKParameterTemperature) as? NSNumber {
            temperature = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber {
            maximumTokens = value.intValue
        }
        if let value = request.parameter(forKey: NFKParameterSeed) as? NSNumber {
            seed = value.uint64Value
        }

        if let (promptTokens, format) = try toolPrompt(for: request) {
            let produced = generate(promptTokens, temperature: temperature, maxTokens: maximumTokens, seed: seed,
                                    stops: templateStopTokens)
            var outputs = NFKMLXToolCalling.outputs(answer: holder.tokenizer.decode(produced), reasoning: "",
                                                    request: request, format: format, source: identifier)
            outputs[NFKOutputUsage] = NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                          outputTokens: produced.count, reasoningTokens: nil)
            return NFKInferenceResult(outputs: outputs)
        }
        let promptTokens = try tokens(for: request)
        let produced = generate(promptTokens, temperature: temperature, maxTokens: maximumTokens, seed: seed,
                                stops: rendersConversation(request) ? templateStopTokens : stopTokens)
        return NFKInferenceResult(outputs: [
            NFKOutputText: holder.tokenizer.decode(produced),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                outputTokens: produced.count, reasoningTokens: nil),
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

    /// Whether a request's messages render as a Gemma 4 conversation: a message list without a raw
    /// prompt, on a tokenizer that has no `<start_of_turn>` turns.
    private func rendersConversation(_ request: NFKInferenceRequest) -> Bool {
        request.prompt == nil && request.messages != nil && (startOfTurn == nil || endOfTurn == nil)
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
        if let prompt = request.prompt {
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

    /// Prefill-only generation: the whole growing sequence runs through the decoder each step.
    private func generate(_ promptTokens: [Int], temperature: Float, maxTokens: Int, seed: UInt64?,
                          stops: Set<Int>) -> [Int] {
        var tokens = promptTokens
        var produced = [Int]()
        for step in 0 ..< max(maxTokens, 0) {
            let input = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
            let logits = holder.logits(input)[0, tokens.count - 1]
            let next: Int
            if temperature <= 0 {
                next = logits.argMax(axis: -1).item(Int.self)
            } else {
                if let seed { MLXRandom.seed(seed &+ UInt64(step)) }
                next = MLXRandom.categorical(logits * (1 / temperature)).item(Int.self)
            }
            if stops.contains(next) { break }
            produced.append(next)
            tokens.append(next)
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
    /// not to fit whole. The dense decoders have nothing to page and load resident under every
    /// residency. The release directory has to stay in place for the life of a paged backend.
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
        if try isUnified(configURL) {
            let net = makeUnifiedNet(try unifiedConfiguration(fromHuggingFace: configURL))
            try loadUnifiedWeights(into: net, fromDirectory: directoryURL, precision: precision)
            return NFKMLXGemmaBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: modelName,
                                      modules: [net], releaseDirectoryURL: directoryURL)
        }
        let net = makeNet(try configuration(fromHuggingFace: configURL))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision, residency: residency)
        return NFKMLXGemmaBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: modelName,
                                  expertStore: net.expertStore, modules: [net], releaseDirectoryURL: directoryURL)
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
