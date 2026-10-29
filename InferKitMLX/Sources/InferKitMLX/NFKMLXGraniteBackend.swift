//
//  NFKMLXGraniteBackend.swift
//  InferKitMLX
//
//  A text-generation backend for the Granite 4.0-H decoder. The decoder is hybrid: its Mamba-2 layers
//  carry a convolution window and a recurrent scan state, and its attention layers a key-value cache.
//  A generation prefills the prompt once and then runs one token a step against those caches.
//
//  The tokenizer is Granite's byte-level BPE (the GPT-2 family), read from the release's `tokenizer.json`
//  through the shared release-tokenizer reader. A raw prompt is encoded after the begin-of-sequence
//  marker the tokenizer defines; generation stops at the end-of-sequence marker. A message list renders
//  through the release's chat template, which the instruct releases ship as `chat_template.jinja`.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

/// Holds the decoder's forward and the tokenizer across the async job boundary. `MLXArray` and the
/// tokenizer are not `Sendable`, so the crossing is made explicit, as the other language backends do.
final class NFKGraniteBackendHolder: @unchecked Sendable {
    let logits: (MLXArray) -> MLXArray
    /// Starts a generation and answers its step: the logits of the tokens given, which continue the
    /// ones before them. Nil runs the whole sequence through `logits` each step.
    let startGeneration: (() -> (MLXArray) -> MLXArray)?
    let tokenizer: NFKTokenizer
    init(logits: @escaping (MLXArray) -> MLXArray, startGeneration: (() -> (MLXArray) -> MLXArray)?,
         tokenizer: NFKTokenizer) {
        self.logits = logits
        self.startGeneration = startGeneration
        self.tokenizer = tokenizer
    }
}

/// A text-generation backend over a Granite 4.0-H decoder.
@objc(NFKMLXGraniteBackend)
public final class NFKMLXGraniteBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKGraniteBackendHolder
    private let identifier: String
    private let beginOfSequence: Int?
    private let stopTokens: Set<Int>
    /// The release's chat template, which a message list renders through.
    private let chatTemplate: String?
    /// What ends a turn the template opened: the end-of-sequence id and every id the generation
    /// config lists.
    private let templateStopTokens: Set<Int>
    private let describedModules: [Module]
    private let releaseDirectoryURL: URL?
    /// The sampling the release's generation config sets: step 3 of ``NFKMLXGenerationOptions``' sampling order.
    private let releaseSampling: NFKMLXReleaseSampling?
    private let modelInfoCache = NFKMLXModelInfoCache()

    /// A backend over `net` that generates with the decoder's caches.
    convenience init(net: NFKMLXGraniteHybridNet, tokenizer: NFKTokenizer, identifier: String,
                     releaseDirectoryURL: URL? = nil) {
        self.init(logits: { net($0) }, startGeneration: {
            let cache = NFKMLXGraniteCache(layerTypes: net.config.layerTypes)
            return { net($0, cache: cache) }
        }, tokenizer: tokenizer, identifier: identifier, modules: [net], releaseDirectoryURL: releaseDirectoryURL)
    }

    init(logits: @escaping (MLXArray) -> MLXArray, startGeneration: (() -> (MLXArray) -> MLXArray)? = nil,
         tokenizer: NFKTokenizer, identifier: String, modules: [Module] = [], releaseDirectoryURL: URL? = nil) {
        self.holder = NFKGraniteBackendHolder(logits: logits, startGeneration: startGeneration, tokenizer: tokenizer)
        self.identifier = identifier
        self.describedModules = modules
        self.releaseDirectoryURL = releaseDirectoryURL
        releaseSampling = NFKMLXReleaseSampling(directoryURL: releaseDirectoryURL)
        beginOfSequence = tokenizer.bosTokenId >= 0 ? tokenizer.bosTokenId : nil
        stopTokens = tokenizer.eosTokenId >= 0 ? Set([tokenizer.eosTokenId]) : []
        chatTemplate = releaseDirectoryURL.flatMap(NFKMLXLanguage.chatTemplate(inDirectory:))
        templateStopTokens = stopTokens.union(NFKMLXToolCalling.generationConfigStops(directoryURL: releaseDirectoryURL))
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    /// The decoder's parameter count, weight bytes, precision, and quantization, with the bytes its
    /// release directory occupies on disk and the `model_type` and `max_position_embeddings` its
    /// `config.json` states (`NFKModelInfo*` keys). Introduced in InferKit 0.4.0.
    /// The positions the release's configuration states, which bound a prompt and its reply.
    private var contextLength: Int? { modelInfo[NFKModelInfoContextLength] as? Int }

    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            NFKMLXModelDescription.info(of: describedModules, releaseDirectoryURL: releaseDirectoryURL)
        }
    }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterTemperature, NFKParameterTopP, NFKParameterTopK, NFKParameterMaxTokens, NFKParameterSeed, NFKParameterTools,
         NFKMLXGenerationParameterKey.toolCallFormat]
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

        if let (promptTokens, format) = try templatePrompt(for: request) {
            let (produced, limit) = try generate(promptTokens, sampling: sampling, maxTokens: maximumTokens, seed: seed,
                                    stops: templateStopTokens)
            var outputs = NFKMLXToolCalling.outputs(answer: holder.tokenizer.decode(produced.map { NSNumber(value: $0) }),
                                                    reasoning: "", request: request, format: format, source: identifier)
            outputs[NFKOutputFinishReason] = NFKMLXContextRoom.finishReason(produced: produced.count, limit: limit, outputs: outputs)
            outputs[NFKOutputUsage] = NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                          outputTokens: produced.count, reasoningTokens: nil)
            return NFKInferenceResult(outputs: outputs)
        }
        let promptTokens = try tokens(for: request)
        let (produced, limit) = try generate(promptTokens, sampling: sampling, maxTokens: maximumTokens, seed: seed,
                                stops: stopTokens)
        return NFKInferenceResult(outputs: [
            NFKOutputText: holder.tokenizer.decode(produced.map { NSNumber(value: $0) }),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                outputTokens: produced.count, reasoningTokens: nil),
            NFKOutputFinishReason: NFKMLXContextRoom.finishReason(produced: produced.count, limit: limit),
        ])
    }

    /// The prompt as token ids: the begin-of-sequence marker, then the encoded text. A message list on a
    /// release without a chat template is joined into plain text.
    private func tokens(for request: NFKInferenceRequest) throws -> [Int] {
        var ids = [Int]()
        if let bos = beginOfSequence { ids.append(bos) }
        if request.messages == nil, let prompt = request.prompt {
            ids += holder.tokenizer.encode(prompt).map(\.intValue)
            return ids
        }
        guard let messages = request.messages else { throw NFKMLXError.unsupportedInput }
        let joined = messages.compactMap { $0["content"] as? String }.joined(separator: "\n\n")
        ids += holder.tokenizer.encode(joined).map(\.intValue)
        return ids
    }

    /// The input tokens a run of the request costs: a message list rendered through the release's
    /// template, or the prompt, tokenized as a run tokenizes it. Runs no model. Introduced in InferKit 0.4.0.
    @objc(tokenCountForRequest:error:)
    public func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber {
        if let (promptTokens, _) = try templatePrompt(for: request) {
            return NSNumber(value: promptTokens.count)
        }
        return NSNumber(value: try tokens(for: request).count)
    }

    /// A message list's prompt as token ids, rendered through the release's own template, and the
    /// syntax a tool request's calls are read in. Nil for a raw prompt, and for a release without a
    /// template, whose messages are joined. An instruct release answers only the turns its template
    /// writes: given the joined text, Granite 4.0-H 1B ends its reply before its first token. A
    /// reasoning effort is refused here, since the backend has no way to apply it.
    private func templatePrompt(for request: NFKInferenceRequest) throws -> ([Int], NFKMLXToolCallFormat?)? {
        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        if let (text, format) = try NFKMLXToolCalling.toolPrompt(for: request, template: chatTemplate, source: identifier) {
            return (holder.tokenizer.encode(text).map(\.intValue), format)
        }
        guard let chatTemplate, let messages = request.messages else { return nil }
        let text = try NFKMLXChatTemplateRenderer.render(
            chatTemplate, messages: NFKMLXToolCalling.templateMessages(messages, format: nil, source: identifier))
        return (holder.tokenizer.encode(text).map(\.intValue), nil)
    }

    /// Generation: the prompt runs through the decoder once, then each new token alone against the
    /// decoder's caches. A backend built without a generation step runs the whole growing sequence
    /// each step.
    private func generate(_ promptTokens: [Int], sampling: NFKMLXStepSampling, maxTokens: Int?, seed: UInt64?,
                          stops: Set<Int>) throws -> (tokens: [Int], limit: Int) {
        let step = holder.startGeneration?()
        var pending = promptTokens
        var produced = [Int]()
        let limit = try NFKMLXContextRoom.bounded(maxTokens, promptTokens: promptTokens.count,
                                                  contextLength: contextLength, source: identifier)
        for index in 0 ..< max(limit, 0) {
            let input = MLXArray(pending.map(Int32.init)).reshaped([1, pending.count])
            let logits = (step ?? holder.logits)(input)[0, pending.count - 1]
            let next = sampling.next(logits, seed: seed, step: index)
            if stops.contains(next) { break }
            produced.append(next)
            pending = step != nil ? [next] : pending + [next]
        }
        return (produced, limit)
    }
}

/// Building a text-generation backend from a Granite 4.0-H release directory.
public extension NFKMLXGraniteHybrid {

    /// Builds a text-generation backend from a released Granite 4.0-H directory, reading its
    /// `config.json`, weights, and `tokenizer.json`. Run inference off the render thread.
    static func backend(directoryURL: URL,
                        precision: NFKMLXWeightPrecision = .checkpoint) throws -> any NFKInferenceBackend {
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Granite release has no readable tokenizer")
        }
        let net = makeNet(try configuration(fromDirectory: directoryURL))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
        return NFKMLXGraniteBackend(net: net, tokenizer: tokenizer, identifier: graniteModelName,
                                    releaseDirectoryURL: directoryURL)
    }

    /// Builds a text-generation backend from a released Granite 4.0-H directory with a mixture size's
    /// routed experts held as `residency` plans them (`loadWeights(into:fromDirectory:precision:residency:)`).
    /// A dense size loads resident under every residency. A paged load is inference-only. Introduced in
    /// InferKit 0.4.0.
    static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision = .checkpoint,
                        residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: residency, identifier: graniteModelName)
    }

    /// The Objective-C entry: builds a Granite 4.0-H text-generation backend from a release directory.
    @objc(graniteBackendWithDirectoryURL:error:)
    static func graniteBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL)
    }

    /// The Objective-C entry that holds a mixture size's routed experts as `residency` says. Introduced in
    /// InferKit 0.4.0.
    @objc(graniteBackendWithDirectoryURL:residency:error:)
    static func graniteBackend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, residency: residency)
    }

    internal static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    internal static let optionalFiles = ["vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
                                         "generation_config.json"]
    internal static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// Downloads a Granite 4.0-H release and builds its text-generation backend.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer.json`, `tokenizer_config.json`,
    /// whichever of `vocab.json`, `merges.txt`, `added_tokens.json`, `chat_template.jinja`, and
    /// `generation_config.json` the repo serves, and the
    /// weights (a single `model.safetensors` or every shard a `model.safetensors.index.json` names)
    /// into the hub cache under `cacheDirectoryURL`, or the default cache when nil. A cached file is
    /// not fetched again. The call blocks on the network; call it off the render thread. It serves
    /// the `ibm-granite/granite-4.0-h-*` releases, such as `ibm-granite/granite-4.0-h-1b`, which are
    /// not gated. Introduced in InferKit 0.4.0.
    @objc(graniteBackendWithRepo:revision:cacheDirectoryURL:error:)
    static func graniteBackend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``graniteBackend(repo:revision:cacheDirectoryURL:)``. Introduced in
    /// InferKit 0.4.0.
    @objc(graniteBackendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    static func graniteBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try graniteBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Registers `granite-4.0-h` and `granite-4.0-h-tiny` with `NFKMLXModelRegistry`; the registry's URL
    /// is the release directory. Both hold a mixture's routed experts as ``NFKMLXResidency/automatic``
    /// plans them: resident where the release fits the working set, paged where it does not.
    @objc static func register() {
        for name in [graniteModelName, graniteTinyModelName] {
            NFKMLXModelRegistry.register(name: name) { url in
                guard let url else {
                    throw NFKMLXError.unsupportedConfiguration("\(name) builds from a release directory, not without weights")
                }
                return try backend(directoryURL: url, precision: .checkpoint, residency: .automatic, identifier: name)
            }
        }
    }

    /// The registry name a Granite 4.0-H backend reports.
    @objc static let graniteModelName = "granite-4.0-h"

    /// The registry name of Granite 4.0-H Tiny (`ibm-granite/granite-4.0-h-tiny`), a mixture of experts.
    /// Introduced in InferKit 0.4.0.
    @objc static let graniteTinyModelName = "granite-4.0-h-tiny"
}

extension NFKMLXGraniteHybrid {

    /// Builds the backend from a release directory, held as `residency` says, reporting `identifier`: the
    /// name the registry builds it under.
    static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                        identifier: String) throws -> any NFKInferenceBackend {
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Granite release has no readable tokenizer")
        }
        let net = makeNet(try configuration(fromDirectory: directoryURL))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision, residency: residency)
        return NFKMLXGraniteBackend(net: net, tokenizer: tokenizer, identifier: identifier,
                                    releaseDirectoryURL: directoryURL)
    }
}
