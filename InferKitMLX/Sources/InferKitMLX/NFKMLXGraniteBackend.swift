//
//  NFKMLXGraniteBackend.swift
//  InferKitMLX
//
//  A text-generation backend for the Granite 4.0-H decoder. The decoder is hybrid: its Mamba-2 layers
//  carry a recurrent state rather than a key-value cache, so generation re-runs the growing sequence
//  each step, as the pure-Mamba backend does. The tiny per-token cost of a short reply tolerates it.
//
//  The tokenizer is Granite's byte-level BPE (the GPT-2 family), read from the release's `tokenizer.json`
//  through the shared release-tokenizer reader. A raw prompt is encoded after the begin-of-sequence
//  marker the tokenizer defines; generation stops at the end-of-sequence marker.
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
    let tokenizer: NFKTokenizer
    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKTokenizer) {
        self.logits = logits
        self.tokenizer = tokenizer
    }
}

/// A prefill-only text-generation backend over a Granite 4.0-H decoder.
@objc(NFKMLXGraniteBackend)
public final class NFKMLXGraniteBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKGraniteBackendHolder
    private let identifier: String
    private let beginOfSequence: Int?
    private let stopTokens: Set<Int>
    /// The release's chat template, which a tool request renders through.
    private let chatTemplate: String?
    /// What ends a turn the template opened: the end-of-sequence id and every id the generation
    /// config lists.
    private let templateStopTokens: Set<Int>
    private let describedModules: [Module]
    private let releaseDirectoryURL: URL?
    private let modelInfoCache = NFKMLXModelInfoCache()

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKTokenizer, identifier: String,
         modules: [Module] = [], releaseDirectoryURL: URL? = nil) {
        self.holder = NFKGraniteBackendHolder(logits: logits, tokenizer: tokenizer)
        self.identifier = identifier
        self.describedModules = modules
        self.releaseDirectoryURL = releaseDirectoryURL
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
    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            NFKMLXModelDescription.info(of: describedModules, releaseDirectoryURL: releaseDirectoryURL)
        }
    }

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
            var outputs = NFKMLXToolCalling.outputs(answer: holder.tokenizer.decode(produced.map { NSNumber(value: $0) }),
                                                    reasoning: "", request: request, format: format, source: identifier)
            outputs[NFKOutputUsage] = NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                          outputTokens: produced.count, reasoningTokens: nil)
            return NFKInferenceResult(outputs: outputs)
        }
        let promptTokens = try tokens(for: request)
        let produced = generate(promptTokens, temperature: temperature, maxTokens: maximumTokens, seed: seed,
                                stops: stopTokens)
        return NFKInferenceResult(outputs: [
            NFKOutputText: holder.tokenizer.decode(produced.map { NSNumber(value: $0) }),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                outputTokens: produced.count, reasoningTokens: nil),
        ])
    }

    /// The prompt as token ids: the begin-of-sequence marker, then the encoded text. A message list is
    /// joined into plain text (the base decoder carries no chat template).
    private func tokens(for request: NFKInferenceRequest) throws -> [Int] {
        var ids = [Int]()
        if let bos = beginOfSequence { ids.append(bos) }
        if let prompt = request.prompt {
            ids += holder.tokenizer.encode(prompt).map(\.intValue)
            return ids
        }
        guard let messages = request.messages else { throw NFKMLXError.unsupportedInput }
        let joined = messages.compactMap { $0["content"] as? String }.joined(separator: "\n\n")
        ids += holder.tokenizer.encode(joined).map(\.intValue)
        return ids
    }

    /// The input tokens a run of the request costs: a tool request's prompt rendered through the
    /// release's template, or the joined prompt, tokenized as a run tokenizes it. Runs no model.
    /// Introduced in InferKit 0.4.0.
    @objc(tokenCountForRequest:error:)
    public func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber {
        if let (promptTokens, _) = try toolPrompt(for: request) {
            return NSNumber(value: promptTokens.count)
        }
        return NSNumber(value: try tokens(for: request).count)
    }

    /// A tool request's prompt as token ids and the syntax its calls are read in, or nil for a request
    /// that offers no tools. A tool request renders the release's own template, which is what the
    /// model's tool format was trained on; a plain chat keeps the joined prompt. A reasoning effort is
    /// refused here, since the backend has no way to apply it.
    private func toolPrompt(for request: NFKInferenceRequest) throws -> ([Int], NFKMLXToolCallFormat?)? {
        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        guard let (text, format) = try NFKMLXToolCalling.toolPrompt(for: request, template: chatTemplate, source: identifier) else {
            return nil
        }
        return (holder.tokenizer.encode(text).map(\.intValue), format)
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

/// Building a text-generation backend from a Granite 4.0-H release directory.
public extension NFKMLXGraniteHybrid {

    /// Builds a text-generation backend from a released Granite 4.0-H directory, reading its
    /// `config.json`, weights, and `tokenizer.json`. Run inference off the render thread; the decoder
    /// re-runs the growing sequence each step (its Mamba layers carry a recurrent state, not a cache).
    static func backend(directoryURL: URL,
                        precision: NFKMLXWeightPrecision = .checkpoint) throws -> any NFKInferenceBackend {
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Granite release has no readable tokenizer")
        }
        let net = makeNet(try configuration(fromDirectory: directoryURL))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
        return NFKMLXGraniteBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: graniteModelName,
                                    modules: [net], releaseDirectoryURL: directoryURL)
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
    internal static let optionalFiles = ["vocab.json", "merges.txt", "added_tokens.json"]
    internal static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// Downloads a Granite 4.0-H release and builds its text-generation backend.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer.json`, `tokenizer_config.json`,
    /// whichever of `vocab.json`, `merges.txt`, and `added_tokens.json` the repo serves, and the
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
        return NFKMLXGraniteBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: identifier,
                                    modules: [net], releaseDirectoryURL: directoryURL)
    }
}
