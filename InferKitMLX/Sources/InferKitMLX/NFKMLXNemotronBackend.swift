//
//  NFKMLXNemotronBackend.swift
//  InferKitMLX
//
//  A text-generation backend for the Nemotron-H decoder (Nemotron Nano 2). The decoder is hybrid: its
//  Mamba-2 layers carry a recurrent state rather than a key-value cache, so generation re-runs the
//  growing sequence each step, as the Granite and pure-Mamba backends do.
//
//  The tokenizer is Nemotron's byte-level BPE, read from the release's `tokenizer.json` through the
//  shared release-tokenizer reader. The release sets `add_bos_token` false, so a raw prompt is encoded
//  with no begin-of-sequence marker prepended (which matches the oracle's raw-id generation);
//  generation stops at the end-of-sequence marker.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

/// Holds the decoder's forward and the tokenizer across the async job boundary. `MLXArray` and the
/// tokenizer are not `Sendable`, so the crossing is made explicit, as the other language backends do.
final class NFKNemotronBackendHolder: @unchecked Sendable {
    let logits: (MLXArray) -> MLXArray
    let tokenizer: NFKTokenizer
    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKTokenizer) {
        self.logits = logits
        self.tokenizer = tokenizer
    }
}

/// A prefill-only text-generation backend over a Nemotron-H decoder.
@objc(NFKMLXNemotronBackend)
public final class NFKMLXNemotronBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKNemotronBackendHolder
    private let identifier: String
    private let stopTokens: Set<Int>
    /// The release's chat template, which a tool request renders through.
    private let chatTemplate: String?
    /// What ends a turn the template opened: the end-of-sequence id and every id the generation
    /// config lists.
    private let templateStopTokens: Set<Int>
    private let describedModules: [Module]
    private let releaseDirectoryURL: URL?
    /// The sampling the release's generation config sets: step 3 of ``NFKMLXGenerationOptions``' sampling order.
    private let releaseSampling: NFKMLXReleaseSampling?
    private let modelInfoCache = NFKMLXModelInfoCache()

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKTokenizer, identifier: String,
         modules: [Module] = [], releaseDirectoryURL: URL? = nil) {
        self.holder = NFKNemotronBackendHolder(logits: logits, tokenizer: tokenizer)
        self.identifier = identifier
        self.describedModules = modules
        self.releaseDirectoryURL = releaseDirectoryURL
        releaseSampling = NFKMLXReleaseSampling(directoryURL: releaseDirectoryURL)
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
        var maximumTokens = 256
        var seed: UInt64?
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber {
            maximumTokens = value.intValue
        }
        if let value = request.parameter(forKey: NFKParameterSeed) as? NSNumber {
            seed = value.uint64Value
        }

        if let (promptTokens, format) = try toolPrompt(for: request) {
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

    /// The prompt as token ids: the encoded text, with no begin-of-sequence marker (the release sets
    /// `add_bos_token` false). A message list is joined into plain text (the base decoder carries no
    /// chat template).
    private func tokens(for request: NFKInferenceRequest) throws -> [Int] {
        if request.messages == nil, let prompt = request.prompt {
            return holder.tokenizer.encode(prompt).map(\.intValue)
        }
        guard let messages = request.messages else { throw NFKMLXError.unsupportedInput }
        let joined = messages.compactMap { $0["content"] as? String }.joined(separator: "\n\n")
        return holder.tokenizer.encode(joined).map(\.intValue)
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
    private func generate(_ promptTokens: [Int], sampling: NFKMLXStepSampling, maxTokens: Int, seed: UInt64?,
                          stops: Set<Int>) throws -> (tokens: [Int], limit: Int) {
        var tokens = promptTokens
        var produced = [Int]()
        let limit = try NFKMLXContextRoom.bounded(maxTokens, promptTokens: promptTokens.count,
                                                  contextLength: contextLength, source: identifier)
        for step in 0 ..< max(limit, 0) {
            let input = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
            let logits = holder.logits(input)[0, tokens.count - 1]
            let next = sampling.next(logits, seed: seed, step: step)
            if stops.contains(next) { break }
            produced.append(next)
            tokens.append(next)
        }
        return (produced, limit)
    }
}

/// Building a text-generation backend from a Nemotron-H release directory.
public extension NFKMLXNemotronH {

    /// Builds a text-generation backend from a released Nemotron Nano 2 directory, reading its
    /// `config.json`, weights, and `tokenizer.json`. Run inference off the render thread; the decoder
    /// re-runs the growing sequence each step (its Mamba layers carry a recurrent state, not a cache).
    static func backend(directoryURL: URL,
                        precision: NFKMLXWeightPrecision = .checkpoint) throws -> any NFKInferenceBackend {
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Nemotron release has no readable tokenizer")
        }
        let net = makeNet(try configuration(fromDirectory: directoryURL))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
        return NFKMLXNemotronBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: nemotronModelName,
                                     modules: [net], releaseDirectoryURL: directoryURL)
    }

    /// The Objective-C entry: builds a Nemotron Nano 2 text-generation backend from a release directory.
    @objc(nemotronBackendWithDirectoryURL:error:)
    static func nemotronBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL)
    }

    internal static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    internal static let optionalFiles = ["vocab.json", "merges.txt", "added_tokens.json"]
    internal static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// Downloads a Nemotron Nano 2 release and builds its text-generation backend.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer.json`, `tokenizer_config.json`,
    /// whichever of `vocab.json`, `merges.txt`, and `added_tokens.json` the repo serves, and every
    /// shard the release's `model.safetensors.index.json` names into the hub cache under
    /// `cacheDirectoryURL`, or the default cache when nil. A cached file is not fetched again. The
    /// call blocks on the network; call it off the render thread. It serves
    /// `nvidia/NVIDIA-Nemotron-Nano-9B-v2`, which is not gated. Introduced in InferKit 0.4.0.
    @objc(nemotronBackendWithRepo:revision:cacheDirectoryURL:error:)
    static func nemotronBackend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``nemotronBackend(repo:revision:cacheDirectoryURL:)``. Introduced in
    /// InferKit 0.4.0.
    @objc(nemotronBackendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    static func nemotronBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try nemotronBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Registers `nemotron-nano-2` with `NFKMLXModelRegistry`; the registry's URL is the release directory.
    @objc static func register() {
        NFKMLXModelRegistry.register(name: nemotronModelName) { url in
            guard let url else {
                throw NFKMLXError.unsupportedConfiguration("nemotron-nano-2 builds from a release directory, not without weights")
            }
            return try nemotronBackend(directoryURL: url)
        }
    }

    /// The registry name a Nemotron Nano 2 backend reports.
    @objc static let nemotronModelName = "nemotron-nano-2"
}
