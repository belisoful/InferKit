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
//  scores are merge ranks, not unigram log-probabilities), so this backend uses `NFKMLXGemmaTokenizer`
//  and builds the chat turn sequence from the release's own special-token ids.
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

    /// The store a paged mixture reads its routed experts from, or nil where they are resident. Its
    /// cache budget can be changed between requests. Introduced in InferKit 0.4.0.
    public let expertStore: NFKMLXExpertStore?

    /// Whether the 26B-A4B mixture's routed experts are paged: left in the release and read as the
    /// router reaches them. Introduced in InferKit 0.4.0.
    @objc public var pagesExperts: Bool { expertStore != nil }

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKMLXGemmaTokenizer, identifier: String,
         expertStore: NFKMLXExpertStore? = nil) {
        self.holder = NFKGemmaBackendHolder(logits: logits, tokenizer: tokenizer)
        self.identifier = identifier
        self.expertStore = expertStore
        beginOfSequence = tokenizer.id(forToken: "<bos>")
        startOfTurn = tokenizer.id(forToken: "<start_of_turn>")
        endOfTurn = tokenizer.id(forToken: "<end_of_turn>")
        stopTokens = Set([tokenizer.id(forToken: "<eos>"), tokenizer.id(forToken: "<end_of_turn>")]
            .compactMap { $0 })
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterTemperature, NFKParameterMaxTokens, NFKParameterSeed]
    }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputPrompt, NFKInputMessages] }

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

        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        let promptTokens = try tokens(for: request)
        let produced = generate(promptTokens, temperature: temperature, maxTokens: maximumTokens, seed: seed)
        return NFKInferenceResult(outputs: [
            NFKOutputText: holder.tokenizer.decode(produced),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                outputTokens: produced.count, reasoningTokens: nil),
        ])
    }

    /// The prompt as token ids: a raw prompt is encoded after the begin-of-sequence marker; a message
    /// list is rendered into Gemma's turn format, with the special markers taken as ids rather than
    /// encoded as text.
    private func tokens(for request: NFKInferenceRequest) throws -> [Int] {
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
    private func generate(_ promptTokens: [Int], temperature: Float, maxTokens: Int, seed: UInt64?) -> [Int] {
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
            if stopTokens.contains(next) { break }
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
            return NFKMLXGemmaBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: modelName)
        }
        let net = makeNet(try configuration(fromHuggingFace: configURL))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision, residency: residency)
        return NFKMLXGemmaBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: modelName,
                                  expertStore: net.expertStore)
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

    /// Whether a config describes the unified decoder rather than the E-series or the mixture.
    private static func isUnified(_ configURL: URL) throws -> Bool {
        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        let text = (json["text_config"] as? [String: Any]) ?? json
        let kind = (text["model_type"] as? String) ?? (json["model_type"] as? String) ?? ""
        return kind.hasPrefix("gemma4_unified")
    }
}
