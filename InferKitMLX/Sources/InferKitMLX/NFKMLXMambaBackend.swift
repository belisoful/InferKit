//
//  NFKMLXMambaBackend.swift
//  InferKitMLX
//
//  A text-generation backend for the Mamba-2 decoder (Codestral-Mamba and the other pure-Mamba
//  releases). The decoder is prefill-only — it carries no key-value cache — so generation re-runs the
//  growing sequence each step, which the tiny per-token cost of a code-completion reply tolerates.
//
//  The tokenizer is the Mistral byte-fallback BPE (`NFKMLXMistralTokenizer`), read from the release's
//  own `tokenizer.json`. A raw prompt is encoded after the begin-of-sequence marker `<s>`; generation
//  stops at `</s>`.
//

import Foundation
import InferKit
import MLX
import MLXRandom

/// Holds the decoder's forward and the tokenizer across the async job boundary. `MLXArray` and the
/// tokenizer are not `Sendable`, so the crossing is made explicit, as the other language backends do.
final class NFKMambaBackendHolder: @unchecked Sendable {
    let logits: (MLXArray) -> MLXArray
    let tokenizer: NFKMLXMistralTokenizer
    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKMLXMistralTokenizer) {
        self.logits = logits
        self.tokenizer = tokenizer
    }
}

/// A prefill-only text-generation backend over a Mamba-2 decoder.
@objc(NFKMLXMambaBackend)
public final class NFKMLXMambaBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKMambaBackendHolder
    private let identifier: String
    private let beginOfSequence: Int?
    private let stopTokens: Set<Int>

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKMLXMistralTokenizer, identifier: String) {
        self.holder = NFKMambaBackendHolder(logits: logits, tokenizer: tokenizer)
        self.identifier = identifier
        beginOfSequence = tokenizer.id(forToken: "<s>")
        stopTokens = Set([tokenizer.id(forToken: "</s>")].compactMap { $0 })
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

    /// The prompt as token ids: the begin-of-sequence marker, then the encoded text. Codestral-Mamba is
    /// a base completion model with no chat template, so a message list is joined into plain text.
    private func tokens(for request: NFKInferenceRequest) throws -> [Int] {
        var ids = [Int]()
        if let bos = beginOfSequence { ids.append(bos) }
        if let prompt = request.prompt {
            ids += holder.tokenizer.encode(prompt)
            return ids
        }
        guard let messages = request.messages else { throw NFKMLXError.unsupportedInput }
        let joined = messages.compactMap { $0["content"] as? String }.joined(separator: "\n\n")
        ids += holder.tokenizer.encode(joined)
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

/// Building a text-generation backend from a Mamba-2 release directory.
public extension NFKMLXMamba {

    /// Builds a text-generation backend from a released Mamba-2 directory, reading its `config.json`,
    /// weights, and `tokenizer.json`. Run inference off the render thread; the decoder is prefill-only.
    static func backend(directoryURL: URL,
                        precision: NFKMLXWeightPrecision = .checkpoint) throws -> any NFKInferenceBackend {
        guard let tokenizer = NFKMLXMistralTokenizer(directoryURL: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Mamba release has no readable tokenizer.json")
        }
        let net = makeNet(try configuration(fromDirectory: directoryURL))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
        return NFKMLXMambaBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: modelName)
    }

    /// The Objective-C entry: builds a Mamba text-generation backend from a release directory.
    @objc(mambaBackendWithDirectoryURL:error:)
    static func mambaBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL)
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
