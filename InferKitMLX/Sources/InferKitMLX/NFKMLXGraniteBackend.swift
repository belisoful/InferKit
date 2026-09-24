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

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKTokenizer, identifier: String) {
        self.holder = NFKGraniteBackendHolder(logits: logits, tokenizer: tokenizer)
        self.identifier = identifier
        beginOfSequence = tokenizer.bosTokenId >= 0 ? tokenizer.bosTokenId : nil
        stopTokens = tokenizer.eosTokenId >= 0 ? Set([tokenizer.eosTokenId]) : []
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
        return NFKMLXGraniteBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: graniteModelName)
    }

    /// The Objective-C entry: builds a Granite 4.0-H text-generation backend from a release directory.
    @objc(graniteBackendWithDirectoryURL:error:)
    static func graniteBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL)
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

    /// Registers `granite-4.0-h` with `NFKMLXModelRegistry`; the registry's URL is the release directory.
    @objc static func register() {
        NFKMLXModelRegistry.register(name: graniteModelName) { url in
            guard let url else {
                throw NFKMLXError.unsupportedConfiguration("granite-4.0-h builds from a release directory, not without weights")
            }
            return try graniteBackend(directoryURL: url)
        }
    }

    /// The registry name a Granite 4.0-H backend reports.
    @objc static let graniteModelName = "granite-4.0-h"
}
