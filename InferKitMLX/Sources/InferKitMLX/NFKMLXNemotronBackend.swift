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
    let draft: ((MLXArray) -> MLXArray)?
    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKTokenizer, draft: ((MLXArray) -> MLXArray)? = nil) {
        self.logits = logits
        self.tokenizer = tokenizer
        self.draft = draft
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
    private let layerStream: NFKMLXLayerStream?

    /// How many decoder blocks the model leaves in the release, read in their turn on every pass; 0
    /// where every block is held. Introduced in InferKit 0.4.0.
    @objc public var streamedLayerCount: Int { layerStream?.layers.count ?? 0 }

    /// The bytes each pass reads from the release for the streamed blocks; 0 where every block is held.
    /// Introduced in InferKit 0.4.0.
    @objc public var streamedBytesPerPass: Int { layerStream?.bytesPerPass ?? 0 }

    /// Whether generation drafts with a smaller Nemotron-H release of the same vocabulary
    /// (``NFKMLXNemotronH/backend(directoryURL:draftDirectoryURL:precision:residency:draftBits:)``).
    /// Introduced in InferKit 0.4.0.
    @objc public var hasDraftModel: Bool { holder.draft != nil }

    /// The rounds, proposals, and kept proposals of the last drafted generation. Introduced in InferKit 0.4.0.
    public private(set) var lastSpeculativeReport = NFKMLXSpeculativeReport()

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKTokenizer, identifier: String,
         modules: [Module] = [], releaseDirectoryURL: URL? = nil, layerStream: NFKMLXLayerStream? = nil,
         draft: ((MLXArray) -> MLXArray)? = nil) {
        self.holder = NFKNemotronBackendHolder(logits: logits, tokenizer: tokenizer, draft: draft)
        self.identifier = identifier
        self.layerStream = layerStream
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
            var outputs = NFKMLXToolCalling.outputs(answer: holder.tokenizer.decode(produced.map { NSNumber(value: $0) }),
                                                    reasoning: "", request: request, format: format, source: identifier)
            outputs[NFKOutputFinishReason] = NFKMLXContextRoom.finishReason(produced: produced.count, limit: limit, outputs: outputs)
            outputs[NFKOutputUsage] = NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                          outputTokens: produced.count, reasoningTokens: nil)
            return NFKInferenceResult(outputs: outputs)
        }
        let promptTokens = try tokens(for: request)
        let (produced, limit) = try generate(promptTokens, sampling: sampling, maxTokens: maximumTokens, seed: seed,
                                stops: stopTokens, draftTokens: draftTokens)
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

    /// Prefill-only generation: the whole growing sequence runs through the decoder each step, or, with
    /// a draft and `draftTokens` above 0, each round of proposals (``NFKMLXPrefillDrafting``).
    private func generate(_ promptTokens: [Int], sampling: NFKMLXStepSampling, maxTokens: Int?, seed: UInt64?,
                          stops: Set<Int>, draftTokens: Int) throws -> (tokens: [Int], limit: Int) {
        var tokens = promptTokens
        var produced = [Int]()
        let limit = try NFKMLXContextRoom.bounded(maxTokens, promptTokens: promptTokens.count,
                                                  contextLength: contextLength, source: identifier)
        if let draft = holder.draft, draftTokens > 0 {
            var report = NFKMLXSpeculativeReport()
            defer { lastSpeculativeReport = report }
            return (try NFKMLXPrefillDrafting.generate(prompt: promptTokens, draft: draft, target: holder.logits,
                                                       verify: verifyStream, proposals: draftTokens, sampling: sampling,
                                                       limit: max(limit, 0), seed: seed, stops: stops, report: &report),
                    limit)
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

    /// Throws a streamed block's failed read and clears it, so the next request reads the release again.
    private func verifyStream() throws {
        if let failure = layerStream?.readFailure {
            layerStream?.clearFailure()
            throw failure
        }
    }
}

/// Building a text-generation backend from a Nemotron-H release directory.
public extension NFKMLXNemotronH {

    /// Builds a text-generation backend from a released Nemotron Nano 2 directory, reading its
    /// `config.json`, weights, and `tokenizer.json`. Run inference off the render thread; the decoder
    /// re-runs the growing sequence each step (its Mamba layers carry a recurrent state, not a cache).
    static func backend(directoryURL: URL,
                        precision: NFKMLXWeightPrecision = .checkpoint) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: .automatic)
    }

    /// ``backend(directoryURL:precision:)`` holding the decoder as `residency` says. Under
    /// ``NFKMLXResidency/automatic`` a release that fits the working set loads whole, and one that does
    /// not, such as the 9B at float32 on a 32 GB machine, holds the first blocks that fit and reads the
    /// others from the release on every pass (``NFKMLXNemotronBackend/streamedLayerCount``);
    /// ``NFKMLXResidency/streamed`` streams wherever the decoder does not fit whole. A failed read fails
    /// the request, and the next request reads every block again. Introduced in InferKit 0.4.0.
    static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision,
                        residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: residency,
                    budget: NFKMLXResidencyBudget.current())
    }

    /// Builds a text-generation backend from a Nemotron-H release with a smaller Nemotron-H release of the
    /// same vocabulary drafting for it: Nemotron Nano 9B v2 drafted by Nemotron 3 Nano 4B, for instance.
    ///
    /// @discussion The decoder runs prefill-only, so each token of a plain run is one pass of the release.
    /// With a draft, a round is one pass of the draft per proposal and one pass of the release over every
    /// proposal (`NFKMLXGenerationParameterKey.draftTokens` a round, 4 by default, 0 to decode without it).
    /// At temperature 0 the reply is the release's greedy reply. The draft loads at its release's own
    /// precision and is quantized to `draftBits` (4 by default; nil keeps it at `precision`), its embeddings
    /// included, since a draft's precision moves how many proposals are kept and never the reply. The
    /// release is held as `residency` plans it against what the draft leaves of the working set.
    /// Introduced in InferKit 0.4.0.
    static func backend(directoryURL: URL, draftDirectoryURL: URL, precision: NFKMLXWeightPrecision = .checkpoint,
                        residency: NFKMLXResidency = .automatic, draftBits: Int? = 4) throws -> NFKMLXNemotronBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: residency,
                    budget: NFKMLXResidencyBudget.current(), draftDirectoryURL: draftDirectoryURL, draftBits: draftBits)
    }

    /// The Objective-C entry for a release with a smaller release drafting for it, the draft at 4 bits; see
    /// ``backend(directoryURL:draftDirectoryURL:precision:residency:draftBits:)``. Introduced in InferKit 0.4.0.
    @objc(nemotronBackendWithDirectoryURL:draftDirectoryURL:precision:residency:error:)
    static func nemotronBackend(directoryURL: URL, draftDirectoryURL: URL, precision: NFKMLXWeightPrecision,
                                residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, draftDirectoryURL: draftDirectoryURL, precision: precision,
                    residency: residency)
    }

    internal static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                                 budget: Int, identifier: String = nemotronModelName, draftDirectoryURL: URL? = nil,
                                 draftBits: Int? = 4) throws -> NFKMLXNemotronBackend {
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Nemotron release has no readable tokenizer")
        }
        let configuration = try configuration(fromDirectory: directoryURL)
        var draft: NFKMLXNemotronHNet?
        if let draftDirectoryURL {
            let draftNet = makeNet(try self.configuration(fromDirectory: draftDirectoryURL))
            guard draftNet.config.vocabularySize == configuration.vocabularySize else {
                throw NFKMLXError.unsupportedConfiguration(
                    "the draft's vocabulary has \(draftNet.config.vocabularySize) entries where the model's has "
                    + "\(configuration.vocabularySize); a draft must share the model's vocabulary")
            }
            try loadWeights(into: draftNet, fromDirectory: draftDirectoryURL, precision: draftBits == nil ? precision : .checkpoint)
            if let draftBits {
                try NFKMLXQuantization.quantize(module: draftNet, bits: draftBits, includeEmbeddings: true)
                // The weights the quantized ones replace go before the release is planned, which they would
                // otherwise crowd.
                eval(draftNet)
                NFKMLXGPU.clearCache()
            }
            draft = draftNet
        }
        let draftBytes = draft.map { NFKMLXLanguage.heldBytes(of: $0) } ?? 0
        let net = makeNet(configuration)
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision, residency: residency,
                        budget: budget > 0 ? Swift.max(budget - draftBytes, 1) : 0, reserving: draftBytes)
        return NFKMLXNemotronBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: identifier,
                                     modules: [net], releaseDirectoryURL: directoryURL, layerStream: net.layerStream,
                                     draft: draft.map { draft in { draft($0) } })
    }

    /// The Objective-C entry: builds a Nemotron Nano 2 text-generation backend from a release directory.
    @objc(nemotronBackendWithDirectoryURL:error:)
    static func nemotronBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL)
    }

    /// The Objective-C entry for ``backend(directoryURL:precision:residency:)``. Introduced in InferKit 0.4.0.
    @objc(nemotronBackendWithDirectoryURL:precision:residency:error:)
    static func nemotronBackend(directoryURL: URL, precision: NFKMLXWeightPrecision,
                                residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: residency)
    }

    internal static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    internal static let optionalFiles = ["vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
                                         "generation_config.json"]
    internal static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// Downloads a Nemotron Nano 2 release and builds its text-generation backend.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer.json`, `tokenizer_config.json`,
    /// whichever of `vocab.json`, `merges.txt`, `added_tokens.json`, `chat_template.jinja`, and
    /// `generation_config.json` the repo serves, and the weights: every shard the release's
    /// `model.safetensors.index.json` names, or its single `model.safetensors`, into the hub cache under
    /// `cacheDirectoryURL`, or the default cache when nil. A cached file is not fetched again. The call
    /// blocks on the network; call it off the render thread. It serves `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
    /// and `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, neither of them gated. Introduced in InferKit 0.4.0.
    @objc(nemotronBackendWithRepo:revision:cacheDirectoryURL:error:)
    static func nemotronBackend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
                        repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                        required: requiredFiles, optional: optionalFiles, weights: weightFiles),
                    precision: .checkpoint, residency: .automatic, budget: NFKMLXResidencyBudget.current(),
                    identifier: repo == nemotron3NanoRepository ? nemotron3NanoModelName : nemotronModelName)
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

    /// The registry name of Nemotron 3 Nano 4B, a smaller Nemotron-H release of Nemotron Nano 2's vocabulary
    /// that drafts for it. Introduced in InferKit 0.4.0.
    @objc static let nemotron3NanoModelName = "nemotron-3-nano-4b"

    /// The repository Nemotron 3 Nano 4B is released in.
    static let nemotron3NanoRepository = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
}
