//
//  NFKMLXGemma3Backend.swift
//  InferKitMLX
//
//  A Gemma 3 release as an InferKit backend: text in, text out, and in a multimodal release an image
//  beside the text. The decoder generates through a hybrid key-value cache (one unbounded for the full
//  layers, one bounded to the window for the sliding ones), so a step costs one token's work rather
//  than the whole sequence's.
//
//  Introduced in InferKit 0.3.1.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

/// Holds the model across the async job boundary. `MLXArray` and the modules are not `Sendable`; the
/// backend runs generation on a background task, so the model is carried through an unchecked holder,
/// as the core language backend does.
final class NFKGemma3Holder: @unchecked Sendable {
    let model: NFKMLXGemma3Model
    init(_ model: NFKMLXGemma3Model) { self.model = model }
}

/// A cancellation flag a job's handler sets and the generation loop reads between tokens.
final class NFKGemma3CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
}

/// Text generation, and image understanding, over a Gemma 3 release.
///
/// Reads `NFKInputPrompt` (a raw prompt, encoded after `<bos>`) or `NFKInputMessages` (rendered through
/// the release's own chat template), plus an optional `NFKInputImage` (a `CGImage`, `CVPixelBuffer`, or
/// `MTLTexture`) in a multimodal release — the image's 256 soft tokens are placed before the text, as
/// the reference processor places them. Returns `NFKOutputText`. `NFKParameterTemperature`,
/// `NFKParameterTopP`, `NFKParameterMaxTokens`, and `NFKParameterSeed` override the defaults. A
/// submitted job reports each token through `partialResult` and honors cancellation between tokens. A
/// request that names its conversation under `NFKParameterConversationKey` continues that
/// conversation's own cache and prefills only what its prompt adds; a request with an image starts
/// the conversation's cache over.
@objc(NFKMLXGemma3Backend)
public final class NFKMLXGemma3Backend: NSObject, NFKInferenceBackend {
    private let holder: NFKGemma3Holder
    private let identifier: String
    /// Generation is serialized: two runs through one set of networks would interleave their work.
    private let generationLock = NSLock()
    /// The named conversations' caches, read and written under `generationLock`.
    private let conversations = NFKMLXConversationKeeper<NFKMLXGemma3PromptCache>(byteBudget: 2 << 30)

    init(model: NFKMLXGemma3Model, identifier: String) {
        holder = NFKGemma3Holder(model)
        self.identifier = identifier
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    /// How many decoder layers a streamed load leaves in the release and reads on every pass; 0 where
    /// every layer is held. Introduced in InferKit 0.4.0.
    @objc public var streamedLayerCount: Int { holder.model.decoder.layerStream?.layers.count ?? 0 }

    /// The bytes each pass reads from the release for its streamed layers; 0 where every layer is held.
    /// Introduced in InferKit 0.4.0.
    @objc public var streamedBytesPerPass: Int { holder.model.decoder.layerStream?.bytesPerPass ?? 0 }

    /// Whether generation drafts with a smaller release (`NFKMLXGemma3Model.useDraft(_:)`).
    /// Introduced in InferKit 0.4.0.
    @objc public var hasDraftModel: Bool { holder.model.draft != nil }

    private let modelInfoCache = NFKMLXModelInfoCache()

    /// The decoder's and, where the release carries them, the vision tower's and projector's parameter
    /// count, weight bytes, precision, and quantization (`NFKModelInfo*` keys). Introduced in
    /// InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            let model = holder.model
            let modules: [Module?] = [model.decoder, model.vision, model.projector]
            return NFKMLXModelDescription.info(of: modules.compactMap { $0 })
        }
    }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterTemperature, NFKParameterTopP, NFKParameterMaxTokens, NFKParameterSeed,
         NFKParameterConversationKey, NFKMLXGenerationParameterKey.draftTokens]
    }

    /// The bytes the conversations' prompt caches may hold together.
    ///
    /// @discussion A request that names its conversation under `NFKParameterConversationKey`
    /// continues from that conversation's own cache, so the turns of several chats served at once
    /// each prefill only what they add. When the caches outgrow the budget, the least recently used
    /// conversations are dropped, never the one a request is running. Lowering the budget drops down
    /// to it at once. Defaults to 2 GiB. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheByteBudget: Int {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            return conversations.states.byteBudget
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.states.byteBudget = newValue
            conversations.states.evict()
            conversations.refresh()
        }
    }

    /// How many conversations hold a prompt cache. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheCount: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return conversations.states.count
    }

    /// The bytes the conversations' prompt caches occupy. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheBytes: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return conversations.states.heldBytes
    }

    /// Drops one conversation's prompt cache. Introduced in InferKit 0.4.0.
    @objc(resetPromptCacheForConversation:)
    public func resetPromptCache(forConversation conversation: String) {
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.states.remove(conversation)
        conversations.refresh()
    }

    /// Drops every conversation's prompt cache. Introduced in InferKit 0.4.0.
    @objc public func resetPromptCache() {
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.states.removeAll()
        conversations.refresh()
    }

    /// The conversation caches as of the last run or change: `conversation_caches`,
    /// `conversation_cache_bytes`, and `conversation_cache_byte_budget`. `NFKInferenceServer` serves it
    /// under the model's "status". It never waits on a run in progress. Introduced in InferKit 0.4.0.
    @objc public var backendStatus: [String: Any] { conversations.status }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputPrompt, NFKInputMessages, NFKInputImage] }

    /// Whether the release carries a vision tower, so a request may attach `NFKInputImage`.
    @objc public var acceptsImages: Bool { holder.model.acceptsImages }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try run(request, onToken: nil)
    }

    private func run(_ request: NFKInferenceRequest, onToken: ((Int, [Int]) -> Bool)?) throws -> NFKInferenceResult {
        var options = NFKMLXGenerationOptions()
        if let value = request.parameter(forKey: NFKParameterTemperature) as? NSNumber {
            options.temperature = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterTopP) as? NSNumber {
            options.topP = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber {
            options.maxTokens = value.intValue
        }
        if let value = request.parameter(forKey: NFKParameterSeed) as? NSNumber {
            options.seed = value.uint64Value
        }
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.draftTokens) as? NSNumber {
            options.draftTokens = value.intValue
        }
        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        let model = holder.model
        let image = request.input(forKey: NFKInputImage)
        if image != nil, !model.acceptsImages {
            throw NFKMLXError.unsupportedConfiguration(
                "\(identifier) is a text-only release; an image needs a multimodal Gemma 3 (4B and up)")
        }
        let ids: [Int]
        if let prompt = request.prompt {
            ids = model.promptTokens(prompt, withImage: image != nil)
        } else if let messages = request.messages {
            ids = model.chatTokens(messages: messages, withImage: image != nil)
        } else {
            throw NFKMLXError.unsupportedInput
        }

        let conversation = NFKMLXConversation.name(of: request)
        generationLock.lock(); defer { generationLock.unlock() }
        let configuration = model.decoder.configuration
        let promptCache = conversation.map { name in
            conversations.states.state(for: name) {
                NFKMLXGemma3PromptCache(kinds: configuration.layerTypes, slidingWindow: configuration.slidingWindow)
            }
        }
        defer {
            if let conversation {
                conversations.states.evict(keeping: conversation)
            }
            conversations.refresh()
        }
        var produced = [Int]()
        try model.generate(tokens: ids, image: image, options: options, promptCache: promptCache) { token in
            produced.append(token)
            return onToken?(token, produced) ?? true
        }
        return NFKInferenceResult(outputs: [
            NFKOutputText: model.decode(produced),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: ids.count, cachedTokens: promptCache?.sharedPrefixLength ?? 0,
                                                outputTokens: produced.count, reasoningTokens: nil),
        ])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let flag = NFKGemma3CancelFlag()
        job.cancellationHandler = { flag.cancel() }
        let maximum = max((request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber)?.intValue ?? 256, 1)
        Task.detached(priority: .userInitiated) { [self] in
            do {
                let result = try run(request) { _, produced in
                    let partial = NFKInferenceResult(outputs: [NFKOutputText: self.holder.model.decode(produced)])
                    job.reportProgress(min(Double(produced.count) / Double(maximum), 0.99), partialResult: partial)
                    return !flag.isCancelled
                }
                if !flag.isCancelled { job.finish(with: result) }
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

/// Building Gemma 3 from a release directory: the model object, and the backend around it.
///
/// `NFKMLXGemma3` is a released `google/gemma-3-*` directory (through the ungated `unsloth/gemma-3-*`
/// mirrors): the text decoder for every size, and the SigLIP vision tower and projector for the
/// multimodal 4B / 12B / 27B. Run inference off the render thread.
@objc(NFKMLXGemma3)
public final class NFKMLXGemma3: NSObject {

    /// The registry name a Gemma 3 backend reports.
    @objc public static let modelName = "gemma3"

    private let holder: NFKGemma3Holder

    /// The loaded model.
    public var model: NFKMLXGemma3Model { holder.model }

    /// How many decoder layers a streamed load leaves in the release and reads on every pass; 0 where
    /// every layer is held. Introduced in InferKit 0.4.0.
    @objc public var streamedLayerCount: Int { holder.model.decoder.layerStream?.layers.count ?? 0 }

    /// The bytes each pass reads from the release for its streamed layers; 0 where every layer is held.
    /// Introduced in InferKit 0.4.0.
    @objc public var streamedBytesPerPass: Int { holder.model.decoder.layerStream?.bytesPerPass ?? 0 }

    /// Whether generation drafts with a smaller release (`NFKMLXGemma3Model.useDraft(_:)`).
    /// Introduced in InferKit 0.4.0.
    @objc public var hasDraftModel: Bool { holder.model.draft != nil }

    init(model: NFKMLXGemma3Model) {
        holder = NFKGemma3Holder(model)
        super.init()
    }

    /// Loads a release: `config.json`, the weights (single-file or sharded), `tokenizer.json`, and the
    /// chat template. A multimodal release's vision tower and projector load beside the decoder.
    ///
    /// - Parameter directory: the release directory.
    /// - Parameter precision: `.float32` (the default, what the parity records were measured at) or
    ///   `.checkpoint` to keep the released bf16, which halves the memory.
    /// - Parameter residency: how the decoder is held. `.automatic` (the default) holds it whole where
    ///   the release fits the machine's working set and streams its layers where it does not;
    ///   `.streamed` streams the layers that do not fit; `.resident`, `.staged`, and `.paged` hold it
    ///   whole and refuse a release that does not fit. Introduced in InferKit 0.4.0.
    public static func load(directoryURL directory: URL, precision: NFKMLXWeightPrecision = .float32,
                            residency: NFKMLXResidency = .automatic) throws -> NFKMLXGemma3 {
        NFKMLXGemma3(model: try model(directoryURL: directory, precision: precision, residency: residency))
    }

    /// The Objective-C entry: loads a release directory.
    @objc(gemma3WithDirectoryURL:error:)
    public static func gemma3(directoryURL: URL) throws -> NFKMLXGemma3 {
        try load(directoryURL: directoryURL)
    }

    /// The Objective-C entry: loads a release directory at float32, its decoder held as `residency`
    /// says. Introduced in InferKit 0.4.0.
    @objc(gemma3WithDirectoryURL:residency:error:)
    public static func gemma3(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXGemma3 {
        try load(directoryURL: directoryURL, residency: residency)
    }

    /// Answers a question about an image (greedy, through the chat template). The release must be a
    /// multimodal one.
    @objc(answerForImage:question:error:)
    public func answer(image: CGImage, question: String) throws -> String {
        try holder.model.answer(image: image, question: question)
    }

    /// Answers a plain question (greedy, through the chat template).
    @objc(answerForQuestion:error:)
    public func answer(question: String) throws -> String {
        try holder.model.answer(image: nil, question: question)
    }

    /// Builds a backend from a release directory, its decoder held as `residency` says (see
    /// ``load(directoryURL:precision:residency:)``).
    public static func backend(directoryURL directory: URL, precision: NFKMLXWeightPrecision = .float32,
                               residency: NFKMLXResidency = .automatic) throws -> any NFKInferenceBackend {
        NFKMLXGemma3Backend(model: try model(directoryURL: directory, precision: precision, residency: residency),
                            identifier: modelName)
    }

    /// The Objective-C entry: builds a backend from a release directory.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: .float32)
    }

    /// The Objective-C entry: builds a backend from a release directory at float32, its decoder held as
    /// `residency` says. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: .float32, residency: residency)
    }

    /// Loads a release with a smaller release of the same vocabulary drafting for it (see
    /// `NFKMLXGemma3Model.useDraft(_:)`): the 27B streamed with the 4B held beside it, for instance.
    ///
    /// @discussion The draft loads held: its decoder alone, quantized to `draftBits` from the release's
    /// own precision (4 by default; nil keeps it at `precision`). A draft only proposes tokens, so its
    /// precision moves how many proposals are kept and never the output, and quantized it leaves the
    /// release more of the working set: a 4B draft at 4 bits holds about 3 GB where it would hold
    /// 8 GB at bfloat16. The release is held as `residency` says, planned against what the draft
    /// leaves. Introduced in InferKit 0.4.0.
    public static func load(directoryURL directory: URL, draftDirectoryURL: URL,
                            precision: NFKMLXWeightPrecision = .float32,
                            residency: NFKMLXResidency = .automatic, draftBits: Int? = 4) throws -> NFKMLXGemma3 {
        NFKMLXGemma3(model: try model(directoryURL: directory, draftDirectoryURL: draftDirectoryURL,
                                      precision: precision, residency: residency, draftBits: draftBits))
    }

    /// Builds a backend from a release with a smaller release drafting for it (see
    /// ``load(directoryURL:draftDirectoryURL:precision:residency:draftBits:)``), the draft at 4 bits.
    /// `NFKMLXGenerationParameterKey.draftTokens` sets the proposals per round, and 0 decodes without
    /// the draft. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:draftDirectoryURL:precision:residency:error:)
    public static func backend(directoryURL directory: URL, draftDirectoryURL: URL, precision: NFKMLXWeightPrecision,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        NFKMLXGemma3Backend(model: try model(directoryURL: directory, draftDirectoryURL: draftDirectoryURL,
                                             precision: precision, residency: residency),
                            identifier: modelName)
    }

    /// The model for a release with the draft release's decoder, quantized to `draftBits`, set as its
    /// draft. The release is planned against what the held draft leaves of the working set.
    static func model(directoryURL directory: URL, draftDirectoryURL: URL, precision: NFKMLXWeightPrecision,
                      residency: NFKMLXResidency, draftBits: Int? = 4) throws -> NFKMLXGemma3Model {
        // A quantized draft loads at the release's own precision; widening it first only raises the peak.
        let draftParts = try load(directory: draftDirectoryURL, precision: draftBits == nil ? precision : .checkpoint,
                                  decoder: true, residency: .resident, vision: false)
        guard let drafter = draftParts.decoder else { throw NFKMLXError.noOutput }
        if let draftBits {
            try NFKMLXQuantization.quantize(module: drafter, bits: draftBits)
            // The quantized arrays are lazy over the weights they replace. Evaluated here, those weights
            // are freed now and cleared from MLX's buffer cache, where they would otherwise sit, larger
            // than the quantized draft, beside the release the plan sizes without them.
            eval(drafter)
            NFKMLXGPU.clearCache()
        }
        let draft = NFKMLXGemma3Model(decoder: drafter, vision: nil, projector: nil, tokenizer: draftParts.tokenizer,
                                      tokens: draftParts.tokens, chatTemplate: nil)
        let draftBytes = drafter.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
        let budget = NFKMLXResidencyBudget.current()
        let parts = try load(directory: directory, precision: precision, decoder: true, residency: residency,
                             budget: budget > 0 ? Swift.max(budget - draftBytes, 1) : 0, reserving: draftBytes)
        guard let decoder = parts.decoder else { throw NFKMLXError.noOutput }
        let model = NFKMLXGemma3Model(decoder: decoder, vision: parts.vision, projector: parts.projector,
                                      tokenizer: parts.tokenizer, tokens: parts.tokens,
                                      chatTemplate: chatTemplate(inDirectory: directory))
        try model.useDraft(draft)
        return model
    }

    static let requiredFiles = ["config.json", "tokenizer.json"]
    static let optionalFiles = ["chat_template.jinja", "tokenizer_config.json"]
    static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// The release directory a download of `repo` fills.
    static func releaseDirectory(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> URL {
        try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles)
    }

    /// Downloads a Gemma 3 release and loads it.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer.json`, the chat template
    /// (`chat_template.jinja`, or `tokenizer_config.json` where the template lives there), and the
    /// weights (a single `model.safetensors` or every shard a `model.safetensors.index.json` names)
    /// into the hub cache under `cacheDirectoryURL`, or the default cache when nil. A cached file is
    /// not fetched again. The call blocks on the network; call it off the render thread. It serves
    /// the Gemma 3 releases, such as `google/gemma-3-270m-it`, `google/gemma-3-1b-it`, and
    /// `google/gemma-3-4b-it`. The `google/gemma-3-*` repos are gated: the caller
    /// sets `NFKHFHub.defaultAccessToken` before the first download, or names the ungated mirrors
    /// such as `unsloth/gemma-3-270m-it`. Introduced in InferKit 0.4.0.
    @objc(gemma3WithRepo:revision:cacheDirectoryURL:error:)
    public static func gemma3(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXGemma3 {
        try load(directoryURL: try releaseDirectory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL))
    }

    /// The asynchronous form of ``gemma3(repo:revision:cacheDirectoryURL:)``, delivered on a
    /// background queue. Introduced in InferKit 0.4.0.
    @objc(gemma3WithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func gemma3(repo: String, revision: String?, cacheDirectoryURL: URL?,
                              completionHandler: @escaping (NFKMLXGemma3?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try gemma3(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// Downloads a Gemma 3 release, as ``gemma3(repo:revision:cacheDirectoryURL:)`` describes, and
    /// builds the backend. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try releaseDirectory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL))
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``. Introduced in
    /// InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Whether a release's `config.json` names a Gemma 3 (`gemma3` or `gemma3_text`).
    static func isGemma3(configURL: URL) -> Bool {
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        let kind = (json["model_type"] as? String) ?? ""
        return kind == "gemma3" || kind == "gemma3_text"
    }

    /// The model object for a release directory, every part loaded, the decoder held as `residency`
    /// says (see ``load(directoryURL:precision:residency:)``).
    public static func model(directoryURL directory: URL, precision: NFKMLXWeightPrecision = .float32,
                             residency: NFKMLXResidency = .automatic) throws -> NFKMLXGemma3Model {
        let parts = try load(directory: directory, precision: precision, decoder: true, residency: residency)
        guard let decoder = parts.decoder else { throw NFKMLXError.noOutput }
        return NFKMLXGemma3Model(decoder: decoder, vision: parts.vision, projector: parts.projector,
                                 tokenizer: parts.tokenizer, tokens: parts.tokens,
                                 chatTemplate: chatTemplate(inDirectory: directory))
    }

    /// The vision tower and projector of a multimodal release alone, the decoder left on disk — what
    /// a test of the vision path loads.
    static func visionParts(directoryURL directory: URL, precision: NFKMLXWeightPrecision = .float32)
        throws -> (vision: NFKMLXGemma3VisionNet, projector: NFKMLXGemma3MultimodalProjector) {
        let parts = try load(directory: directory, precision: precision, decoder: false)
        guard let vision = parts.vision, let projector = parts.projector else {
            throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) carries no vision tower")
        }
        return (vision, projector)
    }

    /// Everything a release directory describes, its weights read once and partitioned by prefix.
    /// `decoder` false leaves the language model unloaded (its weights are most of the file). A
    /// streamed decoder's streamed layers are not read here; their stream reads them in their turn.
    static func load(directory: URL, precision: NFKMLXWeightPrecision, decoder wantsDecoder: Bool,
                     residency: NFKMLXResidency = .automatic, budget: Int = NFKMLXResidencyBudget.current(),
                     reserving reserved: Int = 0, vision wantsVision: Bool = true) throws
        -> (decoder: NFKMLXGemma3Net?, vision: NFKMLXGemma3VisionNet?, projector: NFKMLXGemma3MultimodalProjector?,
            tokenizer: NFKMLXGemmaTokenizer, tokens: NFKMLXGemma3Tokens) {
        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("config.json is not a JSON object")
        }
        let textConfiguration = try NFKMLXGemma3Language.configuration(fromJSON: json)
        guard let tokenizer = NFKMLXGemmaTokenizer(directoryURL: directory) else {
            throw NFKMLXError.unsupportedConfiguration("the Gemma 3 release has no readable tokenizer.json")
        }
        let streamed = wantsDecoder
            ? try streamedLayers(directory: directory, precision: precision,
                                 includesVision: wantsVision && json["vision_config"] != nil,
                                 residency: residency, budget: budget, reserving: reserved)
            : []

        let decoder = wantsDecoder ? NFKMLXGemma3Net(textConfiguration) : nil
        let placeholders = try decoder.map {
            try NFKMLXGemma3Language.stream(streamed, of: $0, directory: directory, precision: precision)
        } ?? []

        var vision: NFKMLXGemma3VisionNet?
        var projector: NFKMLXGemma3MultimodalProjector?
        var tokensPerImage = 256
        if wantsVision, let visionJSON = json["vision_config"] as? [String: Any] {
            func integer(_ key: String, _ fallback: Int) -> Int { (visionJSON[key] as? NSNumber)?.intValue ?? fallback }
            let visionConfiguration = NFKMLXSigLIPConfiguration(
                hiddenSize: integer("hidden_size", 1152), layerCount: integer("num_hidden_layers", 27),
                headCount: integer("num_attention_heads", 16), intermediateSize: integer("intermediate_size", 4304),
                patchSize: integer("patch_size", 14), imageSize: integer("image_size", 896),
                layerNormEpsilon: (visionJSON["layer_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6)
            tokensPerImage = (json["mm_tokens_per_image"] as? NSNumber)?.intValue ?? 256
            vision = NFKMLXGemma3VisionNet(visionConfiguration)
            projector = NFKMLXGemma3MultimodalProjector(
                visionHidden: visionConfiguration.hiddenSize, textHidden: textConfiguration.hiddenSize,
                patchesPerSide: visionConfiguration.grid, tokensPerImage: tokensPerImage,
                eps: visionConfiguration.layerNormEpsilon)
        }

        // Every shard is read once and partitioned by prefix; a 4-D vision convolution weight takes
        // the PyTorch → MLX channel transpose. A tensor nobody wants is skipped before it is converted.
        var decoderWeights = [(String, MLXArray)]()
        var visionWeights = [(String, MLXArray)]()
        var projectorWeights = [(String, MLXArray)]()
        let wanted: (String) -> String? = { key in
            if NFKMLXGemma3Language.decoderName(of: key) != nil {
                guard wantsDecoder, NFKMLXGemma3Language.heldDecoderName(of: key, streamed: streamed) != nil else {
                    return nil
                }
                return key
            }
            if visionName(of: key) != nil || projectorName(of: key) != nil { return vision != nil ? key : nil }
            return nil
        }
        for (key, value) in try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision, remap: wanted) {
            if let name = NFKMLXGemma3Language.decoderName(of: key) {
                decoderWeights.append((name, value))
            } else if let name = visionName(of: key) {
                visionWeights.append((name, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value))
            } else if let name = projectorName(of: key) {
                projectorWeights.append((name, value))
            }
        }
        if let decoder {
            try NFKMLXWeights.apply(decoderWeights + placeholders, to: decoder)
            decoder.layerStream?.prime()
        }
        if let vision, let projector {
            try NFKMLXWeights.apply(visionWeights, to: vision)
            try NFKMLXWeights.apply(projectorWeights, to: projector)
        }

        let tokens = markerTokens(json: json, tokenizer: tokenizer, tokensPerImage: tokensPerImage)
        return (decoder, vision, projector, tokenizer, tokens)
    }

    /// The decoder layers a load leaves in the release under `residency`; empty where it holds them all.
    ///
    /// @discussion `.automatic` holds the decoder whole where the release passes the check every load
    /// makes, so a release that loaded before loads the same way, and plans a stream only where that
    /// check fails. `.streamed` always plans. Every other residency holds the decoder whole, and the
    /// check refuses a release that does not fit. `reserved` is what something held beside the decoder,
    /// such as a draft, takes from the check; `budget` already leaves it out.
    static func streamedLayers(directory: URL, precision: NFKMLXWeightPrecision, includesVision: Bool,
                               residency: NFKMLXResidency, budget: Int, reserving reserved: Int = 0) throws -> Set<Int> {
        let fitsWhole = {
            try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: precision, reserve: reserved)
        }
        switch residency {
        case .automatic where (try? fitsWhole()) != nil, .resident, .staged, .paged:
            try fitsWhole()
            return []
        case .automatic, .streamed:
            let footprint = try decoderFootprint(directory: directory, precision: precision, includesVision: includesVision)
            let plan = try NFKMLXResidencyBudget.plan([footprint], residency: residency, budget: budget)
            guard plan.streams(0) else {
                try fitsWhole()
                return []
            }
            return Set(plan.heldLayers(0, of: footprint).upperBound ..< footprint.layerBytes.count)
        }
    }

    /// The decoder, and the vision tower and projector where they load beside it, as one stage whose
    /// repeated layers are counted apart, at the bytes `precision` holds them in.
    static func decoderFootprint(directory: URL, precision: NFKMLXWeightPrecision,
                                 includesVision: Bool) throws -> NFKMLXStageFootprint {
        var layers = [Int: Int]()
        var fixed = 0
        for url in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            for (key, entry) in try NFKMLXSafetensors.entries(inFile: url) {
                let widens = precision == .float32 && (entry.dtype == "BF16" || entry.dtype == "F16")
                let bytes = entry.byteCount * (widens ? 2 : 1)
                if let name = NFKMLXGemma3Language.decoderName(of: key) {
                    if let layer = NFKMLXLayerStream.layer(ofDecoderName: name)?.layer {
                        layers[layer, default: 0] += bytes
                    } else {
                        fixed += bytes
                    }
                } else if includesVision, visionName(of: key) != nil || projectorName(of: key) != nil {
                    fixed += bytes
                }
            }
        }
        let layerBytes = (0 ..< (layers.keys.max().map { $0 + 1 } ?? 0)).map { layers[$0] ?? 0 }
        return NFKMLXStageFootprint(bytes: fixed + layerBytes.reduce(0, +), layerBytes: layerBytes)
    }

    /// The vision tower's module key for a checkpoint key, or nil for a tensor that is not the tower's.
    /// A release written by transformers 4.x names it `vision_tower.vision_model.`; 5.x nests it under
    /// `model.`.
    static func visionName(of key: String) -> String? {
        NFKMLXGemma3Language.stripped(key, prefixes: ["model.vision_tower.vision_model.", "vision_tower.vision_model."])
    }

    /// The projector's module key for a checkpoint key, or nil.
    static func projectorName(of key: String) -> String? {
        NFKMLXGemma3Language.stripped(key, prefixes: ["model.multi_modal_projector.", "multi_modal_projector."])
    }

    /// The marker ids: the tokenizer's added tokens first, the config's indices as the fallback.
    static func markerTokens(json: [String: Any], tokenizer: NFKMLXGemmaTokenizer,
                             tokensPerImage: Int) -> NFKMLXGemma3Tokens {
        var tokens = NFKMLXGemma3Tokens(tokensPerImage: tokensPerImage)
        func configured(_ key: String) -> Int? { (json[key] as? NSNumber)?.intValue }
        tokens.beginOfSequence = tokenizer.id(forToken: "<bos>") ?? configured("bos_token_id") ?? tokens.beginOfSequence
        tokens.endOfSequence = tokenizer.id(forToken: "<eos>") ?? tokens.endOfSequence
        tokens.startOfTurn = tokenizer.id(forToken: "<start_of_turn>") ?? tokens.startOfTurn
        tokens.endOfTurn = tokenizer.id(forToken: "<end_of_turn>") ?? tokens.endOfTurn
        tokens.startOfImage = tokenizer.id(forToken: "<start_of_image>") ?? configured("boi_token_index") ?? tokens.startOfImage
        tokens.endOfImage = tokenizer.id(forToken: "<end_of_image>") ?? configured("eoi_token_index") ?? tokens.endOfImage
        tokens.imageSoftToken = tokenizer.id(forToken: "<image_soft_token>") ?? configured("image_token_index") ?? tokens.imageSoftToken
        tokens.padToken = tokenizer.id(forToken: "<pad>") ?? configured("pad_token_id") ?? tokens.padToken
        return tokens
    }

    /// The release's chat template. See ``NFKMLXReleaseChatTemplate(inDirectory:)``.
    static func chatTemplate(inDirectory directory: URL) -> String? {
        NFKMLXReleaseChatTemplate(inDirectory: directory)
    }
}
