//
//  NFKMLXVisionChatBackend.swift
//  InferKitMLX
//
//  The InferKit backend over a vision-language model that answers a prompt or a conversation about its
//  images, keeping each conversation's cache between requests.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN

/// A chat over images: Qwen3-VL, Pixtral, and SmolVLM2 behind the InferKit contract.
///
/// @discussion The backend reads `NFKInputPrompt`, one user turn, or `NFKInputMessages`, a conversation rendered
/// through the release's chat template, and the images under `NFKInputImage` and `NFKInputImages`, in order. A
/// lone prompt reads every image before its text; in a conversation each message takes as many images as its
/// `NFKMessageMetadataKey` `imageCount` states (the server's chat route records it), or, where no message states
/// one, the first user turn takes them all. It writes the reply under `NFKOutputText`, the usage, and the finish
/// reason. Decoding is greedy unless the request sets `NFKParameterTemperature` (with `NFKParameterTopP` and
/// `NFKParameterTopK`); `NFKParameterMaxTokens` bounds the reply (128 by default), and `NFKParameterSeed` seeds a
/// sampled one.
///
/// A request continues the conversation its `NFKParameterConversationKey` names, or, naming none, the kept
/// conversation its prompt continues furthest, and prefills only what it adds. A conversation is kept under
/// its token ids with each image position keyed by a digest of the image's pixels, so a prompt that shows
/// another image at a kept image's place continues nothing past it, and an image the kept prefix already holds
/// does not run through the vision tower again. Introduced in InferKit 0.4.0.
public final class NFKMLXVisionChatBackend: NSObject, NFKInferenceBackend {
    private let model: NFKMLXVisionChatModel
    private let identifier: String
    private let modelInfoCache = NFKMLXModelInfoCache()
    /// One generation at a time: a run moves a conversation's cache through the shared decoder.
    private let generationLock = NSLock()
    /// The conversations' caches, read and written under `generationLock`.
    private let conversations = NFKMLXConversationKeeper<NFKMLXPromptCache>(byteBudget: 0)

    init(model: NFKMLXVisionChatModel, identifier: String) {
        self.model = model
        self.identifier = identifier
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    private var releaseDirectoryURL: URL? { model.chatReleaseDirectoryURL }
    private var adapterStrengths: [Float] { model.chatAdapterStrengths }

    /// The model's parameter count, weight bytes, precision, and quantization, with the bytes its release
    /// directory occupies on disk, the `model_type` and `max_position_embeddings` its `config.json` states,
    /// and the residency, `streamed` for a decoder that streams its layers (`NFKModelInfo*` keys).
    /// Introduced in InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            let configuration = model.chatDecoder.configuration
            var info = NFKMLXModelDescription.info(of: model.chatModules, releaseDirectoryURL: releaseDirectoryURL)
            info[NFKModelInfoResidency] = model.chatDecoder.layerStream != nil ? NFKModelResidencyStreamed
                : NFKModelResidencyResident
            info[NFKModelInfoContextLength] = info[NFKModelInfoContextLength] ?? configuration.maximumPositions
            let precision: NFKMLXWeightPrecision = info[NFKModelInfoPrecision] as? String == "float32" ? .float32 : .checkpoint
            info[NFKModelInfoKeyValueBytesPerToken] = NFKMLXModelSizing.keyValueBytesPerToken(of: configuration,
                                                                                               precision: precision)
            return info
        }
    }

    /// The request parameters a backend reads, the adapter strengths among them where it holds adapters.
    static func parameterKeys(holdsAdapters: Bool) -> Set<String> {
        let keys: Set<String> = [NFKParameterTemperature, NFKParameterTopP, NFKParameterTopK, NFKParameterMaxTokens,
                                 NFKParameterSeed, NFKParameterConversationKey]
        return holdsAdapters ? keys.union([NFKParameterAdapterStrengths]) : keys
    }

    static let inputKeys: Set<String> = [NFKInputPrompt, NFKInputMessages, NFKInputImage, NFKInputImages]
    static let outputKeys: Set<String> = [NFKOutputText, NFKOutputUsage, NFKOutputFinishReason]

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { Self.parameterKeys(holdsAdapters: !adapterStrengths.isEmpty) }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { Self.inputKeys }

    /// The result keys the backend writes. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { Self.outputKeys }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        // The held adapters read the run's strengths as each pass builds its graph.
        try NFKMLXAdapterRun.$strengths.withValue(try NFKMLXAdapterRun.strengths(for: request, loaded: adapterStrengths)) {
            try run(request)
        }
    }

    private func run(_ request: NFKInferenceRequest) throws -> NFKInferenceResult {
        var sampling = NFKMLXStepSampling(release: nil)
        sampling.read(request)
        let maximumTokens = (request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber)?.intValue
            ?? NFKMLXGenerationBackend.defaultAnswerTokens
        let seed = (request.parameter(forKey: NFKParameterSeed) as? NSNumber)?.uint64Value
        let (inputIds, images) = try prompt(for: request)
        let limit = try NFKMLXContextRoom.bounded(maximumTokens, promptTokens: inputIds.count,
                                                  contextLength: model.chatDecoder.configuration.contextLength,
                                                  source: identifier)
        var produced = [Int]()
        var cached = 0
        if limit > 0 {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
            (produced, cached) = try NFKMLXVisionChat.run(
                model: model, inputIds: inputIds, images: images, conversations: conversations, request: request,
                sampling: sampling, limit: limit, seed: seed,
                afforded: { NFKMLXConversationBudget.afforded(info: modelInfo, runBytes: runBytes) },
                spills: adapterStrengths.isEmpty)
        }
        return NFKInferenceResult(outputs: [
            NFKOutputText: model.chatTokenizer.decode(produced.map { NSNumber(value: $0) }),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: inputIds.count, cachedTokens: cached,
                                                outputTokens: produced.count, reasoningTokens: nil),
            NFKOutputFinishReason: NFKMLXContextRoom.finishReason(produced: produced.count, limit: limit),
        ])
    }

    /// The bytes one run's keys and values take at the context length.
    private var runBytes: Int { NFKMLXConversationBudget.runBytes(info: modelInfo, positions: nil) }

    /// The prompt's token ids and its prepared images. A reasoning effort is refused, since the models have no
    /// way to apply it.
    private func prompt(for request: NFKInferenceRequest) throws -> (ids: [Int], images: [NFKMLXVisionImage]) {
        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        let images = try NFKMLXVisionPrompt.images(of: request).map { model.prepareChatImage($0) }
        let text: String
        if let messages = request.messages {
            text = try NFKMLXVisionPrompt.render(messages, images: images, model: model)
        } else if let question = request.prompt {
            text = model.singleTurnPrompt(question: question, images: images)
        } else {
            throw NFKMLXError.unsupportedInput
        }
        return (model.chatTokenizer.encode(text).map(\.intValue), images)
    }

    /// The input tokens a run of the request costs, its images' positions included. Runs no model.
    /// Introduced in InferKit 0.4.0.
    @objc(tokenCountForRequest:error:)
    public func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber {
        NSNumber(value: try prompt(for: request).ids.count)
    }

    // MARK: Conversation caches

    /// The bytes the conversations' caches may hold together.
    ///
    /// @discussion A request continues the conversation its `NFKParameterConversationKey` names, or, naming
    /// none, the kept conversation its prompt continues furthest, so a follow-up turn prefills only what it
    /// adds. When the caches outgrow the budget, the least recently used conversations are written to
    /// ``conversationSpillDirectoryURL`` and dropped, never the one a request just ran. Until it is set, the
    /// budget is what the memory planned beside the model affords after one run's cache at the context
    /// length, recomputed each run; setting it fixes it, and lowering it drops down to it at once. Introduced
    /// in InferKit 0.4.0.
    @objc public var conversationCacheByteBudget: Int {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            return conversations.states.byteBudget
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.fixBudget(newValue)
        }
    }

    /// How many conversations hold a cache. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheCount: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return conversations.states.count
    }

    /// The bytes the conversations' caches occupy. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheBytes: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return conversations.states.heldBytes
    }

    /// Drops one named conversation's cache, in memory and on disk. Introduced in InferKit 0.4.0.
    @objc(resetPromptCacheForConversation:)
    public func resetPromptCache(forConversation conversation: String) {
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.states.remove(conversation)
        conversations.spill?.remove(named: conversation)
        conversations.refresh()
    }

    /// Drops every conversation's cache in memory; spilled ones stay on disk. Introduced in InferKit 0.4.0.
    @objc public func resetPromptCache() {
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.states.removeAll()
        conversations.refresh()
    }

    /// Writes every conversation's cache to the spill directory and drops it from memory when no run is in
    /// progress, and returns the bytes they held; returns 0 at once while a run holds them.
    /// `NFKInferenceServer` calls it to relieve memory, and the reloadable backend before it unloads the model.
    /// Introduced in InferKit 0.4.0.
    @objc public func releaseIdleConversationCaches() -> Int {
        guard generationLock.try() else { return 0 }
        defer { generationLock.unlock() }
        return conversations.releaseAll(spills: adapterStrengths.isEmpty)
    }

    /// The directory conversations the memory budget lets go are written to, so a later request continues one
    /// without its prefill, after an unload or a restart too; nil writes none. A backend over a release
    /// directory defaults to a directory under the user's caches that only the same release and weights read.
    /// A model that holds adapters writes none. Introduced in InferKit 0.4.0.
    @objc public var conversationSpillDirectoryURL: URL? {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
            return conversations.spill?.directory
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.setSpillDirectory(newValue)
        }
    }

    /// The bytes the spilled conversations may take on disk; the least recently written go first past it.
    /// Defaults to 8 GiB. Introduced in InferKit 0.4.0.
    @objc public var conversationSpillByteBudget: Int {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
            return conversations.spill?.byteBudget ?? NFKMLXConversationSpill.defaultByteBudget
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
            conversations.spill?.byteBudget = newValue
        }
    }

    /// The conversation caches as of the last run or change: `conversation_caches`,
    /// `conversation_cache_bytes`, and `conversation_cache_byte_budget`. `NFKInferenceServer` serves it under
    /// the model's "status". It never waits on a run in progress. Introduced in InferKit 0.4.0.
    @objc public var backendStatus: [String: Any] { conversations.status }
}
