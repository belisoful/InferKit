//
//  NFKMLXGemma3nBackend.swift
//  InferKitMLX
//
//  Text generation from Gemma 3n, with a picture or a clip in the prompt.
//
//  Introduced in InferKit 0.3.1.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN
import MLXRandom

/// The networks cross the async job boundary here; an `MLXArray` is not `Sendable`.
final class NFKGemma3nHolder: @unchecked Sendable {
    let model: NFKMLXGemma3n
    init(_ model: NFKMLXGemma3n) { self.model = model }
}

final class NFKGemma3nCancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
}

// MARK: - Prompting and generation

extension NFKMLXGemma3n {
    /// The text a picture occupies in a prompt: the release processor's own
    /// `\n\n<start_of_image>` + soft-token run + `<end_of_image>\n\n`.
    var imageSequence: String {
        "\n\n<start_of_image>"
            + String(repeating: "<image_soft_token>", count: model.tokens.visionSoftTokens)
            + "<end_of_image>\n\n"
    }

    /// The text a clip occupies in a prompt.
    var audioSequence: String {
        "\n\n<start_of_audio>"
            + String(repeating: "<audio_soft_token>", count: model.tokens.audioSoftTokens)
            + "<end_of_audio>\n\n"
    }

    var endOfTurn: Int { tokenizer.id(forToken: "<end_of_turn>") ?? 106 }
    var endOfSequence: Int { tokenizer.id(forToken: "<eos>") ?? 1 }

    /// One user turn, with the media placed ahead of the question the way the release's processor
    /// places it.
    ///
    /// @discussion The whole turn is built as TEXT and tokenized once, which is load-bearing: the
    /// `\n` closing `user` and the `\n\n` opening the image run merge into ONE id, and a prompt
    /// assembled from separately encoded pieces reads a different sentence to the model. Gemma 3 has
    /// the same trap.
    public func promptTokens(_ prompt: String, withImage image: Bool = false,
                             withAudio audio: Bool = false) -> [Int] {
        var text = "<bos><start_of_turn>user\n"
        if image { text += imageSequence }
        if audio { text += audioSequence }
        text += prompt + "<end_of_turn>\n<start_of_turn>model\n"
        return tokenizer.encode(text)
    }

    /// A conversation rendered into Gemma's turns. A system message opens the first user turn, which
    /// is where Gemma's own template puts it.
    public func chatTokens(messages: [[String: Any]], withImage image: Bool = false,
                           withAudio audio: Bool = false) -> [Int] {
        var text = "<bos>"
        var pending = ""
        var placed = false
        for message in messages {
            let role = (message["role"] as? String) ?? "user"
            let content = (message["content"] as? String) ?? ""
            if role == "system" {
                pending = content + "\n\n"
                continue
            }
            text += "<start_of_turn>\(role == "assistant" ? "model" : "user")\n"
            if role != "assistant", !placed {
                if image { text += imageSequence }
                if audio { text += audioSequence }
                placed = true
            }
            text += pending + content + "<end_of_turn>\n"
            pending = ""
        }
        text += "<start_of_turn>model\n"
        return tokenizer.encode(text)
    }

    /// The text for a run of generated ids, with the markers left out.
    public func decode(_ ids: [Int]) -> String { tokenizer.decode(ids, skipSpecial: true) }

    /// Generates a continuation, calling `onToken` with each id and stopping when it answers false.
    public func generate(tokens: [Int], image: Any? = nil, audio: MLXArray? = nil,
                         options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                         onToken: (Int) -> Bool) throws {
        try generate(tokens: tokens, image: image, audio: audio, options: options, promptCache: nil, onToken: onToken)
    }

    func generate(tokens: [Int], image: Any?, audio: MLXArray?, options: NFKMLXGenerationOptions,
                  promptCache: NFKMLXGemma3nPromptCache?, onToken: (Int) -> Bool) throws {
        let media = try preparedMedia(image: image, audio: audio)
        try generate(tokens: tokens, pixels: media.pixels, mel: media.mel, options: options, promptCache: promptCache,
                     onToken: onToken)
    }

    /// The picture's pixels and the clip's mel features as the towers read them.
    func preparedMedia(image: Any?, audio: MLXArray?) throws -> (pixels: MLXArray?, mel: MLXArray?) {
        (try image.map { try NFKMLXGemma3nImageProcessor.pixelValues(from: $0) },
         audio.map { NFKMLXGemma3nAudioFeatures()($0) })
    }

    /// The key a conversation keeps `tokens` under: the ids, with the picture's and the clip's soft-token
    /// positions keyed by digests of their pixels and mel features, so a prompt with other media shares nothing
    /// past the first position where they differ.
    func conversationKey(tokens: [Int], pixels: MLXArray?, mel: MLXArray?) throws -> [Int] {
        var key = tokens
        for (array, token) in [(pixels, model.tokens.imageToken), (mel, model.tokens.audioToken)] {
            guard let array else { continue }
            key = try NFKMLXVisionChat.mediaKey(inputIds: key, isMedia: { $0 == token },
                                                media: [(NFKMLXVisionImage.digest(of: array), key.filter { $0 == token }.count)])
        }
        return key
    }

    /// Generation that continues `promptCache` where it is given: the cache rolls back to the prefix it shares
    /// with `tokens`' key (``conversationKey(tokens:pixels:mel:)``), and further to a span's start where that
    /// prefix ends inside a picture's or a clip's soft tokens, since their features fill a whole span; only the
    /// rest is prefilled, a tower running only where it reaches the tower's span, and the cache keeps every
    /// position the run writes. A run that throws leaves the cache empty.
    func generate(tokens: [Int], pixels: MLXArray?, mel: MLXArray?, options: NFKMLXGenerationOptions,
                  promptCache: NFKMLXGemma3nPromptCache?, onToken: (Int) -> Bool) throws {
        if let seed = options.seed { MLXRandom.seed(seed) }
        let key = try conversationKey(tokens: tokens, pixels: pixels, mel: mel)
        let soft: Set<Int> = [model.tokens.imageToken, model.tokens.audioToken]
        let start = promptCache?.align(to: key) { shared in
            var first = shared
            while first > 0, shared < tokens.count, soft.contains(tokens[shared]), tokens[first - 1] == tokens[shared] {
                first -= 1
            }
            return first
        } ?? 0
        let cache = promptCache?.cache ?? NFKMLXGemma3nCache(layerCount: model.decoder.configuration.layerCount)
        let fresh = Array(tokens[start...])
        let stops: Set<Int> = [endOfTurn, endOfSequence]
        do {
            var logits = try model.lastLogits(MLXArray(fresh.map(Int32.init)).reshaped([1, fresh.count]),
                                              image: fresh.contains(model.tokens.imageToken) ? pixels : nil,
                                              audioMel: fresh.contains(model.tokens.audioToken) ? mel : nil,
                                              cache: cache)[0]
            promptCache?.record(Array(key[start...]))
            for _ in 0 ..< max(options.maxTokens, 1) {
                let next = NFKMLXLanguageNet.sample(logits, options: options)
                if stops.contains(next) { break }
                if !onToken(next) { break }
                logits = try model(MLXArray([Int32(next)]).reshaped([1, 1]), cache: cache)[0, -1]
                promptCache?.record([next])
            }
        } catch {
            promptCache?.reset()
            throw error
        }
    }
}

// MARK: - The backend

/// Text generation from Gemma 3n through the InferKit contract.
///
/// @discussion `NFKInputImage` adds a picture and `NFKInputAudio` a clip; either needs a release that
/// carries the matching tower. A request continues the conversation its `NFKParameterConversationKey` names,
/// or, naming none, the kept conversation its prompt continues furthest, so a follow-up turn prefills only what
/// it adds (``conversationCacheByteBudget``). A picture's and a clip's positions are kept under digests of
/// their pixels and features, so a follow-up about the same media does not run their towers again, and one with
/// other media continues nothing past the first position where they differ. Inference is synchronous and multi-second, so a caller
/// runs it off the render thread or takes the job form.
@objc(NFKMLXGemma3nBackend)
public final class NFKMLXGemma3nBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKGemma3nHolder
    private let identifier: String
    private let generationLock = NSLock()

    /// The positions the release states, which bound a prompt and its reply, or nil where unknown.
    private let contextLength: Int?
    /// The sampling the release's generation config sets: step 3 of ``NFKMLXGenerationOptions``' sampling order.
    private let releaseSampling: NFKMLXReleaseSampling?
    /// The strengths the held adapters were loaded at, in load order; empty for a model that holds none.
    private let adapterStrengths: [Float]
    /// The release directory, which names the default spill directory, or nil for a model built without one.
    private let releaseDirectoryURL: URL?
    /// The conversations' caches, read and written under `generationLock`.
    private let conversations = NFKMLXConversationKeeper<NFKMLXGemma3nPromptCache>(byteBudget: 0)

    init(model: NFKMLXGemma3n, identifier: String, contextLength: Int? = nil, releaseSampling: NFKMLXReleaseSampling? = nil,
         adapterStrengths: [Float] = [], releaseDirectoryURL: URL? = nil) {
        holder = NFKGemma3nHolder(model)
        self.identifier = identifier
        self.contextLength = contextLength
        self.releaseSampling = releaseSampling
        self.adapterStrengths = adapterStrengths
        self.releaseDirectoryURL = releaseDirectoryURL
        super.init()
    }

    @objc public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        let keys: Set<String> = [NFKParameterTemperature, NFKParameterTopP, NFKParameterTopK, NFKParameterMaxTokens,
                                 NFKParameterSeed, NFKParameterConversationKey]
        return adapterStrengths.isEmpty ? keys : keys.union([NFKParameterAdapterStrengths])
    }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> {
        [NFKInputPrompt, NFKInputMessages, NFKInputImage, NFKInputAudio]
    }

    /// The result keys the backend writes. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputText, NFKOutputUsage, NFKOutputFinishReason] }

    @objc public var isReady: Bool { true }

    private let modelInfoCache = NFKMLXModelInfoCache()

    /// The decoder's and the carried towers' parameter count, weight bytes, precision, and
    /// quantization (`NFKModelInfo*` keys). Introduced in InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value { NFKMLXModelDescription.info(of: [holder.model.model]) }
    }

    @objc public var acceptsImages: Bool { holder.model.model.vision != nil }
    @objc public var acceptsAudio: Bool { holder.model.model.audio != nil }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try run(request, onToken: nil)
    }

    private func run(_ request: NFKInferenceRequest, onToken: ((Int, [Int]) -> Bool)?) throws
        -> NFKInferenceResult {
        var options = NFKMLXGenerationOptions()
        if let releaseSampling {
            options.temperature = releaseSampling.temperature
            options.topP = releaseSampling.topP
            options.topK = releaseSampling.topK
        }
        if let value = request.parameter(forKey: NFKParameterTemperature) as? NSNumber {
            options.temperature = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterTopP) as? NSNumber {
            options.topP = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterTopK) as? NSNumber {
            options.topK = value.intValue
        }
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber {
            options.maxTokens = value.intValue
        }
        if let value = request.parameter(forKey: NFKParameterSeed) as? NSNumber {
            options.seed = value.uint64Value
        }

        let strengths = try NFKMLXAdapterRun.strengths(for: request, loaded: adapterStrengths)
        let gemma = holder.model
        let (ids, image, waveform) = try prompt(for: request)
        options.maxTokens = try NFKMLXContextRoom.bounded(options.chosenMaxTokens, promptTokens: ids.count,
                                                          contextLength: contextLength, source: identifier)

        let media = try gemma.preparedMedia(image: image, audio: waveform)
        let key = try gemma.conversationKey(tokens: ids, pixels: media.pixels, mel: media.mel)
        let reuses = NFKMLXConversation.reuses(request)
        generationLock.lock(); defer { generationLock.unlock() }
        let layerCount = gemma.model.decoder.configuration.layerCount
        conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
        // A conversation's cache computed at other adapter strengths starts over.
        let (name, kept) = conversations.conversation(
            named: NFKMLXConversation.name(of: request), reuses: reuses, prompt: key, signature: "gemma3n",
            usable: { $0.adapterStrengths == strengths },
            make: {
                let made = NFKMLXGemma3nPromptCache(layerCount: layerCount)
                made.adapterStrengths = strengths
                return made
            },
            read: { url, record in try NFKMLXGemma3nPromptCache.load(from: url, tokens: record.tokens, layerCount: layerCount) })
        let promptCache = name == nil ? nil : kept
        var failed = true
        defer {
            conversations.finish(name, failed: failed, afforded: NFKMLXConversationBudget.afforded(
                info: modelInfo, runBytes: keyValueBytes(positions: contextLength)), spills: adapterStrengths.isEmpty)
        }
        var produced = [Int]()
        // The held adapters read the run's strengths as each pass builds its graph.
        try NFKMLXAdapterRun.$strengths.withValue(strengths) {
            try gemma.generate(tokens: ids, pixels: media.pixels, mel: media.mel, options: options,
                               promptCache: promptCache) { token in
                produced.append(token)
                return onToken?(token, produced) ?? true
            }
        }
        failed = false
        return NFKInferenceResult(outputs: [
            NFKOutputText: gemma.decode(produced),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: ids.count, cachedTokens: promptCache?.sharedPrefixLength ?? 0,
                                                outputTokens: produced.count, reasoningTokens: nil),
            NFKOutputFinishReason: NFKMLXContextRoom.finishReason(produced: produced.count, limit: max(options.maxTokens, 1)),
        ])
    }

    /// The bytes one conversation's keys and values take over `positions` positions: the layers before the
    /// shared tail store them, at the decoder's activation width.
    private func keyValueBytes(positions: Int?) -> Int {
        let c = holder.model.model.decoder.configuration
        let storing = c.sharedKeyValueLayers > 0 ? c.firstSharedKeyValueLayer : c.layerCount
        let elementBytes = holder.model.model.decoder.norm.weight?.dtype.size ?? 4
        return 2 * storing * c.keyValueHeadCount * c.headDimensions * elementBytes * (positions ?? 0)
    }

    // MARK: Conversation caches

    /// The bytes the conversations' prompt caches may hold together.
    ///
    /// @discussion A text request continues the conversation its `NFKParameterConversationKey` names, or,
    /// naming none, the kept conversation its prompt continues furthest, so the turns of several chats served
    /// at once each prefill only what they add. When the caches outgrow the budget, the least recently used
    /// conversations are written to ``conversationSpillDirectoryURL`` and dropped, never the one a request
    /// just ran. Until it is set, the budget is what the memory planned beside the model affords after one
    /// run's keys and values at the context length, recomputed each run; setting it fixes it, and lowering it
    /// drops down to it at once. Introduced in InferKit 0.4.0.
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

    /// Drops one named conversation's prompt cache, in memory and on disk. Introduced in InferKit 0.4.0.
    @objc(resetPromptCacheForConversation:)
    public func resetPromptCache(forConversation conversation: String) {
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.states.remove(conversation)
        conversations.spill?.remove(named: conversation)
        conversations.refresh()
    }

    /// Drops every conversation's prompt cache in memory; spilled ones stay on disk. Introduced in InferKit 0.4.0.
    @objc public func resetPromptCache() {
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.states.removeAll()
        conversations.refresh()
    }

    /// Writes every conversation's prompt cache to the spill directory and drops it from memory when no run is
    /// in progress, and returns the bytes they held; returns 0 at once while a run holds them.
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

    /// The input tokens a run of the request costs: its prompt or chat turns, with the soft tokens an
    /// image or a clip takes, tokenized as a run tokenizes them. Runs no model. Introduced in
    /// InferKit 0.4.0.
    @objc(tokenCountForRequest:error:)
    public func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber {
        NSNumber(value: try prompt(for: request).ids.count)
    }

    /// A run's prompt as token ids, with the image and the 16 kHz waveform it attaches.
    private func prompt(for request: NFKInferenceRequest) throws -> (ids: [Int], image: Any?, waveform: MLXArray?) {
        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        let gemma = holder.model
        let image = request.input(forKey: NFKInputImage)
        if image != nil, !acceptsImages {
            throw NFKMLXError.unsupportedConfiguration("\(identifier) carries no vision tower")
        }
        var waveform: MLXArray?
        if let clip = NFKMLXGemma3nBackend.audio(from: request) {
            guard acceptsAudio else {
                throw NFKMLXError.unsupportedConfiguration("\(identifier) carries no audio tower")
            }
            // The encoder is trained at 16 kHz; a clip arriving at another rate is resampled, which
            // preserves its duration.
            waveform = MLXArray(NFKMLXAudioRate.matched(clip.samples, from: clip.sampleRate, to: 16_000))
        }
        if let messages = request.messages {
            return (gemma.chatTokens(messages: messages, withImage: image != nil, withAudio: waveform != nil), image, waveform)
        }
        if let prompt = request.prompt {
            return (gemma.promptTokens(prompt, withImage: image != nil, withAudio: waveform != nil), image, waveform)
        }
        throw NFKMLXError.unsupportedInput
    }

    /// The samples an `NFKAudioAsset` or raw WAV bytes carry.
    private static func audio(from request: NFKInferenceRequest) -> (samples: [Float], sampleRate: Int)? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL,
           let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.read(data)
        }
        if let data = value as? Data { return NFKMLXWaveFile.read(data) }
        return nil
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let flag = NFKGemma3nCancelFlag()
        job.cancellationHandler = { flag.cancel() }
        let maximum = max((request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber)?.intValue ?? 256, 1)
        Task.detached(priority: .userInitiated) { [self] in
            do {
                let result = try run(request) { _, produced in
                    let partial = NFKInferenceResult(
                        outputs: [NFKOutputText: self.holder.model.decode(produced)])
                    job.reportProgress(min(Double(produced.count) / Double(maximum), 0.99),
                                       partialResult: partial)
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

// MARK: - Building from a release

extension NFKMLXGemma3n {
    /// The registry name the reference models register this under.
    @objc public static let modelName = "gemma3n"

    /// Reads a released `google/gemma-3n-*` directory — the E2B and E4B releases, and the ungated
    /// `unsloth/gemma-3n-*` mirrors of them.
    public static func load(directoryURL directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXGemma3n {
        let networks = try model(directoryURL: directory, precision: precision)
        guard let tokenizer = NFKMLXGemmaTokenizer(directoryURL: directory) else {
            throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) carries no tokenizer.json")
        }
        return NFKMLXGemma3n(model: networks, tokenizer: tokenizer)
    }

    @objc(gemma3nWithDirectoryURL:error:)
    public static func gemma3n(directoryURL: URL) throws -> NFKMLXGemma3n {
        try load(directoryURL: directoryURL)
    }

    /// The model's answer about a picture.
    @objc(answerForImage:question:error:)
    public func answer(image: CGImage, question: String) throws -> String {
        var produced = [Int]()
        try generate(tokens: promptTokens(question, withImage: true), image: image) {
            produced.append($0); return produced.count < 256
        }
        return decode(produced)
    }

    /// The model's answer to a question with no media.
    @objc(answerForQuestion:error:)
    public func answer(question: String) throws -> String {
        var produced = [Int]()
        try generate(tokens: promptTokens(question)) { produced.append($0); return produced.count < 256 }
        return decode(produced)
    }

    /// A text-generation backend around a released directory, with `adapters` held beside its decoder at their
    /// strengths (`holdAdapters(_:in:)`); a run's `NFKParameterAdapterStrengths` sets them.
    public static func backend(directoryURL directory: URL,
                               precision: NFKMLXWeightPrecision = .float32,
                               adapters: [(url: URL, strength: Float)] = []) throws
        -> any NFKInferenceBackend {
        let gemma = try load(directoryURL: directory, precision: precision)
        return NFKMLXGemma3nBackend(model: gemma, identifier: modelName,
                                    contextLength: NFKMLXModelDescription.contextLength(inDirectory: directory),
                                    releaseSampling: NFKMLXReleaseSampling(directoryURL: directory),
                                    adapterStrengths: try holdAdapters(adapters, in: gemma), releaseDirectoryURL: directory)
    }

    /// Holds released LoRA adapters beside `gemma`'s decoder layers, unmerged, and returns their strengths in
    /// load order (``NFKMLXDecoderAdapters``, each layer read through ``NFKMLXGemma3/adapterModulePath(_:)``).
    /// transformers keeps key and value projections on the layers that read an earlier layer's keys and values
    /// and computes nothing from them; the port leaves them out, and an adapter's halves for them are left out
    /// with them. An adapter of the vision or audio tower is refused. Introduced in InferKit 0.4.0.
    static func holdAdapters(_ adapters: [(url: URL, strength: Float)], in gemma: NFKMLXGemma3n) throws -> [Float] {
        guard !adapters.isEmpty else { return [] }
        let decoder = gemma.model.decoder
        return try NFKMLXDecoderAdapters.hold(adapters, in: decoder, modulePath: NFKMLXGemma3.adapterModulePath) { path in
            let parts = path.split(separator: ".")
            guard parts.count == 4, parts[0] == "layers", parts[2] == "self_attn", parts[3] == "k_proj" || parts[3] == "v_proj",
                  let layer = Int(parts[1]), decoder.layers.indices.contains(layer) else { return false }
            return decoder.layers[layer].attention.sharesKeyValues
        }
    }

    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: .float32)
    }

    static let requiredFiles = ["config.json", "tokenizer.json"]
    static let optionalFiles = [String]()
    static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// The release directory a download of `repo` fills.
    static func releaseDirectory(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> URL {
        try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles)
    }

    /// Downloads a Gemma 3n release and loads it.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer.json`, and every shard the
    /// release's `model.safetensors.index.json` names into the hub cache under `cacheDirectoryURL`,
    /// or the default cache when nil. A cached file is not fetched again. The call blocks on the
    /// network; call it off the render thread. It serves the E2B and E4B releases. The
    /// `google/gemma-3n-*` repos are gated: the caller sets `NFKHFHub.defaultAccessToken` before the
    /// first download, or names the ungated mirrors `unsloth/gemma-3n-E2B-it` and
    /// `unsloth/gemma-3n-E4B-it`. Introduced in InferKit 0.4.0.
    @objc(gemma3nWithRepo:revision:cacheDirectoryURL:error:)
    public static func gemma3n(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXGemma3n {
        try NFKMLXBuildHolds.keeping {
            try load(directoryURL: try releaseDirectory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL))
        }
    }

    /// The asynchronous form of ``gemma3n(repo:revision:cacheDirectoryURL:)``, delivered on a
    /// background queue. Introduced in InferKit 0.4.0.
    @objc(gemma3nWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func gemma3n(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXGemma3n?, Error?) -> Void) {
        NFKMLXReleaseDownload.asyncValue(completionHandler) {
            try gemma3n(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Downloads a Gemma 3n release, as ``gemma3n(repo:revision:cacheDirectoryURL:)`` describes, and
    /// builds the backend. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try backend(directoryURL: try releaseDirectory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL))
        }
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

    /// Whether a release directory's `config.json` names Gemma 3n.
    public static func isGemma3n(configURL: URL) -> Bool {
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        let outer = (json["model_type"] as? String) ?? ""
        return outer == "gemma3n" || outer == "gemma3n_text"
    }
}
