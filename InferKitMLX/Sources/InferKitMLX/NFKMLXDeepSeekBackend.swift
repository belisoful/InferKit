//
//  NFKMLXDeepSeekBackend.swift
//  InferKitMLX
//
//  Text generation over the DeepSeek V4 / V4.1 decoder, and the loader that builds one from a
//  released directory.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

public extension NFKMLXDeepSeekNet {

    /// Generates from `prompt`, one token at a time, through the decoder's own cache.
    ///
    /// @discussion The prompt runs as one chunk and each produced token as a chunk of one. The cache
    /// carries the five pieces of state a step cannot recompute — the sliding window, the shared
    /// compressed key-value, the index keys, a compressor's unfinished group, and the n-gram id
    /// history — all indexed by absolute position, which is what makes a step equal to the same
    /// position inside a longer prefill. `NFKMLXDeepSeekCache` states each one.
    ///
    /// - Parameter prompt: the prompt's token ids.
    /// - Parameter embeddings: `[1, prompt.count, hidden]`, the prompt's embeddings with each image
    ///   span already written in, or nil for a prompt of text alone.
    /// - Parameter images: `[1, prompt.count]`, true at every position inside an image span. The
    ///   router selects an image token's experts with a second bias, so a span that is embedded and
    ///   not marked reaches the wrong experts.
    /// - Parameter options: the sampling, the token budget, the stop tokens, and the prefill chunk size.
    /// - Parameter onToken: called with each token as it is produced; returning false stops the run.
    /// - Returns: the produced tokens, not including the prompt.
    func generate(prompt: [Int], embeddings: MLXArray? = nil, images: MLXArray? = nil,
                  options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                  onToken: ((Int) -> Bool)? = nil) -> [Int] {
        generate(prompt: prompt, embeddings: embeddings, images: images, options: options,
                 promptCache: nil, onToken: onToken)
    }

    /// ``generate(prompt:embeddings:images:options:onToken:)`` continuing a conversation's kept
    /// prefill where the prompt extends it, and leaving `promptCache` holding this prompt's prefill.
    /// A prompt with pictures empties it and prefills from the start.
    internal func generate(prompt: [Int], embeddings: MLXArray?, images: MLXArray?,
                           options: NFKMLXGenerationOptions, promptCache: NFKMLXDeepSeekPromptCache?,
                           onToken: ((Int) -> Bool)?) -> [Int] {
        guard !prompt.isEmpty else { return [] }
        if let seed = options.seed { MLXRandom.seed(seed) }

        if embeddings != nil {
            promptCache?.reset()
        }
        let continued = embeddings == nil ? promptCache : nil
        let (cache, start) = continued?.begin(prompt) ?? (NFKMLXDeepSeekCache(configuration), 0)
        // Only the prompt carries pictures. A token the model produces is text, so every step after
        // this one embeds its own id and marks nothing, which is what holds images to the prompt.
        var logits = prefill(Array(prompt[start...]), embeddings: embeddings, images: images, cache: cache,
                             chunkSize: options.prefillChunkSize)
        eval(logits)
        continued?.finish(prompt, cache: cache)
        var produced = [Int]()
        let cursor = options.constraint?.makeCursor()

        for _ in 0 ..< options.maxTokens {
            var row = logits[0, -1]
            if let cursor {
                row = row + cursor.allowedTokenMask().asType(row.dtype)
            }
            let next = NFKMLXLanguageNet.sample(row, options: options)
            if options.stopTokens.contains(next) || (cursor != nil && next == cursor?.endToken) {
                break
            }
            produced.append(next)
            cursor?.accept(next)
            if let onToken, !onToken(next) { break }
            logits = self(MLXArray([Int32(next)]).reshaped([1, 1]), cache: cache)
            eval(logits)
        }
        return produced
    }

    /// Fills the cache with `prompt` and returns the logits after its last token: one forward pass,
    /// or several bounded ones when a chunk size is set.
    ///
    /// @discussion A long prompt run as a single forward builds its whole attention at once, and on
    /// this decoder that peak is the largest one: the sliding window, the shared compressed
    /// key-value, the index keys and the candidate blocks are each built over the chunk in hand.
    /// Chunking bounds the peak by the chunk rather than by the prompt.
    ///
    /// It is exact rather than an approximation, and here that is a stronger claim than on a dense
    /// stack. A boundary can fall inside a ratio-2 compressor's group, and the partial group the
    /// cache parks is what carries that group across it. A boundary also splits the n-gram
    /// look-back, which the id history carries. Every piece of state the cache holds is indexed by
    /// ABSOLUTE position, which is what makes a chunk equal to the same positions inside a longer
    /// one.
    ///
    /// A chunk that cannot finish a compressed group is not a smaller version of this: it emits no
    /// compressed position at all, and its queries attend over a cache a single pass would already
    /// have filled. `chunkSize` is therefore raised to
    /// `NFKMLXDeepSeekConfiguration.minimumPrefillChunk`, which is a bound on memory being
    /// honored as closely as the arithmetic allows rather than a silent approximation.
    ///
    /// Exact describes the mathematics, not the last bit. A chunk of five queries and a chunk of
    /// thirteen reach the same result through differently shaped matrix multiplies, and the engram's
    /// gate is discontinuous at a zero dot product, so a position sitting within float noise of zero
    /// can land on either side of the sign and move its gate by about 5e-4.
    /// `testAChunkedPrefillMatchesASinglePass` holds the agreement to that, at chunk sizes dividing
    /// neither the prompt nor the compression ratio, and holds the chosen token exactly.
    func prefill(_ prompt: [Int], embeddings: MLXArray? = nil, images: MLXArray? = nil,
                        cache: NFKMLXDeepSeekCache, chunkSize: Int?) -> MLXArray {
        guard let chunkSize, chunkSize > 0 else {
            return self(MLXArray(prompt.map(Int32.init)).reshaped([1, prompt.count]),
                        images: images, embeddings: embeddings, cache: cache)
        }
        // Raised to what a compressor needs. A chunk that finishes no group emits no compressed
        // position, and its queries then attend over a compressed cache that a single pass would
        // already have filled, which is a different computation rather than a smaller one. The
        // chunk size is a bound on memory, so raising it to the smallest that keeps the answer is
        // the reading that serves a caller asking for one.
        let size = Swift.max(chunkSize, configuration.minimumPrefillChunk)
        guard prompt.count > size else {
            return self(MLXArray(prompt.map(Int32.init)).reshaped([1, prompt.count]),
                        images: images, embeddings: embeddings, cache: cache)
        }
        var logits = MLXArray(0)
        var start = 0
        while start < prompt.count {
            let end = Swift.min(start + size, prompt.count)
            let slice = prompt[start ..< end].map(Int32.init)
            // The spans were written into the whole prompt before the first chunk ran, so a chunk
            // takes its slice of them. The reference instead requires every picture to lie in the
            // first chunk, because that is where it runs its tower.
            logits = self(MLXArray(slice).reshaped([1, slice.count]),
                          images: images.map { $0[0..., start ..< end] },
                          embeddings: embeddings.map { $0[0..., start ..< end] },
                          cache: cache)
            eval(logits)                            // free a chunk's activations before the next
            start = end

        }
        return logits
    }
}

/// Carries the decoder and tokenizer across the concurrency boundary the protocol runs behind.
final class NFKDeepSeekBackendHolder: @unchecked Sendable {
    let net: NFKMLXDeepSeekNet
    let tokenizer: NFKTokenizer
    init(net: NFKMLXDeepSeekNet, tokenizer: NFKTokenizer) {
        self.net = net
        self.tokenizer = tokenizer
    }
}

/// A text-generation backend over the DeepSeek V4 / V4.1 decoder.
///
/// @discussion Generation is incremental: the prompt is prefilled once and each step reads one
/// token through the cache. Inference is synchronous and multi-second, so a caller runs it off the
/// main thread and prefers `submitInferenceJobForRequest:`.
///
/// A request that names its conversation under `NFKParameterConversationKey` keeps the prefill of
/// its prompt, and the conversation's next prompt prefills only what it adds where both the old
/// prompt and the addition reach the configuration's largest compression ratio, the shortest chunk
/// its chunked prefill holds exact. A run with pictures, or one the draft stack decodes, prefills
/// from the start.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXDeepSeekBackend)
public final class NFKMLXDeepSeekBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKDeepSeekBackendHolder
    private let identifier: String
    private let defaults: NFKMLXGenerationOptions
    /// The vocabulary's bytes, read once and kept for every constrained request.
    private var vocabulary: NFKMLXVocabulary?
    private let generationLock = NSLock()
    /// The named conversations' kept prefills, read and written under `generationLock`.
    private let conversations = NFKMLXConversationKeeper<NFKMLXDeepSeekPromptCache>(byteBudget: 2 << 30)
    /// The tower, aligner, delimiters and preprocessor, where the release carries them.
    private let images: NFKMLXDeepSeekImageStack?
    /// The release's own draft stack, where it carries one.
    private let draft: NFKMLXDeepSeekDraftStack?
    private let releaseDirectoryURL: URL?

    init(net: NFKMLXDeepSeekNet, tokenizer: NFKTokenizer, identifier: String,
         images: NFKMLXDeepSeekImageStack? = nil, draft: NFKMLXDeepSeekDraftStack? = nil,
         options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(), releaseDirectoryURL: URL? = nil) {
        holder = NFKDeepSeekBackendHolder(net: net, tokenizer: tokenizer)
        self.identifier = identifier
        self.releaseDirectoryURL = releaseDirectoryURL
        self.images = images
        self.draft = draft
        defaults = options
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    private let modelInfoCache = NFKMLXModelInfoCache()

    /// The decoder's parameter count, weight bytes, precision, and quantization, with the bytes its
    /// release directory occupies on disk and the `model_type` and `max_position_embeddings` its
    /// `config.json` states (`NFKModelInfo*` keys). A paged decoder's routed experts count toward its
    /// parameters; its weight bytes are what memory holds, which leaves out experts read from the
    /// release. Introduced in InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            NFKMLXModelDescription.info(of: [holder.net], releaseDirectoryURL: releaseDirectoryURL)
        }
    }

    /// The request parameters the backend reads.
    ///
    /// @discussion Two of the language backend's keys are absent because they describe mechanisms
    /// this decoder supplies itself. Its attention is natively windowed and its compressed cache is
    /// the architecture's own long-range memory, so a context window and a quantized key-value cache
    /// would be a second, conflicting policy over state the model already bounds.
    ///
    /// `draftTokens` is read as a switch rather than a count, and only where the release carries a
    /// draft stack: how many tokens a round proposes is `dsparkBlockSize`, which is the
    /// architecture's rather than the caller's. Any value above zero turns speculation on. It
    /// applies at temperature zero, where a kept proposal is this decoder's own argmax and the
    /// output is the sequence plain decoding gives; above zero the run falls back, because the
    /// rejection scheme needs a probability the stack's Markov walk does not expose.
    @objc public var supportedParameterKeys: Set<String> {
        var keys: Set<String> = [
            NFKParameterTemperature, NFKParameterTopP, NFKParameterMaxTokens, NFKParameterSeed,
            NFKParameterJSONSchema, NFKMLXGenerationParameterKey.chatTemplate,
            NFKMLXGenerationParameterKey.prefillChunkSize, NFKParameterConversationKey,
        ]
        if draft != nil { keys.insert(NFKMLXGenerationParameterKey.draftTokens) }
        return keys
    }

    /// The bytes the conversations' prompt caches may hold together.
    ///
    /// @discussion A request that names its conversation under `NFKParameterConversationKey`
    /// continues from the prefill of that conversation's last prompt, so the turns of several chats
    /// served at once each prefill only what they add. When the caches outgrow the budget, the least
    /// recently used conversations are dropped, never the one a request is running. Lowering the
    /// budget drops down to it at once. Defaults to 2 GiB. Introduced in InferKit 0.4.0.
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

    /// The request inputs the backend reads.
    ///
    /// @discussion `NFKInputImage` appears only where the release carries an image tower. A release
    /// without one has no aligner to pool a picture into the decoder's width and no second router
    /// bias to route it with, so accepting an image there would be accepting something it could
    /// only ignore.
    @objc public var supportedInputKeys: Set<String> {
        images == nil ? [NFKInputPrompt, NFKInputMessages]
            : [NFKInputPrompt, NFKInputMessages, NFKInputImage]
    }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        var options = defaults
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
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.chatTemplate)
            as? String {
            options.chatTemplate = .jinja(template: value)
        }
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.prefillChunkSize)
            as? NSNumber {
            options.prefillChunkSize = value.intValue
        }
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.draftTokens)
            as? NSNumber {
            options.draftTokens = value.intValue
        }
        if let schema = request.parameter(forKey: NFKParameterJSONSchema) {
            guard let dictionary = schema as? [String: Any] else {
                throw NFKMLXError.unsupportedConfiguration(
                    "NFKParameterJSONSchema is a JSON Schema object")
            }
            options.jsonSchema = try NFKMLXJSONSchema(json: dictionary)
        }
        if options.stopTokens.isEmpty, holder.tokenizer.eosTokenId >= 0 {
            options.stopTokens = [holder.tokenizer.eosTokenId]
        }

        guard let text = NFKMLXLanguageBackend.prompt(from: request,
                                                      template: options.chatTemplate) else {
            throw NFKMLXError.unsupportedInput
        }
        var promptTokens = holder.tokenizer.encode(text).map(\.intValue)
        var embeddings: MLXArray?
        var spanMask: MLXArray?
        if let value = request.input(forKey: NFKInputImage) {
            guard let images else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(identifier) was built from a release with no image tower")
            }
            let picture = try NFKMLXImageBridge.rgbBytes(from: value,
                                                         colorSpace: CGColorSpaceCreateDeviceRGB())
            // A caller writing plain text has no reason to know how the release spells its image
            // placeholder, so one picture and no placeholder puts the picture first. More than one
            // picture means the prompt says where they go, and the count has to agree.
            if !promptTokens.contains(images.configuration.imageTokenID) {
                promptTokens.insert(images.configuration.imageTokenID, at: 0)
            }
            let expanded = try images.expanded(
                promptTokens,
                pictures: [(picture.bytes, picture.width, picture.height)])
            let inputs = images.inputs(for: expanded.tokens, spans: expanded.spans) {
                holder.net.embed($0)
            }
            promptTokens = expanded.tokens
            embeddings = inputs.embeddings
            spanMask = inputs.images
        }
        let (produced, cached) = generate(promptTokens, embeddings: embeddings, images: spanMask,
                                          options: options, conversation: NFKMLXConversation.name(of: request))
        return NFKInferenceResult(outputs: [
            NFKOutputText: holder.tokenizer.decode(produced.map(NSNumber.init(value:))),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: cached,
                                                outputTokens: produced.count, reasoningTokens: nil),
        ])
    }

    /// One generation at a time. A run threads its own cache through the shared decoder, so two
    /// interleaving their steps would read each other's positions.
    private func generate(_ tokens: [Int], embeddings: MLXArray? = nil, images: MLXArray? = nil,
                          options: NFKMLXGenerationOptions, conversation: String?) -> (produced: [Int], cached: Int) {
        generationLock.lock(); defer { generationLock.unlock() }
        var resolved = options
        resolved.constraint = constraint(for: options)
        // A constrained run stays plain: a proposal the grammar forbids would be verified against
        // logits the mask never reached, so the two mechanisms would disagree about what is legal.
        if let draft, resolved.draftTokens > 0, resolved.temperature == 0,
           resolved.constraint == nil {
            return (holder.net.generate(prompt: tokens, draft: draft, embeddings: embeddings,
                                        images: images, options: resolved), 0)
        }
        let configuration = holder.net.configuration
        let promptCache = conversation.map { name in
            conversations.states.state(for: name) { NFKMLXDeepSeekPromptCache(configuration: configuration) }
        }
        defer {
            if let conversation {
                conversations.states.evict(keeping: conversation)
            }
            conversations.refresh()
        }
        let produced = holder.net.generate(prompt: tokens, embeddings: embeddings, images: images,
                                           options: resolved, promptCache: promptCache, onToken: nil)
        return (produced, promptCache?.sharedPrefixLength ?? 0)
    }

    /// The constraint a request asks for: a schema set on the options, or none.
    private func constraint(for options: NFKMLXGenerationOptions) -> (any NFKMLXTokenConstraint)? {
        if let custom = options.constraint { return custom }
        guard let schema = options.jsonSchema else { return nil }
        let kept = vocabulary ?? NFKMLXVocabulary(tokenizer: holder.tokenizer,
                                                  size: holder.net.configuration.vocabularySize)
        vocabulary = kept
        return NFKMLXJSONSchemaConstraint(schema: schema, vocabulary: kept)
    }
}

public extension NFKMLXDeepSeekNet {

    /// Generates with the release's own draft stack proposing a block of tokens per round, which
    /// this decoder verifies in a single pass.
    ///
    /// @discussion Decoding is bound by memory traffic: a step reads every weight whether it scores
    /// one position or six, so scoring a proposed block costs about what scoring one token costs.
    /// The draft stack proposes `dsparkBlockSize` tokens, this decoder scores the block in one
    /// forward, and the leading proposals that match its own argmax are kept. The output is the
    /// SAME sequence plain decoding produces, token for token, which is what the test asserts.
    ///
    /// **Why this was the hard one.** Rejecting a proposal means putting the cache back, and this
    /// cache cannot be trimmed: the sliding window is a ring, so appending a block evicted the
    /// oldest positions and dropping the tail does not bring them back. The five buffers a step
    /// carries are each REPLACED on an append rather than written through, so a snapshot is a
    /// handful of reference copies and a restore is exact. A round that rejects therefore costs one
    /// extra forward over the accepted prefix, which is still fewer passes than stepping through it
    /// a token at a time whenever two or more proposals are kept.
    ///
    /// Verification is greedy, and a caller asking for temperature above zero gets plain decoding
    /// instead. The rejection scheme that keeps a sampled run distributed correctly needs the
    /// draft's own probability at each proposed token; the stack's Markov walk biases each position
    /// by the token chosen before it, so that probability is not the one its returned logits carry.
    ///
    /// - Parameter prompt: the prompt's token ids.
    /// - Parameter draft: the release's draft stack, which shares this decoder's embedding and head.
    /// - Parameter embeddings: `[1, prompt.count, hidden]` with each image span written in, or nil for
    ///   text alone, as in ``generate(prompt:embeddings:images:options:onToken:)``.
    /// - Parameter images: `[1, prompt.count]`, true inside each image span, or nil.
    /// - Parameter options: the sampling, the token budget, and the stop tokens; a temperature above 0
    ///   runs plain decoding.
    /// - Parameter report: receives the rounds run and the proposals kept.
    /// - Parameter onToken: called with each token as it is produced; returning false stops the run.
    /// - Returns: the produced tokens, not including the prompt.
    func generate(prompt: [Int], draft: NFKMLXDeepSeekDraftStack,
                  embeddings: MLXArray? = nil, images: MLXArray? = nil,
                  options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                  report: inout NFKMLXSpeculativeReport,
                  onToken: ((Int) -> Bool)? = nil) -> [Int] {
        guard !prompt.isEmpty, options.maxTokens > 0 else { return [] }
        guard options.temperature == 0 else {
            return generate(prompt: prompt, embeddings: embeddings, images: images,
                            options: options, onToken: onToken)
        }
        if let seed = options.seed { MLXRandom.seed(seed) }

        let cache = NFKMLXDeepSeekCache(configuration)
        cache.collectsDraftStates = true
        // A picture lies in the prompt, so it is prefilled here and nothing after this point
        // carries one: the drafted and verified positions are text.
        let promptLogits = prefill(prompt, embeddings: embeddings, images: images, cache: cache,
                                   chunkSize: options.prefillChunkSize)
        eval(promptLogits)

        var produced = [Int]()
        func emit(_ token: Int) -> Bool {
            guard produced.count < options.maxTokens, !options.stopTokens.contains(token) else {
                return false
            }
            produced.append(token)
            return onToken?(token) ?? true
        }

        var next = NFKMLXLanguageNet.sample(promptLogits[0, -1], options: options)
        guard emit(next) else { return produced }

        while produced.count < options.maxTokens {
            guard let mainStates = cache.draftStates else { break }
            let proposal = draft.propose(continuing: MLXArray([Int32(next)]),
                                         mainStates: mainStates, through: self)
            eval(proposal.tokens)
            // The stack returns the committed token followed by its proposals.
            let proposals = proposal.tokens[0].asArray(Int32.self).dropFirst().map { Int($0) }
            guard !proposals.isEmpty else { break }

            // Row j predicts what follows input j, so row j judges proposal j and the last row
            // gives the token after the whole block.
            let saved = cache.snapshot()
            let batch = [next] + proposals
            let rows = self(MLXArray(batch.map(Int32.init)).reshaped([1, batch.count]),
                            cache: cache)[0]
            eval(rows)
            let (accepted, following) = NFKMLXLanguageNet.verifyGreedily(rows: rows,
                                                                        proposals: proposals)
            report.rounds += 1
            report.proposed += proposals.count
            report.accepted += accepted

            // A block accepted whole is already committed by the pass that verified it. Anything
            // less is put back and the kept prefix run again, because the ring cannot be trimmed.
            if accepted < proposals.count {
                cache.restore(saved)
                let kept = [next] + proposals.prefix(accepted)
                eval(self(MLXArray(kept.map(Int32.init)).reshaped([1, kept.count]), cache: cache))
            }

            for token in proposals.prefix(accepted) {
                guard emit(token) else { return produced }
            }
            guard emit(following) else { return produced }
            next = following
        }
        return produced
    }

    /// The same run without a report.
    func generate(prompt: [Int], draft: NFKMLXDeepSeekDraftStack,
                  embeddings: MLXArray? = nil, images: MLXArray? = nil,
                  options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                  onToken: ((Int) -> Bool)? = nil) -> [Int] {
        var report = NFKMLXSpeculativeReport()
        return generate(prompt: prompt, draft: draft, embeddings: embeddings, images: images,
                        options: options, report: &report, onToken: onToken)
    }
}
