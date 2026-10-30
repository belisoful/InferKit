//
//  NFKMLXLanguageBackend.swift
//  InferKitMLX
//
//  Text generation over `NFKMLXLanguageNet`, and the InferKit backend around it.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The request-parameter keys for the MLX generation options that have no core key, so an
/// Objective-C caller configures them per request the same way it sets `NFKParameterTemperature`.
///
/// @discussion The core keys (`NFKParameterTemperature` / `TopP` / `MaxTokens` / `Seed`) cover
/// sampling; these cover the MLX backend's own generation options — the cache bound and quantization,
/// prefill chunking, and the chat template. Each overrides the backend's build-time default for that
/// one request, in Swift and Objective-C alike.
@objc(NFKMLXGenerationParameterKey)
public final class NFKMLXGenerationParameterKey: NSObject {
    /// The most cache positions to retain, as an `NSNumber` integer. See ``NFKMLXGenerationOptions/contextWindow``.
    @objc public static let contextWindow = "NFKMLXParameterContextWindow"
    /// Whether a speech recognizer writes its transcript's timed spans under `NFKOutputSegments`
    /// (`NSNumber` boolean). Whisper reads it; its default is the backend's ``NFKMLXWhisperBackend/emitsTimestamps``.
    /// Introduced in InferKit 0.4.0.
    @objc public static let timestamps = "NFKMLXParameterTimestamps"
    /// What a speech recognizer makes of the speech (`NSString`): "transcribe" for text in the language
    /// spoken, or "translate" for English text. Whisper reads it. Introduced in InferKit 0.4.0.
    @objc public static let speechTask = "NFKMLXParameterSpeechTask"
    /// The bit width to store the key-value cache at (`NSNumber`, 2 to 8). Setting it enables cache
    /// quantization; see ``NFKMLXGenerationOptions/cacheQuantization``.
    @objc public static let cacheQuantizationBits = "NFKMLXParameterCacheQuantizationBits"
    /// The cache quantization group size (`NSNumber`, default 64), which must divide the head dimension.
    @objc public static let cacheQuantizationGroupSize = "NFKMLXParameterCacheQuantizationGroupSize"
    /// Whether the cache groups each KEY channel along the sequence (`NSNumber` boolean, default
    /// true, the measured better layout) rather than each position along the head dimension. See
    /// ``NFKMLXKeyValueCache/Quantization/KeyAxis``.
    @objc public static let cacheQuantizationPerChannelKeys = "NFKMLXParameterCacheQuantizationPerChannelKeys"
    /// The most prompt tokens to run per prefill pass, as an `NSNumber`. See ``NFKMLXGenerationOptions/prefillChunkSize``.
    @objc public static let prefillChunkSize = "NFKMLXParameterPrefillChunkSize"
    /// The chat template, as an `NSString`: `"chatml"` applies the ChatML template, a string carrying
    /// Jinja delimiters (`{%`/`{{`) is rendered as the release's own `chat_template`, and anything else
    /// flattens the messages. See ``NFKMLXGenerationOptions/chatTemplate``.
    @objc public static let chatTemplate = "NFKMLXParameterChatTemplate"
    /// How many tokens a draft model proposes per verification round, as an `NSNumber`; 0 turns
    /// speculative decoding off for the request even when the backend holds a draft. See
    /// ``NFKMLXGenerationOptions/draftTokens``.
    @objc public static let draftTokens = "NFKMLXParameterDraftTokens"
    /// Whether the backend keeps its key-value cache between requests and prefills only what a new
    /// prompt adds to the last one, as an `NSNumber` boolean. A request that names its conversation
    /// under `NFKParameterConversationKey` reuses that conversation's cache unless this is NO. See
    /// ``NFKMLXGenerationOptions/reusesPromptCache``.
    @objc public static let reusesPromptCache = "NFKMLXParameterReusesPromptCache"
    /// The output's required shape, as an `NSString`: `"json"` constrains sampling to well-formed
    /// JSON with an object or array root, `"json-object"` to an object, `"json-array"` to an array.
    /// Anything else leaves the output free. See ``NFKMLXGenerationOptions/jsonOutput`` and
    /// ``NFKMLXJSONConstraint``.
    @objc public static let outputFormat = "NFKMLXParameterOutputFormat"
    /// The strings the output must be one of, as an `NSArray<NSString *>`. See
    /// ``NFKMLXGenerationOptions/choices`` and ``NFKMLXChoiceConstraint``.
    @objc public static let choices = "NFKMLXParameterChoices"
    /// Which markers the model wraps its reasoning in, as an `NSString` naming one of `"think"`,
    /// `"harmony"`, or `"none"`, or as an `NSArray<NSString *>` of the markers themselves (opening,
    /// closing, and optionally what stands before the answer). Without the key the format comes
    /// from the release's own chat template. See ``NFKMLXReasoningFormat``.
    @objc public static let reasoningFormat = "NFKMLXParameterReasoningFormat"
    /// The syntax the model writes a tool call in, as an `NSString` naming an ``NFKMLXToolCallFormat``
    /// (`"hermes"`, `"qwen-xml"`, `"gemma"`, `"harmony"`, `"deepseek"`, `"mistral"`, `"granite"`,
    /// `"llama"`) or `"none"`. Without the key the format comes from the release's own chat template.
    /// Introduced in InferKit 0.4.0.
    @objc public static let toolCallFormat = "NFKMLXParameterToolCallFormat"
}

/// The template that turns a message list into prompt text.
public enum NFKMLXChatTemplate: Sendable {
    /// Flatten each message to its content, joined by newlines — what a base model reads.
    case none
    /// The `<|im_start|>role\n…<|im_end|>\n` template the Qwen and Llama instruct families use, with a
    /// trailing `<|im_start|>assistant\n` so the model continues as the assistant. The markers are the
    /// release's own special tokens, which the tokenizer resolves to single ids.
    case chatML
    /// The release's own Jinja `chat_template`, rendered by ``NFKMLXChatTemplateRenderer``. This is the
    /// faithful path: an instruct release ships the exact template it was trained on, so rendering it
    /// reproduces the model's expected input rather than approximating it with `.chatML`. The
    /// associated values are the release's markers, for templates that reference `bos_token`/`eos_token`.
    case jinja(template: String, bosToken: String = "", eosToken: String = "")
}

/// The sampling a release's `generation_config.json` sets, read as transformers' `GenerationConfig` reads
/// it: a file without `do_sample` decodes greedily, and a value a sampling file leaves out takes
/// transformers' default (temperature 1, top-p 1, top-k 50). These values are step 3 of the sampling order
/// ``NFKMLXGenerationOptions`` states.
struct NFKMLXReleaseSampling: Equatable, Sendable {
    var temperature: Float
    var topP: Float
    var topK: Int

    /// The sampling the configuration sets, or nil where it does not sample.
    init?(generationConfig json: [String: Any]) {
        guard (json["do_sample"] as? Bool) == true else { return nil }
        temperature = (json["temperature"] as? NSNumber)?.floatValue ?? 1
        topP = (json["top_p"] as? NSNumber)?.floatValue ?? 1
        topK = (json["top_k"] as? NSNumber)?.intValue ?? 50
    }

    /// The sampling the `generation_config.json` in `directoryURL` sets, or nil without a file that samples.
    init?(directoryURL: URL?) {
        guard let url = directoryURL?.appendingPathComponent("generation_config.json"),
              let data = try? Data(contentsOf: url),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        self.init(generationConfig: json)
    }

    init(temperature: Float, topP: Float, topK: Int) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
    }
}

/// How a generation run samples and how its prompt is prepared.
///
/// @discussion A run's temperature, top-p, and top-k each come from the first of these sources that sets
/// the value:
///
/// 1. The request's `NFKParameterTemperature`, `NFKParameterTopP`, or `NFKParameterTopK` → that value.
/// 2. A value set on the options the backend is built with → that value. A set value counts even when it
///    equals the default, so a backend built with `temperature = 0` decodes greedily.
/// 3. The release's `generation_config.json`, where it sets `do_sample` → its value, or transformers'
///    default where the file leaves it out (temperature 1, top-p 1, top-k 50).
/// 4. Otherwise → the library default: temperature 0, top-p 1, top-k 0.
///
/// ``usesReleaseSampling`` set to false skips step 3. A backend that takes no build options skips step 2.
/// Objective-C cannot build these options, so an Objective-C caller sets sampling through step 1.
public struct NFKMLXGenerationOptions: Sendable {
    /// The most tokens to produce, not counting the prompt. Defaults to 256 for a direct `generate` call.
    /// A backend runs a request whose options and parameters leave it unset to the room its context
    /// leaves. A set value counts even at 256.
    public var maxTokens: Int {
        get { chosenMaxTokens ?? 256 }
        set { chosenMaxTokens = newValue }
    }
    /// The token limit a caller set, or nil for a backend's room in the context.
    var chosenMaxTokens: Int?
    /// 0 samples greedily; above 0 divides the logits before sampling. Defaults to 0.
    public var temperature: Float {
        get { chosenTemperature ?? 0 }
        set { chosenTemperature = newValue }
    }
    /// Nucleus sampling: the smallest set of tokens whose probability sums past this is sampled from.
    /// Ignored at temperature 0. Defaults to 1.
    public var topP: Float {
        get { chosenTopP ?? 1 }
        set { chosenTopP = newValue }
    }
    /// Top-k sampling: only the `topK` most probable tokens, with any tied with the last of them, are
    /// sampled from, before the nucleus filter. 0 keeps every token. Ignored at temperature 0. Defaults
    /// to 0. Introduced in InferKit 0.4.0.
    public var topK: Int {
        get { chosenTopK ?? 0 }
        set { chosenTopK = newValue }
    }
    /// The sampling values a caller set: step 2 of the sampling order.
    private var chosenTemperature: Float?
    private var chosenTopP: Float?
    private var chosenTopK: Int?
    /// Whether a release's `generation_config.json` supplies the temperature, top-p, and top-k these
    /// options leave unset, as the reference's `generate` does: step 3 of the sampling order. Defaults to
    /// true. Introduced in InferKit 0.4.0.
    public var usesReleaseSampling = true

    /// The options with `release`'s temperature, top-p, and top-k in each of the three this struct leaves
    /// unset, or as they are when ``usesReleaseSampling`` is false.
    func sampling(from release: NFKMLXReleaseSampling?) -> NFKMLXGenerationOptions {
        guard usesReleaseSampling, let release else { return self }
        var options = self
        options.chosenTemperature = chosenTemperature ?? release.temperature
        options.chosenTopP = chosenTopP ?? release.topP
        options.chosenTopK = chosenTopK ?? release.topK
        return options
    }
    /// A seed makes a sampled run repeatable.
    public var seed: UInt64?
    /// Generation stops when one of these is produced. Empty, a backend over a release stops on the
    /// tokenizer's end-of-sequence token and every id the release's `generation_config.json` lists.
    public var stopTokens: Set<Int> = []
    /// The most cached positions to retain, or `nil` to retain every one.
    ///
    /// @discussion A cache that keeps everything grows with the conversation, and past a certain
    /// length that is what ends the run. A window bounds it by dropping the oldest positions.
    ///
    /// This changes what the model reads, so it is off by default. It costs nothing while the
    /// conversation is shorter than the window, and past that the model stops seeing its beginning —
    /// which for a model whose attention is not natively windowed is an approximation, not a
    /// configuration. Size it from what the machine can hold; ``NFKMLXGPU/recommendedWorkingSetSize``
    /// is the budget to divide.
    public var contextWindow: Int?

    /// How a message list is turned into prompt text.
    ///
    /// @discussion A base model reads plain text, so the default flattens a message list to its
    /// contents. An INSTRUCT release is trained on a chat template and is prompted outside its format
    /// without one — the roles and turn markers it learned are absent, and it answers a different
    /// question than it was asked. `.chatML` applies the `<|im_start|>role … <|im_end|>` template the
    /// Qwen and Llama instruct families use, with the release's own special tokens; it is off by
    /// default because a base model wants the plain text, and a raw `NFKInputPrompt` is always used
    /// verbatim whatever this says.
    public var chatTemplate: NFKMLXChatTemplate = .none
    /// Whether a message request renders through the release's own chat template when ``chatTemplate`` is
    /// `.none` and the request names none. Defaults to true; false flattens the messages as `.none` does.
    /// Introduced in InferKit 0.4.0.
    public var usesReleaseChatTemplate = true

    /// Stores the key-value cache quantized, or `nil` to keep it in full precision.
    ///
    /// @discussion A quantized cache holds each retained position in `bits` per element instead of a
    /// float, so a long conversation reaches much further before the cache, rather than the weights,
    /// becomes the memory ceiling. It is lossy — a step reads the dequantized span — so it is off by
    /// default; 8-bit tracks full precision closely while halving or better the cache's footprint. The
    /// group size must divide the model's head dimension (64 or 128 divide by the default 64). The
    /// keys are grouped per channel along the sequence by default, which is what holds their
    /// precision at 4 bits and makes 8 bits near-lossless; see ``NFKMLXKeyValueCache/Quantization/KeyAxis``.
    public var cacheQuantization: NFKMLXKeyValueCache.Quantization?

    /// Runs the prompt through the cache in slices of at most this many tokens, or `nil` for
    /// ``NFKMLXLanguageNet/defaultPrefillChunk``.
    ///
    /// @discussion A long prompt run as a single forward builds an attention over its whole length at
    /// once, whose peak is what a large context runs out of. Feeding the prompt in chunks makes each
    /// pass attend to its slice plus the cached prefix, so the peak is bounded by the chunk rather than
    /// the prompt. The cache makes a chunk attend to exactly the keys a single forward would, so the
    /// logits match. A pass is shortened further while its attention scores would pass
    /// ``NFKMLXLanguageNet/prefillScoreBudget``, so a long prompt stays within memory at any setting.
    public var prefillChunkSize: Int?

    /// How many tokens a draft model proposes per round when the backend was built with one.
    ///
    /// @discussion Speculative decoding scores a run of draft proposals in one pass of the large
    /// model and keeps the leading ones that agree, so the output is the plain run's output at a
    /// fraction of the passes. More proposals per round buy more when the draft agrees often and
    /// waste verification work when it does not; four is a reasonable default for a draft from the
    /// same family. 0 turns speculation off for a request even when a draft is present.
    public var draftTokens: Int = 4

    /// Whether the backend keeps its key-value cache between requests, so a prompt that extends the
    /// previous one prefills only its tail. Off by default; a request that names its conversation
    /// under `NFKParameterConversationKey` reuses that conversation's own cache without it. See
    /// ``NFKMLXPromptCache``.
    public var reusesPromptCache: Bool = false

    /// Constrains what the model may emit; nil leaves sampling free. See ``NFKMLXTokenConstraint``.
    ///
    /// @discussion The mask is applied to the logits before temperature and nucleus filtering, so
    /// sampling and greedy decoding alike stay inside the grammar. A constraint runs the plain
    /// loop: speculative decoding is skipped for the run, since a proposal's admissibility depends
    /// on the proposals before it. The backend builds the JSON and choice constraints from
    /// ``jsonOutput`` and ``choices``; this field takes a custom one.
    public var constraint: (any NFKMLXTokenConstraint)?

    /// Whether the backend constrains the output to well-formed JSON. See ``NFKMLXJSONConstraint``.
    public var jsonOutput: Bool = false

    /// What the JSON root may be when ``jsonOutput`` is set. A model asked for "a JSON object" and
    /// left free to open an array has been measured to answer `[]`; naming the root closes that.
    public var jsonRoot: NFKMLXJSONConstraint.Root = .container

    /// Strings the output must be one of, or nil. See ``NFKMLXChoiceConstraint``.
    public var choices: [String]?

    /// The markers the model wraps its reasoning in, or nil to read them from the chat template.
    ///
    /// @discussion Setting this reaches a model whose prompt the caller rendered itself, where there
    /// is no template to read the markers from. See ``NFKMLXReasoningFormat``.
    public var reasoningFormat: NFKMLXReasoningFormat?

    /// Whether ``reasoningFormat`` was set deliberately, so a nil format means "no chain to split"
    /// rather than "take it from the template".
    var reasoningFormatIsStated = false

    /// The syntax the model writes a tool call in, or nil to read it from the chat template. See
    /// ``NFKMLXToolCallFormat``. Introduced in InferKit 0.4.0.
    public var toolCallFormat: NFKMLXToolCallFormat?

    /// Whether ``toolCallFormat`` was set deliberately, so a nil format means "read no calls".
    var toolCallFormatIsStated = false

    /// A schema the JSON output must conform to, or nil. See ``NFKMLXJSONSchemaConstraint``.
    ///
    /// @discussion Where ``jsonOutput`` guarantees the syntax, a schema guarantees the keys, the
    /// types, the enumerations, and the array bounds as well. The backend fills it from the core's
    /// `NFKParameterJSONSchema` request parameter, the same key the remote backends read, so a
    /// caller's structured-output request is engine-agnostic; a schema the grammar cannot enforce is
    /// reported as an error rather than ignored.
    public var jsonSchema: NFKMLXJSONSchema?

    public init() {}
}

/// Holds the network for capture in a `@Sendable` body.
private final class NFKLMHolder: @unchecked Sendable {
    let net: NFKMLXLanguageNet
    let tokenizer: NFKTokenizer?
    let draft: NFKMLXLanguageNet?
    init(_ net: NFKMLXLanguageNet, _ tokenizer: NFKTokenizer?, draft: NFKMLXLanguageNet? = nil) {
        self.net = net
        self.tokenizer = tokenizer
        self.draft = draft
    }
}

public extension NFKMLXLanguageNet {

    /// Generates continuation tokens for `prompt`, calling `onToken` with each as it is produced.
    ///
    /// @discussion The prompt fills the cache through ``prefill(_:cache:chunkSize:)``, then each step
    /// feeds back a single token. Returning false from `onToken` stops the run, which is how a job's
    /// cancellation reaches generation.
    func generate(prompt: [Int], options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                  promptCache: NFKMLXPromptCache? = nil,
                  onToken: ((Int) -> Bool)? = nil) -> [Int] {
        guard !prompt.isEmpty else { return [] }
        if let seed = options.seed { MLXRandom.seed(seed) }

        let (cache, tail) = startingCache(for: prompt, options: options, promptCache: promptCache)
        var logits = prefill(tail, cache: cache, chunkSize: options.prefillChunkSize)
        promptCache?.record(tail)
        var produced = [Int]()
        // A streamed layer that could not be read leaves nothing to decode from, and neither does a
        // failed evaluation.
        guard !streamFailed, !NFKMLXErrorScope.hasFailed else { return produced }
        let cursor = options.constraint?.makeCursor()

        for _ in 0 ..< options.maxTokens {
            var row = logits[0, -1]
            if let cursor {
                row = row + cursor.allowedTokenMask().asType(row.dtype)
            }
            let next = NFKMLXLanguageNet.sample(row, options: options)
            // A failed evaluation samples as token 0.
            if NFKMLXErrorScope.hasFailed { break }
            if options.stopTokens.contains(next) || (cursor != nil && next == cursor?.endToken) { break }
            produced.append(next)
            cursor?.accept(next)
            if let onToken, !onToken(next) { break }
            logits = self(MLXArray([Int32(next)]).reshaped([1, 1]), cache: cache)
            promptCache?.record([next])
            if streamFailed { break }
        }
        return produced
    }

    /// The best continuation of `prompt` a `decoding.beams`-wide beam search finds, without the end token
    /// that closed it, searched as transformers' `generate(num_beams:)` searches a decoder-only model.
    ///
    /// @discussion The prompt runs once and its cache rows are copied to every beam; each step then runs
    /// the beams as one batch. `decoding` supplies the beams, `maxTokens`, the length penalty, early
    /// stopping, the end tokens, and the n-gram ban, whose window reaches into the prompt as the
    /// reference's does. The prompt takes the start token's place, so `startToken` is not read. Throws a
    /// failed read of a streamed layer. Introduced in InferKit 0.4.0.
    func beamSearch(prompt: [Int], decoding: NFKMLXSeq2SeqDecoding) throws -> [Int] {
        try beamHypothesis(prompt: prompt, decoding: decoding)?.tokens ?? []
    }

    /// ``beamSearch(prompt:decoding:)``'s best hypothesis with its score, or nil where none was scored.
    internal func beamHypothesis(prompt: [Int], decoding: NFKMLXSeq2SeqDecoding) throws -> NFKMLXSeq2SeqDecoder.Hypothesis? {
        guard !prompt.isEmpty, decoding.maxTokens > 0 else { return nil }
        var search = decoding
        search.beams = Swift.max(decoding.beams, 1)
        let cache = NFKMLXKeyValueCache(layerCount: configuration.layerCount)
        let promptLogits = self(MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count]), cache: cache)
        try verifyStream()
        let last = promptLogits[0..., (prompt.count - 1)..., 0...]
        cache.reorder(MLXArray.zeros([search.beams], dtype: .int32))
        return try NFKMLXSeq2SeqDecoder.beamSearch(prefix: prompt, logits: broadcast(last, to: [search.beams, 1, last.dim(-1)]),
                                                   decoding: search) { rows, tokens in
            cache.reorder(rows)
            let logits = self(tokens, cache: cache)
            try verifyStream()
            return logits
        }
    }

    /// The cache a run starts from and the part of `prompt` still to prefill into it: a fresh cache
    /// and the whole prompt, or a prompt cache rolled back to what it shares with `prompt` and the
    /// prompt's remainder.
    func startingCache(for prompt: [Int], options: NFKMLXGenerationOptions,
                       promptCache: NFKMLXPromptCache?) -> (NFKMLXKeyValueCache, [Int]) {
        guard let promptCache else {
            return (NFKMLXKeyValueCache(layerCount: configuration.layerCount,
                                        window: options.contextWindow,
                                        quantization: options.cacheQuantization), prompt)
        }
        let shared = promptCache.align(to: prompt)
        return (promptCache.cache, Array(prompt[shared...]))
    }

    /// Fills the cache with the prompt and returns the logits of the last pass, whose last row follows
    /// the prompt's last token.
    ///
    /// @discussion The prompt runs in passes of at most `chunkSize` tokens, or ``defaultPrefillChunk``
    /// where it is nil or not positive. Each pass attends through the cache to the prefix before it, so
    /// the logits are those of a single pass and only the peak differs. A pass is shortened further
    /// while its attention scores would pass ``prefillScoreBudget`` (``prefillPassLength(cached:remaining:limit:budget:)``):
    /// a reduced-precision layer's reference attention holds them whole, and a single pass over 23 000
    /// tokens on a 16-head decoder asks Metal for one 34 GB buffer. A failed evaluation ends the passes.
    func prefill(_ prompt: [Int], cache: NFKMLXKeyValueCache, chunkSize: Int?) -> MLXArray {
        let limit = chunkSize.flatMap { $0 > 0 ? $0 : nil } ?? NFKMLXLanguageNet.defaultPrefillChunk
        var start = 0
        var hidden: MLXArray
        repeat {
            let length = prefillPassLength(cached: cache.maskCacheLength, remaining: prompt.count - start, limit: limit)
            let slice = prompt[start ..< start + length].map { Int32($0) }
            hidden = hiddenStates(fromEmbeddings: embed(MLXArray(slice).reshaped([1, length])), cache: cache)
            start += length
            if start < prompt.count {
                eval(hidden)                                    // free a pass's activations before the next
            }
        } while start < prompt.count && !streamFailed && !NFKMLXErrorScope.hasFailed
        return logits(fromHidden: hidden)
    }

    /// The most prompt tokens one prefill pass runs when the options set no chunk size: mlx_lm's
    /// `prefill_step_size`. Introduced in InferKit 0.4.0.
    static let defaultPrefillChunk = 2048

    /// The most bytes one prefill pass's attention scores take, counted at four bytes a score: one per
    /// head, query, and key the pass sees. Introduced in InferKit 0.4.0.
    static let prefillScoreBudget = 1 << 30

    /// The tokens the next prefill pass runs: the smaller of `remaining` and `limit`, halved while the
    /// scores of its queries over `cached` earlier keys and its own would pass `budget` bytes, and at
    /// least one. Introduced in InferKit 0.4.0.
    func prefillPassLength(cached: Int, remaining: Int, limit: Int,
                           budget: Int = NFKMLXLanguageNet.prefillScoreBudget) -> Int {
        var length = Swift.min(remaining, limit)
        while length > 1 && configuration.headCount * length * (cached + length) * 4 > budget {
            length = (length + 1) / 2
        }
        return length
    }

    /// Picks the next token from a `[vocabulary]` row of logits.
    static func sample(_ logits: MLXArray, options: NFKMLXGenerationOptions) -> Int {
        guard options.temperature > 0 else {
            return logits.argMax().item(Int.self)
        }
        // `categorical` samples from logits, so the filtered distribution goes back through a log.
        return MLXRandom.categorical(log(probabilities(of: logits, options: options))).item(Int.self)
    }

    /// The distribution sampling draws from: the logits at the options' temperature, with the
    /// nucleus filter applied when one is set. Works on one `[vocabulary]` row or on
    /// `[rows, vocabulary]` at once, which speculative verification scores together.
    static func probabilities(of logits: MLXArray, options: NFKMLXGenerationOptions) -> MLXArray {
        let temperature = Swift.max(options.temperature, Float.leastNormalMagnitude)
        var distribution = softmax(logits / temperature, axis: -1)
        if options.topK > 0 && options.topK < distribution.dim(-1) {
            distribution = mostProbable(distribution, count: options.topK)
        }
        return options.topP < 1 ? nucleus(distribution, topP: options.topP) : distribution
    }

    /// Zeroes every token less probable than the `count`th most probable, keeping ties as the reference's
    /// top-k filter does, then renormalizes.
    private static func mostProbable(_ probabilities: MLXArray, count: Int) -> MLXArray {
        let threshold = sorted(probabilities, axis: -1).take(MLXArray([Int32(probabilities.dim(-1) - count)]), axis: -1)
        let filtered = MLX.where(probabilities .>= threshold, probabilities, MLXArray(Float(0)))
        return filtered / filtered.sum(axis: -1, keepDims: true)
    }

    /// Zeroes every token outside the smallest set whose probability passes `topP`, then renormalizes.
    private static func nucleus(_ probabilities: MLXArray, topP: Float) -> MLXArray {
        let order = argSort(probabilities, axis: -1)
        let sorted = takeAlong(probabilities, order, axis: -1)
        let cumulative = cumsum(sorted, axis: -1)
        // Keeping where the cumulative sum from the LOW end is under (1 - topP) discards the tail,
        // which is the same set the reference keeps from the high end.
        let keep = cumulative .> (1 - topP)
        let filtered = MLX.where(keep, sorted, MLXArray(Float(0)))
        // Undo the sort so the probabilities line up with their token ids again.
        let restored = takeAlong(filtered, argSort(order, axis: -1), axis: -1)
        return restored / restored.sum(axis: -1, keepDims: true)
    }
}

/// On-device text generation as an InferKit backend.
///
/// Reads `NFKInputPrompt` or `NFKInputMessages` and returns `NFKOutputText`. `NFKParameterTemperature`,
/// `NFKParameterTopP`, `NFKParameterTopK`, `NFKParameterMaxTokens`, and `NFKParameterSeed` override the
/// defaults. A release whose `generation_config.json` samples defaults to its temperature, top-p, and top-k.
/// A request that sets no `NFKParameterMaxTokens`, over options that set no `maxTokens`, runs to an end token
/// or to the room the context leaves.
///
/// Tools: `NFKParameterTools` renders into the release's chat template, and the calls the model
/// writes come back under `NFKOutputToolCalls`, read in the syntax ``NFKMLXToolCallFormat`` names,
/// with the text around them under `NFKOutputText`. A call whose arguments do not read is still a
/// call, with empty `arguments` and the model's text under `argumentsJSON`. A tool conversation in
/// the chat shape renders back through the template, so the backend is the model of an
/// `NFKToolLoop`. A tool request on a backend built without a chat template renders the release's own.
///
/// A reasoning model's chain comes back under `NFKOutputReasoning`, so `NFKOutputText` holds the
/// answer alone; the markers come from the release's chat template, or from
/// ``NFKMLXGenerationParameterKey/reasoningFormat``. `NFKParameterReasoningEffort` binds the
/// template's own reasoning variables, so it reaches a release whose template reads one.
/// `NFKOutputUsage` reports what the run cost: the prompt, what a retained prompt cache served, the
/// reply, and the chain's share of it.
///
/// Generation is many forward passes; run it off the render thread, or submit a job, which reports the
/// reply as it generates and stops generating when it is cancelled.
@objc(NFKMLXLanguageBackend)
public final class NFKMLXLanguageBackend: NSObject, NFKInferenceBackend {

    private let lifecycle: NFKMLXLifecycle<NFKLMHolder>
    private let hasTokenizer: Bool
    /// The tokenizer, kept outside the lifecycle so a count never loads the model.
    private let countingTokenizer: NFKTokenizer?
    private let holdsDraft: Bool
    private let pagesExpertsAtLoad: Bool
    private let streamedLayersAtLoad: Int
    private let streamedBytesAtLoad: Int
    private let identifier: String
    private let defaults: NFKMLXGenerationOptions
    /// The cache kept between requests when a request asks for reuse. Generation is serialized
    /// through `generationLock`: two runs through one cache would interleave their rows.
    private var promptCache: NFKMLXPromptCache?
    /// The caches of requests that name their conversation, read and written under `generationLock`.
    private static let defaultConversationCacheByteBudget = 2 << 30
    private let conversationCaches = NFKMLXConversationCaches(byteBudget: defaultConversationCacheByteBudget)
    /// The cache figures ``backendStatus`` reports, refreshed under `generationLock` after each run
    /// and each change, and read under `statusLock`, so a status request never waits on a run.
    private var cacheStatus: [String: Any] = ["conversation_caches": 0, "conversation_cache_bytes": 0,
                                              "conversation_cache_byte_budget": defaultConversationCacheByteBudget,
                                              "prompt_cache_length": 0]
    private let statusLock = NSLock()
    private let generationLock = NSLock()
    private let modelInfoCache = NFKMLXModelInfoCache()
    private let releaseURL: URL?
    /// The release's own template, which a message request renders through when it chooses none.
    private let releaseTemplate: NFKMLXChatTemplate?
    /// The ids the release's `generation_config.json` ends generation on.
    private let releaseStops: Set<Int>
    /// The sampling the release's `generation_config.json` sets, or nil where it decodes greedily.
    private let releaseSampling: NFKMLXReleaseSampling?
    /// The tokens the release marks special, by literal, which the reference's decode skips.
    private let releaseSpecialTokens: [String: Int]

    init(net: NFKMLXLanguageNet, tokenizer: NFKTokenizer?, identifier: String,
         options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
         draft: NFKMLXLanguageNet? = nil, releaseURL: URL? = nil, releaseTemplate: NFKMLXChatTemplate? = nil) {
        self.lifecycle = NFKMLXLifecycle(NFKLMHolder(net, tokenizer, draft: draft))
        self.hasTokenizer = tokenizer != nil
        self.countingTokenizer = tokenizer
        self.holdsDraft = draft != nil
        self.pagesExpertsAtLoad = net.expertStore != nil
        self.streamedLayersAtLoad = net.layerStream?.layers.count ?? 0
        self.streamedBytesAtLoad = net.layerStream?.bytesPerPass ?? 0
        self.identifier = identifier
        self.defaults = options
        self.releaseURL = releaseURL
        self.releaseTemplate = releaseTemplate ?? releaseURL.flatMap(NFKMLXLanguage.releaseChatTemplate(inDirectory:))
        self.releaseStops = NFKMLXToolCalling.generationConfigStops(directoryURL: releaseURL)
        self.releaseSampling = NFKMLXReleaseSampling(directoryURL: releaseURL)
        self.releaseSpecialTokens = releaseURL.map(NFKMLXLanguage.specialTokenIDs(inDirectory:)) ?? [:]
        super.init()
        lifecycle.setFreeHandler { [weak self] in
            self?.dropCaches()
        }
    }

    /// Whether the backend can answer: it reads text prompts only, so it needs a tokenizer.
    public var isReady: Bool { hasTokenizer }

    /// The loaded network's parameter count, weight bytes, precision, and quantization, with the
    /// release's `model_type` and `max_position_embeddings` where its `config.json` states them, the
    /// bytes its release directory or GGUF file occupies on disk, and the key-value bytes each token
    /// adds while the cache is unquantized, and the residency, `paged`, `streamed`, or `resident`
    /// (`NFKModelInfo*` keys). A paged model's routed experts
    /// count toward its parameters; its weight bytes are what memory holds, which leaves out experts
    /// read from the release. `NFKModelInfoResidentBytes` and `NFKModelInfoPeakBytes` come from the last
    /// measured load. A model that is unloaded reports the description read while it was loaded.
    /// Introduced in InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        let info = lifecycle.ifLoaded { holder in
            modelInfoCache.value { describe(holder) }
        } ?? modelInfoCache.currentValue ?? [:]
        return lifecycle.modelInfo(over: info)
    }

    private func describe(_ holder: NFKLMHolder) -> [String: Any] {
        let configuration = holder.net.configuration
        var info = NFKMLXModelDescription.info(of: [holder.net])
        if let releaseURL {
            info[NFKModelInfoStorageBytes] = NFKMLXModelDescription.storageBytes(at: releaseURL)
        }
        info[NFKModelInfoArchitecture] = configuration.modelType
        info[NFKModelInfoContextLength] = configuration.maximumPositions
        info[NFKModelInfoResidency] = pagesExpertsAtLoad ? NFKModelResidencyPaged
            : holder.net.layerStream != nil ? NFKModelResidencyStreamed : NFKModelResidencyResident
        if defaults.cacheQuantization == nil {
            let precision: NFKMLXWeightPrecision = info[NFKModelInfoPrecision] as? String == "float32" ? .float32 : .checkpoint
            info[NFKModelInfoKeyValueBytesPerToken] = NFKMLXModelSizing.keyValueBytesPerToken(of: configuration, precision: precision)
        }
        return info
    }

    /// Whether the model's routed experts are paged: left in the release and read as the router
    /// reaches them, through ``expertStore``. Introduced in InferKit 0.4.0.
    @objc public var pagesExperts: Bool { pagesExpertsAtLoad }

    /// The store a paged model reads its routed experts from, or nil where they are resident or the
    /// model is unloaded. Its cache budget can be changed between requests. Introduced in InferKit 0.4.0.
    @objc public var expertStore: NFKMLXExpertStore? { lifecycle.ifLoaded { $0.net.expertStore } ?? nil }

    /// How many decoder layers the loaded model leaves in the release, read in their turn on every pass,
    /// or the first load's count while the model is unloaded; 0 where every layer is held. Introduced
    /// in InferKit 0.4.0.
    @objc public var streamedLayerCount: Int {
        lifecycle.ifLoaded { $0.net.layerStream?.layers.count ?? 0 } ?? streamedLayersAtLoad
    }

    /// The bytes each pass of the loaded model reads from the release for the streamed layers, or the
    /// first load's bytes while the model is unloaded; 0 where every layer is held. Introduced in
    /// InferKit 0.4.0.
    @objc public var streamedBytesPerPass: Int {
        lifecycle.ifLoaded { $0.net.layerStream?.bytesPerPass ?? 0 } ?? streamedBytesAtLoad
    }
    public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads: the core sampling keys, the schema and the two
    /// keys that shape the output, the context length and cache settings a catalog build sets as load
    /// defaults (`NFKModelLoadOptionContextLength`, `NFKModelLoadOptionCacheBits`,
    /// `NFKModelLoadOptionCacheGroupSize`), and every ``NFKMLXGenerationParameterKey``. Introduced in
    /// InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterTemperature, NFKParameterTopP, NFKParameterTopK, NFKParameterMaxTokens, NFKParameterSeed,
         NFKParameterJSONSchema, NFKParameterOutputFormat, NFKParameterChoices,
         NFKModelLoadOptionContextLength, NFKModelLoadOptionCacheBits, NFKModelLoadOptionCacheGroupSize,
         NFKMLXGenerationParameterKey.contextWindow,
         NFKMLXGenerationParameterKey.cacheQuantizationBits,
         NFKMLXGenerationParameterKey.cacheQuantizationGroupSize,
         NFKMLXGenerationParameterKey.cacheQuantizationPerChannelKeys,
         NFKMLXGenerationParameterKey.prefillChunkSize,
         NFKMLXGenerationParameterKey.chatTemplate,
         NFKMLXGenerationParameterKey.draftTokens,
         NFKMLXGenerationParameterKey.reusesPromptCache,
         NFKMLXGenerationParameterKey.outputFormat,
         NFKMLXGenerationParameterKey.choices,
         NFKMLXGenerationParameterKey.reasoningFormat,
         NFKMLXGenerationParameterKey.toolCallFormat,
         NFKParameterReasoningEffort,
         NFKParameterConversationKey,
         NFKParameterTools]
    }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputPrompt, NFKInputMessages] }

    /// The result keys the backend writes: the reply, the tool calls and the assistant turn, the reasoning
    /// when the model shows any, the usage, and the parsed object of a schema-constrained request. Introduced
    /// in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> {
        NFKMLXToolCalling.outputKeys.union([NFKOutputReasoning, NFKOutputUsage, NFKOutputStructured, NFKOutputFinishReason])
    }

    /// Whether the backend holds a draft model, so requests can decode speculatively.
    @objc public var hasDraftModel: Bool { holdsDraft }

    // MARK: Unloading

    /// Frees the network, the draft, and every prompt cache, then MLX's cache; the next run or
    /// ``prepare()`` loads the release again. It does nothing for a backend built from a network in
    /// memory: only one that `NFKMLXModelRegistry` or a download factory built from a release can unload.
    /// With a run in flight, the model is freed when the run ends. Introduced in InferKit 0.4.0.
    @objc public func unload() {
        lifecycle.unload()
    }

    /// Whether the backend holds its network. Introduced in InferKit 0.4.0.
    @objc public var isLoaded: Bool { lifecycle.isLoaded }

    /// The rise in MLX's active memory over the model's last load, or 0 while it is unloaded or before a
    /// measured load. Introduced in InferKit 0.4.0.
    @objc public var residentByteCount: Int { lifecycle.residentByteCount }

    /// The peak of MLX's active memory during the model's last measured load, above its level before it.
    /// Introduced in InferKit 0.4.0.
    @objc public var peakByteCount: Int { lifecycle.peakByteCount }

    /// The load options the backend was built with, keyed by `NFKModelLoadOption*` keys, as requested with
    /// defaults filled; empty for a backend built without options. Introduced in InferKit 0.4.0.
    @objc public var loadOptionValues: [String: Any] { lifecycle.loadOptionValues }

    /// Loads the release when the model is unloaded. Introduced in InferKit 0.4.0.
    @objc(prepareWithError:)
    public func prepare() throws {
        try lifecycle.prepare()
    }

    /// Drops the prompt caches and the vocabulary, which hold arrays over the freed network.
    private func dropCaches() {
        generationLock.lock(); defer { generationLock.unlock() }
        promptCache = nil
        conversationCaches.removeAll()
        vocabulary = nil
        refreshCacheStatus()
    }

    /// How many positions the retained prompt cache holds, or 0 when none is kept.
    @objc public var promptCacheLength: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return promptCache?.count ?? 0
    }

    /// Drops every retained prompt cache, the conversations' included, so the next request prefills
    /// from the start.
    @objc public func resetPromptCache() {
        generationLock.lock(); defer { generationLock.unlock() }
        promptCache = nil
        conversationCaches.removeAll()
        refreshCacheStatus()
    }

    /// Drops the unnamed prompt cache and every conversation's when no run is in progress, and returns the
    /// bytes they held; returns 0 at once while a run holds them. `NFKInferenceServer` calls it to relieve
    /// memory. Introduced in InferKit 0.4.0.
    @objc public func releaseIdleConversationCaches() -> Int {
        guard generationLock.try() else { return 0 }
        defer { generationLock.unlock() }
        let held = conversationCaches.heldBytes + (promptCache?.allocatedBytes ?? 0)
        promptCache = nil
        conversationCaches.removeAll()
        refreshCacheStatus()
        return held
    }

    /// Releases up to `byteCount` of the routed experts the loaded model caches, keeping `keepingByteCount`,
    /// and returns the bytes freed; 0 where the model holds its experts resident or is unloaded. The
    /// cache's budget is unchanged. Introduced in InferKit 0.4.0.
    @objc public func releaseExpertCacheByteCount(_ byteCount: Int, keepingByteCount: Int) -> Int {
        expertStore?.releaseCachedBytes(byteCount, keeping: keepingByteCount) ?? 0
    }

    /// Drops one conversation's prompt cache. Introduced in InferKit 0.4.0.
    @objc(resetPromptCacheForConversation:)
    public func resetPromptCache(forConversation conversation: String) {
        generationLock.lock(); defer { generationLock.unlock() }
        conversationCaches.remove(conversation)
        refreshCacheStatus()
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
            return conversationCaches.byteBudget
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            conversationCaches.byteBudget = newValue
            conversationCaches.evict()
            refreshCacheStatus()
        }
    }

    /// How many conversations hold a prompt cache. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheCount: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return conversationCaches.count
    }

    /// The bytes the conversations' prompt caches occupy. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheBytes: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return conversationCaches.heldBytes
    }

    /// The prompt caches as of the last run or change: `conversation_caches`,
    /// `conversation_cache_bytes`, `conversation_cache_byte_budget`, and `prompt_cache_length`, the
    /// positions the unnamed retained cache holds; and, where the loaded model pages its experts, the
    /// expert cache now: `expert_cache_bytes` and `expert_cache_byte_budget`. `NFKInferenceServer` serves
    /// it under the model's "status". It never waits on a run in progress. Introduced in InferKit 0.4.0.
    @objc public var backendStatus: [String: Any] {
        var status: [String: Any] = {
            statusLock.lock(); defer { statusLock.unlock() }
            return cacheStatus
        }()
        if let store = expertStore {
            status.merge(store.statusFigures) { $1 }
        }
        return status
    }

    /// Records the cache figures ``backendStatus`` reports. Runs under `generationLock`.
    private func refreshCacheStatus() {
        let figures: [String: Any] = ["conversation_caches": conversationCaches.count,
                                      "conversation_cache_bytes": conversationCaches.heldBytes,
                                      "conversation_cache_byte_budget": conversationCaches.byteBudget,
                                      "prompt_cache_length": promptCache?.count ?? 0]
        statusLock.lock(); defer { statusLock.unlock() }
        cacheStatus = figures
    }

    /// The input tokens a run of the request costs: its prompt rendered through the chat template the
    /// run uses, tools included, and tokenized as the run tokenizes it. Runs no model, and counts while
    /// the model is unloaded without loading it. Introduced in InferKit 0.4.0.
    @objc(tokenCountForRequest:error:)
    public func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber {
        guard let tokenizer = countingTokenizer else {
            throw NFKMLXError.unsupportedConfiguration(
                "text generation needs a tokenizer; build the backend with one")
        }
        return NSNumber(value: try preparedPrompt(for: request, tokenizer: tokenizer).tokens.count)
    }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try lifecycle.withState { try run(request, holder: $0) }
    }

    /// The shortest time between two partial results a job reports.
    var partialResultInterval: TimeInterval = 0.05

    /// Runs the request as a job that reports the reply as it generates.
    ///
    /// @discussion Each partial result holds the answer so far under `NFKOutputText` and a reasoning
    /// model's chain so far under `NFKOutputReasoning`, split at the markers the finished result is split
    /// at, and the progress is the share of the token limit produced. Cancelling the job ends generation
    /// at the next token. Introduced in InferKit 0.4.0.
    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        NFKMLXGenerationJob.submit(partialResultInterval: partialResultInterval) { [self] job in
            try lifecycle.withState { try run(request, holder: $0, job: job) }
        }
    }

    private func run(_ request: NFKInferenceRequest, holder: NFKLMHolder,
                     job: NFKMLXGenerationJob? = nil) throws -> NFKInferenceResult {
        guard let tokenizer = holder.tokenizer else {
            throw NFKMLXError.unsupportedConfiguration(
                "text generation needs a tokenizer; build the backend with one")
        }
        var (options, tokens, toolCallFormat) = try preparedPrompt(for: request, tokenizer: tokenizer)
        options.maxTokens = try NFKMLXContextRoom.bounded(options.chosenMaxTokens, promptTokens: tokens.count,
                                                          contextLength: holder.net.configuration.contextLength,
                                                          window: options.contextWindow, source: identifier)
        let reasoningFormat = Self.reasoningFormat(for: options)
        let markers = (reasoningFormat.map { [$0.opening, $0.closing, $0.answerPrefix] } ?? []) + (toolCallFormat?.markers ?? [])
        let decode = NFKMLXLanguage.replyDecoder(tokenizer,
                                                 skipping: NFKMLXLanguage.skippedSpecialTokens(releaseSpecialTokens,
                                                                                               keepingMarkers: markers))
        let onToken = job.map { job in
            Self.partialReporter(job, prompt: tokens, limit: options.maxTokens, format: reasoningFormat,
                                 callMarkers: NFKMLXToolCalling.templateTools(for: request) != nil ? toolCallFormat?.markers ?? [] : [],
                                 decode: decode)
        }
        let (produced, cached) = try generate(tokens, options: options, conversation: Self.conversation(of: request),
                                              holder: holder, onToken: onToken)
        let reply = decode(produced)

        let (reasoning, answer, reasoningTokens) = NFKMLXUsage.split(reply, format: reasoningFormat,
                                                                     produced: produced, decode: decode)

        var outputs = NFKMLXToolCalling.outputs(answer: answer, reasoning: reasoning, request: request,
                                                format: toolCallFormat, source: identifier)
        if !reasoning.isEmpty {
            outputs[NFKOutputReasoning] = reasoning
        }
        let calls = (outputs[NFKOutputToolCalls] as? [Any])?.isEmpty == false
        outputs[NFKOutputFinishReason] = calls ? NFKFinishReasonToolCalls
            : (produced.count >= options.maxTokens ? NFKFinishReasonLength : NFKFinishReasonStop)
        outputs[NFKOutputUsage] = NFKMLXUsage.outputs(inputTokens: tokens.count,
                                                      cachedTokens: cached,
                                                      outputTokens: produced.count,
                                                      reasoningTokens: reasoningTokens)
        // JSON that was asked for comes back parsed too, as the remote backends return it; JSON-looking
        // text that was not asked for is not guessed at.
        if options.jsonSchema != nil || options.jsonOutput,
           let parsed = try? JSONSerialization.jsonObject(with: Data(answer.utf8), options: [.fragmentsAllowed]) {
            outputs[NFKOutputStructured] = parsed
        }
        return NFKInferenceResult(outputs: outputs)
    }

    /// What a run of the request generates from: its options, its prompt as token ids, and the syntax
    /// its calls are read in.
    private func preparedPrompt(for request: NFKInferenceRequest, tokenizer: NFKTokenizer) throws
        -> (options: NFKMLXGenerationOptions, tokens: [Int], toolCallFormat: NFKMLXToolCallFormat?) {
        var options = defaults.sampling(from: releaseSampling)
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
        Self.applyMLXParameters(from: request, to: &options)
        if let schema = request.parameter(forKey: NFKParameterJSONSchema) {
            guard let dictionary = schema as? [String: Any] else {
                throw NFKMLXError.unsupportedConfiguration("NFKParameterJSONSchema is a JSON Schema object")
            }
            options.jsonSchema = try NFKMLXJSONSchema(json: dictionary)
        }
        // An instruct release answers the turns it was trained on, so a message list takes the release's
        // template unless the request or the caller chose one; a raw prompt is read verbatim either way.
        if case .none = options.chatTemplate, options.usesReleaseChatTemplate, request.messages != nil, let releaseTemplate,
           request.parameter(forKey: NFKMLXGenerationParameterKey.chatTemplate) == nil {
            options.chatTemplate = releaseTemplate
        }
        options.chatTemplate = NFKMLXToolCalling.template(options.chatTemplate, for: request, releaseDirectoryURL: releaseURL)
        // The reference's generate ends on every id the release's generation config lists: an instruct
        // release ends a turn on a marker its tokenizer's `eos_token` may not name.
        if options.stopTokens.isEmpty {
            options.stopTokens = releaseStops.union(tokenizer.eosTokenId >= 0 ? [tokenizer.eosTokenId] : [])
        }

        let variables = try Self.templateVariables(for: request, template: options.chatTemplate)
        let toolCallFormat = Self.toolCallFormat(for: options)
        guard let text = Self.prompt(from: request, template: options.chatTemplate, variables: variables,
                                     source: identifier, toolCallFormat: toolCallFormat) else {
            throw NFKMLXError.unsupportedInput
        }
        return (options, tokenizer.encode(text).map(\.intValue), toolCallFormat)
    }

    /// The token callback a job's run generates with: it reports each token to `job` with the reply so
    /// far, and answers whether generation continues.
    static func partialReporter(_ job: NFKMLXGenerationJob, prompt: [Int], limit: Int, format: NFKMLXReasoningFormat?,
                                callMarkers: [String], decode: @escaping ([Int]) -> String) -> (Int) -> Bool {
        let opensChain = NFKMLXPartialReply.opensChain(prompt, format: format, decode: decode)
        var produced = [Int]()
        return { token in
            produced.append(token)
            return job.proceed(produced: produced.count, limit: limit) {
                NFKMLXPartialReply.outputs(decode(produced), format: format, opensChain: opensChain, callMarkers: callMarkers)
            }
        }
    }

    /// The reasoning markers a run's output carries: the ones the request named, or the ones the
    /// release's own chat template shows. A run with no template and no stated format splits
    /// nothing, since there is no way to know what the model wraps a chain in.
    static func reasoningFormat(for options: NFKMLXGenerationOptions) -> NFKMLXReasoningFormat? {
        if options.reasoningFormatIsStated {
            return options.reasoningFormat
        }
        guard case .jinja(let template, _, _) = options.chatTemplate else {
            return nil
        }
        return NFKMLXReasoningFormat.detected(inChatTemplate: template)
    }

    /// The syntax a run's calls are read in: the one the request named, or the one the release's own
    /// chat template shows.
    static func toolCallFormat(for options: NFKMLXGenerationOptions) -> NFKMLXToolCallFormat? {
        if options.toolCallFormatIsStated {
            return options.toolCallFormat
        }
        guard case .jinja(let template, _, _) = options.chatTemplate else {
            return nil
        }
        return NFKMLXToolCallFormat.detected(inChatTemplate: template)
    }

    /// The template bindings a request's reasoning effort asks for.
    ///
    /// @discussion The release's template is what takes the effort, and the two shipped families
    /// spell it differently: Qwen3 tests an `enable_thinking` flag and closes the block when it is
    /// false, while gpt-oss writes a `reasoning_effort` level into its system message. Both are
    /// bound, and a template that reads neither renders unchanged. A request that names an effort
    /// without a template to read it is refused, since the level would reach the model nowhere.
    static func templateVariables(for request: NFKInferenceRequest,
                                  template: NFKMLXChatTemplate) throws -> [String: Any] {
        guard let effort = request.parameter(forKey: NFKParameterReasoningEffort) else {
            return [:]
        }
        guard let level = effort as? String, !level.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration(
                "NFKParameterReasoningEffort is a level name: light, moderate, or deep")
        }
        guard case .jinja = template else {
            throw NFKMLXError.unsupportedConfiguration(
                "a reasoning effort reaches the model through the release's chat template; "
                + "set NFKMLXParameterChatTemplate to it")
        }
        return ["enable_thinking": level != NFKReasoningEffortLight,
                "reasoning_effort": Self.harmonyLevel(for: level)]
    }

    /// The level a harmony template reads for each level the contract names. Another string goes out
    /// as written, which reaches a release that names its own.
    static func harmonyLevel(for effort: String) -> String {
        switch effort {
        case NFKReasoningEffortLight: return "low"
        case NFKReasoningEffortModerate: return "medium"
        case NFKReasoningEffortDeep: return "high"
        default: return effort
        }
    }

    /// The vocabulary's bytes, read from the tokenizer once and kept for every constrained request.
    private var vocabulary: NFKMLXVocabulary?

    /// The constraint a request asks for, if any: a custom one wins, then a schema, then JSON, then
    /// the choices.
    private func constraint(for options: NFKMLXGenerationOptions, tokenizer: NFKTokenizer,
                            holder: NFKLMHolder) -> (any NFKMLXTokenConstraint)? {
        if let custom = options.constraint { return custom }
        guard options.jsonSchema != nil || options.jsonOutput || options.choices != nil else { return nil }
        let vocabulary: NFKMLXVocabulary
        if let kept = self.vocabulary {
            vocabulary = kept
        } else {
            vocabulary = NFKMLXVocabulary(tokenizer: tokenizer, size: holder.net.configuration.vocabularySize)
            self.vocabulary = vocabulary
        }
        if let schema = options.jsonSchema { return NFKMLXJSONSchemaConstraint(schema: schema, vocabulary: vocabulary) }
        if options.jsonOutput { return NFKMLXJSONConstraint(vocabulary: vocabulary, root: options.jsonRoot) }
        return NFKMLXChoiceConstraint(choices: options.choices ?? [], vocabulary: vocabulary)
    }

    /// The conversation whose prompt cache a request continues: its `NFKParameterConversationKey`,
    /// unless the request turns reuse off.
    static func conversation(of request: NFKInferenceRequest) -> String? {
        NFKMLXConversation.name(of: request)
    }

    /// Runs the tokens through the plain or the speculative loop, against the conversation's prompt
    /// cache when the request names one, or the retained cache when the options ask for one.
    /// Answers the tokens produced and how many of the prompt's the cache already held, which is
    /// the cached share a run reports. `onToken` sees each token, and answering false ends the run.
    private func generate(_ tokens: [Int], options requested: NFKMLXGenerationOptions, conversation: String?,
                          holder: NFKLMHolder, onToken: ((Int) -> Bool)? = nil) throws -> (produced: [Int], cached: Int) {
        generationLock.lock(); defer { generationLock.unlock() }
        let net = holder.net
        var options = requested
        if let tokenizer = holder.tokenizer {
            options.constraint = constraint(for: requested, tokenizer: tokenizer, holder: holder)
        }
        var cache: NFKMLXPromptCache?
        if let conversation {
            cache = conversationCaches.cache(for: conversation, layerCount: net.configuration.layerCount,
                                             options: options)
        } else if options.reusesPromptCache {
            if let kept = promptCache, kept.matches(layerCount: net.configuration.layerCount, options: options) {
                cache = kept
            } else {
                cache = NFKMLXPromptCache(layerCount: net.configuration.layerCount,
                                          window: options.contextWindow,
                                          quantization: options.cacheQuantization)
            }
            promptCache = cache
        } else {
            promptCache = nil
        }
        let produced: [Int]
        if let draft = holder.draft, options.draftTokens > 0, options.constraint == nil {
            produced = net.generate(prompt: tokens, options: options, draft: draft, promptCache: cache, onToken: onToken)
        } else {
            produced = net.generate(prompt: tokens, options: options, promptCache: cache, onToken: onToken)
        }
        if let failure = NFKMLXErrorScope.failure {
            // The failed evaluation left rows no later request may extend.
            cache?.reset()
            refreshCacheStatus()
            throw failure
        }
        do {
            try net.verifyStream()
        } catch {
            // The failed pass skipped layers, so the cache holds rows no later request may extend; with
            // it emptied, the next request reads every streamed layer again.
            cache?.reset()
            net.layerStream?.clearFailure()
            refreshCacheStatus()
            throw error
        }
        if let conversation {
            conversationCaches.evict(keeping: conversation)
        }
        refreshCacheStatus()
        return (produced, cache?.sharedPrefixLength ?? 0)
    }

    /// Overrides `options` with the MLX generation parameters a request carries, so an Objective-C
    /// caller reaches every option the Swift `NFKMLXGenerationOptions` struct exposes.
    static func applyMLXParameters(from request: NFKInferenceRequest, to options: inout NFKMLXGenerationOptions) {
        applyLoadDefaults(from: request, to: &options)
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.contextWindow) as? NSNumber {
            options.contextWindow = value.intValue
        }
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.prefillChunkSize) as? NSNumber {
            options.prefillChunkSize = value.intValue
        }
        if let bits = request.parameter(forKey: NFKMLXGenerationParameterKey.cacheQuantizationBits) as? NSNumber {
            let groupSize = (request.parameter(forKey: NFKMLXGenerationParameterKey.cacheQuantizationGroupSize) as? NSNumber)?.intValue ?? 64
            let perChannel = (request.parameter(forKey: NFKMLXGenerationParameterKey.cacheQuantizationPerChannelKeys) as? NSNumber)?.boolValue ?? true
            options.cacheQuantization = .init(bits: bits.intValue, groupSize: groupSize,
                                              keyAxis: perChannel ? .sequence : .headDimension)
        }
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.chatTemplate) as? String {
            // A string carrying Jinja delimiters is the release's own template; render it faithfully.
            if value.contains("{%") || value.contains("{{") {
                options.chatTemplate = .jinja(template: value)
            } else {
                options.chatTemplate = value.lowercased() == "chatml" ? .chatML : .none
            }
        }
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.draftTokens) as? NSNumber {
            options.draftTokens = value.intValue
        }
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.reusesPromptCache) as? NSNumber {
            options.reusesPromptCache = value.boolValue
        }
        if let value = (request.parameter(forKey: NFKMLXGenerationParameterKey.outputFormat)
                        ?? request.parameter(forKey: NFKParameterOutputFormat)) as? String {
            switch value.lowercased() {
            case "json": options.jsonOutput = true; options.jsonRoot = .container
            case "json-object": options.jsonOutput = true; options.jsonRoot = .object
            case "json-array": options.jsonOutput = true; options.jsonRoot = .array
            default: options.jsonOutput = false
            }
        }
        if let value = (request.parameter(forKey: NFKMLXGenerationParameterKey.choices)
                        ?? request.parameter(forKey: NFKParameterChoices)) as? [String], !value.isEmpty {
            options.choices = value
        }
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.reasoningFormat),
           let format = NFKMLXReasoningFormat.named(value) {
            options.reasoningFormat = format
            options.reasoningFormatIsStated = true
        }
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.toolCallFormat),
           let format = NFKMLXToolCallFormat.named(value) {
            options.toolCallFormat = format
            options.toolCallFormatIsStated = true
        }
    }

    /// Overrides `options` with the context length and cache settings a request carries under the
    /// catalog's load-option keys, which a catalog build fills from its load options. The
    /// ``NFKMLXGenerationParameterKey`` keys, read after these, win where a request carries both.
    static func applyLoadDefaults(from request: NFKInferenceRequest, to options: inout NFKMLXGenerationOptions) {
        var values = [String: Any]()
        for key in [NFKModelLoadOptionContextLength, NFKModelLoadOptionCacheBits, NFKModelLoadOptionCacheGroupSize] {
            values[key] = request.parameter(forKey: key)
        }
        guard !values.isEmpty else {
            return
        }
        options = NFKMLXLoadOptions(values).generationOptions(base: options)
    }

    /// The prompt text, from either input key. A message list, which wins over `NFKInputPrompt` when a
    /// request carries both, is flattened plainly, or rendered through the chat template when one is
    /// asked for, with the request's tools and its tool conversation in the shape templates read. A
    /// request without one has its `NFKInputPrompt` used verbatim. `source` names the backend whose
    /// assistant turns get their reasoning back.
    static func prompt(from request: NFKInferenceRequest,
                       template: NFKMLXChatTemplate = .none,
                       variables: [String: Any] = [:],
                       source: String = "",
                       toolCallFormat: NFKMLXToolCallFormat? = nil) -> String? {
        guard let original = request.messages else { return request.prompt }
        let messages = NFKMLXToolCalling.templateMessages(original, format: toolCallFormat, source: source)
        switch template {
        case .none:
            return messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
        case .chatML:
            return chatMLPrompt(from: messages)
        case .jinja(let templateSource, let bosToken, let eosToken):
            // A render failure must not abort generation; fall back to the ChatML approximation.
            return (try? NFKMLXChatTemplateRenderer.render(templateSource, messages: messages,
                                                           addGenerationPrompt: true,
                                                           bosToken: bosToken, eosToken: eosToken,
                                                           tools: NFKMLXToolCalling.templateTools(for: request),
                                                           variables: variables))
                ?? chatMLPrompt(from: messages)
        }
    }

    /// Renders a message list in the ChatML format, with a trailing assistant turn opened so the model
    /// continues as the assistant. A message missing a role is treated as the user's.
    static func chatMLPrompt(from messages: [[AnyHashable: Any]]) -> String {
        var rendered = ""
        for message in messages {
            let role = message["role"] as? String ?? "user"
            let content = message["content"] as? String ?? ""
            rendered += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }
        rendered += "<|im_start|>assistant\n"
        return rendered
    }
}

/// Building, loading, and registering the language model.
@objc(NFKMLXLanguage)
public final class NFKMLXLanguage: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "qwen3"
    static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    static let optionalFiles = ["vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja", "generation_config.json"]
    static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// A decoder at `configuration`, with random weights until a load fills it. Introduced in InferKit
    /// 0.4.0.
    public static func makeNet(_ configuration: NFKMLXLanguageConfiguration = .qwen3_0_6B) -> NFKMLXLanguageNet {
        NFKMLXLanguageNet(configuration)
    }

    /// Reads a released `config.json` into a configuration.
    ///
    /// @discussion The dense decoder and its two mixture-of-experts families (`qwen3_moe`, `mixtral`)
    /// are read. A config naming a hybrid architecture, or an expert family this network does not
    /// implement, is rejected rather than approximated: `Qwen3_5ForConditionalGeneration` interleaves
    /// linear-attention layers with full attention, and loading its weights into a dense stack would
    /// produce fluent-looking nonsense.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXLanguageConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        // A multimodal release states the decoder under `text_config` and names the WRAPPER in its
        // top-level `architectures` (Mistral-Small 3 is `Mistral3ForConditionalGeneration`), which the
        // causal-model guard below would reject for a decoder this network does implement.
        return try configuration(fromJSON: (json["text_config"] as? [String: Any]) ?? json)
    }

    /// The configuration a parsed `config.json` object describes. A multimodal release's `text_config`
    /// reads through here too (Qwen3-VL's decoder is this stack under `qwen3_vl_text`).
    public static func configuration(fromJSON json: [String: Any]) throws -> NFKMLXLanguageConfiguration {
        if let architectures = json["architectures"] as? [String],
           let name = architectures.first,
           !(name.hasSuffix("ForCausalLM")) {
            throw NFKMLXError.unsupportedConfiguration(
                "\(name) is not a dense causal language model; this network implements the dense "
                + "decoder only")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (json[key] as? NSNumber)?.floatValue ?? fallback }
        let modelType = (json["model_type"] as? String) ?? ""
        // Newer transformers writes `layer_types` even for a homogeneous dense stack (the MiniMax
        // Music 3 language model lists 36 × "full_attention"), so only a MIXED stack is rejected —
        // except gpt-oss, whose alternation of sliding-window and full layers this network implements.
        let layerTypes = (json["layer_types"] as? [String]) ?? []
        let slidingAllowed = Set(["full_attention", "sliding_attention"])
        if layerTypes.contains(where: { !slidingAllowed.contains($0) })
            || (modelType != "gpt_oss" && layerTypes.contains("sliding_attention")) {
            throw NFKMLXError.unsupportedConfiguration(
                "the config describes a hybrid-attention model, which this network does not implement")
        }
        let experts = try expertConfiguration(json, modelType: modelType)

        let hidden = integer("hidden_size", 1024)
        // A release that states its head width but not its head count is read as the count that fills
        // the hidden width, which is what transformers resolves for it. SmolVLM2-2.2B's `text_config`
        // omits `num_attention_heads` and gives `head_dim` 64 at hidden 2048, where a fixed fallback
        // builds projections the checkpoint does not fit.
        let statedHeadDimensions = (json["head_dim"] as? NSNumber)?.intValue
        let headFallback = statedHeadDimensions.map { hidden / max($0, 1) } ?? 16
        let heads = integer("num_attention_heads", headFallback)
        var configuration = NFKMLXLanguageConfiguration(
            hiddenSize: hidden,
            layerCount: integer("num_hidden_layers", 28),
            headCount: heads,
            keyValueHeadCount: integer("num_key_value_heads", heads),
            headDimensions: statedHeadDimensions ?? (hidden / max(heads, 1)),
            intermediateSize: integer("intermediate_size", 3072),
            vocabularySize: integer("vocab_size", 151_936),
            // Transformers 5.x nests the rotary base under `rope_parameters`.
            ropeTheta: ((json["rope_parameters"] as? [String: Any])?["rope_theta"] as? NSNumber)?
                .floatValue ?? real("rope_theta", 1_000_000),
            rmsEpsilon: real("rms_norm_eps", 1e-6),
            tiesWordEmbeddings: (json["tie_word_embeddings"] as? NSNumber)?.boolValue ?? false,
            // Qwen2, Qwen2-MoE, and Qwen2.5-VL's text decoder carry query/key/value biases and leave
            // the flag unset (or null) because true is its default; Qwen3 and Llama carry none.
            attentionBias: ((json["attention_bias"] ?? json["qkv_bias"]) as? NSNumber)?.boolValue
                ?? ["qwen2", "qwen2_moe", "qwen2_5_vl_text", "qwen2_vl_text"].contains(modelType))
        // Qwen3 normalizes queries and keys per head; Qwen2 and Llama do not. The model type is what
        // says so — the config carries no flag for it.
        configuration.normalizesQueryAndKey = modelType.hasPrefix("qwen3")
        configuration.modelType = modelType.isEmpty ? nil : modelType
        configuration.maximumPositions = (json["max_position_embeddings"] as? NSNumber)?.intValue
        if let experts {
            guard experts.count > 0, (1 ... experts.count).contains(experts.active), experts.width > 0 else {
                throw NFKMLXError.unsupportedConfiguration(
                    "the config names a mixture of experts but not how many, how many run per token, "
                    + "or how wide they are")
            }
            configuration.expertCount = experts.count
            configuration.activeExpertCount = experts.active
            configuration.expertIntermediateSize = experts.width
            configuration.normalizesExpertWeights = experts.normalizes
            configuration.sharedExpertIntermediateSize = experts.shared
        }
        if modelType == "gpt_oss" {
            // The release spells each of these out; the defaults are the values it ships with.
            let window = integer("sliding_window", 128)
            configuration.slidingWindows = layerTypes.map { $0 == "sliding_attention" ? window : nil }
            configuration.attentionSinks = true
            configuration.outputProjectionBias = configuration.attentionBias
            configuration.routerBias = true
            configuration.clampedSwiGLU = NFKMLXClampedSwiGLU(limit: real("swiglu_limit", 7))
            configuration.normWeightInFloat32 = true
        }
        // Mistral bands every layer to the same window wherever the release states one; transformers
        // applies it at any context length, so a window wider than the context is read too.
        if modelType == "mistral", let window = (json["sliding_window"] as? NSNumber)?.intValue, window > 0 {
            configuration.slidingWindows = Array(repeating: window, count: configuration.layerCount)
        }
        if modelType == NFKMLXHunyuanDense.modelType {
            try NFKMLXHunyuanDense.read(json, into: &configuration)
            return configuration
        }
        if modelType == NFKMLXHunyuanMoE.modelType {
            try NFKMLXHunyuanMoE.read(json, into: &configuration)
        }
        // A release that extended its window says so here. An unimplemented kind throws rather than
        // loading under the wrong rotary, which would run and be wrong.
        configuration.ropeScaling = try NFKMLXRoPEScaling.read(
            json["rope_scaling"],
            maximumPositions: integer("max_position_embeddings", 32_768))
        return configuration
    }

    /// The expert geometry a config describes, or nil for a dense feed-forward.
    ///
    /// @discussion Three families are read. `qwen3_moe` names its experts under `num_experts` and
    /// their width under `moe_intermediate_size`, and renormalizes the selected routing weights when
    /// `norm_topk_prob` says so; a release that interleaves dense layers (`mlp_only_layers`, a
    /// `decoder_sparse_step` above one) is refused, since this stack routes every layer. `qwen2_moe`
    /// reads the same fields, defaults `norm_topk_prob` to false as its releases do, and adds the
    /// shared expert's width (`shared_expert_intermediate_size`); one asking for a sliding window is
    /// refused. `mixtral` names them `num_local_experts` at `intermediate_size` and always
    /// renormalizes; a release with a sliding window is refused, since the attention here is full.
    /// Any other config that names experts (DeepSeek, gpt-oss) is refused by name.
    static func expertConfiguration(_ json: [String: Any], modelType: String) throws
        -> (count: Int, active: Int, width: Int, normalizes: Bool, shared: Int)? {
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func refuseInterleavedDenseLayers() throws {
            let denseLayers = (json["mlp_only_layers"] as? [Any]) ?? []
            guard denseLayers.isEmpty, integer("decoder_sparse_step", 1) == 1 else {
                throw NFKMLXError.unsupportedConfiguration(
                    "the config interleaves dense layers among the expert layers, which this network "
                    + "does not implement")
            }
        }
        switch modelType {
        case "qwen3_moe", "qwen3_vl_moe_text":
            try refuseInterleavedDenseLayers()
            return (integer("num_experts", 0), integer("num_experts_per_tok", 0),
                    integer("moe_intermediate_size", 0),
                    (json["norm_topk_prob"] as? NSNumber)?.boolValue ?? true, 0)
        case "qwen2_moe":
            try refuseInterleavedDenseLayers()
            if (json["use_sliding_window"] as? NSNumber)?.boolValue == true {
                throw NFKMLXError.unsupportedConfiguration(
                    "the config asks for sliding-window attention, which this network does not implement")
            }
            return (integer("num_experts", 0), integer("num_experts_per_tok", 0),
                    integer("moe_intermediate_size", 0),
                    (json["norm_topk_prob"] as? NSNumber)?.boolValue ?? false,
                    integer("shared_expert_intermediate_size", 0))
        case "mixtral":
            if let window = json["sliding_window"] as? NSNumber, window.intValue > 0 {
                throw NFKMLXError.unsupportedConfiguration(
                    "the config asks for sliding-window attention, which this network does not implement")
            }
            return (integer("num_local_experts", 0), integer("num_experts_per_tok", 0),
                    integer("intermediate_size", 0), true, 0)
        case NFKMLXHunyuanMoE.modelType:
            return NFKMLXHunyuanMoE.experts(json)
        case "gpt_oss":
            // The router takes the softmax over the selected logits, which is the renormalized form.
            return (integer("num_local_experts", 0), integer("num_experts_per_tok", 0),
                    integer("intermediate_size", 0), true, 0)
        default:
            if json["num_experts"] != nil || json["num_local_experts"] != nil || json["n_routed_experts"] != nil {
                throw NFKMLXError.unsupportedConfiguration(
                    "the config describes a mixture-of-experts family (\(modelType)) this network does "
                    + "not implement; qwen3_moe, qwen2_moe, mixtral, gpt_oss, and hy_v3 are the families it reads")
            }
            return nil
        }
    }

    /// The module key a release's tensor name maps to: Mixtral's `block_sparse_moe` spelling becomes
    /// the `mlp` layout Qwen3-MoE and this module share, mlx_lm's stacked `switch_mlp` is the module's
    /// `experts`, gpt-oss's `router` is the module's `gate` and its fused expert tensors gain the
    /// `.weight` the module's layers carry, and every other name passes through.
    static func moduleKey(forRelease key: String) -> String {
        // hy_v3 nests its router's projection one level deeper and names its shared expert `shared_mlp`.
        if key.contains(".mlp.router.gate.") {
            return key.replacingOccurrences(of: ".mlp.router.gate.", with: ".mlp.gate.")
        }
        if key.contains(".mlp.shared_mlp.") {
            return key.replacingOccurrences(of: ".mlp.shared_mlp.", with: ".mlp.shared_expert.")
        }
        if key.contains(".switch_mlp.") {
            return moduleKey(forRelease: key.replacingOccurrences(of: ".switch_mlp.", with: ".experts."))
        }
        if key.contains(".mlp.router.") {
            return key.replacingOccurrences(of: ".mlp.router.", with: ".mlp.gate.")
        }
        if key.hasSuffix(".mlp.experts.gate_up_proj") || key.hasSuffix(".mlp.experts.down_proj") {
            return key + ".weight"
        }
        // The MXFP4 release: packed words and their block scales, under the layer's own keys.
        if key.hasSuffix("_proj_blocks"), key.contains(".mlp.experts.") {
            return String(key.dropLast("_blocks".count)) + ".weight"
        }
        if key.hasSuffix("_proj_scales"), key.contains(".mlp.experts.") {
            return String(key.dropLast("_scales".count)) + ".scales"
        }
        guard key.contains(".block_sparse_moe.") else { return key }
        // A quantized release carries `scales` and `biases` beside each expert's `weight`.
        return key.replacingOccurrences(of: ".block_sparse_moe.gate.", with: ".mlp.gate.")
            .replacingOccurrences(of: ".block_sparse_moe.experts.", with: ".mlp.experts.")
            .replacingOccurrences(of: ".w1.", with: ".gate_proj.")
            .replacingOccurrences(of: ".w3.", with: ".up_proj.")
            .replacingOccurrences(of: ".w2.", with: ".down_proj.")
    }

    /// A release's tensors in the module's names and layouts: gpt-oss stores its fused expert
    /// projections as `[experts, in, out]` (an `x @ W` layout) where the switch linear holds
    /// `[experts, out, in]`, so those two are transposed; per-expert tensors are stacked.
    static func releaseWeights(_ pairs: [(String, MLXArray)]) -> [(String, MLXArray)] {
        let mapped = pairs.map { key, value -> (String, MLXArray) in
            let transposed = key.hasSuffix(".mlp.experts.gate_up_proj") || key.hasSuffix(".mlp.experts.down_proj")
            if transposed { return (moduleKey(forRelease: key), value.swappedAxes(-1, -2)) }
            // MXFP4 blocks are `[experts, out, groups, 16]` bytes: sixteen bytes hold a group of 32
            // nibbles, and viewed as little-endian uint32 they ARE MLX's mxfp4 words in its own
            // element order (measured against transformers' decode of the released bytes).
            if key.hasSuffix("_proj_blocks"), key.contains(".mlp.experts."), value.dtype == .uint8 {
                let words = value.view(dtype: .uint32)
                return (moduleKey(forRelease: key), words.reshaped([words.dim(0), words.dim(1), -1]))
            }
            return (moduleKey(forRelease: key), value)
        }
        return stackingExperts(mapped)
    }

    /// The block geometry gpt-oss's MXFP4 experts are stored in.
    static let mxfp4Quantization = NFKMLXWeights.Quantization(bits: 4, groupSize: 32, mode: .mxfp4)

    /// Replaces each fused expert projection with a layer holding the release's packed words and
    /// scales, wherever `pairs` carry them, so the strict apply that follows lands the packed arrays
    /// on matching structure rather than adopting them into a float layer.
    static func installPackedExperts(into net: NFKMLXLanguageNet, from pairs: [(String, MLXArray)]) throws {
        let byKey = Dictionary(pairs, uniquingKeysWith: { first, _ in first })
        for (index, block) in net.model.layers.enumerated() {
            guard let mixture = block.feedForward as? NFKLMMixtureFeedForward,
                  let fused = mixture.experts as? NFKLMFusedSwitchGLU else { continue }
            for projection in ["gate_up_proj", "down_proj"] {
                let base = "model.layers.\(index).mlp.experts.\(projection)"
                // MXFP4 scales are e8m0 bytes; an affine-quantized save carries float scales and is
                // rebuilt by `matchStructure` instead.
                guard let packed = byKey[base + ".weight"], packed.dtype == .uint32,
                      let scales = byKey[base + ".scales"], scales.dtype == .uint8 else { continue }
                let layer = NFKLMQuantizedSwitchLinear(packed: packed, scales: scales, biases: byKey[base + ".biases"],
                                                       groupSize: mxfp4Quantization.groupSize,
                                                       bits: mxfp4Quantization.bits, mode: mxfp4Quantization.mode)
                try fused.update(modules: ModuleChildren.unflattened([(projection, layer)]), verify: .noUnusedKeys)
            }
        }
    }

    private static let expertPattern = try! NSRegularExpression(
        pattern: #"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"#)

    /// Whether a release key holds one expert's projection, which ``stackingExperts(_:)`` stacks.
    static func isPerExpert(_ key: String) -> Bool {
        let name = moduleKey(forRelease: key)
        return expertPattern.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)) != nil
    }

    /// Stacks a release's per-expert tensors (`…experts.N.gate_proj.weight`, and a quantized release's
    /// `scales` and `biases`) into the module's one `[experts, out, in]` tensor per projection and part,
    /// leaving every other pair as it is.
    static func stackingExperts(_ mapped: [(String, MLXArray)]) -> [(String, MLXArray)] {
        var groups = [String: [(index: Int, value: MLXArray)]]()
        var rest = [(String, MLXArray)]()
        for (name, value) in mapped {
            let range = NSRange(name.startIndex..., in: name)
            guard let match = expertPattern.firstMatch(in: name, range: range),
                  let prefix = Range(match.range(at: 1), in: name),
                  let index = Range(match.range(at: 2), in: name).flatMap({ Int(name[$0]) }),
                  let projection = Range(match.range(at: 3), in: name),
                  let part = Range(match.range(at: 4), in: name) else {
                rest.append((name, value))
                continue
            }
            groups["\(name[prefix]).\(name[projection]).\(name[part])", default: []].append((index, value))
        }
        for (key, entries) in groups {
            rest.append((key, stacked(entries.sorted { $0.index < $1.index }.map(\.value))))
        }
        return rest
    }

    private static let stackedExpertPattern = try! NSRegularExpression(
        pattern: #"^(.*\.experts)\.(gate_proj|up_proj|down_proj|gate_up_proj)\.(weight|scales|biases)$"#)

    /// The expert slices a release tensor feeds on a paged load, or none for a tensor the load keeps.
    ///
    /// @discussion Four layouts arrive: one tensor per expert (Qwen2-MoE, Qwen3-MoE, Mixtral), a
    /// stacked `[experts, in, out]` tensor transposed per expert (gpt-oss at bf16), MXFP4 blocks viewed
    /// as words (gpt-oss as released), and a stacked `[experts, out, in]` tensor in the module's own
    /// layout (a checkpoint this package saved, float or quantized). Each slice ends in the layout the
    /// resident loader gives the whole tensor. The expert biases gpt-oss carries stay resident: they
    /// are one row per expert.
    static func expertSlices(forRelease key: String, entry: NFKMLXSafetensorsEntry) -> [NFKMLXExpertSlice] {
        let name = moduleKey(forRelease: key)
        let range = NSRange(name.startIndex..., in: name)
        if let match = expertPattern.firstMatch(in: name, range: range),
           let prefix = Range(match.range(at: 1), in: name),
           let index = Range(match.range(at: 2), in: name).flatMap({ Int(name[$0]) }),
           let projection = Range(match.range(at: 3), in: name),
           let part = Range(match.range(at: 4), in: name) {
            return [NFKMLXExpertSlice(group: "\(name[prefix]).\(name[projection])", part: String(name[part]),
                                      expert: index)]
        }
        guard entry.shape.count >= 3, let match = stackedExpertPattern.firstMatch(in: name, range: range),
              let prefix = Range(match.range(at: 1), in: name),
              let projection = Range(match.range(at: 2), in: name),
              let part = Range(match.range(at: 3), in: name) else { return [] }
        var slice = NFKMLXExpertSlice(group: "\(name[prefix]).\(name[projection])", part: String(name[part]),
                                      expert: nil)
        if key.hasSuffix(".mlp.experts.gate_up_proj") || key.hasSuffix(".mlp.experts.down_proj") {
            slice.inputMajor = true
        } else if key.hasSuffix("_proj_blocks"), entry.dtype == "U8" {
            slice.transform = { blocks in
                let words = blocks.view(dtype: .uint32)
                return words.reshaped([words.dim(0), -1])
            }
        }
        return [slice]
    }

    /// The packing of each expert projection the module already holds quantized, keyed by its group
    /// (`model.layers.N.mlp.experts.gate_proj`): what ``NFKMLXQuantization/matchStructure(toRelease:on:)``
    /// read from an mlx_lm release, which a paged load keeps for the experts it reads from the store.
    static func quantizedExpertGroups(in net: NFKMLXLanguageNet) -> [String: NFKMLXWeights.Quantization] {
        var groups = [String: NFKMLXWeights.Quantization]()
        for (index, block) in net.model.layers.enumerated() {
            guard let mixture = block.feedForward as? NFKLMMixtureFeedForward,
                  let glu = mixture.experts as? NFKLMSwitchGLU else { continue }
            for (name, layer) in [("gate_proj", glu.gate), ("up_proj", glu.up), ("down_proj", glu.down)] {
                guard let packed = layer as? NFKLMQuantizedSwitchLinear else { continue }
                groups["model.layers.\(index).mlp.experts.\(name)"] = NFKMLXWeights.Quantization(
                    bits: packed.bits, groupSize: packed.groupSize, mode: packed.mode)
            }
        }
        return groups
    }

    /// Replaces every mixture projection with a paged one reading from `store`, and returns what
    /// each must find there.
    ///
    /// - Parameters:
    ///   - quantization: how a group's experts are packed, or nil where they are floats.
    ///   - inputMajor: whether a group's experts are stored `[in, out]`.
    static func installPagedExperts(into net: NFKMLXLanguageNet, store: NFKMLXExpertStore,
                                    quantization: (String) -> NFKMLXWeights.Quantization?,
                                    inputMajor: (String) -> Bool) throws
        -> [(group: String, experts: Int, parts: [String])] {
        var layers = [(group: String, experts: Int, parts: [String])]()
        for (index, block) in net.model.layers.enumerated() {
            guard let mixture = block.feedForward as? NFKLMMixtureFeedForward else { continue }
            let projections: [(String, NFKLMExpertLinear)]
            if let glu = mixture.experts as? NFKLMSwitchGLU {
                projections = [("gate_proj", glu.gate), ("up_proj", glu.up), ("down_proj", glu.down)]
            } else if let fused = mixture.experts as? NFKLMFusedSwitchGLU {
                projections = [("gate_up_proj", fused.gateUp), ("down_proj", fused.down)]
            } else {
                continue
            }
            for (name, layer) in projections {
                let group = "model.layers.\(index).mlp.experts.\(name)"
                let packing = quantization(group)
                let paged = NFKLMPagedSwitchLinear(
                    pager: NFKMLXExpertPager(store: store, group: group), experts: layer.expertCount,
                    outputSize: layer.outputSize, inputSize: layer.inputSize, quantization: packing,
                    inputMajor: inputMajor(group))
                try mixture.experts.update(modules: ModuleChildren.unflattened([(name, paged)]),
                                           verify: .noUnusedKeys)
                let parts = packing.map { $0.mode == .affine ? ["weight", "scales", "biases"] : ["weight", "scales"] }
                layers.append((group, layer.expertCount, parts ?? ["weight"]))
            }
        }
        return layers
    }

    /// Loads a release with its routed experts paged: every other tensor loads as a resident load
    /// takes it, and the experts are filed in a new store as byte ranges of the mapped release.
    static func loadPagedWeights(into net: NFKMLXLanguageNet, fromDirectory directory: URL,
                                 precision: NFKMLXWeightPrecision, inventory: NFKMLXExpertInventory,
                                 cacheByteBudget: Int) throws {
        let tied = net.lmHead == nil
        let files = try NFKMLXReleaseWeights.files(inDirectory: directory)
        var recorded: NFKMLXWeights.Quantization?
        var stored = precision
        let releaseEntries = try packedReleaseEntries(inDirectory: directory, net: net)
        if let releaseEntries {
            try NFKMLXQuantization.matchStructure(toRelease: releaseEntries, on: net)
            stored = .checkpoint
        }
        let kept: [(String, MLXArray)]
        if files.count == 1 {
            let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: files[0], reading: {
                !(tied && $0.hasPrefix("lm_head.")) && !inventory.isExpert($0)
            })
            try NFKMLXQuantization.matchStructure(of: checkpoint, on: net)
            recorded = checkpoint.quantization
            if checkpoint.quantization != nil {
                stored = .checkpoint
            }
            kept = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
                if (tied && key.hasPrefix("lm_head.")) || inventory.isExpert(key) { return nil }
                let keeps = stored == .checkpoint || (value.dtype != .float16 && value.dtype != .bfloat16)
                return (key, keeps ? value : value.asType(.float32))
            }
        } else {
            kept = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: stored, remap: {
                (tied && $0.hasPrefix("lm_head.")) || inventory.isExpert($0) ? nil : $0
            })
        }
        var packedGroups = [String: Bool]()
        var inputMajorGroups = Set<String>()
        for tensor in inventory.experts.values {
            for slice in tensor.slices {
                if slice.part == "scales" {
                    packedGroups[slice.group] = tensor.entry.dtype == "U8"
                }
                if slice.inputMajor {
                    inputMajorGroups.insert(slice.group)
                }
            }
        }
        let store = NFKMLXExpertStore(cacheByteBudget: cacheByteBudget)
        let releaseGroups = quantizedExpertGroups(in: net)
        let layers = try installPagedExperts(into: net, store: store, quantization: { group in
            if let release = releaseGroups[group] { return release }
            guard let mxfp4 = packedGroups[group] else { return nil }
            return mxfp4 ? mxfp4Quantization : recorded
        }, inputMajor: inputMajorGroups.contains)
        try inventory.register(into: store, precision: stored)
        let prepared = releaseWeights(kept)
        try NFKMLXWeights.apply(prepared, to: net, verifyShapes: true)
        try NFKMLXExpertInventory.verifyComplete(store, layers: layers)
        net.expertStore = store
    }

    /// Loads a released checkpoint. The module's keys are the checkpoint's, so nothing is remapped
    /// beyond stacking a mixture's per-expert tensors.
    ///
    /// Every weight here is at most two-dimensional, so none of the convolution transposes the vision
    /// models apply have any part in this path. A fine-tuned checkpoint `NFKMLXWeights` wrote carries
    /// the same keys and loads here too. Introduced in InferKit 0.4.0.
    public static func loadWeights(into net: NFKMLXLanguageNet, from url: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        let tied = net.lmHead == nil
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url, reading: {
            !isPerExpert($0) && !(tied && $0.hasPrefix("lm_head."))
        })
        // A quantized checkpoint reshapes the module to match and loads at its stored dtypes: the
        // packed weights are uint32 whatever the request, and the scales keep the precision the
        // quantization was computed at.
        try NFKMLXQuantization.matchStructure(of: checkpoint, on: net)
        let keepStored = precision == .checkpoint || checkpoint.quantization != nil
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            // A tied release still ships `lm_head.weight`: in Qwen3-0.6B it is byte-identical to
            // `model.embed_tokens.weight`, so the module keeps one copy and the file's duplicate is
            // dropped rather than loaded into a projection the tied model does not have.
            if tied && key.hasPrefix("lm_head.") { return nil }
            let keeps = keepStored || (value.dtype != .float16 && value.dtype != .bfloat16)
            return (key, keeps ? value : value.asType(.float32))
        }
        let prepared = releaseWeights(mapped)
        try installPackedExperts(into: net, from: prepared)
        try NFKMLXWeights.apply(prepared, to: net, verifyShapes: true)
    }

    /// Builds a text-generation backend from local weights and a tokenizer.
    ///
    /// A nil `weightsURL` builds random weights, which run and produce nonsense — useful for proving
    /// the pipeline without a download. Without a tokenizer the backend cannot encode a prompt and
    /// reports so rather than guessing.
    public static func backend(weightsURL: URL?, tokenizer: NFKTokenizer?,
                               configuration: NFKMLXLanguageConfiguration = .qwen3_0_6B,
                               options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXLanguageBackend(net: net, tokenizer: tokenizer, identifier: modelName,
                                     options: options)
    }

    /// Loads a release from its directory, following the shard index when there is one.
    ///
    /// @discussion A release past about a gigabyte is split into `model-0000N-of-0000M.safetensors`
    /// with a `model.safetensors.index.json` naming which shard holds each tensor. Every size above
    /// 0.6B ships that way, so a loader that only reads `model.safetensors` covers the smallest model
    /// and nothing else. Introduced in InferKit 0.4.0.
    public static func loadWeights(into net: NFKMLXLanguageNet, fromDirectory directory: URL,
                                   precision requested: NFKMLXWeightPrecision = .float32) throws {
        // An mlx_lm quantized release loads as stored: its packed words are integers whatever the
        // request, and its scales keep the precision mlx_lm computed them at.
        let releaseEntries = try packedReleaseEntries(inDirectory: directory, net: net)
        let precision = releaseEntries == nil ? requested : .checkpoint
        // Refuse a release whose weights alone exceed the memory budget, before materializing any,
        // so "the process died" becomes an error naming the shortfall.
        try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: precision)
        if let releaseEntries {
            try NFKMLXQuantization.matchStructure(toRelease: releaseEntries, on: net)
        }
        // A single-file release routes through the checkpoint reader, which is where a quantized
        // save's metadata lives; a sharded release takes the merge path, quantized (mlx_lm) or not.
        let files = try NFKMLXReleaseWeights.files(inDirectory: directory)
        if files.count == 1 {
            try loadWeights(into: net, from: files[0], precision: precision)
            return
        }
        // A tied release still ships `lm_head.weight`, byte-identical to the embedding, and a tied
        // module has no projection to put it in.
        let tied = net.lmHead == nil
        // Each expert stored as its own tensor stays lazy: stacking reads it as the stack forms, so the
        // stored experts never sit whole beside the stacked ones.
        let kept: (String) -> String? = { tied && $0.hasPrefix("lm_head.") ? nil : $0 }
        let merged = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision,
                                                                 remap: { isPerExpert($0) ? nil : kept($0) })
            + NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision,
                                             remap: { isPerExpert($0) ? kept($0) : nil })
        let prepared = releaseWeights(merged)
        try installPackedExperts(into: net, from: prepared)
        try NFKMLXWeights.apply(prepared, to: net, verifyShapes: true)
    }

    /// The release's tensors under the module's parameter names, header only, where its `config.json`
    /// states an mlx_lm quantization (`quantization`, or a `quantization_config` with no
    /// `quant_method`); nil for any other release. A release's per-expert tensors are filed under the
    /// stacked name the module holds, at the stacked shape.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` for a quantization this package does not read
    ///   (``NFKMLXQuantization/releaseQuantization(configJSON:)``), or for mlx_lm's gpt-oss layout,
    ///   which stores the gate and up projections apart where this module holds them fused.
    static func packedReleaseEntries(inDirectory directory: URL, net: NFKMLXLanguageNet) throws
        -> [String: NFKMLXSafetensorsEntry]? {
        guard try statesMLXQuantization(configURL: directory.appendingPathComponent("config.json")) else { return nil }
        var entries = [String: NFKMLXSafetensorsEntry]()
        var perExpert = [String: (entry: NFKMLXSafetensorsEntry, experts: Int)]()
        for file in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            for (key, entry) in try NFKMLXSafetensors.entries(inFile: file) {
                let name = moduleKey(forRelease: key)
                let range = NSRange(name.startIndex..., in: name)
                if let match = expertPattern.firstMatch(in: name, range: range),
                   let prefix = Range(match.range(at: 1), in: name),
                   let index = Range(match.range(at: 2), in: name).flatMap({ Int(name[$0]) }),
                   let projection = Range(match.range(at: 3), in: name),
                   let part = Range(match.range(at: 4), in: name) {
                    let stacked = "\(name[prefix]).\(name[projection]).\(name[part])"
                    perExpert[stacked] = (entry, max(perExpert[stacked]?.experts ?? 0, index + 1))
                } else {
                    entries[name] = entry
                }
            }
        }
        for (name, stacked) in perExpert {
            entries[name] = NFKMLXSafetensorsEntry(dtype: stacked.entry.dtype, shape: [stacked.experts] + stacked.entry.shape,
                                                   start: 0, end: 0)
        }
        let fused = net.model.layers.contains { ($0.feedForward as? NFKLMMixtureFeedForward)?.experts is NFKLMFusedSwitchGLU }
        if fused, entries.keys.contains(where: { $0.hasSuffix(".mlp.experts.gate_proj.weight") }) {
            throw NFKMLXError.unsupportedConfiguration(
                "\(directory.lastPathComponent) stores gpt-oss's expert gate and up projections apart (mlx_lm's "
                + "layout); this loader reads them fused, as the release ships them")
        }
        return entries
    }

    /// Whether the release's `config.json` states an mlx_lm quantization this package reads.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` for one it does not read
    ///   (``NFKMLXQuantization/releaseQuantization(configJSON:)``).
    static func statesMLXQuantization(configURL: URL) throws -> Bool {
        guard let data = try? Data(contentsOf: configURL),
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return try NFKMLXQuantization.releaseQuantization(configJSON: json) != nil && isMLXQuantization(json)
    }

    /// Whether a `config.json` states mlx_lm's quantization rather than a transformers `quant_method`.
    static func isMLXQuantization(_ json: [String: Any]) -> Bool {
        // A key written as null states nothing, as `releaseQuantization(configJSON:)` reads it.
        let configuration = (!(json["quantization"] is [String: Any]) && !(json["quantization_config"] is [String: Any]))
            ? (json["text_config"] as? [String: Any]) ?? json : json
        if configuration["quantization"] is [String: Any] { return true }
        guard let stated = configuration["quantization_config"] as? [String: Any] else { return false }
        return !(stated["quant_method"] is String)
    }

    /// Whether a release stores its experts MXFP4 (`quantization_config.quant_method`), in which case
    /// it loads at `.checkpoint` precision: the packed experts stay packed, the bf16 attention stays
    /// bf16, and the fit check counts the bytes that will actually be resident.
    static func storesExpertsMXFP4(configURL: URL) -> Bool {
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let quantization = json["quantization_config"] as? [String: Any] else { return false }
        return (quantization["quant_method"] as? String) == "mxfp4"
    }

    /// Builds from a downloaded release directory holding the weights, `config.json`, and the
    /// tokenizer files. A directory whose tokenizer cannot be read is refused: the backend reads text
    /// prompts only, so it could answer none.
    public static func backend(directoryURL: URL,
                               options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, residency: .automatic, options: options)
    }

    /// Builds from a release directory, holding its routed experts as `residency` says.
    ///
    /// @discussion A mixture of experts is one stage, so the residencies differ only in paging:
    /// ``NFKMLXResidency/paged`` leaves the routed experts in the release and reads each as the router
    /// reaches it, ``NFKMLXResidency/resident`` loads them and fails where the release is known not to
    /// fit, and ``NFKMLXResidency/automatic`` pages only where the release is known not to fit whole.
    /// ``NFKMLXResidency/staged`` loads the model resident. A dense release has nothing to page; it
    /// streams the decoder layers it cannot hold under ``NFKMLXResidency/streamed``, and under
    /// ``NFKMLXResidency/automatic`` where the release does not fit whole, and loads resident otherwise.
    /// ``NFKMLXLanguageBackend/pagesExperts`` and ``NFKMLXLanguageBackend/streamedLayerCount`` report
    /// the placement. The release directory has to stay in place for the life of a paged or streamed
    /// backend. The weights load at float32. Introduced in InferKit 0.4.0.
    public static func backend(directoryURL: URL, residency: NFKMLXResidency,
                               options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: nil, residency: residency, options: options)
    }

    /// ``backend(directoryURL:residency:options:)`` with the weights at `precision`. A streamed decoder
    /// reads each streamed layer at `.checkpoint` as stored, so a pass reads half what it reads at
    /// `.float32` from a 16-bit release. Introduced in InferKit 0.4.0.
    public static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision?, residency: NFKMLXResidency,
                               options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        let (net, tokenizer) = try loadedReleaseWithTokenizer(at: directoryURL, precision: precision, residency: residency,
                                                              streams: true)
        return NFKMLXLanguageBackend(net: net, tokenizer: tokenizer, identifier: modelName,
                                     options: options, releaseURL: directoryURL)
    }

    /// Builds from a release directory together with a smaller release of the same family as the
    /// draft for speculative decoding.
    ///
    /// @discussion The draft proposes ``NFKMLXGenerationOptions/draftTokens`` tokens per round and
    /// the main model verifies them in one pass, producing the sequence it would have produced
    /// alone. The two must share a vocabulary, which is checked here; a Qwen3 0.6B drafting for a
    /// Qwen3 4B is the intended pairing. The draft loads at the same precision as the main model, and
    /// first, so the main model is planned against what the draft leaves of the working set.
    public static func backend(directoryURL: URL, draftDirectoryURL: URL,
                               options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        let (draft, _) = try loadedRelease(at: draftDirectoryURL)
        let (net, tokenizer) = try loadedReleaseWithTokenizer(at: directoryURL, streams: true,
                                                              reserving: heldBytes(of: draft))
        try verifyDraft(draft, sharesVocabularyWith: net)
        return NFKMLXLanguageBackend(net: net, tokenizer: tokenizer, identifier: modelName,
                                     options: options, draft: draft, releaseURL: directoryURL)
    }

    /// Builds from a release directory as `loadOptions` say: the weights at its precision (else as
    /// published, the precision each registered family is held to its reference at), its routed experts held as its residency plans
    /// them, its eligible layers quantized to its quantization bits, its draft beside it (quantized to the
    /// draft bits, else at the model's precision), and its context length and cache settings as the
    /// backend's generation defaults. ``NFKMLXOptionRegistrations`` registers the dense and mixture releases
    /// with it. Introduced in InferKit 0.4.0.
    static func backend(directoryURL: URL, loadOptions: NFKMLXLoadOptions) throws -> any NFKInferenceBackend {
        let precision = loadOptions.precision ?? .checkpoint
        // The draft loads first, so the model is planned against what the draft leaves of the working set.
        var draft: NFKMLXLanguageNet?
        if let draftURL = loadOptions.draftURL {
            // A draft quantized at load reads the release as stored; widening it first only raises the peak.
            let draftPrecision = loadOptions.draftBits == nil ? precision : .checkpoint
            let (drafter, _) = try loadedRelease(at: draftURL, precision: draftPrecision)
            try NFKMLXLoadOptions([NFKModelLoadOptionQuantizationBits: loadOptions.draftBits ?? 0])
                .applyWeightQuantization(to: drafter)
            draft = drafter
        }
        let (net, tokenizer) = try loadedReleaseWithTokenizer(at: directoryURL, precision: precision,
                                                              residency: loadOptions.residency, streams: true,
                                                              reserving: draft.map(heldBytes(of:)) ?? 0)
        if let draft {
            try verifyDraft(draft, sharesVocabularyWith: net)
        }
        if loadOptions.quantizationBits != nil {
            try net.requireHeldLayers(for: "quantize at load")
        }
        try loadOptions.applyWeightQuantization(to: net)
        return NFKMLXLanguageBackend(net: net, tokenizer: tokenizer, identifier: modelName,
                                     options: loadOptions.generationOptions(), draft: draft, releaseURL: directoryURL)
    }

    /// The bytes `module`'s parameters hold.
    static func heldBytes(of module: Module) -> Int {
        module.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
    }

    /// Refuses a draft whose vocabulary differs from the model's: a draft proposes the model's own token
    /// ids, so the two share a tokenizer.
    static func verifyDraft(_ draft: NFKMLXLanguageNet, sharesVocabularyWith net: NFKMLXLanguageNet) throws {
        guard draft.configuration.vocabularySize == net.configuration.vocabularySize else {
            throw NFKMLXError.unsupportedConfiguration(
                "the draft's vocabulary has \(draft.configuration.vocabularySize) entries where the "
                + "model's has \(net.configuration.vocabularySize); a draft must share the model's "
                + "tokenizer")
        }
    }

    /// The release's own Jinja `chat_template`, or nil where it ships none.
    ///
    /// @discussion An instruct release is trained on its template and answers a different question
    /// without one, so a caller renders a message list through it: in Swift by setting
    /// ``NFKMLXGenerationOptions/chatTemplate`` to `.jinja(template:)`, and from Objective-C by
    /// passing the string under ``NFKMLXGenerationParameterKey/chatTemplate``. The template is also
    /// what states a reasoning model's markers, so the chain comes back under `NFKOutputReasoning`
    /// only where the template is in play. Introduced in InferKit 0.4.0.
    @objc public static func chatTemplate(inDirectory directoryURL: URL) -> String? {
        NFKMLXReleaseChatTemplate(inDirectory: directoryURL)
    }

    /// The release's template with the begin and end markers its `tokenizer_config.json` names, which a
    /// template such as Llama 3's writes through `bos_token`; nil where the release ships none.
    static func releaseChatTemplate(inDirectory directoryURL: URL) -> NFKMLXChatTemplate? {
        guard let template = NFKMLXReleaseChatTemplate(inDirectory: directoryURL) else { return nil }
        var markers = (bos: "", eos: "")
        if let data = try? Data(contentsOf: directoryURL.appendingPathComponent("tokenizer_config.json")),
           let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            func marker(_ key: String) -> String {
                (json[key] as? String) ?? ((json[key] as? [String: Any])?["content"] as? String) ?? ""
            }
            markers = (marker("bos_token"), marker("eos_token"))
        }
        return .jinja(template: template, bosToken: markers.bos, eosToken: markers.eos)
    }

    /// The network and tokenizer a release directory describes.
    /// The network and tokenizer of a release a backend will read text through: a directory whose
    /// tokenizer cannot be read is refused.
    static func loadedReleaseWithTokenizer(at directoryURL: URL, precision: NFKMLXWeightPrecision? = nil,
                                           residency: NFKMLXResidency = .automatic, streams: Bool = false,
                                           reserving reserved: Int = 0)
        throws -> (NFKMLXLanguageNet, NFKTokenizer) {
        let (net, tokenizer) = try loadedRelease(at: directoryURL, requiringTokenizer: true, precision: precision,
                                                 residency: residency, streams: streams, reserving: reserved)
        guard let tokenizer else { throw NFKMLXError.unsupportedConfiguration("unreachable: the tokenizer was required") }
        return (net, tokenizer)
    }

    /// The network and tokenizer a release directory describes. A caller that reads text sets
    /// `requiringTokenizer`; a caller that feeds the network embeddings (the FLUX.2 text encoder) does not.
    /// `precision` nil loads at float32, or as stored for a release whose experts are packed; a caller
    /// whose release is too large at float32 passes `.checkpoint` (the Qwen3-8B inside FLUX.2 [klein]
    /// 9B is 30.5 GB at float32 and 15.3 GB as released). `streams` lets a dense release that does not
    /// fit stream the layers it cannot hold; it is for a caller that checks the stream after each pass
    /// (``NFKMLXLanguageNet/verifyStream()``), as the backends do, and every other caller holds the
    /// release whole or is refused. `reserved` is what a model held beside this one, such as a draft,
    /// takes from the working set the load plans against.
    static func loadedRelease(at directoryURL: URL, requiringTokenizer: Bool = false,
                              precision: NFKMLXWeightPrecision? = nil,
                              residency: NFKMLXResidency = .automatic, streams: Bool = false,
                              reserving reserved: Int = 0) throws -> (NFKMLXLanguageNet, NFKTokenizer?) {
        var configuration = try self.configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        if try statesMLXQuantization(configURL: directoryURL.appendingPathComponent("config.json")) {
            configuration.halfPrecisionPlacement = .mlx
        }
        let tokenizer: NFKTokenizer?
        var tokenizerRefusal: Error?
        do {
            tokenizer = try readTokenizer(inDirectory: directoryURL)
        } catch {
            tokenizer = nil
            tokenizerRefusal = error
        }
        let net = makeNet(configuration)
        let packed = try storesExpertsMXFP4(configURL: directoryURL.appendingPathComponent("config.json"))
            || packedReleaseEntries(inDirectory: directoryURL, net: net) != nil
        let budget = NFKMLXResidencyBudget.current()
        try loadWeights(into: net, fromDirectory: directoryURL,
                        precision: precision ?? (packed ? .checkpoint : .float32), residency: residency,
                        streams: streams,
                        budget: budget > 0 && reserved > 0 ? Swift.max(budget - reserved, 1) : budget, reserving: reserved)
        // Checked after the load, whose memory check refuses a release the machine cannot hold before
        // reading any weight: that refusal is the one a caller can act on, so it comes first.
        if requiringTokenizer, let tokenizerRefusal {
            throw tokenizerRefusal
        }
        return (net, tokenizer)
    }

    /// Loads a release held as `residency` plans it: paged where the plan pages the routed experts,
    /// streamed where a dense release's layers do not fit whole, and resident otherwise.
    ///
    /// @discussion The plan weighs the release from its shard headers. A model without routed
    /// experts has nothing to page; it streams as ``NFKMLXLayerStream/plannedLayers(residency:budget:footprint:fitsWhole:)``
    /// plans where `streams` allows it, and loads resident otherwise; an mlx_lm quantized release always
    /// loads resident. `reserved` is what something held beside the decoder, such as a draft, takes from
    /// the check a resident load makes; `budget` already leaves it out.
    static func loadWeights(into net: NFKMLXLanguageNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision, residency: NFKMLXResidency, streams: Bool = true,
                            budget: Int = NFKMLXResidencyBudget.current(), reserving reserved: Int = 0) throws {
        let inventory = try NFKMLXExpertInventory.planning(inDirectory: directory, precision: precision,
                                                           classify: expertSlices(forRelease:entry:))
        let plan = try NFKMLXResidencyBudget.plan([inventory.footprint], residency: residency, budget: budget)
        guard plan.pagesExperts else {
            let tied = net.lmHead == nil
            let moduleKey: (String) -> String? = { tied && $0.hasPrefix("lm_head.") ? nil : $0 }
            // An mlx_lm quantized release packs its layers as it loads, and a stream installs tensors into
            // the layers as built, so it loads as stored.
            let packed = try statesMLXQuantization(configURL: directory.appendingPathComponent("config.json"))
            let streamed = streams && inventory.experts.isEmpty && !net.configuration.isMixtureOfExperts && !packed
                ? try streamedLayers(directory: directory, precision: precision, residency: residency, budget: budget,
                                     reserving: reserved, moduleKey: moduleKey)
                : []
            guard streamed.isEmpty else {
                try loadStreamedWeights(into: net, fromDirectory: directory, precision: precision, streamed: streamed,
                                        moduleKey: moduleKey)
                return
            }
            try loadWeights(into: net, fromDirectory: directory, precision: precision)
            return
        }
        try loadPagedWeights(into: net, fromDirectory: directory, precision: precision,
                             inventory: inventory, cacheByteBudget: plan.expertCacheBytes)
    }

    /// The tokenizer a release directory describes, built without loading the weights, or nil where
    /// ``readTokenizer(inDirectory:)`` refuses it.
    static func releaseTokenizer(inDirectory directory: URL) -> NFKTokenizer? {
        try? readTokenizer(inDirectory: directory)
    }

    /// The layer a decoder module key of the form `model.layers.<n>.<rest>` belongs to, and its name
    /// inside the layer.
    static func streamedLayer(ofModuleKey key: String) -> (layer: Int, name: String)? {
        guard key.hasPrefix("model.") else { return nil }
        return NFKMLXLayerStream.layer(ofDecoderName: String(key.dropFirst("model.".count)))
    }

    /// The decoder layers a dense release's load leaves in the release under `residency`, as
    /// ``NFKMLXLayerStream/plannedLayers(residency:budget:footprint:fitsWhole:)`` plans them; empty where it
    /// holds them all.
    ///
    /// - Parameters:
    ///   - moduleKey: the decoder's module key for a release tensor, or nil for one the decoder does not
    ///     take, such as a multimodal release's vision tower.
    ///   - reserved: what something held beside the decoder, such as a draft, takes from the check a
    ///     resident load makes; `budget` already leaves it out.
    ///   - besides: the bytes the caller already holds beside the decoder from the same release, such as
    ///     a vision tower and its connector, which the stream's plan counts with the decoder's fixed bytes.
    static func streamedLayers(directory: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                               budget: Int, reserving reserved: Int = 0, besides: Int = 0,
                               moduleKey: (String) -> String?) throws -> Set<Int> {
        try NFKMLXLayerStream.plannedLayers(residency: residency, budget: budget, footprint: {
            let decoder = try NFKMLXLayerStream.footprint(directory: directory, precision: precision,
                                                          includes: { moduleKey($0) != nil },
                                                          layerOf: { moduleKey($0).flatMap(streamedLayer(ofModuleKey:))?.layer })
            return NFKMLXStageFootprint(bytes: decoder.bytes + besides, layerBytes: decoder.layerBytes)
        }, fitsWhole: {
            try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: precision, reserve: reserved)
        })
    }

    /// Loads a dense release into `net` with its `streamed` layers left in the release, read in their
    /// turn on every pass (``NFKMLXLanguageNet/layerStream``), and every other tensor `moduleKey` names
    /// held. `conversion` converts every floating tensor, held or streamed, to that element type in
    /// place of `precision`'s.
    ///
    /// @discussion A quantized checkpoint is refused: its layers are rebuilt to the stored packing before
    /// a load, and a stream installs the release's tensors into the layers as built.
    static func loadStreamedWeights(into net: NFKMLXLanguageNet, fromDirectory directory: URL,
                                    precision: NFKMLXWeightPrecision, streamed: Set<Int>,
                                    converting conversion: DType? = nil,
                                    moduleKey: @escaping (String) -> String?) throws {
        let files = try NFKMLXReleaseWeights.files(inDirectory: directory)
        if files.count == 1, try NFKMLXWeights.materializedCheckpoint(url: files[0], reading: { _ in false }).quantization != nil {
            throw NFKMLXError.unsupportedConfiguration(
                "\(directory.lastPathComponent) is a quantized checkpoint, which does not stream; "
                + "load it with a residency that holds every layer")
        }
        guard let vacated = try NFKMLXLayerStream.vacating(streamed, of: net.model.layers, prefix: "model.layers.",
                                                           directory: directory, precision: precision,
                                                           converting: conversion, layerOf: { key in
            moduleKey(key).flatMap(streamedLayer(ofModuleKey:))
        }) else { return }
        let held: (String) -> String? = { key in
            guard let name = moduleKey(key) else { return nil }
            return streamedLayer(ofModuleKey: name).map { streamed.contains($0.layer) } == true ? nil : name
        }
        let arrays = try conversion.map { try NFKMLXReleaseWeights.arrays(inDirectory: directory, converting: $0, remap: held) }
            ?? NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision, remap: held)
        try NFKMLXWeights.apply(arrays + vacated.placeholders, to: net, verifyShapes: true)
        net.layerStream = vacated.stream
        vacated.stream.prime()
    }

    /// The tokenizer a release directory describes, built without loading the weights.
    ///
    /// @discussion The scheme comes from ``tokenizerScheme(tokenizerJSON:tokenizerConfiguration:)``, so a
    /// tokenizer this reader would encode differently from the release is refused rather than built. A
    /// byte-level vocabulary is encoded under its own named pre-tokenization; another pattern encodes
    /// the same text to different, valid-looking ids. The special tokens live in `tokenizer_config.json`
    /// and `tokenizer.json`'s `added_tokens` rather than `vocab.json`, so a chat template's or an
    /// embedder's markers resolve rather than encoding as ordinary text.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` naming the part of the tokenizer this reader
    ///   does not reproduce, or the files it found missing.
    static func readTokenizer(inDirectory directory: URL) throws -> NFKTokenizer {
        func object(_ name: String) -> [String: Any]? {
            guard let data = try? Data(contentsOf: directory.appendingPathComponent(name)) else { return nil }
            return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        }
        let tokenizerJSON = object("tokenizer.json")
        let configuration = object("tokenizer_config.json")
        let scheme = try tokenizerScheme(tokenizerJSON: tokenizerJSON, tokenizerConfiguration: configuration)
        let (specials, endToken) = specialTokens(inDirectory: directory, tokenizerJSON: tokenizerJSON)
        guard case .byteLevel(let pretokenization, let composes, let ignoresMerges) = scheme else {
            guard let tokenizerJSON, let tokenizer = NFKMLXMetaspaceTokenizer(
                tokenizerJSON: tokenizerJSON, endToken: endToken, prependsEveryRun: scheme == .prependedMetaspace) else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(directory.lastPathComponent)'s tokenizer.json has no BPE vocabulary and merges")
            }
            return tokenizer
        }
        var spec: [String: Any] = ["type": "bpe-bytelevel", "pretokenizer": pretokenization,
                                   "specialTokens": specials, "ignoreMerges": ignoresMerges]
        if composes { spec["normalizer"] = "nfc" }
        var manifest: [String: Any] = ["tokenizer": spec]
        if let endToken { manifest["eosTokenId"] = endToken }
        // A release that ships only tokenizer.json (gpt-oss), or a vocab.json without the matching
        // merges.txt (Granite Speech), has its vocabulary and merges extracted from tokenizer.json into
        // the vocab.json / merges.txt pair the core reader takes.
        var files = directory
        let hasVocab = FileManager.default.fileExists(atPath: directory.appendingPathComponent("vocab.json").path)
        let hasMerges = FileManager.default.fileExists(atPath: directory.appendingPathComponent("merges.txt").path)
        if tokenizerJSON != nil || !(hasVocab && hasMerges) {
            guard let tokenizerJSON, let extracted = byteLevelFiles(fromTokenizerJSON: tokenizerJSON) else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(directory.lastPathComponent) has no readable tokenizer (tokenizer.json, or vocab.json and "
                    + "merges.txt)")
            }
            files = extracted
        }
        let tokenizer = try NFKTokenizer(forManifest: manifest, directory: files)
        // The core reader keeps the vocabulary and merges it read, so the extracted pair is scratch.
        if files != directory {
            try? FileManager.default.removeItem(at: files)
        }
        return tokenizer
    }

    /// How a release's tokenizer encodes text, as far as this package reproduces it.
    enum TokenizerScheme: Equatable {
        /// Byte-level BPE under a named pre-tokenization (`NFKByteLevelBPETokenizer`'s names), composing
        /// each segment to NFC first where `composes`, and taking a pretoken the vocabulary holds whole
        /// as one token where `ignoresMerges`.
        case byteLevel(pretokenization: String, composes: Bool, ignoresMerges: Bool)
        /// Mistral's SentencePiece-style BPE: spaces become `▁`, one `▁` leads the text, and a
        /// character the vocabulary lacks falls back to its UTF-8 bytes (Mixtral, Codestral-Mamba).
        case metaspace
        /// The same BPE written the Llama way (Seed-X): a normalizer prepends `▁` to every run of text
        /// between added tokens and replaces spaces, with no pre-tokenizer.
        case prependedMetaspace
    }

    /// The `Split` patterns this package reproduces, keyed by their exact text in `tokenizer.json`.
    static let pretokenizationPatterns: [String: String] = [
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#: "qwen2",
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#: "qwen35",
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#: "o200k",
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#: "llama3",
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#: "nemotron",
    ]

    /// HY-MT 1.8B's three `Split` patterns, applied in turn, as its `tokenizer.json` states them.
    static let hunyuanPretokenizationPatterns: [String?] = [
        #"\p{N}{1,3}"#,
        "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
        ##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\##r\##n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\##r\##n]*|\s*[\##r\##n]+|\s+(?!\S)|\s+"##,
    ]

    /// The regex of a `Split` that keeps its matches as pieces (Isolated, not inverted), or nil.
    private static func isolatingPattern(_ step: [String: Any]) -> String? {
        guard step["type"] as? String == "Split", step["behavior"] as? String == "Isolated",
              !((step["invert"] as? NSNumber)?.boolValue ?? false) else { return nil }
        return (step["pattern"] as? [String: Any])?["Regex"] as? String
    }

    /// The scheme a release's parsed `tokenizer.json` and `tokenizer_config.json` describe, read strictly:
    /// every field that changes the ids must hold a value this package reproduces.
    ///
    /// @discussion Two schemes are read. A byte-level BPE takes a `Split` whose regex is one of
    /// ``pretokenizationPatterns`` byte for byte followed by a `ByteLevel` that does not split again, a
    /// lone `ByteLevel` with its own GPT-2 regex, or SmolLM's individual `Digits` before that `ByteLevel`;
    /// its normalizer is none or NFC, and its model has no
    /// byte fallback, dropout, or affixes. The Metaspace scheme is Mistral's exact form: `▁` prepended
    /// to the first segment only, no split, and a byte-fallback model with `<unk>`. A release with no
    /// `tokenizer.json` is read only when `tokenizer_config.json` names Qwen2's tokenizer class, whose
    /// pattern and NFC normalization are fixed by the class.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` naming the first field outside that set.
    static func tokenizerScheme(tokenizerJSON: [String: Any]?, tokenizerConfiguration: [String: Any]?)
        throws -> TokenizerScheme {
        guard let tokenizerJSON else {
            let tokenizerClass = tokenizerConfiguration?["tokenizer_class"] as? String
            guard tokenizerClass == "Qwen2Tokenizer" || tokenizerClass == "Qwen2TokenizerFast" else {
                throw NFKMLXError.unsupportedConfiguration(
                    "the release has no tokenizer.json, and its tokenizer class (\(tokenizerClass ?? "none")) "
                    + "does not fix the pre-tokenization")
            }
            return .byteLevel(pretokenization: "qwen2", composes: true, ignoresMerges: false)
        }
        guard let model = tokenizerJSON["model"] as? [String: Any], model["type"] as? String == "BPE" else {
            let type = ((tokenizerJSON["model"] as? [String: Any])?["type"] as? String) ?? "none"
            throw NFKMLXError.unsupportedConfiguration("the tokenizer model is \(type), not the BPE this reader encodes")
        }
        func present(_ value: Any?) -> Bool { value != nil && !(value is NSNull) }
        func text(_ value: Any?) -> String? { value as? String }
        func flag(_ value: Any?) -> Bool { (value as? NSNumber)?.boolValue ?? false }
        if present(model["dropout"]) {
            throw NFKMLXError.unsupportedConfiguration("the tokenizer applies BPE dropout")
        }
        if !(text(model["continuing_subword_prefix"]) ?? "").isEmpty || !(text(model["end_of_word_suffix"]) ?? "").isEmpty {
            throw NFKMLXError.unsupportedConfiguration("the tokenizer's BPE marks subwords with an affix")
        }
        let preTokenizer = tokenizerJSON["pre_tokenizer"] as? [String: Any]
        if preTokenizer?["type"] as? String == "Metaspace" {
            return try metaspaceScheme(tokenizerJSON, model: model, preTokenizer: preTokenizer ?? [:])
        }
        if preTokenizer == nil, isPrependingNormalizer(tokenizerJSON["normalizer"]) {
            return try prependedMetaspaceScheme(tokenizerJSON, model: model)
        }
        if flag(model["byte_fallback"]) {
            throw NFKMLXError.unsupportedConfiguration(
                "the tokenizer falls back to bytes under a \(preTokenizer?["type"] as? String ?? "missing") "
                + "pre-tokenizer, which is not Mistral's Metaspace form")
        }
        let composes = try composesToNFC(tokenizerJSON["normalizer"])
        guard (tokenizerJSON["decoder"] as? [String: Any])?["type"] as? String == "ByteLevel" else {
            throw NFKMLXError.unsupportedConfiguration("the tokenizer's decoder is not ByteLevel")
        }
        let pretokenization = try byteLevelPretokenization(preTokenizer)
        return .byteLevel(pretokenization: pretokenization, composes: composes,
                          ignoresMerges: flag(model["ignore_merges"]))
    }

    /// The named pre-tokenization a byte-level `pre_tokenizer` states.
    private static func byteLevelPretokenization(_ preTokenizer: [String: Any]?) throws -> String {
        guard let preTokenizer else {
            throw NFKMLXError.unsupportedConfiguration("the tokenizer has no pre-tokenizer")
        }
        func byteLevel(_ step: [String: Any], splits: Bool) -> Bool {
            step["type"] as? String == "ByteLevel" && !((step["add_prefix_space"] as? NSNumber)?.boolValue ?? false)
                && ((step["use_regex"] as? NSNumber)?.boolValue ?? true) == splits
        }
        if byteLevel(preTokenizer, splits: true) {
            return "gpt2"
        }
        let steps = (preTokenizer["type"] as? String == "Sequence") ? (preTokenizer["pretokenizers"] as? [[String: Any]]) ?? [] : [preTokenizer]
        if steps.count == 2, steps[0]["type"] as? String == "Digits",
           (steps[0]["individual_digits"] as? NSNumber)?.boolValue == true, byteLevel(steps[1], splits: true) {
            return "smollm"
        }
        if steps.count == 4, byteLevel(steps[3], splits: false),
           steps.prefix(3).map(isolatingPattern) == hunyuanPretokenizationPatterns {
            return "hunyuan"
        }
        guard steps.count == 2, steps[0]["type"] as? String == "Split", byteLevel(steps[1], splits: false) else {
            let types = steps.map { $0["type"] as? String ?? "?" }.joined(separator: ", ")
            throw NFKMLXError.unsupportedConfiguration(
                "the tokenizer's pre-tokenizer (\(types)) is not a Split followed by a ByteLevel")
        }
        let split = steps[0]
        let behavior = split["behavior"] as? String
        let inverted = (split["invert"] as? NSNumber)?.boolValue ?? false
        // Each pattern matches every character, so keeping the matches (Isolated) and removing what
        // lies between them (Removed, inverted) cut the text identically.
        guard (behavior == "Isolated" && !inverted) || (behavior == "Removed" && inverted) else {
            throw NFKMLXError.unsupportedConfiguration(
                "the tokenizer's Split behaves as \(behavior ?? "?")\(inverted ? ", inverted" : ""), which this "
                + "reader does not reproduce")
        }
        guard let regex = (split["pattern"] as? [String: Any])?["Regex"] as? String,
              let name = pretokenizationPatterns[regex] else {
            throw NFKMLXError.unsupportedConfiguration(
                "the tokenizer's Split pattern is not one of the pre-tokenizations this reader reproduces "
                + "(qwen2, qwen35, o200k, llama3, nemotron)")
        }
        return name
    }

    /// Whether a `normalizer` composes to NFC; nil and an empty sequence change nothing.
    private static func composesToNFC(_ normalizer: Any?) throws -> Bool {
        guard let normalizer = normalizer as? [String: Any] else { return false }
        let steps = normalizer["type"] as? String == "Sequence"
            ? (normalizer["normalizers"] as? [[String: Any]]) ?? [] : [normalizer]
        let types = steps.compactMap { $0["type"] as? String }
        guard types.count == steps.count, types.allSatisfy({ $0 == "NFC" }) else {
            throw NFKMLXError.unsupportedConfiguration(
                "the tokenizer normalizes with \(types.joined(separator: ", ")), which this reader does not reproduce")
        }
        return !types.isEmpty
    }

    /// The Metaspace scheme, accepted in Mistral's exact form only.
    private static func metaspaceScheme(_ tokenizerJSON: [String: Any], model: [String: Any],
                                        preTokenizer: [String: Any]) throws -> TokenizerScheme {
        let decoder = tokenizerJSON["decoder"] as? [String: Any]
        let decoderSteps = ((decoder?["decoders"] as? [[String: Any]]) ?? []).compactMap { $0["type"] as? String }
        guard preTokenizer["replacement"] as? String == "\u{2581}",
              preTokenizer["prepend_scheme"] as? String == "first",
              !((preTokenizer["split"] as? NSNumber)?.boolValue ?? true),
              tokenizerJSON["normalizer"] == nil || tokenizerJSON["normalizer"] is NSNull,
              (model["byte_fallback"] as? NSNumber)?.boolValue == true,
              decoderSteps == ["Replace", "ByteFallback", "Fuse", "Strip"] else {
            throw NFKMLXError.unsupportedConfiguration(
                "the tokenizer's Metaspace form is not Mistral's (a leading ▁ on the first segment, no split, "
                + "byte fallback)")
        }
        return .metaspace
    }

    /// Whether a `normalizer` is the Llama-style pair: prepend `▁`, then replace each space with `▁`.
    private static func isPrependingNormalizer(_ normalizer: Any?) -> Bool {
        guard let normalizer = normalizer as? [String: Any], normalizer["type"] as? String == "Sequence",
              let steps = normalizer["normalizers"] as? [[String: Any]], steps.count == 2 else { return false }
        return steps[0]["type"] as? String == "Prepend" && steps[0]["prepend"] as? String == "\u{2581}"
            && steps[1]["type"] as? String == "Replace"
            && (steps[1]["pattern"] as? [String: Any])?["String"] as? String == " "
            && steps[1]["content"] as? String == "\u{2581}"
    }

    /// The Llama-style scheme (Seed-X): the prepending normalizer, no pre-tokenizer, byte fallback, and
    /// the Metaspace decoder.
    private static func prependedMetaspaceScheme(_ tokenizerJSON: [String: Any], model: [String: Any]) throws
        -> TokenizerScheme {
        let decoder = tokenizerJSON["decoder"] as? [String: Any]
        let decoderSteps = ((decoder?["decoders"] as? [[String: Any]]) ?? []).compactMap { $0["type"] as? String }
        guard (model["byte_fallback"] as? NSNumber)?.boolValue == true,
              decoderSteps == ["Replace", "ByteFallback", "Fuse", "Strip"] else {
            throw NFKMLXError.unsupportedConfiguration(
                "the tokenizer prepends ▁ in its normalizer but lacks the byte fallback and Metaspace decoder of "
                + "the Llama-style form")
        }
        return .prependedMetaspace
    }

    /// Writes a tokenizer.json's byte-level BPE vocabulary and merges as the `vocab.json` and
    /// `merges.txt` pair the core reader takes, into a scratch directory, and returns it. The caller removes
    /// the directory once the tokenizer has read it.
    static func byteLevelFiles(fromTokenizerJSON url: URL) -> URL? {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return byteLevelFiles(fromTokenizerJSON: json)
    }

    /// ``byteLevelFiles(fromTokenizerJSON:)-(URL)`` over an already parsed tokenizer.json.
    static func byteLevelFiles(fromTokenizerJSON json: [String: Any]) -> URL? {
        guard let model = json["model"] as? [String: Any],
              let vocabulary = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [Any] else { return nil }
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        guard (try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)) != nil,
              let vocabularyData = try? JSONSerialization.data(withJSONObject: vocabulary),
              (try? vocabularyData.write(to: scratch.appendingPathComponent("vocab.json"))) != nil else {
            return nil
        }
        var mergesText = "#version: 0.2\n"
        for entry in merges {
            if let pair = entry as? [String], pair.count == 2 {
                mergesText += pair[0] + " " + pair[1] + "\n"
            } else if let text = entry as? String {
                mergesText += text + "\n"
            }
        }
        guard (try? mergesText.write(to: scratch.appendingPathComponent("merges.txt"),
                                     atomically: true, encoding: .utf8)) != nil else { return nil }
        return scratch
    }

    /// The special-token literals a release declares (`added_tokens_decoder` in
    /// `tokenizer_config.json`, `added_tokens.json`, or `added_tokens` in `tokenizer.json`) and the id of
    /// its `eos_token`.
    static func specialTokens(inDirectory directory: URL, tokenizerJSON: [String: Any]? = nil) -> ([String: Int], Int?) {
        var literals = [String: Int]()
        for entry in (tokenizerJSON?["added_tokens"] as? [[String: Any]]) ?? [] {
            if let content = entry["content"] as? String, let id = (entry["id"] as? NSNumber)?.intValue {
                literals[content] = id
            }
        }
        var endToken: Int?
        if let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            for (id, entry) in (json["added_tokens_decoder"] as? [String: [String: Any]]) ?? [:] {
                if let content = entry["content"] as? String, let number = Int(id) {
                    literals[content] = number
                }
            }
            let end = (json["eos_token"] as? String) ?? ((json["eos_token"] as? [String: Any])?["content"] as? String)
            endToken = end.flatMap { literals[$0] }
        }
        if let data = try? Data(contentsOf: directory.appendingPathComponent("added_tokens.json")),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: NSNumber] {
            for (content, id) in json { literals[content] = id.intValue }
        }
        return (literals, endToken)
    }

    /// The tokens a release marks special (`"special": true` in `tokenizer.json`'s `added_tokens` or
    /// `tokenizer_config.json`'s `added_tokens_decoder`), each literal's id.
    static func specialTokenIDs(inDirectory directory: URL) -> [String: Int] {
        func object(_ name: String) -> [String: Any]? {
            guard let data = try? Data(contentsOf: directory.appendingPathComponent(name)) else { return nil }
            return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        }
        var specials = [String: Int]()
        for entry in object("tokenizer.json")?["added_tokens"] as? [[String: Any]] ?? []
        where (entry["special"] as? Bool) == true {
            if let content = entry["content"] as? String, let id = (entry["id"] as? NSNumber)?.intValue {
                specials[content] = id
            }
        }
        for (key, entry) in object("tokenizer_config.json")?["added_tokens_decoder"] as? [String: [String: Any]] ?? [:]
        where (entry["special"] as? Bool) == true {
            if let content = entry["content"] as? String, let id = Int(key) {
                specials[content] = id
            }
        }
        return specials
    }

    /// The special tokens a reply decodes without, as the reference decodes with `skip_special_tokens`:
    /// every one but those a marker the run's reasoning or tool-call reader finds contains, such as
    /// harmony's `<|channel|>` or Mistral's `[TOOL_CALLS]`.
    static func skippedSpecialTokens(_ specials: [String: Int], keepingMarkers markers: [String]) -> Set<Int> {
        Set(specials.filter { literal, _ in !markers.contains { $0.contains(literal) } }.values)
    }

    /// Decodes ids to text without the `skipped` ones.
    static func replyDecoder(_ tokenizer: NFKTokenizer, skipping skipped: Set<Int>) -> ([Int]) -> String {
        { ids in tokenizer.decode(ids.filter { !skipped.contains($0) }.map { NSNumber(value: $0) }) }
    }

    /// The Objective-C entry: builds from a release directory, reading its `config.json`, tokenizer,
    /// and shards. Generation options are set per request through ``NFKMLXGenerationParameterKey``,
    /// which is how an Objective-C caller reaches what the Swift `options:` parameter carries. Run
    /// inference off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, options: NFKMLXGenerationOptions())
    }

    /// The Objective-C entry holding the release's routed experts as `residency` says; see
    /// ``backend(directoryURL:residency:options:)``. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, residency: residency, options: NFKMLXGenerationOptions())
    }

    /// The Objective-C entry loading the weights at `precision` and holding the release as `residency`
    /// says; see ``backend(directoryURL:precision:residency:options:)``. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:precision:residency:error:)
    public static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: residency,
                    options: NFKMLXGenerationOptions())
    }

    /// The Objective-C entry for speculative decoding: a release directory and a smaller release
    /// of the same family as the draft. `NFKMLXGenerationParameterKey.draftTokens` sets the
    /// proposals per round on a request.
    @objc(backendWithDirectoryURL:draftDirectoryURL:error:)
    public static func backend(directoryURL: URL, draftDirectoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, draftDirectoryURL: draftDirectoryURL,
                    options: NFKMLXGenerationOptions())
    }

    /// The release directory a download of `repo` fills: `config.json`, `tokenizer.json`,
    /// `tokenizer_config.json`, whichever of `vocab.json`, `merges.txt`, `added_tokens.json`, and
    /// `chat_template.jinja` the repo serves, and the weights, single-file or sharded.
    static func releaseDirectory(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> URL {
        try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles)
    }

    /// Downloads a release and builds the backend.
    ///
    /// @discussion The download fetches `config.json`, the tokenizer files, and the weights (a single
    /// `model.safetensors` or every shard a `model.safetensors.index.json` names) into the hub cache
    /// under `cacheDirectoryURL`, or the default cache when nil. A cached file is not fetched again.
    /// The call blocks on the network; call it off the render thread. It serves the Qwen3, Qwen2,
    /// Qwen2-MoE, Qwen3-MoE, and gpt-oss releases the directory factory reads, such as
    /// `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B`, and `openai/gpt-oss-20b`, none of which is gated. A gated
    /// repo needs `NFKHFHub.defaultAccessToken` set before the first download. Introduced in
    /// InferKit 0.4.0.
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

    /// ``backend(repo:revision:cacheDirectoryURL:)`` holding the release's routed experts as
    /// `residency` says; see ``backend(directoryURL:residency:options:)``. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try backend(directoryURL: try releaseDirectory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL),
                        residency: residency)
        }
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:residency:)``. Introduced in
    /// InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: residency)
        }
    }

    /// Downloads a release and a smaller release of the same family as the draft, and builds the
    /// speculative-decoding backend.
    ///
    /// @discussion Each repo downloads as ``backend(repo:revision:cacheDirectoryURL:)`` describes,
    /// into the same cache; a cached file is not fetched again. The call blocks on the network; call
    /// it off the render thread. `Qwen/Qwen3-0.6B` drafting for `Qwen/Qwen3-4B` is the intended
    /// pairing. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:draftRepo:draftRevision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, draftRepo: String, draftRevision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let directory = try releaseDirectory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
            let draftDirectory = try releaseDirectory(repo: draftRepo, revision: draftRevision,
                                                      cacheDirectoryURL: cacheDirectoryURL)
            return try backend(directoryURL: directory, draftDirectoryURL: draftDirectory)
        }
    }

    /// The asynchronous form of ``backend(repo:revision:draftRepo:draftRevision:cacheDirectoryURL:)``.
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:draftRepo:draftRevision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, draftRepo: String, draftRevision: String?,
                               cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, draftRepo: draftRepo, draftRevision: draftRevision,
                        cacheDirectoryURL: cacheDirectoryURL)
        }
    }
}

extension NFKMLXLanguageBackend: NFKMLXReloading {

    var canReload: Bool { lifecycle.canReload }

    func setLoadOptions(_ options: NFKMLXLoadOptions) {
        lifecycle.setLoadOptions(options)
    }

    func enableReload(_ build: @escaping () throws -> any NFKInferenceBackend, measurement: NFKMLXLoadMeasurement?) {
        lifecycle.enableReload(building: build, as: NFKMLXLanguageBackend.self, measurement: measurement) {
            try $0.lifecycle.loadedState()
        }
    }
}
