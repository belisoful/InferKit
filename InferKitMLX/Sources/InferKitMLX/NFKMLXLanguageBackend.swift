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
    /// The bit width to store the key-value cache at (`NSNumber`, 4 or 8). Setting it enables cache
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
    /// prompt adds to the last one, as an `NSNumber` boolean. See
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

/// How a generation run samples and how its prompt is prepared.
public struct NFKMLXGenerationOptions: Sendable {
    /// The most tokens to produce, not counting the prompt.
    public var maxTokens: Int = 256
    /// 0 samples greedily; above 0 divides the logits before sampling.
    public var temperature: Float = 0
    /// Nucleus sampling: the smallest set of tokens whose probability sums past this is sampled from.
    /// Ignored at temperature 0.
    public var topP: Float = 1
    /// A seed makes a sampled run repeatable.
    public var seed: UInt64?
    /// Generation stops when one of these is produced.
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

    /// Runs the prompt through the cache in slices of at most this many tokens, or `nil` for one pass.
    ///
    /// @discussion A long prompt run as a single forward builds an attention over its whole length at
    /// once, whose peak is what a large context runs out of. Feeding the prompt in chunks makes each
    /// pass attend to its slice plus the cached prefix, so the peak is bounded by the chunk rather than
    /// the prompt. It is EXACT, not an approximation: the cache makes a chunk attend to exactly the
    /// keys a single forward would, so the logits match. Off by default, since a short prompt needs no
    /// chunking.
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
    /// previous one prefills only its tail. Off by default; see ``NFKMLXPromptCache``.
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
    /// @discussion The prompt runs as one forward pass that fills the cache, then each step feeds back
    /// a single token. Returning false from `onToken` stops the run, which is how a job's cancellation
    /// reaches generation.
    func generate(prompt: [Int], options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                  promptCache: NFKMLXPromptCache? = nil,
                  onToken: ((Int) -> Bool)? = nil) -> [Int] {
        guard !prompt.isEmpty else { return [] }
        if let seed = options.seed { MLXRandom.seed(seed) }

        let (cache, tail) = startingCache(for: prompt, options: options, promptCache: promptCache)
        var logits = prefill(tail, cache: cache, chunkSize: options.prefillChunkSize)
        promptCache?.record(tail)
        var produced = [Int]()
        let cursor = options.constraint?.makeCursor()

        for _ in 0 ..< options.maxTokens {
            var row = logits[0, -1]
            if let cursor {
                row = row + cursor.allowedTokenMask().asType(row.dtype)
            }
            let next = NFKMLXLanguageNet.sample(row, options: options)
            if options.stopTokens.contains(next) || (cursor != nil && next == cursor?.endToken) { break }
            produced.append(next)
            cursor?.accept(next)
            if let onToken, !onToken(next) { break }
            logits = self(MLXArray([Int32(next)]).reshaped([1, 1]), cache: cache)
            promptCache?.record([next])
        }
        return produced
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

    /// Fills the cache with the prompt and returns the logits after its last token: one forward pass,
    /// or several bounded ones when a chunk size is set. Chunking is exact — each chunk attends
    /// through the cache to the same prefix a single pass would — so only the peak differs.
    func prefill(_ prompt: [Int], cache: NFKMLXKeyValueCache, chunkSize: Int?) -> MLXArray {
        guard let chunkSize, chunkSize > 0, prompt.count > chunkSize else {
            return self(MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count]), cache: cache)
        }
        var logits = MLXArray(0)
        var start = 0
        while start < prompt.count {
            let end = Swift.min(start + chunkSize, prompt.count)
            let slice = prompt[start ..< end].map { Int32($0) }
            logits = self(MLXArray(slice).reshaped([1, slice.count]), cache: cache)
            eval(logits)                                        // free a chunk's activations before the next
            start = end
        }
        return logits
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
        let distribution = softmax(logits / temperature, axis: -1)
        return options.topP < 1 ? nucleus(distribution, topP: options.topP) : distribution
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
/// `NFKParameterTopP`, `NFKParameterMaxTokens`, and `NFKParameterSeed` override the defaults.
/// Generation is many forward passes; run it off the render thread, or submit a job, which reports
/// each token through `partialResult`.
@objc(NFKMLXLanguageBackend)
public final class NFKMLXLanguageBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKLMHolder
    private let identifier: String
    private let defaults: NFKMLXGenerationOptions
    /// The cache kept between requests when a request asks for reuse. Generation is serialized
    /// through `generationLock`: two runs through one cache would interleave their rows.
    private var promptCache: NFKMLXPromptCache?
    private let generationLock = NSLock()

    init(net: NFKMLXLanguageNet, tokenizer: NFKTokenizer?, identifier: String,
         options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
         draft: NFKMLXLanguageNet? = nil) {
        self.holder = NFKLMHolder(net, tokenizer, draft: draft)
        self.identifier = identifier
        self.defaults = options
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    /// Whether the backend holds a draft model, so requests can decode speculatively.
    @objc public var hasDraftModel: Bool { holder.draft != nil }

    /// How many positions the retained prompt cache holds, or 0 when none is kept.
    @objc public var promptCacheLength: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return promptCache?.count ?? 0
    }

    /// Drops the retained prompt cache, so the next request prefills from the start.
    @objc public func resetPromptCache() {
        generationLock.lock(); defer { generationLock.unlock() }
        promptCache = nil
    }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let tokenizer = holder.tokenizer else {
            throw NFKMLXError.unsupportedConfiguration(
                "text generation needs a tokenizer; build the backend with one")
        }
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
        Self.applyMLXParameters(from: request, to: &options)
        if let schema = request.parameter(forKey: NFKParameterJSONSchema) {
            guard let dictionary = schema as? [String: Any] else {
                throw NFKMLXError.unsupportedConfiguration("NFKParameterJSONSchema is a JSON Schema object")
            }
            options.jsonSchema = try NFKMLXJSONSchema(json: dictionary)
        }

        guard let text = Self.prompt(from: request, template: options.chatTemplate) else {
            throw NFKMLXError.unsupportedInput
        }
        let tokens = tokenizer.encode(text).map(\.intValue)
        let produced = generate(tokens, options: options)
        let reply = tokenizer.decode(produced.map { NSNumber(value: $0) })
        var outputs: [String: Any] = [NFKOutputText: reply]
        // JSON that was asked for comes back parsed too, as the remote backends return it; JSON-looking
        // text that was not asked for is not guessed at.
        if options.jsonSchema != nil || options.jsonOutput,
           let parsed = try? JSONSerialization.jsonObject(with: Data(reply.utf8), options: [.fragmentsAllowed]) {
            outputs[NFKOutputStructured] = parsed
        }
        return NFKInferenceResult(outputs: outputs)
    }

    /// The vocabulary's bytes, read from the tokenizer once and kept for every constrained request.
    private var vocabulary: NFKMLXVocabulary?

    /// The constraint a request asks for, if any: a custom one wins, then a schema, then JSON, then
    /// the choices.
    private func constraint(for options: NFKMLXGenerationOptions,
                            tokenizer: NFKTokenizer) -> (any NFKMLXTokenConstraint)? {
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

    /// Runs the tokens through the plain or the speculative loop, against the retained prompt cache
    /// when the options ask for one.
    private func generate(_ tokens: [Int], options requested: NFKMLXGenerationOptions) -> [Int] {
        generationLock.lock(); defer { generationLock.unlock() }
        let net = holder.net
        var options = requested
        if let tokenizer = holder.tokenizer {
            options.constraint = constraint(for: requested, tokenizer: tokenizer)
            // A release names its end-of-sequence token; stopping there is what every caller expects
            // of a language model, so it is the default stop when the request names none.
            if options.stopTokens.isEmpty, tokenizer.eosTokenId >= 0 {
                options.stopTokens = [tokenizer.eosTokenId]
            }
        }
        var cache: NFKMLXPromptCache?
        if options.reusesPromptCache {
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
        if let draft = holder.draft, options.draftTokens > 0, options.constraint == nil {
            return net.generate(prompt: tokens, options: options, draft: draft, promptCache: cache)
        }
        return net.generate(prompt: tokens, options: options, promptCache: cache)
    }

    /// Overrides `options` with the MLX generation parameters a request carries, so an Objective-C
    /// caller reaches every option the Swift `NFKMLXGenerationOptions` struct exposes.
    static func applyMLXParameters(from request: NFKInferenceRequest, to options: inout NFKMLXGenerationOptions) {
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
    }

    /// The prompt text, from either input key. A raw `NFKInputPrompt` is used verbatim; a message
    /// list is flattened plainly, or rendered through the chat template when one is asked for.
    static func prompt(from request: NFKInferenceRequest,
                       template: NFKMLXChatTemplate = .none) -> String? {
        if let prompt = request.prompt { return prompt }
        guard let messages = request.messages else { return nil }
        switch template {
        case .none:
            return messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
        case .chatML:
            return chatMLPrompt(from: messages)
        case .jinja(let templateSource, let bosToken, let eosToken):
            // A render failure must not abort generation; fall back to the ChatML approximation.
            return (try? NFKMLXChatTemplateRenderer.render(templateSource, messages: messages,
                                                           addGenerationPrompt: true,
                                                           bosToken: bosToken, eosToken: eosToken))
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

    static func makeNet(_ configuration: NFKMLXLanguageConfiguration = .qwen3_0_6B) -> NFKMLXLanguageNet {
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
        return try configuration(fromJSON: json)
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
        let heads = integer("num_attention_heads", 16)
        var configuration = NFKMLXLanguageConfiguration(
            hiddenSize: hidden,
            layerCount: integer("num_hidden_layers", 28),
            headCount: heads,
            keyValueHeadCount: integer("num_key_value_heads", heads),
            headDimensions: integer("head_dim", hidden / max(heads, 1)),
            intermediateSize: integer("intermediate_size", 3072),
            vocabularySize: integer("vocab_size", 151_936),
            // Transformers 5.x nests the rotary base under `rope_parameters`.
            ropeTheta: ((json["rope_parameters"] as? [String: Any])?["rope_theta"] as? NSNumber)?
                .floatValue ?? real("rope_theta", 1_000_000),
            rmsEpsilon: real("rms_norm_eps", 1e-6),
            tiesWordEmbeddings: (json["tie_word_embeddings"] as? NSNumber)?.boolValue ?? false,
            // Qwen2 and Qwen2-MoE carry query/key/value biases and spell the flag `qkv_bias`, absent
            // from their released configs because true is its default; Qwen3 and Llama carry none.
            attentionBias: ((json["attention_bias"] ?? json["qkv_bias"]) as? NSNumber)?.boolValue
                ?? (modelType == "qwen2" || modelType == "qwen2_moe"))
        // Qwen3 normalizes queries and keys per head; Qwen2 and Llama do not. The model type is what
        // says so — the config carries no flag for it.
        configuration.normalizesQueryAndKey = modelType.hasPrefix("qwen3")
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
        case "gpt_oss":
            // The router takes the softmax over the selected logits, which is the renormalized form.
            return (integer("num_local_experts", 0), integer("num_experts_per_tok", 0),
                    integer("intermediate_size", 0), true, 0)
        default:
            if json["num_experts"] != nil || json["num_local_experts"] != nil || json["n_routed_experts"] != nil {
                throw NFKMLXError.unsupportedConfiguration(
                    "the config describes a mixture-of-experts family (\(modelType)) this network does "
                    + "not implement; qwen3_moe, qwen2_moe, mixtral, and gpt_oss are the families it reads")
            }
            return nil
        }
    }

    /// The module key a release's tensor name maps to: Mixtral's `block_sparse_moe` spelling becomes
    /// the `mlp` layout Qwen3-MoE and this module share, gpt-oss's `router` is the module's `gate` and
    /// its fused expert tensors gain the `.weight` the module's layers carry, and every other name
    /// passes through.
    static func moduleKey(forRelease key: String) -> String {
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
        return key.replacingOccurrences(of: ".block_sparse_moe.gate.", with: ".mlp.gate.")
            .replacingOccurrences(of: ".block_sparse_moe.experts.", with: ".mlp.experts.")
            .replacingOccurrences(of: ".w1.weight", with: ".gate_proj.weight")
            .replacingOccurrences(of: ".w3.weight", with: ".up_proj.weight")
            .replacingOccurrences(of: ".w2.weight", with: ".down_proj.weight")
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
        pattern: #"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"#)

    /// Stacks a release's per-expert tensors (`…experts.N.gate_proj.weight`) into the module's one
    /// `[experts, out, in]` tensor per projection, leaving every other pair as it is.
    static func stackingExperts(_ mapped: [(String, MLXArray)]) -> [(String, MLXArray)] {
        var groups = [String: [(index: Int, value: MLXArray)]]()
        var rest = [(String, MLXArray)]()
        for (name, value) in mapped {
            let range = NSRange(name.startIndex..., in: name)
            guard let match = expertPattern.firstMatch(in: name, range: range),
                  let prefix = Range(match.range(at: 1), in: name),
                  let index = Range(match.range(at: 2), in: name).flatMap({ Int(name[$0]) }),
                  let projection = Range(match.range(at: 3), in: name) else {
                rest.append((name, value))
                continue
            }
            groups["\(name[prefix]).\(name[projection]).weight", default: []].append((index, value))
        }
        for (key, entries) in groups {
            rest.append((key, stacked(entries.sorted { $0.index < $1.index }.map(\.value))))
        }
        return rest
    }

    /// Loads a released checkpoint. The module's keys are the checkpoint's, so nothing is remapped
    /// beyond stacking a mixture's per-expert tensors.
    ///
    /// Every weight here is at most two-dimensional, so none of the convolution transposes the vision
    /// models apply have any part in this path.
    static func loadWeights(into net: NFKMLXLanguageNet, from url: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        // A quantized checkpoint reshapes the module to match and loads at its stored dtypes: the
        // packed weights are uint32 whatever the request, and the scales keep the precision the
        // quantization was computed at.
        NFKMLXQuantization.matchStructure(of: checkpoint, on: net)
        let keepStored = precision == .checkpoint || checkpoint.quantization != nil
        let tied = net.lmHead == nil
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
    /// and nothing else.
    static func loadWeights(into net: NFKMLXLanguageNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        // Refuse a release whose weights alone exceed the memory budget, before materializing any,
        // so "the process died" becomes an error naming the shortfall.
        try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: precision)
        // A single-file release routes through the checkpoint reader, which is where a quantized
        // save's metadata lives; only a sharded release takes the merge path (a quantized module
        // saves as one file, so a quantized sharded release does not arise).
        let files = try NFKMLXReleaseWeights.files(inDirectory: directory)
        if files.count == 1 {
            try loadWeights(into: net, from: files[0], precision: precision)
            return
        }
        // A tied release still ships `lm_head.weight`, byte-identical to the embedding, and a tied
        // module has no projection to put it in.
        let tied = net.lmHead == nil
        let merged = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision) {
            tied && $0.hasPrefix("lm_head.") ? nil : $0
        }
        let prepared = releaseWeights(merged)
        try installPackedExperts(into: net, from: prepared)
        try NFKMLXWeights.apply(prepared, to: net, verifyShapes: true)
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
    /// tokenizer files.
    public static func backend(directoryURL: URL,
                               options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        let (net, tokenizer) = try loadedRelease(at: directoryURL)
        return NFKMLXLanguageBackend(net: net, tokenizer: tokenizer, identifier: modelName,
                                     options: options)
    }

    /// Builds from a release directory together with a smaller release of the same family as the
    /// draft for speculative decoding.
    ///
    /// @discussion The draft proposes ``NFKMLXGenerationOptions/draftTokens`` tokens per round and
    /// the main model verifies them in one pass, producing the sequence it would have produced
    /// alone. The two must share a vocabulary, which is checked here; a Qwen3 0.6B drafting for a
    /// Qwen3 4B is the intended pairing. The draft loads at the same precision as the main model.
    public static func backend(directoryURL: URL, draftDirectoryURL: URL,
                               options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        let (net, tokenizer) = try loadedRelease(at: directoryURL)
        let (draft, _) = try loadedRelease(at: draftDirectoryURL)
        guard draft.configuration.vocabularySize == net.configuration.vocabularySize else {
            throw NFKMLXError.unsupportedConfiguration(
                "the draft's vocabulary has \(draft.configuration.vocabularySize) entries where the "
                + "model's has \(net.configuration.vocabularySize); a draft must share the model's "
                + "tokenizer")
        }
        return NFKMLXLanguageBackend(net: net, tokenizer: tokenizer, identifier: modelName,
                                     options: options, draft: draft)
    }

    /// The network and tokenizer a release directory describes.
    static func loadedRelease(at directoryURL: URL) throws -> (NFKMLXLanguageNet, NFKTokenizer?) {
        let configuration = try self.configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        // Qwen ships a byte-level BPE vocabulary trained on its OWN pre-tokenization splits — a
        // letter run absorbing one leading punctuation character, digits split singly — so the
        // pattern is named: the GPT-2 default encodes the same text to different, valid-looking ids.
        // The special tokens live in tokenizer_config.json rather than vocab.json; without them a
        // chat template's markers would encode as ordinary text.
        let tokenizer = releaseTokenizer(inDirectory: directoryURL)
        let net = makeNet(configuration)
        let packed = storesExpertsMXFP4(configURL: directoryURL.appendingPathComponent("config.json"))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: packed ? .checkpoint : .float32)
        return (net, tokenizer)
    }

    /// The tokenizer a release directory describes, built without loading the weights.
    ///
    /// @discussion The vocabulary is Qwen's byte-level BPE over its own `qwen2` pre-tokenization
    /// splits; the GPT-2 default would encode the same text to different, valid-looking ids. The
    /// special tokens live in `tokenizer_config.json` rather than `vocab.json`, so a chat template's
    /// or an embedder's markers resolve rather than encoding as ordinary text.
    static func releaseTokenizer(inDirectory directory: URL) -> NFKTokenizer? {
        let (specials, endToken) = specialTokens(inDirectory: directory)
        var manifest: [String: Any] = ["tokenizer": ["type": "bpe-bytelevel",
                                                     "pretokenizer": pretokenizationName(inDirectory: directory),
                                                     "specialTokens": specials]]
        if let endToken { manifest["eosTokenId"] = endToken }
        // A release that ships only tokenizer.json (gpt-oss) has its vocabulary and merges extracted
        // into the vocab.json / merges.txt pair the core reader takes.
        var files = directory
        if !FileManager.default.fileExists(atPath: directory.appendingPathComponent("vocab.json").path) {
            guard let extracted = byteLevelFiles(fromTokenizerJSON: directory.appendingPathComponent("tokenizer.json")) else {
                return nil
            }
            files = extracted
        }
        return try? NFKTokenizer(forManifest: manifest, directory: files)
    }

    /// The pre-tokenization a release's tokenizer.json declares, read from its `Split` regex: the
    /// o200k pattern (gpt-oss) matches words by their case pattern, spelled with the `\p{Lu}\p{Lt}…`
    /// classes no other family uses; everything else here is Qwen's, the previous default.
    static func pretokenizationName(inDirectory directory: URL) -> String {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let preTokenizer = json["pre_tokenizer"] as? [String: Any] else { return "qwen2" }
        // A `Sequence` of pre-tokenizers or a single `Split`; the regex is the one that names the family.
        let steps = (preTokenizer["pretokenizers"] as? [[String: Any]]) ?? [preTokenizer]
        for step in steps where step["type"] as? String == "Split" {
            if let pattern = step["pattern"] as? [String: Any], let regex = pattern["Regex"] as? String,
               regex.contains("\\p{Lu}\\p{Lt}") {
                return "o200k"
            }
        }
        return "qwen2"
    }

    /// Writes a tokenizer.json's byte-level BPE vocabulary and merges as the `vocab.json` and
    /// `merges.txt` pair the core reader takes, into a scratch directory, and returns it.
    static func byteLevelFiles(fromTokenizerJSON url: URL) -> URL? {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any],
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
    /// `tokenizer_config.json`, or `added_tokens.json`) and the id of its `eos_token`.
    static func specialTokens(inDirectory directory: URL) -> ([String: Int], Int?) {
        var literals = [String: Int]()
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

    /// The Objective-C entry: builds from a release directory, reading its `config.json`, tokenizer,
    /// and shards. Generation options are set per request through ``NFKMLXGenerationParameterKey``,
    /// which is how an Objective-C caller reaches what the Swift `options:` parameter carries. Run
    /// inference off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, options: NFKMLXGenerationOptions())
    }

    /// The Objective-C entry for speculative decoding: a release directory and a smaller release
    /// of the same family as the draft. `NFKMLXGenerationParameterKey.draftTokens` sets the
    /// proposals per round on a request.
    @objc(backendWithDirectoryURL:draftDirectoryURL:error:)
    public static func backend(directoryURL: URL, draftDirectoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, draftDirectoryURL: draftDirectoryURL,
                    options: NFKMLXGenerationOptions())
    }
}
