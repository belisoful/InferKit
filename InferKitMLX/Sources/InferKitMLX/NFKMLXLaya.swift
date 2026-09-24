//
//  NFKMLXLaya.swift
//  InferKitMLX
//
//  Laya (convaiinnovations/laya, Apache-2.0): the open reproduction of TypeSafe's Jev, a typed-decision
//  model. It answers the same three question types about a state in one bidirectional pass, without
//  generating text: a choice among named options, a score on an ordered scale, and a noul, the
//  probability that a statement holds. Every option is scored at its own mask token, and a softmax
//  over the question's markers is the answer distribution.
//
//  The network is a ModernBERT encoder (the reranker's, under `encoder.`) plus a decision head trained
//  from scratch: a question-type embedding added to every token, two pre-norm transformer layers with
//  a ReLU feed-forward, a scorer (LayerNorm, Linear, GELU, Linear) read at each marker, and an
//  act-or-escalate head over the first token and a summary of the answer distribution. The released
//  geometries are ModernBERT-large (28 layers, 1024 wide; the root and typed-decisions variants) and
//  mmBERT-base (22 layers, 768 wide, a 256k Gemma vocabulary; the multilingual variant).
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - Configuration

/// The geometry and the calibration of a Laya release.
public struct NFKMLXLayaConfiguration: Sendable {
    /// The encoder's geometry, including the classification tokens.
    public var encoder: NFKMLXModernBertConfiguration
    /// How many transformer layers the decision head stacks.
    public var headLayers: Int
    /// The longest sequence a question builds; the state is cut to fit.
    public var maxLength: Int
    /// The budget for the instructions and the options together, before the state.
    public var headMaxLength: Int
    /// The most tokens one option keeps.
    public var optionTokenLimit: Int
    /// The mask token each option is scored at.
    public var maskToken: Int
    /// The padding token, which a single sequence never carries.
    public var padToken: Int
    /// The fitted temperature per question type (choice, score, noul), applied to the logits.
    public var temperatures: [Float]
    /// A temperature per type-and-cardinality bucket (`choice:3-5`), which wins over `temperatures`.
    public var temperatureByOptions: [String: Float]
    /// What the tokenizer spells its mask token as, which the prompt builder scrubs from the text.
    public var maskLiteral: String
    /// The release's own name for the model.
    public var modelName: String
    /// How many prefixes of a conversation an episode trains on; a longer conversation is sampled
    /// evenly down to this many.
    public var maxPrefixes: Int

    public init(encoder: NFKMLXModernBertConfiguration, headLayers: Int = 2, maxLength: Int = 512,
                headMaxLength: Int = 192, optionTokenLimit: Int = 48, maskToken: Int = 50_284,
                padToken: Int = 50_283, temperatures: [Float] = [1, 1, 1],
                temperatureByOptions: [String: Float] = [:], maskLiteral: String = "[MASK]",
                modelName: String = "laya", maxPrefixes: Int = 6) {
        self.encoder = encoder
        self.headLayers = headLayers
        self.maxLength = maxLength
        self.headMaxLength = headMaxLength
        self.optionTokenLimit = optionTokenLimit
        self.maskToken = maskToken
        self.padToken = padToken
        self.temperatures = temperatures
        self.temperatureByOptions = temperatureByOptions
        self.maskLiteral = maskLiteral
        self.modelName = modelName
        self.maxPrefixes = maxPrefixes
    }

    /// The root release: ModernBERT-large with the calibration the release fitted.
    public static let large = NFKMLXLayaConfiguration(
        encoder: NFKMLXModernBertConfiguration(hiddenSize: 1024, layerCount: 28, headCount: 16,
                                               intermediateSize: 2624, vocabularySize: 50_368),
        temperatures: [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
        temperatureByOptions: ["choice:2": 1.9063563346862793, "choice:3-5": 1.7601518630981445,
                               "choice:6-10": 1.0000158548355103, "choice:11+": 0.10058280825614929,
                               "score:3-5": 1.2514300346374512, "noul:2": 1.983399510383606],
        modelName: "rl-agent")

    /// The multilingual release: mmBERT-base, whose local layers share the global rotary base and whose
    /// tokenizer is Gemma's (the classifier token is `<bos>`, the separator `<eos>`), with a longer
    /// sequence budget and no fitted temperature.
    public static let multilingual = NFKMLXLayaConfiguration(
        encoder: NFKMLXModernBertConfiguration(hiddenSize: 768, layerCount: 22, headCount: 12,
                                               intermediateSize: 1152, vocabularySize: 256_000,
                                               globalRopeTheta: 160_000, localRopeTheta: 160_000,
                                               clsToken: 2, sepToken: 1),
        maxLength: 1024, headMaxLength: 256, maskToken: 4, padToken: 0, maskLiteral: "<mask>",
        modelName: "rl-agent")

    /// A small configuration for tests, with a window small enough that the sliding path runs.
    public static let tiny = NFKMLXLayaConfiguration(
        encoder: NFKMLXModernBertConfiguration(hiddenSize: 64, layerCount: 4, headCount: 2, intermediateSize: 128,
                                               vocabularySize: 512, globalAttentionEvery: 3, globalRopeTheta: 160_000,
                                               localRopeTheta: 10_000, localAttention: 4, clsToken: 1, sepToken: 2),
        headLayers: 2, maxLength: 64, headMaxLength: 32, optionTokenLimit: 6,
        maskToken: 4, padToken: 3, modelName: "laya-tiny")

    var headDimensions: Int { max(1, encoder.hiddenSize / 64) }

    /// The bucket a question's temperature is looked up under: its type and its option count.
    static func temperatureBucket(type: NFKDecisionType, optionCount: Int) -> String {
        let size = optionCount <= 2 ? "2" : optionCount <= 5 ? "3-5" : optionCount <= 10 ? "6-10" : "11+"
        return "\(NFKDecisionQuestion.name(for: type)):\(size)"
    }

    /// The temperature a question's logits are divided by.
    func temperature(type: NFKDecisionType, optionCount: Int) -> Float {
        if let bucketed = temperatureByOptions[Self.temperatureBucket(type: type, optionCount: optionCount)] {
            return bucketed
        }
        let index = Int(type.rawValue)
        return index < temperatures.count ? temperatures[index] : 1
    }

    /// Reads a release directory: `encoder/config.json` for the geometry, `rl_agent_config.json` for
    /// the sequence budgets and the calibration, and the tokenizer files for the special tokens. The
    /// rotary bases are read from the transformers 5 `rope_parameters` block when the config carries
    /// one, else from the older top-level fields, else ModernBERT's defaults. The classifier,
    /// separator, mask, and padding ids are the tokenizer's, as the reference reads them: the
    /// multilingual release's encoder config names a `cls_token_id` its tokenizer does not use.
    public init(directoryURL: URL) throws {
        let encoderJSON = try Self.json(at: directoryURL.appendingPathComponent("encoder/config.json"))
        let agentJSON = try Self.json(at: directoryURL.appendingPathComponent("rl_agent_config.json"))
        let tokenizerJSON = (try? Self.json(at: directoryURL.appendingPathComponent("tokenizer/tokenizer_config.json"))) ?? [:]
        let tokenizerModel = (try? Self.json(at: directoryURL.appendingPathComponent("tokenizer/tokenizer.json"))) ?? [:]
        func tokenId(_ literal: Any?) -> Int? {
            guard let literal = literal as? String else { return nil }
            for entry in tokenizerModel["added_tokens"] as? [[String: Any]] ?? [] where entry["content"] as? String == literal {
                return (entry["id"] as? NSNumber)?.intValue
            }
            return ((tokenizerModel["model"] as? [String: Any])?["vocab"] as? [String: Any])?[literal] as? Int
        }

        func number(_ key: String, in json: [String: Any], default value: Double) -> Double {
            (json[key] as? NSNumber)?.doubleValue ?? value
        }
        let rope = encoderJSON["rope_parameters"] as? [String: Any]
        func theta(_ block: String, fallback: String, default value: Double) -> Float {
            if let entry = rope?[block] as? [String: Any], let theta = entry["rope_theta"] as? NSNumber {
                return theta.floatValue
            }
            return Float(number(fallback, in: encoderJSON, default: value))
        }
        encoder = NFKMLXModernBertConfiguration(
            hiddenSize: Int(number("hidden_size", in: encoderJSON, default: 768)),
            layerCount: Int(number("num_hidden_layers", in: encoderJSON, default: 22)),
            headCount: Int(number("num_attention_heads", in: encoderJSON, default: 12)),
            intermediateSize: Int(number("intermediate_size", in: encoderJSON, default: 1152)),
            vocabularySize: Int(number("vocab_size", in: encoderJSON, default: 50_368)),
            globalAttentionEvery: Int(number("global_attn_every_n_layers", in: encoderJSON, default: 3)),
            globalRopeTheta: theta("full_attention", fallback: "global_rope_theta", default: 160_000),
            localRopeTheta: theta("sliding_attention", fallback: "local_rope_theta", default: 10_000),
            localAttention: Int(number("local_attention", in: encoderJSON, default: 128)),
            normEpsilon: Float(number("norm_eps", in: encoderJSON, default: 1e-5)),
            clsToken: tokenId(tokenizerJSON["cls_token"]) ?? Int(number("cls_token_id", in: encoderJSON, default: 50_281)),
            sepToken: tokenId(tokenizerJSON["sep_token"]) ?? Int(number("sep_token_id", in: encoderJSON, default: 50_282)))
        maskToken = tokenId(tokenizerJSON["mask_token"]) ?? Int(number("mask_token_id", in: encoderJSON, default: 50_284))
        padToken = tokenId(tokenizerJSON["pad_token"]) ?? Int(number("pad_token_id", in: encoderJSON, default: 50_283))
        headLayers = Int(number("head_layers", in: agentJSON, default: 2))
        maxLength = Int(number("max_len", in: agentJSON, default: 512))
        headMaxLength = Int(number("head_max_len", in: agentJSON, default: 192))
        optionTokenLimit = 48
        temperatures = (agentJSON["temperature"] as? [NSNumber])?.map(\.floatValue) ?? [1, 1, 1]
        temperatureByOptions = ((agentJSON["temperature_by_options"] as? [String: NSNumber]) ?? [:]).mapValues(\.floatValue)
        maskLiteral = tokenizerJSON["mask_token"] as? String ?? "[MASK]"
        modelName = agentJSON["model_name"] as? String ?? "laya"
        maxPrefixes = Int(number("max_prefixes", in: agentJSON, default: 6))
    }

    private static func json(at url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        return json
    }
}

// MARK: - The decision head

/// A parameter-free GELU occupying a Sequential index, so the checkpoint's numeric keys line up.
final class NFKLayaGELU: Module {}

/// PyTorch's `nn.MultiheadAttention` as the head uses it: one fused input projection with a bias, and
/// an output projection.
final class NFKLayaHeadAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var inputProjectionWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inputProjectionBias: MLXArray
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    let heads: Int
    let headDimensions: Int

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        headDimensions = dimensions / heads
        let bound = 1 / sqrt(Float(dimensions))
        _inputProjectionWeight.wrappedValue = MLXRandom.uniform(low: -bound, high: bound, [3 * dimensions, dimensions])
        _inputProjectionBias.wrappedValue = MLXArray.zeros([3 * dimensions])
        _outputProjection.wrappedValue = Linear(dimensions, dimensions, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        let projected = matmul(x, inputProjectionWeight.transposed()) + inputProjectionBias
        let parts = projected.split(parts: 3, axis: -1)
        func perHead(_ part: MLXArray) -> MLXArray {
            part.reshaped([batch, length, heads, headDimensions]).transposed(0, 2, 1, 3)
        }
        let attention = MLXFast.scaledDotProductAttention(
            queries: perHead(parts[0]), keys: perHead(parts[1]), values: perHead(parts[2]),
            scale: 1 / sqrt(Float(headDimensions)), mask: nil)
        return outputProjection(attention.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDimensions]))
    }
}

/// One pre-norm `nn.TransformerEncoderLayer` of the head: attention and a ReLU feed-forward, each
/// normalized first (with biases, PyTorch's default LayerNorm) and added back.
final class NFKLayaHeadLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKLayaHeadAttention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm

    init(dimensions: Int, heads: Int) {
        _attention.wrappedValue = NFKLayaHeadAttention(dimensions: dimensions, heads: heads)
        _linear1.wrappedValue = Linear(dimensions, 4 * dimensions, bias: true)
        _linear2.wrappedValue = Linear(4 * dimensions, dimensions, bias: true)
        _norm1.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-5)
        _norm2.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-5)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attended = x + attention(norm1(x))
        return attended + linear2(relu(linear1(norm2(attended))))
    }
}

/// The head's layer stack, under the checkpoint's `head.layers.` prefix.
final class NFKLayaHead: Module {
    @ModuleInfo(key: "layers") var layers: [NFKLayaHeadLayer]

    init(dimensions: Int, heads: Int, count: Int) {
        _layers.wrappedValue = (0 ..< count).map { _ in NFKLayaHeadLayer(dimensions: dimensions, heads: heads) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers.reduce(x) { $1($0) }
    }
}

/// Runs a Sequential of Linear, LayerNorm, and GELU modules in order.
private func runLayaSequence(_ layers: [Module], _ input: MLXArray) -> MLXArray {
    var x = input
    for layer in layers {
        switch layer {
        case let linear as Linear: x = linear(x)
        case let norm as LayerNorm: x = norm(x)
        case is NFKLayaGELU: x = gelu(x)
        default: break
        }
    }
    return x
}

/// The whole Laya network: the ModernBERT encoder, the type embedding, the decision head, the marker
/// scorer, and the act-or-escalate head.
public final class NFKMLXLayaNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKModernBertModel
    @ModuleInfo(key: "head") var head: NFKLayaHead
    @ModuleInfo(key: "type_emb") var typeEmbedding: Embedding
    @ModuleInfo(key: "scorer") var scorer: [Module]
    @ModuleInfo(key: "act_head") var actHead: [Module]
    /// The checkpoint's per-type temperature buffer. The calibration the release fitted lives in its
    /// `rl_agent_config.json`, which the configuration carries, so this is loaded and not read.
    @ParameterInfo(key: "temperature") var temperature: MLXArray

    public let configuration: NFKMLXLayaConfiguration

    public init(_ c: NFKMLXLayaConfiguration) {
        configuration = c
        let d = c.encoder.hiddenSize
        _encoder.wrappedValue = NFKModernBertModel(c.encoder)
        _head.wrappedValue = NFKLayaHead(dimensions: d, heads: c.headDimensions, count: c.headLayers)
        _typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: d)
        _scorer.wrappedValue = [LayerNorm(dimensions: d, eps: 1e-5), Linear(d, d, bias: true), NFKLayaGELU(), Linear(d, 1, bias: true)]
        _actHead.wrappedValue = [Linear(d + 4, 256, bias: true), NFKLayaGELU(), Linear(256, 2, bias: true)]
        _temperature.wrappedValue = MLXArray.ones([3])
        super.init()
    }

    /// The token states after the encoder, the type embedding, and the decision head: `[1, length, d]`.
    func headStates(tokens: [Int], type: NFKDecisionType) -> MLXArray {
        let input = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
        let typed = encoder(input) + typeEmbedding(MLXArray([Int32(type.rawValue)])).reshaped([1, 1, -1])
        return head(typed)
    }

    /// The raw option logits `[markers]` (before the temperature) and the act-or-escalate logits `[2]`.
    public func forward(tokens: [Int], markers: [Int], type: NFKDecisionType) -> (logits: MLXArray, act: MLXArray) {
        let states = headStates(tokens: tokens, type: type)[0]              // [length, d]
        let atMarkers = states.take(MLXArray(markers.map { Int32($0) }), axis: 0)
        let logits = runLayaSequence(scorer, atMarkers).reshaped([-1])          // [markers]

        // The act head reads the first token beside a detached summary of the answer distribution:
        // its top probability, the margin to the runner-up, its normalized entropy, and the count.
        let probabilities = stopGradient(softmax(logits, axis: -1))
        let count = Float(Swift.max(markers.count, 2))
        let entropy = -(probabilities * log(maximum(probabilities, 1e-9))).sum() / log(count)
        let ordered = sorted(probabilities, axis: -1)
        let top = ordered[-1]
        let runnerUp = markers.count > 1 ? ordered[-2] : MLXArray(Float(0))
        let features = stacked([top, top - runnerUp, entropy, MLXArray(count / 255)])
        let act = runLayaSequence(actHead, concatenated([states[0], features], axis: 0))
        return (logits, act)
    }
}

// MARK: - The tokenizer and the prompt

/// The tokenizer a Laya release ships: byte-level BPE for the ModernBERT releases, Gemma's for the
/// multilingual one. Special tokens are placed by the prompt builder, so only plain text is encoded.
public final class NFKMLXLayaTokenizer {
    private let encodeText: (String) -> [Int]

    public init(encode: @escaping (String) -> [Int]) {
        encodeText = encode
    }

    /// Wraps a core tokenizer, which is what the byte-level releases use.
    public convenience init(tokenizer: NFKTokenizer) {
        self.init { tokenizer.encode($0).map(\.intValue) }
    }

    /// Wraps Gemma's reader under the multilingual release's Metaspace pre-tokenizer: a metaspace is
    /// prepended to the text, and the text is split at every metaspace into pieces that each start with
    /// one, so no merge crosses a word boundary.
    convenience init(gemma: NFKMLXGemmaTokenizer) {
        self.init { text in
            let metaspace: Character = "\u{2581}"
            var normalized = text.replacingOccurrences(of: " ", with: String(metaspace))
            if !normalized.hasPrefix(String(metaspace)) {
                normalized = String(metaspace) + normalized
            }
            var pieces = [String]()
            var current = ""
            for character in normalized {
                if character == metaspace, !current.isEmpty {
                    pieces.append(current)
                    current = ""
                }
                current.append(character)
            }
            if !current.isEmpty { pieces.append(current) }
            return pieces.flatMap { gemma.encode($0) }
        }
    }

    public func encode(_ text: String) -> [Int] { encodeText(text) }

    /// The tokenizer under `tokenizer/tokenizer.json`, chosen by the file's pre-tokenizer: ByteLevel is
    /// the GPT-2-family reader the reranker uses, anything else is Gemma's metaspace BPE.
    public static func tokenizer(inDirectory directory: URL) -> NFKMLXLayaTokenizer? {
        let folder = directory.appendingPathComponent("tokenizer")
        let url = folder.appendingPathComponent("tokenizer.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let preTokenizer = (json["pre_tokenizer"] as? [String: Any])?["type"] as? String
        if preTokenizer == "ByteLevel" {
            return NFKMLXModernBERTReranker.byteLevelTokenizer(inDirectory: folder).map { NFKMLXLayaTokenizer(tokenizer: $0) }
        }
        return NFKMLXGemmaTokenizer(tokenizerJSON: url).map { NFKMLXLayaTokenizer(gemma: $0) }
    }
}

/// The token sequence a question builds, and where its option markers sit.
public struct NFKMLXLayaPrompt: Equatable, Sendable {
    public var tokens: [Int]
    /// The position of each option's mask token, in option order.
    public var markers: [Int]
    /// The options in the order they were rendered, which is the order the logits come back in.
    public var options: [String]

    /// The option texts a question renders, in label order. A noul is always `[false, true]`, so the
    /// second probability is the noul.
    public static func renderedOptions(for question: NFKDecisionQuestion) -> [String] {
        switch question.type {
        case .choice:
            return question.options.map { name in
                question.descriptions[name].map { "\(name): \($0)" } ?? name
            }
        case .score:
            return question.options.enumerated().map { "level \($0.offset): \($0.element)" }
        case .noul:
            return ["false: " + (question.descriptions["false"] ?? "no, the statement does not hold"),
                    "true: " + (question.descriptions["true"] ?? "yes, the statement holds")]
        @unknown default:
            return question.options
        }
    }

    /// `[CLS] <type> question: <instructions> [SEP] ([MASK] option)* [SEP] state [SEP]`, cut to the
    /// release's budgets the way the reference cuts it: each option to `optionTokenLimit` tokens, the
    /// options shrunk evenly when they leave under 16 tokens of the head budget, the instructions to
    /// what is left (at least 8), and the state to the room under `maxLength`.
    public init(state: Any, question: NFKDecisionQuestion, tokenizer: NFKMLXLayaTokenizer,
                configuration c: NFKMLXLayaConfiguration, truncateLeft: Bool = false) {
        let mask = c.maskLiteral
        func scrub(_ text: String) -> String { text.replacingOccurrences(of: mask, with: " ") }
        let rendered = Self.renderedOptions(for: question)
        let typeName = NFKDecisionQuestion.name(for: question.type)
        var headTokens = tokenizer.encode("\(typeName) question: \(scrub(question.instructions))")
        var optionTokens: [[Int]] = rendered.map { option -> [Int] in
            let encoded: [Int] = tokenizer.encode(" " + scrub(option))
            return [c.maskToken] + Array(encoded.prefix(c.optionTokenLimit))
        }
        var budget = c.headMaxLength - optionTokens.reduce(0) { $0 + $1.count }
        if budget < 16 {
            let per = Swift.max(4, (c.headMaxLength - 16) / Swift.max(1, optionTokens.count))
            optionTokens = optionTokens.map { Array($0.prefix(per)) }
            budget = c.headMaxLength - optionTokens.reduce(0) { $0 + $1.count }
        }
        headTokens = Array(headTokens.prefix(Swift.max(8, budget)))

        var ids = [c.encoder.clsToken] + headTokens + [c.encoder.sepToken]
        var markers = [Int]()
        for option in optionTokens {
            markers.append(ids.count)
            ids += option
        }
        ids.append(c.encoder.sepToken)
        let room = Swift.max(0, c.maxLength - ids.count - 1)
        var stateTokens = tokenizer.encode(scrub(Self.serialize(state: state)))
        // The reference slices `st[-room:]` on the left, and Python's `[-0:]` is the whole list, so a
        // head that fills the budget keeps the entire state before the final cut.
        if truncateLeft {
            stateTokens = room == 0 ? stateTokens : Array(stateTokens.suffix(room))
        } else {
            stateTokens = Array(stateTokens.prefix(room))
        }
        ids += stateTokens + [c.encoder.sepToken]

        tokens = Array(ids.prefix(c.maxLength))
        self.markers = markers.filter { $0 < c.maxLength }
        options = rendered
    }

    /// The text a state is read as: a string as it is, anything else as JSON the way Python's
    /// `json.dumps` writes it (a space after each comma and colon, non-ASCII kept), with object keys
    /// sorted so the same record always reads the same.
    public static func serialize(state: Any) -> String {
        if let text = state as? String { return text }
        return pythonJSON(state)
    }

    /// The state a conversation prefix reads as: the context's fields in sorted order, then the turns
    /// so far under `conversation`, which the reference appends last.
    public static func serialize(context: [String: Any], turns: [Any]) -> String {
        let fields = context.keys.sorted().map { "\"\(escape($0))\": \(pythonJSON(context[$0]!))" }
        return "{" + (fields + ["\"conversation\": \(pythonJSON(turns))"]).joined(separator: ", ") + "}"
    }

    /// The prefix lengths an episode of `turnCount` turns trains on: every length up to
    /// `maxPrefixes`, and beyond that `maxPrefixes` lengths spaced evenly from 1 to the whole
    /// conversation, rounded to the nearest turn and deduplicated.
    public static func prefixLengths(turnCount: Int, maxPrefixes: Int) -> [Int] {
        guard turnCount > 0 else { return [] }
        if turnCount <= maxPrefixes { return Array(1 ... turnCount) }
        guard maxPrefixes > 1 else { return [1] }
        var lengths = Set<Int>()
        for index in 0 ..< maxPrefixes {
            let position = 1 + Double(turnCount - 1) * Double(index) / Double(maxPrefixes - 1)
            lengths.insert(Int(position.rounded(.toNearestOrEven)))
        }
        return lengths.sorted()
    }

    /// The prompts of an episode's prefixes, oldest first, each cut from the left so the newest turns
    /// survive; a prefix whose two markers do not fit is left out, as the reference leaves it out.
    /// Each prompt is paired with the prefix length it reads.
    public static func prefixes(context: [String: Any], turns: [Any], question: NFKDecisionQuestion,
                                tokenizer: NFKMLXLayaTokenizer, configuration c: NFKMLXLayaConfiguration)
        -> [(length: Int, prompt: NFKMLXLayaPrompt)] {
        prefixLengths(turnCount: turns.count, maxPrefixes: c.maxPrefixes).compactMap { length in
            let state = serialize(context: context, turns: Array(turns.prefix(length)))
            let prompt = NFKMLXLayaPrompt(state: state, question: question, tokenizer: tokenizer,
                                          configuration: c, truncateLeft: true)
            return prompt.markers.count == 2 ? (length, prompt) : nil
        }
    }

    private static func pythonJSON(_ value: Any) -> String {
        switch value {
        case let text as String:
            return "\"" + escape(text) + "\""
        case is NSNull:
            return "null"
        case let number as NSNumber:
            if number === kCFBooleanTrue as NSNumber { return "true" }
            if number === kCFBooleanFalse as NSNumber { return "false" }
            let objCType = String(cString: number.objCType)
            if objCType == "d" || objCType == "f" {
                let double = number.doubleValue
                return double == double.rounded() && abs(double) < 1e16 ? String(format: "%.1f", double) : "\(double)"
            }
            return "\(number.int64Value)"
        case let list as [Any]:
            return "[" + list.map(pythonJSON).joined(separator: ", ") + "]"
        case let object as [String: Any]:
            return "{" + object.keys.sorted().map { "\"\(escape($0))\": \(pythonJSON(object[$0]!))" }.joined(separator: ", ") + "}"
        default:
            return "\"" + escape(String(describing: value)) + "\""
        }
    }

    private static func escape(_ text: String) -> String {
        var out = ""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out
    }
}

// MARK: - The model object

/// Holds the network and tokenizer for capture off the render thread.
private final class NFKLayaHolder: @unchecked Sendable {
    let net: NFKMLXLayaNet
    let tokenizer: NFKMLXLayaTokenizer?
    init(_ net: NFKMLXLayaNet, _ tokenizer: NFKMLXLayaTokenizer?) {
        self.net = net
        self.tokenizer = tokenizer
    }
}

/// Laya, a typed-decision model: the on-device counterpart of `NFKTypeSafeBackend`.
///
/// @discussion A caller hands it a state and `NFKDecisionQuestion`s and gets an `NFKDecisionAnswer`
/// per question, the same objects the hosted Jev backend returns, with the probabilities behind each
/// answer and a confidence. Every question is one bidirectional pass over the state, so a decision costs
/// tens of milliseconds rather than a generation. `backend(directoryURL:)` wraps it as an
/// `NFKInferenceBackend` reading `NFKInputState` and `NFKInputQuestions`. Run it off the render thread.
@objc(NFKMLXLaya)
public final class NFKMLXLaya: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "laya"

    private let holder: NFKLayaHolder
    public let configuration: NFKMLXLayaConfiguration

    public init(net: NFKMLXLayaNet, tokenizer: NFKMLXLayaTokenizer?) {
        holder = NFKLayaHolder(net, tokenizer)
        configuration = net.configuration
        super.init()
    }

    /// The network, for a caller that fine-tunes or inspects it.
    public var net: NFKMLXLayaNet { holder.net }

    /// The tokenizer, or nil when the object was built without one, in which case every answer is a
    /// uniform distribution.
    public var tokenizer: NFKMLXLayaTokenizer? { holder.tokenizer }

    // MARK: Deciding

    /// Answers every question about the state, keyed as the questions were.
    @objc(answersForState:questions:)
    public func decide(state: Any, questions: [String: NFKDecisionQuestion]) -> [String: NFKDecisionAnswer] {
        var answers = [String: NFKDecisionAnswer]()
        for (identifier, question) in questions {
            answers[identifier] = answer(state: state, question: question)
        }
        return answers
    }

    /// Answers one question about the state.
    @objc(answerForState:question:)
    public func answer(state: Any, question: NFKDecisionQuestion) -> NFKDecisionAnswer {
        let (probabilities, actProbability, options) = distribution(state: state, question: question)
        return NFKDecisionAnswer(dictionary: Self.wireAnswer(question: question, probabilities: probabilities,
                                                             actProbability: actProbability, options: options))!
    }

    /// The tokens a question builds over the state, or nil without a tokenizer.
    public func prompt(state: Any, question: NFKDecisionQuestion) -> NFKMLXLayaPrompt? {
        guard let tokenizer = holder.tokenizer else { return nil }
        return NFKMLXLayaPrompt(state: state, question: question, tokenizer: tokenizer, configuration: configuration)
    }

    /// The calibrated answer distribution over the rendered options, the act probability, and the
    /// option names in the distribution's order. How many tokens the question cost is `tokenCount`.
    public func distribution(state: Any, question: NFKDecisionQuestion)
        -> (probabilities: [Float], actProbability: Float, options: [String]) {
        let names = question.type == .noul ? ["false", "true"] : question.options
        guard let prompt = prompt(state: state, question: question), !prompt.markers.isEmpty else {
            let uniform = [Float](repeating: 1 / Float(Swift.max(names.count, 1)), count: names.count)
            return (uniform, 0.5, names)
        }
        let (logits, act) = holder.net.forward(tokens: prompt.tokens, markers: prompt.markers, type: question.type)
        let temperature = configuration.temperature(type: question.type, optionCount: prompt.markers.count)
        let probabilities = softmax(logits / temperature, axis: -1)
        let actProbabilities = softmax(act, axis: -1)
        eval(probabilities, actProbabilities)
        return (probabilities.asArray(Float.self), actProbabilities.asArray(Float.self)[0],
                Array(names.prefix(prompt.markers.count)))
    }

    /// The answer in the wire shape the hosted service and the reference API both return, which is
    /// what `NFKDecisionAnswer` reads and keeps under `raw`.
    static func wireAnswer(question: NFKDecisionQuestion, probabilities: [Float], actProbability: Float,
                           options: [String]) -> [String: Any] {
        let extra: [String: Any] = ["rl_agent": ["act_probability": Double(actProbability)]]
        let count = probabilities.count
        switch question.type {
        case .choice:
            let best = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
            var byName = [String: Double]()
            for (name, probability) in zip(options, probabilities) { byName[name] = Double(probability) }
            return ["type": "choice", "choice": options.isEmpty ? "" : options[best],
                    "probabilities": byName, "confidence": confidence(probabilities)].merging(extra) { a, _ in a }
        case .score:
            let expected = probabilities.enumerated().reduce(0.0) { $0 + Double($1.offset) * Double($1.element) }
            var legend = [String: String]()
            var byIndex = [String: Double]()
            for (index, level) in question.options.prefix(count).enumerated() {
                legend[String(index)] = level
                byIndex[String(index)] = Double(probabilities[index])
            }
            return ["type": "score", "score": expected, "legend": legend, "probabilities": byIndex,
                    "confidence": confidence(probabilities)].merging(extra) { a, _ in a }
        case .noul:
            return ["type": "noul", "noul": Double(count > 1 ? probabilities[1] : 0)].merging(extra) { a, _ in a }
        @unknown default:
            return ["type": "choice"]
        }
    }

    /// One minus the normalized entropy of the distribution: 1 when the answer is certain, 0 when it is
    /// uniform. A single option is certain by construction.
    static func confidence(_ probabilities: [Float]) -> Double {
        guard probabilities.count >= 2 else { return 1 }
        let entropy = -probabilities.reduce(0.0) { $0 + Double($1) * Foundation.log(Double(Swift.min(Swift.max($1, 1e-12), 1))) }
        return 1 - entropy / Foundation.log(Double(probabilities.count))
    }

    // MARK: Building

    /// The network, for training or inspection. Nil weights is the random initialization; a released
    /// or fine-tuned file loads through `NFKMLXWeights.loadCheckpoint`. Nothing here is transposed.
    public static func network(weightsURL: URL?, configuration: NFKMLXLayaConfiguration = .large) throws -> NFKMLXLayaNet {
        let net = NFKMLXLayaNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Loads one checkpoint file, released or fine-tuned, at float32.
    public static func loadWeights(into net: NFKMLXLayaNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { ($0.key, $0.value.asType(.float32)) }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    /// Loads a release directory's `model.safetensors`, refusing one the machine cannot hold.
    static func loadWeights(into net: NFKMLXLayaNet, fromDirectory directory: URL) throws {
        try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: .float32)
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: .float32)
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    /// Builds the model from a release directory: one of the three variants' folders, holding
    /// `model.safetensors`, `rl_agent_config.json`, `encoder/config.json`, and `tokenizer/`.
    @objc(layaWithDirectoryURL:error:)
    public static func laya(directoryURL: URL) throws -> NFKMLXLaya {
        try laya(directoryURL: directoryURL, weightsURL: nil)
    }

    /// Builds the model from a release directory with its weights replaced by a fine-tuned file, which
    /// is how Objective-C reaches a customized model.
    @objc(layaWithDirectoryURL:weightsURL:error:)
    public static func laya(directoryURL: URL, weightsURL: URL?) throws -> NFKMLXLaya {
        let configuration = try NFKMLXLayaConfiguration(directoryURL: directoryURL)
        let net = NFKMLXLayaNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        } else {
            try loadWeights(into: net, fromDirectory: directoryURL)
        }
        return NFKMLXLaya(net: net, tokenizer: NFKMLXLayaTokenizer.tokenizer(inDirectory: directoryURL))
    }

    /// Builds the model from optional local weights and a tokenizer, for a caller not loading a whole
    /// release directory. Nil weights builds random weights, which is what a smoke test uses.
    public static func laya(weightsURL: URL?, tokenizer: NFKMLXLayaTokenizer?,
                            configuration: NFKMLXLayaConfiguration = .large) throws -> NFKMLXLaya {
        NFKMLXLaya(net: try network(weightsURL: weightsURL, configuration: configuration), tokenizer: tokenizer)
    }

    /// The model as an inference backend, reading `NFKInputState` and `NFKInputQuestions`.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> NFKMLXLayaBackend {
        NFKMLXLayaBackend(laya: try laya(directoryURL: directoryURL))
    }

    /// This model as an inference backend.
    @objc public func makeBackend() -> NFKMLXLayaBackend { NFKMLXLayaBackend(laya: self) }
}

// MARK: - The backend

/// Laya behind the inference contract: the request `NFKTypeSafeBackend` reads, answered on device.
///
/// @discussion `NFKInputState` (a string, or a JSON-serializable dictionary or array; `NFKInputPrompt`
/// and then `NFKInputMessages` stand in for it) and `NFKInputQuestions` (a dictionary of
/// `NFKDecisionQuestion`s, or of dictionaries in the wire shape) in; `NFKDecisionAnswer`s under
/// `NFKOutputAnswers`, the reply in the hosted service's shape under `NFKOutputStructured`, and the
/// token count under `NFKOutputUsage` out. Blocks for the passes; run it off the render thread.
@objc(NFKMLXLayaBackend)
public final class NFKMLXLayaBackend: NSObject, NFKInferenceBackend {

    @objc public let laya: NFKMLXLaya

    @objc public init(laya: NFKMLXLaya) {
        self.laya = laya
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { "mlx-laya" }
    @objc public var supportedParameterKeys: Set<String> { [] }
    @objc public var supportedInputKeys: Set<String> { [NFKInputState, NFKInputQuestions, NFKInputPrompt, NFKInputMessages] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let state = Self.state(in: request) else {
            throw NSError(domain: NFKInferenceErrorDomain, code: NFKInferenceError.error_InferenceMissingInput.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "the request carries no state under NFKInputState, NFKInputPrompt, or NFKInputMessages"])
        }
        let questions = Self.questions(in: request)
        guard !questions.isEmpty else {
            throw NSError(domain: NFKInferenceErrorDomain, code: NFKInferenceError.error_InferenceMissingInput.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "the request carries no questions under NFKInputQuestions"])
        }
        var answers = [String: NFKDecisionAnswer]()
        var wire = [String: Any]()
        var tokens = 0
        for (identifier, question) in questions {
            let answer = laya.answer(state: state, question: question)
            answers[identifier] = answer
            wire[identifier] = answer.raw
            tokens += laya.prompt(state: state, question: question)?.tokens.count ?? 0
        }
        let reply: [String: Any] = ["model": laya.configuration.modelName, "answers": wire,
                                    "usage": ["input_tokens": tokens, "output_tokens": 0]]
        return NFKInferenceResult(outputs: [NFKOutputAnswers: answers, NFKOutputStructured: reply,
                                            NFKOutputUsage: [NFKUsageInputTokens: tokens, NFKUsageOutputTokens: 0]])
    }

    static func state(in request: NFKInferenceRequest) -> Any? {
        if let state = request.input(forKey: NFKInputState) { return state }
        if let prompt = request.prompt, !prompt.isEmpty { return prompt }
        return request.messages
    }

    /// The questions as `NFKDecisionQuestion`s: one already is, and a dictionary in the wire shape is
    /// read through `NFKDecisionQuestion(dictionary:)`.
    static func questions(in request: NFKInferenceRequest) -> [String: NFKDecisionQuestion] {
        guard let raw = request.input(forKey: NFKInputQuestions) as? [String: Any] else { return [:] }
        var questions = [String: NFKDecisionQuestion]()
        for (identifier, value) in raw {
            if let question = value as? NFKDecisionQuestion {
                questions[identifier] = question
            } else if let wire = value as? [String: Any], let question = NFKDecisionQuestion(dictionary: wire) {
                questions[identifier] = question
            }
        }
        return questions
    }
}
