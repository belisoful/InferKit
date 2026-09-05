//
//  NFKMLXChatterboxT3.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// Chatterbox stage 3: T3, the text-to-speech-token model. A Llama decoder (520M, llama3 rope scaling)
// runs over one embedded sequence: a 34-token conditioning prefix (the speaker embedding, a 32-token
// Perceiver resample of the voice prompt's speech codes, and an emotion scalar), the text tokens with
// learned position embeddings, and the speech tokens generated so far with their own learned position
// embeddings. A speech head over the final hidden state predicts the next S3 speech code; sampling runs
// classifier-free guidance as a batch of two (the second row with its text embedding zeroed).

// MARK: - Configuration

/// The T3 geometry (`T3Config.english_only` over `Llama_520M`).
public struct NFKMLXT3Configuration: Sendable {
    public var textVocabulary: Int = 704
    public var speechVocabulary: Int = 8194
    public var hiddenSize: Int = 1024
    public var startTextToken: Int = 255
    public var stopTextToken: Int = 0
    public var startSpeechToken: Int = 6561
    public var stopSpeechToken: Int = 6562
    /// The learned text position table, `max_text_tokens + 2`.
    public var textPositions: Int = 2050
    /// The learned speech position table, `max_speech_tokens + 4`.
    public var speechPositions: Int = 4100
    public var speakerEmbeddingSize: Int = 256
    public var perceiverQueries: Int = 32
    public var perceiverHeads: Int = 4
    /// How many prompt speech codes condition the model (`speech_cond_prompt_len`).
    public var speechPromptLength: Int = 150
    public var decoder: NFKMLXLanguageConfiguration
    public init(decoder: NFKMLXLanguageConfiguration) { self.decoder = decoder }

    /// The released English model.
    public static var released: NFKMLXT3Configuration {
        var llama = NFKMLXLanguageConfiguration(
            hiddenSize: 1024, layerCount: 30, headCount: 16, keyValueHeadCount: 16, headDimensions: 64,
            intermediateSize: 4096, vocabularySize: 8, ropeTheta: 500_000, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: true, normalizesQueryAndKey: false, attentionBias: false)
        llama.ropeScaling = NFKMLXRoPEScaling(kind: .llama3, factor: 8, originalMaxPositionEmbeddings: 8192,
                                              lowFrequencyFactor: 1, highFrequencyFactor: 4)
        return NFKMLXT3Configuration(decoder: llama)
    }
    public static var tiny: NFKMLXT3Configuration {
        var llama = NFKMLXLanguageConfiguration(
            hiddenSize: 32, layerCount: 2, headCount: 4, keyValueHeadCount: 4, headDimensions: 8,
            intermediateSize: 64, vocabularySize: 8, ropeTheta: 500_000, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: true, normalizesQueryAndKey: false, attentionBias: false)
        llama.ropeScaling = NFKMLXRoPEScaling(kind: .llama3, factor: 8, originalMaxPositionEmbeddings: 8192)
        var configuration = NFKMLXT3Configuration(decoder: llama)
        configuration.hiddenSize = 32
        configuration.textVocabulary = 300
        configuration.speechVocabulary = 6570
        configuration.speakerEmbeddingSize = 8
        configuration.perceiverQueries = 4
        configuration.perceiverHeads = 2
        configuration.textPositions = 64
        configuration.speechPositions = 64
        return configuration
    }
}

// MARK: - Conditioning

/// The Perceiver's attention block (`AttentionBlock2`): one LayerNorm shared by both inputs, separate
/// query/key/value projections, and a residual over the first input.
final class NFKT3AttentionBlock: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "to_q") var toQuery: Linear
    @ModuleInfo(key: "to_k") var toKey: Linear
    @ModuleInfo(key: "to_v") var toValue: Linear
    @ModuleInfo(key: "proj_out") var projection: Linear
    let heads: Int

    init(channels: Int, heads: Int) {
        self.heads = heads
        _norm.wrappedValue = LayerNorm(dimensions: channels)
        _toQuery.wrappedValue = Linear(channels, channels)
        _toKey.wrappedValue = Linear(channels, channels)
        _toValue.wrappedValue = Linear(channels, channels)
        _projection.wrappedValue = Linear(channels, channels)
        super.init()
    }

    func callAsFunction(_ x1: MLXArray, _ x2: MLXArray) -> MLXArray {
        let (batch, width) = (x1.dim(0), x1.dim(2))
        let headDim = width / heads
        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped([batch, t.dim(1), heads, headDim]).transposed(0, 2, 1, 3)
        }
        let normed2 = norm(x2)
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(toQuery(norm(x1))), keys: split(toKey(normed2)), values: split(toValue(normed2)),
            scale: 1 / sqrt(Float(headDim)), mask: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, x1.dim(1), width])
        return x1 + projection(merged)
    }
}

/// The Perceiver resampler: learned queries cross-attend to the prompt, then self-attend once.
final class NFKT3Perceiver: Module {
    @ParameterInfo(key: "pre_attention_query") var query: MLXArray
    @ModuleInfo(key: "attn") var attention: NFKT3AttentionBlock

    init(queries: Int, channels: Int, heads: Int) {
        _query.wrappedValue = MLXArray.zeros([1, queries, channels])
        _attention.wrappedValue = NFKT3AttentionBlock(channels: channels, heads: heads)
        super.init()
    }

    func callAsFunction(_ prompt: MLXArray) -> MLXArray {
        let queries = broadcast(query, to: [prompt.dim(0), query.dim(1), query.dim(2)])
        let pre = attention(queries, prompt)
        return attention(pre, pre)
    }
}

/// `T3CondEnc`: the speaker projection, the Perceiver over the prompt embeddings, and the emotion
/// projection, concatenated along the sequence.
final class NFKT3ConditionEncoder: Module {
    @ModuleInfo(key: "spkr_enc") var speaker: Linear
    @ModuleInfo(key: "emotion_adv_fc") var emotion: Linear
    @ModuleInfo(key: "perceiver") var perceiver: NFKT3Perceiver

    init(_ configuration: NFKMLXT3Configuration) {
        _speaker.wrappedValue = Linear(configuration.speakerEmbeddingSize, configuration.hiddenSize)
        _emotion.wrappedValue = Linear(1, configuration.hiddenSize, bias: false)
        _perceiver.wrappedValue = NFKT3Perceiver(queries: configuration.perceiverQueries,
                                                 channels: configuration.hiddenSize,
                                                 heads: configuration.perceiverHeads)
        super.init()
    }

    /// - Parameters:
    ///   - speakerEmbedding: `[batch, speakerEmbeddingSize]`
    ///   - promptEmbedding: the prompt's speech-token embeddings with positions, `[batch, prompt, hidden]`
    ///   - emotion: the exaggeration scalar per batch row, `[batch]`
    func callAsFunction(speakerEmbedding: MLXArray, promptEmbedding: MLXArray, emotion: MLXArray) -> MLXArray {
        let speakerToken = speaker(speakerEmbedding).expandedDimensions(axis: 1)
        let emotionToken = self.emotion(emotion.reshaped([-1, 1, 1]))
        return concatenated([speakerToken, perceiver(promptEmbedding), emotionToken], axis: 1)
    }
}

/// `LearnedPositionEmbeddings`: an embedding table read at the sequence positions.
final class NFKT3LearnedPositions: Module {
    @ModuleInfo(key: "emb") var table: Embedding

    init(positions: Int, dimensions: Int) {
        _table.wrappedValue = Embedding(embeddingCount: positions, dimensions: dimensions)
        super.init()
    }

    /// The embeddings of positions `0 ..< length`, `[length, dimensions]`.
    func callAsFunction(length: Int) -> MLXArray {
        table(MLXArray(0 ..< Int32(length)))
    }

    /// One position's embedding, `[1, 1, dimensions]`.
    func embedding(at position: Int) -> MLXArray {
        table(MLXArray([Int32(position)]).reshaped([1, 1]))
    }
}

/// What conditions a T3 generation: the VoiceEncoder speaker embedding, the prompt's S3 speech codes
/// (at most `speechPromptLength` of them), and the exaggeration scalar.
public struct NFKMLXT3Condition {
    public var speakerEmbedding: MLXArray
    public var promptTokens: [Int]
    public var exaggeration: Float
    public init(speakerEmbedding: MLXArray, promptTokens: [Int], exaggeration: Float = 0.5) {
        self.speakerEmbedding = speakerEmbedding
        self.promptTokens = promptTokens
        self.exaggeration = exaggeration
    }
}

// MARK: - Sampling

/// The T3 sampling controls, defaulting to `ChatterboxTTS.generate`'s.
public struct NFKMLXT3SamplingOptions: Sendable {
    /// Classifier-free guidance weight; 0 runs the conditional row alone.
    public var guidance: Float = 0.5
    /// 0 selects the most likely code at every step, which the reference has no path for and this
    /// port adds so a run is repeatable without a seed.
    public var temperature: Float = 0.8
    public var repetitionPenalty: Float = 1.2
    public var minP: Float = 0.05
    public var topP: Float = 1
    public var maximumTokens: Int = 1000
    public var seed: UInt64 = 0
    public init() {}
}

// MARK: - The model

/// T3: text tokens and a voice condition in, S3 speech codes out.
public final class NFKMLXT3Net: Module {
    @ModuleInfo(key: "tfmr") var decoder: NFKMLXLanguageNet
    @ModuleInfo(key: "cond_enc") var conditionEncoder: NFKT3ConditionEncoder
    @ModuleInfo(key: "text_emb") var textEmbedding: Embedding
    @ModuleInfo(key: "speech_emb") var speechEmbedding: Embedding
    @ModuleInfo(key: "text_pos_emb") var textPositions: NFKT3LearnedPositions
    @ModuleInfo(key: "speech_pos_emb") var speechPositions: NFKT3LearnedPositions
    @ModuleInfo(key: "text_head") var textHead: Linear
    @ModuleInfo(key: "speech_head") var speechHead: Linear
    public let configuration: NFKMLXT3Configuration

    public init(_ configuration: NFKMLXT3Configuration = .released) {
        self.configuration = configuration
        let width = configuration.hiddenSize
        _decoder.wrappedValue = NFKMLXLanguageNet(configuration.decoder)
        _conditionEncoder.wrappedValue = NFKT3ConditionEncoder(configuration)
        _textEmbedding.wrappedValue = Embedding(embeddingCount: configuration.textVocabulary, dimensions: width)
        _speechEmbedding.wrappedValue = Embedding(embeddingCount: configuration.speechVocabulary, dimensions: width)
        _textPositions.wrappedValue = NFKT3LearnedPositions(positions: configuration.textPositions, dimensions: width)
        _speechPositions.wrappedValue = NFKT3LearnedPositions(positions: configuration.speechPositions, dimensions: width)
        _textHead.wrappedValue = Linear(width, configuration.textVocabulary, bias: false)
        _speechHead.wrappedValue = Linear(width, configuration.speechVocabulary, bias: false)
        super.init()
    }

    private func tokens(_ ids: [Int]) -> MLXArray {
        MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
    }

    /// The conditioning prefix `[1, 2 + perceiverQueries, hidden]` (`prepare_conditioning`): the prompt
    /// codes are embedded with their speech positions before the Perceiver reads them.
    public func conditionEmbedding(_ condition: NFKMLXT3Condition) -> MLXArray {
        let prompt = Array(condition.promptTokens.prefix(configuration.speechPromptLength))
        let promptEmbedding = speechEmbedding(tokens(prompt)) + speechPositions(length: prompt.count)
        return conditionEncoder(speakerEmbedding: condition.speakerEmbedding.reshaped([1, -1]),
                                promptEmbedding: promptEmbedding,
                                emotion: MLXArray([condition.exaggeration]))
    }

    /// The decoder's input sequence (`prepare_input_embeds`): the condition, the text with its learned
    /// positions, and the start-of-speech token at speech position 0. With `guidance` a second row
    /// carries the same sequence with its text EMBEDDING zeroed (the positions stay), which is the
    /// unconditional branch of the classifier-free guidance.
    public func inputEmbeddings(condition: MLXArray, textTokens: [Int], guidance: Bool) -> MLXArray {
        let text = textEmbedding(tokens(textTokens))
        let positions = textPositions(length: textTokens.count)
        let start = speechEmbedding(tokens([configuration.startSpeechToken])) + speechPositions.embedding(at: 0)
        let conditional = concatenated([condition, text + positions, start], axis: 1)
        guard guidance else { return conditional }
        let unconditional = concatenated([condition, MLXArray.zeros(like: text) + positions, start], axis: 1)
        return concatenated([conditional, unconditional], axis: 0)
    }

    /// Speech logits `[batch, length, speechVocabulary]` over already-embedded inputs.
    public func speechLogits(embeddings: MLXArray, cache: NFKMLXKeyValueCache? = nil) -> MLXArray {
        speechHead(decoder.hiddenStates(fromEmbeddings: embeddings, cache: cache))
    }

    /// A speech token embedded at speech position `position`, `[1, 1, hidden]`.
    public func speechTokenEmbedding(_ token: Int, position: Int) -> MLXArray {
        speechEmbedding(tokens([token])) + speechPositions.embedding(at: position)
    }

    /// Generates speech codes for `textTokens` (which must already carry the start and stop text
    /// tokens). Returns the codes without the start token and without the stop token.
    ///
    /// @discussion Reproduces `T3.inference`, including its quirk of feeding the start-of-speech
    /// embedding TWICE (once inside the prepared inputs and once more as the first decode input, both
    /// at speech position 0); the model was sampled that way, so the port keeps it. Each step reads
    /// the last position's speech logits, combines the two guidance rows, applies the repetition
    /// penalty, temperature, min-p and top-p, and samples.
    public func generate(condition: NFKMLXT3Condition, textTokens: [Int],
                         options: NFKMLXT3SamplingOptions = NFKMLXT3SamplingOptions(),
                         shouldContinue: () -> Bool = { true }) -> [Int] {
        let guided = options.guidance > 0
        let prepared = inputEmbeddings(condition: conditionEmbedding(condition), textTokens: textTokens,
                                       guidance: guided)
        let rows = prepared.dim(0)
        let start = speechTokenEmbedding(configuration.startSpeechToken, position: 0)
        var input = concatenated([prepared, broadcast(start, to: [rows, 1, start.dim(2)])], axis: 1)
        let cache = NFKMLXKeyValueCache(layerCount: configuration.decoder.layerCount)
        var generated = [configuration.startSpeechToken]
        var sampler = NFKMLXT3Sampler(seed: options.seed)
        for step in 0 ..< options.maximumTokens {
            guard shouldContinue() else { break }
            let logits = speechLogits(embeddings: input, cache: cache)[0..., -1, 0...]
            let scores: MLXArray = guided
                ? logits[0 ..< 1] + options.guidance * (logits[0 ..< 1] - logits[1 ..< 2])
                : logits[0 ..< 1]
            let processed = NFKMLXT3Sampler.processed(scores[0].asArray(Float.self), generated: generated,
                                                      options: options)
            let next = sampler.sample(processed, temperature: options.temperature)
            generated.append(next)
            if next == configuration.stopSpeechToken { break }
            let embedding = speechTokenEmbedding(next, position: step + 1)
            input = broadcast(embedding, to: [rows, 1, embedding.dim(2)])
        }
        return generated.dropFirst().filter { $0 != configuration.stopSpeechToken }
    }
}

/// The reference's logits processors, in its order, over one step's guided scores.
public struct NFKMLXT3Sampler {
    private var state: UInt64

    public init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }

    /// `RepetitionPenaltyLogitsProcessor` (every token generated so far, the start token included,
    /// has its score divided by the penalty when positive and multiplied when negative), then the
    /// temperature, then `MinPLogitsWarper` (a token whose probability falls below `minP` times the
    /// top probability is removed, the top token always kept), then `TopPLogitsWarper` (the least
    /// likely tokens whose cumulative probability stays within `1 - topP` are removed, the most likely
    /// always kept). Removed tokens read `-inf`.
    public static func processed(_ scores: [Float], generated: [Int], options: NFKMLXT3SamplingOptions) -> [Float] {
        var scores = scores
        for token in Set(generated) where token < scores.count {
            scores[token] = scores[token] < 0 ? scores[token] * options.repetitionPenalty
                                              : scores[token] / options.repetitionPenalty
        }
        if options.temperature > 0, options.temperature != 1 {
            scores = scores.map { $0 / options.temperature }
        }
        let probabilities = softmax(scores)
        if options.minP > 0, let top = probabilities.max() {
            let threshold = options.minP * top
            let keep = probabilities.firstIndex(of: top)
            for index in scores.indices where probabilities[index] < threshold && index != keep {
                scores[index] = -.infinity
            }
        }
        if options.topP < 1 {
            let order = scores.indices.sorted { scores[$0] < scores[$1] }
            let sorted = softmax(order.map { scores[$0] })
            var cumulative: Double = 0
            for (rank, index) in order.enumerated() where rank < order.count - 1 {
                cumulative += Double(sorted[rank])
                if cumulative <= Double(1 - options.topP) { scores[index] = -.infinity }
            }
        }
        return scores
    }

    private static func softmax(_ scores: [Float]) -> [Float] {
        let peak = scores.max() ?? 0
        let exponents = scores.map { $0 == -.infinity ? 0 : expf($0 - peak) }
        let total = exponents.reduce(0, +)
        return exponents.map { $0 / total }
    }

    /// Draws a token from processed scores: the most likely at temperature 0, otherwise a sample from
    /// their softmax through a SplitMix64 stream, so a seeded run repeats.
    public mutating func sample(_ scores: [Float], temperature: Float) -> Int {
        guard temperature > 0 else {
            return scores.indices.max { scores[$0] < scores[$1] } ?? 0
        }
        let probabilities = NFKMLXT3Sampler.softmax(scores)
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        let draw = Double(z >> 11) / Double(1 << 53)
        var cumulative: Double = 0
        for (index, probability) in probabilities.enumerated() {
            cumulative += Double(probability)
            if draw < cumulative { return index }
        }
        return probabilities.indices.last { probabilities[$0] > 0 } ?? 0
    }
}

// MARK: - Text tokenizer

/// The T3 text tokenizer (`EnTokenizer`): the release's `tokenizer.json`, a plain character-level BPE
/// with a `Whitespace` pre-tokenizer and literal added tokens. Spaces become the `[SPACE]` token
/// before encoding, as the reference does.
public final class NFKMLXChatterboxTextTokenizer {
    private let vocabulary: [String: Int]
    private let ranks: [String: Int]
    private let addedTokens: [(content: String, id: Int)]
    private let unknown: Int
    private let pattern = try! NSRegularExpression(pattern: #"\w+|[^\w\s]+"#)
    public let startToken: Int
    public let stopToken: Int
    public let spaceToken: Int

    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any], let vocab = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a tokenizers BPE file")
        }
        vocabulary = vocab
        var ranks = [String: Int]()
        for (rank, merge) in merges.enumerated() {
            if let pair = merge as? String { ranks[pair] = rank }
            else if let parts = merge as? [String], parts.count == 2 { ranks[parts[0] + " " + parts[1]] = rank }
        }
        self.ranks = ranks
        let added = (root["added_tokens"] as? [[String: Any]] ?? []).compactMap { entry -> (String, Int)? in
            guard let content = entry["content"] as? String, let id = entry["id"] as? Int else { return nil }
            return (content, id)
        }
        addedTokens = added.sorted { $0.0.count > $1.0.count }
        unknown = vocab[model["unk_token"] as? String ?? "[UNK]"] ?? 1
        startToken = vocab["[START]"] ?? 255
        stopToken = vocab["[STOP]"] ?? 0
        spaceToken = vocab["[SPACE]"] ?? 2
    }

    /// `punc_norm`: the reference's cleanup before tokenization.
    public static func normalizedPunctuation(_ text: String) -> String {
        if text.isEmpty { return "You need to add some text for me to talk." }
        var text = text
        if let first = text.first, first.isLowercase {
            text = String(first).uppercased() + text.dropFirst()
        }
        text = text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        for (old, new) in [("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "), ("—", "-"),
                           ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'")] {
            text = text.replacingOccurrences(of: old, with: new)
        }
        while text.hasSuffix(" ") { text.removeLast() }
        if !text.hasSuffix(".") && !text.hasSuffix("!") && !text.hasSuffix("?") && !text.hasSuffix("-") && !text.hasSuffix(",") {
            text += "."
        }
        return text
    }

    /// The ids for `text` alone (no start or stop token).
    public func encode(_ text: String) -> [Int] {
        var ids = [Int]()
        for segment in split(text.replacingOccurrences(of: " ", with: "[SPACE]")) {
            switch segment {
            case .added(let id): ids.append(id)
            case .text(let run):
                let range = NSRange(run.startIndex ..< run.endIndex, in: run)
                for match in pattern.matches(in: run, range: range) {
                    guard let wordRange = Range(match.range, in: run) else { continue }
                    ids.append(contentsOf: bpe(String(run[wordRange])))
                }
            }
        }
        return ids
    }

    /// The text with the model's start and stop tokens, ready for T3.
    public func encodeForSynthesis(_ text: String) -> [Int] {
        [startToken] + encode(text) + [stopToken]
    }

    private enum Segment { case added(Int), text(String) }

    private func split(_ text: String) -> [Segment] {
        var segments = [Segment]()
        var pending = ""
        var index = text.startIndex
        while index < text.endIndex {
            if let hit = addedTokens.first(where: { text[index...].hasPrefix($0.content) }) {
                if !pending.isEmpty { segments.append(.text(pending)); pending = "" }
                segments.append(.added(hit.id))
                index = text.index(index, offsetBy: hit.content.count)
            } else {
                pending.append(text[index])
                index = text.index(after: index)
            }
        }
        if !pending.isEmpty { segments.append(.text(pending)) }
        return segments
    }

    private func bpe(_ word: String) -> [Int] {
        var symbols = word.unicodeScalars.map { String($0) }
        while symbols.count > 1 {
            var best: (rank: Int, index: Int)?
            for index in 0 ..< symbols.count - 1 {
                if let rank = ranks[symbols[index] + " " + symbols[index + 1]], rank < (best?.rank ?? .max) {
                    best = (rank, index)
                }
            }
            guard let merge = best else { break }
            symbols.replaceSubrange(merge.index ... merge.index + 1, with: [symbols[merge.index] + symbols[merge.index + 1]])
        }
        return symbols.map { vocabulary[$0] ?? unknown }
    }
}

// MARK: - Loading

extension NFKMLXChatterbox {
    /// Loads the released `t3_cfg.safetensors`. The Llama sits under `tfmr.` in the checkpoint and under
    /// `tfmr.model.` here (the shared decoder keeps the transformers `model.` prefix); nothing needs a
    /// transpose.
    public static func loadT3Weights(into net: NFKMLXT3Net, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var mapped = [(String, MLXArray)]()
        for (key, value) in checkpoint.arrays {
            var name = key
            if name.hasPrefix("tfmr.") { name = "tfmr.model." + name.dropFirst("tfmr.".count) }
            mapped.append((name, value))
        }
        try NFKMLXWeights.apply(mapped, to: net)
        net.train(false)
    }
}
