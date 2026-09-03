//
//  NFKMLXEmbeddingGemma.swift
//  InferKitMLX
//
//  EmbeddingGemma-300M: a second text embedder, and a second embedding architecture. Where
//  Qwen3-Embedding is the causal decoder read one layer earlier, EmbeddingGemma is a BIDIRECTIONAL
//  encoder — the Gemma 3 text backbone with the causal mask removed — mean-pooled over every token
//  and run through a Dense bottleneck.
//
//  The backbone is Gemma 3 (`gemma3_text`), NOT the causal Gemma 4 (`gemma4_text`) `NFKMLXGemmaLanguage`
//  implements. The two are different: Gemma 3 normalizes with `x · (1 + w)` where Gemma 4 uses `x · w`,
//  Gemma 3 turns a full-attention head's whole width where Gemma 4 turns a fraction, and Gemma 3 carries
//  no per-layer input embeddings. This encoder is therefore its own implementation, focused on the
//  embedding path: bidirectional attention, no key-value cache, no logit head.
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// The geometry of the Gemma 3 text encoder EmbeddingGemma is built on.
public struct NFKMLXGemma3EncoderConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    /// Rotary base for the full-attention layers.
    public var ropeTheta: Float
    /// Rotary base for the sliding-window layers, which the config states separately.
    public var ropeLocalTheta: Float
    public var slidingWindow: Int
    /// How the layers alternate. Every `slidingWindowPattern`th layer is full attention; the rest are
    /// sliding-window. EmbeddingGemma is bidirectional, so a sliding layer sees a symmetric window and a
    /// full layer sees everything.
    public var slidingWindowPattern: Int
    /// The queries are scaled by `queryPreAttnScalar ** -0.5` before attention.
    public var queryPreAttnScalar: Float
    public var rmsEpsilon: Float

    public init(hiddenSize: Int = 768, layerCount: Int = 24, headCount: Int = 3,
                keyValueHeadCount: Int = 1, headDimensions: Int = 256, intermediateSize: Int = 1152,
                vocabularySize: Int = 262_144, ropeTheta: Float = 1_000_000, ropeLocalTheta: Float = 10_000,
                slidingWindow: Int = 512, slidingWindowPattern: Int = 6, queryPreAttnScalar: Float = 256,
                rmsEpsilon: Float = 1e-6) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.ropeLocalTheta = ropeLocalTheta
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.queryPreAttnScalar = queryPreAttnScalar
        self.rmsEpsilon = rmsEpsilon
    }

    /// The released `embeddinggemma-300m` backbone geometry.
    public static let embeddingGemma300M = NFKMLXGemma3EncoderConfiguration()

    /// A small configuration that runs with random weights, for tests. Keeps the 6-layer pattern so a
    /// full-attention layer is exercised.
    public static let tiny = NFKMLXGemma3EncoderConfiguration(
        hiddenSize: 64, layerCount: 6, headCount: 2, keyValueHeadCount: 1, headDimensions: 32,
        intermediateSize: 128, vocabularySize: 512, ropeTheta: 1_000_000, ropeLocalTheta: 10_000,
        slidingWindow: 4, slidingWindowPattern: 6, queryPreAttnScalar: 32)

    /// Whether the layer at `index` is full attention (every `slidingWindowPattern`th, counting from 1).
    func isFullAttention(layer index: Int) -> Bool { (index + 1) % slidingWindowPattern == 0 }
}

/// Gemma 3's normalization: `x · (1 + w)`, the weight initialized to zero. This is the difference from
/// Gemma 4's `x · w` that first broke a Gemma port here, so it is written out rather than shared.
final class NFKGemma3Norm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        epsilon = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = x * rsqrt((x * x).mean(axis: -1, keepDims: true) + epsilon)
        return normalized * (1 + weight)
    }
}

/// The rotary embedding at a layer's own base. A full-attention layer turns the entire head; Gemma 3
/// carries no partial rotary factor.
struct NFKGemma3Rotary {
    let dimensions: Int
    let base: Float

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.RoPE(x, dimensions: dimensions, traditional: false, base: base, scale: 1, offset: 0)
    }
}

/// Gemma 3 attention: grouped queries, per-head query and key normalization before the rotary, and a
/// window that most layers use. Bidirectional here, so nothing is masked by causality.
final class NFKGemma3Attention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: NFKGemma3Norm
    @ModuleInfo(key: "k_norm") var keyNorm: NFKGemma3Norm

    let heads: Int
    let keyValueHeads: Int
    let headDimensions: Int
    let scale: Float
    let rope: NFKGemma3Rotary

    init(_ c: NFKMLXGemma3EncoderConfiguration, fullAttention: Bool) {
        heads = c.headCount
        keyValueHeads = c.keyValueHeadCount
        headDimensions = c.headDimensions
        scale = pow(c.queryPreAttnScalar, -0.5)
        rope = NFKGemma3Rotary(dimensions: c.headDimensions,
                               base: fullAttention ? c.ropeTheta : c.ropeLocalTheta)

        _queryProjection.wrappedValue = Linear(c.hiddenSize, c.headCount * c.headDimensions, bias: false)
        _keyProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions, bias: false)
        _valueProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions, bias: false)
        _outputProjection.wrappedValue = Linear(c.headCount * c.headDimensions, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = NFKGemma3Norm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        _keyNorm.wrappedValue = NFKGemma3Norm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])

        var queries = queryProjection(x).reshaped([batch, length, heads, headDimensions])
        var keys = keyProjection(x).reshaped([batch, length, keyValueHeads, headDimensions])
        var values = valueProjection(x).reshaped([batch, length, keyValueHeads, headDimensions])

        // Gemma 3 normalizes each head BEFORE the rotary, over the head width.
        queries = queryNorm(queries)
        keys = keyNorm(keys)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        queries = rope(queries)
        keys = rope(keys)

        let attention = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: mask.map { $0.asType(queries.dtype) })
        return outputProjection(attention.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDimensions]))
    }
}

/// Gemma 3's GeGLU feed-forward: a gate through the tanh-approximate GELU, an up projection, their
/// product projected back down.
final class NFKGemma3FeedForward: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: NFKMLXGemma3EncoderConfiguration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(geluApproximate(gate(x)) * up(x)) }
}

/// One Gemma 3 block: the sandwich normalization, a norm before AND after each of attention and the
/// feed-forward, each pair added back to the block's input.
final class NFKGemma3Block: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKGemma3Attention
    @ModuleInfo(key: "mlp") var feedForward: NFKGemma3FeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: NFKGemma3Norm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: NFKGemma3Norm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: NFKGemma3Norm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: NFKGemma3Norm

    let fullAttention: Bool

    init(_ c: NFKMLXGemma3EncoderConfiguration, fullAttention: Bool) {
        self.fullAttention = fullAttention
        _attention.wrappedValue = NFKGemma3Attention(c, fullAttention: fullAttention)
        _feedForward.wrappedValue = NFKGemma3FeedForward(c)
        _inputNorm.wrappedValue = NFKGemma3Norm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postAttentionNorm.wrappedValue = NFKGemma3Norm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _preFeedForwardNorm.wrappedValue = NFKGemma3Norm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postFeedForwardNorm.wrappedValue = NFKGemma3Norm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, slidingMask: MLXArray?) -> MLXArray {
        let mask = fullAttention ? nil : slidingMask
        let attended = x + postAttentionNorm(attention(inputNorm(x), mask: mask))
        return attended + postFeedForwardNorm(feedForward(preFeedForwardNorm(attended)))
    }
}

/// The Gemma 3 text encoder: a scaled token embedding, the sandwich-normalized blocks, and a final
/// normalization. Bidirectional, so a forward reads the whole sequence at once with no cache.
public final class NFKMLXGemma3EncoderNet: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKGemma3Block]
    @ModuleInfo(key: "norm") var norm: NFKGemma3Norm

    let configuration: NFKMLXGemma3EncoderConfiguration
    private let embeddingScale: Float

    init(_ c: NFKMLXGemma3EncoderConfiguration) {
        configuration = c
        embeddingScale = sqrt(Float(c.hiddenSize))
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.layerCount).map { NFKGemma3Block(c, fullAttention: c.isFullAttention(layer: $0)) }
        _norm.wrappedValue = NFKGemma3Norm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    /// The token hidden states `[1, length, hidden]`, post-final-norm.
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var hidden = embedTokens(tokens) * embeddingScale
        let mask = slidingMask(length: tokens.shape[1])
        for layer in layers {
            hidden = layer(hidden, slidingMask: mask)
        }
        return norm(hidden)
    }

    /// The state entering the stack and the state each layer produces, the final norm applied to the
    /// last — the reference's `output_hidden_states` convention, so a divergence is located to a layer.
    func layerStates(_ tokens: MLXArray) -> [MLXArray] {
        var hidden = embedTokens(tokens) * embeddingScale
        var states = [hidden]
        let mask = slidingMask(length: tokens.shape[1])
        for layer in layers {
            hidden = layer(hidden, slidingMask: mask)
            states.append(hidden)
        }
        states[states.count - 1] = norm(hidden)
        return states
    }

    /// The bidirectional sliding-window mask, or nil when the sequence fits inside the window (which is
    /// the common case for an embedding input, and makes every layer full attention in practice). A
    /// token attends to any other within `slidingWindow` positions in either direction.
    private func slidingMask(length: Int) -> MLXArray? {
        guard length > configuration.slidingWindow else { return nil }
        let rows = MLXArray(0 ..< length).reshaped([length, 1])
        let columns = MLXArray(0 ..< length).reshaped([1, length])
        let within = abs(rows - columns) .< configuration.slidingWindow
        return MLX.where(within, MLXArray(Float(0)), MLXArray(Float(-1e9)))
    }
}

/// The pooling, projection, and markers EmbeddingGemma applies over the encoder's hidden states.
public struct NFKMLXEmbeddingGemmaConfiguration: Sendable {
    /// A token prepended to the encoded ids (the beginning-of-sequence marker), or nil to prepend none.
    public var prependedToken: Int?
    /// A token appended after the encoded ids (the end-of-sequence marker), or nil to append none.
    public var appendedToken: Int?
    /// The Matryoshka width to truncate the embedding to before the final normalization, or nil for the
    /// full width. EmbeddingGemma is trained so 512, 256, and 128 are usable truncations of its 768.
    public var dimensions: Int?

    public init(prependedToken: Int? = 2, appendedToken: Int? = 1, dimensions: Int? = nil) {
        self.prependedToken = prependedToken
        self.appendedToken = appendedToken
        self.dimensions = dimensions
    }
}

/// EmbeddingGemma as a text embedder: the bidirectional encoder, mean pooling over every token, the
/// two Dense projections the sentence-transformers head carries, Matryoshka truncation, and L2
/// normalization.
final class NFKMLXEmbeddingGemmaEmbedder: NFKTextEmbedding {
    let net: NFKMLXGemma3EncoderNet
    /// The first Dense projection `[3072, 768]`, applied as `w · x` (no bias, Identity activation).
    let dense2: MLXArray
    /// The second Dense projection `[768, 3072]`.
    let dense3: MLXArray
    let configuration: NFKMLXEmbeddingGemmaConfiguration

    init(net: NFKMLXGemma3EncoderNet, dense2: MLXArray, dense3: MLXArray,
         configuration: NFKMLXEmbeddingGemmaConfiguration) {
        self.net = net
        self.dense2 = dense2
        self.dense3 = dense3
        self.configuration = configuration
    }

    var embeddingDimensions: Int { configuration.dimensions ?? dense3.dim(0) }

    func embed(tokens: [Int]) -> MLXArray {
        var ids = tokens
        if let prepended = configuration.prependedToken { ids.insert(prepended, at: 0) }
        if let appended = configuration.appendedToken { ids.append(appended) }
        let input = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let hidden = net(input)                          // [1, length, hidden]

        let pooled = hidden[0].mean(axis: 0)             // mean pooling over every token
        var projected = dense3.matmul(dense2.matmul(pooled))   // 768 -> 3072 -> 768, no bias

        if let dimensions = configuration.dimensions, dimensions < projected.dim(0) {
            projected = projected[0 ..< dimensions]
        }
        return projected / sqrt((projected * projected).sum())
    }
}

/// EmbeddingGemma as an InferKit backend, and its Objective-C factories.
///
/// `NFKMLXEmbeddingGemma` is the released `google/embeddinggemma-300m` (through the ungated
/// `unsloth/embeddinggemma-300m` mirror): the bidirectional Gemma 3 encoder, mean-pooled, projected
/// through a Dense bottleneck, and L2-normalized. A query carries a task prompt and a document a
/// different one; ``query(_:)`` and ``document(_:)`` build the two forms the model is trained on.
@objc(NFKMLXEmbeddingGemma)
public final class NFKMLXEmbeddingGemma: NSObject {

    /// A name for the backend the factories produce.
    @objc public static let modelName = "embeddinggemma-300m"

    /// Formats a retrieval query the way EmbeddingGemma is trained to read it.
    @objc public static func query(_ text: String) -> String { "task: search result | query: \(text)" }

    /// Formats a document the way EmbeddingGemma is trained to read it, with no title.
    @objc public static func document(_ text: String) -> String { "title: none | text: \(text)" }

    /// Loads the backbone and the two Dense projections from a release directory.
    static func loadWeights(into net: NFKMLXGemma3EncoderNet, fromDirectory directory: URL)
        throws -> (dense2: MLXArray, dense3: MLXArray) {
        try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: .float32)
        // The backbone's keys are the checkpoint's (no `model.` prefix), so nothing is remapped. The
        // Dense projections live in their own subdirectories, which `files(inDirectory:)` does not read.
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: .float32)
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
        return (try dense(directory, "2_Dense"), try dense(directory, "3_Dense"))
    }

    private static func dense(_ directory: URL, _ subdirectory: String) throws -> MLXArray {
        let url = directory.appendingPathComponent(subdirectory).appendingPathComponent("model.safetensors")
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        guard let weight = checkpoint.arrays["linear.weight"] else {
            throw NFKMLXError.malformedCheckpoint("\(subdirectory)/model.safetensors has no linear.weight")
        }
        return weight.asType(.float32)
    }

    /// Builds an embedder from optional local weights and a tokenizer, for a caller not loading a whole
    /// release directory. A nil `weightsURL` builds random weights (the pipeline runs, the embeddings
    /// are meaningless), and nil Dense weights build random projections.
    public static func backend(weightsURL: URL?, dense2URL: URL?, dense3URL: URL?,
                               tokenizer: NFKTokenizer?,
                               configuration: NFKMLXGemma3EncoderConfiguration = .embeddingGemma300M,
                               embedding: NFKMLXEmbeddingGemmaConfiguration = NFKMLXEmbeddingGemmaConfiguration())
        throws -> any NFKInferenceBackend {
        let net = NFKMLXGemma3EncoderNet(configuration)
        if let weightsURL {
            let mapped = try NFKMLXWeights.loadCheckpoint(url: weightsURL).arrays.map { ($0, $1) }
            try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
        }
        let dense2 = try loadDense(dense2URL, rows: 4 * configuration.hiddenSize, columns: configuration.hiddenSize)
        let dense3 = try loadDense(dense3URL, rows: configuration.hiddenSize, columns: 4 * configuration.hiddenSize)
        let embedder = NFKMLXEmbeddingGemmaEmbedder(net: net, dense2: dense2, dense3: dense3,
                                                    configuration: embedding)
        return NFKMLXTextEmbeddingBackend(embedder: embedder, tokenizer: tokenizer, identifier: modelName)
    }

    private static func loadDense(_ url: URL?, rows: Int, columns: Int) throws -> MLXArray {
        guard let url else { return MLXRandom.normal([rows, columns]) * 0.02 }
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        guard let weight = checkpoint.arrays["linear.weight"] else {
            throw NFKMLXError.malformedCheckpoint("\(url.lastPathComponent) has no linear.weight")
        }
        return weight.asType(.float32)
    }

    /// Builds an EmbeddingGemma backend from a downloaded release directory holding the backbone,
    /// `config.json`, the `2_Dense`/`3_Dense` projections, and the tokenizer files.
    ///
    /// A nil `dimensions` keeps the full 768-wide embedding; a smaller value truncates each embedding to
    /// that Matryoshka width before normalizing. Run inference off the render thread.
    public static func backend(directoryURL: URL, dimensions: Int? = nil)
        throws -> any NFKInferenceBackend {
        let net = NFKMLXGemma3EncoderNet(.embeddingGemma300M)
        let (dense2, dense3) = try loadWeights(into: net, fromDirectory: directoryURL)
        let embedding = NFKMLXEmbeddingGemmaConfiguration(dimensions: dimensions)
        let embedder = NFKMLXEmbeddingGemmaEmbedder(net: net, dense2: dense2, dense3: dense3,
                                                    configuration: embedding)
        let tokenizer = NFKMLXGemmaTokenizer(directoryURL: directoryURL)
        return NFKMLXTextEmbeddingBackend(embedder: embedder,
                                          tokenize: tokenizer.map { t in { t.encode($0) } },
                                          identifier: modelName)
    }

    /// The Objective-C entry: builds from a release directory. Run inference off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, dimensions: nil)
    }

    /// The Objective-C entry that truncates each embedding to a Matryoshka width (0 keeps the full width).
    @objc(backendWithDirectoryURL:outputDimensions:error:)
    public static func backend(directoryURL: URL, outputDimensions dimensions: Int)
        throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, dimensions: dimensions > 0 ? dimensions : nil)
    }

}

/// Gemma's tokenizer, read directly from a release's `tokenizer.json`.
///
/// @discussion Gemma's fast tokenizer is byte-fallback BPE with a metaspace normalizer: a space becomes
/// `▁`, the whole normalized string is one pre-token, and merges combine characters by rank. It is
/// neither the byte-level BPE the GPT-2/Qwen path uses (which maps bytes into a printable alphabet) nor
/// the unigram Viterbi `NFKUnigramTokenizer` runs (Gemma's `tokenizer.model` scores are merge ranks, so
/// a max-score path picks the wrong pieces), so it is its own reader. No offline conversion is needed;
/// the release's `tokenizer.json` is read as it ships.
final class NFKMLXGemmaTokenizer {
    private let vocabulary: [String: Int]
    /// The id-to-piece reverse table, for decoding.
    private let pieces: [Int: String]
    /// A merge `"left\u{0}right"` mapped to its rank; a lower rank is a higher merge priority.
    private let ranks: [String: Int]
    private let unknownId: Int

    /// The metaspace SentencePiece renders a space as.
    private static let metaspace = "\u{2581}"

    /// The id of a special token literal (`<bos>`, `<start_of_turn>`), or nil when the vocabulary has
    /// no such piece.
    func id(forToken content: String) -> Int? { vocabulary[content] }

    /// The text a token-id sequence decodes to: each id's piece, with the metaspace turned back into a
    /// space and byte-fallback pieces (`<0xHH>`) reassembled into their bytes.
    func decode(_ ids: [Int]) -> String {
        var bytes = [UInt8]()
        for id in ids {
            guard let piece = pieces[id] else { continue }
            if piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
               let byte = UInt8(piece.dropFirst(3).dropLast(), radix: 16) {
                bytes.append(byte)
            } else {
                bytes.append(contentsOf: piece.replacingOccurrences(of: Self.metaspace, with: " ").utf8)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    convenience init?(directoryURL: URL) {
        self.init(tokenizerJSON: directoryURL.appendingPathComponent("tokenizer.json"))
    }

    init?(tokenizerJSON url: URL) {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any],
              let vocabulary = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [Any] else { return nil }
        self.vocabulary = vocabulary
        var pieces = [Int: String](minimumCapacity: vocabulary.count)
        for (piece, id) in vocabulary { pieces[id] = piece }
        self.pieces = pieces
        var ranks = [String: Int](minimumCapacity: merges.count)
        for (index, entry) in merges.enumerated() {
            // A merge is `["left", "right"]` in a recent tokenizer.json and `"left right"` in an older one.
            if let pair = entry as? [String], pair.count == 2 {
                ranks[pair[0] + "\u{0}" + pair[1]] = index
            } else if let text = entry as? String, let space = text.firstIndex(of: " ") {
                ranks[String(text[..<space]) + "\u{0}" + String(text[text.index(after: space)...])] = index
            }
        }
        self.ranks = ranks
        unknownId = (model["unk_token"] as? String).flatMap { vocabulary[$0] } ?? 3
    }

    /// The token ids for `text`, with no special markers (the embedder wraps them in BOS and EOS).
    func encode(_ text: String) -> [Int] {
        let normalized = text.replacingOccurrences(of: " ", with: Self.metaspace)
        // The space split the pre-tokenizer would do is a no-op after normalization, so the whole
        // string is one pre-token. Each character stands alone, or falls back to its UTF-8 bytes.
        var symbols = [String]()
        for scalar in normalized.unicodeScalars {
            let piece = String(scalar)
            if vocabulary[piece] != nil {
                symbols.append(piece)
            } else {
                for byte in Array(piece.utf8) { symbols.append(String(format: "<0x%02X>", byte)) }
            }
        }
        merge(&symbols)
        return symbols.map { vocabulary[$0] ?? unknownId }
    }

    /// The BPE merge loop: repeatedly merge the adjacent pair of highest priority (lowest rank) until
    /// none remains. A merged symbol is the concatenation of its parts, which the vocabulary carries.
    private func merge(_ symbols: inout [String]) {
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0 ..< (symbols.count - 1) {
                if let rank = ranks[symbols[index] + "\u{0}" + symbols[index + 1]], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            if bestIndex < 0 { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }
    }
}
