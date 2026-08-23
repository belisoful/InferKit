//
//  NFKMLXLanguageModel.swift
//  InferKitMLX
//
//  A decoder-only transformer for on-device text generation, in `MLXNN`. This is the capability the
//  package was missing: the core runs a Core ML language model and the Foundation Models companion
//  wraps Apple's system model, but nothing here ran a Qwen or Llama through MLX.
//
//  The architecture is the modern dense decoder — grouped-query attention with rotary embeddings, a
//  SwiGLU feed-forward, and RMS normalization throughout — which is what Qwen3 and Llama both are.
//  They differ in a configuration, not in structure: Qwen3 normalizes queries and keys per head and
//  ties its embeddings on the smaller sizes, Llama does neither.
//
//  Module keys are the Hugging Face names (`model.layers.N.self_attn.q_proj`), so a released
//  checkpoint loads with no key remapping at all.
//

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// The geometry of a decoder-only language model.
public struct NFKMLXLanguageConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    /// Key/value heads. Fewer than `headCount` is grouped-query attention, which every current
    /// release uses: the keys and values are shared across a group of query heads.
    public var keyValueHeadCount: Int
    public var headDimensions: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    public var ropeTheta: Float
    public var rmsEpsilon: Float
    /// Whether the output projection reuses the embedding matrix, as the smaller Qwen3 sizes do.
    public var tiesWordEmbeddings: Bool
    /// Whether queries and keys are RMS-normalized per head before the rotary embedding, which Qwen3
    /// does and Llama does not. A checkpoint carries `q_norm`/`k_norm` weights exactly when it applies.
    public var normalizesQueryAndKey: Bool
    /// Whether the attention projections carry biases (Qwen2 does; Qwen3 does not).
    public var attentionBias: Bool

    public init(hiddenSize: Int = 1024, layerCount: Int = 28, headCount: Int = 16,
                keyValueHeadCount: Int = 8, headDimensions: Int = 128, intermediateSize: Int = 3072,
                vocabularySize: Int = 151_936, ropeTheta: Float = 1_000_000, rmsEpsilon: Float = 1e-6,
                tiesWordEmbeddings: Bool = true, normalizesQueryAndKey: Bool = true,
                attentionBias: Bool = false) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.rmsEpsilon = rmsEpsilon
        self.tiesWordEmbeddings = tiesWordEmbeddings
        self.normalizesQueryAndKey = normalizesQueryAndKey
        self.attentionBias = attentionBias
    }

    /// The released `Qwen/Qwen3-0.6B` geometry.
    public static let qwen3_0_6B = NFKMLXLanguageConfiguration()

    /// The released `Qwen/Qwen3-1.7B` geometry.
    public static let qwen3_1_7B = NFKMLXLanguageConfiguration(
        hiddenSize: 2048, layerCount: 28, headCount: 16, keyValueHeadCount: 8,
        headDimensions: 128, intermediateSize: 6144)

    /// The released `Qwen/Qwen3-4B` geometry.
    public static let qwen3_4B = NFKMLXLanguageConfiguration(
        hiddenSize: 2560, layerCount: 36, headCount: 32, keyValueHeadCount: 8,
        headDimensions: 128, intermediateSize: 9728)

    /// The released `Qwen/Qwen3-8B` geometry. Its embeddings are not tied.
    public static let qwen3_8B = NFKMLXLanguageConfiguration(
        hiddenSize: 4096, layerCount: 36, headCount: 32, keyValueHeadCount: 8,
        headDimensions: 128, intermediateSize: 12288, tiesWordEmbeddings: false)

    /// A small configuration that runs with random weights, for tests.
    public static let tiny = NFKMLXLanguageConfiguration(
        hiddenSize: 64, layerCount: 2, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
        intermediateSize: 128, vocabularySize: 512, ropeTheta: 10_000)
}

/// The keys and values a layer has already computed, so a step attends to the whole prefix without
/// recomputing it.
///
/// Generation is quadratic without this and linear with it: the cache is what makes a token cost one
/// step's work rather than the whole sequence's.
public final class NFKMLXKeyValueCache {
    var keys: [MLXArray?]
    var values: [MLXArray?]
    /// How many positions the cache holds, which is the rotary offset the next step needs.
    public private(set) var offset = 0

    public init(layerCount: Int) {
        keys = Array(repeating: nil, count: layerCount)
        values = Array(repeating: nil, count: layerCount)
    }

    func update(layer: Int, keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existing = keys[layer], let existingValues = values[layer] {
            keys[layer] = concatenated([existing, newKeys], axis: 2)
            values[layer] = concatenated([existingValues, newValues], axis: 2)
        } else {
            keys[layer] = newKeys
            values[layer] = newValues
        }
        return (keys[layer]!, values[layer]!)
    }

    /// Advances the position count once every layer has been updated for this step.
    func advance(by count: Int) { offset += count }
}

/// Grouped-query attention with rotary embeddings.
final class NFKLMAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm?

    let heads: Int
    let keyValueHeads: Int
    let headDimensions: Int
    let scale: Float
    let rope: RoPE

    init(_ c: NFKMLXLanguageConfiguration) {
        heads = c.headCount
        keyValueHeads = c.keyValueHeadCount
        headDimensions = c.headDimensions
        scale = 1 / sqrt(Float(c.headDimensions))
        rope = RoPE(dimensions: c.headDimensions, traditional: false, base: c.ropeTheta)

        _queryProjection.wrappedValue = Linear(c.hiddenSize, c.headCount * c.headDimensions,
                                               bias: c.attentionBias)
        _keyProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions,
                                             bias: c.attentionBias)
        _valueProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions,
                                               bias: c.attentionBias)
        _outputProjection.wrappedValue = Linear(c.headCount * c.headDimensions, c.hiddenSize, bias: false)
        // Present exactly when the checkpoint carries them, so a strict load stays strict either way.
        _queryNorm.wrappedValue = c.normalizesQueryAndKey
            ? RMSNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon) : nil
        _keyNorm.wrappedValue = c.normalizesQueryAndKey
            ? RMSNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: NFKMLXKeyValueCache?,
                        layer: Int) -> MLXArray {
        let (batch, length, _) = (x.shape[0], x.shape[1], x.shape[2])

        var queries = queryProjection(x).reshaped([batch, length, heads, headDimensions])
        var keys = keyProjection(x).reshaped([batch, length, keyValueHeads, headDimensions])
        var values = valueProjection(x).reshaped([batch, length, keyValueHeads, headDimensions])

        // Qwen3 normalizes each head BEFORE the rotary embedding; the norm is over the head, not the
        // whole projection, which is why it is applied while the head axis is still separate.
        if let queryNorm { queries = queryNorm(queries) }
        if let keyNorm { keys = keyNorm(keys) }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        if let cache {
            (keys, values) = cache.update(layer: layer, keys: keys, values: values)
        }

        // A `.checkpoint`-precision load makes this a bf16 module, and the fused attention refuses
        // a float32 mask that does not promote to its own type — invisible at float32.
        let attention = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: mask.map { $0.asType(queries.dtype) })
        return outputProjection(attention.transposed(0, 2, 1, 3)
            .reshaped([batch, length, heads * headDimensions]))
    }
}

/// The SwiGLU feed-forward: a gate and an up projection multiplied, then projected back down.
final class NFKLMFeedForward: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: NFKMLXLanguageConfiguration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

/// One transformer block: pre-normalized attention and feed-forward, each added back.
final class NFKLMBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKLMAttention
    @ModuleInfo(key: "mlp") var feedForward: NFKLMFeedForward
    @ModuleInfo(key: "input_layernorm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var feedForwardNorm: RMSNorm

    init(_ c: NFKMLXLanguageConfiguration) {
        _attention.wrappedValue = NFKLMAttention(c)
        _feedForward.wrappedValue = NFKLMFeedForward(c)
        _attentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _feedForwardNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: NFKMLXKeyValueCache?,
                        layer: Int) -> MLXArray {
        let attended = x + attention(attentionNorm(x), mask: mask, cache: cache, layer: layer)
        return attended + feedForward(feedForwardNorm(attended))
    }
}

/// The transformer stack, under the `model.` prefix the released checkpoints use.
final class NFKLMCore: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKLMBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ c: NFKMLXLanguageConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.layerCount).map { _ in NFKLMBlock(c) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }
}

/// A decoder-only language model.
public final class NFKMLXLanguageNet: Module {
    @ModuleInfo(key: "model") var model: NFKLMCore
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: NFKMLXLanguageConfiguration

    init(_ c: NFKMLXLanguageConfiguration) {
        configuration = c
        _model.wrappedValue = NFKLMCore(c)
        // A tied checkpoint carries no `lm_head.weight`: the embedding matrix is the output
        // projection. Building one anyway would leave it randomly initialized and produce noise.
        _lmHead.wrappedValue = c.tiesWordEmbeddings
            ? nil : Linear(c.hiddenSize, c.vocabularySize, bias: false)
        super.init()
    }

    /// Runs the stack over `tokens` `[batch, length]` and returns logits `[batch, length, vocabulary]`.
    func callAsFunction(_ tokens: MLXArray, cache: NFKMLXKeyValueCache? = nil) -> MLXArray {
        var hidden = model.embedTokens(tokens)
        let length = tokens.shape[1]
        // A single token attends to everything cached, so it needs no mask; a prefill does.
        let mask: MLXArray? = length > 1 ? NFKMLXLanguageNet.causalMask(length, offset: cache?.offset ?? 0)
                                         : nil
        for (index, layer) in model.layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache, layer: index)
        }
        cache?.advance(by: length)
        hidden = model.norm(hidden)
        if let lmHead { return lmHead(hidden) }
        return model.embedTokens.asLinear(hidden)
    }

    /// An additive mask that forbids attending to later positions.
    static func causalMask(_ length: Int, offset: Int) -> MLXArray {
        let rows = MLXArray(0 ..< length).reshaped([length, 1]) + offset
        let columns = MLXArray(0 ..< (length + offset)).reshaped([1, length + offset])
        return MLX.where(columns .<= rows, MLXArray(Float(0)), MLXArray(Float(-1e9)))
    }
}
