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
    /// The rotary scaling the release declares, or `nil` for none. Read from the checkpoint's own
    /// `rope_scaling`; see ``NFKMLXRoPEScaling``.
    public var ropeScaling: NFKMLXRoPEScaling?

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
///
/// @discussion The rows live in a buffer that grows in blocks, with a cursor at each end, so a step
/// writes into space that is already there. Concatenating instead would copy the whole cache on every
/// token, which turns decoding back into quadratic work in a place that exists to avoid exactly that.
///
/// **A window bounds it.** `NFKMLXKeyValueCache(layerCount:window:)` keeps at most `window` positions:
/// the oldest are dropped as new ones arrive, so memory stops growing with the conversation. That is a
/// real change to what the model reads, and it is only free when the model's own attention is
/// windowed at the same width. For a model with full attention it is a deliberate approximation —
/// the model stops seeing the beginning of its context — so it is off unless asked for.
public final class NFKMLXKeyValueCache {

    /// The most positions the cache retains, or `nil` for an unbounded one.
    public let window: Int?

    /// TOTAL positions ever appended, which is NOT the retained count once a window starts dropping
    /// them. This is the rotary offset a step needs: a token's angle depends on where it actually is
    /// in the sequence, not on where it sits in the buffer.
    public private(set) var offset = 0

    private var keys: [MLXArray?]
    private var values: [MLXArray?]
    /// The retained span of each layer's buffer, as `[start, end)`.
    private var starts: [Int]
    private var ends: [Int]

    /// Rows are added in blocks so that a steady decode compacts rarely rather than every step.
    private static let growthBlock = 256

    public init(layerCount: Int, window: Int? = nil) {
        precondition(window.map { $0 > 1 } ?? true, "a window keeps more than one position")
        self.window = window
        keys = Array(repeating: nil, count: layerCount)
        values = Array(repeating: nil, count: layerCount)
        starts = Array(repeating: 0, count: layerCount)
        ends = Array(repeating: 0, count: layerCount)
    }

    /// Rows a layer currently holds.
    public func retainedLength(layer: Int = 0) -> Int { ends[layer] - starts[layer] }

    /// How many cached positions a forward pass of more than one token will attend to, which is what
    /// its mask has to be built against.
    ///
    /// @discussion With a window this is the retained length AFTER the trim the next update performs,
    /// so it is smaller than ``offset``. A mask built against the absolute offset would be wider than
    /// the keys it is applied to.
    public var maskCacheLength: Int {
        let retained = ends.isEmpty ? 0 : retainedLength(layer: 0)
        guard let window else { return retained }
        return Swift.min(retained, window - 1)
    }

    /// Drops the oldest rows so at most `window - 1` precede the incoming ones.
    ///
    /// @discussion Trimming to one short of the window means a single-token step reads exactly
    /// `window` positions and needs no sliding mask: everything retained is something it may attend
    /// to. The trim moves a cursor and copies nothing.
    private func trimForAppend(layer: Int) {
        guard let window else { return }
        let excess = retainedLength(layer: layer) - (window - 1)
        if excess > 0 { starts[layer] += excess }
    }

    /// Makes room for `extra` more rows, compacting the retained span to the front when the buffer
    /// has run out at the end.
    private func ensureCapacity(layer: Int, extra: Int, like sample: MLXArray) -> Bool {
        let capacity = keys[layer]?.dim(2) ?? 0
        if ends[layer] + extra <= capacity { return false }

        let retained = retainedLength(layer: layer)
        let needed = retained + extra
        let blocks = (needed + Self.growthBlock - 1) / Self.growthBlock
        let target = blocks * Self.growthBlock

        let shape = [sample.dim(0), sample.dim(1), target, sample.dim(3)]
        let grownKeys = MLXArray.zeros(shape, dtype: sample.dtype)
        let grownValues = MLXArray.zeros(shape, dtype: sample.dtype)
        if let existingKeys = keys[layer], let existingValues = values[layer], retained > 0 {
            grownKeys[0..., 0..., 0 ..< retained, 0...] =
                existingKeys[0..., 0..., starts[layer] ..< ends[layer], 0...]
            grownValues[0..., 0..., 0 ..< retained, 0...] =
                existingValues[0..., 0..., starts[layer] ..< ends[layer], 0...]
        }
        keys[layer] = grownKeys
        values[layer] = grownValues
        starts[layer] = 0
        ends[layer] = retained
        return true
    }

    func update(layer: Int, keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        trimForAppend(layer: layer)
        let length = newKeys.dim(2)
        _ = ensureCapacity(layer: layer, extra: length, like: newKeys)

        let start = ends[layer]
        keys[layer]![0..., 0..., start ..< (start + length), 0...] = newKeys
        values[layer]![0..., 0..., start ..< (start + length), 0...] = newValues
        ends[layer] += length

        return (keys[layer]![0..., 0..., starts[layer] ..< ends[layer], 0...],
                values[layer]![0..., 0..., starts[layer] ..< ends[layer], 0...])
    }

    /// Advances the position count once every layer has been updated for this step.
    func advance(by count: Int) { offset += count }
}

/// The rotary embedding, with a release's declared scaling folded in.
///
/// A model that declares no scaling rotates at `base` and this is `MLXFast.RoPE` unchanged. A model
/// that declares one supplies its own per-pair periods instead, and multiplies by the attention
/// factor — which is legitimate to do to the input because the rotation is linear, so scaling the
/// input and scaling the sines are the same operation.
struct NFKLMRotary {
    let dimensions: Int
    let base: Float
    let periods: MLXArray?
    let attentionFactor: Float

    init(dimensions: Int, base: Float, scaling: NFKMLXRoPEScaling?) {
        self.dimensions = dimensions
        self.base = base
        if let scaling {
            periods = MLXArray(scaling.rotaryPeriods(dimensions: dimensions, base: base))
            attentionFactor = scaling.attentionFactor
        } else {
            periods = nil
            attentionFactor = 1
        }
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        let scaled = attentionFactor == 1 ? x : x * attentionFactor
        return MLXFast.RoPE(scaled, dimensions: dimensions, traditional: false,
                            // The periods replace the base entirely; passing both would be ambiguous.
                            base: periods == nil ? base : nil,
                            scale: 1, offset: offset, freqs: periods)
    }
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
    let rope: NFKLMRotary

    init(_ c: NFKMLXLanguageConfiguration) {
        heads = c.headCount
        keyValueHeads = c.keyValueHeadCount
        headDimensions = c.headDimensions
        scale = 1 / sqrt(Float(c.headDimensions))
        rope = NFKLMRotary(dimensions: c.headDimensions, base: c.ropeTheta,
                           scaling: c.ropeScaling)

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
        logits(fromHidden: hiddenStates(fromEmbeddings: model.embedTokens(tokens), cache: cache))
    }

    /// The token embedding on its own. The music AR loop feeds the model SUMMED code embeddings
    /// rather than token ids, which is why both ends of the ordinary forward are also exposed.
    func embed(_ tokens: MLXArray) -> MLXArray { model.embedTokens(tokens) }

    /// Runs the stack over already-embedded inputs and returns the post-norm hidden states
    /// `[batch, length, hidden]` — what conditions synthesis in a hidden-state-driven pipeline.
    func hiddenStates(fromEmbeddings embeddings: MLXArray,
                      cache: NFKMLXKeyValueCache? = nil) -> MLXArray {
        var hidden = embeddings
        let length = embeddings.shape[1]
        // A single token attends to everything cached, so it needs no mask; a prefill does. The mask
        // is built against the number of cached positions the pass will SEE, which a window makes
        // smaller than the absolute offset — a mask sized to the offset would be wider than the keys.
        let mask: MLXArray? = length > 1
            ? NFKMLXLanguageNet.causalMask(length, offset: cache?.maskCacheLength ?? 0)
            : nil
        for (index, layer) in model.layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache, layer: index)
        }
        cache?.advance(by: length)
        return model.norm(hidden)
    }

    /// The output projection on its own: post-norm hidden states → logits.
    func logits(fromHidden hidden: MLXArray) -> MLXArray {
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
