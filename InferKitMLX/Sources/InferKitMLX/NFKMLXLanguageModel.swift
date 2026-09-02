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

    /// How many experts each feed-forward holds, or 0 for the dense feed-forward.
    ///
    /// @discussion A mixture of experts replaces one wide feed-forward with many narrow ones and a
    /// router that sends each token to ``activeExpertCount`` of them, so a token's cost is the
    /// active experts' width while the model's capacity is all of them. Qwen3-MoE and Mixtral are
    /// this decoder with an expert feed-forward; nothing else about the block changes.
    public var expertCount: Int = 0
    /// How many experts the router selects per token.
    public var activeExpertCount: Int = 0
    /// Each expert's intermediate width, which is the `intermediate_size` of a Mixtral release and
    /// the `moe_intermediate_size` of a Qwen3-MoE release.
    public var expertIntermediateSize: Int = 0
    /// Whether the selected experts' routing weights are renormalized to sum to one
    /// (`norm_topk_prob`). Mixtral always does; Qwen3-MoE says so in its config.
    public var normalizesExpertWeights: Bool = true

    /// Whether the feed-forward is a mixture of experts.
    public var isMixtureOfExperts: Bool { expertCount > 0 }

    /// A small configuration that runs with random weights, for tests.
    public static let tiny = NFKMLXLanguageConfiguration(
        hiddenSize: 64, layerCount: 2, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
        intermediateSize: 128, vocabularySize: 512, ropeTheta: 10_000)

    /// The tiny configuration with a mixture-of-experts feed-forward, for tests.
    public static let tinyMixture: NFKMLXLanguageConfiguration = {
        var configuration = tiny
        configuration.expertCount = 8
        configuration.activeExpertCount = 2
        configuration.expertIntermediateSize = 32
        return configuration
    }()
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

    /// The affine quantization a cache stores its keys and values under.
    ///
    /// @discussion Quantizing the cache shrinks what stays resident — a token's key and value fall
    /// from a float per element to `bits` — which is what lets the context grow further before the
    /// cache, rather than the weights, becomes the ceiling. It changes storage, not positions, so the
    /// offset, window, and mask accounting are identical to an unquantized cache; a step dequantizes
    /// the retained span back to the model's precision on read. `groupSize` must divide the head
    /// dimension (64 or 128 divide by the default 64).
    public struct Quantization: Sendable, Equatable {
        public let bits: Int
        public let groupSize: Int
        public init(bits: Int = 8, groupSize: Int = 64) {
            self.bits = bits
            self.groupSize = groupSize
        }
    }

    /// The most positions the cache retains, or `nil` for an unbounded one.
    public let window: Int?

    /// The quantization the cache stores under, or `nil` to keep keys and values in full precision.
    public let quantization: Quantization?

    /// TOTAL positions ever appended, which is NOT the retained count once a window starts dropping
    /// them. This is the rotary offset a step needs: a token's angle depends on where it actually is
    /// in the sequence, not on where it sits in the buffer.
    public private(set) var offset = 0

    private var keys: [MLXArray?]
    private var values: [MLXArray?]
    /// The scale/bias buffers a quantized cache keeps beside the packed keys and values. Nil for an
    /// unquantized cache, and the bias buffers stay nil when the quantization mode carries none.
    private var keyScales: [MLXArray?]
    private var keyBiases: [MLXArray?]
    private var valueScales: [MLXArray?]
    private var valueBiases: [MLXArray?]
    /// The precision the model computes in, captured on the first append so the dequantized span is
    /// handed back at the type the fused attention expects.
    private var storedDType: DType = .float32
    /// The retained span of each layer's buffer, as `[start, end)`.
    private var starts: [Int]
    private var ends: [Int]

    /// Rows are added in blocks so that a steady decode compacts rarely rather than every step.
    private static let growthBlock = 256

    public init(layerCount: Int, window: Int? = nil, quantization: Quantization? = nil) {
        precondition(window.map { $0 > 1 } ?? true, "a window keeps more than one position")
        self.window = window
        self.quantization = quantization
        keys = Array(repeating: nil, count: layerCount)
        values = Array(repeating: nil, count: layerCount)
        keyScales = Array(repeating: nil, count: layerCount)
        keyBiases = Array(repeating: nil, count: layerCount)
        valueScales = Array(repeating: nil, count: layerCount)
        valueBiases = Array(repeating: nil, count: layerCount)
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
        guard let quantization else {
            let length = newKeys.dim(2)
            _ = ensureCapacity(layer: layer, extra: length, like: newKeys)

            let start = ends[layer]
            keys[layer]![0..., 0..., start ..< (start + length), 0...] = newKeys
            values[layer]![0..., 0..., start ..< (start + length), 0...] = newValues
            ends[layer] += length

            return (keys[layer]![0..., 0..., starts[layer] ..< ends[layer], 0...],
                    values[layer]![0..., 0..., starts[layer] ..< ends[layer], 0...])
        }
        return updateQuantized(layer: layer, quantization, keys: newKeys, values: newValues)
    }

    /// The quantized append: the incoming rows are packed once, stored in the block-growing buffers
    /// beside their scales and biases, and the retained span dequantized back for attention. The
    /// packing is per-row along the head dimension, so a row slots in exactly as a float row does.
    private func updateQuantized(layer: Int, _ quantization: Quantization,
                                 keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        storedDType = newKeys.dtype
        let (bits, groupSize) = (quantization.bits, quantization.groupSize)
        let (packedKeys, scaledKeys, biasedKeys) = quantized(newKeys, groupSize: groupSize, bits: bits)
        let (packedValues, scaledValues, biasedValues) = quantized(newValues, groupSize: groupSize, bits: bits)
        let length = newKeys.dim(2)
        let batch = packedKeys.dim(0), heads = packedKeys.dim(1)

        if ends[layer] + length > (keys[layer]?.dim(2) ?? 0) {
            let retained = retainedLength(layer: layer)
            let blocks = (retained + length + Self.growthBlock - 1) / Self.growthBlock
            let target = blocks * Self.growthBlock
            keys[layer] = grown(keys[layer], layer: layer, target: target, batch: batch, heads: heads,
                                lastDim: packedKeys.dim(3), dtype: .uint32)
            keyScales[layer] = grown(keyScales[layer], layer: layer, target: target, batch: batch, heads: heads,
                                     lastDim: scaledKeys.dim(3), dtype: scaledKeys.dtype)
            keyBiases[layer] = biasedKeys.map { grown(keyBiases[layer], layer: layer, target: target, batch: batch,
                                                      heads: heads, lastDim: $0.dim(3), dtype: $0.dtype) }
            values[layer] = grown(values[layer], layer: layer, target: target, batch: batch, heads: heads,
                                  lastDim: packedValues.dim(3), dtype: .uint32)
            valueScales[layer] = grown(valueScales[layer], layer: layer, target: target, batch: batch, heads: heads,
                                       lastDim: scaledValues.dim(3), dtype: scaledValues.dtype)
            valueBiases[layer] = biasedValues.map { grown(valueBiases[layer], layer: layer, target: target, batch: batch,
                                                          heads: heads, lastDim: $0.dim(3), dtype: $0.dtype) }
            starts[layer] = 0
            ends[layer] = retained
        }

        let rows = ends[layer] ..< (ends[layer] + length)
        keys[layer]![0..., 0..., rows, 0...] = packedKeys
        keyScales[layer]![0..., 0..., rows, 0...] = scaledKeys
        if let biasedKeys { keyBiases[layer]![0..., 0..., rows, 0...] = biasedKeys }
        values[layer]![0..., 0..., rows, 0...] = packedValues
        valueScales[layer]![0..., 0..., rows, 0...] = scaledValues
        if let biasedValues { valueBiases[layer]![0..., 0..., rows, 0...] = biasedValues }
        ends[layer] += length

        let span = starts[layer] ..< ends[layer]
        let outKeys = dequantized(keys[layer]![0..., 0..., span, 0...],
                                  scales: keyScales[layer]![0..., 0..., span, 0...],
                                  biases: keyBiases[layer].map { $0[0..., 0..., span, 0...] },
                                  groupSize: groupSize, bits: bits, dtype: storedDType)
        let outValues = dequantized(values[layer]![0..., 0..., span, 0...],
                                    scales: valueScales[layer]![0..., 0..., span, 0...],
                                    biases: valueBiases[layer].map { $0[0..., 0..., span, 0...] },
                                    groupSize: groupSize, bits: bits, dtype: storedDType)
        return (outKeys, outValues)
    }

    /// Allocates a block-sized buffer and copies the layer's retained span to its front, reading the
    /// current `[start, end)` before the caller resets them. One target row-count grows every buffer
    /// of a quantized layer together; only the trailing width and dtype differ.
    private func grown(_ existing: MLXArray?, layer: Int, target: Int, batch: Int, heads: Int,
                       lastDim: Int, dtype: DType) -> MLXArray {
        let fresh = MLXArray.zeros([batch, heads, target, lastDim], dtype: dtype)
        let retained = retainedLength(layer: layer)
        if let existing, retained > 0 {
            fresh[0..., 0..., 0 ..< retained, 0...] = existing[0..., 0..., starts[layer] ..< ends[layer], 0...]
        }
        return fresh
    }

    /// Advances the position count once every layer has been updated for this step.
    func advance(by count: Int) { offset += count }

    /// Discards the newest `count` positions, so the next update writes over them.
    ///
    /// @discussion This is what lets a cache be REUSED rather than rebuilt: speculative decoding
    /// rolls back the draft tokens the model rejected, and a prompt cache rolls back to the point
    /// where a new prompt diverges from the last one. The rollback moves the end cursor and the
    /// position count and copies nothing, so it is exact — the retained rows are the rows the
    /// original forward wrote. Returns false, and changes nothing, when a window has already dropped
    /// what the rollback would need to reach; the caller rebuilds from scratch then.
    @discardableResult
    public func rollback(by count: Int) -> Bool {
        precondition(count >= 0, "a rollback discards a non-negative number of positions")
        guard count > 0 else { return true }
        guard ends.indices.allSatisfy({ count <= retainedLength(layer: $0) }) else { return false }
        for layer in ends.indices {
            ends[layer] -= count
        }
        offset -= count
        return true
    }

    /// The element type the model computes in, for a persisted cache to restore.
    var storedDTypeForExport: DType { storedDType }

    /// The retained rows of every layer's buffers, keyed for a safetensors file. Used by
    /// ``NFKMLXPromptCache`` to persist a prefilled prompt.
    func exportedArrays() -> [String: MLXArray] {
        var arrays = [String: MLXArray]()
        for layer in keys.indices where retainedLength(layer: layer) > 0 {
            let span = starts[layer] ..< ends[layer]
            arrays["layer.\(layer).keys"] = keys[layer]![0..., 0..., span, 0...]
            arrays["layer.\(layer).values"] = values[layer]![0..., 0..., span, 0...]
            if let scales = keyScales[layer] { arrays["layer.\(layer).key_scales"] = scales[0..., 0..., span, 0...] }
            if let biases = keyBiases[layer] { arrays["layer.\(layer).key_biases"] = biases[0..., 0..., span, 0...] }
            if let scales = valueScales[layer] { arrays["layer.\(layer).value_scales"] = scales[0..., 0..., span, 0...] }
            if let biases = valueBiases[layer] { arrays["layer.\(layer).value_biases"] = biases[0..., 0..., span, 0...] }
        }
        return arrays
    }

    /// Restores what ``exportedArrays()`` wrote, at position count `offset`. The rows land at the
    /// front of fresh buffers, so a restored cache appends exactly as the original did.
    func restore(arrays: [String: MLXArray], offset: Int, storedDType: DType) {
        self.storedDType = storedDType
        for layer in keys.indices {
            guard let restoredKeys = arrays["layer.\(layer).keys"],
                  let restoredValues = arrays["layer.\(layer).values"] else { continue }
            keys[layer] = restoredKeys
            values[layer] = restoredValues
            keyScales[layer] = arrays["layer.\(layer).key_scales"]
            keyBiases[layer] = arrays["layer.\(layer).key_biases"]
            valueScales[layer] = arrays["layer.\(layer).value_scales"]
            valueBiases[layer] = arrays["layer.\(layer).value_biases"]
            starts[layer] = 0
            ends[layer] = restoredKeys.dim(2)
        }
        self.offset = offset
    }
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

/// The feed-forward a block holds, dense or routed.
class NFKLMMLP: Module {
    /// Builds the form the configuration asks for.
    static func make(_ c: NFKMLXLanguageConfiguration) -> NFKLMMLP {
        c.isMixtureOfExperts ? NFKLMMixtureFeedForward(c) : NFKLMFeedForward(c)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fatalError("a feed-forward subclass implements this") }
}

/// The SwiGLU feed-forward: a gate and an up projection multiplied, then projected back down.
final class NFKLMFeedForward: NFKLMMLP {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: NFKMLXLanguageConfiguration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

/// One weight per expert, applied to each token by the experts its router chose.
///
/// @discussion The experts are stacked into a single `[experts, out, in]` tensor rather than held
/// as separate layers, so a step runs ONE gathered matrix multiplication over the chosen experts
/// instead of one multiplication per expert per token. The released checkpoints store each expert
/// separately; the loader stacks them.
class NFKLMSwitchLinear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(experts: Int, inputSize: Int, outputSize: Int) {
        let scale = sqrt(1 / Float(inputSize))
        _weight.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [experts, outputSize, inputSize])
        super.init()
    }

    init(weight: MLXArray) {
        _weight.wrappedValue = weight
        super.init()
    }

    var expertCount: Int { weight.dim(0) }
    var outputSize: Int { weight.dim(1) }
    /// The input width, which a quantized subclass derives from its packing.
    var inputSize: Int { weight.dim(2) }

    /// Applies expert `experts[..., j]` to `x[..., j, :]`: `x` is `[..., 1, in]` broadcast against
    /// `experts` `[..., k]`, giving `[..., k, out]`.
    func callAsFunction(_ x: MLXArray, experts: MLXArray) -> MLXArray {
        gatherMM(x, weight.swappedAxes(-1, -2), rhsIndices: experts)
    }

    /// The same layer with its weights affine-packed.
    func quantized(groupSize: Int, bits: Int) -> NFKLMQuantizedSwitchLinear {
        NFKLMQuantizedSwitchLinear(weight: weight, groupSize: groupSize, bits: bits)
    }
}

/// A switch linear whose expert weights are MLX affine-quantized, applied through the gathered
/// quantized matrix multiplication.
final class NFKLMQuantizedSwitchLinear: NFKLMSwitchLinear, Quantized {
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray?
    let groupSize: Int
    let bits: Int
    var mode: QuantizationMode { .affine }

    init(weight: MLXArray, groupSize: Int, bits: Int) {
        self.groupSize = groupSize
        self.bits = bits
        let (packed, scales, biases) = MLX.quantized(weight, groupSize: groupSize, bits: bits)
        _scales.wrappedValue = scales
        _biases.wrappedValue = biases
        super.init(weight: packed)
        freeze()
    }

    override var inputSize: Int { weight.dim(2) * 32 / bits }

    override func callAsFunction(_ x: MLXArray, experts: MLXArray) -> MLXArray {
        gatherQuantizedMM(x, weight, scales: scales, biases: biases, rhsIndices: experts,
                          transpose: true, groupSize: groupSize, bits: bits)
    }
}

/// The experts' SwiGLU, run for each token through the experts chosen for it.
final class NFKLMSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gate: NFKLMSwitchLinear
    @ModuleInfo(key: "up_proj") var up: NFKLMSwitchLinear
    @ModuleInfo(key: "down_proj") var down: NFKLMSwitchLinear

    init(_ c: NFKMLXLanguageConfiguration) {
        _gate.wrappedValue = NFKLMSwitchLinear(experts: c.expertCount, inputSize: c.hiddenSize,
                                               outputSize: c.expertIntermediateSize)
        _up.wrappedValue = NFKLMSwitchLinear(experts: c.expertCount, inputSize: c.hiddenSize,
                                             outputSize: c.expertIntermediateSize)
        _down.wrappedValue = NFKLMSwitchLinear(experts: c.expertCount, inputSize: c.expertIntermediateSize,
                                               outputSize: c.hiddenSize)
        super.init()
    }

    /// `x` `[batch, length, hidden]` with `experts` `[batch, length, k]` → `[batch, length, k, hidden]`.
    func callAsFunction(_ x: MLXArray, experts: MLXArray) -> MLXArray {
        let expanded = x.expandedDimensions(axes: [-2, -3])
        let hidden = silu(gate(expanded, experts: experts)) * up(expanded, experts: experts)
        return down(hidden, experts: experts).squeezed(axis: -2)
    }
}

/// The mixture-of-experts feed-forward: a router scores every expert, the top `k` run, and their
/// outputs combine by routing weight.
///
/// @discussion The router's softmax is taken over ALL experts and the selected weights are then
/// renormalized when the configuration says so — which is the same as Mixtral's softmax over the
/// selected logits alone, so one implementation serves both families.
final class NFKLMMixtureFeedForward: NFKLMMLP {
    @ModuleInfo(key: "gate") var router: Linear
    @ModuleInfo(key: "experts") var experts: NFKLMSwitchGLU

    let activeExpertCount: Int
    let normalizesWeights: Bool

    init(_ c: NFKMLXLanguageConfiguration) {
        precondition(c.activeExpertCount > 0 && c.activeExpertCount <= c.expertCount,
                     "a mixture routes each token to between one and every expert")
        activeExpertCount = c.activeExpertCount
        normalizesWeights = c.normalizesExpertWeights
        _router.wrappedValue = Linear(c.hiddenSize, c.expertCount, bias: false)
        _experts.wrappedValue = NFKLMSwitchGLU(c)
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scores = softmax(router(x), axis: -1, precise: true)
        let chosen = argPartition(-scores, kth: activeExpertCount - 1, axis: -1)[.ellipsis, 0 ..< activeExpertCount]
        var weights = takeAlong(scores, chosen, axis: -1)
        if normalizesWeights {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        let outputs = experts(x, experts: chosen)
        return (outputs * weights.expandedDimensions(axis: -1).asType(outputs.dtype)).sum(axis: -2)
    }
}

/// One transformer block: pre-normalized attention and feed-forward, each added back.
final class NFKLMBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKLMAttention
    @ModuleInfo(key: "mlp") var feedForward: NFKLMMLP
    @ModuleInfo(key: "input_layernorm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var feedForwardNorm: RMSNorm

    init(_ c: NFKMLXLanguageConfiguration) {
        _attention.wrappedValue = NFKLMAttention(c)
        _feedForward.wrappedValue = NFKLMMLP.make(c)
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

    /// The state entering the stack and the state each layer produces, with the final norm applied
    /// to the last — the reference's `output_hidden_states` convention, so a divergence is located
    /// to a layer rather than guessed at from the logits.
    func layerStates(_ tokens: MLXArray) -> [MLXArray] {
        var hidden = model.embedTokens(tokens)
        var states = [hidden]
        let mask = NFKMLXLanguageNet.causalMask(tokens.shape[1], offset: 0)
        for (index, layer) in model.layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: nil, layer: index)
            states.append(hidden)
        }
        states[states.count - 1] = model.norm(hidden)
        return states
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
