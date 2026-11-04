//
//  NFKMLXHybridLanguageModel.swift
//  InferKitMLX
//
//  The hybrid decoder Qwen3.5, Qwen3.6, and Qwen3.8 are built from (`Qwen3_5ForConditionalGeneration`).
//  It is not the dense stack `NFKMLXLanguageNet` implements: three quarters of its layers replace
//  attention with a **gated delta-rule recurrence** — linear in sequence length, with a fixed-size
//  state instead of a growing key-value cache — and the remaining quarter is full attention whose
//  output is gated.
//
//  At reference parity on the released Qwen3.5-4B, the smallest release of the family and the only
//  size this machine holds: logit cosine 0.9999999999962, with every one of the 33 hidden states
//  matched layer by layer. The larger sizes are covered structurally instead — every parameter this
//  module declares is checked against the released checkpoint's own safetensors header, name by name
//  and shape by shape — so their numerics rest on the 4B measurement rather than on a run of their own.
//
//  Scope: the language model. A release in this family also carries a vision tower (`model.visual`)
//  and a multi-token-prediction head (`mtp`), which are separate features rather than parts of the
//  decoder.
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

/// Which kind of layer sits at a given depth.
public enum NFKMLXHybridLayerKind: String, Sendable {
    /// A gated delta-rule recurrence, linear in sequence length.
    case linearAttention = "linear_attention"
    /// Ordinary attention over the whole prefix, with a gated output.
    case fullAttention = "full_attention"
}

/// Which positions a full-attention layer reads. The linear-attention layers read the prefix in either
/// mode, since their convolution and recurrence run forward in sequence order. Introduced in InferKit 0.4.0.
public enum NFKMLXHybridAttention: Sendable {
    /// Each position reads itself and the positions before it, as the language model generates.
    case causal
    /// Each position reads the whole sequence: the noncausal full attention a decision checkpoint
    /// (`NFKMLXPplxDecider`) was trained with.
    case bidirectional
}

/// The attention arithmetic a full-attention layer follows at half precision. Float32 takes the fused kernel
/// either way. Introduced in InferKit 0.4.0.
public enum NFKMLXHybridHalfPrecisionAttention: Sendable {
    /// transformers' eager attention, which the language model's reference runs.
    case eager
    /// torch's MATH `scaled_dot_product_attention`, the definition of the SDPA a checkpoint loaded with
    /// `attn_implementation="sdpa"` asks for (pplx-decider).
    case sdpaMath
}

/// The geometry of a hybrid decoder.
public struct NFKMLXHybridConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    public var rmsEpsilon: Float

    // Full-attention layers.
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int
    public var ropeTheta: Float
    /// The fraction of each head's channels the rotary embedding turns; the rest pass through.
    public var partialRotaryFactor: Float
    /// Whether the query projection also emits a gate for the attention output.
    public var gatesAttentionOutput: Bool

    // Linear-attention layers.
    public var linearKeyHeadCount: Int
    public var linearKeyHeadDimensions: Int
    public var linearValueHeadCount: Int
    public var linearValueHeadDimensions: Int
    /// The depthwise convolution's kernel over the projected q, k, and v.
    public var linearConvolutionKernel: Int

    /// Whether the output projection reuses the embedding matrix. 4B ties; 27B does not.
    public var tiesWordEmbeddings: Bool

    /// One entry per layer. The releases place a full-attention layer every fourth.
    public var layerTypes: [NFKMLXHybridLayerKind]

    /// Whether the delta rule widens its queries and keys to float32 before normalizing them, as
    /// transformers' chunked rule does from 5.17. False normalizes them at the input's type and then
    /// widens, as 5.16 and earlier do. The two differ only below float32. Introduced in InferKit 0.4.0.
    public var normalizesKeysInFloat32: Bool = true

    /// The full-attention arithmetic at half precision. Introduced in InferKit 0.4.0.
    public var halfPrecisionAttention: NFKMLXHybridHalfPrecisionAttention = .eager

    public init(hiddenSize: Int = 5120, layerCount: Int = 64, intermediateSize: Int = 17408,
                vocabularySize: Int = 248_320, rmsEpsilon: Float = 1e-6,
                headCount: Int = 24, keyValueHeadCount: Int = 4, headDimensions: Int = 256,
                ropeTheta: Float = 10_000_000, partialRotaryFactor: Float = 0.25,
                gatesAttentionOutput: Bool = true,
                linearKeyHeadCount: Int = 16, linearKeyHeadDimensions: Int = 128,
                linearValueHeadCount: Int = 48, linearValueHeadDimensions: Int = 128,
                linearConvolutionKernel: Int = 4, fullAttentionInterval: Int = 4,
                tiesWordEmbeddings: Bool = false,
                layerTypes: [NFKMLXHybridLayerKind]? = nil) {
        self.tiesWordEmbeddings = tiesWordEmbeddings
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.rmsEpsilon = rmsEpsilon
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.gatesAttentionOutput = gatesAttentionOutput
        self.linearKeyHeadCount = linearKeyHeadCount
        self.linearKeyHeadDimensions = linearKeyHeadDimensions
        self.linearValueHeadCount = linearValueHeadCount
        self.linearValueHeadDimensions = linearValueHeadDimensions
        self.linearConvolutionKernel = linearConvolutionKernel
        // The released pattern: every `fullAttentionInterval`-th layer attends fully, the rest recur.
        self.layerTypes = layerTypes ?? (0 ..< layerCount).map {
            ($0 + 1) % fullAttentionInterval == 0 ? .fullAttention : .linearAttention
        }
    }

    /// The width the linear branch's fused q/k/v projection emits.
    var linearQKVWidth: Int {
        2 * linearKeyHeadCount * linearKeyHeadDimensions
            + linearValueHeadCount * linearValueHeadDimensions
    }

    /// The width of the linear branch's value stream, which its gate and output projection match.
    var linearValueWidth: Int { linearValueHeadCount * linearValueHeadDimensions }

    /// How many of a head's channels the rotary embedding turns.
    var rotaryDimensions: Int { Int(Float(headDimensions) * partialRotaryFactor) }

    /// The released `Qwen/Qwen3.8-27B` language model.
    public static let qwen3_8_27B = NFKMLXHybridConfiguration()
}

/// The hybrid family's normalization: `x · (1 + w)`, with the weight stored as an offset from one.
///
/// Qwen3.5 / 3.6 / 3.8 use this; the DENSE Qwen3 stack does not, and neither does Gemma 4 — both of
/// those scale by the weight directly. The two conventions are indistinguishable by shape and differ
/// in every number, so each model's own source decides it. Note the gated norm inside the recurrence
/// (`linear_attn.norm`) is the PLAIN kind even here.
final class NFKHybridNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        epsilon = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        NFKReferenceRounding.gemmaNorm(x, weight: weight, eps: epsilon)
    }
}

/// Full attention with a gated output.
///
/// The query projection emits twice the width a plain one would: the first half is the queries, the
/// second is a gate applied to the attention result. Only the leading `rotaryDimensions` channels of
/// each head are rotated, which is what `partial_rotary_factor` means.
final class NFKHybridAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: NFKHybridNorm
    @ModuleInfo(key: "k_norm") var keyNorm: NFKHybridNorm

    let configuration: NFKMLXHybridConfiguration
    let rope: RoPE

    init(_ c: NFKMLXHybridConfiguration) {
        configuration = c
        rope = RoPE(dimensions: max(c.rotaryDimensions, 1), traditional: false, base: c.ropeTheta)
        let queries = c.headCount * c.headDimensions
        _queryProjection.wrappedValue = Linear(c.hiddenSize,
                                               c.gatesAttentionOutput ? queries * 2 : queries,
                                               bias: false)
        _keyProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions,
                                             bias: false)
        _valueProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions,
                                               bias: false)
        _outputProjection.wrappedValue = Linear(queries, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = NFKHybridNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        _keyNorm.wrappedValue = NFKHybridNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        callAsFunction(x, mask: mask, cache: nil, layer: 0)
    }

    /// The layer over `x` after the positions `cache` holds, its keys and values appended to the cache's
    /// `layer` slot; without a cache, the positions start at 0.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: NFKMLXKeyValueCache?, layer: Int) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.shape[0], x.shape[1])
        let width = c.headCount * c.headDimensions
        let offset = cache?.offset ?? 0

        // The projection is viewed as [.., heads, 2 · headDim] and split on the LAST axis, so each
        // head's queries and gate are interleaved. Splitting the flat width into two halves takes the
        // wrong channels entirely.
        var queries: MLXArray
        var gate: MLXArray?
        if c.gatesAttentionOutput {
            let paired = queryProjection(x)
                .reshaped([batch, length, c.headCount, c.headDimensions * 2])
            queries = paired[.ellipsis, 0 ..< c.headDimensions]
            gate = paired[.ellipsis, c.headDimensions...].reshaped([batch, length, width])
        } else {
            queries = queryProjection(x).reshaped([batch, length, c.headCount, c.headDimensions])
            gate = nil
        }

        queries = queryNorm(queries)
        var keys = keyNorm(keyProjection(x)
            .reshaped([batch, length, c.keyValueHeadCount, c.headDimensions]))
        var values = valueProjection(x)
            .reshaped([batch, length, c.keyValueHeadCount, c.headDimensions])

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Partial rotary: only the leading channels turn, the rest are carried through untouched.
        let turned = c.rotaryDimensions
        if turned > 0 && NFKReferenceRounding.isReduced(queries) {
            queries = NFKReferenceRounding.rotary(queries, dimensions: turned, base: c.ropeTheta, offset: offset)
            keys = NFKReferenceRounding.rotary(keys, dimensions: turned, base: c.ropeTheta, offset: offset)
        } else if turned > 0 && turned < c.headDimensions {
            queries = concatenated([rope(queries[.ellipsis, 0 ..< turned], offset: offset),
                                    queries[.ellipsis, turned...]], axis: -1)
            keys = concatenated([rope(keys[.ellipsis, 0 ..< turned], offset: offset),
                                 keys[.ellipsis, turned...]], axis: -1)
        } else if turned > 0 {
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }
        if let cache {
            (keys, values) = cache.update(layer: layer, keys: keys, values: values)
        }

        // A `.checkpoint`-precision load makes this a bf16 module, and the fused attention refuses
        // a float32 mask that does not promote to its own type — invisible at float32.
        let scale = 1 / sqrt(Float(c.headDimensions))
        let attended: MLXArray
        if cache != nil && NFKMLXHalfPrecision.arithmetic == .fast {
            // Generation through the cache takes MLX's fused attention, which reads the grouped keys and values
            // once; the eager form spreads them to every query head on every step.
            attended = MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale,
                                                         mask: mask.map { .array($0.asType(queries.dtype)) } ?? .none)
        } else if c.halfPrecisionAttention == .sdpaMath && NFKReferenceRounding.isReduced(queries) {
            // transformers repeats each key and value head for its group of query heads before the SDPA call.
            let group = c.headCount / c.keyValueHeadCount
            attended = NFKReferenceRounding.mathAttention(
                queries: queries, keys: repeated(keys, count: group, axis: 1), values: repeated(values, count: group, axis: 1),
                scale: scale, mask: mask.map { $0.asType(queries.dtype) })
        } else {
            attended = NFKReferenceRounding.attention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask.map { $0.asType(queries.dtype) })
        }
        var result = attended.transposed(0, 2, 1, 3).reshaped([batch, length, width])
        if let gate {
            // The config field is named `output_gate_type: swish`, and the implementation applies a
            // plain sigmoid. The implementation is what the weights were trained against.
            result = result * NFKReferenceRounding.sigmoid(gate)
        }
        return outputProjection(result)
    }
}

/// The gated delta-rule recurrence that replaces attention in three quarters of the layers.
///
/// One fused projection produces queries, keys, and values; a depthwise convolution mixes each over a
/// short window; a per-head decay and write strength drive a fixed-size state, which the queries then
/// read. The state is `[value heads, key dimensions, value dimensions]` and does not grow with the
/// sequence, which is what makes the layer linear.
final class NFKHybridLinearAttention: Module {
    @ModuleInfo(key: "in_proj_qkv") var qkvProjection: Linear
    @ModuleInfo(key: "in_proj_z") var gateProjection: Linear
    @ModuleInfo(key: "in_proj_a") var decayProjection: Linear
    @ModuleInfo(key: "in_proj_b") var writeProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "conv1d") var convolution: Conv1d

    @ParameterInfo(key: "A_log") var decayLog: MLXArray
    @ParameterInfo(key: "dt_bias") var stepBias: MLXArray

    let configuration: NFKMLXHybridConfiguration

    init(_ c: NFKMLXHybridConfiguration) {
        configuration = c
        _qkvProjection.wrappedValue = Linear(c.hiddenSize, c.linearQKVWidth, bias: false)
        _gateProjection.wrappedValue = Linear(c.hiddenSize, c.linearValueWidth, bias: false)
        _decayProjection.wrappedValue = Linear(c.hiddenSize, c.linearValueHeadCount, bias: false)
        _writeProjection.wrappedValue = Linear(c.hiddenSize, c.linearValueHeadCount, bias: false)
        _outputProjection.wrappedValue = Linear(c.linearValueWidth, c.hiddenSize, bias: false)
        // The gated normalization runs over one value head's channels.
        _norm.wrappedValue = RMSNorm(dimensions: c.linearValueHeadDimensions, eps: c.rmsEpsilon)
        // Depthwise over the fused projection: one filter per channel.
        _convolution.wrappedValue = Conv1d(inputChannels: c.linearQKVWidth,
                                           outputChannels: c.linearQKVWidth,
                                           kernelSize: c.linearConvolutionKernel,
                                           groups: c.linearQKVWidth, bias: false)
        _decayLog.wrappedValue = MLXArray.zeros([c.linearValueHeadCount])
        _stepBias.wrappedValue = MLXArray.zeros([c.linearValueHeadCount])
        super.init()
    }

    /// The fused projection through the causal short convolution and its SiLU, `[batch, length, width]`.
    func convolved(_ projected: MLXArray) -> MLXArray {
        // Pad on the left so a position sees only itself and its history.
        let padded = padded(projected, widths: [IntOrPair((0, 0)),
                                                IntOrPair((configuration.linearConvolutionKernel - 1, 0)),
                                                IntOrPair((0, 0))])
        return NFKReferenceRounding.silu(convolution(padded))
    }

    /// The delta rule reads and writes a unit-norm key space: `x · rsqrt(Σx² + ε)` in the input's type,
    /// as the reference normalizes before widening, with torch's half-precision `rsqrt`.
    /// `x` unit-normalized and widened to float32: normalized in float32 when `widenedFirst`, else at its
    /// own type and then widened.
    static func unitNorm(_ x: MLXArray, widenedFirst: Bool) -> MLXArray {
        widenedFirst ? unitNorm(x.asType(.float32)) : unitNorm(x).asType(.float32)
    }

    static func unitNorm(_ x: MLXArray) -> MLXArray {
        guard NFKReferenceRounding.isReduced(x) else {
            return x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6)
        }
        // The products round to the input's type and sum in float32, as torch's reduction does.
        let squares = (x * x).asType(.float32).sum(axis: -1, keepDims: true)
        return x * NFKReferenceRounding.rsqrt((squares.asType(x.dtype).asType(.float32) + Float(1e-6)).asType(x.dtype))
    }

    /// The gated delta-rule recurrence over float32 `[batch, length, value heads, width]` queries and keys
    /// (already unit-normed and repeated to the value heads), values, per-step decay `exp(g)`, and write
    /// strength, returning the float32 read `[batch, length, value heads, value width]`.
    func recurrence(queries: MLXArray, keys: MLXArray, values: MLXArray, decay: MLXArray,
                    write: MLXArray) -> MLXArray {
        recurrence(queries: queries, keys: keys, values: values, decay: decay, write: write, initialState: nil).read
    }

    /// The recurrence from `initialState` `[batch, value heads, key width, value width]` (zeros where nil),
    /// with the state after the last position.
    func recurrence(queries: MLXArray, keys: MLXArray, values: MLXArray, decay: MLXArray, write: MLXArray,
                    initialState: MLXArray?) -> (read: MLXArray, state: MLXArray) {
        let c = configuration
        let (batch, length) = (queries.dim(0), queries.dim(1))
        var state = initialState ?? MLXArray.zeros([batch, c.linearValueHeadCount,
                                                    c.linearKeyHeadDimensions, c.linearValueHeadDimensions])
        var outputs = [MLXArray]()
        outputs.reserveCapacity(length)
        // The reference scales the queries by the key width before the recurrence, once.
        let scale = 1 / sqrt(Float(c.linearKeyHeadDimensions))
        for step in 0 ..< length {
            let q = queries[0..., step] * scale
            let k = keys[0..., step]
            let v = values[0..., step]                                     // [batch, value heads, value dims]
            let g = decay[0..., step].reshaped([batch, c.linearValueHeadCount, 1, 1])
            let b = write[0..., step].reshaped([batch, c.linearValueHeadCount, 1])

            // Decay FIRST, then read what the decayed state holds for this key, then write the
            // correction toward v. Reading before decaying is a different recurrence entirely.
            state = state * g
            let held = (state * k.expandedDimensions(axis: -1)).sum(axis: 2)   // [batch, heads, value dims]
            let correction = (v - held) * b
            state = state + k.expandedDimensions(axis: -1) * correction.expandedDimensions(axis: 2)
            outputs.append((state * q.expandedDimensions(axis: -1)).sum(axis: 2))
        }
        return (stacked(outputs, axis: 1), state)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, cache: nil)
    }

    /// The layer over `x` after the positions `cache` holds: the convolution reads the cached inputs before
    /// `x`, and the recurrence starts from the cached state. Both are replaced with the state after `x`. A
    /// single position runs the recurrence as transformers' cached decode does; more run its chunked rule
    /// from the cached state.
    func callAsFunction(_ x: MLXArray, cache: NFKMLXHybridLinearCache?) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.shape[0], x.shape[1])
        let projected = qkvProjection(x)
        let mixed: MLXArray
        if let cache {
            let history = cache.convolutionInputs ?? MLXArray.zeros([batch, c.linearConvolutionKernel - 1, projected.dim(2)],
                                                                    dtype: projected.dtype)
            let joined = concatenated([history, projected], axis: 1)
            mixed = NFKReferenceRounding.silu(convolution(joined))
            cache.convolutionInputs = joined[0..., (joined.dim(1) - (c.linearConvolutionKernel - 1))...]
        } else {
            mixed = convolved(projected)
        }

        let keyWidth = c.linearKeyHeadCount * c.linearKeyHeadDimensions
        let queries = mixed[0..., 0..., 0 ..< keyWidth]
            .reshaped([batch, length, c.linearKeyHeadCount, c.linearKeyHeadDimensions])
        let keys = mixed[0..., 0..., keyWidth ..< (2 * keyWidth)]
            .reshaped([batch, length, c.linearKeyHeadCount, c.linearKeyHeadDimensions])
        let values = mixed[0..., 0..., (2 * keyWidth)...]
            .reshaped([batch, length, c.linearValueHeadCount, c.linearValueHeadDimensions])

        // Per-head decay and write strength. `A_log` is stored as a log so the rate stays positive;
        // the reference keeps `g` as a LOG rate, formed in float32, and exponentiates it inside the
        // recurrence. The recurrence runs in float32 and its output rounds once to the input's type.
        let logDecay = -NFKReferenceRounding.exponential(decayLog.asType(.float32))
            * NFKReferenceRounding.softplus(decayProjection(x).asType(.float32) + stepBias.asType(.float32))
        let write = NFKReferenceRounding.sigmoid(writeProjection(x)).asType(.float32)

        // Key heads are shared across a group of value heads, as grouped-query attention shares them.
        // The reference arithmetic takes transformers 5.17's chunked rule where the release normalizes as it does.
        let group = c.linearValueHeadCount / c.linearKeyHeadCount
        let wide = c.normalizesKeysInFloat32
        let read: MLXArray
        if wide && NFKReferenceRounding.computesReferenceScans && (cache?.state == nil || length > 1) {
            let scanned = NFKReferenceRounding.torchChunkGatedDeltaRule(
                queries: repeated(queries, count: group, axis: 2), keys: repeated(keys, count: group, axis: 2),
                values: values, logDecay: logDecay, beta: write, initialState: cache?.state)
            read = scanned.read.asType(x.dtype)
            cache?.state = scanned.state
        } else {
            let normedQueries = Self.unitNorm(repeated(queries, count: group, axis: 2), widenedFirst: wide)
            let normedKeys = Self.unitNorm(repeated(keys, count: group, axis: 2), widenedFirst: wide)
            let decay = exp(logDecay)
            if NFKMLXGatedDeltaKernel.runs(keyWidth: c.linearKeyHeadDimensions), let cache {
                let scanned = NFKMLXGatedDeltaKernel.scan(queries: normedQueries, keys: normedKeys,
                                                          values: values.asType(.float32), decay: decay, write: write,
                                                          initialState: cache.state)
                read = scanned.read.asType(x.dtype)
                cache.state = scanned.state
            } else if NFKMLXGatedDeltaKernel.runs(keyWidth: c.linearKeyHeadDimensions) {
                read = NFKMLXGatedDeltaKernel.read(queries: normedQueries, keys: normedKeys,
                                                   values: values.asType(.float32), decay: decay,
                                                   write: write).asType(x.dtype)
            } else {
                let scanned = recurrence(queries: normedQueries, keys: normedKeys, values: values.asType(.float32),
                                         decay: decay, write: write, initialState: cache?.state)
                read = scanned.read.asType(x.dtype)
                cache?.state = scanned.state
            }
        }

        let gate = gateProjection(x).reshaped([batch, length, c.linearValueHeadCount,
                                               c.linearValueHeadDimensions])
        // The normalized read times `silu(gate)`, formed in float32 and rounded once.
        let wideGate = gate.asType(.float32)
        let gated = (norm(read).asType(.float32) * (wideGate * sigmoid(wideGate))).asType(x.dtype)
        return outputProjection(gated.reshaped([batch, length, c.linearValueWidth]))
    }
}

/// One layer of the hybrid stack. Which branch it carries is fixed by its depth.
final class NFKHybridBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKHybridAttention?
    @ModuleInfo(key: "linear_attn") var linearAttention: NFKHybridLinearAttention?
    @ModuleInfo(key: "mlp") var feedForward: NFKLMFeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: NFKHybridNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: NFKHybridNorm

    let kind: NFKMLXHybridLayerKind

    init(_ c: NFKMLXHybridConfiguration, kind: NFKMLXHybridLayerKind) {
        self.kind = kind
        // Exactly one branch exists per layer, which is what the checkpoint carries: a linear layer
        // has no `self_attn.*` and a full layer has no `linear_attn.*`.
        _attention.wrappedValue = kind == .fullAttention ? NFKHybridAttention(c) : nil
        _linearAttention.wrappedValue = kind == .linearAttention ? NFKHybridLinearAttention(c) : nil
        _feedForward.wrappedValue = NFKLMFeedForward(NFKMLXLanguageConfiguration(
            hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize))
        _inputNorm.wrappedValue = NFKHybridNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postNorm.wrappedValue = NFKHybridNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        callAsFunction(x, mask: mask, cache: nil, layer: 0)
    }

    /// The compiled single-position step of a linear-attention layer, built at its first use.
    private let compiledStep = NFKHybridCompiledStep()

    /// The layer over `x` after the positions `cache` holds, reading and replacing this layer's slot. Where
    /// `compiles` is set, a linear-attention layer runs one position whose cache holds a state through its
    /// compiled step, which fuses the step's elementwise operations; the weights must stay in place, so a
    /// streamed layer does not compile.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: NFKMLXHybridCache?, layer: Int,
                        compiles: Bool = false) -> MLXArray {
        if compiles, x.dim(1) == 1, linearAttention != nil, let slot = cache?.linear[layer],
           let convolutionInputs = slot.convolutionInputs, let state = slot.state {
            let step = compiledStep.step ?? {
                let made = compile(inputs: [self]) { [unowned self] arrays in
                    let scratch = NFKMLXHybridLinearCache()
                    scratch.convolutionInputs = arrays[1]
                    scratch.state = arrays[2]
                    let output = self.linearLayer(arrays[0], cache: scratch)
                    return [output, scratch.convolutionInputs ?? arrays[1], scratch.state ?? arrays[2]]
                }
                compiledStep.step = made
                return made
            }()
            let outputs = step([x, convolutionInputs, state])
            slot.convolutionInputs = outputs[1]
            slot.state = outputs[2]
            return outputs[0]
        }
        let normalized = inputNorm(x)
        let mixed: MLXArray
        if let attention {
            mixed = attention(normalized, mask: mask, cache: cache?.attention, layer: layer)
        } else if let linearAttention {
            mixed = linearAttention(normalized, cache: cache?.linear[layer])
        } else {
            mixed = normalized
        }
        let residual = x + mixed
        return residual + feedForward(postNorm(residual))
    }

    /// A linear-attention layer over `x` through `cache`'s slot.
    private func linearLayer(_ x: MLXArray, cache: NFKMLXHybridLinearCache?) -> MLXArray {
        let residual = x + linearAttention!(inputNorm(x), cache: cache)
        return residual + feedForward(postNorm(residual))
    }
}

/// A layer's compiled step, held outside the module's reflected properties.
final class NFKHybridCompiledStep {
    var step: (@Sendable ([MLXArray]) -> [MLXArray])?
}

/// The hybrid decoder, under the `model.language_model.` prefix the releases use.
final class NFKHybridCore: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKHybridBlock]
    @ModuleInfo(key: "norm") var norm: NFKHybridNorm

    init(_ c: NFKMLXHybridConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.layerCount).map { NFKHybridBlock(c, kind: c.layerTypes[$0]) }
        _norm.wrappedValue = NFKHybridNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }
}

/// A hybrid causal language model.
///
/// See the file comment: this is built and structurally verified against the released checkpoint, and
/// its numerics are unmeasured because the smallest release does not fit on the machine that built it.
public final class NFKMLXHybridLanguageNet: Module {
    @ModuleInfo(key: "model") var model: NFKHybridCore
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// The geometry the decoder was built at. Introduced in InferKit 0.4.0.
    public let configuration: NFKMLXHybridConfiguration

    /// The reader of the layers a streamed load leaves in the release, read in their turn on every pass;
    /// nil where every layer is held. Introduced in InferKit 0.4.0.
    public internal(set) var layerStream: NFKMLXLayerStream?

    /// Throws the first failed read of a streamed layer. Every pass after a failed read skips every
    /// layer, so a caller holding a streamed decoder checks after each pass. Nothing is thrown where
    /// every layer is held. Introduced in InferKit 0.4.0.
    public func verifyStream() throws {
        if let failure = layerStream?.readFailure {
            throw failure
        }
    }

    init(_ c: NFKMLXHybridConfiguration) {
        configuration = c
        _model.wrappedValue = NFKHybridCore(c)
        // A tied release ships no `lm_head.weight`: the embedding matrix is the output projection.
        _lmHead.wrappedValue = c.tiesWordEmbeddings
            ? nil : Linear(c.hiddenSize, c.vocabularySize, bias: false)
        super.init()
    }

    /// The logits `[batch, length, vocabularySize]` for token ids `[batch, length]`, every position
    /// attending causally over the whole sequence. The linear-attention layers carry a recurrent state
    /// rather than a key-value cache, so a generation step runs the full sequence again. Introduced in
    /// InferKit 0.4.0.
    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var trace = [MLXArray]()
        return forward(tokens, trace: &trace)
    }

    /// The logits `[batch, vocabularySize]` of the last of `tokens` `[batch, length]`, which follow the
    /// positions `cache` holds; the cache takes the new positions. Introduced in InferKit 0.4.0.
    func lastLogits(_ tokens: MLXArray, cache: NFKMLXHybridCache) -> MLXArray {
        let length = tokens.shape[1]
        var hidden = model.embedTokens(tokens)
        let mask: MLXArray? = length > 1
            ? NFKMLXLanguageNet.causalMask(length, offset: cache.maskCacheLength) : nil
        for (index, layer) in model.layers.enumerated() {
            guard let next = NFKMLXLayerStream.run(layerStream, layer: layer, index: index,
                                                   { layer(hidden, mask: mask, cache: cache, layer: index,
                                                           compiles: layerStream == nil) },
                                                   keeping: { [$0] }) else {
                continue
            }
            hidden = next
        }
        cache.attention.advance(by: length)
        let last = model.norm(hidden[0..., (length - 1)...])
        return (lmHead?(last) ?? model.embedTokens.asLinear(last))[0..., 0]
    }

    /// Fills `cache` with `prompt` and returns the logits `[vocabularySize]` that follow its last token.
    ///
    /// @discussion The prompt runs in passes of at most `limit` tokens, halved while a pass's attention scores,
    /// four bytes for each head, query, and key it sees, would pass `budget` bytes. A pass shorter than the
    /// rest of the prompt keeps a whole number of the delta rule's 64-position chunks, so the chunked
    /// recurrence splits where a single pass would. A failed streamed read or MLX error ends the passes.
    func prefill(_ prompt: [Int], cache: NFKMLXHybridCache, limit: Int = NFKMLXLanguageNet.defaultPrefillChunk,
                 budget: Int = NFKMLXLanguageNet.prefillScoreBudget) -> MLXArray {
        var start = 0
        var logits: MLXArray
        repeat {
            let length = prefillPassLength(cached: cache.maskCacheLength, remaining: prompt.count - start,
                                           limit: limit, budget: budget)
            let slice = prompt[start ..< start + length].map { Int32($0) }
            logits = lastLogits(MLXArray(slice).reshaped([1, length]), cache: cache)
            start += length
            if start < prompt.count {
                eval(logits)                                    // free a pass's activations before the next
            }
        } while start < prompt.count && layerStream?.readFailure == nil && !NFKMLXErrorScope.hasFailed
        return logits[0]
    }

    /// Generation through a cache of this decoder: a fresh cache under the window and quantization a run sets,
    /// filled by ``prefill(_:cache:limit:budget:)``, then one token a pass.
    var cachedGeneration: NFKDecoderCachedGeneration {
        NFKDecoderCachedGeneration { [self] prompt, window, quantization in
            let cache = NFKMLXHybridCache(configuration: configuration, window: window, quantization: quantization)
            let logits = prefill(prompt, cache: cache)
            return (logits, { token in self.lastLogits(token.reshaped([1, 1]), cache: cache)[0] })
        }
    }

    /// The tokens the next prefill pass runs: the smaller of `remaining` and `limit`, halved while the scores
    /// of its queries over `cached` earlier keys and its own would pass `budget` bytes, then, short of
    /// `remaining`, rounded down to a multiple of 64 where it holds one. At least one.
    func prefillPassLength(cached: Int, remaining: Int, limit: Int,
                           budget: Int = NFKMLXLanguageNet.prefillScoreBudget) -> Int {
        var length = Swift.min(remaining, limit)
        while length > 1 && configuration.headCount * length * (cached + length) * 4 > budget {
            length = (length + 1) / 2
        }
        if length < remaining && length >= 64 {
            length -= length % 64
        }
        return length
    }

    /// The state entering the stack, then each layer's output, with the final normalization folded
    /// into the last entry — the convention the reference's `output_hidden_states` uses.
    func hiddenStates(_ tokens: MLXArray, attention: NFKMLXHybridAttention = .causal) -> [MLXArray] {
        var trace = [MLXArray]()
        _ = normalizedHidden(tokens, attention: attention, trace: &trace)
        return trace
    }

    /// The final-normalized hidden states `[batch, length, hidden]`, without the output projection:
    /// the text model's `last_hidden_state`.
    func normalizedHidden(_ tokens: MLXArray, attention: NFKMLXHybridAttention = .causal) -> MLXArray {
        var trace: [MLXArray]? = nil
        return normalizedHidden(tokens, attention: attention, trace: &trace)
    }

    private func forward(_ tokens: MLXArray, trace: inout [MLXArray]) -> MLXArray {
        let hidden = normalizedHidden(tokens, attention: .causal, trace: &trace)
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    private func normalizedHidden(_ tokens: MLXArray, attention: NFKMLXHybridAttention,
                                  trace: inout [MLXArray]) -> MLXArray {
        var recorded: [MLXArray]? = trace
        let hidden = normalizedHidden(tokens, attention: attention, trace: &recorded)
        trace = recorded ?? []
        return hidden
    }

    /// The stack's pass. Each state is kept only where `trace` is non-nil, so a pass that reads only the
    /// final state lets every intermediate one go as the evaluation consumes it.
    private func normalizedHidden(_ tokens: MLXArray, attention: NFKMLXHybridAttention,
                                  trace: inout [MLXArray]?) -> MLXArray {
        var hidden = model.embedTokens(tokens)
        trace?.append(hidden)
        let length = tokens.shape[1]
        let mask: MLXArray? = attention == .causal && length > 1 ? NFKMLXLanguageNet.causalMask(length, offset: 0) : nil
        for (index, layer) in model.layers.enumerated() {
            guard let next = NFKMLXLayerStream.run(layerStream, layer: layer, index: index, { layer(hidden, mask: mask) },
                                                   keeping: { [$0] }) else {
                continue
            }
            hidden = next
            trace?.append(hidden)
        }
        hidden = model.norm(hidden)
        if trace != nil {
            trace![trace!.count - 1] = hidden
        }
        return hidden
    }
}

/// What a linear-attention layer carries between passes: the convolution's inputs before the next position
/// and the recurrence's state.
final class NFKMLXHybridLinearCache {
    /// `[batch, kernel - 1, qkv width]`, the fused projection's last rows, which the next position's
    /// convolution reads before its own.
    var convolutionInputs: MLXArray?
    /// `[batch, value heads, key width, value width]` in float32, the state after the last position.
    var state: MLXArray?
}

/// The caches of one generation over a hybrid decoder, so each pass runs only its new positions: the
/// full-attention layers' keys and values, and each linear-attention layer's convolution inputs and state.
///
/// @discussion The keys and values live in an ``NFKMLXKeyValueCache`` whose slots are the full-attention
/// layers' indices; its window and quantization apply to them alone. The recurrent state is fixed in size
/// and holds the whole prefix, so a window bounds what the attention layers read and nothing else.
final class NFKMLXHybridCache {
    let attention: NFKMLXKeyValueCache
    let linear: [NFKMLXHybridLinearCache?]
    /// The first full-attention layer, whose slot states how many keys a pass attends to.
    private let firstAttentionLayer: Int?

    init(configuration: NFKMLXHybridConfiguration, window: Int? = nil,
         quantization: NFKMLXKeyValueCache.Quantization? = nil) {
        attention = NFKMLXKeyValueCache(layerCount: configuration.layerCount, window: window,
                                        quantization: quantization)
        linear = configuration.layerTypes.map { $0 == .linearAttention ? NFKMLXHybridLinearCache() : nil }
        firstAttentionLayer = configuration.layerTypes.firstIndex(of: .fullAttention)
    }

    /// The positions the cache has seen.
    var offset: Int { attention.offset }

    /// How many cached keys a pass of more than one position attends to before its own: what a full-attention
    /// layer retains, after the trim a window makes before the pass.
    var maskCacheLength: Int {
        guard let layer = firstAttentionLayer else { return 0 }
        let retained = attention.retainedLength(layer: layer)
        return attention.window.map { Swift.min(retained, $0 - 1) } ?? retained
    }

    /// The bytes the cached arrays occupy.
    var allocatedBytes: Int {
        attention.allocatedBytes
            + linear.reduce(0) { $0 + ($1?.convolutionInputs?.nbytes ?? 0) + ($1?.state?.nbytes ?? 0) }
    }
}

/// Building the hybrid decoder and reading a release's configuration.
@objc(NFKMLXHybridLanguage)
public final class NFKMLXHybridLanguage: NSObject {

    /// A decoder at `configuration`, with random weights until ``loadWeights(into:fromDirectory:precision:)``
    /// fills it. Introduced in InferKit 0.4.0.
    public static func makeNet(_ configuration: NFKMLXHybridConfiguration = .qwen3_8_27B)
        -> NFKMLXHybridLanguageNet {
        NFKMLXHybridLanguageNet(configuration)
    }

    /// Reads a `Qwen3_5ForConditionalGeneration` config, whose decoder lives under `text_config`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXHybridConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        // The decoder's own numbers are nested; the top level describes the multimodal wrapper.
        let text = (json["text_config"] as? [String: Any]) ?? json
        guard (text["model_type"] as? String)?.hasPrefix("qwen3_5") == true
                || (json["model_type"] as? String)?.hasPrefix("qwen3_5") == true else {
            throw NFKMLXError.unsupportedConfiguration(
                "this reads the qwen3_5 hybrid decoder; use NFKMLXLanguage for a dense one")
        }

        func integer(_ key: String, _ fallback: Int) -> Int { (text[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (text[key] as? NSNumber)?.floatValue ?? fallback }

        var theta: Float = real("rope_theta", 10_000_000)
        var rotaryFactor: Float = real("partial_rotary_factor", 0.25)
        if let rope = text["rope_parameters"] as? [String: Any] {
            theta = (rope["rope_theta"] as? NSNumber)?.floatValue ?? theta
            rotaryFactor = (rope["partial_rotary_factor"] as? NSNumber)?.floatValue ?? rotaryFactor
        }

        let kinds = (text["layer_types"] as? [String])?
            .compactMap { NFKMLXHybridLayerKind(rawValue: $0) }

        return NFKMLXHybridConfiguration(
            hiddenSize: integer("hidden_size", 5120),
            layerCount: integer("num_hidden_layers", 64),
            intermediateSize: integer("intermediate_size", 17408),
            vocabularySize: integer("vocab_size", 248_320),
            rmsEpsilon: real("rms_norm_eps", 1e-6),
            headCount: integer("num_attention_heads", 24),
            keyValueHeadCount: integer("num_key_value_heads", 4),
            headDimensions: integer("head_dim", 256),
            ropeTheta: theta,
            partialRotaryFactor: rotaryFactor,
            gatesAttentionOutput: (text["attn_output_gate"] as? NSNumber)?.boolValue ?? true,
            linearKeyHeadCount: integer("linear_num_key_heads", 16),
            linearKeyHeadDimensions: integer("linear_key_head_dim", 128),
            linearValueHeadCount: integer("linear_num_value_heads", 48),
            linearValueHeadDimensions: integer("linear_value_head_dim", 128),
            linearConvolutionKernel: integer("linear_conv_kernel_dim", 4),
            fullAttentionInterval: integer("full_attention_interval", 4),
            tiesWordEmbeddings: (text["tie_word_embeddings"] as? NSNumber)?.boolValue
                ?? (json["tie_word_embeddings"] as? NSNumber)?.boolValue ?? false,
            layerTypes: kinds)
    }

    /// Loads a released hybrid decoder from its directory, following the shard index.
    ///
    /// The releases nest the decoder under `model.language_model.` beside a vision tower and a
    /// multi-token-prediction head; only the decoder's tensors are taken. Introduced in InferKit 0.4.0.
    public static func loadWeights(into net: NFKMLXHybridLanguageNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        try loadWeights(into: net, fromDirectory: directory, precision: precision, keeping: { _ in true })
    }

    /// The layers of a release's decoder `residency` leaves in the release, as
    /// ``NFKMLXLayerStream/plannedLayers(residency:budget:footprint:fitsWhole:)`` plans them; empty where it
    /// holds them all. `reserved` is what something held beside the decoder takes from the check a whole
    /// load makes; `budget` already leaves it out.
    static func streamedLayers(directory: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                               tied: Bool, budget: Int = NFKMLXResidencyBudget.current(),
                               reserving reserved: Int = 0) throws -> Set<Int> {
        try NFKMLXLayerStream.plannedLayers(residency: residency, budget: budget, footprint: {
            try NFKMLXLayerStream.footprint(
                directory: directory, precision: precision,
                converting: precision == .checkpoint ? try releasedType(inDirectory: directory) : nil,
                includes: { key in moduleKey(forReference: key).map { !(tied && $0.hasPrefix("lm_head.")) } ?? false },
                layerOf: { layer(ofKey: $0)?.layer })
        }, fitsWhole: {
            try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: precision, reserve: reserved)
        })
    }

    /// Loads a released decoder with the layers `residency` plans to leave in the release read in their
    /// turn on every pass (``NFKMLXHybridLanguageNet/layerStream``), and every other tensor held. Under
    /// ``NFKMLXResidency/automatic`` a release that fits loads whole, as
    /// ``loadWeights(into:fromDirectory:precision:)`` loads it. A streamed layer takes the types a whole
    /// load gives it: at `.checkpoint` every floating tensor takes the type most of the release carries.
    /// Only a caller that checks ``NFKMLXHybridLanguageNet/verifyStream()`` after each pass loads a
    /// streamed decoder. A parameter `retaining` names (an adapter's, which the release does not hold)
    /// keeps the value the net already has, in a streamed layer too. Introduced in InferKit 0.4.0.
    static func loadWeights(into net: NFKMLXHybridLanguageNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                            budget: Int = NFKMLXResidencyBudget.current(), reserving reserved: Int = 0,
                            retaining retained: (String) -> Bool = { _ in false }) throws {
        let streamed = try streamedLayers(directory: directory, precision: precision, residency: residency,
                                          tied: net.lmHead == nil, budget: budget, reserving: reserved)
        let kept = net.parameters().flattened().filter { retained($0.0) }
        guard let vacated = try NFKMLXLayerStream.vacating(
            streamed, of: net.model.layers, prefix: "model.layers.", directory: directory, precision: precision,
            converting: precision == .checkpoint ? try releasedType(inDirectory: directory) : nil,
            layerOf: layer(ofKey:), shaping: { name, shape in
                name.hasSuffix("conv1d.weight") && shape.count == 3 ? [shape[0], shape[2], shape[1]] : shape
            }) else {
            try loadWeights(into: net, fromDirectory: directory, precision: precision, keeping: { _ in true }, adding: kept)
            return
        }
        try loadWeights(into: net, fromDirectory: directory, precision: precision, keeping: { name in
            NFKMLXLayerStream.layer(ofDecoderName: String(name.dropFirst("model.".count))).map { streamed.contains($0.layer) } != true
        }, adding: vacated.placeholders + kept)
        net.layerStream = vacated.stream
        vacated.stream.prime()
    }

    /// The layer a release tensor belongs to and its name inside the layer, or nil outside the layers.
    static func layer(ofKey key: String) -> (layer: Int, name: String)? {
        moduleKey(forReference: key).flatMap { name in
            name.hasPrefix("model.") ? NFKMLXLayerStream.layer(ofDecoderName: String(name.dropFirst("model.".count))) : nil
        }
    }

    /// The floating type most of a release's decoder tensors are stored in, read from its shard headers:
    /// the type a `.checkpoint` load gives every floating tensor.
    static func releasedType(inDirectory directory: URL) throws -> DType? {
        var counts = [DType: Int]()
        for url in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            for (key, entry) in try NFKMLXSafetensors.entries(inFile: url) where moduleKey(forReference: key) != nil {
                if let dtype = NFKMLXSafetensors.dtype(entry.dtype), dtype.isFloatingPoint {
                    counts[dtype, default: 0] += 1
                }
            }
        }
        return counts.max { $0.value < $1.value }?.key
    }

    private static func loadWeights(into net: NFKMLXHybridLanguageNet, fromDirectory directory: URL,
                                    precision: NFKMLXWeightPrecision, keeping keep: (String) -> Bool,
                                    adding placeholders: [(String, MLXArray)] = []) throws {
        let tied = net.lmHead == nil
        let read = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision, remap: {
            guard let name = moduleKey(forReference: $0), keep(name) else { return nil }
            return tied && name.hasPrefix("lm_head.") ? nil : name
        })
        // The releases store `A_log` and the recurrence's norm weight in float32 beside bf16 elsewhere,
        // and the reference loads every tensor in the release's own type. At `.checkpoint` those take
        // the type most tensors carry, so the decay reads the same rounded `A_log` the reference does.
        let counts = Dictionary(grouping: read.map(\.1.dtype), by: { $0 }).mapValues(\.count)
        let released = counts.max { $0.value < $1.value }?.key
        // A depthwise convolution is stored [channels, 1, kernel] and wanted [channels, kernel, 1].
        let merged = read.map { name, value in
            var tensor = name.hasSuffix("conv1d.weight") && value.ndim == 3 ? value.transposed(0, 2, 1) : value
            if precision == .checkpoint, let released, tensor.dtype != released, tensor.dtype.isFloatingPoint {
                tensor = tensor.asType(released)
            }
            return (name, tensor)
        }
        try NFKMLXWeights.apply(merged + placeholders, to: net, verifyShapes: !placeholders.isEmpty)
    }

    /// Loads a released decoder at its stored precision and packs its projections to `bits` as it reads
    /// them, so the unpacked release never sits whole in memory.
    ///
    /// @discussion Each layer's tensors gather as the shards are read, and a layer is installed,
    /// quantized, and evaluated as soon as its last tensor arrives; the replaced weights then leave MLX's
    /// buffer cache. A release past the working set at its stored type (Qwen3.8-27B, about 52 GB at
    /// bfloat16) loads this way at the size of its packed form. `A_log` and the recurrence's norm weight
    /// take the type most tensors carry, as a `.checkpoint` load gives them. Every parameter must be
    /// supplied at its shape. Introduced in InferKit 0.4.0.
    ///
    /// - Parameters:
    ///   - net: the decoder to fill, built at the release's geometry.
    ///   - directory: the release directory, one file or a shard index.
    ///   - bits: the bits of each packed weight, 2 to 8.
    ///   - groupSize: the weights sharing one scale and bias.
    ///   - includeEmbeddings: packs the token embedding too. A tied decoder whose output projection runs
    ///     reads the packed matrix for both.
    public static func loadQuantizedWeights(into net: NFKMLXHybridLanguageNet, fromDirectory directory: URL,
                                            bits: Int, groupSize: Int = 64, includeEmbeddings: Bool = false) throws {
        let tied = net.lmHead == nil
        let expected = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1.shape) })
        let layerPrefix = "model.layers."
        func layerIndex(_ key: String) -> Int? {
            guard key.hasPrefix(layerPrefix) else { return nil }
            return Int(key.dropFirst(layerPrefix.count).prefix { $0 != "." })
        }
        var unread = Dictionary(grouping: expected.keys.compactMap { key in layerIndex(key).map { ($0, key) } },
                                by: \.0).mapValues { Set($0.map(\.1)) }
        var gathered = [Int: [(String, MLXArray)]]()
        var outside = [(String, MLXArray)]()
        var misshapen = [String]()
        var released: DType?
        for url in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            let arrays = try NFKMLXWeights.loadCheckpoint(url: url).arrays
            if released == nil {
                let counts = Dictionary(grouping: arrays.values.map(\.dtype).filter(\.isFloatingPoint), by: { $0 })
                released = counts.max { $0.value.count < $1.value.count }?.key
            }
            var completed = [Int]()
            for (key, value) in arrays {
                guard let name = moduleKey(forReference: key), !(tied && name.hasPrefix("lm_head.")),
                      let shape = expected[name] else { continue }
                var tensor = name.hasSuffix("conv1d.weight") && value.ndim == 3 ? value.transposed(0, 2, 1) : value
                if let released, tensor.dtype.isFloatingPoint, tensor.dtype != released {
                    tensor = tensor.asType(released)
                }
                if tensor.shape != shape {
                    misshapen.append("\(name) \(tensor.shape) for \(shape)")
                }
                guard let layer = layerIndex(name) else {
                    outside.append((name, tensor))
                    continue
                }
                gathered[layer, default: []].append((name, tensor))
                unread[layer]?.remove(name)
                if unread[layer]?.isEmpty == true {
                    completed.append(layer)
                }
            }
            guard misshapen.isEmpty else {
                throw NFKMLXError.weightsMismatch("mismatched \(misshapen.sorted().prefix(3))")
            }
            for layer in completed {
                let prefix = "\(layerPrefix)\(layer)."
                let block = net.model.layers[layer]
                block.update(parameters: ModuleParameters.unflattened(
                    (gathered.removeValue(forKey: layer) ?? []).map { (String($0.0.dropFirst(prefix.count)), $0.1) }))
                try NFKMLXQuantization.quantize(module: block, bits: bits, groupSize: groupSize)
                eval(block)
                unread[layer] = nil
                NFKMLXGPU.clearCache()
            }
        }
        let placed = Set(outside.map(\.0))
        let missingOutside = expected.keys.filter { layerIndex($0) == nil && !placed.contains($0) }
        let missing = (unread.values.flatMap { $0 } + missingOutside).sorted()
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch("the release lacks \(missing.count) parameters, such as \(missing.prefix(3))")
        }
        net.update(parameters: ModuleParameters.unflattened(outside))
        if includeEmbeddings {
            try NFKMLXQuantization.quantize(module: net.model, bits: bits, groupSize: groupSize, includeEmbeddings: true)
        }
        eval(net)
        NFKMLXGPU.clearCache()
    }

    /// The module key a checkpoint key maps to, or nil for a tensor this decoder does not carry. A
    /// `Qwen3_5Model` save (a decision checkpoint's backbone) holds the decoder under `language_model.`
    /// with no `model.` wrapper.
    static func moduleKey(forReference key: String) -> String? {
        for prefix in ["model.language_model.", "language_model."] where key.hasPrefix(prefix) {
            return "model." + key.dropFirst(prefix.count)
        }
        if key == "lm_head.weight" { return key }
        return nil                      // the vision tower and the multi-token-prediction head
    }

    /// The checkpoint key a parameter of this module corresponds to.
    ///
    /// The releases nest the decoder under `model.language_model.`, beside a vision tower and a
    /// multi-token-prediction head that this module does not implement.
    static func referenceKey(for parameter: String) -> String {
        parameter.hasPrefix("model.")
            ? "model.language_model." + parameter.dropFirst("model.".count)
            : parameter
    }

    /// A decoder built from a release directory at the geometry its `config.json` declares.
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, precision: NFKMLXWeightPrecision = .float32) throws
        -> NFKMLXHybridLanguageNet {
        let net = makeNet(try configuration(fromHuggingFace: directoryURL.appendingPathComponent("config.json")))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
        return net
    }

    /// The registry name the text backend builds under. Introduced in InferKit 0.4.0.
    @objc public static let modelName = "qwen3.5"

    static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    static let optionalFiles = ["vocab.json", "merges.txt", "generation_config.json", "chat_template.jinja"]
    static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// A text-generation backend from a Qwen3.5, 3.6, or 3.8 release directory, reading its
    /// `config.json`, the decoder's weights, the tokenizer, and the chat template.
    ///
    /// @discussion The backend is an ``NFKMLXDecoderBackend`` over the decoder's cache: the prompt fills it
    /// in passes, and each generated token then runs alone against the full-attention layers' keys and
    /// values and the linear-attention layers' recurrent state. Run inference off the render thread.
    /// Introduced in InferKit 0.4.0.
    public static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision) throws
        -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: .automatic)
    }

    /// ``backend(directoryURL:precision:)`` holding the decoder as `residency` says. Under
    /// ``NFKMLXResidency/automatic`` a release that fits the working set loads whole, and one that does
    /// not, such as Qwen3.8 27B on a 32 GB machine, holds the first layers that fit and reads the others
    /// from the release on every pass (``NFKMLXDecoderBackend/streamedLayerCount``);
    /// ``NFKMLXResidency/streamed`` streams wherever the decoder does not fit whole. A failed read fails
    /// the request, and the next request reads every layer again. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:precision:residency:error:)
    public static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: residency,
                    budget: NFKMLXResidencyBudget.current())
    }

    /// ``backend(directoryURL:precision:residency:)`` with a smaller release of the same vocabulary
    /// drafting for it: Qwen3.5 9B drafted by Qwen3.5 2B, for instance.
    ///
    /// @discussion A round is one pass of the draft per proposal and one pass of the release over every
    /// proposal (`NFKMLXGenerationParameterKey.draftTokens` a round, 4 by default, 0 to decode without
    /// it). At temperature 0 the reply is the release's greedy reply. The draft loads at its release's own
    /// precision and is quantized to `draftBits` (4 by default; nil keeps it at `precision`), its
    /// embeddings included, since a draft's precision moves how many proposals are kept and never the
    /// reply. The release is held as `residency` plans it against what the draft leaves of the working
    /// set. Introduced in InferKit 0.4.0.
    public static func backend(directoryURL: URL, draftDirectoryURL: URL, precision: NFKMLXWeightPrecision = .checkpoint,
                               residency: NFKMLXResidency = .automatic, draftBits: Int? = 4) throws -> NFKMLXDecoderBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: residency,
                    budget: NFKMLXResidencyBudget.current(), draftDirectoryURL: draftDirectoryURL, draftBits: draftBits)
    }

    /// The Objective-C entry for a release with a smaller release drafting for it, the draft at 4 bits;
    /// see ``backend(directoryURL:draftDirectoryURL:precision:residency:draftBits:)``. Introduced in
    /// InferKit 0.4.0.
    @objc(backendWithDirectoryURL:draftDirectoryURL:precision:residency:error:)
    public static func draftedBackend(directoryURL: URL, draftDirectoryURL: URL, precision: NFKMLXWeightPrecision,
                                      residency: NFKMLXResidency) throws -> NFKMLXDecoderBackend {
        try backend(directoryURL: directoryURL, draftDirectoryURL: draftDirectoryURL, precision: precision,
                    residency: residency)
    }

    static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                        budget: Int, identifier: String = modelName, draftDirectoryURL: URL? = nil,
                        draftBits: Int? = 4, quantizationBits: Int? = nil,
                        quantizationGroupSize: Int = NFKMLXLoadOptions.defaultQuantizationGroupSize,
                        generation: NFKMLXGenerationOptions = NFKMLXGenerationOptions()) throws -> NFKMLXDecoderBackend {
        if quantizationBits != nil && residency == .streamed {
            throw NFKMLXError.unsupportedConfiguration(
                "\(identifier) holds a quantized decoder resident: a streamed layer is read from the release as "
                + "stored, which holds no packed weights")
        }
        let configuration = try configuration(fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        var draft: NFKMLXHybridLanguageNet?
        if let draftDirectoryURL {
            let draftNet = makeNet(try self.configuration(fromHuggingFace: draftDirectoryURL.appendingPathComponent("config.json")))
            guard draftNet.configuration.vocabularySize == configuration.vocabularySize else {
                throw NFKMLXError.unsupportedConfiguration(
                    "the draft's vocabulary has \(draftNet.configuration.vocabularySize) entries where the model's has "
                    + "\(configuration.vocabularySize); a draft must share the model's vocabulary")
            }
            try loadWeights(into: draftNet, fromDirectory: draftDirectoryURL, precision: draftBits == nil ? precision : .checkpoint)
            if let draftBits {
                try NFKMLXQuantization.quantize(module: draftNet, bits: draftBits, includeEmbeddings: true)
                // The weights the quantized ones replace go before the release is planned, which they would
                // otherwise crowd.
                eval(draftNet)
                NFKMLXGPU.clearCache()
            }
            draft = draftNet
        }
        let draftBytes = draft.map { NFKMLXLanguage.heldBytes(of: $0) } ?? 0
        let net = makeNet(configuration)
        if let quantizationBits {
            // Each layer packs as its last tensor arrives, so the stored decoder is never held whole; the
            // embedding and the output projection pack after, as the catalog's estimate counts them.
            try loadQuantizedWeights(into: net, fromDirectory: directoryURL, bits: quantizationBits,
                                     groupSize: quantizationGroupSize, includeEmbeddings: true)
            try NFKMLXQuantization.quantize(module: net, bits: quantizationBits, groupSize: quantizationGroupSize)
            eval(net)
            NFKMLXGPU.clearCache()
        } else {
            try loadWeights(into: net, fromDirectory: directoryURL, precision: precision, residency: residency,
                            budget: budget > 0 ? Swift.max(budget - draftBytes, 1) : 0, reserving: draftBytes)
        }
        let backend = try NFKMLXDecoderBackend.release(directoryURL: directoryURL, identifier: identifier, modules: [net],
                                                       layerStream: net.layerStream,
                                                       draft: draft.map { draft in { draft($0) } },
                                                       cached: net.cachedGeneration) { net($0) }
        backend.cacheDefaults = generation
        return backend
    }


    /// The Objective-C entry for ``backend(directoryURL:precision:)`` at the released precision.
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: .checkpoint)
    }

    /// Downloads a release into the hub cache under `cacheDirectoryURL` (the default cache when nil)
    /// and builds its text backend. A cached file is not fetched again. The call blocks on the
    /// network; run it off the render thread. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws
        -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try backend(directoryURL: try NFKMLXReleaseDownload.directory(
                repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                required: requiredFiles, optional: optionalFiles, weights: weightFiles))
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

    /// Registers `qwen3.5` with `NFKMLXModelRegistry`; the registry's URL is the release directory.
    /// Introduced in InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { url in
            guard let url else {
                throw NFKMLXError.unsupportedConfiguration("qwen3.5 builds from a release directory, not without weights")
            }
            return try backend(directoryURL: url)
        }
    }
}
