//
//  NFKMLXGemma2.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// The Gemma-2 text decoder (`Gemma2Model`), SANA's text encoder — the caption features SANA's DiT
// cross-attends to are this model's last hidden state. Gemma 2 is a distinct architecture from the
// Gemma 3 / Gemma 4 text models here: it keeps the `(1 + w)` RMS normalization and the sandwich block
// (a norm before AND after each of attention and the feed-forward), but it has NO query/key norm, it
// SOFT-CAPS the attention logits (`tanh(logit/cap)·cap`), it uses a single rotary base with sliding-window
// attention on alternating layers, and it scales the query by `query_pre_attn_scalar^-0.5`. The
// attention is computed explicitly (not through the fused SDPA) because of the soft-cap.

/// Gemma 2 geometry. Defaults are the released 2B text encoder.
public struct NFKMLXGemma2Configuration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var kvHeadCount: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    public var queryPreAttnScalar: Float
    public var attnLogitSoftcap: Float
    public var slidingWindow: Int
    public var ropeTheta: Float
    public var rmsEps: Float

    public init(hiddenSize: Int = 2304, layerCount: Int = 26, headCount: Int = 8, kvHeadCount: Int = 4,
                headDim: Int = 256, intermediateSize: Int = 9216, vocabularySize: Int = 256000,
                queryPreAttnScalar: Float = 256, attnLogitSoftcap: Float = 50, slidingWindow: Int = 4096,
                ropeTheta: Float = 10000, rmsEps: Float = 1e-6) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.queryPreAttnScalar = queryPreAttnScalar
        self.attnLogitSoftcap = attnLogitSoftcap
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.rmsEps = rmsEps
    }

    public static let gemma2_2B = NFKMLXGemma2Configuration()

    public static let tiny = NFKMLXGemma2Configuration(
        hiddenSize: 32, layerCount: 3, headCount: 2, kvHeadCount: 1, headDim: 8, intermediateSize: 64,
        vocabularySize: 128, queryPreAttnScalar: 8, attnLogitSoftcap: 50, slidingWindow: 3)

    /// Gemma 2 alternates sliding-window and full attention, the even layers sliding.
    func isSliding(_ layer: Int) -> Bool { layer % 2 == 0 }
}

/// Gemma's RMS normalization: `x · rsqrt(mean(x²) + eps) · (1 + weight)`.
final class NFKGemma2RMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(_ dim: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.zeros([dim])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps)
        return normed * (1 + weight)
    }
}

/// The GeGLU feed-forward (`down(gelu_tanh(gate) · up)`).
final class NFKGemma2MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: NFKMLXGemma2Configuration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(geluApproximate(gate(x)) * up(x)) }
}

/// Applies the rotate-half rotary to `x` `[N, heads, headDim]` given `cos`/`sin` `[N, headDim]`.
private func gemma2Rope(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
    let headDim = x.dim(2)
    let half = headDim / 2
    let x1 = x[0..., 0..., 0 ..< half]
    let x2 = x[0..., 0..., half ..< headDim]
    let rotated = concatenated([-x2, x1], axis: -1)
    return x * c.expandedDimensions(axis: 1) + rotated * s.expandedDimensions(axis: 1)
}

/// Gemma 2 grouped-query attention with logit soft-capping and (per layer) a sliding window.
final class NFKGemma2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scaling: Float
    let softcap: Float

    init(_ config: NFKMLXGemma2Configuration) {
        self.heads = config.headCount
        self.kvHeads = config.kvHeadCount
        self.headDim = config.headDim
        self.scaling = pow(config.queryPreAttnScalar, -0.5)
        self.softcap = config.attnLogitSoftcap
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
    }

    /// `x` `[N, hidden]` (batch 1), `mask` `[N, N]` additive → `[N, hidden]`.
    func callAsFunction(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray, mask: MLXArray) -> MLXArray {
        let n = x.dim(0)
        var q = gemma2Rope(qProj(x).reshaped([n, heads, headDim]), cos: c, sin: s)
        var k = gemma2Rope(kProj(x).reshaped([n, kvHeads, headDim]), cos: c, sin: s)
        var v = vProj(x).reshaped([n, kvHeads, headDim])
        // [heads, N, headDim], repeating the KV heads to the query-head count.
        let repeats = heads / kvHeads
        q = q.transposed(1, 0, 2)
        k = repeated(k.transposed(1, 0, 2), count: repeats, axis: 0)
        v = repeated(v.transposed(1, 0, 2), count: repeats, axis: 0)
        var scores = matmul(q, k.transposed(0, 2, 1)) * scaling             // [heads, N, N]
        scores = tanh(scores / softcap) * softcap                          // logit soft-cap
        scores = scores + mask.reshaped([1, n, n])
        let attn = softmax(scores, axis: -1)
        let out = matmul(attn, v).transposed(1, 0, 2).reshaped([n, heads * headDim])
        return oProj(out)
    }
}

/// A Gemma 2 block: sandwich norms around the attention and the feed-forward.
final class NFKGemma2Layer: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKGemma2Attention
    @ModuleInfo(key: "mlp") var mlp: NFKGemma2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: NFKGemma2RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: NFKGemma2RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: NFKGemma2RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: NFKGemma2RMSNorm

    init(_ config: NFKMLXGemma2Configuration) {
        _attention.wrappedValue = NFKGemma2Attention(config)
        _mlp.wrappedValue = NFKGemma2MLP(config)
        _inputNorm.wrappedValue = NFKGemma2RMSNorm(config.hiddenSize, eps: config.rmsEps)
        _postAttentionNorm.wrappedValue = NFKGemma2RMSNorm(config.hiddenSize, eps: config.rmsEps)
        _preFeedForwardNorm.wrappedValue = NFKGemma2RMSNorm(config.hiddenSize, eps: config.rmsEps)
        _postFeedForwardNorm.wrappedValue = NFKGemma2RMSNorm(config.hiddenSize, eps: config.rmsEps)
    }

    func callAsFunction(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x + postAttentionNorm(attention(inputNorm(x), cos: c, sin: s, mask: mask))
        h = h + postFeedForwardNorm(mlp(preFeedForwardNorm(h)))
        return h
    }
}

/// The Gemma 2 text model. Returns the last hidden state (SANA's caption features).
public final class NFKMLXGemma2Net: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKGemma2Layer]
    @ModuleInfo(key: "norm") var norm: NFKGemma2RMSNorm

    let config: NFKMLXGemma2Configuration

    public init(_ config: NFKMLXGemma2Configuration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.layerCount).map { _ in NFKGemma2Layer(config) }
        _norm.wrappedValue = NFKGemma2RMSNorm(config.hiddenSize, eps: config.rmsEps)
    }

    /// `tokens` `[N]` (int32) → last hidden state `[N, hidden]`.
    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let n = tokens.dim(0)
        var h = embedTokens(tokens) * sqrt(Float(config.hiddenSize))

        let (cos, sin) = rotaryTable(positions: n)
        let causal = causalMask(n, window: nil)
        let sliding = causalMask(n, window: config.slidingWindow)

        for (index, layer) in layers.enumerated() {
            h = layer(h, cos: cos, sin: sin, mask: config.isSliding(index) ? sliding : causal)
        }
        return norm(h)
    }

    /// The rotary `cos`/`sin` tables `[N, headDim]` (rotate-half layout: frequencies repeated twice).
    private func rotaryTable(positions n: Int) -> (MLXArray, MLXArray) {
        let half = config.headDim / 2
        let k = MLXArray(stride(from: 0, to: config.headDim, by: 2).map { Float($0) })
        let invFreq = pow(MLXArray(config.ropeTheta), -(k / Float(config.headDim)))   // [half]
        let pos = MLXArray((0 ..< n).map { Float($0) }).reshaped([n, 1])
        let angles = pos * invFreq.reshaped([1, half])                    // [N, half]
        let full = concatenated([angles, angles], axis: -1)               // [N, headDim]
        return (cos(full), sin(full))
    }

    /// An additive causal mask `[N, N]`; a finite `window` also masks positions further back than it.
    private func causalMask(_ n: Int, window: Int?) -> MLXArray {
        var rows: [Float] = []
        for i in 0 ..< n {
            for j in 0 ..< n {
                let allowed = j <= i && (window == nil || i - j < window!)
                rows.append(allowed ? 0 : -1e9)
            }
        }
        return MLXArray(rows).reshaped([n, n])
    }
}
