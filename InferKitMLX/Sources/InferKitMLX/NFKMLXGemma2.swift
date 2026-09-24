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

    /// The released Gemma 2 9B: 3584 wide over 42 layers of 16 heads (8 key-value) at 256, a 14336
    /// feed-forward, the query scaled by `256^-0.5`.
    public static let gemma2_9B = NFKMLXGemma2Configuration(
        hiddenSize: 3584, layerCount: 42, headCount: 16, kvHeadCount: 8, headDim: 256,
        intermediateSize: 14336, queryPreAttnScalar: 256)

    /// The released Gemma 2 27B: 4608 wide over 46 layers of 32 heads (16 key-value) at 128, a 36864
    /// feed-forward, the query scaled by `144^-0.5` — the one size whose scalar is not its head width.
    public static let gemma2_27B = NFKMLXGemma2Configuration(
        hiddenSize: 4608, layerCount: 46, headCount: 32, kvHeadCount: 16, headDim: 128,
        intermediateSize: 36864, queryPreAttnScalar: 144)

    public static let tiny = NFKMLXGemma2Configuration(
        hiddenSize: 32, layerCount: 3, headCount: 2, kvHeadCount: 1, headDim: 8, intermediateSize: 64,
        vocabularySize: 128, queryPreAttnScalar: 8, attnLogitSoftcap: 50, slidingWindow: 3)

    /// Gemma 2 alternates sliding-window and full attention, the even layers sliding.
    func isSliding(_ layer: Int) -> Bool { layer % 2 == 0 }

    /// Reads a Gemma 2 geometry from a release's `config.json`. Rejects a config whose `model_type` is
    /// not `gemma2`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXGemma2Configuration {
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        let modelType = json["model_type"] as? String
        guard modelType == "gemma2" else {
            throw NFKMLXError.unsupportedConfiguration("expected model_type gemma2, found \(modelType ?? "nil")")
        }
        func int(_ key: String) throws -> Int {
            guard let value = (json[key] as? NSNumber)?.intValue else {
                throw NFKMLXError.unsupportedConfiguration("the config carries no \(key)")
            }
            return value
        }
        func float(_ key: String, _ fallback: Float) -> Float {
            (json[key] as? NSNumber)?.floatValue ?? fallback
        }
        let defaults = NFKMLXGemma2Configuration()
        return NFKMLXGemma2Configuration(
            hiddenSize: try int("hidden_size"), layerCount: try int("num_hidden_layers"),
            headCount: try int("num_attention_heads"), kvHeadCount: try int("num_key_value_heads"),
            headDim: try int("head_dim"), intermediateSize: try int("intermediate_size"),
            vocabularySize: try int("vocab_size"),
            queryPreAttnScalar: float("query_pre_attn_scalar", defaults.queryPreAttnScalar),
            attnLogitSoftcap: float("attn_logit_softcapping", defaults.attnLogitSoftcap),
            slidingWindow: (json["sliding_window"] as? NSNumber)?.intValue ?? defaults.slidingWindow,
            ropeTheta: float("rope_theta", defaults.ropeTheta), rmsEps: float("rms_norm_eps", defaults.rmsEps))
    }
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
        NFKReferenceRounding.gemmaNorm(x, weight: weight, eps: eps)
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

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(NFKReferenceRounding.geluTanh(gate(x)) * up(x)) }
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
    let ropeTheta: Float

    init(_ config: NFKMLXGemma2Configuration) {
        self.heads = config.headCount
        self.kvHeads = config.kvHeadCount
        self.headDim = config.headDim
        self.scaling = pow(config.queryPreAttnScalar, -0.5)
        self.softcap = config.attnLogitSoftcap
        self.ropeTheta = config.ropeTheta
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
    }

    /// `x` `[N, hidden]` (batch 1), `mask` `[N, N]` additive → `[N, hidden]`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let n = x.dim(0)
        func heads(_ projected: MLXArray, _ count: Int) -> MLXArray {
            projected.reshaped([1, n, count, headDim]).transposed(0, 2, 1, 3)
        }
        let q = NFKReferenceRounding.rotary(heads(qProj(x), self.heads), dimensions: headDim, base: ropeTheta, offset: 0)
        let k = NFKReferenceRounding.rotary(heads(kProj(x), kvHeads), dimensions: headDim, base: ropeTheta, offset: 0)
        let v = heads(vProj(x), kvHeads)
        let out = NFKReferenceRounding.attention(queries: q, keys: k, values: v, scale: scaling,
                                                 mask: mask, softcap: softcap)
        return oProj(out[0].transposed(1, 0, 2).reshaped([n, self.heads * headDim]))
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

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x + postAttentionNorm(attention(inputNorm(x), mask: mask))
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

    /// A released Gemma 2 decoder from its directory: the geometry from `config.json`, the weights from
    /// its safetensors, the `lm_head` (tied to the embedding) left unread. `precision` `.checkpoint`
    /// keeps the released bf16. Introduced in InferKit 0.4.0.
    public static func load(directoryURL directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXGemma2Net {
        let net = NFKMLXGemma2Net(try NFKMLXGemma2Configuration.configuration(
            fromHuggingFace: directory.appendingPathComponent("config.json")))
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision) { key in
            key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : nil
        }
        try NFKMLXWeights.apply(mapped, to: net)
        return net
    }

    /// `tokens` `[N]` (int32) → last hidden state `[N, hidden]`.
    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var trace = [MLXArray]()
        return forward(tokens, trace: &trace)
    }

    /// The embedding, each block's output, and the normalized last state, `[N, hidden]` each.
    func layerStates(_ tokens: MLXArray) -> [MLXArray] {
        var trace = [MLXArray]()
        _ = forward(tokens, trace: &trace)
        return trace
    }

    private func forward(_ tokens: MLXArray, trace: inout [MLXArray]) -> MLXArray {
        let n = tokens.dim(0)
        var h = embedTokens(tokens) * sqrt(Float(config.hiddenSize))
        trace.append(h)

        let causal = causalMask(n, window: nil)
        let sliding = causalMask(n, window: config.slidingWindow)

        for (index, layer) in layers.enumerated() {
            h = layer(h, mask: config.isSliding(index) ? sliding : causal)
            trace.append(h)
        }
        h = norm(h)
        trace[trace.count - 1] = h
        return h
    }

    /// An additive causal mask `[N, N]`; a finite `window` also masks positions further back than it.
    func causalMask(_ n: Int, window: Int?) -> MLXArray {
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
