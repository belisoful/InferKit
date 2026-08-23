//
//  NFKMLXGemmaModel.swift
//  InferKitMLX
//
//  The Gemma 4 text decoder (`gemma4_text`). A fourth architecture family here, and the one whose
//  distinguishing feature is **per-layer input embeddings**: alongside the ordinary token embedding,
//  every layer receives its own slice of a second, much wider embedding, gated into the residual after
//  the feed-forward. Attention alternates a sliding window with occasional full attention, and the
//  layers past `sharedKeyValueLayers` reuse an earlier layer's keys and values rather than computing
//  their own.
//
//  BUILT, NOT MEASURED. Gemma 4 is not in any released `transformers` (4.57.6 is the newest on PyPI and
//  stops at gemma3n), and the version that supports it requires Python 3.10 while this machine has
//  3.9.6 — so there is no oracle here and no parity record. The verification is structural, and on the
//  same footing as the hybrid decoder's rather than DeepSeek's: this checkpoint is bf16, so a float
//  module's parameters correspond one-to-one with it and the shape comparison is exact.
//
//  Scope: the text decoder. The release is tri-modal — it also carries a vision tower and an audio
//  Conformer — which are separate features, named in a test rather than left unsaid.
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// Whether a layer attends over a window or over everything before it.
public enum NFKMLXGemmaAttentionKind: String, Sendable {
    case sliding = "sliding_attention"
    case full = "full_attention"
}

/// The geometry of a Gemma 4 text decoder.
public struct NFKMLXGemmaConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    public var rmsEpsilon: Float
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int
    /// A FULL-attention layer runs wider heads than a sliding one. The config names this separately
    /// (`global_head_dim`), and using `head_dim` for both makes seven of the layers the wrong size.
    public var globalHeadDimensions: Int
    public var slidingWindow: Int
    /// Rotary base for the sliding layers.
    public var ropeTheta: Float
    /// Rotary base for the FULL-attention layers, which the config states separately.
    public var globalRopeTheta: Float
    /// The fraction of a full-attention head the rotary turns; a sliding head turns entirely.
    public var globalPartialRotaryFactor: Float
    /// The width of each layer's own input embedding.
    public var perLayerInputSize: Int
    /// Layers at or past this index reuse an earlier layer's keys and values.
    public var sharedKeyValueLayers: Int
    /// The feed-forward width in the layers that share keys and values.
    ///
    /// The config does NOT state this. It was read from the released checkpoint: E2B's first fifteen
    /// layers compute their own keys and values with a 6144-wide feed-forward, and the last twenty —
    /// exactly `num_kv_shared_layers` — share them and run 12288 instead. Sharing the cache and
    /// widening the feed-forward go together.
    public var sharedLayerIntermediateSize: Int

    /// The logits are squashed through `tanh` at this scale; 0 disables it.
    public var finalLogitSoftcap: Float
    public var layerTypes: [NFKMLXGemmaAttentionKind]

    public init(hiddenSize: Int = 1536, layerCount: Int = 35, intermediateSize: Int = 6144,
                vocabularySize: Int = 262_144, rmsEpsilon: Float = 1e-6, headCount: Int = 8,
                keyValueHeadCount: Int = 1, headDimensions: Int = 256,
                globalHeadDimensions: Int = 512, slidingWindow: Int = 512,
                ropeTheta: Float = 10_000, globalRopeTheta: Float = 1_000_000,
                globalPartialRotaryFactor: Float = 0.25, perLayerInputSize: Int = 256,
                sharedKeyValueLayers: Int = 20, sharedLayerIntermediateSize: Int? = nil,
                finalLogitSoftcap: Float = 30,
                layerTypes: [NFKMLXGemmaAttentionKind]? = nil) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.rmsEpsilon = rmsEpsilon
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.globalHeadDimensions = globalHeadDimensions
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.globalRopeTheta = globalRopeTheta
        self.globalPartialRotaryFactor = globalPartialRotaryFactor
        self.perLayerInputSize = perLayerInputSize
        self.sharedKeyValueLayers = sharedKeyValueLayers
        self.sharedLayerIntermediateSize = sharedLayerIntermediateSize ?? (intermediateSize * 2)
        self.finalLogitSoftcap = finalLogitSoftcap
        self.layerTypes = layerTypes ?? Array(repeating: .sliding, count: layerCount)
    }

    /// The released `google/gemma-4-E2B-it` text decoder.
    public static let e2b = NFKMLXGemmaConfiguration()

    /// The released `google/gemma-4-12B-it` text decoder.
    public static let twelveB = NFKMLXGemmaConfiguration(
        hiddenSize: 3840, layerCount: 48, intermediateSize: 15360, headCount: 16,
        keyValueHeadCount: 8, slidingWindow: 1024)

    /// The first layer that shares keys and values with an earlier one.
    var firstSharedLayer: Int { layerCount - sharedKeyValueLayers }

    /// The feed-forward width at this depth.
    func intermediateSize(atLayer layer: Int) -> Int {
        layer >= firstSharedLayer ? sharedLayerIntermediateSize : intermediateSize
    }

    /// The rotary base a layer of this kind uses.
    func ropeTheta(for kind: NFKMLXGemmaAttentionKind) -> Float {
        kind == .full ? globalRopeTheta : ropeTheta
    }

    /// How many of a head's channels the rotary turns. A sliding head turns entirely; a full head
    /// turns only `globalPartialRotaryFactor` of its width.
    func rotaryDimensions(for kind: NFKMLXGemmaAttentionKind) -> Int {
        kind == .full ? Int(Float(headDimensions(for: kind)) * globalPartialRotaryFactor)
                      : headDimensions(for: kind)
    }

    /// The head width a layer of this kind runs.
    func headDimensions(for kind: NFKMLXGemmaAttentionKind) -> Int {
        kind == .full ? globalHeadDimensions : headDimensions
    }

    /// The width of the second embedding, which holds every layer's own input.
    var perLayerEmbeddingWidth: Int { layerCount * perLayerInputSize }
}

/// Gemma 4's normalization: a plain RMS scale, `x · w`, with the weight initialized to one.
///
/// Gemma **3** used `x · (1 + w)`, and assuming Gemma 4 inherited it is what first made this port
/// wrong: the per-layer isolation harness put the divergence at `input_layernorm` on layer 0, whose
/// input was exact. Read the model's own source rather than the family's reputation.
final class NFKGemmaNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        epsilon = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = x * rsqrt((x * x).mean(axis: -1, keepDims: true) + epsilon)
        return normalized * weight
    }
}

/// Gemma 4 attention: grouped queries, per-head normalization, and a window that most layers use.
final class NFKGemmaAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: NFKGemmaNorm
    @ModuleInfo(key: "k_norm") var keyNorm: NFKGemmaNorm

    let configuration: NFKMLXGemmaConfiguration
    /// Values are RMS-normalized with NO learned scale, so this carries no parameter and a structural
    /// check cannot see it. It still changes every number the layer produces.
    let valueEpsilon: Float
    let kind: NFKMLXGemmaAttentionKind
    let rope: RoPE
    /// Present only on a full-attention layer, whose rotary is the `proportional` kind.
    let proportionalRope: NFKGemmaRotary?

    init(_ c: NFKMLXGemmaConfiguration, kind: NFKMLXGemmaAttentionKind) {
        configuration = c
        self.kind = kind
        let width = c.headDimensions(for: kind)
        rope = RoPE(dimensions: width, traditional: false, base: c.ropeTheta(for: kind))
        proportionalRope = kind == .full
            ? NFKGemmaRotary(width: width, theta: c.globalRopeTheta,
                             proportion: c.globalPartialRotaryFactor)
            : nil
        _queryProjection.wrappedValue = Linear(c.hiddenSize, c.headCount * width, bias: false)
        _keyProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * width, bias: false)
        _valueProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * width, bias: false)
        _outputProjection.wrappedValue = Linear(c.headCount * width, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = NFKGemmaNorm(dimensions: width, eps: c.rmsEpsilon)
        _keyNorm.wrappedValue = NFKGemmaNorm(dimensions: width, eps: c.rmsEpsilon)
        valueEpsilon = c.rmsEpsilon
        super.init()
    }

    /// Applies the rotary to the leading channels only, which is what a partial factor means.
    private func turned(_ x: MLXArray) -> MLXArray {
        if let proportionalRope { return proportionalRope(x) }
        return rope(x, offset: 0)
    }

    /// - Parameter shared: keys and values from an earlier layer, for a layer that shares them.
    /// - Returns: the attention output and the keys and values this layer used.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?,
                        shared: (keys: MLXArray, values: MLXArray)?)
        -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let c = configuration
        let (batch, length) = (x.shape[0], x.shape[1])
        let width = c.headDimensions(for: kind)

        var queries = queryNorm(queryProjection(x)
            .reshaped([batch, length, c.headCount, width])).transposed(0, 2, 1, 3)
        queries = turned(queries)

        // A sharing layer reuses an earlier layer's keys and values rather than projecting its own.
        // Its checkpoint still carries k_proj and v_proj; the reference ignores them, so this does too.
        var keys: MLXArray
        var values: MLXArray
        if let shared {
            keys = shared.keys
            values = shared.values
        } else {
            keys = keyNorm(keyProjection(x)
                .reshaped([batch, length, c.keyValueHeadCount, width])).transposed(0, 2, 1, 3)
            keys = turned(keys)
            var v = valueProjection(x).reshaped([batch, length, c.keyValueHeadCount, width])
            v = v * rsqrt((v * v).mean(axis: -1, keepDims: true) + valueEpsilon)
            values = v.transposed(0, 2, 1, 3)
        }

        // A sliding layer additionally forbids anything older than the window.
        var effective = mask
        if kind == .sliding, length > 1 {
            effective = NFKMLXGemmaLanguage.windowMask(length, window: c.slidingWindow)
        }
        // The masks are built float32, and a `.checkpoint`-precision load makes this a bf16 module;
        // the fused attention refuses a mask that does not promote to its own type, so the mask takes
        // the queries' dtype. Invisible at float32, which is why no float32 run ever raised it.
        effective = effective.map { $0.asType(queries.dtype) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            // The queries are already per-head normalized, so the reference attends at scale 1.
            scale: 1, mask: effective)
        let output = outputProjection(attended.transposed(0, 2, 1, 3)
            .reshaped([batch, length, c.headCount * width]))
        return (output, keys, values)
    }
}

/// One decoder layer, including the gate that folds in this layer's own input embedding.
final class NFKGemmaBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKGemmaAttention
    @ModuleInfo(key: "mlp") var feedForward: NFKGemmaFeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: NFKGemmaNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: NFKGemmaNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: NFKGemmaNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: NFKGemmaNorm
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerNorm: NFKGemmaNorm
    @ModuleInfo(key: "per_layer_input_gate") var perLayerGate: Linear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ c: NFKMLXGemmaConfiguration, kind: NFKMLXGemmaAttentionKind, layer: Int) {
        _attention.wrappedValue = NFKGemmaAttention(c, kind: kind)
        _feedForward.wrappedValue = NFKGemmaFeedForward(
            hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize(atLayer: layer))
        _inputNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postAttentionNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _preFeedForwardNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postFeedForwardNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postPerLayerNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _perLayerGate.wrappedValue = Linear(c.hiddenSize, c.perLayerInputSize, bias: false)
        _perLayerProjection.wrappedValue = Linear(c.perLayerInputSize, c.hiddenSize, bias: false)
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, perLayerInput: MLXArray, mask: MLXArray?,
                        shared: (keys: MLXArray, values: MLXArray)?)
        -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let (mixed, keys, values) = attention(inputNorm(x), mask: mask, shared: shared)
        let attended = x + postAttentionNorm(mixed)
        let lifted = attended + postFeedForwardNorm(feedForward(preFeedForwardNorm(attended)))

        // This layer's own slice of the second embedding. The activation sits on the GATE, not on the
        // slice, and the projection's result is added back before the layer's scalar applies.
        let gated = geluApproximate(perLayerGate(lifted)) * perLayerInput
        let folded = lifted + postPerLayerNorm(perLayerProjection(gated))

        // `layer_scalar` scales the WHOLE layer output, not only the folded term.
        return (folded * layerScalar, keys, values)
    }
}

/// Gemma 4's `proportional` rotary, which the full-attention layers use.
///
/// The frequencies are computed over the WHOLE head width and then truncated: the first
/// `partial_rotary_factor` of the pairs get a real frequency and the rest get **zero**, which is an
/// identity rotation. That is not the same as rotating a contiguous leading slice — with the
/// rotate-half layout a pair is (i, i + width/2), so a quarter-turned 512-wide head turns channels
/// 0…63 and 256…319, not 0…127. Getting that wrong leaves the sliding layers exact and every full
/// layer subtly off, which is what the isolation harness showed at layer 4.
final class NFKGemmaRotary {
    private let cosines: MLXArray
    private let sines: MLXArray
    let width: Int

    init(width: Int, theta: Float, proportion: Float, maximumPositions: Int = 4096) {
        self.width = width
        let pairs = width / 2
        let turned = Int(proportion * Float(width)) / 2
        var frequencies = [Float](repeating: 0, count: pairs)
        for pair in 0 ..< min(turned, pairs) {
            // The exponent divides by the FULL head width, not by the turned part.
            frequencies[pair] = 1 / powf(theta, Float(2 * pair) / Float(width))
        }
        let positions = MLXArray((0 ..< maximumPositions).map(Float.init)).reshaped([maximumPositions, 1])
        let angles = positions * MLXArray(frequencies).reshaped([1, pairs])
        cosines = cos(angles)
        sines = sin(angles)
    }

    /// Applies the rotation to `x` shaped `[batch, heads, length, width]`.
    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let length = x.shape[2]
        let half = width / 2
        let c = cosines[offset ..< (offset + length)].reshaped([1, 1, length, half])
        let s = sines[offset ..< (offset + length)].reshaped([1, 1, length, half])
        let first = x[.ellipsis, 0 ..< half]
        let second = x[.ellipsis, half...]
        return concatenated([first * c - second * s, second * c + first * s], axis: -1)
    }
}

/// Gemma's gated feed-forward.
///
/// The same shape as the dense decoder's SwiGLU but with Gemma's own activation: the config names
/// `gelu_pytorch_tanh`, not silu. Reusing the SwiGLU cost the layer 0.99 instead of 1.0, which the
/// isolation harness located to the MLP once attention was exact.
final class NFKGemmaFeedForward: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(geluApproximate(gate(x)) * up(x))
    }
}

/// `gelu_pytorch_tanh`, which is the activation Gemma names.
func geluApproximate(_ x: MLXArray) -> MLXArray {
    0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
}

/// The Gemma 4 text decoder.
public final class NFKMLXGemmaNet: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_tokens_per_layer") var embedPerLayer: Embedding
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: NFKGemmaNorm
    @ModuleInfo(key: "layers") var layers: [NFKGemmaBlock]
    @ModuleInfo(key: "norm") var norm: NFKGemmaNorm

    let configuration: NFKMLXGemmaConfiguration

    init(_ c: NFKMLXGemmaConfiguration) {
        configuration = c
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _embedPerLayer.wrappedValue = Embedding(embeddingCount: c.vocabularySize,
                                                dimensions: c.perLayerEmbeddingWidth)
        _perLayerModelProjection.wrappedValue = Linear(c.hiddenSize, c.perLayerEmbeddingWidth, bias: false)
        _perLayerProjectionNorm.wrappedValue = NFKGemmaNorm(dimensions: c.perLayerInputSize,
                                                            eps: c.rmsEpsilon)
        _layers.wrappedValue = (0 ..< c.layerCount).map {
            NFKGemmaBlock(c, kind: c.layerTypes[$0], layer: $0)
        }
        _norm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    /// The hidden state entering the stack, then the state each layer produced.
    ///
    /// The reference's `output_hidden_states` has the same shape, so comparing them one at a time says
    /// WHICH layer first diverges — which a whole-model cosine cannot.
    func hiddenStates(_ tokens: MLXArray) -> [MLXArray] {
        var trace = [MLXArray]()
        _ = forward(tokens, trace: &trace)
        return trace
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var trace = [MLXArray]()
        return forward(tokens, trace: &trace)
    }

    private func forward(_ tokens: MLXArray, trace: inout [MLXArray]) -> MLXArray {
        let c = configuration
        let (batch, length) = (tokens.shape[0], tokens.shape[1])

        // Gemma scales each embedding by the square root of ITS OWN width: the token embedding by
        // sqrt(hiddenSize), the per-layer one by sqrt(perLayerInputSize).
        var hidden = embedTokens(tokens) * sqrt(Float(c.hiddenSize))
        let identity = (embedPerLayer(tokens) * sqrt(Float(c.perLayerInputSize)))
            .reshaped([batch, length, c.layerCount, c.perLayerInputSize])

        // The context projection is scaled DOWN by the model width, reshaped per layer, and normalized
        // before it meets the token's own slice; the sum is then scaled by 1/sqrt(2).
        let projected = perLayerProjectionNorm(
            (perLayerModelProjection(hidden) * (1 / sqrt(Float(c.hiddenSize))))
                .reshaped([batch, length, c.layerCount, c.perLayerInputSize]))
        let normalized = (projected + identity) * (1 / sqrt(Float(2)))

        trace.append(hidden)
        let mask: MLXArray? = length > 1 ? NFKMLXLanguageNet.causalMask(length, offset: 0) : nil
        // The sharing layers reuse the keys and values of the LAST non-sharing layer OF THEIR OWN
        // KIND: a sliding layer shares with the last sliding one, a full layer with the last full one.
        var latest = [NFKMLXGemmaAttentionKind: (keys: MLXArray, values: MLXArray)]()
        for (index, layer) in layers.enumerated() {
            let kind = c.layerTypes[index]
            let shared = index >= c.firstSharedLayer ? latest[kind] : nil
            let (next, keys, values) = layer(hidden, perLayerInput: normalized[0..., 0..., index],
                                             mask: mask, shared: shared)
            hidden = next
            trace.append(hidden)
            if index < c.firstSharedLayer {
                latest[kind] = (keys, values)
            }
        }
        hidden = norm(hidden)
        // The reference records its LAST hidden state after the final normalization, so the trace
        // matches its convention rather than ending on the raw last-layer output.
        if !trace.isEmpty { trace[trace.count - 1] = hidden }

        // Tied embeddings, then the soft cap the release applies to its logits.
        var logits = embedTokens.asLinear(hidden)
        if c.finalLogitSoftcap > 0 {
            logits = tanh(logits / c.finalLogitSoftcap) * c.finalLogitSoftcap
        }
        return logits
    }
}

/// Building the decoder and reading a release's configuration.
@objc(NFKMLXGemmaLanguage)
public final class NFKMLXGemmaLanguage: NSObject {

    static func makeNet(_ configuration: NFKMLXGemmaConfiguration = .e2b) -> NFKMLXGemmaNet {
        NFKMLXGemmaNet(configuration)
    }

    /// A causal mask that additionally forbids anything older than `window`.
    static func windowMask(_ length: Int, window: Int) -> MLXArray {
        let rows = MLXArray(0 ..< length).reshaped([length, 1])
        let columns = MLXArray(0 ..< length).reshaped([1, length])
        // Inside the window and not in the future.
        let recent = MLX.where(columns .> (rows - window), MLXArray(Float(1)), MLXArray(Float(0)))
        let causal = MLX.where(columns .<= rows, MLXArray(Float(1)), MLXArray(Float(0)))
        return MLX.where((recent * causal) .> 0, MLXArray(Float(0)), MLXArray(Float(-1e9)))
    }

    /// Reads a released `config.json`, whose decoder sits under `text_config` in a multimodal release.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXGemmaConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        let text = (json["text_config"] as? [String: Any]) ?? json
        let kind = (text["model_type"] as? String) ?? (json["model_type"] as? String) ?? ""
        // The exact type, not the prefix: `gemma4_unified_text` (the 12B) is a different architecture,
        // and a prefix guard would load its weights into this stack and produce fluent nonsense — the
        // trap the dense Qwen reader is hardened against.
        guard kind == "gemma4_text" || kind == "gemma4" else {
            throw NFKMLXError.unsupportedConfiguration("this reads a gemma4 text decoder, not \(kind)")
        }
        // The dense sizes carry the expert fields too, nulled — the family reserves them — so the
        // guard reads the number rather than testing for the key.
        let experts = (text["num_experts"] as? NSNumber)?.intValue ?? 0
        guard experts <= 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "the config describes a \(experts)-expert mixture, which this network does not implement")
        }

        func integer(_ key: String, _ fallback: Int) -> Int { (text[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (text[key] as? NSNumber)?.floatValue ?? fallback }

        // The two layer kinds carry their own rotary settings, which is what `rope_parameters` splits.
        var theta: Float = real("rope_theta", 10_000)
        var globalTheta: Float = 1_000_000
        var globalFactor: Float = 0.25
        if let rope = text["rope_parameters"] as? [String: Any] {
            if let sliding = rope["sliding_attention"] as? [String: Any] {
                theta = (sliding["rope_theta"] as? NSNumber)?.floatValue ?? theta
            }
            if let full = rope["full_attention"] as? [String: Any] {
                globalTheta = (full["rope_theta"] as? NSNumber)?.floatValue ?? globalTheta
                globalFactor = (full["partial_rotary_factor"] as? NSNumber)?.floatValue ?? globalFactor
            }
        }

        return NFKMLXGemmaConfiguration(
            hiddenSize: integer("hidden_size", 1536),
            layerCount: integer("num_hidden_layers", 35),
            intermediateSize: integer("intermediate_size", 6144),
            vocabularySize: integer("vocab_size", 262_144),
            rmsEpsilon: real("rms_norm_eps", 1e-6),
            headCount: integer("num_attention_heads", 8),
            keyValueHeadCount: integer("num_key_value_heads", 1),
            headDimensions: integer("head_dim", 256),
            globalHeadDimensions: integer("global_head_dim", 512),
            slidingWindow: integer("sliding_window", 512),
            ropeTheta: theta,
            globalRopeTheta: globalTheta,
            globalPartialRotaryFactor: globalFactor,
            perLayerInputSize: integer("hidden_size_per_layer_input", 256),
            sharedKeyValueLayers: integer("num_kv_shared_layers", 20),
            finalLogitSoftcap: real("final_logit_softcapping", 30),
            layerTypes: (text["layer_types"] as? [String])?
                .compactMap { NFKMLXGemmaAttentionKind(rawValue: $0) })
    }

    /// The checkpoint key a parameter of this module corresponds to.
    static func referenceKey(for parameter: String) -> String { "model.language_model." + parameter }

    /// Loads the decoder from a released directory, taking only the language model's tensors.
    ///
    /// The release is tri-modal, so most of it is the vision and audio towers; they are skipped rather
    /// than loaded, and a strict apply then proves the decoder's own set is complete.
    ///
    /// Reads a sharded release as well as a single-file one. E2B ships one `model.safetensors`, and
    /// every larger size splits across a shard index, so a loader reading only the single file covers
    /// the smallest model and nothing else.
    static func loadWeights(into net: NFKMLXGemmaNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        let prefix = "model.language_model."
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision) {
            $0.hasPrefix(prefix) ? String($0.dropFirst(prefix.count)) : nil
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
