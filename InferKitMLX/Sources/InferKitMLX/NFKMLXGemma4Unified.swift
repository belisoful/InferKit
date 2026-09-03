//
//  NFKMLXGemma4Unified.swift
//  InferKitMLX
//
//  The Gemma 4 unified text decoder (`gemma4_unified_text`, the 12B), a DIFFERENT decoder from the
//  E-series `gemma4_text`: it has no per-layer input embeddings and no mixture of experts, only the
//  sandwich-normed block with a per-layer scalar. Its attention is the same one the E-series runs — a
//  learned query and key norm, a scale-free value norm, attention at scale 1, per-layer head widths
//  (a full layer runs 512 where a sliding one runs its own), the proportional rotary on the full
//  layers, and key/value sharing on the trailing layers — so `NFKGemmaAttention` and
//  `NFKGemmaFeedForward` are reused directly, and only the block and the model are new here.
//

import Foundation
import MLX
import MLXNN

/// One unified decoder layer: attention and feed-forward, each sandwiched between a pre- and a
/// post-normalization and added back, then scaled by a learned per-layer scalar. This is the E-series
/// block WITHOUT the per-layer input embedding it folds in and without the routed-expert branch.
final class NFKGemma4UnifiedBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKGemmaAttention
    @ModuleInfo(key: "mlp") var feedForward: NFKGemmaFeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: NFKGemmaNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: NFKGemmaNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: NFKGemmaNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: NFKGemmaNorm
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ c: NFKMLXGemmaConfiguration, kind: NFKMLXGemmaAttentionKind, layer: Int) {
        _attention.wrappedValue = NFKGemmaAttention(c, kind: kind)
        _feedForward.wrappedValue = NFKGemmaFeedForward(
            hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize(atLayer: layer))
        _inputNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postAttentionNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _preFeedForwardNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postFeedForwardNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?,
                        shared: (keys: MLXArray, values: MLXArray)?)
        -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let (mixed, keys, values) = attention(inputNorm(x), mask: mask, shared: shared)
        let attended = x + postAttentionNorm(mixed)
        let lifted = attended + postFeedForwardNorm(feedForward(preFeedForwardNorm(attended)))
        return (lifted * layerScalar, keys, values)
    }
}

/// The Gemma 4 unified text decoder.
public final class NFKMLXGemma4UnifiedNet: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKGemma4UnifiedBlock]
    @ModuleInfo(key: "norm") var norm: NFKGemmaNorm

    let configuration: NFKMLXGemmaConfiguration

    init(_ c: NFKMLXGemmaConfiguration) {
        configuration = c
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.layerCount).map {
            NFKGemma4UnifiedBlock(c, kind: c.layerTypes[$0], layer: $0)
        }
        _norm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    /// The hidden state entering the stack, then the state each layer produced, matching the
    /// reference's `output_hidden_states` so the isolation harness can locate a divergence.
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
        let length = tokens.shape[1]

        var hidden = embedTokens(tokens) * sqrt(Float(c.hiddenSize))
        trace.append(hidden)

        let mask: MLXArray? = length > 1 ? NFKMLXLanguageNet.causalMask(length, offset: 0) : nil
        // A sharing layer reuses the keys and values of the last non-sharing layer OF ITS OWN KIND.
        var latest = [NFKMLXGemmaAttentionKind: (keys: MLXArray, values: MLXArray)]()
        for (index, layer) in layers.enumerated() {
            let kind = c.layerTypes[index]
            let shared = index >= c.firstSharedLayer ? latest[kind] : nil
            let (next, keys, values) = layer(hidden, mask: mask, shared: shared)
            hidden = next
            trace.append(hidden)
            if index < c.firstSharedLayer {
                latest[kind] = (keys, values)
            }
        }
        hidden = norm(hidden)
        if !trace.isEmpty { trace[trace.count - 1] = hidden }

        var logits = embedTokens.asLinear(hidden)
        if c.finalLogitSoftcap > 0 {
            logits = tanh(logits / c.finalLogitSoftcap) * c.finalLogitSoftcap
        }
        return logits
    }
}

/// Building the unified decoder and reading its configuration.
public extension NFKMLXGemmaLanguage {

    /// Builds a unified decoder network.
    static func makeUnifiedNet(_ configuration: NFKMLXGemmaConfiguration) -> NFKMLXGemma4UnifiedNet {
        NFKMLXGemma4UnifiedNet(configuration)
    }

    /// Reads a released `gemma4_unified_text` `config.json` (the decoder sits under `text_config` in a
    /// multimodal release) into a configuration.
    ///
    /// @discussion The unified decoder is a different architecture from the E-series, so its own model
    /// type is required. The full-attention layers run a wider head (`global_head_dim`) and the
    /// proportional rotary; the sliding layers run their own head width and a plain rotary.
    static func unifiedConfiguration(fromHuggingFace url: URL) throws -> NFKMLXGemmaConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        let text = (json["text_config"] as? [String: Any]) ?? json
        let kind = (text["model_type"] as? String) ?? (json["model_type"] as? String) ?? ""
        guard kind == "gemma4_unified_text" || kind == "gemma4_unified" else {
            throw NFKMLXError.unsupportedConfiguration("this reads a gemma4 unified text decoder, not \(kind)")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (text[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (text[key] as? NSNumber)?.floatValue ?? fallback }

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
        let hidden = integer("hidden_size", 3840)
        let heads = integer("num_attention_heads", 16)
        return NFKMLXGemmaConfiguration(
            hiddenSize: hidden,
            layerCount: integer("num_hidden_layers", 48),
            intermediateSize: integer("intermediate_size", 15360),
            vocabularySize: integer("vocab_size", 262_144),
            rmsEpsilon: real("rms_norm_eps", 1e-6),
            headCount: heads,
            keyValueHeadCount: integer("num_key_value_heads", heads),
            headDimensions: integer("head_dim", hidden / max(heads, 1)),
            globalHeadDimensions: integer("global_head_dim", 512),
            slidingWindow: integer("sliding_window", 1024),
            ropeTheta: theta, globalRopeTheta: globalTheta, globalPartialRotaryFactor: globalFactor,
            perLayerInputSize: 0,
            sharedKeyValueLayers: integer("num_kv_shared_layers", 0),
            finalLogitSoftcap: real("final_logit_softcapping", 0),
            layerTypes: (text["layer_types"] as? [String])?
                .compactMap { NFKMLXGemmaAttentionKind(rawValue: $0) })
    }

    /// Loads the unified decoder from a released directory, taking only the language model's tensors.
    static func loadUnifiedWeights(into net: NFKMLXGemma4UnifiedNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let prefix = "model.language_model."
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision) {
            $0.hasPrefix(prefix) ? String($0.dropFirst(prefix.count)) : nil
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
