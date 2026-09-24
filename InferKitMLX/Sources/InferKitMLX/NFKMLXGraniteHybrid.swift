//
//  NFKMLXGraniteHybrid.swift
//  InferKitMLX
//
//  The Granite 4.0-H decoder (`GraniteMoeHybridForCausalLM`, IBM, Apache-2.0), a hybrid Mamba-attention
//  decoder. Most layers are a Mamba-2 selective-scan mixer (reused verbatim from `NFKMLXMamba2Mixer`),
//  and the few the release's `layer_types` names are grouped-query attention with NO positional
//  embedding (NoPE) and a custom attention scale. The feed-forward is a gated-linear shared MLP, with
//  a routed mixture of experts beside it on the MoE sizes. Granite's scalar multipliers scale the embedding, each residual add, the
//  attention logits, and the output logits.
//
//  Built on the SSM mixer the Codestral-Mamba port introduced, so one selective-scan serves both. The
//  dense sizes (h-350m, h-1b) carry the shared MLP alone; the MoE sizes (h-tiny, h-small) add a routed
//  mixture of experts beside the shared MLP, its two summed outputs forming the feed-forward.
//
//  Reference: HF `transformers` `GraniteMoeHybridForCausalLM`.
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

/// Which mixer sits at a given depth.
public enum NFKMLXGraniteLayerKind: String, Sendable {
    case mamba
    case attention
}

/// The geometry of a Granite 4.0-H decoder.
public struct NFKMLXGraniteHybridConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var vocabularySize: Int
    public var rmsEpsilon: Float
    public var tiesWordEmbeddings: Bool

    // Attention layers (grouped-query, no positional embedding).
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int

    // The Mamba-2 mixer.
    public var mambaHeadCount: Int
    public var mambaHeadDimensions: Int
    public var mambaGroupCount: Int
    public var mambaStateSize: Int
    public var mambaConvolutionKernel: Int
    public var mambaExpand: Int
    public var mambaConvolutionBias: Bool
    public var mambaProjectionBias: Bool

    // Feed-forward: a gated-linear shared MLP, and optional routed experts beside it.
    public var sharedIntermediateSize: Int
    public var expertCount: Int
    public var expertsPerToken: Int
    public var expertIntermediateSize: Int

    // Granite's scalar multipliers.
    public var embeddingMultiplier: Float
    public var residualMultiplier: Float
    public var attentionMultiplier: Float
    public var logitsScaling: Float

    /// One entry per layer.
    public var layerTypes: [NFKMLXGraniteLayerKind]

    public init(hiddenSize: Int = 1536, layerCount: Int = 40, vocabularySize: Int = 100_352,
                rmsEpsilon: Float = 1e-5, tiesWordEmbeddings: Bool = true,
                headCount: Int = 12, keyValueHeadCount: Int = 4, headDimensions: Int = 128,
                mambaHeadCount: Int = 48, mambaHeadDimensions: Int = 64, mambaGroupCount: Int = 1,
                mambaStateSize: Int = 128, mambaConvolutionKernel: Int = 4, mambaExpand: Int = 2,
                mambaConvolutionBias: Bool = true, mambaProjectionBias: Bool = false,
                sharedIntermediateSize: Int = 4096, expertCount: Int = 0, expertsPerToken: Int = 0,
                expertIntermediateSize: Int = 0,
                embeddingMultiplier: Float = 12, residualMultiplier: Float = 0.22,
                attentionMultiplier: Float = 0.0078125, logitsScaling: Float = 6,
                layerTypes: [NFKMLXGraniteLayerKind]? = nil) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.vocabularySize = vocabularySize
        self.rmsEpsilon = rmsEpsilon
        self.tiesWordEmbeddings = tiesWordEmbeddings
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.mambaHeadCount = mambaHeadCount
        self.mambaHeadDimensions = mambaHeadDimensions
        self.mambaGroupCount = mambaGroupCount
        self.mambaStateSize = mambaStateSize
        self.mambaConvolutionKernel = mambaConvolutionKernel
        self.mambaExpand = mambaExpand
        self.mambaConvolutionBias = mambaConvolutionBias
        self.mambaProjectionBias = mambaProjectionBias
        self.sharedIntermediateSize = sharedIntermediateSize
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.expertIntermediateSize = expertIntermediateSize
        self.embeddingMultiplier = embeddingMultiplier
        self.residualMultiplier = residualMultiplier
        self.attentionMultiplier = attentionMultiplier
        self.logitsScaling = logitsScaling
        self.layerTypes = layerTypes
            ?? (0 ..< layerCount).map { $0 % 10 == 5 ? .attention : .mamba }
    }

    /// The geometry the reused `NFKMLXMamba2Mixer` reads for a Mamba layer.
    var mambaConfiguration: NFKMLXMamba2Configuration {
        NFKMLXMamba2Configuration(
            hiddenSize: hiddenSize, layerCount: layerCount, vocabularySize: vocabularySize,
            rmsEpsilon: rmsEpsilon, intermediateSize: mambaExpand * hiddenSize,
            headCount: mambaHeadCount, headDimensions: mambaHeadDimensions, stateSize: mambaStateSize,
            groupCount: mambaGroupCount, convolutionKernel: mambaConvolutionKernel,
            useConvolutionBias: mambaConvolutionBias, useProjectionBias: mambaProjectionBias,
            tiesWordEmbeddings: tiesWordEmbeddings)
    }

    var hasExperts: Bool { expertCount > 0 }

    /// The released `ibm-granite/granite-4.0-h-1b` (dense hybrid): attention at layers 5, 15, 25, and 35.
    public static let h1B = NFKMLXGraniteHybridConfiguration(
        expertIntermediateSize: 4096, layerTypes: (0 ..< 40).map { $0 % 10 == 5 ? .attention : .mamba })

    /// Reads a Granite 4.0-H geometry from a Hugging Face `config.json` dictionary.
    public static func configuration(fromHuggingFace json: [String: Any]) throws
        -> NFKMLXGraniteHybridConfiguration {
        let modelType = json["model_type"] as? String
        guard modelType == "granitemoehybrid" else {
            throw NFKMLXError.unsupportedConfiguration(
                "expected model_type granitemoehybrid, found \(modelType ?? "nil")")
        }
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? Int) ?? fallback }
        func float(_ key: String, _ fallback: Float) -> Float {
            if let d = json[key] as? Double { return Float(d) }
            if let i = json[key] as? Int { return Float(i) }
            return fallback
        }
        let hidden = int("hidden_size", 1536)
        let heads = int("num_attention_heads", 12)
        let kinds = (json["layer_types"] as? [String])?.map {
            NFKMLXGraniteLayerKind(rawValue: $0) ?? .mamba
        }
        return NFKMLXGraniteHybridConfiguration(
            hiddenSize: hidden,
            layerCount: int("num_hidden_layers", 40),
            vocabularySize: int("vocab_size", 100_352),
            rmsEpsilon: float("rms_norm_eps", 1e-5),
            tiesWordEmbeddings: (json["tie_word_embeddings"] as? Bool) ?? true,
            headCount: heads,
            keyValueHeadCount: int("num_key_value_heads", 4),
            headDimensions: (json["head_dim"] as? Int) ?? (hidden / heads),
            mambaHeadCount: int("mamba_n_heads", 48),
            mambaHeadDimensions: int("mamba_d_head", 64),
            mambaGroupCount: int("mamba_n_groups", 1),
            mambaStateSize: int("mamba_d_state", 128),
            mambaConvolutionKernel: int("mamba_d_conv", 4),
            mambaExpand: int("mamba_expand", 2),
            mambaConvolutionBias: (json["mamba_conv_bias"] as? Bool) ?? true,
            mambaProjectionBias: (json["mamba_proj_bias"] as? Bool) ?? false,
            sharedIntermediateSize: int("shared_intermediate_size", 4096),
            expertCount: int("num_local_experts", 0),
            expertsPerToken: int("num_experts_per_tok", 0),
            expertIntermediateSize: int("intermediate_size", 0),
            embeddingMultiplier: float("embedding_multiplier", 12),
            residualMultiplier: float("residual_multiplier", 0.22),
            attentionMultiplier: float("attention_multiplier", 0.0078125),
            logitsScaling: float("logits_scaling", 6),
            layerTypes: kinds)
    }
}

/// Standard RMS normalization, `weight · x · rsqrt(mean(x²) + eps)`.
final class NFKGraniteRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        epsilon = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let variance = mean(f * f, axis: -1, keepDims: true)
        return weight * (f * rsqrt(variance + epsilon)).asType(x.dtype)
    }
}

/// Grouped-query attention with no positional embedding (NoPE) and Granite's custom logit scale.
final class NFKGraniteAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    let headCount: Int
    let keyValueHeadCount: Int
    let headDimensions: Int
    let scale: Float

    init(_ config: NFKMLXGraniteHybridConfiguration) {
        headCount = config.headCount
        keyValueHeadCount = config.keyValueHeadCount
        headDimensions = config.headDimensions
        scale = config.attentionMultiplier
        _queryProjection.wrappedValue = Linear(config.hiddenSize, headCount * headDimensions, bias: false)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, keyValueHeadCount * headDimensions, bias: false)
        _valueProjection.wrappedValue = Linear(config.hiddenSize, keyValueHeadCount * headDimensions, bias: false)
        _outputProjection.wrappedValue = Linear(headCount * headDimensions, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let batch = hidden.dim(0), length = hidden.dim(1)
        // No rotary: Granite 4.0-H uses NoPE.
        let q = queryProjection(hidden).reshaped([batch, length, headCount, headDimensions]).transposed(0, 2, 1, 3)
        let k = keyProjection(hidden).reshaped([batch, length, keyValueHeadCount, headDimensions]).transposed(0, 2, 1, 3)
        let v = valueProjection(hidden).reshaped([batch, length, keyValueHeadCount, headDimensions]).transposed(0, 2, 1, 3)
        let out: MLXArray
        if NFKReferenceRounding.isReduced(q) {
            out = NFKReferenceRounding.attention(queries: q, keys: k, values: v, scale: scale,
                                                 mask: NFKMLXLanguageNet.causalMask(length, offset: 0))
        } else {
            out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .causal)
        }
        return outputProjection(out.transposed(0, 2, 1, 3).reshaped([batch, length, headCount * headDimensions]))
    }
}

/// The gated-linear feed-forward: `output_linear(silu(a) · b)` where `a, b = input_linear(x).chunk(2)`.
final class NFKGraniteMLP: Module {
    @ModuleInfo(key: "input_linear") var inputLinear: Linear
    @ModuleInfo(key: "output_linear") var outputLinear: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _inputLinear.wrappedValue = Linear(hiddenSize, 2 * intermediateSize, bias: false)
        _outputLinear.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = inputLinear(x)
        let parts = split(projected, parts: 2, axis: -1)
        return outputLinear(NFKReferenceRounding.silu(parts[0]) * parts[1])
    }
}

/// Granite's top-k router: one linear scores every expert, and the softmax is taken over the chosen
/// `k` logits (not over all experts). A real submodule named `router` holding the `layer` linear so
/// the checkpoint's `block_sparse_moe.router.layer.weight` routes in.
final class NFKGraniteRouter: Module {
    @ModuleInfo(key: "layer") var layer: Linear

    init(hiddenSize: Int, expertCount: Int) {
        _layer.wrappedValue = Linear(hiddenSize, expertCount, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { layer(x) }
}

/// The routed mixture of experts on the MoE sizes: `input_linear` (a fused gate+up projection) →
/// SwiGLU → `output_linear`, weighted by the router's softmax over the chosen experts and summed.
///
/// @discussion The experts are stacked per role into one `[experts, out, in]` tensor apiece, exactly
/// the released `block_sparse_moe.input_linear.weight` / `output_linear.weight` layout, and run
/// through one gathered matrix multiplication over the chosen experts (`NFKLMSwitchLinear`, shared
/// with the language-model MoE). `input_linear` emits `2 · intermediate`, split into the gate and the
/// lift; the router scores every expert, and the softmax over the top-`k` raw logits weights each
/// chosen expert's output.
final class NFKMLXGraniteMoE: Module {
    @ModuleInfo(key: "router") var router: NFKGraniteRouter
    @ModuleInfo(key: "input_linear") var inputLinear: NFKLMExpertLinear
    @ModuleInfo(key: "output_linear") var outputLinear: NFKLMExpertLinear
    let expertsPerToken: Int

    init(_ config: NFKMLXGraniteHybridConfiguration) {
        expertsPerToken = config.expertsPerToken
        _router.wrappedValue = NFKGraniteRouter(hiddenSize: config.hiddenSize, expertCount: config.expertCount)
        _inputLinear.wrappedValue = NFKLMSwitchLinear(experts: config.expertCount, inputSize: config.hiddenSize,
                                                      outputSize: 2 * config.expertIntermediateSize)
        _outputLinear.wrappedValue = NFKLMSwitchLinear(experts: config.expertCount,
                                                       inputSize: config.expertIntermediateSize,
                                                       outputSize: config.hiddenSize)
        super.init()
    }

    /// Replaces both projections with paged ones reading from `store`, filed under `path`, this
    /// module's own key path. Returns the groups the store must hold.
    func page(from store: NFKMLXExpertStore, path: String) throws
        -> [(group: String, experts: Int, parts: [String])] {
        var groups = [(group: String, experts: Int, parts: [String])]()
        for (name, layer) in [("input_linear", inputLinear), ("output_linear", outputLinear)] {
            let group = "\(path).\(name)"
            let paged = NFKLMPagedSwitchLinear(pager: NFKMLXExpertPager(store: store, group: group),
                                               experts: layer.expertCount, outputSize: layer.outputSize,
                                               inputSize: layer.inputSize)
            try update(modules: ModuleChildren.unflattened([(name, paged)]), verify: .noUnusedKeys)
            groups.append((group, layer.expertCount, ["weight"]))
        }
        return groups
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = router(x).asType(.float32)
        let chosen = argPartition(-logits, kth: expertsPerToken - 1, axis: -1)[.ellipsis, 0 ..< expertsPerToken]
        let gates = softmax(takeAlong(logits, chosen, axis: -1), axis: -1, precise: true).asType(x.dtype)
        let expanded = x.expandedDimensions(axes: [-2, -3])
        let fused = inputLinear(expanded, experts: chosen)
        let parts = split(fused, parts: 2, axis: -1)
        let activated = NFKReferenceRounding.silu(parts[0]) * parts[1]
        let expertOutputs = outputLinear(activated, experts: chosen).squeezed(axis: -2)
        return (expertOutputs * gates.expandedDimensions(axis: -1)).sum(axis: -2)
    }
}

/// One Granite 4.0-H block: a Mamba or attention mixer, then the feed-forward, each added back through
/// the residual multiplier. The feed-forward is the shared MLP alone on the dense sizes, and the
/// routed experts summed with the shared MLP on the MoE sizes.
final class NFKMLXGraniteBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: NFKGraniteRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: NFKGraniteRMSNorm
    @ModuleInfo(key: "mamba") var mamba: NFKMLXMamba2Mixer?
    @ModuleInfo(key: "self_attn") var attention: NFKGraniteAttention?
    @ModuleInfo(key: "block_sparse_moe") var moe: NFKMLXGraniteMoE?
    @ModuleInfo(key: "shared_mlp") var sharedMLP: NFKGraniteMLP
    let residualMultiplier: Float

    init(_ config: NFKMLXGraniteHybridConfiguration, kind: NFKMLXGraniteLayerKind) {
        residualMultiplier = config.residualMultiplier
        _inputNorm.wrappedValue = NFKGraniteRMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
        _postNorm.wrappedValue = NFKGraniteRMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
        switch kind {
        case .mamba:
            _mamba.wrappedValue = NFKMLXMamba2Mixer(config.mambaConfiguration)
            _attention.wrappedValue = nil
        case .attention:
            _attention.wrappedValue = NFKGraniteAttention(config)
            _mamba.wrappedValue = nil
        }
        _moe.wrappedValue = config.hasExperts ? NFKMLXGraniteMoE(config) : nil
        _sharedMLP.wrappedValue = NFKGraniteMLP(hiddenSize: config.hiddenSize,
                                               intermediateSize: config.sharedIntermediateSize)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var residual = hidden
        var h = inputNorm(hidden)
        if let mamba {
            h = mamba(h)
        } else if let attention {
            h = attention(h)
        }
        h = residual + NFKReferenceRounding.scaled(h, by: residualMultiplier)
        residual = h
        let normed = postNorm(h)
        h = moe.map { $0(normed) + sharedMLP(normed) } ?? sharedMLP(normed)
        return residual + NFKReferenceRounding.scaled(h, by: residualMultiplier)
    }
}

/// The Granite backbone: the token embedding, the hybrid block stack, and the final normalization. A
/// real submodule so the checkpoint's `model.*` keys route into it (a dotted `@ModuleInfo` key flattens
/// to the right name but cannot receive an unflattened weight — the SigLIP2 / Mamba lesson).
final class NFKMLXGraniteModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKMLXGraniteBlock]
    @ModuleInfo(key: "norm") var norm: NFKGraniteRMSNorm

    init(_ config: NFKMLXGraniteHybridConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize,
                                              dimensions: config.hiddenSize)
        _layers.wrappedValue = config.layerTypes.map { NFKMLXGraniteBlock(config, kind: $0) }
        _norm.wrappedValue = NFKGraniteRMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
        super.init()
    }
}

/// The Granite 4.0-H decoder: embedding (scaled), the hybrid block stack, a final normalization, and
/// the tied output projection (scaled).
public final class NFKMLXGraniteHybridNet: Module {
    let config: NFKMLXGraniteHybridConfiguration

    @ModuleInfo(key: "model") var model: NFKMLXGraniteModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// The routed experts of a paged load, which the mixture layers read in place of parameters; nil
    /// where every expert is resident. Introduced in InferKit 0.4.0.
    public internal(set) var expertStore: NFKMLXExpertStore?

    public init(_ config: NFKMLXGraniteHybridConfiguration) {
        self.config = config
        _model.wrappedValue = NFKMLXGraniteModel(config)
        _lmHead.wrappedValue = config.tiesWordEmbeddings
            ? nil : Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    /// The state entering each block: `[embedding·multiplier, block0, … , blockN-1]`, before the final
    /// normalization, for per-seam parity isolation.
    public func blockStates(_ tokens: MLXArray) -> [MLXArray] {
        var hidden = model.embedTokens(tokens) * config.embeddingMultiplier
        var states = [hidden]
        for layer in model.layers {
            hidden = layer(hidden)
            states.append(hidden)
        }
        return states
    }

    public func hiddenStates(_ tokens: MLXArray) -> MLXArray {
        model.norm(blockStates(tokens).last!)
    }

    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let hidden = hiddenStates(tokens)
        let logits = lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
        return logits / config.logitsScaling
    }
}

/// Builders and the released-weight loader for the Granite 4.0-H decoder.
@objc(NFKMLXGraniteHybrid)
public final class NFKMLXGraniteHybrid: NSObject {
    public static func makeNet(_ config: NFKMLXGraniteHybridConfiguration) -> NFKMLXGraniteHybridNet {
        NFKMLXGraniteHybridNet(config)
    }

    public static func configuration(fromDirectory directory: URL) throws
        -> NFKMLXGraniteHybridConfiguration {
        let url = directory.appendingPathComponent("config.json")
        let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] ?? [:]
        return try NFKMLXGraniteHybridConfiguration.configuration(fromHuggingFace: json)
    }

    /// Loads a released Granite 4.0-H decoder from its directory. The checkpoint keys mirror this
    /// module's own; the depthwise convolution weight is squeezed `[C, 1, K]` → `[C, K]`, and a tied
    /// release ships no `lm_head`.
    public static func loadWeights(into net: NFKMLXGraniteHybridNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .checkpoint) throws {
        try loadWeights(into: net, fromDirectory: directory, precision: precision, skipping: { _ in false })
    }

    /// Loads a released decoder with its routed experts held as `residency` plans them. A mixture size
    /// stores `input_linear` and `output_linear` stacked in the module's own layout, so an expert pages
    /// as the release stores it; a dense size loads resident under every residency. Introduced in
    /// InferKit 0.4.0.
    public static func loadWeights(into net: NFKMLXGraniteHybridNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision, residency: NFKMLXResidency) throws {
        let weight = ".weight"
        net.expertStore = try NFKMLXExpertInventory.load(
            directory: directory, precision: precision, residency: residency,
            classify: { key, entry in
                guard key.hasSuffix(weight), entry.shape.count == 3,
                      key.hasSuffix(".block_sparse_moe.input_linear" + weight)
                        || key.hasSuffix(".block_sparse_moe.output_linear" + weight) else { return [] }
                return [NFKMLXExpertSlice(group: String(key.dropLast(weight.count)), part: "weight", expert: nil)]
            },
            install: { store in
                try net.model.layers.enumerated().flatMap { index, block in
                    try block.moe?.page(from: store, path: "model.layers.\(index).block_sparse_moe") ?? []
                }
            },
            load: { try loadWeights(into: net, fromDirectory: directory, precision: precision, skipping: $0) })
    }

    static func loadWeights(into net: NFKMLXGraniteHybridNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision, skipping skipped: (String) -> Bool) throws {
        let tied = net.lmHead == nil
        let read = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision) {
            (tied && $0.hasPrefix("lm_head.")) || skipped($0) ? nil : $0
        }
        let merged = read.map { name, value in
            (name, name.hasSuffix("conv1d.weight") && value.ndim == 3
                 ? value.reshaped([value.dim(0), value.dim(2)]) : value)
        }
        try NFKMLXWeights.apply(merged, to: net)
    }
}
