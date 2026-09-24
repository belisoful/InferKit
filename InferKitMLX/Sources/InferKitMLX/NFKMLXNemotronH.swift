//
//  NFKMLXNemotronH.swift
//  InferKitMLX
//
//  The Nemotron-H decoder (`NemotronHForCausalLM`, NVIDIA, NVIDIA Open Model License), the hybrid
//  Mamba-attention decoder behind Nemotron Nano 2. Its layers interleave three mixers, one per block:
//  a Mamba-2 selective-scan (reused verbatim from `NFKMLXMamba2Mixer`), grouped-query attention with
//  no positional embedding (NoPE), and a ReLU-squared dense feed-forward. A `hybrid_override_pattern`
//  string names the mixer at each depth.
//
//  Unlike Granite 4.0-H, the feed-forward is its own block rather than a per-block companion to the
//  mixer, there are no scalar multipliers on the embedding, residual, or logits, and each block carries
//  a single pre-normalization: `residual + mixer(norm(residual))`. The Mamba mixer's gated output norm
//  normalizes within `n_groups` groups (HF's `Zamba2RMSNormGated`), the one numeric departure from the
//  ungrouped Codestral/Granite mixer.
//
//  Reference: HF `transformers` `NemotronHForCausalLM` (`modeling_nemotron_h.py`).
//
//  The module tree mirrors the checkpoint's own nesting (`model.embeddings`, `model.layers.N.norm`,
//  `model.layers.N.mixer.*`, `model.norm_f`, `lm_head`) with REAL submodules, not dotted keys, so an
//  unflattened weight routes in (the SigLIP2 / Mamba keys-split-on-dot lesson).
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

/// Which mixer sits at a given depth. The raw values are the checkpoint's `layers_block_type` strings.
public enum NFKMLXNemotronLayerKind: String, Sendable {
    case mamba = "linear_attention"
    case attention = "full_attention"
    case mlp
    case moe
}

/// The geometry of a Nemotron-H decoder.
public struct NFKMLXNemotronHConfiguration: Sendable {
    public var hiddenSize: Int
    public var vocabularySize: Int
    public var rmsEpsilon: Float

    // Attention layers (grouped-query, no positional embedding, standard scale).
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int

    // The Mamba-2 mixer.
    public var mambaHeadCount: Int
    public var mambaHeadDimensions: Int
    public var mambaGroupCount: Int
    public var mambaStateSize: Int
    public var mambaConvolutionKernel: Int
    public var mambaConvolutionBias: Bool
    public var mambaProjectionBias: Bool
    public var timeStepMinimum: Float

    // The ReLU-squared dense feed-forward.
    public var intermediateSize: Int
    public var mlpBias: Bool

    /// One entry per layer, from `hybrid_override_pattern`.
    public var layerTypes: [NFKMLXNemotronLayerKind]

    public var layerCount: Int { layerTypes.count }

    public init(hiddenSize: Int = 4480, vocabularySize: Int = 131_072, rmsEpsilon: Float = 1e-5,
                headCount: Int = 40, keyValueHeadCount: Int = 8, headDimensions: Int = 128,
                mambaHeadCount: Int = 128, mambaHeadDimensions: Int = 80, mambaGroupCount: Int = 8,
                mambaStateSize: Int = 128, mambaConvolutionKernel: Int = 4,
                mambaConvolutionBias: Bool = true, mambaProjectionBias: Bool = false,
                timeStepMinimum: Float = 0.001, intermediateSize: Int = 15680, mlpBias: Bool = false,
                layerTypes: [NFKMLXNemotronLayerKind]) {
        self.hiddenSize = hiddenSize
        self.vocabularySize = vocabularySize
        self.rmsEpsilon = rmsEpsilon
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.mambaHeadCount = mambaHeadCount
        self.mambaHeadDimensions = mambaHeadDimensions
        self.mambaGroupCount = mambaGroupCount
        self.mambaStateSize = mambaStateSize
        self.mambaConvolutionKernel = mambaConvolutionKernel
        self.mambaConvolutionBias = mambaConvolutionBias
        self.mambaProjectionBias = mambaProjectionBias
        self.timeStepMinimum = timeStepMinimum
        self.intermediateSize = intermediateSize
        self.mlpBias = mlpBias
        self.layerTypes = layerTypes
    }

    /// The geometry the reused `NFKMLXMamba2Mixer` reads for a Mamba layer. The gated output norm groups
    /// by `n_groups`, and the discretization step clamps its lower bound at `time_step_min`.
    var mambaConfiguration: NFKMLXMamba2Configuration {
        NFKMLXMamba2Configuration(
            hiddenSize: hiddenSize, layerCount: layerCount, vocabularySize: vocabularySize,
            rmsEpsilon: rmsEpsilon, intermediateSize: mambaHeadCount * mambaHeadDimensions,
            headCount: mambaHeadCount, headDimensions: mambaHeadDimensions, stateSize: mambaStateSize,
            groupCount: mambaGroupCount, convolutionKernel: mambaConvolutionKernel,
            useConvolutionBias: mambaConvolutionBias, useProjectionBias: mambaProjectionBias,
            timeStepLowerLimit: timeStepMinimum, timeStepUpperLimit: .infinity,
            gatedNormGroups: mambaGroupCount)
    }

    /// The released `nvidia/NVIDIA-Nemotron-Nano-9B-v2` pattern.
    public static let nano9B = NFKMLXNemotronHConfiguration(
        layerTypes: NFKMLXNemotronHConfiguration.layerKinds(
            fromPattern: "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"))

    /// Parses a `hybrid_override_pattern` string into per-layer kinds. `M` mamba, `-` mlp, `*` attention,
    /// `E` mixture of experts.
    public static func layerKinds(fromPattern pattern: String) -> [NFKMLXNemotronLayerKind] {
        pattern.map {
            switch $0 {
            case "M": return .mamba
            case "*": return .attention
            case "E": return .moe
            default: return .mlp
            }
        }
    }

    /// Reads a Nemotron-H geometry from a Hugging Face `config.json` dictionary.
    public static func configuration(fromHuggingFace json: [String: Any]) throws
        -> NFKMLXNemotronHConfiguration {
        let modelType = json["model_type"] as? String
        guard modelType == "nemotron_h" else {
            throw NFKMLXError.unsupportedConfiguration(
                "expected model_type nemotron_h, found \(modelType ?? "nil")")
        }
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? Int) ?? fallback }
        func float(_ key: String, _ fallback: Float) -> Float {
            if let d = json[key] as? Double { return Float(d) }
            if let i = json[key] as? Int { return Float(i) }
            return fallback
        }
        let hidden = int("hidden_size", 4480)
        let heads = int("num_attention_heads", 40)
        let kinds: [NFKMLXNemotronLayerKind]
        if let explicit = json["layers_block_type"] as? [String] {
            kinds = explicit.map { NFKMLXNemotronLayerKind(rawValue: $0) ?? .mlp }
        } else if let pattern = json["hybrid_override_pattern"] as? String {
            kinds = layerKinds(fromPattern: pattern)
        } else {
            throw NFKMLXError.unsupportedConfiguration(
                "nemotron_h config has neither layers_block_type nor hybrid_override_pattern")
        }
        let groups = (json["n_groups"] as? Int) ?? (json["mamba_num_groups"] as? Int) ?? 8
        return NFKMLXNemotronHConfiguration(
            hiddenSize: hidden,
            vocabularySize: int("vocab_size", 131_072),
            rmsEpsilon: float("layer_norm_epsilon", 1e-5),
            headCount: heads,
            keyValueHeadCount: int("num_key_value_heads", 8),
            headDimensions: (json["head_dim"] as? Int) ?? (hidden / heads),
            mambaHeadCount: int("mamba_num_heads", 128),
            mambaHeadDimensions: int("mamba_head_dim", 80),
            mambaGroupCount: groups,
            mambaStateSize: (json["ssm_state_size"] as? Int) ?? (json["mamba_state_dim"] as? Int) ?? 128,
            mambaConvolutionKernel: int("conv_kernel", 4),
            mambaConvolutionBias: (json["use_conv_bias"] as? Bool) ?? true,
            mambaProjectionBias: (json["use_bias"] as? Bool) ?? false,
            timeStepMinimum: float("time_step_min", 0.001),
            intermediateSize: int("intermediate_size", 15680),
            mlpBias: (json["mlp_bias"] as? Bool) ?? false,
            layerTypes: kinds)
    }
}

/// Standard RMS normalization, `weight · x · rsqrt(mean(x²) + eps)`.
final class NFKNemotronRMSNorm: Module, UnaryLayer {
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

/// Grouped-query attention with no positional embedding (NoPE) and the standard `1/√headDim` scale.
final class NFKNemotronAttention: Module, UnaryLayer {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    let headCount: Int
    let keyValueHeadCount: Int
    let headDimensions: Int
    let scale: Float

    init(_ config: NFKMLXNemotronHConfiguration) {
        headCount = config.headCount
        keyValueHeadCount = config.keyValueHeadCount
        headDimensions = config.headDimensions
        scale = 1.0 / Float(headDimensions).squareRoot()
        _queryProjection.wrappedValue = Linear(config.hiddenSize, headCount * headDimensions, bias: false)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, keyValueHeadCount * headDimensions, bias: false)
        _valueProjection.wrappedValue = Linear(config.hiddenSize, keyValueHeadCount * headDimensions, bias: false)
        _outputProjection.wrappedValue = Linear(headCount * headDimensions, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let batch = hidden.dim(0), length = hidden.dim(1)
        // No rotary: Nemotron-H uses NoPE.
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

/// The dense feed-forward: `down_proj(relu(up_proj(x))²)`. Ungated, unlike the gated-linear MLPs the
/// language and Granite decoders use.
final class NFKNemotronMLP: Module, UnaryLayer {
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(_ config: NFKMLXNemotronHConfiguration) {
        _upProjection.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        _downProjection.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let lifted = relu(upProjection(x))
        return downProjection(lifted * lifted)
    }
}

/// One Nemotron-H block: a single pre-normalization, one mixer, and a plain residual add. The mixer is
/// a Mamba-2 scan, NoPE attention, or the ReLU-squared feed-forward, all keyed `mixer` in the checkpoint.
final class NFKMLXNemotronBlock: Module {
    @ModuleInfo(key: "norm") var norm: NFKNemotronRMSNorm
    @ModuleInfo(key: "mixer") var mixer: Module & UnaryLayer

    init(_ config: NFKMLXNemotronHConfiguration, kind: NFKMLXNemotronLayerKind) {
        _norm.wrappedValue = NFKNemotronRMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
        switch kind {
        case .mamba:
            _mixer.wrappedValue = NFKMLXMamba2Mixer(config.mambaConfiguration)
        case .attention:
            _mixer.wrappedValue = NFKNemotronAttention(config)
        case .mlp, .moe:
            _mixer.wrappedValue = NFKNemotronMLP(config)
        }
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        hidden + mixer(norm(hidden))
    }
}

/// The Nemotron-H backbone: the token embedding, the hybrid block stack, and the final normalization.
final class NFKMLXNemotronModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKMLXNemotronBlock]
    @ModuleInfo(key: "norm_f") var finalNorm: NFKNemotronRMSNorm

    init(_ config: NFKMLXNemotronHConfiguration) {
        _embeddings.wrappedValue = Embedding(embeddingCount: config.vocabularySize,
                                             dimensions: config.hiddenSize)
        _layers.wrappedValue = config.layerTypes.map { NFKMLXNemotronBlock(config, kind: $0) }
        _finalNorm.wrappedValue = NFKNemotronRMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
        super.init()
    }
}

/// The Nemotron-H decoder: the embedding, the hybrid block stack, the final normalization, and the
/// untied output projection. No scalar multipliers scale any stage.
public final class NFKMLXNemotronHNet: Module {
    let config: NFKMLXNemotronHConfiguration

    @ModuleInfo(key: "model") var model: NFKMLXNemotronModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: NFKMLXNemotronHConfiguration) {
        self.config = config
        _model.wrappedValue = NFKMLXNemotronModel(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    /// The state entering each block: `[embedding, block0, … , blockN-1]`, before the final
    /// normalization, for per-seam parity isolation.
    public func blockStates(_ tokens: MLXArray) -> [MLXArray] {
        var hidden = model.embeddings(tokens)
        var states = [hidden]
        for layer in model.layers {
            hidden = layer(hidden)
            states.append(hidden)
        }
        return states
    }

    public func hiddenStates(_ tokens: MLXArray) -> MLXArray {
        model.finalNorm(blockStates(tokens).last!)
    }

    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        lmHead(hiddenStates(tokens))
    }
}

/// Builders and the released-weight loader for the Nemotron-H decoder.
@objc(NFKMLXNemotronH)
public final class NFKMLXNemotronH: NSObject {
    public static func makeNet(_ config: NFKMLXNemotronHConfiguration) -> NFKMLXNemotronHNet {
        NFKMLXNemotronHNet(config)
    }

    public static func configuration(fromDirectory directory: URL) throws
        -> NFKMLXNemotronHConfiguration {
        let url = directory.appendingPathComponent("config.json")
        let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] ?? [:]
        return try NFKMLXNemotronHConfiguration.configuration(fromHuggingFace: json)
    }

    /// Loads a released Nemotron-H decoder from its directory. The released checkpoint carries the
    /// original `backbone.*` naming (the module tree here follows the transformers-integrated `model.*`
    /// naming, which the tiny oracle records), so the backbone prefix is remapped on load. The depthwise
    /// convolution weight is squeezed `[C, 1, K]` → `[C, K]`, and any multi-token prediction (`mtp.*`)
    /// tensors a release carries are skipped, as transformers skips them.
    public static func loadWeights(into net: NFKMLXNemotronHNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .checkpoint) throws {
        let read = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision) { key in
            if key.hasPrefix("mtp.") || key.hasPrefix("model.mtp") { return nil }
            return key.hasPrefix("backbone.") ? "model." + key.dropFirst("backbone.".count) : key
        }
        let merged = read.map { name, value in
            (name, name.hasSuffix("conv1d.weight") && value.ndim == 3
                 ? value.reshaped([value.dim(0), value.dim(2)]) : value)
        }
        try NFKMLXWeights.apply(merged, to: net)
    }
}

/// The reused Mamba-2 mixer is `UnaryLayer`-shaped already; the conformance lets a block hold it beside
/// the attention and feed-forward mixers under one `mixer` key.
extension NFKMLXMamba2Mixer: UnaryLayer {}
