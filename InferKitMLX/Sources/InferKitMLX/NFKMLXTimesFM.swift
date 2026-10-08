//
//  NFKMLXTimesFM.swift
//  InferKitMLX
//
//  TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`, Apache-2.0), Google's decoder-only time-series
//  foundation model, ported into MLXNN from the official `google-research/timesfm` PyTorch module
//  (`TimesFM_2p5_200M_torch_module`) and its forecasting wrapper. A series is cut into 32-step patches,
//  each normalized by the running statistics of the patches so far (RevIN), embedded by a residual
//  block, and passed through 20 causal transformer layers; the last patch's output is a 128-step
//  forecast at ten quantile heads, and a continuous quantile head spans 1024 steps. Horizons past 128
//  decode autoregressively over a key-value cache.
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// The geometry of a TimesFM 2.5 release.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTimesFMConfiguration: Sendable, Equatable {
    public var patchLength: Int = 32
    public var outputPatchLength: Int = 128
    public var outputQuantileLength: Int = 1024
    public var quantiles: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    /// The output channel of the point forecast (`decode_index`): channel 0 is the mean and channels
    /// 1…9 the quantiles, so 5 is the median.
    public var decodeIndex: Int = 5
    public var hiddenSize: Int = 1280
    public var intermediateSize: Int = 1280
    public var numLayers: Int = 20
    public var numHeads: Int = 16
    public var rmsNormEps: Float = 1e-6
    /// The most steps of context plus horizon the model accepts.
    public var contextLimit: Int = 16384

    /// `google/timesfm-2.5-200m`.
    public static let v2_5 = NFKMLXTimesFMConfiguration()

    public init() {}

    /// Reads a release's `config.json`, in either the official (`timesfm`) or the transformers
    /// (`timesfm2_5`) spelling.
    public init(configurationURL: URL) throws {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: configurationURL)) as? [String: Any] ?? [:]
        let modelType = json["model_type"] as? String ?? "timesfm"
        guard ["timesfm", "timesfm2_5"].contains(modelType) else {
            throw NFKMLXError.unsupportedConfiguration("\(configurationURL.lastPathComponent) describes a \(modelType) "
                                                      + "model, not TimesFM 2.5")
        }
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        patchLength = int("patch_length", patchLength)
        outputPatchLength = int("horizon_length", outputPatchLength)
        outputQuantileLength = int("quantile_horizon_length", int("output_quantile_len", outputQuantileLength))
        quantiles = (json["quantiles"] as? [NSNumber])?.map(\.floatValue) ?? quantiles
        decodeIndex = int("decode_index", decodeIndex)
        hiddenSize = int("hidden_size", hiddenSize)
        intermediateSize = int("intermediate_size", intermediateSize)
        numLayers = int("num_hidden_layers", numLayers)
        numHeads = int("num_attention_heads", numHeads)
        rmsNormEps = Float((json["rms_norm_eps"] as? NSNumber)?.doubleValue ?? Double(rmsNormEps))
        contextLimit = int("context_length", contextLimit)
        if let headDimension = json["head_dim"] as? NSNumber, headDimension.intValue * numHeads != hiddenSize {
            throw NFKMLXError.unsupportedConfiguration("head_dim \(headDimension) × \(numHeads) heads is not the "
                                                      + "hidden size \(hiddenSize)")
        }
    }

    /// Output channels per step: the mean and each quantile.
    public var outputChannels: Int { quantiles.count + 1 }
    public var headDimension: Int { hiddenSize / numHeads }
}

// MARK: - Layers

/// `dense.ResidualBlock`: `output(swish(hidden(x))) + residual(x)`.
final class NFKTimesFMResidualBlock: Module {
    @ModuleInfo(key: "hidden_layer") var hidden: Linear
    @ModuleInfo(key: "output_layer") var output: Linear
    @ModuleInfo(key: "residual_layer") var residual: Linear

    init(input: Int, hidden: Int, output: Int, bias: Bool) {
        _hidden.wrappedValue = Linear(input, hidden, bias: bias)
        _output.wrappedValue = Linear(hidden, output, bias: bias)
        _residual.wrappedValue = Linear(input, output, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        output(silu(hidden(x))) + residual(x)
    }
}

/// `normalization.RMSNorm`: `x · rsqrt(mean(x²) + ε) · scale`, the scale used as stored.
final class NFKTimesFMRMSNorm: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray
    let eps: Float

    init(_ dimensions: Int, eps: Float) {
        self.eps = eps
        _scale.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: scale, eps: eps)
    }
}

/// `transformer.PerDimScale`, held on the attention as `per_dim_scale.per_dim_scale`.
final class NFKTimesFMPerDimScale: Module {
    @ParameterInfo(key: "per_dim_scale") var perDimScale: MLXArray

    init(_ dimensions: Int) {
        _perDimScale.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    /// `1.442695041 / √d · softplus(per_dim_scale)`, the only scaling the attention applies.
    var factor: MLXArray {
        softplus(perDimScale) * (1.442695041 / Float(perDimScale.dim(0)).squareRoot())
    }
}

/// The keys and values one layer has seen, and the index the next patch takes.
struct NFKTimesFMLayerCache {
    var keys: MLXArray
    var values: MLXArray
}

/// `transformer.MultiHeadAttention` in its unfused form (`fuse_qkv=False`: separate `query`, `key`, and
/// `value` projections), rotary positions applied before the query and key RMS norms, and the
/// per-dimension query scale. The current checkpoint stores the projections fused (`qkv_proj`); the
/// loader splits them, which leaves the forward unchanged and gives each projection its own layer, as a
/// LoRA over the transformers release adapts them.
final class NFKTimesFMAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "out") var out: Linear
    @ModuleInfo(key: "query_ln") var queryNorm: NFKTimesFMRMSNorm
    @ModuleInfo(key: "key_ln") var keyNorm: NFKTimesFMRMSNorm
    @ModuleInfo(key: "per_dim_scale") var perDimScale: NFKTimesFMPerDimScale
    let heads: Int
    let headDimension: Int

    init(_ c: NFKMLXTimesFMConfiguration) {
        heads = c.numHeads
        headDimension = c.headDimension
        _query.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _key.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _value.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _out.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = NFKTimesFMRMSNorm(c.headDimension, eps: 1e-6)
        _keyNorm.wrappedValue = NFKTimesFMRMSNorm(c.headDimension, eps: 1e-6)
        _perDimScale.wrappedValue = NFKTimesFMPerDimScale(c.headDimension)
        super.init()
    }

    /// The rotary embedding of `transformer.RotaryPositionalEmbedding`: timescales
    /// `10000^(2i / d)`, the first and second halves rotated as a pair.
    static func rotate(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let fraction = MLXArray(0 ..< half).asType(.float32) * (2 / Float(x.dim(-1)))
        let timescale = pow(MLXArray(Float(10000)), fraction)
        let angles = positions.reshaped([positions.dim(0), positions.dim(1), 1, 1]) / timescale
        let (sine, cosine) = (sin(angles), cos(angles))
        let first = x[.ellipsis, ..<half], second = x[.ellipsis, half...]
        return concatenated([first * cosine - second * sine, second * cosine + first * sine], axis: -1)
    }

    /// Attends `x` `[B, N, D]` at `positions` `[B, N]` over the cached keys and these, causally.
    func callAsFunction(_ x: MLXArray, positions: MLXArray, cache: inout NFKTimesFMLayerCache?) -> MLXArray {
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([batch, length, heads, headDimension]) }
        let queries = queryNorm(Self.rotate(split(query(x)), positions: positions)) * perDimScale.factor
        var keys = keyNorm(Self.rotate(split(key(x)), positions: positions))
        var values = split(value(x))
        if let previous = cache {
            keys = concatenated([previous.keys, keys], axis: 1)
            values = concatenated([previous.values, values], axis: 1)
        }
        cache = NFKTimesFMLayerCache(keys: keys, values: values)
        let offset = keys.dim(1) - length
        let queryIndex = MLXArray(0 ..< length).reshaped([length, 1]) + offset
        let keyIndex = MLXArray(0 ..< keys.dim(1)).reshaped([1, keys.dim(1)])
        let mask = MLX.where(queryIndex .>= keyIndex, MLXArray(Float(0)), MLXArray(-Float.infinity))
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries.transposed(0, 2, 1, 3), keys: keys.transposed(0, 2, 1, 3),
            values: values.transposed(0, 2, 1, 3), scale: 1, mask: .array(mask))
        return out(attended.transposed(0, 2, 1, 3).reshaped([batch, length, width]))
    }
}

/// `transformer.Transformer`: RMS norms before and after the attention and the feed-forward block.
final class NFKTimesFMLayer: Module {
    @ModuleInfo(key: "pre_attn_ln") var preAttentionNorm: NFKTimesFMRMSNorm
    @ModuleInfo(key: "attn") var attention: NFKTimesFMAttention
    @ModuleInfo(key: "post_attn_ln") var postAttentionNorm: NFKTimesFMRMSNorm
    @ModuleInfo(key: "pre_ff_ln") var preFeedForwardNorm: NFKTimesFMRMSNorm
    @ModuleInfo(key: "ff0") var ff0: Linear
    @ModuleInfo(key: "ff1") var ff1: Linear
    @ModuleInfo(key: "post_ff_ln") var postFeedForwardNorm: NFKTimesFMRMSNorm

    init(_ c: NFKMLXTimesFMConfiguration) {
        _preAttentionNorm.wrappedValue = NFKTimesFMRMSNorm(c.hiddenSize, eps: c.rmsNormEps)
        _attention.wrappedValue = NFKTimesFMAttention(c)
        _postAttentionNorm.wrappedValue = NFKTimesFMRMSNorm(c.hiddenSize, eps: c.rmsNormEps)
        _preFeedForwardNorm.wrappedValue = NFKTimesFMRMSNorm(c.hiddenSize, eps: c.rmsNormEps)
        _ff0.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _ff1.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        _postFeedForwardNorm.wrappedValue = NFKTimesFMRMSNorm(c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, cache: inout NFKTimesFMLayerCache?) -> MLXArray {
        let attended = postAttentionNorm(attention(preAttentionNorm(x), positions: positions, cache: &cache)) + x
        return postFeedForwardNorm(ff1(silu(ff0(preFeedForwardNorm(attended))))) + attended
    }
}

// MARK: - Network

/// The TimesFM 2.5 network, the official `TimesFM_2p5_200M_torch_module`. Module keys mirror the
/// official checkpoint (`tokenizer.*`, `stacked_xf.N.*`, `output_projection_point.*`,
/// `output_projection_quantiles.*`) in its unfused attention form (`stacked_xf.N.attn.query` / `key` /
/// `value`).
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXTimesFMNet: Module {
    @ModuleInfo(key: "tokenizer") var tokenizer: NFKTimesFMResidualBlock
    @ModuleInfo(key: "stacked_xf") var layers: [NFKTimesFMLayer]
    @ModuleInfo(key: "output_projection_point") var pointProjection: NFKTimesFMResidualBlock
    @ModuleInfo(key: "output_projection_quantiles") var quantileProjection: NFKTimesFMResidualBlock

    public let configuration: NFKMLXTimesFMConfiguration

    public init(_ configuration: NFKMLXTimesFMConfiguration = .v2_5) {
        self.configuration = configuration
        let c = configuration
        _tokenizer.wrappedValue = NFKTimesFMResidualBlock(input: 2 * c.patchLength, hidden: c.hiddenSize,
                                                          output: c.hiddenSize, bias: true)
        _layers.wrappedValue = (0 ..< c.numLayers).map { _ in NFKTimesFMLayer(c) }
        _pointProjection.wrappedValue = NFKTimesFMResidualBlock(input: c.hiddenSize, hidden: c.hiddenSize,
                                                                output: c.outputPatchLength * c.outputChannels, bias: false)
        _quantileProjection.wrappedValue = NFKTimesFMResidualBlock(input: c.hiddenSize, hidden: c.hiddenSize,
                                                                   output: c.outputQuantileLength * c.outputChannels,
                                                                   bias: false)
        super.init()
    }

    /// One pass of the module's `forward`: normalized patches `[B, N, patch]` and their masks (true where
    /// padded) in, the embeddings, the transformer's output, and both projections out. `caches` carries
    /// the key-value state between an autoregressive step and the next; `positions` are the patches'
    /// rotary positions.
    func forward(_ patches: MLXArray, masks: MLXArray, positions: MLXArray, caches: inout [NFKTimesFMLayerCache?])
        -> (embeddings: MLXArray, output: MLXArray, point: MLXArray, quantiles: MLXArray) {
        let embeddings = tokenizer(concatenated([patches, masks.asType(patches.dtype)], axis: -1))
        var hidden = embeddings
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, positions: positions, cache: &caches[index])
        }
        return (embeddings, hidden, pointProjection(hidden), quantileProjection(hidden))
    }
}

// MARK: - Weights

extension NFKMLXTimesFMNet {
    /// Loads a release directory's `model.safetensors`: the official layout (its fused `qkv_proj` split
    /// into the three projections), the transformers one (`google/timesfm-2.5-200m-transformers`), or a
    /// directory ``NFKMLXTimesFM/save(_:toDirectoryURL:)`` wrote.
    ///
    /// Introduced in InferKit 0.4.0.
    public func loadWeights(fromDirectory directory: URL) throws {
        try loadWeights(url: directory.appendingPathComponent("model.safetensors"))
    }

    func loadWeights(url: URL) throws {
        let arrays = try NFKMLXWeights.materializedCheckpoint(url: url).arrays
        var mapped = arrays.keys.contains { $0.hasPrefix("model.layers.") } ? try Self.official(fromTransformers: arrays) : arrays
        for (name, fused) in mapped where name.hasSuffix(".attn.qkv_proj.weight") {
            let prefix = String(name.dropLast("qkv_proj.weight".count))
            let parts = fused.split(parts: 3, axis: 0)
            mapped[prefix + "query.weight"] = parts[0]
            mapped[prefix + "key.weight"] = parts[1]
            mapped[prefix + "value.weight"] = parts[2]
            mapped[name] = nil
        }
        try NFKMLXWeights.apply(mapped.map { ($0.key, $0.value) }, to: self)
    }

    /// The transformers checkpoint's names mapped onto the official module's.
    static func official(fromTransformers arrays: [String: MLXArray]) throws -> [String: MLXArray] {
        var mapped = [String: MLXArray]()
        let blockNames = ["input_layer": "hidden_layer", "output_layer": "output_layer", "residual_layer": "residual_layer"]
        let layerNames = ["input_layernorm.weight": "pre_attn_ln.scale", "post_attention_layernorm.weight": "post_attn_ln.scale",
                          "pre_feedforward_layernorm.weight": "pre_ff_ln.scale",
                          "post_feedforward_layernorm.weight": "post_ff_ln.scale",
                          "mlp.ff0.weight": "ff0.weight", "mlp.ff1.weight": "ff1.weight",
                          "self_attn.o_proj.weight": "attn.out.weight", "self_attn.q_norm.weight": "attn.query_ln.scale",
                          "self_attn.k_norm.weight": "attn.key_ln.scale",
                          "self_attn.scaling": "attn.per_dim_scale.per_dim_scale"]
        for (key, value) in arrays {
            let parts = key.split(separator: ".").map(String.init)
            if key.hasPrefix("model.input_ff_layer."), parts.count == 4, let name = blockNames[parts[2]] {
                mapped["tokenizer.\(name).\(parts[3])"] = value
            } else if key.hasPrefix("output_projection_"), parts.count == 3, let name = blockNames[parts[1]] {
                mapped["\(parts[0]).\(name).\(parts[2])"] = value
            } else if key.hasPrefix("model.layers."), parts.count >= 5, let layer = Int(parts[2]) {
                let rest = parts[3...].joined(separator: ".")
                if let name = layerNames[rest] {
                    mapped["stacked_xf.\(layer).\(name)"] = value
                } else if let projection = ["self_attn.q_proj.weight": "attn.query.weight",
                                            "self_attn.k_proj.weight": "attn.key.weight",
                                            "self_attn.v_proj.weight": "attn.value.weight"][rest] {
                    mapped["stacked_xf.\(layer).\(projection)"] = value
                } else {
                    throw NFKMLXError.weightsMismatch("the TimesFM checkpoint carries an unread tensor \(key)")
                }
            } else {
                throw NFKMLXError.weightsMismatch("the TimesFM checkpoint carries an unread tensor \(key)")
            }
        }
        return mapped
    }
}
