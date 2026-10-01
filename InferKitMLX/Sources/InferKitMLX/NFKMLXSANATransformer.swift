//
//  NFKMLXSANATransformer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The SANA linear-attention DiT (`SanaTransformer2DModel`, NVIDIA): the denoising transformer of an
// efficient high-resolution text-to-image model, and the fourth DiT family here. Two things set it apart:
// the self-attention is LINEAR (ReLU feature maps, O(N) rather than O(N²)), which is what lets it run at
// high resolution, and the feed-forward is a GLUMBConv — a gated inverted-bottleneck depthwise
// convolution over the 2-D token grid rather than a pointwise MLP. Each block also cross-attends to the
// Gemma text embedding through ordinary softmax attention. Conditioning is PixArt-α adaptive-norm-single:
// one shared timestep embedding plus a per-block learned `scale_shift_table`.
//
// For parity the caption embedding is supplied directly (no Gemma), so the DiT is validated in isolation.

/// SANA DiT geometry. Defaults are the released 1.6B model.
public struct NFKMLXSANAConfiguration: Sendable {
    public var inChannels: Int
    public var heads: Int
    public var headDim: Int
    public var layers: Int
    public var crossHeads: Int
    public var crossHeadDim: Int
    public var captionChannels: Int
    public var mlpRatio: Float
    public var patchSize: Int
    public var normEps: Float
    public var timestepScale: Float
    public var attentionBias: Bool

    public init(inChannels: Int = 32, heads: Int = 70, headDim: Int = 32, layers: Int = 20,
                crossHeads: Int = 20, crossHeadDim: Int = 112, captionChannels: Int = 2304,
                mlpRatio: Float = 2.5, patchSize: Int = 1, normEps: Float = 1e-6,
                timestepScale: Float = 1.0, attentionBias: Bool = false) {
        self.inChannels = inChannels
        self.heads = heads
        self.headDim = headDim
        self.layers = layers
        self.crossHeads = crossHeads
        self.crossHeadDim = crossHeadDim
        self.captionChannels = captionChannels
        self.mlpRatio = mlpRatio
        self.patchSize = patchSize
        self.normEps = normEps
        self.timestepScale = timestepScale
        self.attentionBias = attentionBias
    }

    public static let base = NFKMLXSANAConfiguration()

    public static let tiny = NFKMLXSANAConfiguration(
        inChannels: 8, heads: 2, headDim: 8, layers: 2, crossHeads: 2, crossHeadDim: 8,
        captionChannels: 12, mlpRatio: 2.0)

    var innerDim: Int { heads * headDim }
    var glumbHidden: Int { Int(mlpRatio * Float(innerDim)) }
}

/// The `TimestepEmbedding`: `Linear → SiLU → Linear` over the sinusoidal features.
final class NFKSANATimestepMLP: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ innerDim: Int) {
        _linear1.wrappedValue = Linear(256, innerDim)
        _linear2.wrappedValue = Linear(innerDim, innerDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(NFKReferenceRounding.silu(linear1(x))) }
}

/// `PixArtAlphaCombinedTimestepSizeEmbeddings` with no additional conditions: a sinusoidal projection.
final class NFKSANATimestepEmb: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: NFKSANATimestepMLP
    init(_ innerDim: Int) { _timestepEmbedder.wrappedValue = NFKSANATimestepMLP(innerDim) }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        timestepEmbedder(ltxTimestepEmbedding(t))                          // cos-first, 256 channels
    }
}

/// `AdaLayerNormSingle`: one shared timestep embedding, expanded to the six modulation parameters.
final class NFKSANAAdaLNSingle: Module {
    @ModuleInfo(key: "emb") var emb: NFKSANATimestepEmb
    @ModuleInfo(key: "linear") var linear: Linear

    init(_ innerDim: Int) {
        _emb.wrappedValue = NFKSANATimestepEmb(innerDim)
        _linear.wrappedValue = Linear(innerDim, 6 * innerDim)
    }

    /// `t` scalar → `(modulation [B, 6·inner], embedded [B, inner])`.
    func callAsFunction(_ t: MLXArray) -> (modulation: MLXArray, embedded: MLXArray) {
        let embedded = emb(t)
        return (linear(silu(embedded)), embedded)
    }
}

/// `PixArtAlphaTextProjection`: `Linear → GELU(tanh) → Linear`.
final class NFKSANATextProjection: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inFeatures: Int, hidden: Int) {
        _linear1.wrappedValue = Linear(inFeatures, hidden)
        _linear2.wrappedValue = Linear(hidden, hidden)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(NFKReferenceRounding.geluTanh(linear1(x))) }
}

/// SANA's ReLU linear attention: `O = ((V·1̂) @ ReLU(K)) @ ReLU(Q)`, normalized by the ones row.
final class NFKSANALinearAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]                         // [Linear]

    let heads: Int
    let headDim: Int

    init(_ config: NFKMLXSANAConfiguration) {
        self.heads = config.heads
        self.headDim = config.headDim
        _toQ.wrappedValue = Linear(config.innerDim, config.innerDim, bias: config.attentionBias)
        _toK.wrappedValue = Linear(config.innerDim, config.innerDim, bias: config.attentionBias)
        _toV.wrappedValue = Linear(config.innerDim, config.innerDim, bias: config.attentionBias)
        _toOut.wrappedValue = [Linear(config.innerDim, config.innerDim)]
    }

    /// `x` `[N, inner]` (batch 1) → `[N, inner]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let n = x.dim(0)
        // [heads, headDim, N] for query/value, [heads, N, headDim] for key.
        let q = relu(toQ(x).reshaped([n, heads, headDim]).transposed(1, 2, 0))
        let k = relu(toK(x).reshaped([n, heads, headDim]).transposed(1, 0, 2))
        let v = toV(x).reshaped([n, heads, headDim]).transposed(1, 2, 0)
        let vPadded = concatenated([v, MLXArray.ones([heads, 1, n])], axis: 1) // append a ones row
        let scores = matmul(vPadded, k)                                    // [heads, headDim+1, headDim]
        let attended = matmul(scores, q)                                   // [heads, headDim+1, N]
        let numerator = attended[0..., 0 ..< headDim, 0...]
        let denominator = attended[0..., headDim ..< (headDim + 1), 0...] + 1e-15
        let out = (numerator / denominator).reshaped([heads * headDim, n]).transposed(1, 0)
        return (toOut[0] as! Linear)(out)
    }
}

/// Standard softmax cross-attention to the caption tokens.
final class NFKSANACrossAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]                         // [Linear]

    let heads: Int
    let headDim: Int

    init(_ config: NFKMLXSANAConfiguration) {
        self.heads = config.crossHeads
        self.headDim = config.crossHeadDim
        let inner = config.crossHeads * config.crossHeadDim
        _toQ.wrappedValue = Linear(config.innerDim, inner)
        _toK.wrappedValue = Linear(config.innerDim, inner)
        _toV.wrappedValue = Linear(config.innerDim, inner)
        _toOut.wrappedValue = [Linear(inner, config.innerDim)]
    }

    /// `x` `[N, inner]`, `context` `[Lc, inner]` → `[N, inner]`.
    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let n = x.dim(0), lc = context.dim(0)
        let q = toQ(x).reshaped([n, heads, headDim]).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let k = toK(context).reshaped([lc, heads, headDim]).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let v = toV(context).reshaped([lc, heads, headDim]).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let scale = 1.0 / sqrt(Float(headDim))
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        let merged = out[0].transposed(1, 0, 2).reshaped([n, heads * headDim])
        return (toOut[0] as! Linear)(merged)
    }
}

/// The GLUMBConv Mix-FFN: an inverted-bottleneck gated depthwise convolution over the 2-D token grid.
final class NFKSANAGLUMBConv: Module {
    @ModuleInfo(key: "conv_inverted") var convInverted: Conv2d
    @ModuleInfo(key: "conv_depth") var convDepth: Conv2d
    @ModuleInfo(key: "conv_point") var convPoint: Conv2d

    init(_ config: NFKMLXSANAConfiguration) {
        let hidden = config.glumbHidden
        _convInverted.wrappedValue = Conv2d(inputChannels: config.innerDim, outputChannels: hidden * 2, kernelSize: 1)
        _convDepth.wrappedValue = Conv2d(inputChannels: hidden * 2, outputChannels: hidden * 2,
                                         kernelSize: 3, padding: 1, groups: hidden * 2)
        _convPoint.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: config.innerDim,
                                         kernelSize: 1, bias: false)
    }

    /// `x` `[1, H, W, inner]` → `[1, H, W, inner]` (channels last).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = silu(convInverted(x))
        h = convDepth(h)
        let half = h.dim(3) / 2
        let value = h[0..., 0..., 0..., 0 ..< half]
        let gate = h[0..., 0..., 0..., half ..< (2 * half)]
        return convPoint(value * silu(gate))
    }
}

/// A SANA block: linear self-attention, softmax cross-attention, GLUMBConv, all under adaptive norms
/// driven by the shared timestep embedding plus the block's own `scale_shift_table`.
final class NFKSANABlock: Module {
    @ModuleInfo(key: "attn1") var attn1: NFKSANALinearAttention
    @ModuleInfo(key: "attn2") var attn2: NFKSANACrossAttention
    @ModuleInfo(key: "ff") var ff: NFKSANAGLUMBConv
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    let norm1: LayerNorm
    let norm2: LayerNorm

    init(_ config: NFKMLXSANAConfiguration) {
        _attn1.wrappedValue = NFKSANALinearAttention(config)
        _attn2.wrappedValue = NFKSANACrossAttention(config)
        _ff.wrappedValue = NFKSANAGLUMBConv(config)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([6, config.innerDim])
        self.norm1 = LayerNorm(dimensions: config.innerDim, eps: config.normEps, affine: false)
        self.norm2 = LayerNorm(dimensions: config.innerDim, eps: config.normEps, affine: false)
    }

    /// `x` `[N, inner]`, `context` `[Lc, inner]`, `temb` `[6·inner]`, grid `H×W` (`H·W == N`).
    func callAsFunction(_ x: MLXArray, context: MLXArray, temb: MLXArray, height: Int, width: Int) -> MLXArray {
        let inner = x.dim(1)
        let mod = (scaleShiftTable + temb.reshaped([6, inner]))            // [6, inner]
        let shiftMSA = mod[0].reshaped([1, inner]), scaleMSA = mod[1].reshaped([1, inner])
        let gateMSA = mod[2].reshaped([1, inner])
        let shiftMLP = mod[3].reshaped([1, inner]), scaleMLP = mod[4].reshaped([1, inner])
        let gateMLP = mod[5].reshaped([1, inner])

        var h = x + gateMSA * attn1(norm1(x) * (1.0 + scaleMSA) + shiftMSA)
        h = h + attn2(h, context: context)
        let normed = norm2(h) * (1.0 + scaleMLP) + shiftMLP
        let grid = normed.reshaped([1, height, width, inner])
        let ffOut = ff(grid).reshaped([height * width, inner])
        return h + gateMLP * ffOut
    }
}

/// The SANA transformer.
public final class NFKMLXSANATransformerNet: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: NFKSANAPatchEmbed
    @ModuleInfo(key: "time_embed") var timeEmbed: NFKSANAAdaLNSingle
    @ModuleInfo(key: "caption_projection") var captionProjection: NFKSANATextProjection
    @ModuleInfo(key: "caption_norm") var captionNorm: RMSNorm
    @ModuleInfo(key: "transformer_blocks") var blocks: [NFKSANABlock]
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let config: NFKMLXSANAConfiguration
    let normOut: LayerNorm

    public init(_ config: NFKMLXSANAConfiguration) {
        self.config = config
        _patchEmbed.wrappedValue = NFKSANAPatchEmbed(config)
        _timeEmbed.wrappedValue = NFKSANAAdaLNSingle(config.innerDim)
        _captionProjection.wrappedValue = NFKSANATextProjection(inFeatures: config.captionChannels, hidden: config.innerDim)
        _captionNorm.wrappedValue = RMSNorm(dimensions: config.innerDim, eps: 1e-5)
        _blocks.wrappedValue = (0 ..< config.layers).map { _ in NFKSANABlock(config) }
        _scaleShiftTable.wrappedValue = MLXArray.zeros([2, config.innerDim])
        _projOut.wrappedValue = Linear(config.innerDim, config.patchSize * config.patchSize * config.inChannels)
        self.normOut = LayerNorm(dimensions: config.innerDim, eps: 1e-6, affine: false)
    }

    /// Velocity prediction. `x` `[C, H, W]`, `capFeats` `[Lc, captionChannels]`, `t` scalar.
    public func callAsFunction(_ x: MLXArray, capFeats: MLXArray, t: MLXArray) -> MLXArray {
        let c = x.dim(0), height = x.dim(1), width = x.dim(2)
        let p = config.patchSize
        let ph = height / p, pw = width / p
        let inner = config.innerDim

        let nhwc = x.transposed(1, 2, 0).expandedDimensions(axis: 0)       // [1, H, W, C]
        var hidden = patchEmbed(nhwc).reshaped([ph * pw, inner])           // [N, inner]

        let (modulation, embedded) = timeEmbed(t * config.timestepScale)   // [1, 6·inner], [1, inner]
        let context = captionNorm(captionProjection(capFeats))            // [Lc, inner]

        for block in blocks {
            hidden = block(hidden, context: context, temb: modulation[0], height: ph, width: pw)
        }
        _ = context

        // Output modulation from the shared embedded timestep and the top-level scale_shift_table.
        let outMod = scaleShiftTable + embedded[0].reshaped([1, inner])    // [2, inner] + [1, inner]
        let shift = outMod[0].reshaped([1, inner]), scale = outMod[1].reshaped([1, inner])
        hidden = normOut(hidden) * (1.0 + scale) + shift
        let out = projOut(hidden)                                          // [N, p·p·C]

        // Unpatchify [N, p·p·C] -> [C, H, W].
        let grid = out.reshaped([ph, pw, p, p, c])
        return grid.transposed(4, 0, 2, 1, 3).reshaped([c, height, width])
    }
}

/// SANA's patch embedding: a stride-`patch` convolution (no positional embedding).
final class NFKSANAPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(_ config: NFKMLXSANAConfiguration) {
        _proj.wrappedValue = Conv2d(inputChannels: config.inChannels, outputChannels: config.innerDim,
                                    kernelSize: IntOrPair(config.patchSize), stride: IntOrPair(config.patchSize))
    }

    /// `[1, H, W, C]` → `[1, H/p, W/p, inner]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

// MARK: - Release

extension NFKMLXSANATransformerNet {

    /// The geometry a diffusers `transformer/config.json` declares (`SanaTransformer2DModel`).
    ///
    /// @discussion A release that turns on a part this port does not carry is refused rather than
    /// built without it: `guidance_embeds` (SANA-Sprint), a `qk_norm`, or an `interpolation_scale`,
    /// which adds a sine-cosine position embedding. Introduced in InferKit 0.4.0.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXSANAConfiguration {
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func float(_ key: String, _ fallback: Float) -> Float { (json[key] as? NSNumber)?.floatValue ?? fallback }
        if (json["guidance_embeds"] as? Bool) == true {
            throw NFKMLXError.unsupportedConfiguration("a SANA release with guidance_embeds is not ported")
        }
        if let norm = json["qk_norm"] as? String {
            throw NFKMLXError.unsupportedConfiguration("a SANA release with qk_norm \(norm) is not ported")
        }
        if json["interpolation_scale"] is NSNumber {
            throw NFKMLXError.unsupportedConfiguration("a SANA release with a sine-cosine position embedding is not ported")
        }
        let inChannels = int("in_channels", 32)
        if int("out_channels", inChannels) != inChannels {
            throw NFKMLXError.unsupportedConfiguration("a SANA release whose output channels differ from its input is not ported")
        }
        let configuration = NFKMLXSANAConfiguration(
            inChannels: inChannels, heads: int("num_attention_heads", 70), headDim: int("attention_head_dim", 32),
            layers: int("num_layers", 20), crossHeads: int("num_cross_attention_heads", 20),
            crossHeadDim: int("cross_attention_head_dim", 112), captionChannels: int("caption_channels", 2304),
            mlpRatio: float("mlp_ratio", 2.5), patchSize: int("patch_size", 1), normEps: float("norm_eps", 1e-6),
            timestepScale: float("timestep_scale", 1), attentionBias: (json["attention_bias"] as? Bool) ?? false)
        if let crossDim = (json["cross_attention_dim"] as? NSNumber)?.intValue,
           crossDim != configuration.crossHeads * configuration.crossHeadDim {
            throw NFKMLXError.unsupportedConfiguration(
                "cross_attention_dim \(crossDim) is not num_cross_attention_heads × cross_attention_head_dim")
        }
        return configuration
    }

    /// Loads a diffusers `transformer/` directory (one file or a shard index). The module keys are
    /// diffusers' own names; the patch convolution moves from `[out, in, kH, kW]` to MLX's
    /// `[out, kH, kW, in]`. Introduced in InferKit 0.4.0.
    public static func loadWeights(into net: NFKMLXSANATransformerNet, fromDirectory directory: URL) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory).map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(arrays, to: net)
    }
}
