//
//  NFKMLXSD3Transformer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The Stable Diffusion 3 MMDiT (`SD3Transformer2DModel`), the multimodal diffusion transformer of the
// SD3 / SD3.5 text-to-image models, and the sixth DiT family here. It is DUAL-STREAM: the image latent
// tokens and the text tokens each carry their own projections, feed-forward, and adaptive-norm
// modulation, while attention is computed JOINTLY over the concatenation of the two streams. This is a
// different design from Z-Image's single-stream (which concatenates and SHARES weights per layer): a
// JointTransformerBlock holds a full attention and feed-forward for each stream.
//
// The last block runs `context_pre_only` — the text stream contributes keys and values to the joint
// attention but its own output and feed-forward are dropped, since nothing downstream reads the text
// again. SD3.5 large adds two things over SD3.0: a per-head RMS query/key normalization (`qk_norm`),
// and DUAL ATTENTION on the early layers — a second, image-only self-attention (`attn2`) whose output
// is gated in beside the joint one.
//
// For parity the caption embedding (T5 + the two CLIP sequence outputs) and the pooled projection (the
// two CLIP pooled outputs) are supplied directly, so the DiT is validated in isolation, as the LTX and
// Z-Image DiTs are; the text encoders are the SD3 pipeline's own stage.

/// SD3 MMDiT geometry. Defaults are the SD3-medium (2B) release.
public struct NFKMLXSD3Configuration: Sendable {
    public var sampleSize: Int
    public var patchSize: Int
    public var inChannels: Int
    public var outChannels: Int
    public var numLayers: Int
    public var attentionHeadDim: Int
    public var numAttentionHeads: Int
    public var jointAttentionDim: Int
    public var captionProjectionDim: Int
    public var pooledProjectionDim: Int
    public var posEmbedMaxSize: Int
    public var dualAttentionLayers: [Int]
    public var qkNorm: Bool

    public init(sampleSize: Int = 128, patchSize: Int = 2, inChannels: Int = 16, outChannels: Int = 16,
                numLayers: Int = 24, attentionHeadDim: Int = 64, numAttentionHeads: Int = 24,
                jointAttentionDim: Int = 4096, captionProjectionDim: Int = 1536,
                pooledProjectionDim: Int = 2048, posEmbedMaxSize: Int = 192,
                dualAttentionLayers: [Int] = [], qkNorm: Bool = false) {
        self.sampleSize = sampleSize
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.numLayers = numLayers
        self.attentionHeadDim = attentionHeadDim
        self.numAttentionHeads = numAttentionHeads
        self.jointAttentionDim = jointAttentionDim
        self.captionProjectionDim = captionProjectionDim
        self.pooledProjectionDim = pooledProjectionDim
        self.posEmbedMaxSize = posEmbedMaxSize
        self.dualAttentionLayers = dualAttentionLayers
        self.qkNorm = qkNorm
    }

    /// SD3-medium (`stabilityai/stable-diffusion-3-medium-diffusers`), 2B: 24 layers, 24 heads, no
    /// qk-norm, no dual attention.
    public static let sd3Medium = NFKMLXSD3Configuration()

    /// SD3.5-large (`stabilityai/stable-diffusion-3.5-large`), 8B: 38 layers, 38 heads (inner 2432),
    /// RMS qk-norm, no dual attention (standard MMDiT).
    public static let sd35Large = NFKMLXSD3Configuration(
        numLayers: 38, numAttentionHeads: 38, captionProjectionDim: 2432, qkNorm: true)

    /// SD3.5-medium (`stabilityai/stable-diffusion-3.5-medium`), 2.5B (MMDiT-X): 24 layers, 24 heads,
    /// RMS qk-norm, dual attention on the first thirteen layers, `pos_embed_max_size` 384.
    public static let sd35Medium = NFKMLXSD3Configuration(
        numLayers: 24, numAttentionHeads: 24, captionProjectionDim: 1536, posEmbedMaxSize: 384,
        dualAttentionLayers: Array(0 ... 12), qkNorm: true)

    /// A tiny random configuration for reference parity: two layers (the first dual-attention), two
    /// heads of eight, RMS qk-norm, a small position grid the crop is exercised on.
    public static let tiny = NFKMLXSD3Configuration(
        sampleSize: 16, inChannels: 4, outChannels: 4, numLayers: 2, attentionHeadDim: 8,
        numAttentionHeads: 2, jointAttentionDim: 24, captionProjectionDim: 16, pooledProjectionDim: 20,
        posEmbedMaxSize: 8, dualAttentionLayers: [0], qkNorm: true)

    var innerDim: Int { numAttentionHeads * attentionHeadDim }
    var patchOutputDim: Int { patchSize * patchSize * outChannels }
}

/// An affine-free layer normalization over the last axis (the reference's `elementwise_affine=False`).
func sd3AffineFreeLayerNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let variance = (x - mean).square().mean(axis: -1, keepDims: true)
    return (x - mean) * rsqrt(variance + eps)
}

/// The sinusoidal timestep features (`get_timestep_embedding`, `flip_sin_to_cos=True`,
/// `downscale_freq_shift=0`), a `[cos, sin]` table.
func sd3TimestepEmbedding(_ timesteps: MLXArray, dimensions: Int, maxPeriod: Float = 10000) -> MLXArray {
    let half = dimensions / 2
    let exponent = -log(maxPeriod) * MLXArray(0 ..< half).asType(.float32) / Float(half)
    let emb = exp(exponent)
    let args = timesteps.reshaped([-1, 1]).asType(.float32) * emb.reshaped([1, -1])
    return concatenated([cos(args), sin(args)], axis: -1)                  // flip_sin_to_cos → [cos, sin]
}

/// The first feed-forward entry: a `proj` linear (diffusers' `GELU` module holds its linear under
/// `proj`), followed by the tanh-approximate GELU applied in the feed-forward's forward.
final class NFKSD3GELUProj: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ inDim: Int, _ outDim: Int) { _proj.wrappedValue = Linear(inDim, outDim) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

/// A parameter-free marker occupying a `nn.Sequential` / `nn.ModuleList` index (an activation, a
/// dropout).
final class NFKSD3Marker: Module {}

/// The feed-forward `net`: `net.0` is the tanh-GELU projection, `net.2` the output linear (`net.1` a
/// dropout).
final class NFKSD3FeedForward: Module {
    @ModuleInfo(key: "net") var net: [Module]                             // [NFKSD3GELUProj, marker, Linear]

    init(_ dim: Int) {
        _net.wrappedValue = [NFKSD3GELUProj(dim, dim * 4), NFKSD3Marker(), Linear(dim * 4, dim)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (net[2] as! Linear)(geluApproximate((net[0] as! NFKSD3GELUProj)(x)))
    }
}

/// The 2D patch embedding: a stride-`patch` convolution over the latent, then a cropped sincos
/// positional table added to the tokens. The table is a persistent buffer loaded from the checkpoint
/// (the reference precomputes it at `pos_embed_max_size²` and center-crops it to the latent grid).
final class NFKSD3PatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray

    let patchSize: Int
    let maxSize: Int

    init(_ config: NFKMLXSD3Configuration) {
        self.patchSize = config.patchSize
        self.maxSize = config.posEmbedMaxSize
        _proj.wrappedValue = Conv2d(inputChannels: config.inChannels, outputChannels: config.innerDim,
                                    kernelSize: IntOrPair(config.patchSize), stride: IntOrPair(config.patchSize))
        _posEmbed.wrappedValue = MLXArray.zeros([1, config.posEmbedMaxSize * config.posEmbedMaxSize, config.innerDim])
    }

    /// `latent` `[B, C, H, W]` → tokens `[B, (H/p)·(W/p), inner]` with the cropped positions added.
    func callAsFunction(_ latent: MLXArray) -> MLXArray {
        let h = latent.dim(2) / patchSize, w = latent.dim(3) / patchSize
        let nhwc = latent.transposed(0, 2, 3, 1)                           // [B, H, W, C]
        let convolved = proj(nhwc)                                         // [B, H/p, W/p, inner]
        let tokens = convolved.reshaped([convolved.dim(0), h * w, convolved.dim(3)])
        // Center-crop the precomputed sincos grid to the latent grid.
        let top = (maxSize - h) / 2, left = (maxSize - w) / 2
        let dim = posEmbed.dim(2)
        let grid = posEmbed.reshaped([1, maxSize, maxSize, dim])
        let cropped = grid[0..., top ..< top + h, left ..< left + w, 0...].reshaped([1, h * w, dim])
        return tokens + cropped
    }
}

/// A two-linear MLP with an activation between, the reference's `TimestepEmbedding` (silu) and
/// `PixArtAlphaTextProjection` (silu) both keyed `linear_1` / `linear_2`.
final class NFKSD3MLP: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ inDim: Int, _ hiddenDim: Int, _ outDim: Int) {
        _linear1.wrappedValue = Linear(inDim, hiddenDim)
        _linear2.wrappedValue = Linear(hiddenDim, outDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

/// The combined timestep + pooled-text embedding: the sinusoidal timestep through an MLP, plus the
/// pooled projection through a `Linear → SiLU → Linear`, summed.
final class NFKSD3TimeTextEmbed: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: NFKSD3MLP
    @ModuleInfo(key: "text_embedder") var textEmbedder: NFKSD3MLP

    init(_ config: NFKMLXSD3Configuration) {
        _timestepEmbedder.wrappedValue = NFKSD3MLP(256, config.innerDim, config.innerDim)
        _textEmbedder.wrappedValue = NFKSD3MLP(config.pooledProjectionDim, config.innerDim, config.innerDim)
    }

    /// `timestep` `[B]`, `pooled` `[B, pooledDim]` → `[B, inner]`.
    func callAsFunction(timestep: MLXArray, pooled: MLXArray) -> MLXArray {
        let proj = sd3TimestepEmbedding(timestep, dimensions: 256)         // [B, 256]
        return timestepEmbedder(proj) + textEmbedder(pooled)
    }
}

/// An adaptive-norm modulation projection (`silu` then a `linear`), the reference's `AdaLayerNorm*`
/// with the affine-free normalization applied by the caller. Holds only the `linear`.
final class NFKSD3AdaLinear: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    init(_ inDim: Int, _ outDim: Int) { _linear.wrappedValue = Linear(inDim, outDim) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(silu(x)) }
}

/// The joint (MMDiT) attention: separate query/key/value projections for the image stream, optional
/// added query/key/value projections for the text stream, optional per-head RMS query/key norms, and a
/// single softmax over the concatenated sequence. When `encoder` is nil (the dual-attention `attn2`),
/// only the image stream runs.
final class NFKSD3Attention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]                        // [Linear, dropout-marker]
    @ModuleInfo(key: "add_q_proj") var addQ: Linear?
    @ModuleInfo(key: "add_k_proj") var addK: Linear?
    @ModuleInfo(key: "add_v_proj") var addV: Linear?
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear?
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm?
    @ModuleInfo(key: "norm_k") var normK: RMSNorm?
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm?
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm?

    let heads: Int
    let headDim: Int
    let hasAdded: Bool
    let contextPreOnly: Bool

    init(dim: Int, heads: Int, headDim: Int, qkNorm: Bool, added: Bool, contextPreOnly: Bool) {
        self.heads = heads
        self.headDim = headDim
        self.hasAdded = added
        self.contextPreOnly = contextPreOnly
        let inner = heads * headDim
        _toQ.wrappedValue = Linear(dim, inner)
        _toK.wrappedValue = Linear(dim, inner)
        _toV.wrappedValue = Linear(dim, inner)
        _toOut.wrappedValue = [Linear(inner, dim), NFKSD3Marker()]
        if added {
            _addQ.wrappedValue = Linear(dim, inner)
            _addK.wrappedValue = Linear(dim, inner)
            _addV.wrappedValue = Linear(dim, inner)
            _toAddOut.wrappedValue = contextPreOnly ? nil : Linear(inner, dim)
        }
        if qkNorm {
            _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
            _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
            if added {
                _normAddedQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
                _normAddedK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
            }
        }
    }

    /// `hidden` `[B, Si, dim]`, `encoder` `[B, St, dim]?` → the image output `[B, Si, dim]` and, when a
    /// text stream is present and not `context_pre_only`, the text output `[B, St, dim]`.
    func callAsFunction(_ hidden: MLXArray, encoder: MLXArray?) -> (image: MLXArray, context: MLXArray?) {
        let b = hidden.dim(0), imageLength = hidden.dim(1)
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([b, -1, heads, headDim]).transposed(0, 2, 1, 3) }
        var q = split(toQ(hidden))
        var k = split(toK(hidden))
        var v = split(toV(hidden))
        if let normQ { q = normQ(q) }
        if let normK { k = normK(k) }
        if let encoder, hasAdded {
            var eq = split(addQ!(encoder))
            var ek = split(addK!(encoder))
            let ev = split(addV!(encoder))
            if let normAddedQ { eq = normAddedQ(eq) }
            if let normAddedK { ek = normAddedK(ek) }
            q = concatenated([q, eq], axis: 2)
            k = concatenated([k, ek], axis: 2)
            v = concatenated([v, ev], axis: 2)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / sqrt(Float(headDim)), mask: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([b, -1, heads * headDim])
        if encoder != nil {
            let imagePart = merged[0..., 0 ..< imageLength, 0...]
            let contextPart = merged[0..., imageLength..., 0...]
            let image = (toOut[0] as! Linear)(imagePart)
            let context = contextPreOnly ? nil : toAddOut!(contextPart)
            return (image, context)
        }
        return ((toOut[0] as! Linear)(merged), nil)
    }
}

/// One JointTransformerBlock: a joint attention with dual-stream adaptive norms and per-stream
/// feed-forwards. The last block runs `context_pre_only` (the text stream ends after the attention);
/// the dual-attention blocks add a second image-only self-attention.
final class NFKSD3JointBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: NFKSD3AdaLinear                   // 6·dim, or 9·dim when dual
    @ModuleInfo(key: "norm1_context") var norm1Context: NFKSD3AdaLinear   // 6·dim, or 2·dim when pre-only
    @ModuleInfo(key: "attn") var attn: NFKSD3Attention
    @ModuleInfo(key: "attn2") var attn2: NFKSD3Attention?
    @ModuleInfo(key: "ff") var ff: NFKSD3FeedForward
    @ModuleInfo(key: "ff_context") var ffContext: NFKSD3FeedForward?

    let dim: Int
    let contextPreOnly: Bool
    let useDualAttention: Bool

    init(_ config: NFKMLXSD3Configuration, contextPreOnly: Bool, useDualAttention: Bool) {
        self.dim = config.innerDim
        self.contextPreOnly = contextPreOnly
        self.useDualAttention = useDualAttention
        _norm1.wrappedValue = NFKSD3AdaLinear(dim, (useDualAttention ? 9 : 6) * dim)
        _norm1Context.wrappedValue = NFKSD3AdaLinear(dim, (contextPreOnly ? 2 : 6) * dim)
        _attn.wrappedValue = NFKSD3Attention(dim: dim, heads: config.numAttentionHeads,
                                             headDim: config.attentionHeadDim, qkNorm: config.qkNorm,
                                             added: true, contextPreOnly: contextPreOnly)
        if useDualAttention {
            _attn2.wrappedValue = NFKSD3Attention(dim: dim, heads: config.numAttentionHeads,
                                                  headDim: config.attentionHeadDim, qkNorm: config.qkNorm,
                                                  added: false, contextPreOnly: false)
        }
        _ff.wrappedValue = NFKSD3FeedForward(dim)
        _ffContext.wrappedValue = contextPreOnly ? nil : NFKSD3FeedForward(dim)
    }

    /// `hidden` `[B, Si, dim]`, `encoder` `[B, St, dim]`, `temb` `[B, dim]` → the updated text stream
    /// (nil when `context_pre_only`) and image stream.
    func callAsFunction(_ hidden: MLXArray, encoder: MLXArray, temb: MLXArray) -> (context: MLXArray?, image: MLXArray) {
        var hiddenStates = hidden
        var encoderStates = encoder

        // Image-stream adaptive norm.
        let modulation = norm1(temb)                                       // [B, 6·dim] or [B, 9·dim]
        func chunk(_ x: MLXArray, _ i: Int) -> MLXArray { x[0..., (i * dim) ..< ((i + 1) * dim)][0..., .newAxis, 0...] }
        let shiftMSA = chunk(modulation, 0), scaleMSA = chunk(modulation, 1), gateMSA = chunk(modulation, 2)
        let shiftMLP = chunk(modulation, 3), scaleMLP = chunk(modulation, 4), gateMLP = chunk(modulation, 5)
        let normed = sd3AffineFreeLayerNorm(hiddenStates)
        let normHidden = normed * (1 + scaleMSA) + shiftMSA

        // Text-stream adaptive norm.
        var cGateMSA: MLXArray?, cShiftMLP: MLXArray?, cScaleMLP: MLXArray?, cGateMLP: MLXArray?
        let normEncoder: MLXArray
        if contextPreOnly {
            let cmod = norm1Context(temb)                                  // [B, 2·dim]: scale, shift
            let cScale = cmod[0..., 0 ..< dim][0..., .newAxis, 0...]
            let cShift = cmod[0..., dim ..< 2 * dim][0..., .newAxis, 0...]
            normEncoder = sd3AffineFreeLayerNorm(encoderStates) * (1 + cScale) + cShift
        } else {
            let cmod = norm1Context(temb)                                  // [B, 6·dim]
            let cShiftMSA = chunk(cmod, 0), cScaleMSA = chunk(cmod, 1)
            cGateMSA = chunk(cmod, 2); cShiftMLP = chunk(cmod, 3); cScaleMLP = chunk(cmod, 4); cGateMLP = chunk(cmod, 5)
            normEncoder = sd3AffineFreeLayerNorm(encoderStates) * (1 + cScaleMSA) + cShiftMSA
        }

        // Joint attention.
        let (attnImage, attnContext) = attn(normHidden, encoder: normEncoder)
        hiddenStates = hiddenStates + gateMSA * attnImage

        if useDualAttention, let attn2 {
            // The ZeroX modulation appends (shift_msa2, scale_msa2, gate_msa2) after the six.
            let shiftMSA2 = chunk(modulation, 6), scaleMSA2 = chunk(modulation, 7), gateMSA2 = chunk(modulation, 8)
            let normHidden2 = normed * (1 + scaleMSA2) + shiftMSA2
            let (attnImage2, _) = attn2(normHidden2, encoder: nil)
            hiddenStates = hiddenStates + gateMSA2 * attnImage2
        }

        // Image-stream feed-forward.
        let normFF = sd3AffineFreeLayerNorm(hiddenStates) * (1 + scaleMLP) + shiftMLP
        hiddenStates = hiddenStates + gateMLP * ff(normFF)

        // Text-stream feed-forward.
        if contextPreOnly {
            return (nil, hiddenStates)
        }
        encoderStates = encoderStates + cGateMSA! * attnContext!
        let normContextFF = sd3AffineFreeLayerNorm(encoderStates) * (1 + cScaleMLP!) + cShiftMLP!
        encoderStates = encoderStates + cGateMLP! * ffContext!(normContextFF)
        return (encoderStates, hiddenStates)
    }
}

/// The SD3 / SD3.5 MMDiT.
public final class NFKMLXSD3TransformerNet: Module {
    @ModuleInfo(key: "pos_embed") var posEmbed: NFKSD3PatchEmbed
    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: NFKSD3TimeTextEmbed
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [NFKSD3JointBlock]
    @ModuleInfo(key: "norm_out") var normOut: NFKSD3AdaLinear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public let config: NFKMLXSD3Configuration

    public init(_ config: NFKMLXSD3Configuration) {
        self.config = config
        _posEmbed.wrappedValue = NFKSD3PatchEmbed(config)
        _timeTextEmbed.wrappedValue = NFKSD3TimeTextEmbed(config)
        _contextEmbedder.wrappedValue = Linear(config.jointAttentionDim, config.innerDim)
        _transformerBlocks.wrappedValue = (0 ..< config.numLayers).map { index in
            NFKSD3JointBlock(config, contextPreOnly: index == config.numLayers - 1,
                             useDualAttention: config.dualAttentionLayers.contains(index))
        }
        _normOut.wrappedValue = NFKSD3AdaLinear(config.innerDim, 2 * config.innerDim)
        _projOut.wrappedValue = Linear(config.innerDim, config.patchOutputDim)
    }

    /// Velocity prediction. `hiddenStates` `[B, C, H, W]`, `encoderHidden` `[B, L, jointDim]`, `pooled`
    /// `[B, pooledDim]`, `timestep` `[B]` (0…1000) → `[B, C, H, W]`.
    ///
    /// `blockControlnetHiddenStates` are a ControlNet's per-block residuals (``NFKMLXSD3ControlNetNet``):
    /// each non-`context_pre_only` block's image stream is offset by one, strided over the residual list
    /// as the reference's `interval_control` does (nil for plain text-to-image, which leaves the forward
    /// byte-identical).
    public func callAsFunction(_ hiddenStates: MLXArray, encoderHidden: MLXArray, pooled: MLXArray,
                               timestep: MLXArray, blockControlnetHiddenStates: [MLXArray]? = nil) -> MLXArray {
        let height = hiddenStates.dim(2), width = hiddenStates.dim(3)
        var image = posEmbed(hiddenStates)                                 // [B, Si, inner]
        let temb = timeTextEmbed(timestep: timestep, pooled: pooled)       // [B, inner]
        var context: MLXArray? = contextEmbedder(encoderHidden)            // [B, L, inner]

        for (index, block) in transformerBlocks.enumerated() {
            let (newContext, newImage) = block(image, encoder: context!, temb: temb)
            image = newImage
            context = newContext                                           // nil after the pre-only block
            if let residuals = blockControlnetHiddenStates, !block.contextPreOnly {
                let interval = Double(transformerBlocks.count) / Double(residuals.count)
                image = image + residuals[Int(Double(index) / interval)]
            }
        }

        // Final adaptive norm and projection to patches.
        let modulation = normOut(temb)                                     // [B, 2·inner]: scale, shift
        let dim = config.innerDim
        let scale = modulation[0..., 0 ..< dim][0..., .newAxis, 0...]
        let shift = modulation[0..., dim ..< 2 * dim][0..., .newAxis, 0...]
        image = sd3AffineFreeLayerNorm(image) * (1 + scale) + shift
        image = projOut(image)                                             // [B, Si, p·p·out]

        // Unpatchify: [B, H/p, W/p, p, p, out] → einsum nhwpqc->nchpwq → [B, out, H, W].
        let p = config.patchSize, oc = config.outChannels
        let hp = height / p, wp = width / p
        let reshaped = image.reshaped([image.dim(0), hp, wp, p, p, oc])
        return reshaped.transposed(0, 5, 1, 3, 2, 4).reshaped([image.dim(0), oc, height, width])
    }

    /// The geometry of a released SD3 / SD3.5 transformer, from its `transformer/config.json`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXSD3Configuration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "SD3Transformer2DModel" {
            throw NFKMLXError.unsupportedConfiguration("this reads an SD3 transformer, not \(kind)")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        let outChannels = (json["out_channels"] as? NSNumber)?.intValue ?? integer("in_channels", 16)
        let dual = (json["dual_attention_layers"] as? [Any])?.compactMap { ($0 as? NSNumber)?.intValue } ?? []
        let qkNorm = (json["qk_norm"] as? String) != nil
        return NFKMLXSD3Configuration(
            sampleSize: integer("sample_size", 128), patchSize: integer("patch_size", 2),
            inChannels: integer("in_channels", 16), outChannels: outChannels,
            numLayers: integer("num_layers", 24), attentionHeadDim: integer("attention_head_dim", 64),
            numAttentionHeads: integer("num_attention_heads", 24),
            jointAttentionDim: integer("joint_attention_dim", 4096),
            captionProjectionDim: integer("caption_projection_dim", 1536),
            pooledProjectionDim: integer("pooled_projection_dim", 2048),
            posEmbedMaxSize: integer("pos_embed_max_size", 192),
            dualAttentionLayers: dual, qkNorm: qkNorm)
    }

    /// Loads a released SD3 / SD3.5 transformer directory (`transformer/`) into `net`. The only tensor
    /// needing a layout change is the 4-D patch-embed convolution, transposed to MLX's NHWC.
    public static func loadWeights(into net: NFKMLXSD3TransformerNet, from directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision)
        let weights = arrays.map { name, value -> (String, MLXArray) in
            (name, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(weights, to: net)
    }
}
