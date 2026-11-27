//
//  NFKMLXWanTransformer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The Wan DiT (`WanTransformer3DModel`, Alibaba Wan): the denoising transformer of the Wan text-to-video
// model, and the fifth DiT family here. It is a 3-D sequence transformer over a video latent patchified
// by a `Conv3d`, with a 3-axis interleaved rotary over the (frame, height, width) grid. Each block runs
// self-attention (with rotary), cross-attention to the text embedding, and a gelu feed-forward, all under
// PixArt-α adaptive norms driven by a shared timestep projection plus a per-block `scale_shift_table`.
// The query/key normalization is `rms_norm_across_heads` — an RMS norm over the WHOLE inner width, before
// the head split, where the Z-Image DiT norms per head.
//
// The transformer has no image-embedding branch. Wan 2.2 I2V-A14B conditions on an image through its
// input channels (the noise, a mask, and the image's latents, 36 in all) and predicts 16. For parity the
// text embedding is supplied directly (no umT5), so the DiT is validated in isolation.

/// Wan DiT geometry. Defaults are the released 14B model.
public struct NFKMLXWanConfiguration: Sendable {
    public var inChannels: Int
    /// The channels the transformer predicts: `inChannels` for text to video, 16 for Wan 2.2 I2V-A14B, whose
    /// input carries a mask and the image's latents beside the noise. Introduced in InferKit 0.4.0.
    public var outChannels: Int
    public var heads: Int
    public var headDim: Int
    public var layers: Int
    public var ffnDim: Int
    public var textDim: Int
    public var freqDim: Int
    public var patchSize: [Int]                                            // (t, h, w)
    public var eps: Float
    public var ropeTheta: Float
    /// The latent width `ref_conv` patchifies into one frame of tokens ahead of the video, as the Wan 2.2
    /// Fun Control releases read a reference image (`in_dim_ref_conv`); nil for a transformer without
    /// one. Introduced in InferKit 0.4.0.
    public var referenceChannels: Int?
    /// The input width of the camera control adapter (`in_dim_control_adapter`, 24 for the Fun
    /// Control-Camera releases), nil for a transformer without one. Introduced in InferKit 0.4.0.
    public var controlAdapterChannels: Int?
    /// The pixel unshuffle factor of the camera control adapter (`downscale_factor_control_adapter`).
    /// Introduced in InferKit 0.4.0.
    public var controlAdapterDownscale: Int
    /// Whether the timestep sinusoid and the rotary angles form in float64, as the original Wan
    /// implementation computes them (Wan-Video, VideoX-Fun); diffusers forms them in float32.
    /// Introduced in InferKit 0.4.0.
    public var formsSinusoidsInFloat64: Bool
    /// The blocks a VACE transformer's control blocks feed (`vace_layers`, the eight of `[0, 5, …, 35]` in the
    /// Wan 2.2 VACE-Fun releases), nil for a transformer without VACE. Introduced in InferKit 0.4.0.
    public var vaceLayers: [Int]?
    /// The VACE context's channels (`vace_in_dim`): the inactive and reactive latents and the folded mask, 96
    /// over the Wan 2.1 autoencoder. Introduced in InferKit 0.4.0.
    public var vaceInChannels: Int

    public init(inChannels: Int = 16, heads: Int = 40, headDim: Int = 128, layers: Int = 40,
                ffnDim: Int = 13824, textDim: Int = 4096, freqDim: Int = 256,
                patchSize: [Int] = [1, 2, 2], eps: Float = 1e-6, ropeTheta: Float = 10000.0,
                outChannels: Int? = nil, referenceChannels: Int? = nil, controlAdapterChannels: Int? = nil,
                controlAdapterDownscale: Int = 8, formsSinusoidsInFloat64: Bool = false, vaceLayers: [Int]? = nil,
                vaceInChannels: Int = 96) {
        self.vaceLayers = vaceLayers
        self.vaceInChannels = vaceInChannels
        self.referenceChannels = referenceChannels
        self.controlAdapterChannels = controlAdapterChannels
        self.controlAdapterDownscale = controlAdapterDownscale
        self.formsSinusoidsInFloat64 = formsSinusoidsInFloat64
        self.inChannels = inChannels
        self.outChannels = outChannels ?? inChannels
        self.heads = heads
        self.headDim = headDim
        self.layers = layers
        self.ffnDim = ffnDim
        self.textDim = textDim
        self.freqDim = freqDim
        self.patchSize = patchSize
        self.eps = eps
        self.ropeTheta = ropeTheta
    }

    public static let base = NFKMLXWanConfiguration()

    public static let tiny = NFKMLXWanConfiguration(
        inChannels: 4, heads: 2, headDim: 16, layers: 2, ffnDim: 48, textDim: 10, patchSize: [1, 2, 2])

    var innerDim: Int { heads * headDim }
    /// The rotary axis widths the reference derives: `h = w = 2·(headDim/6)`, `t = headDim − h − w`.
    var ropeAxes: [Int] {
        let hw = 2 * (headDim / 6)
        return [headDim - 2 * hw, hw, hw]
    }
}

/// Wan attention: fused-free q/k/v, an across-heads RMS query/key norm, an optional interleaved rotary
/// (self-attention only), and softmax SDPA.
final class NFKWanAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]                         // [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    let heads: Int
    let headDim: Int
    init(_ config: NFKMLXWanConfiguration) {
        self.heads = config.heads
        self.headDim = config.headDim
        let inner = config.innerDim
        _toQ.wrappedValue = Linear(inner, inner)
        _toK.wrappedValue = Linear(inner, inner)
        _toV.wrappedValue = Linear(inner, inner)
        _toOut.wrappedValue = [Linear(inner, inner)]
        _normQ.wrappedValue = RMSNorm(dimensions: inner, eps: config.eps)
        _normK.wrappedValue = RMSNorm(dimensions: inner, eps: config.eps)
    }

    /// `x` `[N, inner]`; `context` nil for self-attention (with rotary), else the cross-attention source.
    func callAsFunction(_ x: MLXArray, context: MLXArray?, cos c: MLXArray?, sin s: MLXArray?) -> MLXArray {
        let source = context ?? x
        let n = x.dim(0), m = source.dim(0)
        var q = Self.normalized(toQ(x), normQ).reshaped([n, heads, headDim])
        var k = Self.normalized(toK(source), normK).reshaped([m, heads, headDim])
        let v = toV(source).reshaped([m, heads, headDim])
        if let c, let s {
            q = zImageApplyRope(q, cos: c, sin: s)
            k = zImageApplyRope(k, cos: c, sin: s)
        }
        let qh = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let kh = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let vh = v.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let scale = 1.0 / sqrt(Float(headDim))
        let out = NFKReferenceRounding.flashAttention(queries: qh, keys: kh, values: vh, scale: scale, mask: nil)
        let merged = out[0].transposed(1, 0, 2).reshaped([n, heads * headDim])
        return (toOut[0] as! Linear)(merged)
    }

    /// The query and key norm, torch's `nn.RMSNorm`: on a half-precision input it normalizes and scales
    /// in float32 and rounds once.
    static func normalized(_ x: MLXArray, _ norm: RMSNorm) -> MLXArray {
        guard NFKReferenceRounding.isReduced(x) else { return norm(x) }
        return NFKReferenceRounding.scaledNorm(x, weight: norm.weight, eps: norm.eps)
    }
}

/// A `GELU(approximate: tanh)` projection, the first entry of the reference's feed-forward `net`.
final class NFKWanGELU: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ dim: Int, _ hidden: Int) { _proj.wrappedValue = Linear(dim, hidden) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { NFKReferenceRounding.geluTanh(proj(x)) }
}

/// The feed-forward: `net.0` is the gelu projection, `net.2` the output linear.
final class NFKWanFeedForward: Module {
    @ModuleInfo(key: "net") var net: [Module]                             // [NFKWanGELU, marker, Linear]

    init(dim: Int, hidden: Int) {
        _net.wrappedValue = [NFKWanGELU(dim, hidden), Module(), Linear(hidden, dim)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (net[2] as! Linear)((net[0] as! NFKWanGELU)(x))
    }
}

/// A Wan block: self-attention with rotary, cross-attention to the text, a gelu feed-forward, under the
/// shared timestep modulation plus the block's `scale_shift_table`.
final class NFKWanBlock: Module {
    @ModuleInfo(key: "attn1") var attn1: NFKWanAttention
    @ModuleInfo(key: "attn2") var attn2: NFKWanAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: NFKWanFeedForward
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    /// A VACE control block's projection of the context ahead of its first block (`before_proj`).
    @ModuleInfo(key: "before_proj") var beforeProj: Linear?
    /// A VACE control block's projection of its output into the hint it hands its main block (`after_proj`).
    @ModuleInfo(key: "after_proj") var afterProj: Linear?

    let norm1: LayerNorm
    let norm3: LayerNorm
    let eps: Float

    /// A block; `vaceIndex` makes it a VACE control block, the first of which also projects the context.
    init(_ config: NFKMLXWanConfiguration, vaceIndex: Int? = nil) {
        eps = config.eps
        if let vaceIndex {
            _afterProj.wrappedValue = Linear(config.innerDim, config.innerDim)
            if vaceIndex == 0 {
                _beforeProj.wrappedValue = Linear(config.innerDim, config.innerDim)
            }
        }
        _attn1.wrappedValue = NFKWanAttention(config)
        _attn2.wrappedValue = NFKWanAttention(config)
        _norm2.wrappedValue = LayerNorm(dimensions: config.innerDim, eps: config.eps, affine: true)
        _ffn.wrappedValue = NFKWanFeedForward(dim: config.innerDim, hidden: config.ffnDim)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([1, 6, config.innerDim])
        self.norm1 = LayerNorm(dimensions: config.innerDim, eps: config.eps, affine: false)
        self.norm3 = LayerNorm(dimensions: config.innerDim, eps: config.eps, affine: false)
    }

    /// `x` `[N, inner]`, `context` `[Lc, inner]`, `temb` `[6, inner]`. `firstFrame`, where given, is the
    /// modulation `[6, inner]` of the tokens `mask` (`[N, 1]`) marks, which read another timestep.
    /// `tokens`, where given, replaces `temb` with a modulation per token: row `index[n]` of `proj`
    /// (`[K, 6, inner]`) for token `n`.
    func callAsFunction(_ x: MLXArray, context: MLXArray, temb: MLXArray, cos c: MLXArray, sin s: MLXArray,
                        firstFrame: (temb: MLXArray, mask: MLXArray)? = nil,
                        tokens: NFKWanTokenModulation? = nil) -> MLXArray {
        if NFKReferenceRounding.isReduced(x) {
            return reduced(x, context: context, temb: temb, cos: c, sin: s, firstFrame: firstFrame, tokens: tokens)
        }
        let mod = tokens.map { $0.rows(adding: scaleShiftTable[0]) }
            ?? Self.modulation(scaleShiftTable[0] + temb,
                               firstFrame: firstFrame.map { (scaleShiftTable[0] + $0.temb, $0.mask) })
        let shiftMSA = mod(0), scaleMSA = mod(1), gateMSA = mod(2)
        let cShiftMSA = mod(3), cScaleMSA = mod(4), cGateMSA = mod(5)

        var h = x + attn1(norm1(x) * (1.0 + scaleMSA) + shiftMSA, context: nil, cos: c, sin: s) * gateMSA
        h = h + attn2(norm2(h), context: context, cos: nil, sin: nil)
        h = h + ffn(norm3(h) * (1.0 + cScaleMSA) + cShiftMSA) * cGateMSA
        return h
    }

    /// The block on a half-precision stream, rounded where the reference rounds: the modulation, the
    /// norms, and the gated residual sums are float32, each step rounded once to the stream's type.
    private func reduced(_ x: MLXArray, context: MLXArray, temb: MLXArray, cos c: MLXArray, sin s: MLXArray,
                         firstFrame: (temb: MLXArray, mask: MLXArray)?, tokens: NFKWanTokenModulation?) -> MLXArray {
        let dtype = x.dtype
        let table = scaleShiftTable[0].asType(.float32)
        let row = tokens.map { $0.rows(adding: table, as: .float32) }
            ?? Self.modulation(table + temb.asType(.float32),
                               firstFrame: firstFrame.map { (table + $0.temb.asType(.float32), $0.mask) })
        func normed(_ y: MLXArray) -> MLXArray {
            NFKReferenceRounding.layerNorm(y.asType(.float32), weight: nil, bias: nil, eps: eps)
        }
        let attended = attn1((normed(x) * (1 + row(1)) + row(0)).asType(dtype), context: nil, cos: c, sin: s)
        var h = (x.asType(.float32) + attended.asType(.float32) * row(2)).asType(dtype)
        let crossNormed = NFKReferenceRounding.layerNorm(h.asType(.float32), weight: norm2.weight?.asType(.float32),
                                                         bias: norm2.bias?.asType(.float32), eps: eps).asType(dtype)
        h = h + attn2(crossNormed, context: context, cos: nil, sin: nil)
        let fed = ffn((normed(h) * (1 + row(4)) + row(3)).asType(dtype))
        return (h.asType(.float32) + fed.asType(.float32) * row(5)).asType(dtype)
    }
}

extension NFKWanBlock {
    /// Row `i` of a block's modulation `[6, inner]` as `[1, inner]`, or, with a first-frame modulation, as
    /// `[N, inner]` taking the first frame's row where `mask` is set.
    static func modulation(_ mod: MLXArray, firstFrame: (MLXArray, MLXArray)?) -> (Int) -> MLXArray {
        let inner = mod.dim(-1)
        return { index in
            let row = mod[index].reshaped([1, inner])
            guard let (first, mask) = firstFrame else { return row }
            return which(mask, first[index].reshaped([1, inner]), row)
        }
    }
}

/// A modulation per token: `proj` holds the modulation of each distinct timestep (`[K, rows, inner]`), and
/// `index` (`[N]`) names the timestep each token reads, so a clip whose tokens read a few timesteps never
/// holds a modulation per token.
struct NFKWanTokenModulation {
    let proj: MLXArray
    let index: MLXArray

    /// Row `i` of every token's modulation plus `table` (`[rows, inner]`), as `[N, inner]`.
    func rows(adding table: MLXArray, as dtype: DType? = nil) -> (Int) -> MLXArray {
        { i in
            let row = dtype.map { proj[0..., i, 0...].asType($0) } ?? proj[0..., i, 0...]
            return take(table[i].reshaped([1, -1]) + row, index, axis: 0)
        }
    }
}

/// The Wan time/text condition embedder (text-to-video: no image branch).
final class NFKWanConditionEmbedder: Module {
    @ModuleInfo(key: "time_embedder") var timeEmbedder: NFKSANATimestepMLP
    @ModuleInfo(key: "time_proj") var timeProj: Linear
    @ModuleInfo(key: "text_embedder") var textEmbedder: NFKSANATextProjection

    init(_ config: NFKMLXWanConfiguration) {
        _timeEmbedder.wrappedValue = NFKSANATimestepMLP(config.innerDim)
        _timeProj.wrappedValue = Linear(config.innerDim, 6 * config.innerDim)
        _textEmbedder.wrappedValue = NFKSANATextProjection(inFeatures: config.textDim, hidden: config.innerDim)
    }

    /// `t` scalar, `text` `[Lc, textDim]` → `(temb [inner], timestepProj [6·inner], context [Lc, inner])`.
    func callAsFunction(_ t: MLXArray, text: MLXArray, freqDim: Int) -> (temb: MLXArray, proj: MLXArray, context: MLXArray) {
        let (temb, proj) = timestep(t, dtype: text.dtype, freqDim: freqDim)
        return (temb, proj, textEmbedder(text))                           // context [Lc, inner]
    }

    /// The timestep embedding `[inner]` and its projection `[6·inner]` of a scalar `t`.
    func timestep(_ t: MLXArray, dtype: DType, freqDim: Int) -> (temb: MLXArray, proj: MLXArray) {
        // The float32 sinusoids take the embedder's type, and its output the text's, as the reference casts them.
        let sinusoid = ltxTimestepEmbedding(t, channels: freqDim)         // cos-first
        let temb = timeEmbedder(sinusoid.asType(timeEmbedder.linear1.weight.dtype)).asType(dtype)  // [1, inner]
        let proj = timeProj(NFKReferenceRounding.silu(temb))              // [1, 6·inner]
        return (temb[0], proj[0])
    }

    /// The timestep embeddings `[K, inner]` and projections `[K, 6·inner]` of `timesteps`, their sinusoids
    /// formed in float64 as the original Wan implementation forms them (`sinusoidal_embedding_1d`) and
    /// rounded to float32.
    func timesteps(float64 timesteps: [Double], dtype: DType, freqDim: Int) -> (temb: MLXArray, proj: MLXArray) {
        let half = freqDim / 2
        let frequencies = (0 ..< half).map { Foundation.pow(10000.0, -Double($0) / Double(half)) }
        var sinusoid = [Float]()
        sinusoid.reserveCapacity(timesteps.count * freqDim)
        for t in timesteps {
            sinusoid += frequencies.map { Float(Foundation.cos(t * $0)) }
            sinusoid += frequencies.map { Float(Foundation.sin(t * $0)) }
        }
        let embedded = MLXArray(sinusoid).reshaped([timesteps.count, freqDim])
        let temb = timeEmbedder(embedded.asType(timeEmbedder.linear1.weight.dtype)).asType(dtype)
        return (temb, timeProj(NFKReferenceRounding.silu(temb)))
    }
}

/// A residual block of two 3 × 3 convolutions around a ReLU, the camera control adapter's
/// feature extractor (VideoX-Fun `wan_camera_adapter.ResidualBlock`).
final class NFKWanAdapterResidualBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d

    init(_ dim: Int) {
        _conv1.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, padding: 1)
    }

    /// `x` `[B, H, W, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv2(relu(conv1(x))) + x }
}

/// The camera control adapter (VideoX-Fun `SimpleAdapter`): each frame of the camera embedding is pixel
/// unshuffled by `downscale`, patchified to the transformer's width, and passed through one residual
/// block. Its output adds to the patch embedding token for token.
final class NFKWanControlAdapter: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "residual_blocks") var residualBlocks: [NFKWanAdapterResidualBlock]

    let downscale: Int

    init(inChannels: Int, dim: Int, patch: (Int, Int), downscale: Int) {
        self.downscale = downscale
        _conv.wrappedValue = Conv2d(inputChannels: inChannels * downscale * downscale, outputChannels: dim,
                                    kernelSize: IntOrPair([patch.0, patch.1]), stride: IntOrPair([patch.0, patch.1]))
        _residualBlocks.wrappedValue = [NFKWanAdapterResidualBlock(dim)]
    }

    /// `x` `[C, F, H, W]` → `[F, H / (downscale · patch), W / (downscale · patch), dim]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = x.dim(0), f = x.dim(1), h = x.dim(2), w = x.dim(3), r = downscale
        // torch's PixelUnshuffle orders the unshuffled channels (c, ry, rx).
        let unshuffled = x.transposed(1, 0, 2, 3).reshaped([f, c, h / r, r, w / r, r])
            .transposed(0, 2, 4, 1, 3, 5).reshaped([f, h / r, w / r, c * r * r])
        var out = conv(unshuffled)
        for block in residualBlocks {
            out = block(out)
        }
        return out
    }
}

/// What a conditioned Wan pass reads beyond the latent, the text, and the timestep: the VideoX-Fun
/// Fun releases' conditions.
struct NFKWanConditioning {
    /// The timestep of each video token, as distinct values and each token's index into them; nil for one
    /// timestep for every token.
    var tokenTimesteps: (values: [Double], index: MLXArray)?
    /// The reference image's latent `[referenceChannels, H, W]`, patchified by `ref_conv` into a frame of
    /// tokens ahead of the video.
    var reference: MLXArray?
    /// The camera embedding `[controlAdapterChannels, F, H · downscale, W · downscale]` the control
    /// adapter reads.
    var cameraControl: MLXArray?
    /// A VACE transformer's context `[vaceInChannels, F, H, W]`, read by its control blocks.
    var vaceContext: MLXArray?
    /// The weight of the VACE hints (`vace_context_scale`).
    var vaceScale: Float = 1
}

/// The Wan transformer.
public final class NFKMLXWanTransformerNet: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv3d
    @ModuleInfo(key: "condition_embedder") var conditionEmbedder: NFKWanConditionEmbedder
    @ModuleInfo(key: "blocks") var blocks: [NFKWanBlock]
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "ref_conv") var refConv: Conv2d?
    @ModuleInfo(key: "control_adapter") var controlAdapter: NFKWanControlAdapter?
    @ModuleInfo(key: "vace_blocks") var vaceBlocks: [NFKWanBlock]?
    @ModuleInfo(key: "vace_patch_embedding") var vacePatchEmbedding: Conv3d?

    let config: NFKMLXWanConfiguration
    let rope: NFKZImageRope
    let normOut: LayerNorm
    /// The blocks left in the release and read in their turn on every pass, where the load streams them.
    var layerStream: NFKMLXLayerStream?

    /// Whether a block's activations are recomputed from its inputs in the backward pass rather than held
    /// from the forward one (gradient checkpointing, the community trainers' `gradient_checkpointing`). A
    /// training step then holds one block's activations at a time, at the cost of a second forward pass. The
    /// forward's values and the gradients are unchanged. Off by default.
    ///
    /// @discussion Each block runs as a custom function over its trainable arrays and its input whose
    /// backward reruns the block, ordered so the backward holds one block's intermediates at a time. The
    /// modulation, the text, and the rotary tables the block reads depend on nothing trainable and enter as
    /// constants.
    ///
    /// Introduced in InferKit 0.4.0.
    public var recomputesBlocksInBackward = false

    public init(_ config: NFKMLXWanConfiguration) {
        self.config = config
        self.rope = NFKZImageRope(axesDims: config.ropeAxes, theta: config.ropeTheta)
        _patchEmbedding.wrappedValue = Conv3d(inputChannels: config.inChannels, outputChannels: config.innerDim,
                                              kernelSize: IntOrTriple(config.patchSize),
                                              stride: IntOrTriple(config.patchSize))
        _conditionEmbedder.wrappedValue = NFKWanConditionEmbedder(config)
        _blocks.wrappedValue = (0 ..< config.layers).map { _ in NFKWanBlock(config) }
        _scaleShiftTable.wrappedValue = MLXArray.zeros([1, 2, config.innerDim])
        _projOut.wrappedValue = Linear(config.innerDim, config.outChannels * config.patchSize.reduce(1, *))
        self.normOut = LayerNorm(dimensions: config.innerDim, eps: config.eps, affine: false)
        let spatialPatch = [config.patchSize[1], config.patchSize[2]]
        _refConv.wrappedValue = config.referenceChannels.map {
            Conv2d(inputChannels: $0, outputChannels: config.innerDim, kernelSize: IntOrPair(spatialPatch),
                   stride: IntOrPair(spatialPatch))
        }
        _controlAdapter.wrappedValue = config.controlAdapterChannels.map {
            NFKWanControlAdapter(inChannels: $0, dim: config.innerDim, patch: (spatialPatch[0], spatialPatch[1]),
                                 downscale: config.controlAdapterDownscale)
        }
        _vaceBlocks.wrappedValue = config.vaceLayers.map { layers in layers.indices.map { NFKWanBlock(config, vaceIndex: $0) } }
        _vacePatchEmbedding.wrappedValue = config.vaceLayers.map { _ in
            Conv3d(inputChannels: config.vaceInChannels, outputChannels: config.innerDim,
                   kernelSize: IntOrTriple(config.patchSize), stride: IntOrTriple(config.patchSize))
        }
    }

    /// The patch embedding; on a half-precision input the convolution and its bias are summed in float32
    /// and rounded once, as torch's `conv3d` does.
    private func embedded(_ x: MLXArray) -> MLXArray {
        guard NFKReferenceRounding.isReduced(x), let bias = patchEmbedding.bias else { return patchEmbedding(x) }
        let pt = config.patchSize[0], ph = config.patchSize[1], pw = config.patchSize[2]
        return (conv3d(x.asType(.float32), patchEmbedding.weight.asType(.float32), stride: .init((pt, ph, pw)))
            + bias.asType(.float32)).asType(x.dtype)
    }

    /// Throws the first failed read of a streamed block, and clears it so the next pass reads every block
    /// again. Every pass after a failed read skips every block, so a caller holding a streamed transformer
    /// checks after each pass. Nothing is thrown where every block is held.
    func verifyStream() throws {
        guard let stream = layerStream, let failure = stream.readFailure else { return }
        stream.clearFailure()
        throw failure
    }

    /// Velocity prediction. `x` `[inChannels, F, H, W]`, `text` `[Lc, textDim]`, `t` scalar; the prediction
    /// is `[outChannels, F, H, W]`.
    public func callAsFunction(_ x: MLXArray, text: MLXArray, t: MLXArray) -> MLXArray {
        callAsFunction(x, text: text, t: t, firstFrameTimestep: nil)
    }

    /// Velocity prediction with the first latent frame's tokens at `firstFrameTimestep` and the others at `t`,
    /// as Wan 2.2 TI2V's `expand_timesteps` runs a clip that starts from an image (the first frame clean, at 0).
    func callAsFunction(_ x: MLXArray, text: MLXArray, t: MLXArray, firstFrameTimestep: MLXArray?) -> MLXArray {
        let c = config.outChannels, f = x.dim(1), h = x.dim(2), w = x.dim(3)
        let pt = config.patchSize[0], ph = config.patchSize[1], pw = config.patchSize[2]
        let ft = f / pt, ht = h / ph, wt = w / pw
        let inner = config.innerDim

        // Patch embed: NCTHW -> NDHWC Conv3d -> [N, inner].
        let ndhwc = x.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)   // [1, F, H, W, C]
        var hidden = embedded(ndhwc).reshaped([ft * ht * wt, inner])

        // Position grid (frame, height, width), no offset.
        var pf = [Float](), phh = [Float](), pww = [Float]()
        for fi in 0 ..< ft {
            for hi in 0 ..< ht {
                for wi in 0 ..< wt {
                    pf.append(Float(fi)); phh.append(Float(hi)); pww.append(Float(wi))
                }
            }
        }
        let n = ft * ht * wt
        let positions = concatenated([MLXArray(pf).reshaped([n, 1]),
                                      MLXArray(phh).reshaped([n, 1]),
                                      MLXArray(pww).reshaped([n, 1])], axis: 1)
        let (cosT, sinT) = rope.table(positions: positions)

        let (temb, proj, context) = conditionEmbedder(t, text: text, freqDim: config.freqDim)
        let blockMod = proj.reshaped([6, inner])                          // shared across blocks
        let firstFrame = firstFrameTimestep.map { timestep -> (temb: MLXArray, proj: MLXArray, mask: MLXArray) in
            let (firstTemb, firstProj) = conditionEmbedder.timestep(timestep, dtype: text.dtype, freqDim: config.freqDim)
            let mask = MLXArray(0 ..< (ft * ht * wt)).reshaped([-1, 1]) .< (ht * wt)
            return (firstTemb, firstProj.reshaped([6, inner]), mask)
        }

        for (index, block) in blocks.enumerated() {
            let input = hidden
            let run = { (x: MLXArray) in
                block(x, context: context, temb: blockMod, cos: cosT, sin: sinT,
                      firstFrame: firstFrame.map { ($0.proj, $0.mask) })
            }
            guard let output = NFKMLXLayerStream.run(layerStream, layer: block, index: index, {
                recomputesBlocksInBackward && layerStream == nil
                    ? NFKMLXCustomFunction.recomputed(block) { [run($0[0])] }([input])[0] : run(input)
            }, keeping: { [$0] }) else {
                continue
            }
            hidden = output
        }

        // Output modulation from the shared timestep embedding.
        let outMod = NFKWanBlock.modulation(scaleShiftTable[0] + temb.reshaped([1, inner]),       // [2, inner]
                                            firstFrame: firstFrame.map { (scaleShiftTable[0] + $0.temb.reshaped([1, inner]), $0.mask) })
        let shift = outMod(0), scale = outMod(1)
        if NFKReferenceRounding.isReduced(hidden) {
            // The reference normalizes in float32 and rounds the modulated result once.
            hidden = (NFKReferenceRounding.layerNorm(hidden.asType(.float32), weight: nil, bias: nil, eps: config.eps)
                * (1.0 + scale) + shift).asType(hidden.dtype)
        } else {
            hidden = normOut(hidden) * (1.0 + scale) + shift
        }
        let out = projOut(hidden)                                         // [N, C·pt·ph·pw]

        // Unpatchify: [ft·ht·wt, C·pt·ph·pw] -> [C, F, H, W].
        let grid = out.reshaped([ft, ht, wt, pt, ph, pw, c])
        // f h w pt ph pw c -> c (f pt) (h ph) (w pw)
        return grid.transposed(6, 0, 3, 1, 4, 2, 5).reshaped([c, f, h, w])
    }

    /// The (frame, height, width) position of every token of a `frames × height × width` grid, `[N, 3]`.
    static func gridPositions(frames: Int, height: Int, width: Int) -> MLXArray {
        let n = frames * height * width
        let index = MLXArray(0 ..< n).asType(.int32)
        let hw = height * width
        return stacked([index / hw, (index % hw) / width, index % width], axis: 1).asType(.float32)
    }

    /// The rotary tables of a `frames × height × width` grid with the angles formed in float64, as the
    /// original implementation's `rope_params` forms them.
    func float64RopeTables(frames: Int, height: Int, width: Int) -> (cos: MLXArray, sin: MLXArray) {
        let axes = config.ropeAxes
        let inverse = axes.map { d in
            stride(from: 0, to: d, by: 2).map { 1.0 / Foundation.pow(Double(config.ropeTheta), Double($0) / Double(d)) }
        }
        let width64 = inverse.reduce(0) { $0 + $1.count }
        var cosines = [Float](), sines = [Float]()
        cosines.reserveCapacity(frames * height * width * width64)
        sines.reserveCapacity(frames * height * width * width64)
        for fi in 0 ..< frames {
            for hi in 0 ..< height {
                for wi in 0 ..< width {
                    for (axis, position) in [fi, hi, wi].enumerated() {
                        for frequency in inverse[axis] {
                            let angle = Double(position) * frequency
                            cosines.append(Float(Foundation.cos(angle)))
                            sines.append(Float(Foundation.sin(angle)))
                        }
                    }
                }
            }
        }
        let n = frames * height * width
        return (MLXArray(cosines).reshaped([n, width64]), MLXArray(sines).reshaped([n, width64]))
    }

    /// Velocity prediction under the VideoX-Fun Fun releases' conditions, the original implementation's
    /// forward. `x` `[inChannels, F, H, W]` (the noise and every condition channel), `text` `[Lc, textDim]`;
    /// the prediction is `[outChannels, F, H, W]`.
    ///
    /// @discussion The reference image's tokens, where given, form a frame ahead of the video: they take
    /// rotary frame 0, shift the video's frames by one, read the timestep of the last video token (the
    /// original pads the per-token timesteps by repeating the last), and leave before the head's output is
    /// unpatchified. The camera adapter's output adds to the patch embedding.
    /// The hint each VACE-fed block adds after it, keyed by block (`forward_vace`): the context patchified by
    /// `vace_patch_embedding`, the first control block reading it projected (`before_proj`) plus the clip's
    /// tokens, each later one the previous block's output, every block run as a Wan block over the same
    /// modulation, text, and rotary tables, and each output projected by `after_proj` into its hint.
    func vaceHints(_ conditioning: NFKWanConditioning, hidden: MLXArray, context: MLXArray, proj: MLXArray,
                   cos: MLXArray, sin: MLXArray, tokens: NFKWanTokenModulation) -> [Int: MLXArray] {
        guard let vaceBlocks, let vacePatchEmbedding, let layers = config.vaceLayers,
              let vaceContext = conditioning.vaceContext else { return [:] }
        let embedded = vacePatchEmbedding(vaceContext.transposed(1, 2, 3, 0).expandedDimensions(axis: 0).asType(hidden.dtype))
        var c = embedded.reshaped([-1, config.innerDim])
        var hints = [Int: MLXArray]()
        for (index, block) in vaceBlocks.enumerated() {
            if let beforeProj = block.beforeProj {
                c = beforeProj(c) + hidden
            }
            c = block(c, context: context, temb: proj, cos: cos, sin: sin, tokens: tokens)
            if let afterProj = block.afterProj {
                hints[layers[index]] = afterProj(c)
            }
        }
        return hints
    }

    func callAsFunction(_ x: MLXArray, text: MLXArray, t: Double, conditioning: NFKWanConditioning) -> MLXArray {
        let c = config.outChannels, f = x.dim(1), h = x.dim(2), w = x.dim(3)
        let pt = config.patchSize[0], ph = config.patchSize[1], pw = config.patchSize[2]
        let ft = f / pt, ht = h / ph, wt = w / pw
        let inner = config.innerDim
        let videoTokens = ft * ht * wt

        var embeddedVideo = embedded(x.transposed(1, 2, 3, 0).expandedDimensions(axis: 0))   // [1, ft, ht, wt, inner]
        if let controlAdapter, let camera = conditioning.cameraControl {
            embeddedVideo = embeddedVideo + controlAdapter(camera).expandedDimensions(axis: 0)
        }
        var hidden = embeddedVideo.reshaped([videoTokens, inner])

        var values = [t]
        var index = MLXArray.zeros([videoTokens], dtype: .int32)
        if let tokenTimesteps = conditioning.tokenTimesteps {
            values = tokenTimesteps.values
            index = tokenTimesteps.index.asType(.int32)
        }
        var frames = ft
        var referenceTokens = 0
        if let refConv, let reference = conditioning.reference {
            referenceTokens = ht * wt
            let tokens = refConv(reference.transposed(1, 2, 0).expandedDimensions(axis: 0)).reshaped([referenceTokens, inner])
            hidden = concatenated([tokens, hidden], axis: 0)
            index = concatenated([repeated(index[(videoTokens - 1) ..< videoTokens], count: referenceTokens, axis: 0), index])
            frames += 1
        }

        let (cosT, sinT) = config.formsSinusoidsInFloat64
            ? float64RopeTables(frames: frames, height: ht, width: wt)
            : rope.table(positions: Self.gridPositions(frames: frames, height: ht, width: wt))

        let (temb, proj): (MLXArray, MLXArray)
        if config.formsSinusoidsInFloat64 {
            (temb, proj) = conditionEmbedder.timesteps(float64: values, dtype: text.dtype, freqDim: config.freqDim)
        } else {
            let each = values.map { conditionEmbedder.timestep(MLXArray(Float($0)), dtype: text.dtype, freqDim: config.freqDim) }
            (temb, proj) = (stacked(each.map(\.temb)), stacked(each.map(\.proj)))
        }
        let tokens = NFKWanTokenModulation(proj: proj.reshaped([values.count, 6, inner]), index: index)
        let context = conditionEmbedder.textEmbedder(text)
        let hints = vaceHints(conditioning, hidden: hidden, context: context, proj: proj, cos: cosT, sin: sinT, tokens: tokens)

        for (blockIndex, block) in blocks.enumerated() {
            let input = hidden
            let run = { (x: MLXArray) in
                block(x, context: context, temb: proj, cos: cosT, sin: sinT, tokens: tokens)
            }
            guard let output = NFKMLXLayerStream.run(layerStream, layer: block, index: blockIndex, {
                recomputesBlocksInBackward && layerStream == nil
                    ? NFKMLXCustomFunction.recomputed(block) { [run($0[0])] }([input])[0] : run(input)
            }, keeping: { [$0] }) else {
                continue
            }
            hidden = output
            if let hint = hints[blockIndex] {
                hidden = hidden + hint * conditioning.vaceScale
            }
        }

        let head = NFKWanTokenModulation(proj: temb.reshaped([values.count, 1, inner]), index: index)
        let shift = head.rows(adding: scaleShiftTable[0][0 ..< 1])(0)
        let scale = head.rows(adding: scaleShiftTable[0][1 ..< 2])(0)
        if NFKReferenceRounding.isReduced(hidden) {
            hidden = (NFKReferenceRounding.layerNorm(hidden.asType(.float32), weight: nil, bias: nil, eps: config.eps)
                * (1.0 + scale.asType(.float32)) + shift.asType(.float32)).asType(hidden.dtype)
        } else {
            hidden = normOut(hidden) * (1.0 + scale) + shift
        }
        let out = projOut(hidden[referenceTokens...])

        let grid = out.reshaped([ft, ht, wt, pt, ph, pw, c])
        return grid.transposed(6, 0, 3, 1, 4, 2, 5).reshaped([c, f, h, w])
    }
}
