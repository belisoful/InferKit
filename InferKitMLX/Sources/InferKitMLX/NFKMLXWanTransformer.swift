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
// This is the text-to-video path (no image-conditioning branch). For parity the text embedding is
// supplied directly (no umT5), so the DiT is validated in isolation.

/// Wan DiT geometry. Defaults are the released 14B model.
public struct NFKMLXWanConfiguration: Sendable {
    public var inChannels: Int
    public var heads: Int
    public var headDim: Int
    public var layers: Int
    public var ffnDim: Int
    public var textDim: Int
    public var freqDim: Int
    public var patchSize: [Int]                                            // (t, h, w)
    public var eps: Float
    public var ropeTheta: Float

    public init(inChannels: Int = 16, heads: Int = 40, headDim: Int = 128, layers: Int = 40,
                ffnDim: Int = 13824, textDim: Int = 4096, freqDim: Int = 256,
                patchSize: [Int] = [1, 2, 2], eps: Float = 1e-6, ropeTheta: Float = 10000.0) {
        self.inChannels = inChannels
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
        var q = normQ(toQ(x)).reshaped([n, heads, headDim])
        var k = normK(toK(source)).reshaped([m, heads, headDim])
        let v = toV(source).reshaped([m, heads, headDim])
        if let c, let s {
            q = zImageApplyRope(q, cos: c, sin: s)
            k = zImageApplyRope(k, cos: c, sin: s)
        }
        let qh = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let kh = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let vh = v.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let scale = 1.0 / sqrt(Float(headDim))
        let out = MLXFast.scaledDotProductAttention(queries: qh, keys: kh, values: vh, scale: scale, mask: .none)
        let merged = out[0].transposed(1, 0, 2).reshaped([n, heads * headDim])
        return (toOut[0] as! Linear)(merged)
    }
}

/// A `GELU(approximate: tanh)` projection, the first entry of the reference's feed-forward `net`.
final class NFKWanGELU: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ dim: Int, _ hidden: Int) { _proj.wrappedValue = Linear(dim, hidden) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { geluApproximate(proj(x)) }
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

    let norm1: LayerNorm
    let norm3: LayerNorm

    init(_ config: NFKMLXWanConfiguration) {
        _attn1.wrappedValue = NFKWanAttention(config)
        _attn2.wrappedValue = NFKWanAttention(config)
        _norm2.wrappedValue = LayerNorm(dimensions: config.innerDim, eps: config.eps, affine: true)
        _ffn.wrappedValue = NFKWanFeedForward(dim: config.innerDim, hidden: config.ffnDim)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([1, 6, config.innerDim])
        self.norm1 = LayerNorm(dimensions: config.innerDim, eps: config.eps, affine: false)
        self.norm3 = LayerNorm(dimensions: config.innerDim, eps: config.eps, affine: false)
    }

    /// `x` `[N, inner]`, `context` `[Lc, inner]`, `temb` `[6, inner]`.
    func callAsFunction(_ x: MLXArray, context: MLXArray, temb: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
        let inner = x.dim(1)
        let mod = scaleShiftTable[0] + temb                               // [6, inner]
        let shiftMSA = mod[0].reshaped([1, inner]), scaleMSA = mod[1].reshaped([1, inner])
        let gateMSA = mod[2].reshaped([1, inner])
        let cShiftMSA = mod[3].reshaped([1, inner]), cScaleMSA = mod[4].reshaped([1, inner])
        let cGateMSA = mod[5].reshaped([1, inner])

        var h = x + attn1(norm1(x) * (1.0 + scaleMSA) + shiftMSA, context: nil, cos: c, sin: s) * gateMSA
        h = h + attn2(norm2(h), context: context, cos: nil, sin: nil)
        h = h + ffn(norm3(h) * (1.0 + cScaleMSA) + cShiftMSA) * cGateMSA
        return h
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
        let sinusoid = ltxTimestepEmbedding(t, channels: freqDim)         // cos-first
        let temb = timeEmbedder(sinusoid)                                 // [1, inner]
        let proj = timeProj(silu(temb))                                   // [1, 6·inner]
        let context = textEmbedder(text)                                  // [Lc, inner]
        return (temb[0], proj[0], context)
    }
}

/// The Wan transformer.
public final class NFKMLXWanTransformerNet: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv3d
    @ModuleInfo(key: "condition_embedder") var conditionEmbedder: NFKWanConditionEmbedder
    @ModuleInfo(key: "blocks") var blocks: [NFKWanBlock]
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let config: NFKMLXWanConfiguration
    let rope: NFKZImageRope
    let normOut: LayerNorm

    public init(_ config: NFKMLXWanConfiguration) {
        self.config = config
        self.rope = NFKZImageRope(axesDims: config.ropeAxes, theta: config.ropeTheta)
        _patchEmbedding.wrappedValue = Conv3d(inputChannels: config.inChannels, outputChannels: config.innerDim,
                                              kernelSize: IntOrTriple(config.patchSize),
                                              stride: IntOrTriple(config.patchSize))
        _conditionEmbedder.wrappedValue = NFKWanConditionEmbedder(config)
        _blocks.wrappedValue = (0 ..< config.layers).map { _ in NFKWanBlock(config) }
        _scaleShiftTable.wrappedValue = MLXArray.zeros([1, 2, config.innerDim])
        _projOut.wrappedValue = Linear(config.innerDim, config.inChannels * config.patchSize.reduce(1, *))
        self.normOut = LayerNorm(dimensions: config.innerDim, eps: config.eps, affine: false)
    }

    /// Velocity prediction. `x` `[C, F, H, W]`, `text` `[Lc, textDim]`, `t` scalar.
    public func callAsFunction(_ x: MLXArray, text: MLXArray, t: MLXArray) -> MLXArray {
        let c = x.dim(0), f = x.dim(1), h = x.dim(2), w = x.dim(3)
        let pt = config.patchSize[0], ph = config.patchSize[1], pw = config.patchSize[2]
        let ft = f / pt, ht = h / ph, wt = w / pw
        let inner = config.innerDim

        // Patch embed: NCTHW -> NDHWC Conv3d -> [N, inner].
        let ndhwc = x.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)   // [1, F, H, W, C]
        var hidden = patchEmbedding(ndhwc).reshaped([ft * ht * wt, inner])

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

        for block in blocks {
            hidden = block(hidden, context: context, temb: blockMod, cos: cosT, sin: sinT)
        }

        // Output modulation from the shared timestep embedding.
        let outMod = scaleShiftTable[0] + temb.reshaped([1, inner])       // [2, inner]
        let shift = outMod[0].reshaped([1, inner]), scale = outMod[1].reshaped([1, inner])
        hidden = normOut(hidden) * (1.0 + scale) + shift
        let out = projOut(hidden)                                         // [N, C·pt·ph·pw]

        // Unpatchify: [ft·ht·wt, C·pt·ph·pw] -> [C, F, H, W].
        let grid = out.reshaped([ft, ht, wt, pt, ph, pw, c])
        // f h w pt ph pw c -> c (f pt) (h ph) (w pw)
        return grid.transposed(6, 0, 3, 1, 4, 2, 5).reshaped([c, f, h, w])
    }
}
