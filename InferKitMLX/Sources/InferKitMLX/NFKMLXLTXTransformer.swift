//
//  NFKMLXLTXTransformer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The LTX-Video DiT (`LTXVideoTransformer3DModel`): the denoising transformer of the video-generation
// pipeline, the stage after the VAE. It operates on the VAE's latent tokens (a flattened spatiotemporal
// grid), conditioned on a text embedding (cross-attention) and a timestep (adaptive layer norm). Position
// is a 3D rotary embedding over the (frame, height, width) grid. This is the 2B model; the pipeline (a
// rectified-flow sampler) and the T5 text encoder are the remaining stages.
//
// For parity the text embedding is supplied directly (the caption projection is inside the DiT), so the
// transformer is validated in isolation, as the SD UNet is.

/// LTX DiT geometry. Defaults are the released 2B model.
public struct NFKMLXLTXTransformerConfiguration: Sendable {
    public var inChannels: Int
    public var heads: Int
    public var headDim: Int
    public var layers: Int
    public var crossAttentionDim: Int
    public var captionChannels: Int

    public init(inChannels: Int = 128, heads: Int = 32, headDim: Int = 64, layers: Int = 28,
                crossAttentionDim: Int = 2048, captionChannels: Int = 4096) {
        self.inChannels = inChannels
        self.heads = heads
        self.headDim = headDim
        self.layers = layers
        self.crossAttentionDim = crossAttentionDim
        self.captionChannels = captionChannels
    }

    public static let base = NFKMLXLTXTransformerConfiguration()

    public static let tiny = NFKMLXLTXTransformerConfiguration(
        inChannels: 16, heads: 2, headDim: 8, layers: 2, crossAttentionDim: 16, captionChannels: 32)

    var innerDim: Int { heads * headDim }
}

/// The sinusoidal timestep embedding diffusers' `Timesteps(256, flip_sin_to_cos, shift 0)` produces.
private func ltxTimestepEmbedding(_ timesteps: MLXArray, channels: Int = 256) -> MLXArray {
    let half = channels / 2
    let exponent = -log(10000.0) * MLXArray(0 ..< half).asType(.float32) / Float(half)
    let emb = timesteps.reshaped([-1, 1]).asType(.float32) * exp(exponent).reshaped([1, -1])
    return concatenated([cos(emb), sin(emb)], axis: -1)                   // flip_sin_to_cos: cos first
}

/// The `AdaLayerNormSingle` timestep conditioning: a sinusoidal embedding through an MLP gives the
/// per-token modulation (`embedded_timestep`), and a SiLU + linear expands it into the six adaptive-norm
/// parameters each block uses.
final class NFKLTXTimeEmbed: Module {
    @ModuleInfo(key: "emb") var emb: NFKLTXTimestepMLP
    @ModuleInfo(key: "linear") var linear: Linear

    init(_ innerDim: Int) {
        _emb.wrappedValue = NFKLTXTimestepMLP(innerDim)
        _linear.wrappedValue = Linear(innerDim, 6 * innerDim)
    }

    /// A timestep → `(temb [B, 6·inner], embedded [B, inner])`.
    func callAsFunction(_ timestep: MLXArray) -> (temb: MLXArray, embedded: MLXArray) {
        let embedded = emb(timestep)
        return (linear(silu(embedded)), embedded)
    }
}

/// `TimestepEmbedding`: linear, SiLU, linear over the sinusoidal timestep embedding.
final class NFKLTXTimestepMLP: Module {
    @ModuleInfo(key: "timestep_embedder") var embedder: NFKLTXLinearAct

    init(_ innerDim: Int) { _embedder.wrappedValue = NFKLTXLinearAct(256, innerDim, innerDim) }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray { embedder(ltxTimestepEmbedding(timestep)) }
}

/// A `linear_1` → SiLU → `linear_2` block (the timestep embedder's shape).
final class NFKLTXLinearAct: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ inDim: Int, _ hidden: Int, _ outDim: Int) {
        _linear1.wrappedValue = Linear(inDim, hidden)
        _linear2.wrappedValue = Linear(hidden, outDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

/// The PixArt caption projection: `linear_1` → tanh-GELU → `linear_2`, mapping the text embedding to the
/// transformer width.
final class NFKLTXCaptionProjection: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ captionChannels: Int, _ innerDim: Int) {
        _linear1.wrappedValue = Linear(captionChannels, innerDim)
        _linear2.wrappedValue = Linear(innerDim, innerDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(geluApproximate(linear1(x))) }
}

/// The 3D rotary position embedding over the (frame, height, width) latent grid.
struct NFKLTXRotary {
    let dim: Int
    let baseFrames: Float = 20, baseHeight: Float = 2048, baseWidth: Float = 2048
    let theta: Float = 10000.0

    /// `(cos, sin)` each `[1, frames·height·width, dim]` for the given latent grid and interpolation scale.
    func embedding(frames: Int, height: Int, width: Int, scale: (Float, Float, Float)) -> (MLXArray, MLXArray) {
        let sequence = frames * height * width
        // The grid coordinate of each token, scaled to the base resolution.
        var coordinates = [Float](repeating: 0, count: sequence * 3)
        var index = 0
        for f in 0 ..< frames {
            for h in 0 ..< height {
                for w in 0 ..< width {
                    coordinates[index * 3 + 0] = Float(f) * scale.0 / baseFrames
                    coordinates[index * 3 + 1] = Float(h) * scale.1 / baseHeight
                    coordinates[index * 3 + 2] = Float(w) * scale.2 / baseWidth
                    index += 1
                }
            }
        }
        let grid = coordinates.withUnsafeBufferPointer { MLXArray($0, [1, sequence, 3, 1]) }

        let count = dim / 6
        let ramp = MLXArray(0 ..< count).asType(.float32) / Float(max(count - 1, 1))
        var freqs = pow(MLXArray(theta), ramp) * (Float.pi / 2)          // [count]
        freqs = (grid * 2 - 1) * freqs.reshaped([1, 1, 1, count])         // [1, S, 3, count]
        let flattened = freqs.transposed(0, 1, 3, 2).reshaped([1, sequence, 3 * count])
        var cosFreqs = repeatedInterleave2(cos(flattened))               // [1, S, 6·count]
        var sinFreqs = repeatedInterleave2(sin(flattened))
        let padding = dim - 6 * count
        if padding > 0 {                                                 // the leading channels are unrotated
            cosFreqs = concatenated([MLXArray.ones([1, sequence, padding]), cosFreqs], axis: -1)
            sinFreqs = concatenated([MLXArray.zeros([1, sequence, padding]), sinFreqs], axis: -1)
        }
        return (cosFreqs, sinFreqs)
    }

    private func repeatedInterleave2(_ x: MLXArray) -> MLXArray {
        let (b, s, c) = (x.shape[0], x.shape[1], x.shape[2])
        return broadcast(x.reshaped([b, s, c, 1]), to: [b, s, c, 2]).reshaped([b, s, c * 2])
    }
}

/// Rotates adjacent channel pairs by the rotary `(cos, sin)`.
private func applyLTXRotary(_ x: MLXArray, _ rotary: (MLXArray, MLXArray)) -> MLXArray {
    let (cos, sin) = rotary
    let pairs = x.reshaped([x.shape[0], x.shape[1], -1, 2])
    let real = pairs[0..., 0..., 0..., 0], imaginary = pairs[0..., 0..., 0..., 1]
    let rotated = stacked([-imaginary, real], axis: -1).reshaped(x.shape)
    return x * cos + rotated * sin
}

/// LTX attention: query/key/value projections with an across-heads RMS norm on the query and key, an
/// optional rotary on both, and scaled dot-product attention. Self-attention rotates and reads itself;
/// cross-attention reads the text embedding and does not rotate.
final class NFKLTXAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    let heads: Int
    let headDim: Int

    init(queryDim: Int, contextDim: Int, heads: Int, headDim: Int) {
        self.heads = heads
        self.headDim = headDim
        let inner = heads * headDim
        _toQ.wrappedValue = Linear(queryDim, inner)
        _toK.wrappedValue = Linear(contextDim, inner)
        _toV.wrappedValue = Linear(contextDim, inner)
        _toOut.wrappedValue = [Linear(inner, queryDim)]
        _normQ.wrappedValue = RMSNorm(dimensions: inner, eps: 1e-5)
        _normK.wrappedValue = RMSNorm(dimensions: inner, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray?, rotary: (MLXArray, MLXArray)?) -> MLXArray {
        let kv = context ?? x
        var query = normQ(toQ(x))
        var key = normK(toK(kv))
        let value = toV(kv)
        if let rotary {
            query = applyLTXRotary(query, rotary)
            key = applyLTXRotary(key, rotary)
        }
        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped([t.shape[0], t.shape[1], heads, headDim]).transposed(0, 2, 1, 3)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(query), keys: split(key), values: split(value),
            scale: 1 / sqrt(Float(headDim)), mask: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([x.shape[0], x.shape[1], heads * headDim])
        return toOut[0](merged)
    }
}

/// The feed-forward: a tanh-GELU projection and a projection back.
final class NFKLTXFeedForward: Module {
    @ModuleInfo(key: "net") var net: [Module]                            // [proj(Linear), GELU marker, Linear]

    init(_ dim: Int) {
        _net.wrappedValue = [NFKLTXGEGLUProj(dim, dim * 4), NFKLTXActMarker(), Linear(dim * 4, dim)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = (net[0] as! NFKLTXGEGLUProj)(x)
        h = geluApproximate(h)
        return (net[2] as! Linear)(h)
    }
}

/// The first feed-forward entry: a `proj` linear (diffusers' `GELU` module holds its linear under `proj`).
final class NFKLTXGEGLUProj: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ inDim: Int, _ outDim: Int) { _proj.wrappedValue = Linear(inDim, outDim) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

/// A parameter-free marker occupying the activation's Sequential index.
final class NFKLTXActMarker: Module {}

/// One LTX transformer block: adaptive-norm self-attention with rotary, cross-attention to text, and an
/// adaptive-norm feed-forward. The six modulation parameters come from the timestep through the block's
/// own `scale_shift_table`.
final class NFKLTXBlock: Module {
    let norm1 = NFKLTXIdentityNorm()
    @ModuleInfo(key: "attn1") var attn1: NFKLTXAttention
    let norm2 = NFKLTXIdentityNorm()
    @ModuleInfo(key: "attn2") var attn2: NFKLTXAttention
    @ModuleInfo(key: "ff") var ff: NFKLTXFeedForward
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    init(_ c: NFKMLXLTXTransformerConfiguration) {
        _attn1.wrappedValue = NFKLTXAttention(queryDim: c.innerDim, contextDim: c.innerDim, heads: c.heads, headDim: c.headDim)
        _attn2.wrappedValue = NFKLTXAttention(queryDim: c.innerDim, contextDim: c.crossAttentionDim, heads: c.heads, headDim: c.headDim)
        _ff.wrappedValue = NFKLTXFeedForward(c.innerDim)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([6, c.innerDim])
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray, temb: MLXArray, rotary: (MLXArray, MLXArray)) -> MLXArray {
        // temb is [B, 1, 6·inner]; the block adds its table and splits into the six modulation tensors.
        let ada = scaleShiftTable.reshaped([1, 1, 6, scaleShiftTable.shape[1]])
            + temb.reshaped([temb.shape[0], temb.shape[1], 6, -1])
        let shiftMSA = ada[0..., 0..., 0], scaleMSA = ada[0..., 0..., 1], gateMSA = ada[0..., 0..., 2]
        let shiftMLP = ada[0..., 0..., 3], scaleMLP = ada[0..., 0..., 4], gateMLP = ada[0..., 0..., 5]

        var hidden = x
        var normed = norm1(hidden) * (1 + scaleMSA) + shiftMSA
        hidden = hidden + attn1(normed, context: nil, rotary: rotary) * gateMSA
        hidden = hidden + attn2(hidden, context: context, rotary: nil)
        normed = norm2(hidden) * (1 + scaleMLP) + shiftMLP
        return hidden + ff(normed) * gateMLP
    }
}

/// A parameter-free RMS/LayerNorm placeholder: the reference's affine-free norm carries no weights.
final class NFKLTXIdentityNorm {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // The reference block norms are RMSNorm(elementwise_affine=false).
        x * rsqrt(x.square().mean(axis: -1, keepDims: true) + 1e-6)
    }
}

/// The LTX-Video DiT.
final class NFKMLXLTXTransformerNet: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "time_embed") var timeEmbed: NFKLTXTimeEmbed
    @ModuleInfo(key: "caption_projection") var captionProjection: NFKLTXCaptionProjection
    @ModuleInfo(key: "transformer_blocks") var blocks: [NFKLTXBlock]
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let configuration: NFKMLXLTXTransformerConfiguration
    let rotary: NFKLTXRotary

    init(_ c: NFKMLXLTXTransformerConfiguration) {
        configuration = c
        rotary = NFKLTXRotary(dim: c.innerDim)
        _projIn.wrappedValue = Linear(c.inChannels, c.innerDim)
        _timeEmbed.wrappedValue = NFKLTXTimeEmbed(c.innerDim)
        _captionProjection.wrappedValue = NFKLTXCaptionProjection(c.captionChannels, c.innerDim)
        _blocks.wrappedValue = (0 ..< c.layers).map { _ in NFKLTXBlock(c) }
        _scaleShiftTable.wrappedValue = MLXArray.zeros([2, c.innerDim])
        _projOut.wrappedValue = Linear(c.innerDim, c.inChannels)
    }

    /// Latent tokens `[B, S, inChannels]`, text `[B, L, captionChannels]`, a timestep, and the latent grid
    /// `(frames, height, width)` → the predicted velocity `[B, S, inChannels]`.
    func callAsFunction(_ latent: MLXArray, text: MLXArray, timestep: MLXArray,
                        grid: (Int, Int, Int), ropeScale: (Float, Float, Float)) -> MLXArray {
        let rope = rotary.embedding(frames: grid.0, height: grid.1, width: grid.2, scale: ropeScale)
        var hidden = projIn(latent)
        let (temb, embedded) = timeEmbed(timestep)
        let tembTokens = temb.reshaped([latent.shape[0], 1, temb.shape[temb.ndim - 1]])
        let context = captionProjection(text)
        for block in blocks {
            hidden = block(hidden, context: context, temb: tembTokens, rotary: rope)
        }
        let ada = scaleShiftTable.reshaped([1, 1, 2, configuration.innerDim])
            + embedded.reshaped([latent.shape[0], 1, 1, configuration.innerDim])
        let shift = ada[0..., 0..., 0], scale = ada[0..., 0..., 1]
        hidden = ltxLayerNorm(hidden) * (1 + scale) + shift
        return projOut(hidden)
    }
}

/// The final affine-free LayerNorm.
private func ltxLayerNorm(_ x: MLXArray) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let variance = (x - mean).square().mean(axis: -1, keepDims: true)
    return (x - mean) * rsqrt(variance + 1e-6)
}

/// Weight loading for the LTX-Video DiT.
@objc(NFKMLXLTXTransformer)
public final class NFKMLXLTXTransformer: NSObject {

    static func makeNet(_ configuration: NFKMLXLTXTransformerConfiguration = .base) -> NFKMLXLTXTransformerNet {
        NFKMLXLTXTransformerNet(configuration)
    }

    /// Loads a checkpoint (sharded diffusers safetensors). All weights are at most 2-D, so no transpose
    /// applies; the module keys mirror the reference's `LTXVideoTransformer3DModel`.
    static func loadWeights(into net: NFKMLXLTXTransformerNet, from directory: URL) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, remap: remapReferenceKey)
        try NFKMLXWeights.apply(arrays, to: net)
    }

    /// The reference keys mirror the module names; finalized against the released checkpoint.
    static func remapReferenceKey(_ key: String) -> String? { key }
}
