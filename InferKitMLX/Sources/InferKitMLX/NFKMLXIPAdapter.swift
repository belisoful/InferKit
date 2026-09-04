//
//  NFKMLXIPAdapter.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// IP-Adapter: lightweight image conditioning for a diffusion model, the cheap way to steer a Stable
// Diffusion generation with a reference image rather than only text. Two pieces: an image projection
// that maps a CLIP image embedding to a short sequence of "image text" tokens, and a DECOUPLED
// cross-attention that adds a second, image-conditioned attention beside the text cross-attention —
// `text_attn + scale·ip_attn`, sharing the query, through its own `to_k_ip` / `to_v_ip` projections.
// The projection and the extra key/value weights are the only trained parameters; the base UNet is
// frozen, so an adapter is a small file over a shipped Stable Diffusion model.

/// The base IP-Adapter image projection: a CLIP image embedding → `numTokens` cross-attention tokens.
public final class NFKMLXIPAdapterImageProjection: Module {
    @ModuleInfo(key: "image_embeds") var imageEmbeds: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    let numTokens: Int

    public init(imageEmbedDim: Int = 1024, crossAttentionDim: Int = 768, numTokens: Int = 4) {
        self.numTokens = numTokens
        _imageEmbeds.wrappedValue = Linear(imageEmbedDim, numTokens * crossAttentionDim)
        _norm.wrappedValue = LayerNorm(dimensions: crossAttentionDim)
    }

    /// `imageEmbeds` `[B, imageEmbedDim]` → `[B, numTokens, crossAttentionDim]`.
    public func callAsFunction(_ imageEmbeds: MLXArray) -> MLXArray {
        let b = imageEmbeds.dim(0)
        let projected = self.imageEmbeds(imageEmbeds).reshaped([b, numTokens, -1])
        return norm(projected)
    }
}

/// A decoupled cross-attention: the text cross-attention plus a scaled image-conditioned attention that
/// shares the query. This is the IP-Adapter mechanism the UNet's cross-attention layers gain.
public final class NFKMLXIPAdapterAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]                        // [Linear]
    @ModuleInfo(key: "to_k_ip") var toKIP: [Linear]                       // [Linear]
    @ModuleInfo(key: "to_v_ip") var toVIP: [Linear]                       // [Linear]

    let heads: Int
    let headDim: Int

    public init(queryDim: Int, crossAttentionDim: Int, heads: Int, headDim: Int) {
        self.heads = heads
        self.headDim = headDim
        let inner = heads * headDim
        _toQ.wrappedValue = Linear(queryDim, inner, bias: false)
        _toK.wrappedValue = Linear(crossAttentionDim, inner, bias: false)
        _toV.wrappedValue = Linear(crossAttentionDim, inner, bias: false)
        _toOut.wrappedValue = [Linear(inner, queryDim)]
        _toKIP.wrappedValue = [Linear(crossAttentionDim, inner, bias: false)]
        _toVIP.wrappedValue = [Linear(crossAttentionDim, inner, bias: false)]
    }

    private func split(_ t: MLXArray) -> MLXArray {
        let (b, n) = (t.dim(0), t.dim(1))
        return t.reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
    }

    /// `hidden` `[B, S, queryDim]`, `text` `[B, T, cross]`, `ipTokens` `[B, K, cross]` → `[B, S, queryDim]`.
    public func callAsFunction(_ hidden: MLXArray, text: MLXArray, ipTokens: MLXArray, scale: Float) -> MLXArray {
        let (b, s) = (hidden.dim(0), hidden.dim(1))
        let q = split(toQ(hidden))
        let textAttn = MLXFast.scaledDotProductAttention(
            queries: q, keys: split(toK(text)), values: split(toV(text)),
            scale: 1.0 / sqrt(Float(headDim)), mask: .none)
        let ipAttn = MLXFast.scaledDotProductAttention(
            queries: q, keys: split(toKIP[0](ipTokens)), values: split(toVIP[0](ipTokens)),
            scale: 1.0 / sqrt(Float(headDim)), mask: .none)
        let merged = (textAttn + scale * ipAttn).transposed(0, 2, 1, 3).reshaped([b, s, heads * headDim])
        return (toOut[0] as! Linear)(merged)
    }
}
