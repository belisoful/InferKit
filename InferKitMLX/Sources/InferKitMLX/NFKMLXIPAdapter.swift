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

/// A loaded IP-Adapter: the image projection plus the per-cross-attention key/value weights attached to
/// a Stable Diffusion UNet. `conditioning(imageEmbedding:scale:)` turns a CLIP image embedding into the
/// ``NFKSDImageConditioning`` a UNet run reads.
public final class NFKMLXIPAdapter {
    /// The projection from a CLIP image embedding to the short image-token sequence the cross-attention
    /// reads.
    public let imageProjection: NFKMLXIPAdapterImageProjection

    public init(imageProjection: NFKMLXIPAdapterImageProjection) {
        self.imageProjection = imageProjection
    }

    /// Loads an adapter from the released safetensors and attaches its key/value projections to the
    /// UNet's cross-attention layers. The file holds `image_proj.{proj,norm}.*` for the projection and
    /// `ip_adapter.<N>.to_{k,v}_ip.weight` for the per-layer weights, where the indices `N`, sorted, are
    /// the cross-attention order the UNet enumerates (down blocks, then mid, then up). The UNet must
    /// already carry its base weights; this only adds the adapter's projections.
    public static func load(from url: URL, into unet: NFKMLXSDUNet, imageEmbedDim: Int = 1024,
                            numTokens: Int = 4) throws -> NFKMLXIPAdapter {
        let attentions = unet.crossAttentions
        let crossAttentionDim = attentions.first?.contextDimensions ?? 768
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let arrays = checkpoint.arrays

        // The image projection: the released `proj` linear is this module's `image_embeds`.
        let projection = NFKMLXIPAdapterImageProjection(imageEmbedDim: imageEmbedDim,
                                                        crossAttentionDim: crossAttentionDim,
                                                        numTokens: numTokens)
        let projectionWeights: [(String, MLXArray)] = arrays.compactMap { key, value in
            guard key.hasPrefix("image_proj.") else { return nil }
            let name = String(key.dropFirst("image_proj.".count))
                .replacingOccurrences(of: "proj.", with: "image_embeds.")
            return (name, value)
        }
        try NFKMLXWeights.apply(projectionWeights, to: projection)

        // The per-layer key/value: one pair per cross-attention, in the file's numeric-index order,
        // which is the down-then-mid-then-up module order the UNet enumerates.
        let indices = Set(arrays.keys.compactMap { key -> Int? in
            let suffix = ".to_k_ip.weight"
            guard key.hasPrefix("ip_adapter."), key.hasSuffix(suffix) else { return nil }
            return Int(key.dropFirst("ip_adapter.".count).dropLast(suffix.count))
        }).sorted()
        guard indices.count == attentions.count else {
            throw NFKMLXError.weightsMismatch(
                "the adapter carries \(indices.count) cross-attention layers but the UNet has "
                + "\(attentions.count); the adapter does not match this UNet")
        }
        for (attention, index) in zip(attentions, indices) {
            let keyWeight = arrays["ip_adapter.\(index).to_k_ip.weight"]
            let valueWeight = arrays["ip_adapter.\(index).to_v_ip.weight"]
            guard let keyWeight, let valueWeight else {
                throw NFKMLXError.weightsMismatch("the adapter is missing layer \(index)'s key/value")
            }
            try attention.attachIPAdapter(keyWeight: keyWeight, valueWeight: valueWeight)
        }
        return NFKMLXIPAdapter(imageProjection: projection)
    }

    /// Turns a CLIP image embedding `[batch, imageEmbedDim]` into the conditioning a UNet run reads, at
    /// the given blend `scale` (0 leaves the text generation unchanged; the adapter's own default is
    /// around 0.5–1.0).
    public func conditioning(imageEmbedding: MLXArray, scale: Float) -> NFKSDImageConditioning {
        NFKSDImageConditioning(tokens: imageProjection(imageEmbedding), scale: scale)
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
