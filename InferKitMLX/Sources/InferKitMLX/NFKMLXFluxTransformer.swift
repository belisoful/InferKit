//
//  NFKMLXFluxTransformer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The FLUX.1 transformer (`FluxTransformer2DModel`, Black Forest Labs), the denoising transformer of the
// 12B FLUX text-to-image models, and the seventh DiT family here. It has two block kinds. The
// DOUBLE-STREAM blocks are MMDiT joint-attention blocks like SD3's (image and text each carry their own
// projections, feed-forward, and modulation; attention runs over the concatenation). The SINGLE-STREAM
// blocks concatenate the two streams and run a parallel attention-and-MLP over the join, in the modern
// LLM style (one adaptive-norm gate over `[attention ‖ mlp] → proj_out`). Position is an axial rotary
// over the (frame, height, width) token ids rather than a learned table.
//
// FLUX.1 conditions on the T5-XXL sequence embedding and the CLIP-L POOLED embedding (no CLIP-G, no CLIP
// sequence). The `[dev]` release is guidance-distilled and carries a guidance embedding; `[schnell]` is
// a four-step distillation with none. For parity the text conditioning is supplied directly, so the DiT
// is validated in isolation, as the SD3 and LTX DiTs are.

/// FLUX transformer geometry. Defaults are the released 12B model.
public struct NFKMLXFluxConfiguration: Sendable {
    public var inChannels: Int
    public var outChannels: Int
    public var numLayers: Int
    public var numSingleLayers: Int
    public var attentionHeadDim: Int
    public var numAttentionHeads: Int
    public var jointAttentionDim: Int
    public var pooledProjectionDim: Int
    public var guidanceEmbeds: Bool
    public var axesDimsRope: [Int]

    public init(inChannels: Int = 64, outChannels: Int = 64, numLayers: Int = 19,
                numSingleLayers: Int = 38, attentionHeadDim: Int = 128, numAttentionHeads: Int = 24,
                jointAttentionDim: Int = 4096, pooledProjectionDim: Int = 768,
                guidanceEmbeds: Bool = false, axesDimsRope: [Int] = [16, 56, 56]) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.numLayers = numLayers
        self.numSingleLayers = numSingleLayers
        self.attentionHeadDim = attentionHeadDim
        self.numAttentionHeads = numAttentionHeads
        self.jointAttentionDim = jointAttentionDim
        self.pooledProjectionDim = pooledProjectionDim
        self.guidanceEmbeds = guidanceEmbeds
        self.axesDimsRope = axesDimsRope
    }

    /// FLUX.1 [dev] (`black-forest-labs/FLUX.1-dev`), guidance-distilled.
    public static let dev = NFKMLXFluxConfiguration(guidanceEmbeds: true)

    /// FLUX.1 [schnell] (`black-forest-labs/FLUX.1-schnell`), the four-step distillation, no guidance.
    public static let schnell = NFKMLXFluxConfiguration(guidanceEmbeds: false)

    /// A tiny random configuration for reference parity, guidance-distilled (the larger surface): two
    /// double blocks, two single blocks, two heads of six, the three-axis rope.
    public static let tiny = NFKMLXFluxConfiguration(
        inChannels: 8, outChannels: 8, numLayers: 2, numSingleLayers: 2, attentionHeadDim: 6,
        numAttentionHeads: 2, jointAttentionDim: 24, pooledProjectionDim: 10, guidanceEmbeds: true,
        axesDimsRope: [2, 2, 2])

    var innerDim: Int { numAttentionHeads * attentionHeadDim }
    var patchOutputDim: Int { outChannels }
}

/// The axial rotary over the token ids, the reference's `FluxPosEmbed` with `get_1d_rotary_pos_embed`
/// (`repeat_interleave_real`, adjacent-pair rotation). Each axis contributes `axisDim` cos/sin
/// channels, concatenated.
struct NFKFluxRope {
    let axesDim: [Int]
    let theta: Float

    /// `ids` `[S, nAxes]` → `(cos, sin)` each `[S, sum(axesDim)]`.
    func table(ids: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        var cosParts: [MLXArray] = [], sinParts: [MLXArray] = []
        for axis in 0 ..< axesDim.count {
            let d = axesDim[axis]
            let k = MLXArray(stride(from: 0, to: d, by: 2).map { Float($0) })  // [d/2]
            let freqs = pow(MLXArray(theta), -(k / Float(d)))                   // theta^(-2k/d)
            let pos = ids[0..., axis].reshaped([-1, 1])                         // [S, 1]
            let angles = pos * freqs.reshaped([1, -1])                          // [S, d/2]
            cosParts.append(repeatInterleave2(cos(angles)))                     // [S, d]
            sinParts.append(repeatInterleave2(sin(angles)))
        }
        return (concatenated(cosParts, axis: -1), concatenated(sinParts, axis: -1))
    }

    private func repeatInterleave2(_ x: MLXArray) -> MLXArray {
        let s = x.dim(0), c = x.dim(1)
        return broadcast(x.reshaped([s, c, 1]), to: [s, c, 2]).reshaped([s, c * 2])
    }
}

/// Rotates adjacent channel pairs of `x` `[B, H, S, headDim]` by `(cos, sin)` `[S, headDim]`.
func applyFluxRotary(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
    let cc = c.reshaped([1, 1, c.dim(0), c.dim(1)])
    let ss = s.reshaped([1, 1, s.dim(0), s.dim(1)])
    let pairs = x.reshaped([x.dim(0), x.dim(1), x.dim(2), -1, 2])
    let real = pairs[0..., 0..., 0..., 0..., 0]
    let imag = pairs[0..., 0..., 0..., 0..., 1]
    let rotated = stacked([-imag, real], axis: -1).reshaped(x.shape)
    return x * cc + rotated * ss
}

/// The FLUX attention: query/key/value projections with per-head RMS query/key norm and rotary, over
/// the joint (double-block) or concatenated (single-block) sequence. The double block adds the text
/// stream through `add_*_proj` and concatenates it BEFORE the image stream (`[text, image]`), the
/// reference's order.
final class NFKFluxAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]?                       // [Linear, dropout-marker]; nil when pre_only
    @ModuleInfo(key: "add_q_proj") var addQ: Linear?
    @ModuleInfo(key: "add_k_proj") var addK: Linear?
    @ModuleInfo(key: "add_v_proj") var addV: Linear?
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear?
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm?
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm?

    let heads: Int
    let headDim: Int
    let hasAdded: Bool
    let preOnly: Bool

    init(dim: Int, heads: Int, headDim: Int, added: Bool, preOnly: Bool) {
        self.heads = heads
        self.headDim = headDim
        self.hasAdded = added
        self.preOnly = preOnly
        let inner = heads * headDim
        _toQ.wrappedValue = Linear(dim, inner)
        _toK.wrappedValue = Linear(dim, inner)
        _toV.wrappedValue = Linear(dim, inner)
        _toOut.wrappedValue = preOnly ? nil : [Linear(inner, dim), NFKSD3Marker()]
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        if added {
            _addQ.wrappedValue = Linear(dim, inner)
            _addK.wrappedValue = Linear(dim, inner)
            _addV.wrappedValue = Linear(dim, inner)
            _toAddOut.wrappedValue = Linear(inner, dim)
            _normAddedQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
            _normAddedK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        }
    }

    /// Double block: `hidden` `[B, Si, dim]`, `encoder` `[B, St, dim]` → image and text outputs. Single
    /// block: `hidden` is the concatenation, `encoder` nil → the joined output only.
    func callAsFunction(_ hidden: MLXArray, encoder: MLXArray?, cos c: MLXArray, sin s: MLXArray)
        -> (image: MLXArray, context: MLXArray?) {
        let b = hidden.dim(0)
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([b, -1, heads, headDim]).transposed(0, 2, 1, 3) }
        var q = normQ(split(toQ(hidden)))
        var k = normK(split(toK(hidden)))
        var v = split(toV(hidden))
        var textLength = 0
        if let encoder, hasAdded {
            textLength = encoder.dim(1)
            let eq = normAddedQ!(split(addQ!(encoder)))
            let ek = normAddedK!(split(addK!(encoder)))
            let ev = split(addV!(encoder))
            q = concatenated([eq, q], axis: 2)                             // text before image
            k = concatenated([ek, k], axis: 2)
            v = concatenated([ev, v], axis: 2)
        }
        q = applyFluxRotary(q, cos: c, sin: s)
        k = applyFluxRotary(k, cos: c, sin: s)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / sqrt(Float(headDim)), mask: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([b, -1, heads * headDim])
        if let encoder, hasAdded, let toOut {
            _ = encoder
            let context = merged[0..., 0 ..< textLength, 0...]
            let image = merged[0..., textLength..., 0...]
            return ((toOut[0] as! Linear)(image), toAddOut!(context))
        }
        return (merged, nil)                                              // pre_only: no output projection
    }
}

/// One FLUX double-stream (MMDiT) block: joint attention with dual-stream adaptive norms and
/// per-stream feed-forwards, plus the axial rotary.
final class NFKFluxDoubleBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: NFKSD3AdaLinear                   // 6·dim
    @ModuleInfo(key: "norm1_context") var norm1Context: NFKSD3AdaLinear   // 6·dim
    @ModuleInfo(key: "attn") var attn: NFKFluxAttention
    @ModuleInfo(key: "ff") var ff: NFKSD3FeedForward
    @ModuleInfo(key: "ff_context") var ffContext: NFKSD3FeedForward

    let dim: Int

    init(_ config: NFKMLXFluxConfiguration) {
        self.dim = config.innerDim
        _norm1.wrappedValue = NFKSD3AdaLinear(dim, 6 * dim)
        _norm1Context.wrappedValue = NFKSD3AdaLinear(dim, 6 * dim)
        _attn.wrappedValue = NFKFluxAttention(dim: dim, heads: config.numAttentionHeads,
                                              headDim: config.attentionHeadDim, added: true, preOnly: false)
        _ff.wrappedValue = NFKSD3FeedForward(dim)
        _ffContext.wrappedValue = NFKSD3FeedForward(dim)
    }

    func callAsFunction(_ hidden: MLXArray, encoder: MLXArray, temb: MLXArray, cos c: MLXArray, sin s: MLXArray)
        -> (context: MLXArray, image: MLXArray) {
        var hiddenStates = hidden
        var encoderStates = encoder
        func chunk(_ x: MLXArray, _ i: Int) -> MLXArray { x[0..., (i * dim) ..< ((i + 1) * dim)][0..., .newAxis, 0...] }

        let mod = norm1(temb)
        let shiftMSA = chunk(mod, 0), scaleMSA = chunk(mod, 1), gateMSA = chunk(mod, 2)
        let shiftMLP = chunk(mod, 3), scaleMLP = chunk(mod, 4), gateMLP = chunk(mod, 5)
        let normHidden = sd3AffineFreeLayerNorm(hiddenStates) * (1 + scaleMSA) + shiftMSA

        let cmod = norm1Context(temb)
        let cShiftMSA = chunk(cmod, 0), cScaleMSA = chunk(cmod, 1), cGateMSA = chunk(cmod, 2)
        let cShiftMLP = chunk(cmod, 3), cScaleMLP = chunk(cmod, 4), cGateMLP = chunk(cmod, 5)
        let normEncoder = sd3AffineFreeLayerNorm(encoderStates) * (1 + cScaleMSA) + cShiftMSA

        let (attnImage, attnContext) = attn(normHidden, encoder: normEncoder, cos: c, sin: s)
        hiddenStates = hiddenStates + gateMSA * attnImage
        let normFF = sd3AffineFreeLayerNorm(hiddenStates) * (1 + scaleMLP) + shiftMLP
        hiddenStates = hiddenStates + gateMLP * ff(normFF)

        encoderStates = encoderStates + cGateMSA * attnContext!
        let normContextFF = sd3AffineFreeLayerNorm(encoderStates) * (1 + cScaleMLP) + cShiftMLP
        encoderStates = encoderStates + cGateMLP * ffContext(normContextFF)
        return (encoderStates, hiddenStates)
    }
}

/// One FLUX single-stream block: the two streams are concatenated and a parallel attention-and-MLP runs
/// over the join under a single adaptive-norm gate. `[attention ‖ mlp]` is projected out and gated back
/// into the residual.
final class NFKFluxSingleBlock: Module {
    @ModuleInfo(key: "norm") var norm: NFKSD3AdaLinear                     // 3·dim: shift, scale, gate
    @ModuleInfo(key: "proj_mlp") var projMLP: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "attn") var attn: NFKFluxAttention

    let dim: Int
    let mlpHidden: Int

    init(_ config: NFKMLXFluxConfiguration) {
        self.dim = config.innerDim
        self.mlpHidden = config.innerDim * 4
        _norm.wrappedValue = NFKSD3AdaLinear(dim, 3 * dim)
        _projMLP.wrappedValue = Linear(dim, mlpHidden)
        _projOut.wrappedValue = Linear(dim + mlpHidden, dim)
        _attn.wrappedValue = NFKFluxAttention(dim: dim, heads: config.numAttentionHeads,
                                              headDim: config.attentionHeadDim, added: false, preOnly: true)
    }

    /// `hidden` `[B, Si, dim]`, `encoder` `[B, St, dim]` → the updated streams. The two are concatenated
    /// as `[text, image]`, run as one, and split back.
    func callAsFunction(_ hidden: MLXArray, encoder: MLXArray, temb: MLXArray, cos c: MLXArray, sin s: MLXArray)
        -> (context: MLXArray, image: MLXArray) {
        let textLength = encoder.dim(1)
        let joined = concatenated([encoder, hidden], axis: 1)
        let residual = joined
        let mod = norm(temb)
        let shift = mod[0..., 0 ..< dim][0..., .newAxis, 0...]
        let scale = mod[0..., dim ..< 2 * dim][0..., .newAxis, 0...]
        let gate = mod[0..., 2 * dim ..< 3 * dim][0..., .newAxis, 0...]
        let normed = sd3AffineFreeLayerNorm(joined) * (1 + scale) + shift
        let mlp = geluApproximate(projMLP(normed))
        let (attnOut, _) = attn(normed, encoder: nil, cos: c, sin: s)
        let combined = concatenated([attnOut, mlp], axis: 2)              // [B, seq, dim + mlpHidden]
        let out = residual + gate * projOut(combined)
        return (out[0..., 0 ..< textLength, 0...], out[0..., textLength..., 0...])
    }
}

/// The combined timestep (+ guidance) + pooled-text embedding, the reference's
/// `CombinedTimestep[Guidance]TextProjEmbeddings`. The timestep and guidance are scaled by 1000 by the
/// caller (the transformer's forward), matching diffusers.
final class NFKFluxTimeTextEmbed: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: NFKSD3MLP
    @ModuleInfo(key: "guidance_embedder") var guidanceEmbedder: NFKSD3MLP?
    @ModuleInfo(key: "text_embedder") var textEmbedder: NFKSD3MLP

    init(_ config: NFKMLXFluxConfiguration) {
        _timestepEmbedder.wrappedValue = NFKSD3MLP(256, config.innerDim, config.innerDim)
        _guidanceEmbedder.wrappedValue = config.guidanceEmbeds ? NFKSD3MLP(256, config.innerDim, config.innerDim) : nil
        _textEmbedder.wrappedValue = NFKSD3MLP(config.pooledProjectionDim, config.innerDim, config.innerDim)
    }

    func callAsFunction(timestep: MLXArray, guidance: MLXArray?, pooled: MLXArray) -> MLXArray {
        var conditioning = timestepEmbedder(sd3TimestepEmbedding(timestep, dimensions: 256))
        if let guidance, let guidanceEmbedder {
            conditioning = conditioning + guidanceEmbedder(sd3TimestepEmbedding(guidance, dimensions: 256))
        }
        return conditioning + textEmbedder(pooled)
    }
}

/// The FLUX.1 transformer.
public final class NFKMLXFluxTransformerNet: Module {
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: NFKFluxTimeTextEmbed
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [NFKFluxDoubleBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [NFKFluxSingleBlock]
    @ModuleInfo(key: "norm_out") var normOut: NFKSD3AdaLinear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public let config: NFKMLXFluxConfiguration
    let rope: NFKFluxRope

    public init(_ config: NFKMLXFluxConfiguration) {
        self.config = config
        self.rope = NFKFluxRope(axesDim: config.axesDimsRope, theta: 10000)
        _xEmbedder.wrappedValue = Linear(config.inChannels, config.innerDim)
        _contextEmbedder.wrappedValue = Linear(config.jointAttentionDim, config.innerDim)
        _timeTextEmbed.wrappedValue = NFKFluxTimeTextEmbed(config)
        _transformerBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in NFKFluxDoubleBlock(config) }
        _singleTransformerBlocks.wrappedValue = (0 ..< config.numSingleLayers).map { _ in NFKFluxSingleBlock(config) }
        _normOut.wrappedValue = NFKSD3AdaLinear(config.innerDim, 2 * config.innerDim)
        _projOut.wrappedValue = Linear(config.innerDim, config.patchOutputDim)
    }

    /// Velocity prediction on the PACKED latent. `hiddenStates` `[B, imgSeq, inChannels]`, `encoderHidden`
    /// `[B, txtSeq, jointDim]`, `pooled` `[B, pooledDim]`, `timestep` `[B]` (0…1), `guidance` `[B]?`,
    /// `imageIds` `[imgSeq, 3]` → `[B, imgSeq, outChannels]`.
    ///
    /// `controlnetBlockSamples` / `controlnetSingleBlockSamples` are a ControlNet's per-block residuals
    /// (``NFKMLXFluxControlNetNet``): each double- and single-stream block's image stream is offset by
    /// one, strided over the residual list with the reference's `ceil` interval (nil for plain
    /// text-to-image, which leaves the forward byte-identical).
    public func callAsFunction(_ hiddenStates: MLXArray, encoderHidden: MLXArray, pooled: MLXArray,
                               timestep: MLXArray, guidance: MLXArray?, imageIds: MLXArray,
                               controlnetBlockSamples: [MLXArray]? = nil,
                               controlnetSingleBlockSamples: [MLXArray]? = nil) -> MLXArray {
        var image = xEmbedder(hiddenStates)                                // [B, imgSeq, inner]
        let temb = timeTextEmbed(timestep: timestep * 1000, guidance: guidance.map { $0 * 1000 }, pooled: pooled)
        var context = contextEmbedder(encoderHidden)                       // [B, txtSeq, inner]

        // Axial rope over the concatenated [text, image] token ids (text ids are all zero).
        let textLength = encoderHidden.dim(1)
        let textIds = MLXArray.zeros([textLength, imageIds.dim(1)])
        let ids = concatenated([textIds, imageIds], axis: 0)
        let (cos, sin) = rope.table(ids: ids)

        for (index, block) in transformerBlocks.enumerated() {
            let (newContext, newImage) = block(image, encoder: context, temb: temb, cos: cos, sin: sin)
            context = newContext
            image = newImage
            if let residuals = controlnetBlockSamples {
                let interval = Int((Double(transformerBlocks.count) / Double(residuals.count)).rounded(.up))
                image = image + residuals[index / interval]
            }
        }
        for (index, block) in singleTransformerBlocks.enumerated() {
            let (newContext, newImage) = block(image, encoder: context, temb: temb, cos: cos, sin: sin)
            context = newContext
            image = newImage
            if let residuals = controlnetSingleBlockSamples {
                let interval = Int((Double(singleTransformerBlocks.count) / Double(residuals.count)).rounded(.up))
                image = image + residuals[index / interval]
            }
        }

        let mod = normOut(temb)                                            // [B, 2·inner]: scale, shift
        let dim = config.innerDim
        let scale = mod[0..., 0 ..< dim][0..., .newAxis, 0...]
        let shift = mod[0..., dim ..< 2 * dim][0..., .newAxis, 0...]
        image = sd3AffineFreeLayerNorm(image) * (1 + scale) + shift
        return projOut(image)                                              // [B, imgSeq, outChannels]
    }

    /// The `(0, row, col)` id grid for a packed latent of `height`×`width` (in latent-patch units), the
    /// reference's `_prepare_latent_image_ids`.
    public static func imageIds(height: Int, width: Int) -> MLXArray {
        var ids = [Float](repeating: 0, count: height * width * 3)
        var index = 0
        for h in 0 ..< height {
            for w in 0 ..< width {
                ids[index * 3 + 1] = Float(h)
                ids[index * 3 + 2] = Float(w)
                index += 1
            }
        }
        return MLXArray(ids, [height * width, 3])
    }

    /// The geometry of a released FLUX transformer, from its `transformer/config.json`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXFluxConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "FluxTransformer2DModel" {
            throw NFKMLXError.unsupportedConfiguration("this reads a FLUX transformer, not \(kind)")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        let inChannels = integer("in_channels", 64)
        let axes = (json["axes_dims_rope"] as? [Any])?.compactMap { ($0 as? NSNumber)?.intValue } ?? [16, 56, 56]
        return NFKMLXFluxConfiguration(
            inChannels: inChannels, outChannels: (json["out_channels"] as? NSNumber)?.intValue ?? inChannels,
            numLayers: integer("num_layers", 19), numSingleLayers: integer("num_single_layers", 38),
            attentionHeadDim: integer("attention_head_dim", 128),
            numAttentionHeads: integer("num_attention_heads", 24),
            jointAttentionDim: integer("joint_attention_dim", 4096),
            pooledProjectionDim: integer("pooled_projection_dim", 768),
            guidanceEmbeds: (json["guidance_embeds"] as? NSNumber)?.boolValue ?? false,
            axesDimsRope: axes)
    }

    /// Loads a released FLUX transformer directory (`transformer/`) into `net`. Every weight is at most
    /// 2-D, so no layout change is needed.
    public static func loadWeights(into net: NFKMLXFluxTransformerNet, from directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision)
        try NFKMLXWeights.apply(arrays, to: net)
    }
}
