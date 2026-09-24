//
//  NFKMLXFlux2Transformer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The FLUX.2 transformer (`Flux2Transformer2DModel`, Black Forest Labs). It keeps FLUX.1's two block
// kinds — double-stream MMDiT joint attention, then single-stream blocks over the concatenation — and
// changes four things.
//
// 1. MODULATION LIVES ON THE MODEL. FLUX.1 gives every block its own `norm1.linear`. FLUX.2 evaluates
//    three modulation heads once from the timestep embedding (double image, double text, single) and
//    every block of that kind reads the same vector, so the checkpoint carries three modulation
//    tensors in place of one per block.
// 2. The feed-forward is a SwiGLU whose gate is the FIRST half of one fused projection.
// 3. The single-stream block is a parallel block in the ViT-22B sense: one projection produces query,
//    key, value and both SwiGLU halves; one projection takes the attention output concatenated with
//    the gated MLP. FLUX.1 keeps `proj_mlp` and `proj_out` separate and applies a GELU.
// 4. The rotary runs over FOUR axes (time, row, column, token) at theta 2000, and the text ids carry
//    the token index in the fourth axis rather than being all zero.
//
// There is no pooled text embedding: the conditioning is the timestep plus the guidance scale. The
// released models differ in which language model writes the text sequence — FLUX.2 [dev] uses
// Mistral-Small 3 and FLUX.2 [klein] uses Qwen3, each supplying the hidden states of three layers
// concatenated per token, which is why `jointAttentionDim` is three times a language model's width.
// The text conditioning is supplied directly here, so the transformer is verified in isolation, as the
// FLUX.1, SD3 and LTX transformers are.

/// FLUX.2 transformer geometry. Defaults are the released 32B FLUX.2 [dev].
public struct NFKMLXFlux2Configuration: Sendable {
    public var inChannels: Int
    public var outChannels: Int?
    public var patchSize: Int
    public var numLayers: Int
    public var numSingleLayers: Int
    public var attentionHeadDim: Int
    public var numAttentionHeads: Int
    public var jointAttentionDim: Int
    public var timestepGuidanceChannels: Int
    public var mlpRatio: Float
    public var axesDimsRope: [Int]
    public var ropeTheta: Float
    public var eps: Float
    public var guidanceEmbeds: Bool

    public init(inChannels: Int = 128, outChannels: Int? = nil, patchSize: Int = 1,
                numLayers: Int = 8, numSingleLayers: Int = 48, attentionHeadDim: Int = 128,
                numAttentionHeads: Int = 48, jointAttentionDim: Int = 15360,
                timestepGuidanceChannels: Int = 256, mlpRatio: Float = 3.0,
                axesDimsRope: [Int] = [32, 32, 32, 32], ropeTheta: Float = 2000, eps: Float = 1e-6,
                guidanceEmbeds: Bool = true) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.patchSize = patchSize
        self.numLayers = numLayers
        self.numSingleLayers = numSingleLayers
        self.attentionHeadDim = attentionHeadDim
        self.numAttentionHeads = numAttentionHeads
        self.jointAttentionDim = jointAttentionDim
        self.timestepGuidanceChannels = timestepGuidanceChannels
        self.mlpRatio = mlpRatio
        self.axesDimsRope = axesDimsRope
        self.ropeTheta = ropeTheta
        self.eps = eps
        self.guidanceEmbeds = guidanceEmbeds
    }

    /// FLUX.2 [dev] (`black-forest-labs/FLUX.2-dev`), 32B, guidance-distilled. Its text sequence is
    /// three Mistral-Small 3 layers wide.
    public static let dev = NFKMLXFlux2Configuration()

    /// FLUX.2 [klein] 9B (`black-forest-labs/FLUX.2-klein-9B` and its BASE sibling), 9.08B, not
    /// guidance-distilled. Its text sequence is three Qwen3 layers wide at 12288.
    ///
    /// @discussion The parameter total pins 32 heads, the 12288 text width and the absent guidance
    /// embedding, but not the split between double and single blocks: a double block costs exactly
    /// two single blocks, so seventeen splits reach the same total. The split is READ from the
    /// released `transformer/config.json` — 8 double and 24 single — rather than chosen, which is
    /// why no preset existed until a 9B release was reachable.
    public static let klein9B = NFKMLXFlux2Configuration(
        numLayers: 8, numSingleLayers: 24, numAttentionHeads: 32, jointAttentionDim: 12288,
        guidanceEmbeds: false)
    // Its `transformer/config.json` is behind the repository's gate, so the split is unmeasured.

    /// FLUX.2 [klein] 4B (`black-forest-labs/FLUX.2-klein-4B`), the smallest release. Its text
    /// sequence is three Qwen3 layers wide.
    public static let klein4B = NFKMLXFlux2Configuration(
        numLayers: 5, numSingleLayers: 20, numAttentionHeads: 24, jointAttentionDim: 7680,
        guidanceEmbeds: false)

    /// A tiny random configuration for reference parity, guidance-distilled (the larger surface): two
    /// double blocks, two single blocks, two heads of eight, the four-axis rope.
    public static let tiny = NFKMLXFlux2Configuration(
        inChannels: 8, numLayers: 2, numSingleLayers: 2, attentionHeadDim: 8, numAttentionHeads: 2,
        jointAttentionDim: 24, timestepGuidanceChannels: 16, axesDimsRope: [2, 2, 2, 2],
        guidanceEmbeds: true)

    var innerDim: Int { numAttentionHeads * attentionHeadDim }
    var mlpHiddenDim: Int { Int(Float(innerDim) * mlpRatio) }
    var resolvedOutChannels: Int { outChannels ?? inChannels }
}

/// The SwiGLU of the FLUX.2 feed-forwards: the gate is the first half of the incoming projection and
/// the value the second, so the module itself holds no weights.
func flux2SwiGLU(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return NFKReferenceRounding.silu(x[.ellipsis, 0 ..< half]) * x[.ellipsis, half...]
}

/// A FLUX.2 feed-forward: one projection to twice the inner width, the SwiGLU, one projection back.
/// Every linear is bias-free.
final class NFKFlux2FeedForward: Module {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(dim: Int, inner: Int) {
        _linearIn.wrappedValue = Linear(dim, inner * 2, bias: false)
        _linearOut.wrappedValue = Linear(inner, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linearOut(flux2SwiGLU(linearIn(x))) }
}

/// A modulation head: `silu` then one bias-free projection to `3 · sets` shift/scale/gate vectors. One
/// head serves every block of its kind.
final class NFKFlux2Modulation: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    init(dim: Int, outDim: Int) { _linear.wrappedValue = Linear(dim, outDim, bias: false) }

    convenience init(dim: Int, sets: Int) { self.init(dim: dim, outDim: dim * 3 * sets) }

    func callAsFunction(_ temb: MLXArray) -> MLXArray { linear(NFKReferenceRounding.silu(temb)) }

    /// Set `index` of `mod` `[B, 3 · sets · dim]` as `(shift, scale, gate)`, each `[B, 1, dim]`.
    static func set(_ mod: MLXArray, _ index: Int, dim: Int) -> (MLXArray, MLXArray, MLXArray) {
        func part(_ offset: Int) -> MLXArray {
            let base = (index * 3 + offset) * dim
            return mod[0..., base ..< (base + dim)][0..., .newAxis, 0...]
        }
        return (part(0), part(1), part(2))
    }
}

/// What one attention does with FLUX.2 [klein] 9B KV's reference tokens.
enum NFKFlux2LayerReference {
    /// Plain joint attention.
    case none
    /// `count` reference tokens follow the text. They attend only to one another while the text and
    /// the generated tokens attend to everything, and their post-rotary keys and values are returned.
    case extract(count: Int)
    /// The keys and values an extracting pass cached, spliced between the text and the generated
    /// tokens.
    case cached(key: MLXArray, value: MLXArray)
}

/// What one block does with the reference tokens: the attention's part and, in an extracting pass,
/// the modulation the reference tokens take in place of the generated tokens'.
enum NFKFlux2BlockReference {
    case none
    case extract(count: Int, modulation: MLXArray)
    case cached(key: MLXArray, value: MLXArray)

    var attention: NFKFlux2LayerReference {
        switch self {
        case .none:
            return .none
        case .extract(let count, _):
            return .extract(count: count)
        case .cached(let key, let value):
            return .cached(key: key, value: value)
        }
    }
}

/// Attention over `[text, image]` queries, keys and values `[B, heads, S, headDim]`, with the
/// reference-cache variants of the reference's `_flux2_kv_causal_attention`. `textCount` is the
/// number of leading text positions. Returns the attended sequence and, when extracting, the
/// reference tokens' keys and values.
func flux2Attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, headDim: Int, textCount: Int,
                    reference: NFKFlux2LayerReference)
    -> (attended: MLXArray, reference: (key: MLXArray, value: MLXArray)?) {
    let scale = 1 / sqrt(Float(headDim))
    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        NFKReferenceRounding.flashAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
    }
    func positions(_ x: MLXArray, _ range: Range<Int>) -> MLXArray { x[0..., 0..., range, 0...] }

    switch reference {
    case .none:
        return (attend(q, k, v), nil)
    case .extract(let count):
        let end = textCount + count
        let length = q.dim(2)
        let others = concatenated([positions(q, 0 ..< textCount), positions(q, end ..< length)], axis: 2)
        let attendedOthers = attend(others, k, v)
        let referenceKey = positions(k, textCount ..< end)
        let referenceValue = positions(v, textCount ..< end)
        let attendedReference = attend(positions(q, textCount ..< end), referenceKey, referenceValue)
        let attended = concatenated([
            positions(attendedOthers, 0 ..< textCount), attendedReference,
            positions(attendedOthers, textCount ..< attendedOthers.dim(2)),
        ], axis: 2)
        return (attended, (referenceKey, referenceValue))
    case .cached(let key, let value):
        let length = k.dim(2)
        let keys = concatenated([positions(k, 0 ..< textCount), key, positions(k, textCount ..< length)],
                                axis: 2)
        let values = concatenated([positions(v, 0 ..< textCount), value,
                                   positions(v, textCount ..< length)], axis: 2)
        return (attend(q, keys, values), nil)
    }
}

/// A modulation vector `[B, 1, dim]` spread over `length` positions, the `count` positions from
/// `start` taking `reference` instead: the reference's `_blend_mod_params`.
func flux2BlendedModulation(_ base: MLXArray, _ reference: MLXArray, start: Int, count: Int,
                            length: Int) -> MLXArray {
    let (batch, dim) = (base.dim(0), base.dim(2))
    var parts = [MLXArray]()
    if start > 0 {
        parts.append(broadcast(base, to: [batch, start, dim]))
    }
    parts.append(broadcast(reference, to: [batch, count, dim]))
    if length > start + count {
        parts.append(broadcast(base, to: [batch, length - start - count, dim]))
    }
    return concatenated(parts, axis: 1)
}

/// The FLUX.2 joint attention of a double-stream block: per-stream query/key/value projections with
/// per-head RMS query/key norms, the text stream concatenated BEFORE the image stream, one softmax over
/// the join, and per-stream output projections. Every linear is bias-free.
final class NFKFlux2Attention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]                        // [Linear, dropout-marker]
    @ModuleInfo(key: "add_q_proj") var addQ: Linear
    @ModuleInfo(key: "add_k_proj") var addK: Linear
    @ModuleInfo(key: "add_v_proj") var addV: Linear
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm

    let heads: Int
    let headDim: Int

    init(dim: Int, heads: Int, headDim: Int, eps: Float) {
        self.heads = heads
        self.headDim = headDim
        let inner = heads * headDim
        _toQ.wrappedValue = Linear(dim, inner, bias: false)
        _toK.wrappedValue = Linear(dim, inner, bias: false)
        _toV.wrappedValue = Linear(dim, inner, bias: false)
        _toOut.wrappedValue = [Linear(inner, dim, bias: false), NFKSD3Marker()]
        _addQ.wrappedValue = Linear(dim, inner, bias: false)
        _addK.wrappedValue = Linear(dim, inner, bias: false)
        _addV.wrappedValue = Linear(dim, inner, bias: false)
        _toAddOut.wrappedValue = Linear(inner, dim, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _normAddedQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _normAddedK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
    }

    /// `hidden` `[B, Si, dim]`, `encoder` `[B, St, dim]` → the image and text outputs.
    func callAsFunction(_ hidden: MLXArray, encoder: MLXArray, cos c: MLXArray, sin s: MLXArray,
                        reference: NFKFlux2LayerReference = .none)
        -> (image: MLXArray, context: MLXArray, reference: (key: MLXArray, value: MLXArray)?) {
        let batch = hidden.dim(0)
        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped([batch, -1, heads, headDim]).transposed(0, 2, 1, 3)
        }
        let textLength = encoder.dim(1)
        var q = concatenated([normAddedQ(split(addQ(encoder))), normQ(split(toQ(hidden)))], axis: 2)
        var k = concatenated([normAddedK(split(addK(encoder))), normK(split(toK(hidden)))], axis: 2)
        let v = concatenated([split(addV(encoder)), split(toV(hidden))], axis: 2)
        q = applyFluxRotary(q, cos: c, sin: s)
        k = applyFluxRotary(k, cos: c, sin: s)
        let (attended, stored) = flux2Attention(q, k, v, headDim: headDim, textCount: textLength,
                                                reference: reference)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, -1, heads * headDim])
        let context = toAddOut(merged[0..., 0 ..< textLength, 0...])
        let image = (toOut[0] as! Linear)(merged[0..., textLength..., 0...])
        return (image, context, stored)
    }
}

/// The FLUX.2 parallel self-attention of a single-stream block: one bias-free projection produces
/// query, key, value and both SwiGLU halves, and one projection takes the attention output
/// concatenated with the gated MLP.
final class NFKFlux2ParallelAttention: Module {
    @ModuleInfo(key: "to_qkv_mlp_proj") var toQKVMLP: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    let heads: Int
    let headDim: Int
    let mlpHidden: Int

    init(dim: Int, heads: Int, headDim: Int, mlpHidden: Int, eps: Float) {
        self.heads = heads
        self.headDim = headDim
        self.mlpHidden = mlpHidden
        let inner = heads * headDim
        _toQKVMLP.wrappedValue = Linear(dim, inner * 3 + mlpHidden * 2, bias: false)
        _toOut.wrappedValue = Linear(inner + mlpHidden, dim, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
    }

    /// `textCount` is the number of leading text positions, which the reference-cache variants split on.
    func callAsFunction(_ hidden: MLXArray, cos c: MLXArray, sin s: MLXArray, textCount: Int = 0,
                        reference: NFKFlux2LayerReference = .none)
        -> (output: MLXArray, reference: (key: MLXArray, value: MLXArray)?) {
        let batch = hidden.dim(0)
        let inner = heads * headDim
        let projected = toQKVMLP(hidden)
        let qkv = projected[0..., 0..., 0 ..< (3 * inner)]
        let mlp = flux2SwiGLU(projected[0..., 0..., (3 * inner)...])
        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped([batch, -1, heads, headDim]).transposed(0, 2, 1, 3)
        }
        var q = normQ(split(qkv[0..., 0..., 0 ..< inner]))
        var k = normK(split(qkv[0..., 0..., inner ..< (2 * inner)]))
        let v = split(qkv[0..., 0..., (2 * inner)...])
        q = applyFluxRotary(q, cos: c, sin: s)
        k = applyFluxRotary(k, cos: c, sin: s)
        let (attended, stored) = flux2Attention(q, k, v, headDim: headDim, textCount: textCount,
                                                reference: reference)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, -1, inner])
        return (toOut(concatenated([merged, mlp], axis: 2)), stored)
    }
}

/// One FLUX.2 double-stream block: joint attention and per-stream feed-forwards under the model's
/// shared modulation. The block holds no modulation projection of its own.
final class NFKFlux2DoubleBlock: Module {
    @ModuleInfo(key: "attn") var attn: NFKFlux2Attention
    @ModuleInfo(key: "ff") var ff: NFKFlux2FeedForward
    @ModuleInfo(key: "ff_context") var ffContext: NFKFlux2FeedForward

    let dim: Int
    let eps: Float

    init(_ config: NFKMLXFlux2Configuration) {
        self.dim = config.innerDim
        self.eps = config.eps
        _attn.wrappedValue = NFKFlux2Attention(
            dim: dim, heads: config.numAttentionHeads, headDim: config.attentionHeadDim, eps: config.eps)
        _ff.wrappedValue = NFKFlux2FeedForward(dim: dim, inner: config.mlpHiddenDim)
        _ffContext.wrappedValue = NFKFlux2FeedForward(dim: dim, inner: config.mlpHiddenDim)
    }

    /// `modImage` and `modText` are the model's two-set modulation vectors.
    func callAsFunction(_ hidden: MLXArray, encoder: MLXArray, modImage: MLXArray, modText: MLXArray,
                        cos c: MLXArray, sin s: MLXArray, reference: NFKFlux2BlockReference = .none)
        -> (context: MLXArray, image: MLXArray, reference: (key: MLXArray, value: MLXArray)?) {
        var image = hidden
        var context = encoder
        var (shiftMSA, scaleMSA, gateMSA) = NFKFlux2Modulation.set(modImage, 0, dim: dim)
        var (shiftMLP, scaleMLP, gateMLP) = NFKFlux2Modulation.set(modImage, 1, dim: dim)
        if case .extract(let count, let modulation) = reference {
            // The image stream leads with the reference tokens, which take the reference timestep's
            // modulation; the text stream's is unchanged.
            let length = hidden.dim(1)
            func blend(_ base: MLXArray, _ other: MLXArray) -> MLXArray {
                flux2BlendedModulation(base, other, start: 0, count: count, length: length)
            }
            let (rShiftMSA, rScaleMSA, rGateMSA) = NFKFlux2Modulation.set(modulation, 0, dim: dim)
            let (rShiftMLP, rScaleMLP, rGateMLP) = NFKFlux2Modulation.set(modulation, 1, dim: dim)
            (shiftMSA, scaleMSA, gateMSA) = (blend(shiftMSA, rShiftMSA), blend(scaleMSA, rScaleMSA),
                                             blend(gateMSA, rGateMSA))
            (shiftMLP, scaleMLP, gateMLP) = (blend(shiftMLP, rShiftMLP), blend(scaleMLP, rScaleMLP),
                                             blend(gateMLP, rGateMLP))
        }
        let (cShiftMSA, cScaleMSA, cGateMSA) = NFKFlux2Modulation.set(modText, 0, dim: dim)
        let (cShiftMLP, cScaleMLP, cGateMLP) = NFKFlux2Modulation.set(modText, 1, dim: dim)

        let normImage = (1 + scaleMSA) * sd3AffineFreeLayerNorm(image, eps: eps) + shiftMSA
        let normText = (1 + cScaleMSA) * sd3AffineFreeLayerNorm(context, eps: eps) + cShiftMSA
        let (attnImage, attnContext, stored) = attn(normImage, encoder: normText, cos: c, sin: s,
                                                    reference: reference.attention)

        image = image + gateMSA * attnImage
        image = image + gateMLP * ff(sd3AffineFreeLayerNorm(image, eps: eps) * (1 + scaleMLP) + shiftMLP)

        context = context + cGateMSA * attnContext
        context = context + cGateMLP * ffContext(
            sd3AffineFreeLayerNorm(context, eps: eps) * (1 + cScaleMLP) + cShiftMLP)
        return (context, image, stored)
    }
}

/// One FLUX.2 single-stream block: the parallel attention-and-MLP over the concatenated streams under
/// one shift/scale/gate set.
final class NFKFlux2SingleBlock: Module {
    @ModuleInfo(key: "attn") var attn: NFKFlux2ParallelAttention

    let dim: Int
    let eps: Float

    init(_ config: NFKMLXFlux2Configuration) {
        self.dim = config.innerDim
        self.eps = config.eps
        _attn.wrappedValue = NFKFlux2ParallelAttention(
            dim: dim, heads: config.numAttentionHeads, headDim: config.attentionHeadDim,
            mlpHidden: config.mlpHiddenDim, eps: config.eps)
    }

    /// `joined` `[B, St + Si, dim]`, already the concatenation of the text and image streams, of which
    /// the first `textCount` positions are text.
    func callAsFunction(_ joined: MLXArray, mod: MLXArray, cos c: MLXArray, sin s: MLXArray,
                        textCount: Int = 0, reference: NFKFlux2BlockReference = .none)
        -> (output: MLXArray, reference: (key: MLXArray, value: MLXArray)?) {
        var (shift, scale, gate) = NFKFlux2Modulation.set(mod, 0, dim: dim)
        if case .extract(let count, let modulation) = reference {
            // In the joined stream the reference tokens sit between the text and the generated tokens.
            let length = joined.dim(1)
            func blend(_ base: MLXArray, _ other: MLXArray) -> MLXArray {
                flux2BlendedModulation(base, other, start: textCount, count: count, length: length)
            }
            let (rShift, rScale, rGate) = NFKFlux2Modulation.set(modulation, 0, dim: dim)
            (shift, scale, gate) = (blend(shift, rShift), blend(scale, rScale), blend(gate, rGate))
        }
        let normed = (1 + scale) * sd3AffineFreeLayerNorm(joined, eps: eps) + shift
        let (attended, stored) = attn(normed, cos: c, sin: s, textCount: textCount,
                                      reference: reference.attention)
        return (joined + gate * attended, stored)
    }
}

/// The timestep-and-guidance conditioning: the sinusoidal timestep through a bias-free MLP, plus the
/// guidance scale through a second one where the release carries it. FLUX.2 has no pooled text
/// embedding, so this is the whole conditioning vector.
final class NFKFlux2TimeGuidanceEmbed: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: NFKFlux2MLP
    @ModuleInfo(key: "guidance_embedder") var guidanceEmbedder: NFKFlux2MLP?

    let channels: Int

    init(_ config: NFKMLXFlux2Configuration) {
        self.channels = config.timestepGuidanceChannels
        _timestepEmbedder.wrappedValue = NFKFlux2MLP(channels, config.innerDim)
        _guidanceEmbedder.wrappedValue = config.guidanceEmbeds
            ? NFKFlux2MLP(channels, config.innerDim) : nil
    }

    func callAsFunction(timestep: MLXArray, guidance: MLXArray?) -> MLXArray {
        // The float32 sinusoids take the timestep's type, as the reference casts them.
        let embedded = timestepEmbedder(sd3TimestepEmbedding(timestep, dimensions: channels).asType(timestep.dtype))
        guard let guidance, let guidanceEmbedder else { return embedded }
        return embedded + guidanceEmbedder(sd3TimestepEmbedding(guidance, dimensions: channels).asType(guidance.dtype))
    }
}

/// The bias-free `Linear → SiLU → Linear` of a FLUX.2 timestep embedder.
final class NFKFlux2MLP: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ inDim: Int, _ outDim: Int) {
        _linear1.wrappedValue = Linear(inDim, outDim, bias: false)
        _linear2.wrappedValue = Linear(outDim, outDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(NFKReferenceRounding.silu(linear1(x))) }
}

/// The FLUX.2 transformer.
/// FLUX.2 [klein] 9B KV's reference cache: the post-rotary keys and values of every reference token at
/// every layer, taken by ``NFKMLXFlux2TransformerNet/extractingReferences(_:referenceCount:encoderHidden:timestep:guidance:imageIds:textIds:referenceTimestep:)``
/// on the first denoising step and read by every later one, so the reference tokens run once rather
/// than on every step.
public final class NFKMLXFlux2ReferenceCache {
    /// How many reference tokens the cache holds.
    public let referenceCount: Int
    let double: [(key: MLXArray, value: MLXArray)]
    let single: [(key: MLXArray, value: MLXArray)]

    init(referenceCount: Int, double: [(key: MLXArray, value: MLXArray)],
         single: [(key: MLXArray, value: MLXArray)]) {
        self.referenceCount = referenceCount
        self.double = double
        self.single = single
    }

    /// Every cached array. Evaluating them once after the extracting pass materializes the cache, so
    /// the later steps read values rather than holding that pass's graph.
    public var arrays: [MLXArray] {
        (double + single).flatMap { [$0.key, $0.value] }
    }
}

public final class NFKMLXFlux2TransformerNet: Module {
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "time_guidance_embed") var timeGuidanceEmbed: NFKFlux2TimeGuidanceEmbed
    @ModuleInfo(key: "double_stream_modulation_img") var doubleModImage: NFKFlux2Modulation
    @ModuleInfo(key: "double_stream_modulation_txt") var doubleModText: NFKFlux2Modulation
    @ModuleInfo(key: "single_stream_modulation") var singleMod: NFKFlux2Modulation
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [NFKFlux2DoubleBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [NFKFlux2SingleBlock]
    @ModuleInfo(key: "norm_out") var normOut: NFKFlux2Modulation
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public let config: NFKMLXFlux2Configuration
    let rope: NFKFluxRope

    public init(_ config: NFKMLXFlux2Configuration) {
        self.config = config
        self.rope = NFKFluxRope(axesDim: config.axesDimsRope, theta: config.ropeTheta)
        let inner = config.innerDim
        _xEmbedder.wrappedValue = Linear(config.inChannels, inner, bias: false)
        _contextEmbedder.wrappedValue = Linear(config.jointAttentionDim, inner, bias: false)
        _timeGuidanceEmbed.wrappedValue = NFKFlux2TimeGuidanceEmbed(config)
        _doubleModImage.wrappedValue = NFKFlux2Modulation(dim: inner, sets: 2)
        _doubleModText.wrappedValue = NFKFlux2Modulation(dim: inner, sets: 2)
        _singleMod.wrappedValue = NFKFlux2Modulation(dim: inner, sets: 1)
        _transformerBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in NFKFlux2DoubleBlock(config) }
        _singleTransformerBlocks.wrappedValue =
            (0 ..< config.numSingleLayers).map { _ in NFKFlux2SingleBlock(config) }
        // `norm_out` is the reference's `AdaLayerNormContinuous`: the same `silu → linear` shape as a
        // modulation head, producing scale and shift in that order (no gate).
        _normOut.wrappedValue = NFKFlux2Modulation(dim: inner, outDim: 2 * inner)
        _projOut.wrappedValue = Linear(
            inner, config.patchSize * config.patchSize * config.resolvedOutChannels, bias: false)
    }

    /// Velocity prediction on the packed latent. `hiddenStates` `[B, imgSeq, inChannels]`,
    /// `encoderHidden` `[B, txtSeq, jointAttentionDim]`, `timestep` `[B]` (0…1), `guidance` `[B]?`,
    /// `imageIds` `[imgSeq, 4]`, `textIds` `[txtSeq, 4]` → `[B, imgSeq, outChannels]`.
    public func callAsFunction(_ hiddenStates: MLXArray, encoderHidden: MLXArray, timestep: MLXArray,
                               guidance: MLXArray?, imageIds: MLXArray, textIds: MLXArray) -> MLXArray {
        forward(hiddenStates, encoderHidden: encoderHidden, timestep: timestep, guidance: guidance,
                imageIds: imageIds, textIds: textIds, mode: .none).velocity
    }

    /// The first denoising step of FLUX.2 [klein] 9B KV's reference cache: the velocity of the
    /// generated tokens, and every layer's reference keys and values for the steps that follow.
    ///
    /// @discussion `hiddenStates` LEADS with `referenceCount` reference tokens and `imageIds` with
    /// their ids, the reverse of the order ordinary reference conditioning appends them in. The
    /// reference tokens are treated as a clean image: they take the modulation of
    /// `referenceTimestep` (0 by default, as the reference's `ref_fixed_timestep`) and attend only to
    /// one another, while the text and the generated tokens attend to everything. The keys and values
    /// cached are post-rotary. This computes a different function from ordinary reference
    /// conditioning with the same weights, which is why `FLUX.2-klein-9b-kv` is a separately trained
    /// release whose tensors are the base 9B's to the name and shape.
    public func extractingReferences(_ hiddenStates: MLXArray, referenceCount: Int,
                                     encoderHidden: MLXArray, timestep: MLXArray, guidance: MLXArray?,
                                     imageIds: MLXArray, textIds: MLXArray,
                                     referenceTimestep: Float = 0)
        -> (velocity: MLXArray, cache: NFKMLXFlux2ReferenceCache) {
        let result = forward(hiddenStates, encoderHidden: encoderHidden, timestep: timestep,
                             guidance: guidance, imageIds: imageIds, textIds: textIds,
                             mode: .extract(count: referenceCount, timestep: referenceTimestep))
        return (result.velocity, result.cache!)
    }

    /// A later denoising step of the reference cache: the generated tokens alone, attending to the
    /// cached reference keys and values. `imageIds` are the generated tokens' only.
    public func callAsFunction(_ hiddenStates: MLXArray, encoderHidden: MLXArray, timestep: MLXArray,
                               guidance: MLXArray?, imageIds: MLXArray, textIds: MLXArray,
                               referenceCache: NFKMLXFlux2ReferenceCache) -> MLXArray {
        forward(hiddenStates, encoderHidden: encoderHidden, timestep: timestep, guidance: guidance,
                imageIds: imageIds, textIds: textIds, mode: .cached(referenceCache)).velocity
    }

    private enum ReferenceMode {
        case none
        case extract(count: Int, timestep: Float)
        case cached(NFKMLXFlux2ReferenceCache)
    }

    private func forward(_ hiddenStates: MLXArray, encoderHidden: MLXArray, timestep: MLXArray,
                         guidance: MLXArray?, imageIds: MLXArray, textIds: MLXArray,
                         mode: ReferenceMode) -> (velocity: MLXArray, cache: NFKMLXFlux2ReferenceCache?) {
        // The reference takes the timestep and guidance in the latents' type before scaling them.
        let timestep = timestep.asType(hiddenStates.dtype)
        let scaledGuidance = guidance.map { $0.asType(hiddenStates.dtype) * 1000 }
        let temb = timeGuidanceEmbed(timestep: timestep * 1000, guidance: scaledGuidance)
        let modImage = doubleModImage(temb)
        let modText = doubleModText(temb)
        let modSingle = singleMod(temb)

        // The reference tokens' own modulation, from their fixed timestep.
        var referenceCount = 0
        var referenceModImage: MLXArray?
        var referenceModSingle: MLXArray?
        if case .extract(let count, let referenceTimestep) = mode {
            referenceCount = count
            let referenceTemb = timeGuidanceEmbed(timestep: timestep * 0 + referenceTimestep * 1000,
                                                  guidance: scaledGuidance)
            referenceModImage = doubleModImage(referenceTemb)
            referenceModSingle = singleMod(referenceTemb)
        }

        var image = xEmbedder(hiddenStates)
        var context = contextEmbedder(encoderHidden)

        // One rotary table over the concatenated [text, image] ids, which is the reference's two tables
        // concatenated: every row is independent of the rest.
        let (cos, sin) = rope.table(ids: concatenated([textIds, imageIds], axis: 0))

        func blockReference(_ modulation: MLXArray?, _ cached: (key: MLXArray, value: MLXArray)?)
            -> NFKFlux2BlockReference {
            if let modulation {
                return .extract(count: referenceCount, modulation: modulation)
            }
            if let cached {
                return .cached(key: cached.key, value: cached.value)
            }
            return .none
        }
        var cache: NFKMLXFlux2ReferenceCache? = nil
        if case .cached(let existing) = mode {
            cache = existing
        }

        var doubleStored = [(key: MLXArray, value: MLXArray)]()
        for (index, block) in transformerBlocks.enumerated() {
            let (nextContext, nextImage, stored) = block(
                image, encoder: context, modImage: modImage, modText: modText, cos: cos, sin: sin,
                reference: blockReference(referenceModImage, cache?.double[index]))
            (context, image) = (nextContext, nextImage)
            if let stored {
                doubleStored.append(stored)
            }
        }

        let textLength = context.dim(1)
        var joined = concatenated([context, image], axis: 1)
        var singleStored = [(key: MLXArray, value: MLXArray)]()
        for (index, block) in singleTransformerBlocks.enumerated() {
            let (next, stored) = block(joined, mod: modSingle, cos: cos, sin: sin, textCount: textLength,
                                       reference: blockReference(referenceModSingle, cache?.single[index]))
            joined = next
            if let stored {
                singleStored.append(stored)
            }
        }
        // The text, and in an extracting pass the reference tokens, are dropped from the output.
        image = joined[0..., (textLength + referenceCount)..., 0...]

        let mod = normOut(temb)                                            // [B, 2·inner]: scale, shift
        let dim = config.innerDim
        let scale = mod[0..., 0 ..< dim][0..., .newAxis, 0...]
        let shift = mod[0..., dim ..< 2 * dim][0..., .newAxis, 0...]
        image = sd3AffineFreeLayerNorm(image, eps: config.eps) * (1 + scale) + shift
        if referenceCount > 0 {
            cache = NFKMLXFlux2ReferenceCache(referenceCount: referenceCount, double: doubleStored,
                                              single: singleStored)
        }
        return (projOut(image), cache)
    }

    /// The `(0, row, column, 0)` id grid of a packed latent `height`×`width` in latent-patch units, the
    /// pipeline's `_prepare_latent_ids`.
    public static func imageIds(height: Int, width: Int) -> MLXArray {
        var ids = [Float](repeating: 0, count: height * width * 4)
        var index = 0
        for row in 0 ..< height {
            for column in 0 ..< width {
                ids[index * 4 + 1] = Float(row)
                ids[index * 4 + 2] = Float(column)
                index += 1
            }
        }
        return MLXArray(ids, [height * width, 4])
    }

    /// The ids of a REFERENCE image's latent, the pipeline's `_prepare_image_ids`.
    ///
    /// @discussion FLUX.2 conditions on reference images by appending their latents to the generated
    /// one's token sequence. What keeps the two apart is the TIME axis: the generated latent sits at
    /// `t = 0` and reference image `index` at `t = scale · (index + 1)`, the reference's default
    /// `scale` being 10. Without that offset a reference patch and a generated patch at the same row
    /// and column would carry the same rotary position, and the attention could not tell them apart.
    public static func referenceImageIds(height: Int, width: Int, index: Int,
                                         scale: Int = 10) -> MLXArray {
        var ids = [Float](repeating: 0, count: height * width * 4)
        let time = Float(scale * (index + 1))
        var position = 0
        for row in 0 ..< height {
            for column in 0 ..< width {
                ids[position * 4] = time
                ids[position * 4 + 1] = Float(row)
                ids[position * 4 + 2] = Float(column)
                position += 1
            }
        }
        return MLXArray(ids, [height * width, 4])
    }

    /// The `(0, 0, 0, token)` ids of a text sequence, the pipeline's `_prepare_text_ids`. FLUX.1 leaves
    /// the text ids at zero; FLUX.2 numbers the tokens on the fourth axis.
    public static func textIds(length: Int) -> MLXArray {
        var ids = [Float](repeating: 0, count: length * 4)
        for token in 0 ..< length {
            ids[token * 4 + 3] = Float(token)
        }
        return MLXArray(ids, [length, 4])
    }

    /// The geometry of a released FLUX.2 transformer, from its `transformer/config.json`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXFlux2Configuration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "Flux2Transformer2DModel" {
            throw NFKMLXError.unsupportedConfiguration("this reads a FLUX.2 transformer, not \(kind)")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        let axes = (json["axes_dims_rope"] as? [Any])?.compactMap { ($0 as? NSNumber)?.intValue }
            ?? [32, 32, 32, 32]
        return NFKMLXFlux2Configuration(
            inChannels: integer("in_channels", 128),
            outChannels: (json["out_channels"] as? NSNumber)?.intValue,
            patchSize: integer("patch_size", 1),
            numLayers: integer("num_layers", 8), numSingleLayers: integer("num_single_layers", 48),
            attentionHeadDim: integer("attention_head_dim", 128),
            numAttentionHeads: integer("num_attention_heads", 48),
            jointAttentionDim: integer("joint_attention_dim", 15360),
            timestepGuidanceChannels: integer("timestep_guidance_channels", 256),
            mlpRatio: (json["mlp_ratio"] as? NSNumber)?.floatValue ?? 3.0,
            axesDimsRope: axes, ropeTheta: (json["rope_theta"] as? NSNumber)?.floatValue ?? 2000,
            eps: (json["eps"] as? NSNumber)?.floatValue ?? 1e-6,
            guidanceEmbeds: (json["guidance_embeds"] as? NSNumber)?.boolValue ?? true)
    }

    /// Loads a released FLUX.2 transformer directory (`transformer/`) into `net`. Every weight is at
    /// most 2-D, so no layout change is needed.
    public static func loadWeights(into net: NFKMLXFlux2TransformerNet, from directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision)
        try NFKMLXWeights.apply(arrays, to: net)
    }
}

// MARK: - The latent codec

/// The step between FLUX.2's autoencoder and its transformer.
///
/// Every earlier release in this package scales its latent by a scalar `scaling_factor` and shifts it
/// by a `shift_factor`. FLUX.2 ships neither. It folds a 2×2 patch of the latent into the channel
/// axis, which is how 32 latent channels become the transformer's 128, and then whitens the result
/// with the RUNNING STATISTICS of a `BatchNorm` the release stores beside the encoder and decoder
/// (`bn.running_mean`, `bn.running_var`). The mean and variance are per patched channel, so the
/// whitening happens after the patching and is undone before the unpatching.
public final class NFKMLXFlux2LatentCodec: Module {
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVariance: MLXArray

    /// The release's `batch_norm_eps`, added under the square root.
    public let epsilon: Float
    /// The spatial patch folded into the channel axis, `2` for every release so far.
    public let patch: Int

    public init(patchedChannels: Int = 128, epsilon: Float = 1e-4, patch: Int = 2) {
        self.epsilon = epsilon
        self.patch = patch
        _runningMean.wrappedValue = MLXArray.zeros([patchedChannels])
        _runningVariance.wrappedValue = MLXArray.ones([patchedChannels])
    }

    /// Folds each `patch`×`patch` block into the channel axis. `latent` `[B, H, W, C]` →
    /// `[B, H/patch, W/patch, C · patch²]`.
    ///
    /// @discussion The reference's `permute(0, 1, 3, 5, 2, 4)` places the two sub-pixel axes directly
    /// after the channel, so a channel's `patch²` offsets are adjacent in the output. A pixel-unshuffle
    /// that groups by spatial position produces the same shape with different contents; this works in
    /// channels-last, where the reference works in channels-first, so the axis order is re-derived
    /// rather than transcribed.
    public func patchify(_ latent: MLXArray) -> MLXArray {
        let (batch, height, width, channels) = (latent.dim(0), latent.dim(1), latent.dim(2), latent.dim(3))
        return latent
            .reshaped([batch, height / patch, patch, width / patch, patch, channels])
            .transposed(0, 1, 3, 5, 2, 4)                                 // [B, H/p, W/p, C, p, p]
            .reshaped([batch, height / patch, width / patch, channels * patch * patch])
    }

    /// The inverse of ``patchify(_:)``.
    public func unpatchify(_ patched: MLXArray) -> MLXArray {
        let (batch, height, width) = (patched.dim(0), patched.dim(1), patched.dim(2))
        let channels = patched.dim(3) / (patch * patch)
        return patched
            .reshaped([batch, height, width, channels, patch, patch])
            .transposed(0, 1, 4, 2, 5, 3)                                 // [B, H, p, W, p, C]
            .reshaped([batch, height * patch, width * patch, channels])
    }

    /// `(patched - mean) / sqrt(variance + epsilon)`, over the channel axis.
    public func whiten(_ patched: MLXArray) -> MLXArray {
        (patched - runningMean) * rsqrt(runningVariance + epsilon)
    }

    /// The inverse of ``whiten(_:)``.
    public func unwhiten(_ whitened: MLXArray) -> MLXArray {
        whitened * sqrt(runningVariance + epsilon) + runningMean
    }

    /// Flattens the spatial grid into the transformer's token sequence. `[B, H, W, C]` → `[B, H·W, C]`.
    public func pack(_ whitened: MLXArray) -> MLXArray {
        whitened.reshaped([whitened.dim(0), whitened.dim(1) * whitened.dim(2), whitened.dim(3)])
    }

    /// The inverse of ``pack(_:)`` for a known latent grid.
    public func unpack(_ packed: MLXArray, height: Int, width: Int) -> MLXArray {
        packed.reshaped([packed.dim(0), height, width, packed.dim(2)])
    }

    /// The autoencoder's latent to the transformer's token sequence.
    public func encode(latent: MLXArray) -> MLXArray { pack(whiten(patchify(latent))) }

    /// The transformer's token sequence back to the autoencoder's latent.
    public func decode(tokens: MLXArray, height: Int, width: Int) -> MLXArray {
        unpatchify(unwhiten(unpack(tokens, height: height, width: width)))
    }

    /// Reads `bn.running_mean` / `bn.running_var` from a released `vae/` directory.
    public static func codec(fromReleaseDirectory directory: URL,
                             epsilon: Float = 1e-4, patch: Int = 2) throws -> NFKMLXFlux2LatentCodec {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: .float32)
        let wanted = arrays.compactMap { name, value -> (String, MLXArray)? in
            name.hasPrefix("bn.") && !name.hasSuffix("num_batches_tracked")
                ? (String(name.dropFirst(3)), value) : nil
        }
        guard let mean = wanted.first(where: { $0.0 == "running_mean" })?.1 else {
            throw NFKMLXError.weightsMismatch("the release carries no bn.running_mean")
        }
        let codec = NFKMLXFlux2LatentCodec(patchedChannels: mean.dim(0), epsilon: epsilon, patch: patch)
        try NFKMLXWeights.apply(wanted, to: codec)
        return codec
    }
}

// MARK: - The text front end

/// FLUX.2's text conditioning: three intermediate hidden states of a language model, concatenated
/// per token.
///
/// The conditioning is not a language model's output. The release reads three layers of its text
/// encoder and stacks them on the channel axis, which is why ``NFKMLXFlux2Configuration/jointAttentionDim``
/// is three times that model's width. FLUX.2 [klein] reads a Qwen3 and FLUX.2 [dev] reads
/// Mistral-Small 3 (``NFKMLXLanguageConfiguration/mistralSmall3``), both of which this package runs.
///
/// The prompt pads on the RIGHT to a fixed length and the whole padded sequence is encoded. The pad
/// positions' own states are part of the conditioning, and they depend on the attention mask: a pad
/// attends to the real tokens and to the pads before it. Measured on the reference, masking changes
/// the real tokens' states not at all (cosine 1.0) and the pad positions' by 1.2e-2, which moves the
/// whole embedding to 0.9993. The mask is therefore required, not an optimization.
public final class NFKMLXFlux2TextEncoder {
    private let decoder: NFKMLXLanguageNet
    /// The layer indices read, in the reference's `output_hidden_states` convention: index 0 is the
    /// embedding and index `k` the state after `k` decoder layers.
    public let layers: [Int]
    /// The length the prompt pads to, the release's `max_sequence_length`.
    public let contextLength: Int
    private let padToken: Int

    public init(decoder: NFKMLXLanguageNet, layers: [Int] = [9, 18, 27],
                contextLength: Int = 512, padToken: Int = 151643) {
        self.decoder = decoder
        self.layers = layers
        self.contextLength = contextLength
        self.padToken = padToken
    }

    /// The conditioning for an already-tokenized prompt. `tokens` is the chat-templated prompt
    /// WITHOUT padding; this pads it, encodes the whole padded sequence under the key-padding mask,
    /// and returns `[1, contextLength, layers.count · hiddenSize]`.
    public func encode(tokens: [Int]) -> MLXArray {
        let kept = tokens.count > contextLength ? Array(tokens.prefix(contextLength)) : tokens
        var padded = kept
        while padded.count < contextLength { padded.append(padToken) }
        let ids = MLXArray(padded.map(Int32.init)).reshaped([1, contextLength])
        let attends = MLXArray((0 ..< contextLength).map { $0 < kept.count })
        return embedding(ids: ids, keyPadding: attends)
    }

    /// The conditioning for a padded batch whose mask the caller supplies. `keyPadding` is true where
    /// a position may be attended to.
    public func embedding(ids: MLXArray, keyPadding: MLXArray?) -> MLXArray {
        let states = decoder.layerStates(ids, keyPadding: keyPadding)
        // `[layers][1, length, hidden]` → `[1, length, layers · hidden]`, the reference's stack on a
        // new axis followed by a permute that puts the layer axis beside the channel.
        return concatenated(layers.map { states[$0] }, axis: -1)
    }
}
