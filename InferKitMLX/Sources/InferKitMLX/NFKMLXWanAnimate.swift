//
//  NFKMLXWanAnimate.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The Wan 2.2 Animate 2 DiT (`WanAnimate2Transformer3DModel`, Alibaba Wan): the denoising transformer
// that drives a reference character image with the motion of a driving video. The block is the Wan
// adaLN block — a `[1, 6, dim]` modulation added to the timestep projection and chunked six ways, a
// non-affine LayerNorm scaled and shifted before self-attention with a gated residual, an affine
// `norm3` before cross-attention, and a gated feed-forward — plus two additions: an image
// cross-attention branch in every block, fed by an `img_emb` projector over CLIP embeddings, and an
// in-context reference mechanism built on a key/value cache.
//
// The reference mechanism is what makes this a separate port rather than an option on the Wan DiT.
// Generation runs in two passes. The EXTRACT pass runs the reference latents through the stack and
// stores each block's pre-rotary keys and values. The GENERATE pass runs a chunk of the video and
// attends, per frame, over the whole video's generation buffer plus the cached reference tokens at
// that frame's index; the reference keys are re-rotated on a grid the `referOffset*` values place
// away from the generation grid. The buffer is the full video's, so a chunk shorter than the video
// leaves zero-filled key positions inside it, and the reference's own buffer leaves zero-filled
// frames past what the cache holds. Those positions are NOT masked out: a zero key scores zero
// against every query, so each one adds `exp(0)` to the softmax denominator while contributing
// nothing to the numerator. `maskedAttention` reproduces that dilution by counting the zero
// positions rather than materializing them.
//
// The released 14B model is 32.8 GB in bf16 and its pipeline is about 50 GB, so it does not run on a
// 32 GiB machine. The port is measured numerically at a tiny random configuration against diffusers'
// own implementation, and the released file is held to the module by shape.

/// Wan Animate DiT geometry. Defaults are the released `Wan2.2-Animate-2-14B` model.
public struct NFKMLXWanAnimateConfiguration: Sendable {
    /// Channels the patch embedding reads: the latent plus the conditioning latent concatenated.
    public var inChannels: Int
    public var outChannels: Int
    public var dim: Int
    public var heads: Int
    public var layers: Int
    public var ffnDim: Int
    public var textDim: Int
    /// Tokens the text stream is zero-padded to before the text embedding.
    public var textLength: Int
    public var freqDim: Int
    /// Width of the CLIP image embedding the `img_emb` projector reads.
    public var imageDim: Int
    public var patchSize: [Int]                                            // (t, h, w)
    public var eps: Float
    public var ropeTheta: Float
    public var crossAttentionNorm: Bool
    public var useImageEmbedding: Bool
    /// Rotary offsets placing the reference grid away from the generation grid. A negative width
    /// offset means the reference grid's own width, resolved per pass.
    public var referOffsetT: Int
    public var referOffsetH: Int
    public var referOffsetW: Int
    /// Frame stride the reference grid's rotary positions advance by.
    public var referStride: Int

    public init(inChannels: Int = 36, outChannels: Int = 16, dim: Int = 5120, heads: Int = 40,
                layers: Int = 40, ffnDim: Int = 13824, textDim: Int = 4096, textLength: Int = 512,
                freqDim: Int = 256, imageDim: Int = 1280, patchSize: [Int] = [1, 2, 2],
                eps: Float = 1e-6, ropeTheta: Float = 10000.0, crossAttentionNorm: Bool = true,
                useImageEmbedding: Bool = true, referOffsetT: Int = 1, referOffsetH: Int = 0,
                referOffsetW: Int = -1, referStride: Int = 1) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.dim = dim
        self.heads = heads
        self.layers = layers
        self.ffnDim = ffnDim
        self.textDim = textDim
        self.textLength = textLength
        self.freqDim = freqDim
        self.imageDim = imageDim
        self.patchSize = patchSize
        self.eps = eps
        self.ropeTheta = ropeTheta
        self.crossAttentionNorm = crossAttentionNorm
        self.useImageEmbedding = useImageEmbedding
        self.referOffsetT = referOffsetT
        self.referOffsetH = referOffsetH
        self.referOffsetW = referOffsetW
        self.referStride = referStride
    }

    /// The released `Wan-AI/Wan2.2-Animate-2-14B` geometry, which the defaults spell out.
    public static let base = NFKMLXWanAnimateConfiguration()

    public static let tiny = NFKMLXWanAnimateConfiguration(
        inChannels: 8, outChannels: 4, dim: 32, heads: 2, layers: 2, ffnDim: 48, textDim: 10,
        textLength: 16, patchSize: [1, 2, 2])

    var headDim: Int { dim / heads }
    /// The rotary axis widths the reference derives: `h = w = 2·(headDim/6)`, `t = headDim − h − w`.
    var ropeAxes: [Int] {
        let hw = 2 * (headDim / 6)
        return [headDim - 2 * hw, hw, hw]
    }
}

/// A patch grid, in patch units.
struct NFKWanAnimateGrid: Sendable, Equatable {
    var frames: Int
    var height: Int
    var width: Int
    var area: Int { height * width }
    var tokens: Int { frames * height * width }
}

/// The reference key/value store the two passes share: the extract pass writes each block's
/// pre-rotary keys and values, the generation pass reads them and applies the reference rotary.
@objc(NFKMLXWanAnimateKVCache)
public final class NFKMLXWanAnimateKVCache: NSObject {
    private var keys: [MLXArray?]
    private var values: [MLXArray?]

    @objc public init(layerCount: Int) {
        self.keys = Array(repeating: nil, count: layerCount)
        self.values = Array(repeating: nil, count: layerCount)
    }

    @objc public var layerCount: Int { keys.count }

    /// Whether every layer holds a reference pass, which is what the generation pass requires.
    @objc public var isPopulated: Bool { keys.allSatisfy { $0 != nil } }

    @objc public func clear() {
        keys = Array(repeating: nil, count: keys.count)
        values = Array(repeating: nil, count: values.count)
    }

    func store(key: MLXArray, value: MLXArray, layer: Int) {
        keys[layer] = key
        values[layer] = value
    }

    func read(layer: Int) throws -> (key: MLXArray, value: MLXArray) {
        guard let key = keys[layer], let value = values[layer] else {
            throw NFKMLXError.unsupportedConfiguration(
                "the reference cache is empty; run the extract pass before generating")
        }
        return (key, value)
    }
}

/// The self-attention of a Wan Animate block. Its keys and values are stored before the rotary is
/// applied, because the generation pass rotates them on a different grid.
final class NFKWanAnimateSelfAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]                         // [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    let heads: Int
    let headDim: Int

    init(_ config: NFKMLXWanAnimateConfiguration) {
        self.heads = config.heads
        self.headDim = config.headDim
        _toQ.wrappedValue = Linear(config.dim, config.dim)
        _toK.wrappedValue = Linear(config.dim, config.dim)
        _toV.wrappedValue = Linear(config.dim, config.dim)
        _toOut.wrappedValue = [Linear(config.dim, config.dim)]
        _normQ.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.eps)
        _normK.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.eps)
    }

    /// `x` `[N, dim]` → `(query, key, value)`, each `[N, heads, headDim]`, key and value pre-rotary.
    func project(_ x: MLXArray) -> (query: MLXArray, key: MLXArray, value: MLXArray) {
        let n = x.dim(0)
        return (normQ(toQ(x)).reshaped([n, heads, headDim]),
                normK(toK(x)).reshaped([n, heads, headDim]),
                toV(x).reshaped([n, heads, headDim]))
    }

    func output(_ merged: MLXArray) -> MLXArray { (toOut[0] as! Linear)(merged) }
}

/// The cross-attention of a Wan Animate block: the text stream, plus an additive branch over the
/// projected CLIP image embeddings sharing the same queries.
final class NFKWanAnimateCrossAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]                         // [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "add_k_proj") var addK: Linear?
    @ModuleInfo(key: "add_v_proj") var addV: Linear?
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm?

    let heads: Int
    let headDim: Int

    init(_ config: NFKMLXWanAnimateConfiguration) {
        self.heads = config.heads
        self.headDim = config.headDim
        _toQ.wrappedValue = Linear(config.dim, config.dim)
        _toK.wrappedValue = Linear(config.dim, config.dim)
        _toV.wrappedValue = Linear(config.dim, config.dim)
        _toOut.wrappedValue = [Linear(config.dim, config.dim)]
        _normQ.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.eps)
        _normK.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.eps)
        if config.useImageEmbedding {
            _addK.wrappedValue = Linear(config.dim, config.dim)
            _addV.wrappedValue = Linear(config.dim, config.dim)
            _normAddedK.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.eps)
        }
    }

    /// `x` `[N, dim]`, `text` `[Lt, dim]`, `image` `[Li, dim]` (already projected) → `[N, dim]`.
    func callAsFunction(_ x: MLXArray, text: MLXArray, image: MLXArray?) -> MLXArray {
        let n = x.dim(0), lt = text.dim(0)
        let scale = 1.0 / sqrt(Float(headDim))
        let q = normQ(toQ(x)).reshaped([n, heads, headDim]).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let k = normK(toK(text)).reshaped([lt, heads, headDim]).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let v = toV(text).reshaped([lt, heads, headDim]).transposed(1, 0, 2).expandedDimensions(axis: 0)
        var attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                         scale: scale, mask: .none)[0]
        if let image, let addK, let addV, let normAddedK {
            let li = image.dim(0)
            let ki = normAddedK(addK(image)).reshaped([li, heads, headDim])
                .transposed(1, 0, 2).expandedDimensions(axis: 0)
            let vi = addV(image).reshaped([li, heads, headDim])
                .transposed(1, 0, 2).expandedDimensions(axis: 0)
            attended = attended + MLXFast.scaledDotProductAttention(queries: q, keys: ki, values: vi,
                                                                    scale: scale, mask: .none)[0]
        }
        return (toOut[0] as! Linear)(attended.transposed(1, 0, 2).reshaped([n, heads * headDim]))
    }
}

/// A Wan Animate block. `norm3` is the affine cross-attention norm and `norm2` the non-affine
/// pre-feed-forward one, which is the reverse of the naming the text-to-video port uses.
final class NFKWanAnimateBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: NFKWanAnimateSelfAttention
    @ModuleInfo(key: "cross_attn") var crossAttention: NFKWanAnimateCrossAttention
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: [Module]                              // [Linear, marker, Linear]
    @ParameterInfo(key: "modulation") var modulation: MLXArray

    let norm1: LayerNorm
    let norm2: LayerNorm
    let dim: Int

    init(_ config: NFKMLXWanAnimateConfiguration) {
        self.dim = config.dim
        _selfAttention.wrappedValue = NFKWanAnimateSelfAttention(config)
        _crossAttention.wrappedValue = NFKWanAnimateCrossAttention(config)
        _norm3.wrappedValue = LayerNorm(dimensions: config.dim, eps: config.eps,
                                        affine: config.crossAttentionNorm)
        _ffn.wrappedValue = [Linear(config.dim, config.ffnDim), Module(), Linear(config.ffnDim, config.dim)]
        _modulation.wrappedValue = MLXArray.zeros([1, 6, config.dim])
        self.norm1 = LayerNorm(dimensions: config.dim, eps: config.eps, affine: false)
        self.norm2 = LayerNorm(dimensions: config.dim, eps: config.eps, affine: false)
    }

    func feedForward(_ x: MLXArray) -> MLXArray {
        (ffn[2] as! Linear)(geluApproximate((ffn[0] as! Linear)(x)))
    }

    /// The six modulation rows this block applies, each `[1, dim]`.
    func modulationRows(_ temb: MLXArray) -> [MLXArray] {
        let combined = modulation[0] + temb                                // [6, dim]
        return (0 ..< 6).map { combined[$0].reshaped([1, dim]) }
    }
}

/// The `img_emb` projector: a normalized two-layer MLP over the CLIP image embedding, ending in a
/// second normalization. Its activation is the exact gelu, where the block's feed-forward takes the
/// tanh approximation.
final class NFKWanAnimateImageProjection: Module {
    @ModuleInfo(key: "proj") var proj: [Module]                            // [LayerNorm, Linear, marker, Linear, LayerNorm]

    init(_ config: NFKMLXWanAnimateConfiguration) {
        _proj.wrappedValue = [LayerNorm(dimensions: config.imageDim),
                              Linear(config.imageDim, config.imageDim),
                              Module(),
                              Linear(config.imageDim, config.dim),
                              LayerNorm(dimensions: config.dim)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = (proj[0] as! LayerNorm)(x)
        let hidden = gelu((proj[1] as! Linear)(normalized))
        return (proj[4] as! LayerNorm)((proj[3] as! Linear)(hidden))
    }
}

/// The output head: the shared timestep embedding modulates a non-affine norm, then one linear emits
/// a patch's worth of output channels.
final class NFKWanAnimateHead: Module {
    @ModuleInfo(key: "head") var head: Linear
    @ParameterInfo(key: "modulation") var modulation: MLXArray

    let norm: LayerNorm
    let dim: Int

    init(_ config: NFKMLXWanAnimateConfiguration) {
        self.dim = config.dim
        let patch = config.patchSize.reduce(1, *)
        _head.wrappedValue = Linear(config.dim, patch * config.outChannels)
        _modulation.wrappedValue = MLXArray.zeros([1, 2, config.dim])
        self.norm = LayerNorm(dimensions: config.dim, eps: config.eps, affine: false)
    }

    /// `x` `[N, dim]`, `temb` `[dim]`.
    func callAsFunction(_ x: MLXArray, temb: MLXArray) -> MLXArray {
        let combined = modulation[0] + temb.reshaped([1, dim])             // [2, dim]
        let shift = combined[0].reshaped([1, dim]), scale = combined[1].reshaped([1, dim])
        return head(norm(x) * (1.0 + scale) + shift)
    }
}

/// The Wan Animate transformer.
public final class NFKMLXWanAnimateNet: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv3d
    @ModuleInfo(key: "text_embedding") var textEmbedding: [Module]         // [Linear, marker, Linear]
    @ModuleInfo(key: "time_embedding") var timeEmbedding: [Module]         // [Linear, marker, Linear]
    @ModuleInfo(key: "time_projection") var timeProjection: [Module]       // [marker, Linear]
    @ModuleInfo(key: "img_emb") var imageProjection: NFKWanAnimateImageProjection?
    @ModuleInfo(key: "blocks") var blocks: [NFKWanAnimateBlock]
    @ModuleInfo(key: "head") var head: NFKWanAnimateHead

    let config: NFKMLXWanAnimateConfiguration
    let rope: NFKZImageRope

    public init(_ config: NFKMLXWanAnimateConfiguration) {
        self.config = config
        self.rope = NFKZImageRope(axesDims: config.ropeAxes, theta: config.ropeTheta)
        _patchEmbedding.wrappedValue = Conv3d(inputChannels: config.inChannels,
                                              outputChannels: config.dim,
                                              kernelSize: IntOrTriple(config.patchSize),
                                              stride: IntOrTriple(config.patchSize))
        _textEmbedding.wrappedValue = [Linear(config.textDim, config.dim), Module(),
                                       Linear(config.dim, config.dim)]
        _timeEmbedding.wrappedValue = [Linear(config.freqDim, config.dim), Module(),
                                       Linear(config.dim, config.dim)]
        _timeProjection.wrappedValue = [Module(), Linear(config.dim, 6 * config.dim)]
        _imageProjection.wrappedValue = config.useImageEmbedding
            ? NFKWanAnimateImageProjection(config) : nil
        _blocks.wrappedValue = (0 ..< config.layers).map { _ in NFKWanAnimateBlock(config) }
        _head.wrappedValue = NFKWanAnimateHead(config)
    }

    /// The reference pass. Runs the reference latents through the stack, filling `cache` with every
    /// block's pre-rotary keys and values, and returns the sample the pass predicts.
    ///
    /// The reference is modulated at a fixed timestep of 1 and rotated on the offset grid, so the
    /// call takes no timestep of its own.
    public func extractReference(latent: MLXArray, condition: MLXArray, text: MLXArray,
                                 imageEmbeddings: MLXArray?, into cache: NFKMLXWanAnimateKVCache)
        throws -> MLXArray {
        try run(latent: latent, condition: condition, text: text, imageEmbeddings: imageEmbeddings,
                timestep: MLXArray(Float(1)), cache: cache, reference: nil, video: nil)
    }

    /// One denoising step over a chunk of the video, attending over the cached reference tokens.
    ///
    /// - Parameters:
    ///   - latent: the chunk's noisy latents `[channels, frames, height, width]`.
    ///   - condition: the chunk's conditioning latents, concatenated with `latent` by channel.
    ///   - text: the text encoder's states `[tokens, textDim]`, zero-padded to the configured length.
    ///   - imageEmbeddings: the image encoder's features, or nil for a configuration without them.
    ///   - timestep: the step's timestep.
    ///   - cache: the reference keys and values ``extractReference(latent:condition:text:imageEmbeddings:into:)`` filled.
    ///   - referenceGrid: the patch grid of the latents the extract pass ran over.
    ///   - videoFrames: the full video's latent frame count (`originLength / 4 + 1` upstream).
    ///   - videoArea: the full video's patch count per frame (`height · width / 256` upstream).
    public func generate(latent: MLXArray, condition: MLXArray, text: MLXArray,
                         imageEmbeddings: MLXArray?, timestep: MLXArray,
                         cache: NFKMLXWanAnimateKVCache, referenceGrid: (Int, Int, Int),
                         videoFrames: Int, videoArea: Int) throws -> MLXArray {
        let grid = NFKWanAnimateGrid(frames: referenceGrid.0, height: referenceGrid.1,
                                     width: referenceGrid.2)
        return try run(latent: latent, condition: condition, text: text,
                       imageEmbeddings: imageEmbeddings, timestep: timestep, cache: cache,
                       reference: grid, video: (videoFrames, videoArea))
    }

    /// Position ids `[N, 3]` for a patch grid, offset per axis and advancing the frame axis by
    /// `timeStride`. A negative width offset resolves to the grid's own width, as the reference's
    /// `refer_offset_w = -1` does.
    private func positions(_ grid: NFKWanAnimateGrid, offset: (Int, Int, Int),
                           timeStride: Int) -> MLXArray {
        let offsetW = offset.2 >= 0 ? offset.2 : grid.width
        var frames = [Float](), rows = [Float](), columns = [Float]()
        frames.reserveCapacity(grid.tokens)
        for f in 0 ..< grid.frames {
            for h in 0 ..< grid.height {
                for w in 0 ..< grid.width {
                    frames.append(Float(offset.0 + f * timeStride))
                    rows.append(Float(offset.1 + h))
                    columns.append(Float(offsetW + w))
                }
            }
        }
        let n = grid.tokens
        return concatenated([MLXArray(frames).reshaped([n, 1]),
                             MLXArray(rows).reshaped([n, 1]),
                             MLXArray(columns).reshaped([n, 1])], axis: 1)
    }

    /// The generation pass's self-attention: dense over the chunk's own tokens plus, per frame, the
    /// reference tokens at that frame's index, with the video buffer's zero-filled positions entering
    /// the denominator.
    ///
    /// `query`/`key`/`value` are `[N, heads, headDim]` over the chunk; `referenceKey`/`referenceValue`
    /// are `[Nr, heads, headDim]` over the cached reference.
    private func maskedAttention(query: MLXArray, key: MLXArray, value: MLXArray,
                                 referenceKey: MLXArray, referenceValue: MLXArray,
                                 grid: NFKWanAnimateGrid, reference: NFKWanAnimateGrid,
                                 videoFrames: Int, videoArea: Int) -> MLXArray {
        let scale = 1.0 / sqrt(Float(config.headDim))
        let keyHeads = key.transposed(1, 0, 2)                             // [heads, N, headDim]
        let valueHeads = value.transposed(1, 0, 2)
        let keyTransposed = keyHeads.transposed(0, 2, 1)                   // [heads, headDim, N]
        // Positions the full video's generation buffer holds that this chunk does not fill.
        let emptyGeneration = (videoFrames + 1) * videoArea - grid.tokens

        var perFrame = [MLXArray]()
        for frame in 0 ..< grid.frames {
            let rows = (frame * grid.area) ..< ((frame + 1) * grid.area)
            let queryHeads = query[rows].transposed(1, 0, 2)               // [heads, area, headDim]
            var scores = matmul(queryHeads, keyTransposed) * scale
            var values = valueHeads
            var empty = emptyGeneration
            // Frame `f` reads the reference frame `f - 1`; a slot past what the cache holds is a
            // zero-filled frame of the reference buffer, and past the video's frame count there is
            // no reference slot at all.
            let slot = frame - 1
            if slot >= 0, slot < videoFrames {
                if slot < reference.frames {
                    let referenceRows = (slot * reference.area) ..< ((slot + 1) * reference.area)
                    let rk = referenceKey[referenceRows].transposed(1, 0, 2)
                    scores = concatenated([scores, matmul(queryHeads, rk.transposed(0, 2, 1)) * scale],
                                          axis: -1)
                    values = concatenated([values, referenceValue[referenceRows].transposed(1, 0, 2)],
                                          axis: 1)
                    empty += videoArea - reference.area
                } else {
                    empty += videoArea
                }
            }
            var peak = scores.max(axis: -1, keepDims: true)
            if empty > 0 {
                peak = maximum(peak, MLXArray(Float(0)))
            }
            let weights = exp(scores - peak)
            var total = weights.sum(axis: -1, keepDims: true)
            if empty > 0 {
                total = total + Float(empty) * exp(-peak)
            }
            perFrame.append((matmul(weights, values) / total).transposed(1, 0, 2))
        }
        return concatenated(perFrame, axis: 0)
    }

    /// `latent` and `condition` `[C, F, H, W]`, `text` `[Lc, textDim]`, `imageEmbeddings`
    /// `[Li, imageDim]`. `reference` and `video` are set for the generation pass only.
    private func run(latent: MLXArray, condition: MLXArray, text: MLXArray,
                     imageEmbeddings: MLXArray?, timestep: MLXArray,
                     cache: NFKMLXWanAnimateKVCache, reference: NFKWanAnimateGrid?,
                     video: (frames: Int, area: Int)?) throws -> MLXArray {
        guard cache.layerCount == config.layers else {
            throw NFKMLXError.unsupportedConfiguration(
                "the cache holds \(cache.layerCount) layers, the model has \(config.layers)")
        }
        let pt = config.patchSize[0], ph = config.patchSize[1], pw = config.patchSize[2]
        let f = latent.dim(1), h = latent.dim(2), w = latent.dim(3)
        let grid = NFKWanAnimateGrid(frames: f / pt, height: h / ph, width: w / pw)
        let dim = config.dim

        // 1. Patch embedding over the latent and its conditioning latent, concatenated by channel.
        let stacked = concatenated([latent, condition], axis: 0)           // [inChannels, F, H, W]
        let ndhwc = stacked.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
        var hidden = patchEmbedding(ndhwc).reshaped([grid.tokens, dim])

        // 2. Time and text conditioning. The reference pass is modulated at a fixed timestep, which
        //    `extractReference` supplies.
        let sinusoid = ltxTimestepEmbedding(timestep, channels: config.freqDim)
        let temb = (timeEmbedding[2] as! Linear)(silu((timeEmbedding[0] as! Linear)(sinusoid)))
        let projection = (timeProjection[1] as! Linear)(silu(temb))        // [1, 6·dim]
        let blockModulation = projection.reshaped([6, dim])

        var padded = text
        if text.dim(0) < config.textLength {
            padded = concatenated([text, MLXArray.zeros([config.textLength - text.dim(0), text.dim(1)])],
                                  axis: 0)
        }
        let context = (textEmbedding[2] as! Linear)(
            geluApproximate((textEmbedding[0] as! Linear)(padded)))
        let image = imageEmbeddings.flatMap { embeddings in
            imageProjection.map { $0(embeddings) }
        }

        // 3. Rotary grids. The reference stream sits at the configured offsets and stride; the
        //    generation stream sits at the origin with a stride of one.
        let referenceGrid = reference ?? grid
        let offsets = (config.referOffsetT, config.referOffsetH, config.referOffsetW)
        let referencePositions = positions(referenceGrid, offset: offsets, timeStride: config.referStride)
        let (referenceCos, referenceSin) = rope.table(positions: referencePositions)
        var generationCos: MLXArray?, generationSin: MLXArray?
        if reference != nil {
            let table = rope.table(positions: positions(grid, offset: (0, 0, 0), timeStride: 1))
            generationCos = table.cos
            generationSin = table.sin
        }

        // 4. Blocks.
        for (index, block) in blocks.enumerated() {
            let rows = block.modulationRows(blockModulation)
            let normalized = block.norm1(hidden) * (1.0 + rows[1]) + rows[0]
            let (query, key, value) = block.selfAttention.project(normalized)

            var attended: MLXArray
            if let video, let reference {
                let q = zImageApplyRope(query, cos: generationCos!, sin: generationSin!)
                let k = zImageApplyRope(key, cos: generationCos!, sin: generationSin!)
                let cached = try cache.read(layer: index)
                let referenceKey = zImageApplyRope(cached.key, cos: referenceCos, sin: referenceSin)
                attended = maskedAttention(query: q, key: k, value: value,
                                           referenceKey: referenceKey, referenceValue: cached.value,
                                           grid: grid, reference: reference,
                                           videoFrames: video.frames, videoArea: video.area)
            } else {
                cache.store(key: key, value: value, layer: index)
                let q = zImageApplyRope(query, cos: referenceCos, sin: referenceSin)
                        .transposed(1, 0, 2).expandedDimensions(axis: 0)
                let k = zImageApplyRope(key, cos: referenceCos, sin: referenceSin)
                        .transposed(1, 0, 2).expandedDimensions(axis: 0)
                let v = value.transposed(1, 0, 2).expandedDimensions(axis: 0)
                attended = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(config.headDim)),
                    mask: .none)[0].transposed(1, 0, 2)
            }
            let merged = block.selfAttention.output(attended.reshaped([grid.tokens, dim]))
            hidden = hidden + merged * rows[2]
            hidden = hidden + block.crossAttention(block.norm3(hidden), text: context, image: image)
            hidden = hidden + block.feedForward(block.norm2(hidden) * (1.0 + rows[4]) + rows[3]) * rows[5]
        }

        // 5. Head and unpatchify.
        let out = head(hidden, temb: temb[0])                              // [N, C·pt·ph·pw]
        let c = config.outChannels
        let patched = out.reshaped([grid.frames, grid.height, grid.width, pt, ph, pw, c])
        return patched.transposed(6, 0, 3, 1, 4, 2, 5).reshaped([c, f, h, w])
    }
}

/// Building the Wan Animate transformer, and naming what its released checkpoint holds.
@objc(NFKMLXWanAnimate)
public final class NFKMLXWanAnimate: NSObject {

    public static func makeNet(_ configuration: NFKMLXWanAnimateConfiguration = .base) -> NFKMLXWanAnimateNet {
        NFKMLXWanAnimateNet(configuration)
    }

    /// A cache sized for a configuration's block count.
    @objc public static func makeCache(layerCount: Int) -> NFKMLXWanAnimateKVCache {
        NFKMLXWanAnimateKVCache(layerCount: layerCount)
    }

    /// The module key a released tensor name maps to.
    ///
    /// @discussion The release ships the original Wan naming, where the module follows the
    /// implementation it is measured against: every block sits under an extra `block.` level, the
    /// attention projections are single letters, and the image branch of the cross-attention is named
    /// for the image rather than for being an addition. Everything else — `modulation`, `norm3`, the
    /// numbered feed-forward, `img_emb.proj.N`, `patch_embedding`, `text_embedding`, `time_embedding`,
    /// `time_projection.1`, `head` — is already the module's own name.
    static func moduleKey(forRelease name: String) -> String {
        var key = name.replacingOccurrences(of: #"^(blocks\.\d+\.)block\."#, with: "$1",
                                            options: .regularExpression)
        for site in ["self_attn", "cross_attn"] {
            for (released, module) in [("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("o", "to_out.0")] {
                key = key.replacingOccurrences(of: "\(site).\(released).", with: "\(site).\(module).")
            }
        }
        key = key.replacingOccurrences(of: "cross_attn.k_img.", with: "cross_attn.add_k_proj.")
        key = key.replacingOccurrences(of: "cross_attn.v_img.", with: "cross_attn.add_v_proj.")
        return key.replacingOccurrences(of: "cross_attn.norm_k_img.", with: "cross_attn.norm_added_k.")
    }

    /// The released tensor name a module key came from, which is `moduleKey(forRelease:)` inverted.
    static func releaseKey(forModule name: String) -> String {
        var key = name.replacingOccurrences(of: #"^(blocks\.\d+\.)"#, with: "$1block.",
                                            options: .regularExpression)
        key = key.replacingOccurrences(of: "cross_attn.add_k_proj.", with: "cross_attn.k_img.")
        key = key.replacingOccurrences(of: "cross_attn.add_v_proj.", with: "cross_attn.v_img.")
        key = key.replacingOccurrences(of: "cross_attn.norm_added_k.", with: "cross_attn.norm_k_img.")
        for site in ["self_attn", "cross_attn"] {
            key = key.replacingOccurrences(of: "\(site).to_out.0.", with: "\(site).o.")
            for (module, released) in [("to_q", "q"), ("to_k", "k"), ("to_v", "v")] {
                key = key.replacingOccurrences(of: "\(site).\(module).", with: "\(site).\(released).")
            }
        }
        return key
    }
}
