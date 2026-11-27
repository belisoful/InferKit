//
//  NFKMLXWanDancer.swift
//  InferKitMLX
//
//  The Wan-Dancer-14B transformer (Wan-Video/Wan-Dancer, `diffsynth/models/wan_video_dit.py`, `dit_model_type=1`):
//  Wan 2.1 image-to-video blocks over a music track. The reference image's CLIP tokens lead the context, a
//  two-layer music encoder reads 35 librosa features per frame, eight music injectors add the music after
//  their blocks, and the global stage, which plans keyframes over the whole track at a reduced frame rate,
//  has its own patch embedding and head and spreads its rotary frames at that rate.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// The geometry of a Wan-Dancer transformer. Defaults are the released 14B.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanDancerConfiguration: Sendable {
    public var dim: Int
    public var inChannels: Int
    public var outChannels: Int
    public var heads: Int
    public var layers: Int
    public var ffnDim: Int
    public var textDim: Int
    public var freqDim: Int
    public var eps: Float
    public var patchSize: [Int]
    /// The CLIP feature width the image projections read.
    public var imageDim: Int
    /// The blocks a music injector follows; a block past the last makes none.
    public var musicInjectLayers: [Int]
    /// The music feature width (``NFKMLXWanDancerMusic/featureCount``).
    public var musicFeatures: Int
    public var musicWidth: Int
    public var musicHeads: Int
    public var musicFeedForward: Int
    public var musicLayers: Int
    /// The number of spans the music encoder reads separately: the released scripts run eight-way sequence
    /// parallel, which splits the music features into eight spans and encodes each alone.
    public var musicEncoderSpans: Int
    /// The rows of the rotary tables (`int(22350 / 8 + 0.5)`).
    public var ropeEnd: Int

    public init(dim: Int = 5120, inChannels: Int = 36, outChannels: Int = 16, heads: Int = 40, layers: Int = 40,
                ffnDim: Int = 13824, textDim: Int = 4096, freqDim: Int = 256, eps: Float = 1e-6,
                patchSize: [Int] = [1, 2, 2], imageDim: Int = 1280,
                musicInjectLayers: [Int] = [0, 4, 8, 12, 16, 20, 24, 27], musicFeatures: Int = 35,
                musicWidth: Int = 256, musicHeads: Int = 4, musicFeedForward: Int = 1024, musicLayers: Int = 2,
                musicEncoderSpans: Int = 8, ropeEnd: Int = 2794) {
        self.dim = dim
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.heads = heads
        self.layers = layers
        self.ffnDim = ffnDim
        self.textDim = textDim
        self.freqDim = freqDim
        self.eps = eps
        self.patchSize = patchSize
        self.imageDim = imageDim
        self.musicInjectLayers = musicInjectLayers
        self.musicFeatures = musicFeatures
        self.musicWidth = musicWidth
        self.musicHeads = musicHeads
        self.musicFeedForward = musicFeedForward
        self.musicLayers = musicLayers
        self.musicEncoderSpans = musicEncoderSpans
        self.ropeEnd = ropeEnd
    }

    public var headDim: Int { dim / heads }

    /// The blocks that carry an injector, as the reference builds one only for a block that exists.
    var injectedBlocks: [Int] { musicInjectLayers.filter { $0 < layers } }

    /// The geometry a release's `config.json` states (`dim`, `ffn_dim`, `freq_dim`, `in_dim`, `num_heads`,
    /// `num_layers`, `out_dim`, `eps`, and, where given, `text_dim`); the rest are the released values.
    public static func reading(_ url: URL) throws -> NFKMLXWanDancerConfiguration {
        let json = try NFKMLXWanRelease.json(url)
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        let base = NFKMLXWanDancerConfiguration()
        return NFKMLXWanDancerConfiguration(
            dim: int("dim", base.dim), inChannels: int("in_dim", base.inChannels), outChannels: int("out_dim", base.outChannels),
            heads: int("num_heads", base.heads), layers: int("num_layers", base.layers), ffnDim: int("ffn_dim", base.ffnDim),
            textDim: int("text_dim", base.textDim), freqDim: int("freq_dim", base.freqDim),
            eps: (json["eps"] as? NSNumber)?.floatValue ?? base.eps,
            musicInjectLayers: (json["music_inject_layers"] as? [NSNumber])?.map(\.intValue) ?? base.musicInjectLayers)
    }
}

/// DiffSynth-Studio's `RMSNorm`: normalized in float32, rounded to the input's type, then scaled.
final class NFKWanDancerRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(_ dim: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let wide = x.asType(.float32)
        let normed = wide * rsqrt((wide * wide).mean(axis: -1, keepDims: true) + eps)
        return normed.asType(x.dtype) * weight
    }
}

/// A Wan attention in the original layout: `q`, `k`, `v`, `o` with query and key norms, and, for the
/// image-to-video cross-attention, `k_img`, `v_img`, `norm_k_img` over the context's first 257 tokens.
final class NFKWanDancerAttention: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "norm_q") var normQ: NFKWanDancerRMSNorm
    @ModuleInfo(key: "norm_k") var normK: NFKWanDancerRMSNorm
    @ModuleInfo(key: "k_img") var kImage: Linear?
    @ModuleInfo(key: "v_img") var vImage: Linear?
    @ModuleInfo(key: "norm_k_img") var normKImage: NFKWanDancerRMSNorm?

    static let imageTokens = 257
    let heads: Int

    init(dim: Int, heads: Int, eps: Float, imageInput: Bool) {
        self.heads = heads
        _q.wrappedValue = Linear(dim, dim)
        _k.wrappedValue = Linear(dim, dim)
        _v.wrappedValue = Linear(dim, dim)
        _o.wrappedValue = Linear(dim, dim)
        _normQ.wrappedValue = NFKWanDancerRMSNorm(dim, eps: eps)
        _normK.wrappedValue = NFKWanDancerRMSNorm(dim, eps: eps)
        if imageInput {
            _kImage.wrappedValue = Linear(dim, dim)
            _vImage.wrappedValue = Linear(dim, dim)
            _normKImage.wrappedValue = NFKWanDancerRMSNorm(dim, eps: eps)
        }
    }

    private func split(_ x: MLXArray) -> MLXArray {
        x.reshaped([x.dim(0), heads, x.dim(1) / heads]).transposed(1, 0, 2).expandedDimensions(axis: 0)
    }

    private func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let headDim = q.dim(-1)
        let out = NFKReferenceRounding.flashAttention(queries: q, keys: k, values: v,
                                                      scale: 1 / sqrt(Float(headDim)), mask: nil)
        return out[0].transposed(1, 0, 2).reshaped([q.dim(2), heads * headDim])
    }

    /// The self-attention of `x` `[N, dim]` under the rotary tables `cos`/`sin` `[N, headDim / 2]`.
    func selfAttention(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
        let n = x.dim(0), headDim = x.dim(1) / heads
        let query = zImageApplyRope(normQ(q(x)).reshaped([n, heads, headDim]), cos: c, sin: s)
        let key = zImageApplyRope(normK(k(x)).reshaped([n, heads, headDim]), cos: c, sin: s)
        let attended = attend(split(query.reshaped([n, -1])), split(key.reshaped([n, -1])), split(v(x)))
        return o(attended)
    }

    /// The cross-attention of `x` `[N, dim]` over `context` `[L, dim]`; with the image projections, the
    /// context's first 257 tokens feed the image branch and the rest the text branch.
    func crossAttention(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let image: MLXArray?, text: MLXArray
        if kImage != nil {
            image = context[0 ..< Self.imageTokens]
            text = context[Self.imageTokens...]
        } else {
            image = nil
            text = context
        }
        let query = split(normQ(q(x)))
        var attended = attend(query, split(normK(k(text))), split(v(text)))
        if let image, let kImage, let vImage, let normKImage {
            attended = attended + attend(query, split(normKImage(kImage(image))), split(vImage(image)))
        }
        return o(attended)
    }
}

/// A Wan 2.1 image-to-video block in the original layout (DiffSynth-Studio's `DiTBlock`).
final class NFKWanDancerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: NFKWanDancerAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: NFKWanDancerAttention
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: [Module]                              // [Linear, marker, Linear]
    @ParameterInfo(key: "modulation") var modulation: MLXArray

    let norm1: LayerNorm
    let norm2: LayerNorm

    init(_ c: NFKMLXWanDancerConfiguration) {
        _selfAttn.wrappedValue = NFKWanDancerAttention(dim: c.dim, heads: c.heads, eps: c.eps, imageInput: false)
        _crossAttn.wrappedValue = NFKWanDancerAttention(dim: c.dim, heads: c.heads, eps: c.eps, imageInput: true)
        _norm3.wrappedValue = NFKLayerNorm(dimensions: c.dim, eps: c.eps)
        _ffn.wrappedValue = [Linear(c.dim, c.ffnDim), Module(), Linear(c.ffnDim, c.dim)]
        _modulation.wrappedValue = MLXArray.zeros([1, 6, c.dim])
        norm1 = LayerNorm(dimensions: c.dim, eps: c.eps, affine: false)
        norm2 = LayerNorm(dimensions: c.dim, eps: c.eps, affine: false)
    }

    /// `x` `[N, dim]`, `context` `[L, dim]`, `timeModulation` `[6, dim]`, each operation at the stream's type
    /// as the reference's eager ops round.
    func callAsFunction(_ x: MLXArray, context: MLXArray, timeModulation: MLXArray, cos c: MLXArray,
                        sin s: MLXArray) -> MLXArray {
        let rows = modulation[0].asType(timeModulation.dtype) + timeModulation
        func row(_ i: Int) -> MLXArray { rows[i].reshaped([1, -1]) }
        var h = x + row(2) * selfAttn.selfAttention(norm1(x) * (1 + row(1)) + row(0), cos: c, sin: s)
        h = h + crossAttn.crossAttention(norm3(h), context: context)
        let fed = (ffn[2] as! Linear)(NFKReferenceRounding.geluTanh((ffn[0] as! Linear)(norm2(h) * (1 + row(4)) + row(3))))
        return h + row(5) * fed
    }
}

/// The image projection `img_emb` (and `img_emb_refimage`): a normalized two-layer MLP with the exact gelu.
final class NFKWanDancerImageProjection: Module {
    @ModuleInfo(key: "proj") var proj: [Module]                            // [LayerNorm, Linear, marker, Linear, LayerNorm]

    init(input: Int, output: Int) {
        _proj.wrappedValue = [NFKLayerNorm(dimensions: input), Linear(input, input), Module(), Linear(input, output),
                              NFKLayerNorm(dimensions: output)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = NFKReferenceRounding.gelu((proj[1] as! Linear)((proj[0] as! LayerNorm)(x)))
        return (proj[4] as! LayerNorm)((proj[3] as! Linear)(hidden))
    }
}

/// An output head: a non-affine norm modulated by the timestep embedding, then one linear.
final class NFKWanDancerHead: Module {
    @ModuleInfo(key: "head") var head: Linear
    @ParameterInfo(key: "modulation") var modulation: MLXArray
    let norm: LayerNorm

    init(_ c: NFKMLXWanDancerConfiguration) {
        _head.wrappedValue = Linear(c.dim, c.outChannels * c.patchSize.reduce(1, *))
        _modulation.wrappedValue = MLXArray.zeros([1, 2, c.dim])
        norm = LayerNorm(dimensions: c.dim, eps: c.eps, affine: false)
    }

    /// `x` `[N, dim]`, `embedding` `[1, dim]`.
    func callAsFunction(_ x: MLXArray, embedding: MLXArray) -> MLXArray {
        let rows = modulation[0].asType(embedding.dtype) + embedding
        return head(norm(x) * (1 + rows[1].reshaped([1, -1])) + rows[0].reshaped([1, -1]))
    }
}

/// One pre-norm layer of the music encoder (LODGE's `MusicEncoderLayer` with `norm_first`): torch's
/// `MultiheadAttention` whose query and key inputs are rotated first, then a gelu feed-forward.
final class NFKWanDancerMusicLayer: Module {
    final class Attention: Module {
        @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray
        @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ width: Int) {
            _inProjWeight.wrappedValue = MLXArray.zeros([3 * width, width])
            _inProjBias.wrappedValue = MLXArray.zeros([3 * width])
            _outProj.wrappedValue = Linear(width, width)
        }
    }

    @ModuleInfo(key: "self_attn") var selfAttn: Attention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    let heads: Int

    init(_ c: NFKMLXWanDancerConfiguration) {
        heads = c.musicHeads
        _selfAttn.wrappedValue = Attention(c.musicWidth)
        _linear1.wrappedValue = Linear(c.musicWidth, c.musicFeedForward)
        _linear2.wrappedValue = Linear(c.musicFeedForward, c.musicWidth)
        _norm1.wrappedValue = NFKLayerNorm(dimensions: c.musicWidth, eps: 1e-5)
        _norm2.wrappedValue = NFKLayerNorm(dimensions: c.musicWidth, eps: 1e-5)
    }

    /// `x` `[T, width]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = norm1(x)
        let rotated = Self.rotated(normed)
        let width = x.dim(1), headDim = width / heads, t = x.dim(0)
        let w = selfAttn.inProjWeight, b = selfAttn.inProjBias
        func project(_ input: MLXArray, _ part: Int) -> MLXArray {
            let rows = (part * width) ..< ((part + 1) * width)
            let projected = matmul(input, w[rows].transposed(1, 0)) + b[rows]
            return projected.reshaped([t, heads, headDim]).transposed(1, 0, 2).expandedDimensions(axis: 0)
        }
        let attended = NFKReferenceRounding.attention(queries: project(rotated, 0), keys: project(rotated, 1),
                                                      values: project(normed, 2), scale: 1 / sqrt(Float(headDim)),
                                                      mask: nil)
        let h = x + selfAttn.outProj(attended[0].transposed(1, 0, 2).reshaped([t, width]))
        return h + linear2(NFKReferenceRounding.gelu(linear1(norm2(h))))
    }

    /// `rotary_embedding_torch`'s `rotate_queries_or_keys` over the whole width: interleaved pairs rotated by
    /// position `p` at `p / 10000^(2i / width)`. The angles form in the input's type, as the reference's
    /// buffer is cast with the model.
    static func rotated(_ x: MLXArray) -> MLXArray {
        let t = x.dim(0), width = x.dim(1)
        let inverse = MLXArray((0 ..< width / 2).map { Float(1) / powf(10000, Float(2 * $0) / Float(width)) })
        let positions = MLXArray((0 ..< t).map { Float($0) }).asType(x.dtype)
        let angles = positions.reshaped([t, 1]) * inverse.asType(x.dtype).reshaped([1, -1])
        let pairs = x.reshaped([t, width / 2, 2])
        let first = pairs[0..., 0..., 0], second = pairs[0..., 0..., 1]
        let c = MLX.cos(angles), s = MLX.sin(angles)
        return stacked([first * c - second * s, second * c + first * s], axis: -1).reshaped([t, width])
    }
}

/// A music injector (Wan S2V's `AudioInjector` as Wan-Dancer adapts it): a cross-attention per block it
/// follows, each group of tokens attending to one music token.
final class NFKWanDancerMusicInjector: Module {
    @ModuleInfo(key: "injector") var injector: [NFKWanDancerAttention]

    init(_ c: NFKMLXWanDancerConfiguration) {
        _injector.wrappedValue = c.injectedBlocks.map { _ in
            NFKWanDancerAttention(dim: c.dim, heads: c.heads, eps: c.eps, imageInput: false)
        }
    }
}

/// The inputs of one Wan-Dancer pass besides the latent and the timestep.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanDancerInputs {
    /// The condition `[20, F', h, w]`: the keyframe mask folded to four channels and the keyframes' latents.
    public var condition: MLXArray
    /// The CLIP features `[257, imageDim]` of the first keyframe.
    public var keyframeFeatures: MLXArray
    /// The CLIP features `[257, imageDim]` of the reference image.
    public var referenceFeatures: MLXArray
    /// The music features `[T, 35]` (``NFKMLXWanDancerMusic/features(samples:sampleRate:)``).
    public var music: MLXArray
    /// The clip's frame rate: 30 for the local stage, `30 / k` for the global one.
    public var fps: Double
    /// The spans the music encoder reads, or nil for ``NFKMLXWanDancerConfiguration/musicEncoderSpans``.
    /// DiffSynth-Studio trains without sequence parallelism, so its music encoder reads one span.
    public var musicEncoderSpans: Int?

    public init(condition: MLXArray, keyframeFeatures: MLXArray, referenceFeatures: MLXArray, music: MLXArray,
                fps: Double, musicEncoderSpans: Int? = nil) {
        self.condition = condition
        self.keyframeFeatures = keyframeFeatures
        self.referenceFeatures = referenceFeatures
        self.music = music
        self.fps = fps
        self.musicEncoderSpans = musicEncoderSpans
    }

    /// Whether the clip runs at the global stage's reduced rate, which reads the global patch embedding and
    /// head and spreads the rotary frames.
    var isGlobal: Bool { Int((fps + 0.5).rounded(.down)) != 30 }
}

/// The Wan-Dancer transformer.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXWanDancerNet: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv3d
    @ModuleInfo(key: "patch_embedding_global") var patchEmbeddingGlobal: Conv3d
    @ModuleInfo(key: "text_embedding") var textEmbedding: [Module]         // [Linear, marker, Linear]
    @ModuleInfo(key: "time_embedding") var timeEmbedding: [Module]         // [Linear, marker, Linear]
    @ModuleInfo(key: "time_projection") var timeProjection: [Module]       // [marker, Linear]
    @ModuleInfo(key: "blocks") var blocks: [NFKWanDancerBlock]
    @ModuleInfo(key: "head") var head: NFKWanDancerHead
    @ModuleInfo(key: "head_global") var headGlobal: NFKWanDancerHead
    @ModuleInfo(key: "img_emb") var imageEmbedding: NFKWanDancerImageProjection
    @ModuleInfo(key: "img_emb_refimage") var referenceEmbedding: NFKWanDancerImageProjection
    @ModuleInfo(key: "ref_conv") var referenceConv: Conv2d
    @ModuleInfo(key: "music_projection") var musicProjection: Linear
    @ModuleInfo(key: "music_encoder") var musicEncoder: [NFKWanDancerMusicLayer]
    @ModuleInfo(key: "music_injector") var musicInjector: NFKWanDancerMusicInjector

    public let configuration: NFKMLXWanDancerConfiguration
    /// The blocks left in the release and read in their turn on every pass, where the load streams them.
    var layerStream: NFKMLXLayerStream?
    /// Whether a block's activations are recomputed in the backward pass. Introduced in InferKit 0.4.0.
    public var recomputesBlocksInBackward = false

    /// The block the reference skips on the unconditional pass (`enable_skip_layer`).
    public static let unconditionalSkippedBlock = 9
    /// The music features' frames and width after the encoder, which the reference interpolates to before
    /// interpolating to the clip (`N = 149`, `M = 4800`).
    static let musicPlan = (frames: 149, width: 4800)

    public init(_ c: NFKMLXWanDancerConfiguration) {
        configuration = c
        let patch = IntOrTriple(c.patchSize)
        _patchEmbedding.wrappedValue = Conv3d(inputChannels: c.inChannels, outputChannels: c.dim, kernelSize: patch, stride: patch)
        _patchEmbeddingGlobal.wrappedValue = Conv3d(inputChannels: c.inChannels, outputChannels: c.dim, kernelSize: patch,
                                                    stride: patch)
        _textEmbedding.wrappedValue = [Linear(c.textDim, c.dim), Module(), Linear(c.dim, c.dim)]
        _timeEmbedding.wrappedValue = [Linear(c.freqDim, c.dim), Module(), Linear(c.dim, c.dim)]
        _timeProjection.wrappedValue = [Module(), Linear(c.dim, 6 * c.dim)]
        _blocks.wrappedValue = (0 ..< c.layers).map { _ in NFKWanDancerBlock(c) }
        _head.wrappedValue = NFKWanDancerHead(c)
        _headGlobal.wrappedValue = NFKWanDancerHead(c)
        _imageEmbedding.wrappedValue = NFKWanDancerImageProjection(input: c.imageDim, output: c.dim)
        _referenceEmbedding.wrappedValue = NFKWanDancerImageProjection(input: c.imageDim, output: c.dim)
        _referenceConv.wrappedValue = Conv2d(inputChannels: 16, outputChannels: c.dim, kernelSize: 2, stride: 2)
        _musicProjection.wrappedValue = Linear(c.musicFeatures, c.musicWidth)
        _musicEncoder.wrappedValue = (0 ..< c.musicLayers).map { _ in NFKWanDancerMusicLayer(c) }
        _musicInjector.wrappedValue = NFKWanDancerMusicInjector(c)
    }

    // MARK: Embeddings

    /// `t` as the reference hands it to the model, rounded to the model's type.
    static func rounded(_ t: Double, to dtype: DType) -> Double {
        Double(MLXArray(Float(t)).asType(dtype).asType(.float32).item(Float.self))
    }

    /// The timestep embedding `[1, dim]` and the block modulation `[6, dim]` of `t`, its sinusoids formed in
    /// float64 and rounded to `dtype` (`sinusoidal_embedding_1d(...).to(position.dtype)`).
    func timeEmbedded(_ t: Double, dtype: DType) -> (embedding: MLXArray, modulation: MLXArray) {
        let half = configuration.freqDim / 2
        let frequencies = (0 ..< half).map { pow(10000.0, -Double($0) / Double(half)) }
        let sinusoid = frequencies.map { Float(cos(t * $0)) } + frequencies.map { Float(sin(t * $0)) }
        let x = MLXArray(sinusoid).reshaped([1, -1]).asType(dtype)
        let embedding = (timeEmbedding[2] as! Linear)(NFKReferenceRounding.silu((timeEmbedding[0] as! Linear)(x)))
        let modulation = (timeProjection[1] as! Linear)(NFKReferenceRounding.silu(embedding))
        return (embedding, modulation.reshaped([6, configuration.dim]))
    }

    /// The context `[1026, dim]`: the reference image's projected CLIP tokens, the first keyframe's, then the text.
    func context(text: MLXArray, inputs: NFKMLXWanDancerInputs, dtype: DType) -> MLXArray {
        let embeddedText = (textEmbedding[2] as! Linear)(NFKReferenceRounding.geluTanh((textEmbedding[0] as! Linear)(text.asType(dtype))))
        let keyframe = imageEmbedding(inputs.keyframeFeatures.asType(dtype))
        let reference = referenceEmbedding(inputs.referenceFeatures.asType(dtype))
        return concatenated([reference, keyframe, embeddedText], axis: 0)
    }

    // MARK: Rotary positions

    /// One axis's complex rotary table `[rows][d / 2]` (`precompute_freqs_cis`), in double precision.
    static func axisTable(_ d: Int, rows: Int) -> [[(re: Double, im: Double)]] {
        let frequencies = stride(from: 0, to: d, by: 2).prefix(d / 2).map { 1.0 / pow(10000.0, Double($0) / Double(d)) }
        return (0 ..< rows).map { p in frequencies.map { (re: cos(Double(p) * $0), im: sin(Double(p) * $0)) } }
    }

    /// The frame axis's rows for `frames` latent frames at `fps` (`get_single_freqs`): rows interpolated, as
    /// complex values, at `30 / (fps + 1e-6)` frames apart, the last row the table's row for the span's end.
    static func spreadRows(_ table: [[(re: Double, im: Double)]], frames: Int, fps: Double) -> [[(re: Double, im: Double)]] {
        let interval = 30.0 / (fps + 1e-6)
        let total = Int(interval * Double(frames) + 0.5)
        guard frames > 1 else { return [table[total - 1]] }
        var rows = [table[0]]
        for i in 1 ..< Swift.max(1, frames - 1) {
            let position = Double(i) * interval
            let low = Int(position), high = Swift.min(low + 1, total - 1)
            let upper = position - Double(low), lower = 1.0 - upper
            rows.append(zip(table[low], table[high]).map { (re: $0.re * lower + $1.re * upper, im: $0.im * lower + $1.im * upper) })
        }
        rows.append(table[total - 1])
        return rows
    }

    /// The rotary tables `cos`/`sin` `[f·h·w, headDim / 2]` of a patch grid, the frame axis spread at `fps` for
    /// the global stage.
    func ropeTables(frames f: Int, height h: Int, width w: Int, spreadAt fps: Double?) -> (cos: MLXArray, sin: MLXArray) {
        let d = configuration.headDim
        let frameDim = d - 2 * (d / 3), sideDim = d / 3
        let frameTable = Self.axisTable(frameDim, rows: configuration.ropeEnd)
        let frameRows = fps.map { Self.spreadRows(frameTable, frames: f, fps: $0) } ?? Array(frameTable[0 ..< f])
        let sideTable = Self.axisTable(sideDim, rows: Swift.max(h, w))
        var cosines = [Float](), sines = [Float]()
        cosines.reserveCapacity(f * h * w * d / 2)
        sines.reserveCapacity(f * h * w * d / 2)
        for fi in 0 ..< f {
            for hi in 0 ..< h {
                for wi in 0 ..< w {
                    for value in frameRows[fi] + sideTable[hi] + sideTable[wi] {
                        cosines.append(Float(value.re))
                        sines.append(Float(value.im))
                    }
                }
            }
        }
        return (MLXArray(cosines).reshaped([f * h * w, d / 2]), MLXArray(sines).reshaped([f * h * w, d / 2]))
    }

    // MARK: Music

    /// torch's `interpolate(mode: "bilinear", align_corners: false)` weights `[output, input]` along one axis,
    /// the source positions computed in float32 as the reference's kernel computes them.
    static func bilinearMatrix(from input: Int, to output: Int) -> MLXArray {
        let scale = Float(input) / Float(output)
        var weights = [Float](repeating: 0, count: output * input)
        for o in 0 ..< output {
            let source = Swift.max(0, scale * (Float(o) + 0.5) - 0.5)
            let low = Swift.min(Int(source), input - 1)
            let high = low < input - 1 ? low + 1 : low
            let upper = source - Float(low)
            weights[o * input + low] += 1 - upper
            weights[o * input + high] += upper
        }
        return MLXArray(weights).reshaped([output, input])
    }

    /// `x` `[A, B]` resized to `[rows, columns]` by bilinear interpolation, in float32 and rounded to `x`'s type.
    static func bilinear(_ x: MLXArray, rows: Int, columns: Int) -> MLXArray {
        let vertical = matmul(bilinearMatrix(from: x.dim(0), to: rows), x.asType(.float32))
        return matmul(vertical, bilinearMatrix(from: x.dim(1), to: columns).transposed(1, 0)).asType(x.dtype)
    }

    /// The music tokens `[8·frames, dim]` of a clip with `frames` latent frames: the features projected and
    /// encoded span by span (`spans`, ``NFKMLXWanDancerConfiguration/musicEncoderSpans`` by default), interpolated
    /// to 149 × 4800, then to eight tokens a latent frame at the model width.
    func musicTokens(_ features: MLXArray, frames: Int, dtype: DType, spans: Int? = nil) -> MLXArray {
        let spans = spans ?? configuration.musicEncoderSpans
        let length = features.dim(0) / spans * spans
        let projected = musicProjection(features[0 ..< length].asType(dtype))
        let spanLength = length / spans
        let encoded = concatenated((0 ..< spans).map { span in
            musicEncoder.reduce(projected[(span * spanLength) ..< ((span + 1) * spanLength)]) { $1($0) }
        }, axis: 0)
        let planned = Self.bilinear(encoded, rows: Self.musicPlan.frames, columns: Self.musicPlan.width)
        return Self.bilinear(planned, rows: 8 * frames, columns: configuration.dim)
    }

    /// `hidden` with injector `index`'s music added: each of the `8·frames` groups of tokens reads one music
    /// token. A softmax over one key is 1, so the attention returns the token's value.
    func injectingMusic(_ hidden: MLXArray, index: Int, music: MLXArray) -> MLXArray {
        let attention = musicInjector.injector[index]
        let groups = music.dim(0), perGroup = hidden.dim(0) / groups
        let values = attention.o(attention.v(music))                       // [groups, dim]
        let spread = broadcast(values.expandedDimensions(axis: 1), to: [groups, perGroup, values.dim(1)])
        return hidden + spread.reshaped([hidden.dim(0), values.dim(1)])
    }

    // MARK: The pass

    /// The velocity of the latent `x` `[16, F', h, w]` at timestep `t` (`model_fn_wan_video` with the music,
    /// the reference image, the dynamic frame rate, and the global model enabled), the unconditional pass
    /// skipping block ``unconditionalSkippedBlock`` where `skipsBlock`.
    public func callAsFunction(_ x: MLXArray, text: MLXArray, t: Double, inputs: NFKMLXWanDancerInputs,
                               skipsBlock: Bool = false) -> MLXArray {
        let c = configuration
        let dtype = x.dtype
        let (embedding, timeModulation) = timeEmbedded(Self.rounded(t, to: dtype), dtype: dtype)
        let context = self.context(text: text, inputs: inputs, dtype: dtype)
        let frames = x.dim(1), height = x.dim(2), width = x.dim(3)
        let ph = c.patchSize[1], pw = c.patchSize[2]
        let hh = height / ph, ww = width / pw
        let input = concatenated([x, inputs.condition.asType(dtype)], axis: 0)
        let embed = inputs.isGlobal ? patchEmbeddingGlobal : patchEmbedding
        var hidden = embed(input.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)).reshaped([-1, c.dim])
        let (cosT, sinT) = ropeTables(frames: frames, height: hh, width: ww, spreadAt: inputs.isGlobal ? inputs.fps : nil)
        let music = musicTokens(inputs.music, frames: frames, dtype: dtype, spans: inputs.musicEncoderSpans)
        let injected = Dictionary(uniqueKeysWithValues: c.injectedBlocks.enumerated().map { ($1, $0) })

        for (blockIndex, block) in blocks.enumerated() {
            if skipsBlock && blockIndex == Self.unconditionalSkippedBlock { continue }
            let musicIndex = injected[blockIndex]
            let run = { (y: MLXArray) -> MLXArray in
                var z = block(y, context: context, timeModulation: timeModulation, cos: cosT, sin: sinT)
                if let musicIndex {
                    z = self.injectingMusic(z, index: musicIndex, music: music)
                }
                return z
            }
            let current = hidden
            guard let output = NFKMLXLayerStream.run(layerStream, layer: block, index: blockIndex, {
                recomputesBlocksInBackward && layerStream == nil
                    ? NFKMLXCustomFunction.recomputed(block) { [run($0[0])] }([current])[0] : run(current)
            }, keeping: { [$0] }) else {
                continue
            }
            hidden = output
        }

        let out = (inputs.isGlobal ? headGlobal : head)(hidden, embedding: embedding)
        let pt = c.patchSize[0], oc = c.outChannels
        return out.reshaped([frames / pt, hh, ww, pt, ph, pw, oc]).transposed(6, 0, 3, 1, 4, 2, 5)
            .reshaped([oc, frames, height, width])
    }
}

extension NFKMLXWanDancerNet {
    /// A released tensor in the module's layout: a 5-D convolution to `[out, t, h, w, in]`, a 4-D one to
    /// `[out, h, w, in]`. Every name loads as released.
    static func adapted(key: String, value: MLXArray) -> (String, MLXArray) {
        switch value.ndim {
        case 5: return (key, value.transposed(0, 2, 3, 4, 1))
        case 4: return (key, value.transposed(0, 2, 3, 1))
        default: return (key, value)
        }
    }

    /// Loads one of a release's transformers (`global_model.safetensors` or `local_model.safetensors`) into `net`.
    static func loadWeights(into net: NFKMLXWanDancerNet, file: URL, precision: NFKMLXWeightPrecision = .float32) throws {
        let stored = try NFKMLXWeights.materializedCheckpoint(url: file).arrays
        let arrays = stored.keys.sorted().map { key -> (String, MLXArray) in
            let (name, shaped) = adapted(key: key, value: stored[key]!)
            return (name, precision == .checkpoint || !shaped.dtype.isFloatingPoint ? shaped : shaped.asType(.float32))
        }
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
    }
}
