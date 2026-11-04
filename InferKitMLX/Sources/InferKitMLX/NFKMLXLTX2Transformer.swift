//
//  NFKMLXLTX2Transformer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The LTX-2 audio-video transformer (`LTX2VideoTransformer3DModel`, Lightricks). One transformer
// denoises a video latent and an audio latent TOGETHER, so a generated clip carries its own sound
// rather than having one dubbed on. The LTX-Video DiT this package already ships
// (`NFKMLXLTXTransformer`) is the video-only ancestor; this is a different model, not a configuration
// of it.
//
// Each block runs six attentions. Video self-attention and audio self-attention carry the rotary;
// video-over-text and audio-over-text cross attention read the two text streams; and the two
// cross-modal directions join them — audio-to-video, where the video asks and the audio answers, and
// video-to-audio the other way. Every attention normalizes its query and key with an ACROSS-HEADS RMS
// norm (over the whole projected width, before the heads are split) and scales each head by
// `2 · sigmoid(linear(query input))`, so a zero-initialized gate leaves the attention unchanged.
//
// Modulation is PixArt-alpha's adaptive-norm-single raised to ten heads. Six of them live on the MODEL
// (video and audio timestep embeddings, two cross-modal scale/shift heads, two cross-modal gates) and
// each block adds its own `scale_shift_table` on top, so a block's parameters are the per-layer DELTA
// of a vector computed once. The rotary is the `split` kind: the channel axis halves into a real and an
// imaginary block rather than interleaving adjacent pairs, and its frequencies are a linear ramp in the
// exponent scaled by the patch's MIDPOINT in pixel-and-second space, not by its token index.
//
// For parity the two text streams are supplied directly, as they are for the LTX-Video, SD3 and FLUX
// transformers. The sampler's guidance passes reach the forward through three switches with no weights
// behind them: spatio-temporal guidance replaces both self-attentions of the listed blocks with their
// value projection (still gated and projected out), modality isolation skips both cross-modal
// attentions, and the cross timestep has each modality's cross-modal modulation read the OTHER
// modality's sigma.

/// LTX-2 transformer geometry.
public struct NFKMLXLTX2Configuration: Sendable {
    public var inChannels: Int
    public var outChannels: Int
    public var numAttentionHeads: Int
    public var attentionHeadDim: Int
    public var crossAttentionDim: Int
    public var patchSize: Int
    public var patchSizeT: Int
    public var vaeScaleFactors: [Int]
    public var posEmbedMaxPos: Int
    public var baseHeight: Int
    public var baseWidth: Int

    public var audioInChannels: Int
    public var audioOutChannels: Int
    public var audioNumAttentionHeads: Int
    public var audioAttentionHeadDim: Int
    public var audioCrossAttentionDim: Int
    public var audioScaleFactor: Int
    public var audioPosEmbedMaxPos: Int
    public var audioSamplingRate: Int
    public var audioHopLength: Int

    public var numLayers: Int
    public var normEps: Float
    public var captionChannels: Int
    public var ropeTheta: Float
    public var causalOffset: Int
    public var timestepScaleMultiplier: Float
    public var crossAttentionTimestepScaleMultiplier: Float
    /// Whether the text projections live in the transformer. LTX-2.0 carries them; LTX-2.3 and LTX-2.5
    /// project the text in the pipeline's connectors instead, so the transformer has none.
    public var usesPromptEmbeddings: Bool
    /// Whether the text cross-attention takes its own timestep-dependent modulation. LTX-2.5 turns this
    /// off, which drops the two prompt adaptive-norm heads and leaves the text key and value
    /// timestep-independent, so a sampler can cache them across denoising steps.
    public var usesPromptAdaptiveNorm: Bool
    /// Whether the video feed-forward's linears carry a bias. LTX-2.5 drops it.
    public var feedForwardBias: Bool
    public var audioFeedForwardBias: Bool
    /// Whether a learned absolute-position embedding marks generated-keyframe tokens (LTX-2.5).
    public var usesKeyframeEmbedding: Bool

    public init(inChannels: Int = 128, outChannels: Int = 128, numAttentionHeads: Int = 32,
                attentionHeadDim: Int = 128, crossAttentionDim: Int = 4096, patchSize: Int = 1,
                patchSizeT: Int = 1, vaeScaleFactors: [Int] = [8, 32, 32], posEmbedMaxPos: Int = 20,
                baseHeight: Int = 2048, baseWidth: Int = 2048,
                audioInChannels: Int = 128, audioOutChannels: Int = 128,
                audioNumAttentionHeads: Int = 32, audioAttentionHeadDim: Int = 64,
                audioCrossAttentionDim: Int = 2048, audioScaleFactor: Int = 4,
                audioPosEmbedMaxPos: Int = 20, audioSamplingRate: Int = 16000,
                audioHopLength: Int = 160, numLayers: Int = 48, normEps: Float = 1e-6,
                captionChannels: Int = 3840, ropeTheta: Float = 10000, causalOffset: Int = 1,
                timestepScaleMultiplier: Float = 1000,
                crossAttentionTimestepScaleMultiplier: Float = 1000,
                usesPromptEmbeddings: Bool = false, usesPromptAdaptiveNorm: Bool = true,
                feedForwardBias: Bool = true, audioFeedForwardBias: Bool = true,
                usesKeyframeEmbedding: Bool = false) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.numAttentionHeads = numAttentionHeads
        self.attentionHeadDim = attentionHeadDim
        self.crossAttentionDim = crossAttentionDim
        self.patchSize = patchSize
        self.patchSizeT = patchSizeT
        self.vaeScaleFactors = vaeScaleFactors
        self.posEmbedMaxPos = posEmbedMaxPos
        self.baseHeight = baseHeight
        self.baseWidth = baseWidth
        self.audioInChannels = audioInChannels
        self.audioOutChannels = audioOutChannels
        self.audioNumAttentionHeads = audioNumAttentionHeads
        self.audioAttentionHeadDim = audioAttentionHeadDim
        self.audioCrossAttentionDim = audioCrossAttentionDim
        self.audioScaleFactor = audioScaleFactor
        self.audioPosEmbedMaxPos = audioPosEmbedMaxPos
        self.audioSamplingRate = audioSamplingRate
        self.audioHopLength = audioHopLength
        self.numLayers = numLayers
        self.normEps = normEps
        self.captionChannels = captionChannels
        self.ropeTheta = ropeTheta
        self.causalOffset = causalOffset
        self.timestepScaleMultiplier = timestepScaleMultiplier
        self.crossAttentionTimestepScaleMultiplier = crossAttentionTimestepScaleMultiplier
        self.usesPromptEmbeddings = usesPromptEmbeddings
        self.usesPromptAdaptiveNorm = usesPromptAdaptiveNorm
        self.feedForwardBias = feedForwardBias
        self.audioFeedForwardBias = audioFeedForwardBias
        self.usesKeyframeEmbedding = usesKeyframeEmbedding
    }

    /// LTX-2.3 22B (`Lightricks/LTX-2.3`), the ungated release this port is held to structurally.
    public static let ltx23 = NFKMLXLTX2Configuration()

    /// LTX-2.5 22B (`Lightricks/LTX-2.5-Diffusers`, its `transformer/config.json`): LTX-2.3's geometry with
    /// the video feed-forward's bias dropped and the keyframe embedding added. The prompt modulation stays
    /// timestep-dependent, as in LTX-2.3.
    public static let ltx25 = NFKMLXLTX2Configuration(feedForwardBias: false, usesKeyframeEmbedding: true)

    /// A tiny random configuration for reference parity.
    public static let tiny = NFKMLXLTX2Configuration(
        inChannels: 8, outChannels: 8, numAttentionHeads: 2, attentionHeadDim: 16,
        crossAttentionDim: 32, baseHeight: 64, baseWidth: 64,
        audioInChannels: 6, audioOutChannels: 6, audioNumAttentionHeads: 2,
        audioAttentionHeadDim: 8, audioCrossAttentionDim: 16, numLayers: 2, captionChannels: 14)

    /// The tiny configuration with the three switches that differ across the LTX-2 transformers: the
    /// video feed-forward's bias dropped, the keyframe embedding added, and the prompt modulation made
    /// timestep-independent.
    public static let tiny25 = NFKMLXLTX2Configuration(
        inChannels: 8, outChannels: 8, numAttentionHeads: 2, attentionHeadDim: 16,
        crossAttentionDim: 32, baseHeight: 64, baseWidth: 64,
        audioInChannels: 6, audioOutChannels: 6, audioNumAttentionHeads: 2,
        audioAttentionHeadDim: 8, audioCrossAttentionDim: 16, numLayers: 2, captionChannels: 14,
        usesPromptAdaptiveNorm: false, feedForwardBias: false, usesKeyframeEmbedding: true)

    var innerDim: Int { numAttentionHeads * attentionHeadDim }
    var audioInnerDim: Int { audioNumAttentionHeads * audioAttentionHeadDim }
    /// Nine modulation parameters where the text cross-attention is modulated, six otherwise. Both
    /// released arrangements modulate it, so both carry nine.
    var modulationParameterCount: Int { 9 }
}

/// An affine-free RMS normalization over the channel axis.
func ltx2AffineFreeRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    guard NFKReferenceRounding.isReduced(x) else { return x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps) }
    let wide = x.asType(.float32)
    return (wide * rsqrt(wide.square().mean(axis: -1, keepDims: true) + eps)).asType(x.dtype)
}

/// The `split` rotary of LTX-2: the channel axis of each head halves into a real and an imaginary
/// block, which rotate against each other. FLUX and LTX-Video interleave adjacent pairs instead.
///
/// `x` `[B, S, heads · headDim]`, `cos`/`sin` `[B, heads, S, headDim / 2]` → `[B, S, heads · headDim]`.
func applyLTX2SplitRotary(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray, heads: Int) -> MLXArray {
    let batch = x.dim(0), sequence = x.dim(1)
    let headDim = x.dim(2) / heads
    let half = headDim / 2
    // The reference rotates in float32 and rounds once to the input's type.
    let headed = x.asType(.float32).reshaped([batch, sequence, heads, headDim]).transposed(0, 2, 1, 3)
    let first = headed[0..., 0..., 0..., 0 ..< half]
    let second = headed[0..., 0..., 0..., half...]
    let rotatedFirst = first * c - s * second
    let rotatedSecond = second * c + s * first
    return concatenated([rotatedFirst, rotatedSecond], axis: -1)
        .transposed(0, 2, 1, 3).reshaped([batch, sequence, heads * headDim]).asType(x.dtype)
}

/// The LTX-2 rotary table. The position of a token is the MIDPOINT of the pixel-space (or, for audio,
/// second-space) interval its patch covers, divided by the base extent, and the frequency ramp is
/// linear in the exponent rather than geometric in the index.
struct NFKLTX2Rope {
    let dim: Int
    let heads: Int
    let theta: Float

    /// `positions` `[axes][sequence]`, each already a fraction of its base extent → `(cos, sin)` each
    /// `[1, heads, sequence, dim / (2 · heads)]`.
    func table(positions: [[Float]]) -> (cos: MLXArray, sin: MLXArray) {
        let axes = positions.count
        let sequence = positions[0].count
        let ropeElements = axes * 2
        let steps = dim / ropeElements

        // `theta ^ linspace(0, 1, steps) · π / 2`, in double precision as the reference computes it.
        var frequencies = [Float](repeating: 0, count: steps)
        for index in 0 ..< steps {
            let fraction = steps == 1 ? 0 : Double(index) / Double(steps - 1)
            frequencies[index] = Float(pow(Double(theta), fraction) * Double.pi / 2)
        }

        // `(2 · position - 1) · frequency`, the axes interleaved per token: the reference transposes
        // the (axis, frequency) pair and flattens, so an axis's frequencies are strided, not blocked.
        var angles = [Float](repeating: 0, count: sequence * axes * steps)
        for token in 0 ..< sequence {
            for step in 0 ..< steps {
                for axis in 0 ..< axes {
                    let position = positions[axis][token] * 2 - 1
                    angles[(token * steps + step) * axes + axis] = position * frequencies[step]
                }
            }
        }
        let table = MLXArray(angles, [1, sequence, axes * steps])
        let width = axes * steps
        let expected = dim / 2
        var cosTable = cos(table), sinTable = sin(table)
        if expected > width {
            // The reference pads the LEADING channels with an identity rotation.
            let padding = expected - width
            cosTable = concatenated([MLXArray.ones([1, sequence, padding]), cosTable], axis: -1)
            sinTable = concatenated([MLXArray.zeros([1, sequence, padding]), sinTable], axis: -1)
        }
        func headed(_ t: MLXArray) -> MLXArray {
            t.reshaped([1, sequence, heads, expected / heads]).transposed(0, 2, 1, 3)
        }
        return (headed(cosTable), headed(sinTable))
    }

    /// The video grid's per-axis positions: each patch's midpoint in pixel space (its frame axis in
    /// seconds), as a fraction of the base extent.
    static func videoPositions(frames: Int, height: Int, width: Int,
                               configuration: NFKMLXLTX2Configuration, fps: Float) -> [[Float]] {
        let scales = configuration.vaeScaleFactors.map(Float.init)
        let patches = [configuration.patchSizeT, configuration.patchSize, configuration.patchSize]
        let extents = [Float(configuration.posEmbedMaxPos), Float(configuration.baseHeight),
                       Float(configuration.baseWidth)]
        var out = [[Float]](repeating: [], count: 3)
        for frame in stride(from: 0, to: frames, by: configuration.patchSizeT) {
            for row in stride(from: 0, to: height, by: configuration.patchSize) {
                for column in stride(from: 0, to: width, by: configuration.patchSize) {
                    let starts = [Float(frame), Float(row), Float(column)]
                    for axis in 0 ..< 3 {
                        var begin = starts[axis] * scales[axis]
                        var end = (starts[axis] + Float(patches[axis])) * scales[axis]
                        if axis == 0 {
                            // The causal VAE gives the first frame a temporal stride of one, so the
                            // timestamps shift back by the stride and clamp at zero, then convert to
                            // seconds.
                            begin = max(0, begin + Float(configuration.causalOffset) - scales[0]) / fps
                            end = max(0, end + Float(configuration.causalOffset) - scales[0]) / fps
                        }
                        out[axis].append((begin + end) / 2 / extents[axis])
                    }
                }
            }
        }
        return out
    }

    /// The audio grid's positions: each latent frame's midpoint in seconds, as a fraction of the base
    /// extent.
    static func audioPositions(frames: Int, configuration: NFKMLXLTX2Configuration) -> [[Float]] {
        let scale = Float(configuration.audioScaleFactor)
        let secondsPerBin = Float(configuration.audioHopLength) / Float(configuration.audioSamplingRate)
        var positions = [Float]()
        for frame in stride(from: 0, to: frames, by: configuration.patchSizeT) {
            let beginBin = max(0, Float(frame) * scale + Float(configuration.causalOffset) - scale)
            let endBin = max(0, (Float(frame) + Float(configuration.patchSizeT)) * scale
                             + Float(configuration.causalOffset) - scale)
            let begin = beginBin * secondsPerBin, end = endBin * secondsPerBin
            positions.append((begin + end) / 2 / Float(configuration.audioPosEmbedMaxPos))
        }
        return [positions]
    }
}

/// One adaptive-norm-single head: the sinusoidal timestep through an MLP, then SiLU and one linear into
/// `count` modulation vectors. The embedded timestep is returned too, because the output layer reads it
/// directly.
final class NFKLTX2AdaptiveNorm: Module {
    @ModuleInfo(key: "emb") var emb: NFKLTX2TimestepEmbedder
    @ModuleInfo(key: "linear") var linear: Linear

    init(dim: Int, count: Int) {
        _emb.wrappedValue = NFKLTX2TimestepEmbedder(dim)
        _linear.wrappedValue = Linear(dim, count * dim)
    }

    func callAsFunction(_ timestep: MLXArray) -> (modulation: MLXArray, embedded: MLXArray) {
        let embedded = emb(timestep)
        return (linear(NFKReferenceRounding.silu(embedded)), embedded)
    }
}

/// The `emb` wrapper holding the timestep embedder, so the release's
/// `…emb.timestep_embedder.linear_1` keys match.
final class NFKLTX2TimestepEmbedder: Module {
    @ModuleInfo(key: "timestep_embedder") var embedder: NFKLTXLinearAct

    init(_ dim: Int) { _embedder.wrappedValue = NFKLTXLinearAct(256, dim, dim) }

    /// The float32 sinusoid takes the embedder's type, as the reference casts it to the hidden states'.
    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        embedder(ltxTimestepEmbedding(timestep).asType(NFKReferenceRounding.parameterType(of: embedder)))
    }
}

/// An LTX-2 attention. The query and key take an across-heads RMS norm before the heads are split, the
/// rotary (where one is supplied) applies at that same full width, and each head is scaled by
/// `2 · sigmoid` of a per-head logit read off the query input.
final class NFKLTX2Attention: Module {
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]                        // [Linear, dropout-marker]
    @ModuleInfo(key: "to_gate_logits") var toGateLogits: Linear?

    let heads: Int
    let headDim: Int

    init(queryDim: Int, contextDim: Int, heads: Int, headDim: Int, bias: Bool, outBias: Bool,
         eps: Float, gated: Bool) {
        self.heads = heads
        self.headDim = headDim
        let inner = heads * headDim
        _normQ.wrappedValue = RMSNorm(dimensions: inner, eps: eps)
        _normK.wrappedValue = RMSNorm(dimensions: inner, eps: eps)
        _toQ.wrappedValue = Linear(queryDim, inner, bias: bias)
        _toK.wrappedValue = Linear(contextDim, inner, bias: bias)
        _toV.wrappedValue = Linear(contextDim, inner, bias: bias)
        _toOut.wrappedValue = [Linear(inner, queryDim, bias: outBias), NFKSD3Marker()]
        _toGateLogits.wrappedValue = gated ? Linear(queryDim, heads) : nil
    }

    /// `hidden` is the query side, `context` the key and value side (the same tensor for
    /// self-attention). `queryRotary` and `keyRotary` are nil where the attention does not rotate.
    /// `mask` is additive over the keys, `[B, 1, 1, keys]`. A `perturbed` attention returns its value
    /// projection in place of the attended values, the spatio-temporal guidance pass; with `keep`
    /// `[B, 1, 1]` it blends the two per example, `value + keep · (attended − value)`, so a 0 perturbs that
    /// example and a 1 leaves it attended.
    func callAsFunction(_ hidden: MLXArray, context: MLXArray,
                        queryRotary: (MLXArray, MLXArray)?,
                        keyRotary: (MLXArray, MLXArray)?, mask: MLXArray? = nil,
                        perturbed: Bool = false, keep: MLXArray? = nil) -> MLXArray {
        let batch = hidden.dim(0)
        let gates = toGateLogits.map { 2 * NFKReferenceRounding.sigmoid($0(hidden)) }
        let value = toV(context)
        if perturbed, keep == nil {
            var merged = value.reshaped([batch, -1, heads, headDim])
            if let gates {
                merged = merged * gates[0..., 0..., 0..., .newAxis]
            }
            return (toOut[0] as! Linear)(merged.reshaped([batch, -1, heads * headDim]))
        }
        var query = normQ(toQ(hidden))
        var key = normK(toK(context))
        if let queryRotary {
            query = applyLTX2SplitRotary(query, cos: queryRotary.0, sin: queryRotary.1, heads: heads)
            let forKey = keyRotary ?? queryRotary
            key = applyLTX2SplitRotary(key, cos: forKey.0, sin: forKey.1, heads: heads)
        }
        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped([batch, -1, heads, headDim]).transposed(0, 2, 1, 3)
        }
        let attended = NFKReferenceRounding.flashAttention(
            queries: split(query), keys: split(key), values: split(value),
            scale: 1 / sqrt(Float(headDim)), mask: mask?.asType(query.dtype))
        var merged = attended.transposed(0, 2, 1, 3)                       // [B, S, heads, headDim]
        if perturbed, let keep {
            let projected = value.reshaped([batch, -1, heads, headDim])
            merged = projected + keep.asType(merged.dtype).reshaped([batch, 1, 1, 1]) * (merged - projected)
        }
        if let gates {
            merged = merged * gates[0..., 0..., 0..., .newAxis]
        }
        return (toOut[0] as! Linear)(merged.reshaped([batch, -1, heads * headDim]))
    }
}

/// The gelu-approximate feed-forward, whose `net` is a `[Module]` array so the reference's Sequential
/// indices (`net.0.proj`, an activation slot, `net.2`) match.
final class NFKLTX2FeedForward: Module {
    @ModuleInfo(key: "net") var net: [Module]

    init(dim: Int, bias: Bool) {
        _net.wrappedValue = [NFKLTX2GELUProjection(dim, dim * 4, bias: bias), NFKSD3Marker(),
                             Linear(dim * 4, dim, bias: bias)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (net[2] as! Linear)(NFKReferenceRounding.geluTanh((net[0] as! NFKLTX2GELUProjection)(x)))
    }
}

/// The feed-forward's first entry, whose linear the reference keeps under `proj`.
final class NFKLTX2GELUProjection: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ inDim: Int, _ outDim: Int, bias: Bool) { _proj.wrappedValue = Linear(inDim, outDim, bias: bias) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

/// One LTX-2 block: six attentions and two feed-forwards over the paired video and audio streams.
final class NFKLTX2Block: Module {
    @ModuleInfo(key: "attn1") var attn1: NFKLTX2Attention
    @ModuleInfo(key: "audio_attn1") var audioAttn1: NFKLTX2Attention
    @ModuleInfo(key: "attn2") var attn2: NFKLTX2Attention
    @ModuleInfo(key: "audio_attn2") var audioAttn2: NFKLTX2Attention
    @ModuleInfo(key: "audio_to_video_attn") var audioToVideoAttn: NFKLTX2Attention
    @ModuleInfo(key: "video_to_audio_attn") var videoToAudioAttn: NFKLTX2Attention
    @ModuleInfo(key: "ff") var ff: NFKLTX2FeedForward
    @ModuleInfo(key: "audio_ff") var audioFF: NFKLTX2FeedForward
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ParameterInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray
    @ParameterInfo(key: "prompt_scale_shift_table") var promptScaleShiftTable: MLXArray
    @ParameterInfo(key: "audio_prompt_scale_shift_table") var audioPromptScaleShiftTable: MLXArray
    @ParameterInfo(key: "video_a2v_cross_attn_scale_shift_table") var videoCrossTable: MLXArray
    @ParameterInfo(key: "audio_a2v_cross_attn_scale_shift_table") var audioCrossTable: MLXArray

    let config: NFKMLXLTX2Configuration

    init(_ config: NFKMLXLTX2Configuration) {
        self.config = config
        let dim = config.innerDim, audioDim = config.audioInnerDim
        let heads = config.numAttentionHeads, headDim = config.attentionHeadDim
        let audioHeads = config.audioNumAttentionHeads, audioHeadDim = config.audioAttentionHeadDim
        let eps = config.normEps
        _attn1.wrappedValue = NFKLTX2Attention(
            queryDim: dim, contextDim: dim, heads: heads, headDim: headDim, bias: true, outBias: true,
            eps: eps, gated: true)
        _audioAttn1.wrappedValue = NFKLTX2Attention(
            queryDim: audioDim, contextDim: audioDim, heads: audioHeads, headDim: audioHeadDim,
            bias: true, outBias: true, eps: eps, gated: true)
        _attn2.wrappedValue = NFKLTX2Attention(
            queryDim: dim, contextDim: config.crossAttentionDim, heads: heads, headDim: headDim,
            bias: true, outBias: true, eps: eps, gated: true)
        _audioAttn2.wrappedValue = NFKLTX2Attention(
            queryDim: audioDim, contextDim: config.audioCrossAttentionDim, heads: audioHeads,
            headDim: audioHeadDim, bias: true, outBias: true, eps: eps, gated: true)
        // Both cross-modal attentions run at the AUDIO head geometry, whichever way they point.
        _audioToVideoAttn.wrappedValue = NFKLTX2Attention(
            queryDim: dim, contextDim: audioDim, heads: audioHeads, headDim: audioHeadDim,
            bias: true, outBias: true, eps: eps, gated: true)
        _videoToAudioAttn.wrappedValue = NFKLTX2Attention(
            queryDim: audioDim, contextDim: dim, heads: audioHeads, headDim: audioHeadDim,
            bias: true, outBias: true, eps: eps, gated: true)
        _ff.wrappedValue = NFKLTX2FeedForward(dim: dim, bias: config.feedForwardBias)
        _audioFF.wrappedValue = NFKLTX2FeedForward(dim: audioDim, bias: config.audioFeedForwardBias)
        let count = config.modulationParameterCount
        _scaleShiftTable.wrappedValue = MLXArray.zeros([count, dim])
        _audioScaleShiftTable.wrappedValue = MLXArray.zeros([count, audioDim])
        _promptScaleShiftTable.wrappedValue = MLXArray.zeros([2, dim])
        _audioPromptScaleShiftTable.wrappedValue = MLXArray.zeros([2, audioDim])
        _videoCrossTable.wrappedValue = MLXArray.zeros([5, dim])
        _audioCrossTable.wrappedValue = MLXArray.zeros([5, audioDim])
    }

    /// A per-layer table added to a globally computed modulation vector, split into its parameter sets.
    /// `table` `[count, dim]`, `temb` `[B, T, count · dim]` → `count` arrays of `[B, T, dim]`.
    static func modulation(table: MLXArray, temb: MLXArray) -> [MLXArray] {
        let count = table.dim(0), dim = table.dim(1)
        let values = table.reshaped([1, 1, count, dim])
            + temb.reshaped([temb.dim(0), temb.dim(1), count, dim])
        return (0 ..< count).map { values[0..., 0..., $0, 0...] }
    }

    struct Streams {
        var video: MLXArray
        var audio: MLXArray
    }

    /// One block over both streams. `promptModulation` is nil where the release leaves the text
    /// key/value modulation timestep-independent, which is LTX-2.5's arrangement: the per-layer table
    /// is then used on its own. `textMask` and `audioTextMask` are additive over the text tokens;
    /// `perturbed` replaces both self-attentions with their value projections (blended per example by
    /// `keep` where it is given), and `crossModal` false skips both cross-modal attentions.
    func callAsFunction(_ streams: Streams, text: MLXArray, audioText: MLXArray,
                        temb: MLXArray, audioTemb: MLXArray,
                        crossScaleShift: MLXArray, audioCrossScaleShift: MLXArray,
                        crossGate: MLXArray, audioCrossGate: MLXArray,
                        promptModulation: MLXArray?, audioPromptModulation: MLXArray?,
                        videoRotary: (MLXArray, MLXArray), audioRotary: (MLXArray, MLXArray),
                        crossVideoRotary: (MLXArray, MLXArray),
                        crossAudioRotary: (MLXArray, MLXArray), textMask: MLXArray? = nil,
                        audioTextMask: MLXArray? = nil, perturbed: Bool = false, keep: MLXArray? = nil,
                        crossModal: Bool = true) -> Streams {
        let eps = config.normEps
        var video = streams.video
        var audio = streams.audio

        let videoMod = NFKLTX2Block.modulation(table: scaleShiftTable, temb: temb)
        let audioMod = NFKLTX2Block.modulation(table: audioScaleShiftTable, temb: audioTemb)

        // 1. Self-attention on each stream.
        let normedVideo = ltx2AffineFreeRMSNorm(video, eps: eps) * (1 + videoMod[1]) + videoMod[0]
        video = video + videoMod[2] * attn1(normedVideo, context: normedVideo,
                                            queryRotary: videoRotary, keyRotary: nil, perturbed: perturbed,
                                            keep: keep)
        let normedAudio = ltx2AffineFreeRMSNorm(audio, eps: eps) * (1 + audioMod[1]) + audioMod[0]
        audio = audio + audioMod[2] * audioAttn1(normedAudio, context: normedAudio,
                                                 queryRotary: audioRotary, keyRotary: nil,
                                                 perturbed: perturbed, keep: keep)

        // 2. Cross-attention over the two text streams. The query is modulated by the block's last
        // three parameters; the key and value side is modulated by the prompt table.
        let promptMod = promptModulation.map { NFKLTX2Block.modulation(table: promptScaleShiftTable, temb: $0) }
            ?? [promptScaleShiftTable[0][.newAxis, .newAxis, 0...],
                promptScaleShiftTable[1][.newAxis, .newAxis, 0...]]
        let audioPromptMod = audioPromptModulation.map {
            NFKLTX2Block.modulation(table: audioPromptScaleShiftTable, temb: $0)
        } ?? [audioPromptScaleShiftTable[0][.newAxis, .newAxis, 0...],
              audioPromptScaleShiftTable[1][.newAxis, .newAxis, 0...]]

        let videoText = text * (1 + promptMod[1]) + promptMod[0]
        let audioTextModulated = audioText * (1 + audioPromptMod[1]) + audioPromptMod[0]
        video = video + videoMod[8] * attn2(
            ltx2AffineFreeRMSNorm(video, eps: eps) * (1 + videoMod[7]) + videoMod[6],
            context: videoText, queryRotary: nil, keyRotary: nil, mask: textMask)
        audio = audio + audioMod[8] * audioAttn2(
            ltx2AffineFreeRMSNorm(audio, eps: eps) * (1 + audioMod[7]) + audioMod[6],
            context: audioTextModulated, queryRotary: nil, keyRotary: nil, mask: audioTextMask)
        if crossModal {
            (video, audio) = crossModalUpdate(video: video, audio: audio, crossScaleShift: crossScaleShift,
                                              audioCrossScaleShift: audioCrossScaleShift, crossGate: crossGate,
                                              audioCrossGate: audioCrossGate, crossVideoRotary: crossVideoRotary,
                                              crossAudioRotary: crossAudioRotary)
        }

        // 4. Feed-forward.
        video = video + videoMod[5] * ff(
            ltx2AffineFreeRMSNorm(video, eps: eps) * (1 + videoMod[4]) + videoMod[3])
        audio = audio + audioMod[5] * audioFF(
            ltx2AffineFreeRMSNorm(audio, eps: eps) * (1 + audioMod[4]) + audioMod[3])
        return Streams(video: video, audio: audio)
    }

    /// The two cross-modal directions.
    func crossModalUpdate(video inputVideo: MLXArray, audio inputAudio: MLXArray, crossScaleShift: MLXArray,
                          audioCrossScaleShift: MLXArray, crossGate: MLXArray, audioCrossGate: MLXArray,
                          crossVideoRotary: (MLXArray, MLXArray),
                          crossAudioRotary: (MLXArray, MLXArray)) -> (MLXArray, MLXArray) {
        let eps = config.normEps
        var video = inputVideo, audio = inputAudio

        // 3. The two cross-modal directions, both reading the SAME normalized streams: the video's
        // update must not be visible to the audio's.
        let normVideo = ltx2AffineFreeRMSNorm(video, eps: eps)
        let normAudio = ltx2AffineFreeRMSNorm(audio, eps: eps)
        let videoCross = NFKLTX2Block.modulation(
            table: videoCrossTable[0 ..< 4, 0...], temb: crossScaleShift)
        let audioCross = NFKLTX2Block.modulation(
            table: audioCrossTable[0 ..< 4, 0...], temb: audioCrossScaleShift)
        let a2vGate = NFKLTX2Block.modulation(table: videoCrossTable[4..., 0...], temb: crossGate)[0]
        let v2aGate = NFKLTX2Block.modulation(table: audioCrossTable[4..., 0...], temb: audioCrossGate)[0]

        let a2vVideo = normVideo * (1 + videoCross[0]) + videoCross[1]
        let a2vAudio = normAudio * (1 + audioCross[0]) + audioCross[1]
        video = video + a2vGate * audioToVideoAttn(
            a2vVideo, context: a2vAudio, queryRotary: crossVideoRotary, keyRotary: crossAudioRotary)

        let v2aVideo = normVideo * (1 + videoCross[2]) + videoCross[3]
        let v2aAudio = normAudio * (1 + audioCross[2]) + audioCross[3]
        audio = audio + v2aGate * videoToAudioAttn(
            v2aAudio, context: v2aVideo, queryRotary: crossAudioRotary, keyRotary: crossVideoRotary)
        return (video, audio)
    }
}

/// One forward pass of the LTX-2 transformer, the inputs of the single-pass call: a sampler's guided
/// pair, perturbed pass, and isolated pass are three of these.
public struct NFKMLXLTX2Pass {
    public var video: MLXArray
    public var audio: MLXArray
    public var text: MLXArray
    public var audioText: MLXArray
    public var timestep: MLXArray
    public var audioTimestep: MLXArray
    public var sigma: MLXArray
    public var audioSigma: MLXArray?
    public var textMask: MLXArray?
    public var keyframes: MLXArray?
    public var perturbedBlocks: Set<Int>
    /// Which examples the ``perturbedBlocks`` perturb, `[B]`: 0 where spatio-temporal guidance applies and
    /// 1 where the example stays attended, the reference's `perturbation_mask`. Nil perturbs every example.
    public var perturbationMask: MLXArray?
    public var isolatesModalities: Bool

    public init(video: MLXArray, audio: MLXArray, text: MLXArray, audioText: MLXArray, timestep: MLXArray,
                audioTimestep: MLXArray, sigma: MLXArray, audioSigma: MLXArray? = nil, textMask: MLXArray? = nil,
                keyframes: MLXArray? = nil, perturbedBlocks: Set<Int> = [], perturbationMask: MLXArray? = nil,
                isolatesModalities: Bool = false) {
        self.perturbationMask = perturbationMask
        self.video = video
        self.audio = audio
        self.text = text
        self.audioText = audioText
        self.timestep = timestep
        self.audioTimestep = audioTimestep
        self.sigma = sigma
        self.audioSigma = audioSigma
        self.textMask = textMask
        self.keyframes = keyframes
        self.perturbedBlocks = perturbedBlocks
        self.isolatesModalities = isolatesModalities
    }
}

/// The LTX-2 audio-video transformer.
public final class NFKMLXLTX2TransformerNet: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "audio_proj_in") var audioProjIn: Linear
    @ModuleInfo(key: "caption_projection") var captionProjection: NFKLTXCaptionProjection?
    @ModuleInfo(key: "audio_caption_projection") var audioCaptionProjection: NFKLTXCaptionProjection?
    @ModuleInfo(key: "time_embed") var timeEmbed: NFKLTX2AdaptiveNorm
    @ModuleInfo(key: "audio_time_embed") var audioTimeEmbed: NFKLTX2AdaptiveNorm
    @ModuleInfo(key: "av_cross_attn_video_scale_shift") var crossVideoScaleShift: NFKLTX2AdaptiveNorm
    @ModuleInfo(key: "av_cross_attn_audio_scale_shift") var crossAudioScaleShift: NFKLTX2AdaptiveNorm
    @ModuleInfo(key: "av_cross_attn_video_a2v_gate") var crossVideoGate: NFKLTX2AdaptiveNorm
    @ModuleInfo(key: "av_cross_attn_audio_v2a_gate") var crossAudioGate: NFKLTX2AdaptiveNorm
    @ModuleInfo(key: "prompt_adaln") var promptAdaptiveNorm: NFKLTX2AdaptiveNorm?
    @ModuleInfo(key: "audio_prompt_adaln") var audioPromptAdaptiveNorm: NFKLTX2AdaptiveNorm?
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [NFKLTX2Block]
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "audio_proj_out") var audioProjOut: Linear
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ParameterInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray
    @ParameterInfo(key: "keyframes_abs_pos_embedding") var keyframeEmbedding: MLXArray?

    public let config: NFKMLXLTX2Configuration

    /// The reader of the blocks a streamed load leaves in the release, read in their turn on every pass;
    /// nil where every block is held.
    var layerStream: NFKMLXLayerStream?

    /// Whether a block's activations are recomputed from its inputs in the backward pass rather than held
    /// from the forward one (gradient checkpointing, MLX's `checkpoint`), the trainer's
    /// `enable_gradient_checkpointing`. A training step then holds one block's activations at a time, at the
    /// cost of a second forward pass. The forward's values are unchanged. Off by default.
    ///
    /// Introduced in InferKit 0.4.0.
    public var recomputesBlocksInBackward = false

    /// The type the transformer computes in: its input projection's, which a quantized build keeps at the
    /// stored type.
    var computeType: DType {
        (projIn as? QuantizedLinear)?.scales.dtype ?? projIn.weight.dtype
    }

    public init(_ config: NFKMLXLTX2Configuration) {
        self.config = config
        let dim = config.innerDim, audioDim = config.audioInnerDim
        _projIn.wrappedValue = Linear(config.inChannels, dim)
        _audioProjIn.wrappedValue = Linear(config.audioInChannels, audioDim)
        _captionProjection.wrappedValue = config.usesPromptEmbeddings
            ? NFKLTXCaptionProjection(config.captionChannels, dim) : nil
        _audioCaptionProjection.wrappedValue = config.usesPromptEmbeddings
            ? NFKLTXCaptionProjection(config.captionChannels, audioDim) : nil
        let count = config.modulationParameterCount
        _timeEmbed.wrappedValue = NFKLTX2AdaptiveNorm(dim: dim, count: count)
        _audioTimeEmbed.wrappedValue = NFKLTX2AdaptiveNorm(dim: audioDim, count: count)
        _crossVideoScaleShift.wrappedValue = NFKLTX2AdaptiveNorm(dim: dim, count: 4)
        _crossAudioScaleShift.wrappedValue = NFKLTX2AdaptiveNorm(dim: audioDim, count: 4)
        _crossVideoGate.wrappedValue = NFKLTX2AdaptiveNorm(dim: dim, count: 1)
        _crossAudioGate.wrappedValue = NFKLTX2AdaptiveNorm(dim: audioDim, count: 1)
        _promptAdaptiveNorm.wrappedValue = config.usesPromptAdaptiveNorm
            ? NFKLTX2AdaptiveNorm(dim: dim, count: 2) : nil
        _audioPromptAdaptiveNorm.wrappedValue = config.usesPromptAdaptiveNorm
            ? NFKLTX2AdaptiveNorm(dim: audioDim, count: 2) : nil
        _transformerBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in NFKLTX2Block(config) }
        _projOut.wrappedValue = Linear(dim, config.outChannels)
        _audioProjOut.wrappedValue = Linear(audioDim, config.audioOutChannels)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([2, dim])
        _audioScaleShiftTable.wrappedValue = MLXArray.zeros([2, audioDim])
        _keyframeEmbedding.wrappedValue = config.usesKeyframeEmbedding
            ? MLXArray.zeros([1, dim]) : nil
    }

    /// The paired velocity prediction. `video` `[B, videoTokens, inChannels]`, `audio`
    /// `[B, audioTokens, audioInChannels]`, `text` `[B, textTokens, crossAttentionDim]` (or
    /// `captionChannels` where the release keeps the caption projections), `timestep`
    /// `[B, videoTokens]` (or `[B]`, one value a row) and `audioTimestep` `[B, audioTokens]` (or `[B]`)
    /// already scaled by the release's multiplier, `sigma` `[B]`. `keyframes` marks the tokens whose
    /// latent holds a single pixel frame.
    ///
    /// @discussion The sampler's passes: `textMask` `[B, textTokens]` (1 for a real token) masks both
    /// text cross-attentions; `crossTimestep` has the video's cross-modal modulation read `audioSigma`
    /// (`sigma` where nil) and the audio's read `sigma`, the pipeline's `use_cross_timestep`;
    /// `perturbedBlocks` lists the blocks whose self-attentions return their value projection
    /// (spatio-temporal guidance), over the examples where `perturbationMask` `[B]` is 0, or the whole batch
    /// where it is nil; `isolatesModalities` skips both cross-modal
    /// attentions in every block. `gatesReadCrossSigma` has each modality's cross-modal gate read the
    /// other modality's sigma while its cross-modal scale and shift read its own timestep, the
    /// arrangement of Lightricks' `ltx_core`, which its trainer trains under; it differs from both
    /// `crossTimestep` settings only where a modality's timesteps vary across its tokens.
    public func callAsFunction(video videoLatents: MLXArray, audio audioLatents: MLXArray,
                               text: MLXArray, audioText: MLXArray, timestep: MLXArray,
                               audioTimestep: MLXArray, sigma: MLXArray, frames: Int, height: Int,
                               width: Int, audioFrames: Int, fps: Float = 24,
                               keyframes: MLXArray? = nil, textMask: MLXArray? = nil,
                               audioSigma: MLXArray? = nil, crossTimestep: Bool = false,
                               gatesReadCrossSigma: Bool = false, perturbedBlocks: Set<Int> = [],
                               perturbationMask: MLXArray? = nil,
                               isolatesModalities: Bool = false) -> (video: MLXArray, audio: MLXArray) {
        let pass = NFKMLXLTX2Pass(video: videoLatents, audio: audioLatents, text: text, audioText: audioText,
                                  timestep: timestep, audioTimestep: audioTimestep, sigma: sigma,
                                  audioSigma: audioSigma, textMask: textMask, keyframes: keyframes,
                                  perturbedBlocks: perturbedBlocks, perturbationMask: perturbationMask,
                                  isolatesModalities: isolatesModalities)
        return predictions([pass], frames: frames, height: height, width: width, audioFrames: audioFrames, fps: fps,
                           crossTimestep: crossTimestep, gatesReadCrossSigma: gatesReadCrossSigma)[0]
    }

    /// The velocity of each of `passes`, each the value the single-pass call gives it alone. Every block
    /// runs every pass in turn, so a streamed block is read once for all of them.
    ///
    /// `videoCoordinates`, where given, places the video tokens in place of the `frames × height × width` grid:
    /// a DFR pass appends keyframe, slot, and reference tokens after the grid, each with its own interval.
    public func predictions(_ passes: [NFKMLXLTX2Pass], frames: Int, height: Int, width: Int, audioFrames: Int,
                            fps: Float = 24, crossTimestep: Bool = false, gatesReadCrossSigma: Bool = false,
                            videoCoordinates: NFKMLXLTX2VideoCoordinates? = nil) -> [(video: MLXArray, audio: MLXArray)] {
        let videoRope = NFKLTX2Rope(dim: config.innerDim, heads: config.numAttentionHeads,
                                    theta: config.ropeTheta)
        let audioRope = NFKLTX2Rope(dim: config.audioInnerDim, heads: config.audioNumAttentionHeads,
                                    theta: config.ropeTheta)
        // The cross-modal rotary runs at the AUDIO cross-attention width over the TIME axis alone, one
        // table per modality from that modality's own time positions.
        let crossRope = NFKLTX2Rope(dim: config.audioCrossAttentionDim,
                                    heads: config.audioNumAttentionHeads, theta: config.ropeTheta)
        let videoPositions = videoCoordinates?.positions(config) ?? NFKLTX2Rope.videoPositions(
            frames: frames, height: height, width: width, configuration: config, fps: fps)
        let audioPositions = NFKLTX2Rope.audioPositions(frames: audioFrames, configuration: config)
        let videoRotary = videoRope.table(positions: videoPositions)
        let audioRotary = audioRope.table(positions: audioPositions)
        let crossVideoRotary = crossRope.table(positions: [videoPositions[0]])
        let crossAudioRotary = crossRope.table(positions: audioPositions)

        var prepared = passes.map {
            prepare($0, crossTimestep: crossTimestep, gatesReadCrossSigma: gatesReadCrossSigma)
        }
        for (index, block) in transformerBlocks.enumerated() {
            func run(_ p: Prepared, _ streams: NFKLTX2Block.Streams) -> NFKLTX2Block.Streams {
                block(streams, text: p.text, audioText: p.audioText, temb: p.temb,
                      audioTemb: p.audioTemb, crossScaleShift: p.crossScaleShift,
                      audioCrossScaleShift: p.audioCrossScaleShift, crossGate: p.crossGate,
                      audioCrossGate: p.audioCrossGate, promptModulation: p.promptModulation,
                      audioPromptModulation: p.audioPromptModulation, videoRotary: videoRotary,
                      audioRotary: audioRotary, crossVideoRotary: crossVideoRotary,
                      crossAudioRotary: crossAudioRotary, textMask: p.textMask,
                      audioTextMask: p.textMask, perturbed: p.perturbedBlocks.contains(index),
                      keep: p.perturbationMask, crossModal: p.crossModal)
            }
            guard let outputs = NFKMLXLayerStream.run(layerStream, layer: block, index: index, {
                prepared.map { p in
                    recomputesBlocksInBackward && layerStream == nil
                        ? Self.recomputed(block) { run(p, $0) }(p.streams) : run(p, p.streams)
                }
            }, keeping: { outputs in outputs.flatMap { [$0.video, $0.audio] } }) else {
                continue
            }
            for (pass, output) in outputs.enumerated() { prepared[pass].streams = output }
        }

        // The output modulation reads the EMBEDDED timestep rather than the expanded one, and its two
        // parameters are shift then scale — the opposite order to the blocks'.
        func finish(_ x: MLXArray, table: MLXArray, embedded: MLXArray, projection: Linear) -> MLXArray {
            let dim = table.dim(1)
            let values = table.reshaped([1, 1, 2, dim])
                + embedded.reshaped([x.dim(0), -1, embedded.dim(-1)])[0..., 0..., .newAxis, 0...]
            let shift = values[0..., 0..., 0, 0...], scale = values[0..., 0..., 1, 0...]
            return projection(ltx2LayerNorm(x) * (1 + scale) + shift)
        }
        return prepared.map { p in
            (finish(p.streams.video, table: scaleShiftTable, embedded: p.embedded, projection: projOut),
             finish(p.streams.audio, table: audioScaleShiftTable, embedded: p.audioEmbedded, projection: audioProjOut))
        }
    }

    /// `body` over `block` as MLX's `checkpoint` builds it (`nn.utils.checkpoint`): the block's trainable
    /// arrays and both streams are the inputs of a custom function whose backward runs `body` again over the
    /// same inputs, so none of the block's activations outlive its forward pass. The other inputs the block
    /// reads (the modulation, the text, the rotary tables) depend on nothing trainable and enter as constants.
    static func recomputed(_ block: NFKLTX2Block, _ body: @escaping (NFKLTX2Block.Streams) -> NFKLTX2Block.Streams)
        -> (NFKLTX2Block.Streams) -> NFKLTX2Block.Streams {
        let trainable = block.trainableParameters().flattened()
        let keys = trainable.map(\.0)
        let forward: ([MLXArray]) -> [MLXArray] = { arrays in
            let held = block.trainableParameters()
            block.update(parameters: ModuleParameters.unflattened(Array(zip(keys, arrays.prefix(keys.count)))))
            let output = body(NFKLTX2Block.Streams(video: arrays[keys.count], audio: arrays[keys.count + 1]))
            block.update(parameters: held)
            return [output.video, output.audio]
        }
        let checkpointed = NFKMLXCustomFunction.make {
            Forward(forward)
            VJP { primals, cotangents in vjp(forward, primals: primals, cotangents: cotangents).1 }
        }
        return { streams in
            let outputs = checkpointed(trainable.map(\.1) + [streams.video, streams.audio])
            return NFKLTX2Block.Streams(video: outputs[0], audio: outputs[1])
        }
    }

    /// One pass's block inputs: the projected streams and every modulation the blocks read.
    struct Prepared {
        var streams: NFKLTX2Block.Streams
        let text: MLXArray, audioText: MLXArray
        let temb: MLXArray, audioTemb: MLXArray, embedded: MLXArray, audioEmbedded: MLXArray
        let crossScaleShift: MLXArray, audioCrossScaleShift: MLXArray, crossGate: MLXArray, audioCrossGate: MLXArray
        let promptModulation: MLXArray?, audioPromptModulation: MLXArray?
        let textMask: MLXArray?
        let perturbedBlocks: Set<Int>
        let perturbationMask: MLXArray?
        let crossModal: Bool
    }

    func prepare(_ pass: NFKMLXLTX2Pass, crossTimestep: Bool, gatesReadCrossSigma: Bool = false) -> Prepared {
        let batch = pass.video.dim(0)
        let sigma = pass.sigma, audioSigma = pass.audioSigma ?? pass.sigma
        var video = projIn(pass.video)
        let audio = audioProjIn(pass.audio)
        if let keyframeEmbedding, let keyframes = pass.keyframes {
            video = video + (keyframes .> 0).asType(video.dtype)[0..., 0..., .newAxis] * keyframeEmbedding
        }

        func reshaped(_ x: MLXArray) -> MLXArray { x.reshaped([batch, -1, x.dim(-1)]) }
        let (tembFlat, embedded) = timeEmbed(pass.timestep.reshaped([-1]))
        let (audioTembFlat, audioEmbedded) = audioTimeEmbed(pass.audioTimestep.reshaped([-1]))

        let gateScale = config.crossAttentionTimestepScaleMultiplier / config.timestepScaleMultiplier
        let videoCrossTimestep = crossTimestep ? audioSigma.reshaped([-1]) : pass.timestep.reshaped([-1])
        let audioCrossTimestep = crossTimestep ? sigma.reshaped([-1]) : pass.audioTimestep.reshaped([-1])
        let videoGateTimestep = gatesReadCrossSigma ? audioSigma.reshaped([-1]) : videoCrossTimestep
        let audioGateTimestep = gatesReadCrossSigma ? sigma.reshaped([-1]) : audioCrossTimestep

        var textStream = pass.text, audioTextStream = pass.audioText
        if let captionProjection, let audioCaptionProjection {
            textStream = captionProjection(pass.text)
            audioTextStream = audioCaptionProjection(pass.audioText)
        }
        return Prepared(
            streams: NFKLTX2Block.Streams(video: video, audio: audio), text: textStream, audioText: audioTextStream,
            temb: reshaped(tembFlat), audioTemb: reshaped(audioTembFlat), embedded: embedded,
            audioEmbedded: audioEmbedded,
            crossScaleShift: reshaped(crossVideoScaleShift(videoCrossTimestep).modulation),
            audioCrossScaleShift: reshaped(crossAudioScaleShift(audioCrossTimestep).modulation),
            crossGate: reshaped(crossVideoGate(videoGateTimestep * gateScale).modulation),
            audioCrossGate: reshaped(crossAudioGate(audioGateTimestep * gateScale).modulation),
            promptModulation: promptAdaptiveNorm.map { reshaped($0(sigma.reshaped([-1])).modulation) },
            audioPromptModulation: audioPromptAdaptiveNorm.map { reshaped($0(audioSigma.reshaped([-1])).modulation) },
            // The reference turns the multiplicative mask into a `-10000` bias in the hidden type.
            textMask: pass.textMask.map { ((1 - $0.asType(video.dtype)) * -10000).reshaped([batch, 1, 1, -1]) },
            perturbedBlocks: pass.perturbedBlocks, perturbationMask: pass.perturbationMask,
            crossModal: !pass.isolatesModalities)
    }

    /// Throws the first failed read of a streamed block, and clears it so the next pass reads every block
    /// again. Every pass after a failed read skips every block, so a caller holding a streamed transformer
    /// checks after each step. Nothing is thrown where every block is held.
    public func verifyStream() throws {
        guard let stream = layerStream, let failure = stream.readFailure else { return }
        stream.clearFailure()
        throw failure
    }

    /// The geometry of a released LTX-2 transformer, from its `transformer/config.json`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXLTX2Configuration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "LTX2VideoTransformer3DModel" {
            throw NFKMLXError.unsupportedConfiguration("this reads an LTX-2 transformer, not \(kind)")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func flag(_ key: String, _ fallback: Bool) -> Bool { (json[key] as? NSNumber)?.boolValue ?? fallback }
        func number(_ key: String, _ fallback: Float) -> Float {
            (json[key] as? NSNumber)?.floatValue ?? fallback
        }
        if let rope = json["rope_type"] as? String, rope != "split" {
            throw NFKMLXError.unsupportedConfiguration(
                "this builds the split rotary; the release asks for \(rope)")
        }
        // Both released arrangements modulate the text cross-attention, which is what fixes the
        // modulation vector at nine parameters. A release that turned it off would carry six, and a
        // module built here would read the wrong slices of a vector that still loaded.
        if !flag("cross_attn_mod", true) || !flag("audio_cross_attn_mod", true) {
            throw NFKMLXError.unsupportedConfiguration(
                "this builds the nine-parameter modulation both released arrangements carry; the "
                + "release turns the text cross-attention modulation off")
        }
        // The remaining switches this reader does not vary. Each is refused by name rather than
        // ignored: a release that turned the attention gates off would ship no `to_gate_logits`, and
        // the loader's error would name a missing tensor rather than the configuration that asked for
        // it.
        if !flag("gated_attn", true) || !flag("audio_gated_attn", true) {
            throw NFKMLXError.unsupportedConfiguration(
                "this builds the per-head attention gates both released arrangements carry; the "
                + "release turns them off")
        }
        if let norm = json["qk_norm"] as? String, norm != "rms_norm_across_heads" {
            throw NFKMLXError.unsupportedConfiguration(
                "this builds the across-heads query and key norm; the release asks for \(norm)")
        }
        if flag("norm_elementwise_affine", false) {
            throw NFKMLXError.unsupportedConfiguration(
                "this builds the affine-free normalizations the releases carry; the release's own "
                + "normalizations are affine")
        }
        if let activation = json["activation_fn"] as? String, activation != "gelu-approximate" {
            throw NFKMLXError.unsupportedConfiguration(
                "this builds the tanh-approximate GELU feed-forward; the release asks for \(activation)")
        }
        let scales = (json["vae_scale_factors"] as? [Any])?.compactMap { ($0 as? NSNumber)?.intValue }
            ?? [8, 32, 32]
        return NFKMLXLTX2Configuration(
            inChannels: integer("in_channels", 128), outChannels: integer("out_channels", 128),
            numAttentionHeads: integer("num_attention_heads", 32),
            attentionHeadDim: integer("attention_head_dim", 128),
            crossAttentionDim: integer("cross_attention_dim", 4096),
            patchSize: integer("patch_size", 1), patchSizeT: integer("patch_size_t", 1),
            vaeScaleFactors: scales, posEmbedMaxPos: integer("pos_embed_max_pos", 20),
            baseHeight: integer("base_height", 2048), baseWidth: integer("base_width", 2048),
            audioInChannels: integer("audio_in_channels", 128),
            audioOutChannels: integer("audio_out_channels", 128),
            audioNumAttentionHeads: integer("audio_num_attention_heads", 32),
            audioAttentionHeadDim: integer("audio_attention_head_dim", 64),
            audioCrossAttentionDim: integer("audio_cross_attention_dim", 2048),
            audioScaleFactor: integer("audio_scale_factor", 4),
            audioPosEmbedMaxPos: integer("audio_pos_embed_max_pos", 20),
            audioSamplingRate: integer("audio_sampling_rate", 16000),
            audioHopLength: integer("audio_hop_length", 160),
            numLayers: integer("num_layers", 48), normEps: number("norm_eps", 1e-6),
            captionChannels: integer("caption_channels", 3840),
            ropeTheta: number("rope_theta", 10000), causalOffset: integer("causal_offset", 1),
            timestepScaleMultiplier: number("timestep_scale_multiplier", 1000),
            crossAttentionTimestepScaleMultiplier:
                number("cross_attn_timestep_scale_multiplier", 1000),
            usesPromptEmbeddings: flag("use_prompt_embeddings", false),
            usesPromptAdaptiveNorm: flag("use_prompt_adaln_single", true),
            feedForwardBias: flag("ff_bias", true),
            audioFeedForwardBias: flag("audio_ff_bias", true),
            usesKeyframeEmbedding: flag("use_keyframes_abs_pos_embedding", false))
    }

    /// Loads a released LTX-2 transformer directory into `net`. Every weight is at most 2-D, so no
    /// layout change is needed.
    public static func loadWeights(into net: NFKMLXLTX2TransformerNet, from directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        try loadWeights(into: net, from: directory, precision: precision, streamed: [])
    }

    /// Loads a released transformer directory into `net` one block at a time, quantizing each block's linear
    /// layers to `bits` as it arrives, so the float transformer is never resident whole. The layers outside
    /// the blocks (the input and output projections, the modulation heads) stay at the stored type, as the
    /// reference trainer's quantizer excludes them.
    static func loadQuantizedWeights(into net: NFKMLXLTX2TransformerNet, from directory: URL, bits: Int,
                                     groupSize: Int) throws {
        // Evaluating the whole network would materialize every block's initial weights, so the layers
        // outside the blocks are placed and evaluated by themselves, and each block on its own.
        let outside = try NFKMLXReleaseWeights.materializedArrays(
            inDirectory: directory, precision: .checkpoint, remap: { block(ofKey: $0) == nil ? $0 : nil })
        let provided = Set(outside.map(\.0))
        let missing = net.parameters().flattened().map(\.0)
            .filter { !$0.hasPrefix("transformer_blocks.") && !provided.contains($0) }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch("the release lacks \(missing.count) transformer weight(s): "
                                              + missing.prefix(3).joined(separator: ", "))
        }
        net.update(parameters: ModuleParameters.unflattened(outside))
        eval(outside.map(\.1))
        for (index, layer) in net.transformerBlocks.enumerated() {
            let arrays = try NFKMLXReleaseWeights.materializedArrays(
                inDirectory: directory, precision: .checkpoint,
                remap: { key in block(ofKey: key).flatMap { $0.layer == index ? $0.name : nil } })
            try NFKMLXWeights.apply(arrays, to: layer, verifyShapes: true)
            try NFKMLXQuantization.quantize(module: layer, bits: bits, groupSize: groupSize)
            eval(layer)
        }
    }

    /// The block a released transformer key belongs to and its name inside the block, or nil for a key
    /// outside `transformer_blocks`.
    static func block(ofKey key: String) -> (layer: Int, name: String)? {
        NFKMLXLayerStream.layer(ofDecoderName: key, layersKey: "transformer_blocks")
    }

    /// What a released transformer directory holds at `precision`, each block counted as a repeated
    /// layer.
    static func footprint(directory: URL, precision: NFKMLXWeightPrecision) throws -> NFKMLXStageFootprint {
        try NFKMLXLayerStream.footprint(directory: directory, precision: precision, includes: { _ in true },
                                        layerOf: { block(ofKey: $0)?.layer })
    }

    /// Loads a released transformer directory into `net`, leaving the `streamed` blocks in the release to
    /// be read in their turn on every pass (``layerStream``).
    static func loadWeights(into net: NFKMLXLTX2TransformerNet, from directory: URL, precision: NFKMLXWeightPrecision,
                            streamed: Set<Int>) throws {
        let vacated = try NFKMLXLayerStream.vacating(streamed, of: net.transformerBlocks, prefix: "transformer_blocks.",
                                                     directory: directory, precision: precision,
                                                     layerOf: block(ofKey:))
        let held: (String) -> String? = { key in
            block(ofKey: key).map { streamed.contains($0.layer) } == true ? nil : key
        }
        let arrays = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision, remap: held)
        try NFKMLXWeights.apply(arrays + (vacated?.placeholders ?? []), to: net, verifyShapes: true)
        net.layerStream = vacated?.stream
        vacated?.stream.prime()
    }
}

/// The output layer's affine-free layer normalization, whose epsilon the reference fixes at `1e-6`
/// rather than reading `norm_eps`.
func ltx2LayerNorm(_ x: MLXArray) -> MLXArray { sd3AffineFreeLayerNorm(x, eps: 1e-6) }
