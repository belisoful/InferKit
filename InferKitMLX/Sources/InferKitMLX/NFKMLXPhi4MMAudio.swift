//
//  NFKMLXPhi4MMAudio.swift
//  InferKitMLX
//
//  Phi-4-multimodal's speech tower: Microsoft's `ConformerEncoder` (the `cascades` audio processor).
//  A per-feature mean/variance normalization, a NeMo depthwise-striding convolutional subsampler that
//  reduces time by eight and lifts 80 mel channels to the model width, then 24 macaron Conformer blocks
//  with a T5 relative-position attention bias and a causal gated convolution module. A two-layer MLP
//  projects the encoder output into the decoder's embedding space; the projector has a `speech` and a
//  `vision` head, chosen by the input mode.
//
//  Attention is full and bidirectional: the released config sets `chunk_size = -1`, which makes the
//  streaming mask span the whole sequence, so the only positional signal is the T5 bias. The `causal`
//  flag applies to the convolution module alone (left-pad, right-trim), not the attention. Past 500
//  subsampled frames (40 s) the encoder unfolds the sequence into independent 500-frame windows, the
//  last zero-padded; a lone clip's padding stays visible to attention, as the reference computes it.
//  Several clips run as one batch padded to the longest, with each clip's padding masked out of the keys.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// The Conformer geometry, from the release config's `audio_processor.config`.
public struct NFKMLXPhi4MMAudioConfiguration: Sendable {
    public var mels = 80
    public var modelDimensions = 1024
    public var layerCount = 24
    public var heads = 16
    public var feedForwardDimensions = 1536
    public var convKernel = 3
    public var timeReduction = 8
    public var subsamplingChannels = 1024
    public var t5Buckets = 1000               // 2 · t5_bias_max_distance (asymmetric)
    public var t5MaxDistance = 500
    public var attentionWindow = 500          // the encoder's fixed `max_seq_len` unfold size
    public var projectionDimensions = 3072    // the decoder hidden size

    public init() {}
    public static let released = NFKMLXPhi4MMAudioConfiguration()

    /// A shrunk geometry that runs with random weights, for examples and tests. It keeps the 80-band input
    /// and the eight-fold subsampler, so it reads the real feature extractor's output.
    public static let tiny: NFKMLXPhi4MMAudioConfiguration = {
        var c = NFKMLXPhi4MMAudioConfiguration()
        c.modelDimensions = 32
        c.layerCount = 2
        c.heads = 2
        c.feedForwardDimensions = 64
        c.subsamplingChannels = 16
        c.t5MaxDistance = 16
        c.t5Buckets = 32
        c.projectionDimensions = 48
        return c
    }()
}

/// Per-feature normalization, `(x - mean) · invstd`, with the statistics stored in the checkpoint.
final class NFKPhi4MMAudioNorm: Module {
    @ParameterInfo(key: "global_mean") var mean: MLXArray
    @ParameterInfo(key: "global_invstd") var invStandardDeviation: MLXArray

    init(_ mels: Int) {
        _mean.wrappedValue = MLXArray.zeros([mels])
        _invStandardDeviation.wrappedValue = MLXArray.ones([mels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { (x - mean) * invStandardDeviation }
}

/// The NeMo depthwise-striding subsampler: a full 3×3 stride-2 convolution, then two (depthwise 3×3
/// stride-2, pointwise 1×1) stages, ReLU after each, over the `(time, mel)` plane as a one-channel
/// image; the result flattens `(channel, mel)` per frame into a linear projection to the model width.
/// The `nn.Sequential` ReLU markers (indices 1, 4, 7) are kept so the checkpoint keys match.
final class NFKPhi4MMAudioSubsampling: Module {
    @ModuleInfo(key: "conv") var conv: [Module]
    @ModuleInfo(key: "out") var out: Linear

    init(_ c: NFKMLXPhi4MMAudioConfiguration) {
        let channels = c.subsamplingChannels
        var stack: [Module] = [
            NFKConv2d(inputChannels: 1, outputChannels: channels, kernelSize: 3, stride: 2, padding: 1), Module(),
        ]
        let stages = Int(log2(Double(c.timeReduction)))
        for _ in 1 ..< stages {
            stack.append(NFKConv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3,
                                stride: 2, padding: 1, groups: channels))
            stack.append(NFKConv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1))
            stack.append(Module())
        }
        _conv.wrappedValue = stack
        let reducedMels = (0 ..< stages).reduce(c.mels) { m, _ in (m + 2 - 3) / 2 + 1 }
        _out.wrappedValue = Linear(channels * reducedMels, c.modelDimensions)
    }

    /// `[1, T, mels]` → `[1, T/8, dModel]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.expandedDimensions(axis: 3)                               // [1, T, mels, 1] NHWC
        for module in conv {
            if let layer = module as? Conv2d { h = layer(h) } else { h = relu(h) }
        }
        let (b, t, f, c) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        return out(h.transposed(0, 1, 3, 2).reshaped([b, t, c * f]))        // (channel, mel) order
    }
}

/// The T5 relative-position attention bias: a learned scalar per head for each clipped relative
/// distance, added to the attention logits. Distances are used directly (no bucketing) and the range is
/// asymmetric, so the table holds `2 · max_distance` entries indexed by `distance + max_distance`.
final class NFKPhi4MMT5Bias: Module {
    @ModuleInfo(key: "bias_values") var biasValues: Embedding
    let maxDistance: Int

    init(_ c: NFKMLXPhi4MMAudioConfiguration) {
        _biasValues.wrappedValue = Embedding(embeddingCount: c.t5Buckets, dimensions: c.heads)
        maxDistance = c.t5MaxDistance
    }

    /// `[1, heads, length, length]`.
    func callAsFunction(length: Int) -> MLXArray {
        let context = MLXArray(Int32(0) ..< Int32(length)).reshaped([length, 1])
        let memory = MLXArray(Int32(0) ..< Int32(length)).reshaped([1, length])
        var relative = memory - context                                     // [length, length]
        relative = clip(relative, min: MLXArray(Int32(-maxDistance)), max: MLXArray(Int32(maxDistance - 1)))
        let index = (relative + Int32(maxDistance)).reshaped([length * length])
        let looked = biasValues(index).reshaped([length, length, -1])       // [L, L, heads]
        return looked.transposed(2, 0, 1).expandedDimensions(axis: 0)       // [1, heads, L, L]
    }
}

/// A gated linear unit over a linear projection: `Linear` to twice the width, then `first · swish(gate)`.
final class NFKPhi4MMGLULinear: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    init(_ inputDimensions: Int, _ outputDimensions: Int) {
        _linear.wrappedValue = Linear(inputDimensions, 2 * outputDimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = linear(x)
        let half = projected.dim(projected.ndim - 1) / 2
        return projected[.ellipsis, 0 ..< half] * NFKReferenceRounding.swish(projected[.ellipsis, half...])
    }
}

/// The macaron feed-forward: layer norm, a GLU linear to the inner width, then a linear back. The
/// `nn.Sequential` dropout markers (indices 1, 3) are kept so the checkpoint keys match.
final class NFKPhi4MMAudioFeedForward: Module {
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "net") var net: [Module]

    init(_ c: NFKMLXPhi4MMAudioConfiguration) {
        _norm.wrappedValue = NFKLayerNorm(dimensions: c.modelDimensions)
        _net.wrappedValue = [
            NFKPhi4MMGLULinear(c.modelDimensions, c.feedForwardDimensions), Module(),
            Linear(c.feedForwardDimensions, c.modelDimensions), Module(),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm(x)
        h = (net[0] as! NFKPhi4MMGLULinear)(h)
        return (net[2] as! Linear)(h)
    }
}

/// The causal gated convolution module: layer norm, a pointwise GLU convolution, a depthwise separable
/// convolution whose future context is trimmed away, Swish, and a final pointwise convolution. Every
/// convolution is channels-last, so the reference's channel-first permutations are absent.
final class NFKPhi4MMAudioConv: Module {
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "glu") var glu: NFKPhi4MMConvGLU
    @ModuleInfo(key: "dw_sep_conv_1d") var depthwiseSeparable: NFKPhi4MMDepthwiseSeparable
    @ModuleInfo(key: "ext_pw_conv_1d") var pointwise: Conv1d
    let kernel: Int

    init(_ c: NFKMLXPhi4MMAudioConfiguration) {
        let d = c.modelDimensions
        _norm.wrappedValue = NFKLayerNorm(dimensions: d)
        _glu.wrappedValue = NFKPhi4MMConvGLU(d)
        _depthwiseSeparable.wrappedValue = NFKPhi4MMDepthwiseSeparable(d, kernel: c.convKernel)
        _pointwise.wrappedValue = NFKConv1d(inputChannels: d, outputChannels: d, kernelSize: 1)
        kernel = c.convKernel
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let length = x.dim(1)
        var h = glu(norm(x))
        h = depthwiseSeparable(h)
        h = h[0..., 0 ..< length, 0...]                                     // causal trim of the future context
        h = NFKReferenceRounding.swish(h)
        return pointwise(h)
    }
}

/// The pointwise GLU convolution: a 1×1 convolution to twice the width, split into a value and a gate
/// with their own biases, then `(value + b1) · swish(gate + b2)`.
final class NFKPhi4MMConvGLU: Module {
    @ModuleInfo(key: "ext_pw_conv_1d") var conv: Conv1d
    @ParameterInfo(key: "b1") var bias1: MLXArray
    @ParameterInfo(key: "b2") var bias2: MLXArray

    init(_ d: Int) {
        _conv.wrappedValue = NFKConv1d(inputChannels: d, outputChannels: 2 * d, kernelSize: 1)
        _bias1.wrappedValue = MLXArray.zeros([1, d, 1])
        _bias2.wrappedValue = MLXArray.zeros([1, d, 1])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = conv(x)                                            // [B, T, 2d]
        let half = projected.dim(projected.ndim - 1) / 2
        // The stored biases are `[1, channels, 1]` (channel-first); channels-last needs `[1, 1, channels]`.
        let b1 = bias1.reshaped([1, 1, half])
        let b2 = bias2.reshaped([1, 1, half])
        return (projected[.ellipsis, 0 ..< half] + b1) * NFKReferenceRounding.swish(projected[.ellipsis, half...] + b2)
    }
}

/// A depthwise 1-D convolution (kernel `k`, one group per channel, left-padded for causality) followed
/// by a pointwise 1×1 convolution. The left padding is the kernel minus one; the caller trims the
/// matching future context after both convolutions.
final class NFKPhi4MMDepthwiseSeparable: Module {
    @ModuleInfo(key: "dw_conv") var depthwise: Conv1d
    @ModuleInfo(key: "pw_conv") var pointwise: Conv1d

    init(_ d: Int, kernel: Int) {
        _depthwise.wrappedValue = NFKConv1d(inputChannels: d, outputChannels: d, kernelSize: kernel,
                                         padding: kernel - 1, groups: d)
        _pointwise.wrappedValue = NFKConv1d(inputChannels: d, outputChannels: d, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { pointwise(depthwise(x)) }
}

/// Full multi-head attention with the T5 relative-position bias added to the logits. The query is
/// scaled by `1/sqrt(head)`; keys, values, and queries all use the model width.
final class NFKPhi4MMAudioAttention: Module {
    @ModuleInfo(key: "linear_q") var q: Linear
    @ModuleInfo(key: "linear_k") var k: Linear
    @ModuleInfo(key: "linear_v") var v: Linear
    @ModuleInfo(key: "linear_out") var out: Linear
    let heads: Int
    let headDimensions: Int

    init(_ c: NFKMLXPhi4MMAudioConfiguration) {
        let d = c.modelDimensions
        heads = c.heads
        headDimensions = d / c.heads
        _q.wrappedValue = Linear(d, d)
        _k.wrappedValue = Linear(d, d)
        _v.wrappedValue = Linear(d, d)
        _out.wrappedValue = Linear(d, d)
    }

    /// `keyMask` (`[batch, 1, 1, time]`, true where a key is real) removes padded keys the way the
    /// reference's masked softmax does: their logits go to minus infinity and their weights to zero, so a
    /// row with every key masked yields zeros rather than NaN.
    func callAsFunction(_ x: MLXArray, bias: MLXArray, keyMask: MLXArray? = nil) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        func split(_ y: MLXArray) -> MLXArray { y.reshaped([b, t, heads, headDimensions]).transposed(0, 2, 1, 3) }
        let qh = NFKReferenceRounding.scaled(split(q(x)), by: 1 / sqrt(Float(headDimensions)))
        let kh = split(k(x)), vh = split(v(x))
        var scores = matmul(qh, kh.transposed(0, 1, 3, 2)) + bias
        var weights: MLXArray
        if let keyMask {
            scores = MLX.where(keyMask, scores, MLXArray(-Float.infinity).asType(scores.dtype))
            weights = MLX.where(keyMask, softmax(scores, axis: -1, precise: true), MLXArray(0).asType(scores.dtype))
        } else {
            weights = softmax(scores, axis: -1, precise: true)
        }
        let attended = matmul(weights, vh)
        return out(attended.transposed(0, 2, 1, 3).reshaped([b, t, heads * headDimensions]))
    }
}

/// One macaron Conformer block: half a feed-forward, attention over the normalized input, the
/// convolution module, half a feed-forward, then a final layer norm.
final class NFKPhi4MMAudioLayer: Module {
    @ModuleInfo(key: "feed_forward_in") var feedForwardIn: NFKPhi4MMAudioFeedForward
    @ModuleInfo(key: "self_attn") var attention: NFKPhi4MMAudioAttention
    @ModuleInfo(key: "conv") var conv: NFKPhi4MMAudioConv
    @ModuleInfo(key: "feed_forward_out") var feedForwardOut: NFKPhi4MMAudioFeedForward
    @ModuleInfo(key: "layer_norm_att") var attentionNorm: LayerNorm
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm

    init(_ c: NFKMLXPhi4MMAudioConfiguration) {
        _feedForwardIn.wrappedValue = NFKPhi4MMAudioFeedForward(c)
        _attention.wrappedValue = NFKPhi4MMAudioAttention(c)
        _conv.wrappedValue = NFKPhi4MMAudioConv(c)
        _feedForwardOut.wrappedValue = NFKPhi4MMAudioFeedForward(c)
        _attentionNorm.wrappedValue = NFKLayerNorm(dimensions: c.modelDimensions)
        _norm.wrappedValue = NFKLayerNorm(dimensions: c.modelDimensions)
    }

    func callAsFunction(_ x: MLXArray, bias: MLXArray, keyMask: MLXArray? = nil) -> MLXArray {
        var h = x + 0.5 * feedForwardIn(x)
        h = h + attention(attentionNorm(h), bias: bias, keyMask: keyMask)
        h = h + conv(h)
        h = h + 0.5 * feedForwardOut(h)
        return norm(h)
    }
}

/// The Conformer encoder: normalization, subsampling, then the stack of blocks under one T5 bias.
final class NFKPhi4MMConformerEncoder: Module {
    @ModuleInfo(key: "encoder_embedding") var norm: NFKPhi4MMAudioNorm
    @ModuleInfo(key: "embed") var subsample: NFKPhi4MMAudioSubsampling
    @ModuleInfo(key: "encoders") var layers: [NFKPhi4MMAudioLayer]
    @ModuleInfo(key: "relative_attention_bias_layer") var t5Bias: NFKPhi4MMT5Bias
    let timeReduction: Int
    let window: Int

    init(_ c: NFKMLXPhi4MMAudioConfiguration) {
        _norm.wrappedValue = NFKPhi4MMAudioNorm(c.mels)
        _subsample.wrappedValue = NFKPhi4MMAudioSubsampling(c)
        _layers.wrappedValue = (0 ..< c.layerCount).map { _ in NFKPhi4MMAudioLayer(c) }
        _t5Bias.wrappedValue = NFKPhi4MMT5Bias(c)
        timeReduction = c.timeReduction
        window = c.attentionWindow
    }

    /// `[clips, frames, mels]` → `[clips, ceil(frames/8), dModel]`. `validFrames` gives each clip's real
    /// mel frames when the clips are zero-padded to a common length; nil (one clip) masks nothing.
    func callAsFunction(_ mel: MLXArray, validFrames: [Int]? = nil) -> MLXArray {
        var h = subsample(norm(mel))
        let (clips, length, width) = (h.dim(0), h.dim(1), h.dim(2))
        var valid: MLXArray? = validFrames.map { frames in
            let lengths = MLXArray(frames.map { Int32(($0 + timeReduction - 1) / timeReduction) })
            return MLXArray(Int32(0) ..< Int32(length)).reshaped([1, length]) .< lengths.reshaped([clips, 1])
        }
        var span = length
        if length > window {
            let padding = (window - length % window) % window
            if padding > 0 {
                h = concatenated([h, MLXArray.zeros([clips, padding, width], dtype: h.dtype)], axis: 1)
                valid = valid.map { concatenated([$0, MLXArray.zeros([clips, padding], dtype: .bool)], axis: 1) }
            }
            h = h.reshaped([-1, window, width])
            valid = valid?.reshaped([-1, window])
            span = window
        }
        let bias = t5Bias(length: span)
        let keyMask = valid.map { $0.reshaped([$0.dim(0), 1, 1, span]) }
        for layer in layers { h = layer(h, bias: bias, keyMask: keyMask) }
        return h.reshaped([clips, -1, width])[0..., 0 ..< length, 0...]
    }
}

/// The audio projector's two heads, one per input mode, each an `nn.Sequential` of `Linear → GELU →
/// Linear` stored directly under `speech` and `vision` (the layers at numeric keys 0 and 2, a GELU
/// marker at 1). The head chosen at inference matches the active LoRA.
final class NFKPhi4MMAudioProjection: Module {
    @ModuleInfo(key: "speech") var speech: [Module]
    @ModuleInfo(key: "vision") var vision: [Module]

    init(_ c: NFKMLXPhi4MMAudioConfiguration) {
        func head() -> [Module] {
            [Linear(c.modelDimensions, c.projectionDimensions), Module(),
             Linear(c.projectionDimensions, c.projectionDimensions)]
        }
        _speech.wrappedValue = head()
        _vision.wrappedValue = head()
    }

    func callAsFunction(_ x: MLXArray, mode: NFKMLXPhi4MMModality) -> MLXArray {
        let head = mode == .vision ? vision : speech
        return (head[2] as! Linear)(NFKReferenceRounding.wide((head[0] as! Linear)(x)) { gelu($0) })
    }
}

/// The complete speech tower: the Conformer encoder and the two-headed projector, under the release's
/// `model.embed_tokens_extend.audio_embed.` subtree.
public final class NFKMLXPhi4MMAudioNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKPhi4MMConformerEncoder
    @ModuleInfo(key: "audio_projection") var projection: NFKPhi4MMAudioProjection

    let configuration: NFKMLXPhi4MMAudioConfiguration

    init(_ c: NFKMLXPhi4MMAudioConfiguration = .released) {
        configuration = c
        _encoder.wrappedValue = NFKPhi4MMConformerEncoder(c)
        _projection.wrappedValue = NFKPhi4MMAudioProjection(c)
        super.init()
    }

    /// The Conformer encoder output for a log-mel input `[1, frames, mels]`.
    func encode(_ mel: MLXArray) -> MLXArray { encoder(mel) }

    /// The projected audio embeddings `[1, frames/8, decoderHidden]`, ready to scatter into the decoder
    /// at the audio-token positions.
    func projected(_ mel: MLXArray, mode: NFKMLXPhi4MMModality) -> MLXArray {
        projection(encoder(mel), mode: mode)
    }

    /// The Conformer encoder output for several clips at once: each `[frames, mels]`, zero-padded to the
    /// longest and run as one masked batch, as the reference processor submits a request's clips. One
    /// clip runs unpadded and unmasked. `[clips, ceil(longest/8), dModel]`.
    func encode(clips mels: [MLXArray]) -> MLXArray {
        let frames = mels.map { $0.dim(0) }
        let longest = frames.max() ?? 0
        let padded = mels.map { mel in
            mel.dim(0) == longest ? mel
                : concatenated([mel, MLXArray.zeros([longest - mel.dim(0), mel.dim(1)], dtype: mel.dtype)], axis: 0)
        }
        return encoder(stacked(padded, axis: 0), validFrames: mels.count > 1 ? frames : nil)
    }

    /// Each clip's projected embeddings `[ceil(frames/8), decoderHidden]`, in order, from one batched
    /// encoder pass.
    func projected(clips mels: [MLXArray], mode: NFKMLXPhi4MMModality) -> [MLXArray] {
        guard !mels.isEmpty else { return [] }
        let projectedClips = projection(encode(clips: mels), mode: mode)
        return mels.enumerated().map { index, mel in
            projectedClips[index, 0 ..< NFKMLXPhi4MMAudioFeatures.tokenCount(frames: mel.dim(0)), 0...]
        }
    }
}
