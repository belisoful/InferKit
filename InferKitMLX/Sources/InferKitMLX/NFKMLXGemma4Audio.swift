//
//  NFKMLXGemma4Audio.swift
//  InferKitMLX
//
//  The Gemma 4 audio Conformer (`Gemma4AudioModel`), the audio tower of the tri-modal release and the
//  most involved of its towers: a 2-D convolutional subsampler, then Conformer layers — a macaron
//  feed-forward, a blocked relative-position attention (Transformer-XL rel-shift, a per-dimension
//  softplus scale, a logit softcap), a light depthwise convolution, a second macaron feed-forward, and
//  sandwich norms — and an output projection.
//
//  The blocked attention mask is a sliding-window structure the model builds; it is fed in from the
//  caller (or the reference, in the parity test) rather than reconstructed here, which is the remaining
//  integration on the consumer path.
//

import Foundation
import MLX
import MLXNN

/// The geometry of a Gemma 4 audio Conformer.
public struct NFKMLXGemma4AudioConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var convKernelSize: Int
    public var chunkSize: Int
    public var contextLeft: Int
    public var contextRight: Int
    public var residualWeight: Float
    public var gradientClipping: Float
    public var attentionLogitCap: Float
    public var rmsEpsilon: Float
    public var subsampleChannels: [Int]
    public var outputProjDims: Int
    /// Whether the projections clamp their inputs and outputs to the release's trained bounds.
    public var useClippedLinears: Bool

    public init(hiddenSize: Int = 1024, layerCount: Int = 12, headCount: Int = 8, convKernelSize: Int = 5,
                chunkSize: Int = 12, contextLeft: Int = 13, contextRight: Int = 0,
                residualWeight: Float = 0.5, gradientClipping: Float = 1e10,
                attentionLogitCap: Float = 50, rmsEpsilon: Float = 1e-6,
                subsampleChannels: [Int] = [128, 128], outputProjDims: Int = 1024,
                useClippedLinears: Bool = false) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.convKernelSize = convKernelSize
        self.chunkSize = chunkSize
        self.contextLeft = contextLeft
        self.contextRight = contextRight
        self.residualWeight = residualWeight
        self.gradientClipping = gradientClipping
        self.attentionLogitCap = attentionLogitCap
        self.rmsEpsilon = rmsEpsilon
        self.subsampleChannels = subsampleChannels
        self.outputProjDims = outputProjDims
        self.useClippedLinears = useClippedLinears
    }

    /// A tiny geometry, matching the `gemma4_audio` reference record.
    public static let tiny = NFKMLXGemma4AudioConfiguration(
        hiddenSize: 32, layerCount: 2, headCount: 4, convKernelSize: 3, chunkSize: 4,
        contextLeft: 3, contextRight: 0, residualWeight: 0.5, gradientClipping: 1e10,
        attentionLogitCap: 50, rmsEpsilon: 1e-6, subsampleChannels: [8, 16], outputProjDims: 48)

    var headDimensions: Int { hiddenSize / headCount }
    var maxPastHorizon: Int { contextLeft - 1 }
    var maxFutureHorizon: Int { contextRight }
    var contextSize: Int { chunkSize + maxPastHorizon + maxFutureHorizon }
}

/// A LayerNorm with a learned scale and NO bias — the normalization the subsampler's convolutions use
/// (`nn.LayerNorm(bias=False)`), over the last axis.
final class NFKAudioLayerNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        epsilon = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = (x - mean).square().mean(axis: -1, keepDims: true)
        return (x - mean) * rsqrt(variance + epsilon) * weight
    }
}

/// One subsampling stage: a stride-2 3×3 convolution, a channel LayerNorm, and a ReLU.
final class NFKGemma4AudioSubSampleLayer: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: NFKAudioLayerNorm

    init(inputChannels: Int, outputChannels: Int, eps: Float) {
        _conv.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: outputChannels,
                                    kernelSize: 3, stride: 2, padding: 1, bias: false)
        _norm.wrappedValue = NFKAudioLayerNorm(dimensions: outputChannels, eps: eps)
        super.init()
    }

    /// `x` is NHWC `[batch, time, frequency, channels]`; the norm is over the channel axis.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(norm(conv(x)))
    }
}

/// The subsampler: two stride-2 stages that quarter the time and frequency axes, then a linear
/// projection of the flattened frequency-and-channel features to the model width.
final class NFKGemma4AudioSubSample: Module {
    @ModuleInfo(key: "layer0") var layer0: NFKGemma4AudioSubSampleLayer
    @ModuleInfo(key: "layer1") var layer1: NFKGemma4AudioSubSampleLayer
    @ModuleInfo(key: "input_proj_linear") var inputProjection: Linear

    init(_ c: NFKMLXGemma4AudioConfiguration) {
        _layer0.wrappedValue = NFKGemma4AudioSubSampleLayer(inputChannels: 1,
            outputChannels: c.subsampleChannels[0], eps: c.rmsEpsilon)
        _layer1.wrappedValue = NFKGemma4AudioSubSampleLayer(inputChannels: c.subsampleChannels[0],
            outputChannels: c.subsampleChannels[1], eps: c.rmsEpsilon)
        let projectionInput = (c.subsampleChannels[0] / 4) * c.subsampleChannels[1]
        _inputProjection.wrappedValue = Linear(projectionInput, c.hiddenSize, bias: false)
        super.init()
    }

    /// `features` `[batch, time, frequency]` → `[batch, time/4, hidden]`.
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var hidden = features.expandedDimensions(axis: -1)   // NHWC with one channel
        hidden = layer0(hidden)
        hidden = layer1(hidden)
        let (batch, time) = (hidden.shape[0], hidden.shape[1])
        // NHWC already lays out (frequency, channels) as the trailing axes the reference flattens.
        return inputProjection(hidden.reshaped([batch, time, hidden.shape[2] * hidden.shape[3]]))
    }
}

/// A Conformer macaron feed-forward: a pre-normed two-layer MLP with a residual scaled by a fixed
/// weight. The clamps guard the residual against overflow and are identity in normal operation.
final class NFKGemma4AudioFeedForward: Module {
    @ModuleInfo(key: "ffw_layer_1") var layer1: NFKGemmaClippableLinear
    @ModuleInfo(key: "ffw_layer_2") var layer2: NFKGemmaClippableLinear
    @ModuleInfo(key: "pre_layer_norm") var preNorm: NFKGemmaNorm
    @ModuleInfo(key: "post_layer_norm") var postNorm: NFKGemmaNorm
    let residualWeight: Float
    let clampBound: Float

    init(_ c: NFKMLXGemma4AudioConfiguration) {
        _layer1.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, c.hiddenSize * 4, clipped: c.useClippedLinears)
        _layer2.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize * 4, c.hiddenSize, clipped: c.useClippedLinears)
        _preNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        residualWeight = c.residualWeight
        clampBound = c.gradientClipping
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = preNorm(clip(x, min: -clampBound, max: clampBound))
        h = layer2(silu(layer1(h)))
        h = postNorm(clip(h, min: -clampBound, max: clampBound)) * residualWeight
        return h + residual
    }
}

/// The Conformer light convolution: a gated linear unit, a causal depthwise convolution, and a
/// pointwise projection, each pre-normed.
final class NFKGemma4AudioLightConv: Module {
    @ModuleInfo(key: "linear_start") var linearStart: NFKGemmaClippableLinear
    @ModuleInfo(key: "linear_end") var linearEnd: NFKGemmaClippableLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwise: Conv1d
    @ModuleInfo(key: "pre_layer_norm") var preNorm: NFKGemmaNorm
    @ModuleInfo(key: "conv_norm") var convNorm: NFKGemmaNorm
    let leftPad: Int
    let clampBound: Float

    init(_ c: NFKMLXGemma4AudioConfiguration) {
        _linearStart.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, c.hiddenSize * 2, clipped: c.useClippedLinears)
        _linearEnd.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, c.hiddenSize, clipped: c.useClippedLinears)
        _depthwise.wrappedValue = Conv1d(inputChannels: c.hiddenSize, outputChannels: c.hiddenSize,
                                         kernelSize: c.convKernelSize, groups: c.hiddenSize, bias: false)
        _preNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _convNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        leftPad = c.convKernelSize - 1
        clampBound = c.gradientClipping
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = linearStart(preNorm(x))
        let half = h.shape[h.ndim - 1] / 2
        h = h[.ellipsis, 0 ..< half] * sigmoid(h[.ellipsis, half...])   // gated linear unit
        // A causal depthwise convolution: pad only the past, so a frame sees no future frame.
        let padded = padded(h, left: leftPad)
        h = depthwise(padded)
        h = convNorm(clip(h, min: -clampBound, max: clampBound))
        return linearEnd(silu(h)) + residual
    }

    /// Left-pads the time axis (axis 1) of an `[batch, time, channels]` tensor with zeros.
    private func padded(_ x: MLXArray, left: Int) -> MLXArray {
        let zeros = MLXArray.zeros([x.shape[0], left, x.shape[2]], dtype: x.dtype)
        return concatenated([zeros, x], axis: 1)
    }
}

/// The Conformer's blocked relative-position attention: queries are grouped into non-overlapping
/// blocks and each block attends over a fixed context window, with a Transformer-XL relative-position
/// term and a logit softcap. The blocked attention mask is supplied by the caller.
final class NFKGemma4AudioAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: NFKGemmaClippableLinear
    @ModuleInfo(key: "k_proj") var keyProjection: NFKGemmaClippableLinear
    @ModuleInfo(key: "v_proj") var valueProjection: NFKGemmaClippableLinear
    @ModuleInfo(key: "post") var post: NFKGemmaClippableLinear
    @ModuleInfo(key: "relative_k_proj") var relativeKeyProjection: Linear
    @ParameterInfo(key: "per_dim_scale") var perDimensionScale: MLXArray

    let headCount: Int
    let width: Int
    let chunkSize: Int
    let contextSize: Int
    let maxPastHorizon: Int
    let maxFutureHorizon: Int
    let softcap: Float
    let queryScale: Float
    let keyScale: Float

    init(_ c: NFKMLXGemma4AudioConfiguration) {
        headCount = c.headCount
        width = c.headDimensions
        chunkSize = c.chunkSize
        contextSize = c.contextSize
        maxPastHorizon = c.maxPastHorizon
        maxFutureHorizon = c.maxFutureHorizon
        softcap = c.attentionLogitCap
        queryScale = powf(Float(width), -0.5) / logf(2)
        keyScale = logf(1 + expf(1)) / logf(2)
        _queryProjection.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, headCount * width, clipped: c.useClippedLinears)
        _keyProjection.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, headCount * width, clipped: c.useClippedLinears)
        _valueProjection.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, headCount * width, clipped: c.useClippedLinears)
        _post.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, c.hiddenSize, clipped: c.useClippedLinears)
        _relativeKeyProjection.wrappedValue = Linear(c.hiddenSize, headCount * width, bias: false)
        _perDimensionScale.wrappedValue = MLXArray.zeros([width])
        super.init()
    }

    /// - Parameters:
    ///   - positionEmbeddings: the sinusoidal relative encoding `[1, positions, hidden]`.
    ///   - mask: the blocked validity mask `[batch, blocks, chunk, context]` (1 valid, 0 padding).
    func callAsFunction(_ x: MLXArray, positionEmbeddings: MLXArray, mask: MLXArray?) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        let scale = queryScale * softplus(perDimensionScale)
        let queriesFull = (queryProjection(x).reshaped([batch, length, headCount, width])) * scale
        let keysFull = (keyProjection(x).reshaped([batch, length, headCount, width])) * keyScale
        let valuesFull = valueProjection(x).reshaped([batch, length, headCount, width])

        let blockedQueries = convertToBlock(queriesFull)             // [B, blocks, chunk, heads, hd]
        let blockedKeys = extractBlockContext(keysFull)              // [B, blocks, context, heads, hd]
        let blockedValues = extractBlockContext(valuesFull)
        let blocks = blockedQueries.shape[1]

        let positions = relativeKeyProjection(positionEmbeddings).reshaped([-1, headCount, width])

        let queries = blockedQueries.transposed(0, 3, 1, 2, 4)       // [B, heads, blocks, chunk, hd]
        let keys = blockedKeys.transposed(0, 3, 1, 4, 2)             // [B, heads, blocks, hd, context]
        let contentScores = matmul(queries, keys)                    // [B, heads, blocks, chunk, context]

        let queriesFlat = queries.reshaped([batch, headCount, blocks * chunkSize, width])
        var relativeScores = matmul(queriesFlat, positions.transposed(1, 2, 0))  // [B, heads, blocks*chunk, P]
        relativeScores = relativeScores.reshaped([batch, headCount, blocks, chunkSize, positions.shape[0]])
        relativeScores = relativeShift(relativeScores)              // [B, heads, blocks, chunk, context]

        var scores = contentScores + relativeScores
        scores = tanh(scores / softcap) * softcap
        if let mask {
            let headMask = mask.expandedDimensions(axis: 1)          // [B, 1, blocks, chunk, context]
            scores = MLX.where(headMask .> 0, scores, MLXArray(Float(-1e9)))
        }
        let weights = softmax(scores, axis: -1, precise: true)
        let values = blockedValues.transposed(0, 3, 1, 2, 4)         // [B, heads, blocks, context, hd]
        var output = matmul(weights, values)                        // [B, heads, blocks, chunk, hd]
        output = output.transposed(0, 2, 3, 1, 4).reshaped([batch, blocks * chunkSize, headCount * width])
        return post(output[0..., 0 ..< length])
    }

    /// Splits `[batch, time, heads, hd]` into `[batch, blocks, chunk, heads, hd]`, padding the tail.
    private func convertToBlock(_ x: MLXArray) -> MLXArray {
        let (batch, time) = (x.shape[0], x.shape[1])
        let blocks = (time + chunkSize - 1) / chunkSize
        let pad = blocks * chunkSize - time
        var h = x
        if pad > 0 {
            let zeros = MLXArray.zeros([batch, pad, x.shape[2], x.shape[3]], dtype: x.dtype)
            h = concatenated([h, zeros], axis: 1)
        }
        return h.reshaped([batch, blocks, chunkSize, x.shape[2], x.shape[3]])
    }

    /// Extracts an overlapping `contextSize` window per block, padding past and future, via a gather.
    private func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let (batch, time, heads, hd) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let after = maxFutureHorizon + chunkSize - 1
        let leftZeros = MLXArray.zeros([batch, maxPastHorizon, heads, hd], dtype: x.dtype)
        let rightZeros = MLXArray.zeros([batch, after, heads, hd], dtype: x.dtype)
        let padded = concatenated([leftZeros, x, rightZeros], axis: 1)
        let blocks = (time + chunkSize - 1) / chunkSize
        var indices = [Int32]()
        for block in 0 ..< blocks {
            for offset in 0 ..< contextSize { indices.append(Int32(block * chunkSize + offset)) }
        }
        let gatherIndices = MLXArray(indices).reshaped([blocks, contextSize])
        return take(padded, gatherIndices, axis: 1)                 // [B, blocks, context, heads, hd]
    }

    /// The Transformer-XL relative shift: pad, reshape, drop, reshape, mapping `positions` distances to
    /// the `[chunk, context]` grid.
    private func relativeShift(_ x: MLXArray) -> MLXArray {
        let (batch, heads, blocks, chunk, positions) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        let padWidth = contextSize + 1 - positions
        var h = x
        if padWidth > 0 {
            let zeros = MLXArray.zeros([batch, heads, blocks, chunk, padWidth], dtype: x.dtype)
            h = concatenated([h, zeros], axis: -1)
        }
        h = h.reshaped([batch, heads, blocks, chunk * (contextSize + 1)])
        h = h[.ellipsis, 0 ..< (chunk * contextSize)]
        return h.reshaped([batch, heads, blocks, chunk, contextSize])
    }
}

/// `softplus(x) = log(1 + e^x)`, the per-dimension query scaling the audio attention learns.
private func softplus(_ x: MLXArray) -> MLXArray { log(1 + exp(x)) }

/// One Conformer layer: two macaron feed-forwards around a blocked attention and a light convolution,
/// with sandwich normalizations and overflow clamps.
final class NFKGemma4AudioLayer: Module {
    @ModuleInfo(key: "feed_forward1") var feedForward1: NFKGemma4AudioFeedForward
    @ModuleInfo(key: "feed_forward2") var feedForward2: NFKGemma4AudioFeedForward
    @ModuleInfo(key: "self_attn") var attention: NFKGemma4AudioAttention
    @ModuleInfo(key: "lconv1d") var lightConv: NFKGemma4AudioLightConv
    @ModuleInfo(key: "norm_pre_attn") var normPreAttention: NFKGemmaNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttention: NFKGemmaNorm
    @ModuleInfo(key: "norm_out") var normOut: NFKGemmaNorm
    let clampBound: Float

    init(_ c: NFKMLXGemma4AudioConfiguration) {
        _feedForward1.wrappedValue = NFKGemma4AudioFeedForward(c)
        _feedForward2.wrappedValue = NFKGemma4AudioFeedForward(c)
        _attention.wrappedValue = NFKGemma4AudioAttention(c)
        _lightConv.wrappedValue = NFKGemma4AudioLightConv(c)
        _normPreAttention.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _normPostAttention.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _normOut.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        clampBound = c.gradientClipping
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positionEmbeddings: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = feedForward1(x)
        let residual = h
        h = normPreAttention(clip(h, min: -clampBound, max: clampBound))
        h = attention(h, positionEmbeddings: positionEmbeddings, mask: mask)
        h = normPostAttention(clip(h, min: -clampBound, max: clampBound)) + residual
        h = lightConv(h)
        h = feedForward2(h)
        return normOut(clip(h, min: -clampBound, max: clampBound))
    }
}

/// The Gemma 4 audio Conformer: the convolutional subsampler, the Conformer layers, and the output
/// projection. The relative position encoding is parameter-free.
public final class NFKMLXGemma4AudioNet: Module {
    @ModuleInfo(key: "subsample_conv_projection") var subsample: NFKGemma4AudioSubSample
    @ModuleInfo(key: "layers") var layers: [NFKGemma4AudioLayer]
    @ModuleInfo(key: "output_proj") var outputProjection: Linear

    let configuration: NFKMLXGemma4AudioConfiguration

    init(_ c: NFKMLXGemma4AudioConfiguration) {
        configuration = c
        _subsample.wrappedValue = NFKGemma4AudioSubSample(c)
        _layers.wrappedValue = (0 ..< c.layerCount).map { _ in NFKGemma4AudioLayer(c) }
        _outputProjection.wrappedValue = Linear(c.hiddenSize, c.outputProjDims, bias: true)
        super.init()
    }

    /// The full tower: the convolutional subsampler, the sliding-window blocked mask, and the Conformer
    /// with its output projection. `features` is the mel front end's output `[batch, frames, features]`.
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        let hidden = subsample(features)
        let mask = blockedMask(sequenceLength: hidden.shape[1])
        return conformer(hidden, positionEmbeddings: relativePositionEncoding(), mask: mask)
    }

    /// Runs the Conformer over already-subsampled hidden states with a supplied position encoding and
    /// blocked mask. This is the seam the parity test drives, isolating the Conformer from the mask
    /// construction and the subsampler.
    public func conformer(_ hidden: MLXArray, positionEmbeddings: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = hidden
        for layer in layers { h = layer(h, positionEmbeddings: positionEmbeddings, mask: mask) }
        return outputProjection(h)
    }

    /// The sliding-window blocked validity mask `[1, blocks, chunk, context]` (1 valid, 0 masked): a
    /// query attends to a key within `[-maxPast, +maxFuture]` of it and inside the sequence. This is the
    /// reference's `create_sliding_window_mask` collapsed into the block layout the attention reads.
    func blockedMask(sequenceLength length: Int) -> MLXArray {
        let c = configuration
        let (chunk, context) = (c.chunkSize, c.contextSize)
        let (past, future) = (c.maxPastHorizon, c.maxFutureHorizon)
        let blocks = (length + chunk - 1) / chunk
        var values = [Float](); values.reserveCapacity(blocks * chunk * context)
        for block in 0 ..< blocks {
            for offset in 0 ..< chunk {
                let query = block * chunk + offset
                for position in 0 ..< context {
                    let key = block * chunk + position - past
                    let distance = query - key
                    // The window function: valid past is `dist < leftWindow` (STRICT), future is
                    // `-dist < rightWindow`; here leftWindow == maxPast and rightWindow == maxFuture.
                    let inWindow = (distance >= 0 && distance < past) || (distance < 0 && -distance < future)
                    let inRange = key >= 0 && key < length && query < length
                    values.append(inRange && inWindow ? 1 : 0)
                }
            }
        }
        return MLXArray(values).reshaped([1, blocks, chunk, context])
    }

    /// The parameter-free sinusoidal relative position encoding, `[1, context/2 + 1, hidden]`.
    public func relativePositionEncoding() -> MLXArray {
        let hidden = configuration.hiddenSize
        let count = hidden / 2
        let increment = logf(10000) / Float(max(count - 1, 1))
        let inverseTimescales = (0 ..< count).map { expf(Float($0) * -increment) }
        let positionCount = configuration.contextSize / 2 + 1
        var sines = [Float](), cosines = [Float]()
        for position in stride(from: configuration.contextSize / 2, through: 0, by: -1) {
            for scale in inverseTimescales {
                let angle = Float(position) * scale
                sines.append(sinf(angle)); cosines.append(cosf(angle))
            }
        }
        let sine = MLXArray(sines).reshaped([positionCount, count])
        let cosine = MLXArray(cosines).reshaped([positionCount, count])
        return concatenated([sine, cosine], axis: -1).expandedDimensions(axis: 0)
    }
}

/// Building and loading the audio Conformer.
extension NFKMLXGemmaLanguage {
    /// Builds a Gemma 4 audio subsampler (the convolutional front end).
    static func makeAudioSubSample(_ configuration: NFKMLXGemma4AudioConfiguration) -> NFKGemma4AudioSubSample {
        NFKGemma4AudioSubSample(configuration)
    }

    /// Builds a Gemma 4 audio Conformer network.
    static func makeAudioNet(_ configuration: NFKMLXGemma4AudioConfiguration) -> NFKMLXGemma4AudioNet {
        NFKMLXGemma4AudioNet(configuration)
    }

    /// Loads audio weights, transposing 2-D and 1-D convolution kernels from PyTorch to MLX layout.
    static func loadAudioWeights(_ arrays: [(String, MLXArray)], into module: Module) throws {
        let mapped = arrays.map { key, value -> (String, MLXArray) in
            if key.hasSuffix("conv.weight") && value.ndim == 4 {
                return (key, value.transposed(0, 2, 3, 1))       // [out,in,kH,kW] → [out,kH,kW,in]
            }
            if key.hasSuffix("depthwise_conv1d.weight") && value.ndim == 3 {
                return (key, value.transposed(0, 2, 1))          // [ch,1,k] → [ch,k,1]
            }
            return (key, value)
        }
        try NFKMLXWeights.apply(mapped, to: module)
    }
}
