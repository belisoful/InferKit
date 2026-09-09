//
//  NFKMLXGemma3nAudio.swift
//  InferKitMLX
//
//  The Gemma 3n audio encoder, a Universal Speech Model Conformer. It is a different network from the
//  Gemma 4 Conformer already here, not a configuration of it: the front end is a pair of strided 2-D
//  convolutions under a CUMULATIVE group normalization, the attention is chunked with a relative
//  position embedding shifted the Transformer-XL way, and every block clamps its activations.
//
//  Three things are load-bearing and none of them shows in a shape.
//
//  - **The group normalization is cumulative over time.** A frame is normalized by the statistics of
//    every frame up to and including itself, so the front end is causal in a way an ordinary group
//    norm is not.
//  - **The activation clamp runs at inference.** `gradient_clipping` reads like a training device and
//    is applied in the forward pass, six times per block.
//  - **The queries carry a learned per-dimension scale through a softplus**, on top of a fixed scale
//    that is the usual `1/sqrt(headDim)` divided by `softplus(0)`.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

/// The geometry of a Gemma 3n audio encoder.
public struct NFKMLXGemma3nAudioConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    /// The mel bands the front end reads.
    public var inputFeatureSize: Int
    /// The output channels of each front-end convolution.
    public var convChannels: [Int]
    public var convKernels: [(time: Int, frequency: Int)]
    public var convStrides: [(time: Int, frequency: Int)]
    public var convGroupNormEpsilon: Float
    /// The query block the chunked attention works in.
    public var attentionChunkSize: Int
    /// Frames of context to the left, INCLUDING the query's own, so the reach back is one less.
    public var attentionContextLeft: Int
    public var attentionContextRight: Int
    public var attentionLogitCap: Float
    public var convolutionKernelSize: Int
    /// The weight on the feed-forward's residual branch.
    public var residualWeight: Float
    /// Every activation is clamped to this magnitude, at inference as well as in training.
    public var gradientClipping: Float
    /// Frames kept at the end, one in every `reductionFactor`.
    public var reductionFactor: Int
    public var rmsEpsilon: Float

    public init(hiddenSize: Int = 1536, layerCount: Int = 12, headCount: Int = 8,
                inputFeatureSize: Int = 128, convChannels: [Int] = [128, 32],
                convKernels: [(time: Int, frequency: Int)] = [(3, 3), (3, 3)],
                convStrides: [(time: Int, frequency: Int)] = [(2, 2), (2, 2)],
                convGroupNormEpsilon: Float = 1e-3, attentionChunkSize: Int = 12,
                attentionContextLeft: Int = 13, attentionContextRight: Int = 0,
                attentionLogitCap: Float = 50, convolutionKernelSize: Int = 5,
                residualWeight: Float = 0.5, gradientClipping: Float = 1e10,
                reductionFactor: Int = 4, rmsEpsilon: Float = 1e-6) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.inputFeatureSize = inputFeatureSize
        self.convChannels = convChannels
        self.convKernels = convKernels
        self.convStrides = convStrides
        self.convGroupNormEpsilon = convGroupNormEpsilon
        self.attentionChunkSize = attentionChunkSize
        self.attentionContextLeft = attentionContextLeft
        self.attentionContextRight = attentionContextRight
        self.attentionLogitCap = attentionLogitCap
        self.convolutionKernelSize = convolutionKernelSize
        self.residualWeight = residualWeight
        self.gradientClipping = gradientClipping
        self.reductionFactor = reductionFactor
        self.rmsEpsilon = rmsEpsilon
    }

    var headDimensions: Int { hiddenSize / headCount }
    /// The frames the attention reaches back, which is one less than the stated left context.
    var maximumPast: Int { Swift.max(0, attentionContextLeft - 1) }
    var maximumFuture: Int { attentionContextRight }
    /// The keys one query block attends over.
    var contextSize: Int { attentionChunkSize + maximumPast + maximumFuture }

    /// The frequency width leaving each front-end convolution, which the group normalizations and the
    /// projection are sized against. The frequency axis is padded by one on each side.
    var convFrequencyWidths: [Int] {
        var widths = [Int]()
        var width = inputFeatureSize
        for index in convKernels.indices {
            width = (width + 2 - convKernels[index].frequency) / convStrides[index].frequency + 1
            widths.append(width)
        }
        return widths
    }

    /// The released `gemma-3n-E2B-it` / `E4B-it` audio encoder.
    public static let released = NFKMLXGemma3nAudioConfiguration()

    /// A small configuration carrying every mechanism.
    public static let tiny = NFKMLXGemma3nAudioConfiguration(
        hiddenSize: 32, layerCount: 2, headCount: 2, inputFeatureSize: 16,
        convChannels: [8, 4], attentionChunkSize: 4, attentionContextLeft: 3,
        attentionContextRight: 1, convolutionKernelSize: 3, gradientClipping: 10, reductionFactor: 2)
}

// MARK: - Cumulative group normalization

/// Group normalization taken cumulatively over time: a frame is normalized by the statistics of every
/// frame up to and including itself, over the frequency and channel axes together.
///
/// @discussion The variance is NOT the running variance it resembles. Each frame's squared deviations
/// are taken from THAT frame's cumulative mean and only then summed over time, which is what the
/// reference computes and is not the same number.
final class NFKGemma3nAudioCumulativeGroupNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float
    /// The elements one frame contributes: frequency width times channels.
    let elementsPerFrame: Float

    init(channels: Int, frequencyWidth: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.ones([channels])
        epsilon = eps
        elementsPerFrame = Float(frequencyWidth * channels)
        super.init()
    }

    /// `x` is `[batch, time, frequency, channels]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let wide = x.asType(.float32)
        let counts = MLXArray(Float(elementsPerFrame))

        let perFrame = wide.sum(axes: [2, 3], keepDims: true)
        let cumulativeCount = cumsum(broadcast(counts, to: perFrame.shape), axis: 1)
        let mean = cumsum(perFrame, axis: 1) / cumulativeCount

        let deviation = wide - mean
        let variance = cumsum((deviation * deviation).sum(axes: [2, 3], keepDims: true), axis: 1) / cumulativeCount
        return ((deviation * rsqrt(variance + epsilon)) * weight.asType(.float32)).asType(x.dtype)
    }
}

// MARK: - Front end

/// One front-end convolution: a strided 2-D convolution over the (time, frequency) plane, the
/// cumulative group normalization, and a ReLU.
///
/// The time axis is padded on the RIGHT only, by `kernel - 1`, which is JAX's reverse-causal padding.
final class NFKGemma3nAudioConvBlock: Module {
    @ModuleInfo(key: "conv") var convolution: Conv2d
    @ModuleInfo(key: "norm") var norm: NFKGemma3nAudioCumulativeGroupNorm

    let timePadding: Int

    init(_ c: NFKMLXGemma3nAudioConfiguration, index: Int) {
        let inputChannels = index == 0 ? 1 : c.convChannels[index - 1]
        let kernel = c.convKernels[index], stride = c.convStrides[index]
        timePadding = kernel.time - 1
        _convolution.wrappedValue = Conv2d(inputChannels: inputChannels,
                                           outputChannels: c.convChannels[index],
                                           kernelSize: IntOrPair((kernel.time, kernel.frequency)),
                                           stride: IntOrPair((stride.time, stride.frequency)),
                                           padding: IntOrPair((0, 0)), bias: false)
        _norm.wrappedValue = NFKGemma3nAudioCumulativeGroupNorm(
            channels: c.convChannels[index], frequencyWidth: c.convFrequencyWidths[index],
            eps: c.convGroupNormEpsilon)
        super.init()
    }

    /// `x` is `[batch, time, frequency, channels]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = padded(x, widths: [IntOrPair(0), IntOrPair((0, timePadding)), IntOrPair((1, 1)),
                                        IntOrPair(0)])
        return relu(norm(convolution(padded)))
    }
}

/// The two convolutions and the projection that turn a mel spectrogram into the Conformer's sequence.
final class NFKGemma3nAudioSubSample: Module {
    @ModuleInfo(key: "conv_0") var first: NFKGemma3nAudioConvBlock
    @ModuleInfo(key: "conv_1") var second: NFKGemma3nAudioConvBlock
    @ModuleInfo(key: "input_proj_linear") var projection: Linear

    init(_ c: NFKMLXGemma3nAudioConfiguration) {
        _first.wrappedValue = NFKGemma3nAudioConvBlock(c, index: 0)
        _second.wrappedValue = NFKGemma3nAudioConvBlock(c, index: 1)
        let width = c.convFrequencyWidths[c.convFrequencyWidths.count - 1] * c.convChannels[c.convChannels.count - 1]
        _projection.wrappedValue = Linear(width, c.hiddenSize, bias: false)
        super.init()
    }

    /// `mel` is `[batch, frames, bands]`; the result is `[batch, frames / 4, hidden]`.
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        let x = second(first(mel.expandedDimensions(axis: -1)))
        let (batch, time) = (x.shape[0], x.shape[1])
        // The frequency and channel axes flatten together, frequency first — the reference's own order.
        return projection(x.reshaped([batch, time, x.shape[2] * x.shape[3]]))
    }
}

// MARK: - Chunked attention

/// The relative position embedding the chunked attention adds to its content scores.
final class NFKGemma3nAudioRelativePosition: Module {
    @ModuleInfo(key: "pos_proj") var positionProjection: Linear

    let heads: Int
    let headDimensions: Int
    let channels: Int
    let maximumBackward: Int
    let maximumForward: Int
    /// `1 / timescale` per sinusoid. Held as a Swift array rather than an `MLXArray`, because a
    /// stored `MLXArray` on a `Module` enters `parameters()` and a strict load then reports it as a
    /// parameter the checkpoint does not cover.
    private let inverseTimescales: [Float]

    init(_ c: NFKMLXGemma3nAudioConfiguration) {
        heads = c.headCount
        headDimensions = c.headDimensions
        channels = c.hiddenSize
        maximumBackward = c.maximumPast
        maximumForward = c.maximumFuture
        _positionProjection.wrappedValue = Linear(c.hiddenSize, c.headCount * c.headDimensions, bias: false)

        let count = c.hiddenSize / 2
        let increment = Foundation.log(1.0e4) / Double(Swift.max(count - 1, 1))
        inverseTimescales = (0 ..< count).map { Float(Foundation.exp(Double($0) * -increment)) }
        super.init()
    }

    /// The scores for one pass: `queries` `[batch, blocks, chunk, heads, dim]` against `keys`
    /// `[batch, blocks, context, heads, dim]`, as `[batch, heads, blocks, chunk, context]`.
    func callAsFunction(queries: MLXArray, keys: MLXArray) -> MLXArray {
        let (batch, blocks, chunk) = (queries.shape[0], queries.shape[1], queries.shape[2])
        let context = keys.shape[2]

        // The spans run from the furthest key behind to the furthest ahead.
        let spans = MLXArray(stride(from: maximumBackward, through: -maximumForward, by: -1).map(Float.init))
            .reshaped([1, -1, 1])
        let scaled = spans * MLXArray(inverseTimescales).reshaped([1, 1, inverseTimescales.count])
        let timing = concatenated([sin(scaled), cos(scaled)], axis: -1).asType(queries.dtype)
        let span = timing.shape[1]
        let embedded = positionProjection(timing).reshaped([span, heads, headDimensions])

        let queriesByHead = queries.transposed(0, 3, 1, 2, 4)                       // [B, N, U, W, H]
        let content = matmul(queriesByHead, keys.transposed(0, 3, 1, 4, 2))         // [B, N, U, W, C]

        let flattened = queriesByHead.reshaped([batch, heads, blocks * chunk, headDimensions])
        let position = matmul(flattened, embedded.transposed(1, 2, 0))              // [B, N, U·W, F]
            .reshaped([batch, heads, blocks, chunk, span])

        return content + shifted(position, chunk: chunk, context: context, span: span)
    }

    /// The Transformer-XL relative shift: a pad, a reshape, a drop, and a reshape, which turns a
    /// score per SPAN into a score per key position.
    private func shifted(_ x: MLXArray, chunk: Int, context: Int, span: Int) -> MLXArray {
        let (batch, heads, blocks) = (x.shape[0], x.shape[1], x.shape[2])
        let padding = (context + 1) - span
        let padded = padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair(0), IntOrPair(0),
                                        IntOrPair((0, padding))])
        return padded.reshaped([batch, heads, blocks, chunk * (context + 1)])[0..., 0..., 0..., 0 ..< (chunk * context)]
            .reshaped([batch, heads, blocks, chunk, context])
    }
}

/// Chunked local attention: the queries are cut into blocks and each block attends over its own
/// frames plus a fixed reach behind and ahead.
final class NFKGemma3nAudioAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "relative_position_embedding") var relativePosition: NFKGemma3nAudioRelativePosition
    @ParameterInfo(key: "per_dim_scale") var perDimensionScale: MLXArray

    let heads: Int
    let headDimensions: Int
    let chunk: Int
    let maximumPast: Int
    let maximumFuture: Int
    let contextSize: Int
    let softcap: Float
    /// `1/sqrt(headDim)` divided by `softplus(0)`, which is what the reference folds into the queries.
    let queryScale: Float

    init(_ c: NFKMLXGemma3nAudioConfiguration) {
        heads = c.headCount
        headDimensions = c.headDimensions
        chunk = c.attentionChunkSize
        maximumPast = c.maximumPast
        maximumFuture = c.maximumFuture
        contextSize = c.contextSize
        softcap = c.attentionLogitCap
        queryScale = Float(pow(Double(c.headDimensions), -0.5) / Foundation.log(2.0))

        _queryProjection.wrappedValue = Linear(c.hiddenSize, c.headCount * c.headDimensions, bias: false)
        _keyProjection.wrappedValue = Linear(c.hiddenSize, c.headCount * c.headDimensions, bias: false)
        _valueProjection.wrappedValue = Linear(c.hiddenSize, c.headCount * c.headDimensions, bias: false)
        _relativePosition.wrappedValue = NFKGemma3nAudioRelativePosition(c)
        _perDimensionScale.wrappedValue = MLXArray.zeros([c.headDimensions])
        super.init()
    }

    /// Which keys a query in a block may read: causal within the reach behind, and no further ahead
    /// than the block's right context allows.
    private var localMask: MLXArray {
        let rows = MLXArray(Int32(0) ..< Int32(chunk)).reshaped([chunk, 1])
        let columns = MLXArray(Int32(0) ..< Int32(contextSize)).reshaped([1, contextSize])
        // A query at row `q` sits at context position `q + maximumPast`, so the frames it may read
        // run from `q` (its own reach back) to `q + maximumPast + maximumFuture` (its reach ahead).
        // Both bounds are relative to the ROW, not to the query's own context position.
        return (columns .>= rows) .&& (columns .<= (rows + Int32(maximumPast + maximumFuture)))
    }

    /// `x` is `[batch, time, hidden]`; `valid` marks each frame `[batch, time]`, or nil for all valid.
    /// The result is `[batch, time, heads, dim]`.
    func callAsFunction(_ x: MLXArray, valid: MLXArray?) -> MLXArray {
        let (batch, time) = (x.shape[0], x.shape[1])
        let shape = [batch, time, heads, headDimensions]

        let scale = softplus(perDimensionScale).reshaped([1, 1, 1, headDimensions])
        let queries = queryProjection(x).reshaped(shape) * queryScale * scale
        let keys = keyProjection(x).reshaped(shape)
        let values = valueProjection(x).reshaped(shape)

        let queryBlocks = NFKGemma3nAudioBlocking.blocks(queries, chunk: chunk)
        let keyBlocks = NFKGemma3nAudioBlocking.context(keys, chunk: chunk, past: maximumPast,
                                                        future: maximumFuture)
        let valueBlocks = NFKGemma3nAudioBlocking.context(values, chunk: chunk, past: maximumPast,
                                                          future: maximumFuture)
        let blocks = queryBlocks.shape[1]

        var logits = relativePosition(queries: queryBlocks, keys: keyBlocks)
        logits = tanh(logits / softcap) * softcap

        // The validity mask is never skipped, even when every frame carries audio: the context a block
        // reads is ZERO-PADDED at both ends, and the reference marks those padded frames invalid by
        // padding the mask itself with false. Attending to them instead changes every softmax in the
        // first and last blocks.
        let validity = (valid ?? (MLXArray.ones([batch, time]) .> Float(0))).asType(.int32)
        let validBlocks = NFKGemma3nAudioBlocking.context(validity, chunk: chunk, past: maximumPast,
                                                          future: maximumFuture) .== Int32(1)
        let admitted = localMask.reshaped([1, 1, 1, chunk, contextSize])
            .&& validBlocks.reshaped([batch, 1, blocks, 1, contextSize])
        logits = MLX.where(admitted, logits, MLXArray(Float(-3.4028235e38)))

        let probabilities = softmax(logits.asType(.float32), axis: -1).asType(values.dtype)
        // [B, N, U, W, C] against [B, U, C, N, H] → [B, U, W, N, H]
        let attended = matmul(probabilities.transposed(0, 2, 1, 3, 4),
                              valueBlocks.transposed(0, 1, 3, 2, 4))
        return attended.transposed(0, 1, 3, 2, 4)
            .reshaped([batch, blocks * chunk, heads, headDimensions])[0..., 0 ..< time]
    }
}

/// Cutting a sequence into query blocks and the wider key context each block reads.
enum NFKGemma3nAudioBlocking {
    /// `[batch, time, ...]` → `[batch, blocks, chunk, ...]`, the tail padded with zeros.
    static func blocks(_ x: MLXArray, chunk: Int) -> MLXArray {
        let (batch, time) = (x.shape[0], x.shape[1])
        let count = (time + chunk - 1) / chunk
        let padded = pad(x, left: 0, right: count * chunk - time)
        return padded.reshaped([batch, count, chunk] + Array(x.shape.dropFirst(2)))
    }

    /// `[batch, time, ...]` → `[batch, blocks, context, ...]`, each block's own frames with the reach
    /// behind and ahead around them. MLX has no `unfold`, so the frames are gathered.
    static func context(_ x: MLXArray, chunk: Int, past: Int, future: Int) -> MLXArray {
        let (batch, time) = (x.shape[0], x.shape[1])
        let count = (time + chunk - 1) / chunk
        let contextSize = chunk + past + future
        let padded = pad(x, left: past, right: future + chunk - 1 + (count * chunk - time))

        var indices = [Int32]()
        indices.reserveCapacity(count * contextSize)
        for block in 0 ..< count {
            for position in 0 ..< contextSize {
                indices.append(Int32(block * chunk + position))
            }
        }
        let gathered = padded.take(MLXArray(indices), axis: 1)
        return gathered.reshaped([batch, count, contextSize] + Array(x.shape.dropFirst(2)))
    }

    /// Zero frames added to the front and back of the time axis.
    private static func pad(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        guard left > 0 || right > 0 else { return x }
        var widths = [IntOrPair(0), IntOrPair((left, right))]
        widths.append(contentsOf: x.shape.dropFirst(2).map { _ in IntOrPair(0) })
        return padded(x, widths: widths)
    }
}

/// The numerically stable softplus, which MLX does not vend as a free function.
private func softplus(_ x: MLXArray) -> MLXArray {
    maximum(x, 0) + log1p(exp(-abs(x)))
}

// MARK: - Conformer

/// The Conformer's feed-forward branch: normalize, widen fourfold through a SiLU, narrow, normalize,
/// and add back at a fixed weight.
final class NFKGemma3nAudioFeedForward: Module {
    @ModuleInfo(key: "pre_layer_norm") var preNorm: NFKGemma3nNorm
    @ModuleInfo(key: "ffw_layer_1") var expand: Linear
    @ModuleInfo(key: "ffw_layer_2") var contract: Linear
    @ModuleInfo(key: "post_layer_norm") var postNorm: NFKGemma3nNorm

    let clip: Float
    let residualWeight: Float

    init(_ c: NFKMLXGemma3nAudioConfiguration) {
        clip = c.gradientClipping
        residualWeight = c.residualWeight
        _preNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _expand.wrappedValue = Linear(c.hiddenSize, c.hiddenSize * 4, bias: false)
        _contract.wrappedValue = Linear(c.hiddenSize * 4, c.hiddenSize, bias: false)
        _postNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let widened = contract(silu(expand(preNorm(clipped(x, clip)))))
        return x + postNorm(clipped(widened, clip)) * residualWeight
    }
}

/// The Conformer's attention branch: the chunked attention under a pre-normalization, projected back
/// and normalized before it rejoins the residual.
final class NFKGemma3nAudioConformerAttention: Module {
    @ModuleInfo(key: "pre_attn_norm") var preNorm: NFKGemma3nNorm
    @ModuleInfo(key: "attn") var attention: NFKGemma3nAudioAttention
    @ModuleInfo(key: "post") var post: Linear
    @ModuleInfo(key: "post_norm") var postNorm: NFKGemma3nNorm

    let clip: Float

    init(_ c: NFKMLXGemma3nAudioConfiguration) {
        clip = c.gradientClipping
        _preNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _attention.wrappedValue = NFKGemma3nAudioAttention(c)
        _post.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _postNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, valid: MLXArray?) -> MLXArray {
        let attended = attention(preNorm(clipped(x, clip)), valid: valid)
        let (batch, time) = (attended.shape[0], attended.shape[1])
        let flattened = attended.reshaped([batch, time, attended.shape[2] * attended.shape[3]])
        return x + postNorm(clipped(post(flattened), clip))
    }
}

/// The Conformer's convolution branch: a gated linear unit, a CAUSAL depthwise convolution, and a
/// projection back, added to the input.
final class NFKGemma3nAudioLightConvolution: Module {
    @ModuleInfo(key: "pre_layer_norm") var preNorm: NFKGemma3nNorm
    @ModuleInfo(key: "linear_start") var start: Linear
    @ModuleInfo(key: "depthwise_conv1d") var depthwise: Conv1d
    @ModuleInfo(key: "conv_norm") var convNorm: NFKGemma3nNorm
    @ModuleInfo(key: "linear_end") var end: Linear

    let clip: Float
    let causalPadding: Int

    init(_ c: NFKMLXGemma3nAudioConfiguration) {
        clip = c.gradientClipping
        causalPadding = c.convolutionKernelSize - 1
        _preNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _start.wrappedValue = Linear(c.hiddenSize, c.hiddenSize * 2, bias: false)
        _depthwise.wrappedValue = Conv1d(inputChannels: c.hiddenSize, outputChannels: c.hiddenSize,
                                         kernelSize: c.convolutionKernelSize, stride: 1, padding: 0,
                                         groups: c.hiddenSize, bias: false)
        _convNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _end.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let widened = start(preNorm(x))
        let half = widened.shape[2] / 2
        let gated = widened[0..., 0..., 0 ..< half] * sigmoid(widened[0..., 0..., half...])
        // The convolution sees no future: the padding is entirely on the left.
        let padded = padded(gated, widths: [IntOrPair(0), IntOrPair((causalPadding, 0)), IntOrPair(0)])
        return x + end(silu(convNorm(clipped(depthwise(padded), clip))))
    }
}

/// One Conformer block: a feed-forward, the attention, the convolution, a second feed-forward, and a
/// closing normalization.
final class NFKGemma3nAudioConformerBlock: Module {
    @ModuleInfo(key: "ffw_layer_start") var feedForwardStart: NFKGemma3nAudioFeedForward
    @ModuleInfo(key: "attention") var attention: NFKGemma3nAudioConformerAttention
    @ModuleInfo(key: "lconv1d") var convolution: NFKGemma3nAudioLightConvolution
    @ModuleInfo(key: "ffw_layer_end") var feedForwardEnd: NFKGemma3nAudioFeedForward
    @ModuleInfo(key: "norm") var norm: NFKGemma3nNorm

    let clip: Float

    init(_ c: NFKMLXGemma3nAudioConfiguration) {
        clip = c.gradientClipping
        _feedForwardStart.wrappedValue = NFKGemma3nAudioFeedForward(c)
        _attention.wrappedValue = NFKGemma3nAudioConformerAttention(c)
        _convolution.wrappedValue = NFKGemma3nAudioLightConvolution(c)
        _feedForwardEnd.wrappedValue = NFKGemma3nAudioFeedForward(c)
        _norm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, valid: MLXArray?) -> MLXArray {
        var hidden = attention(feedForwardStart(x), valid: valid)
        if let valid {
            // The invalid frames are zeroed BEFORE the convolution, so a padded frame contributes
            // nothing to its neighbors through the kernel.
            hidden = hidden * valid.expandedDimensions(axis: -1).asType(hidden.dtype)
        }
        hidden = feedForwardEnd(convolution(hidden))
        return norm(clipped(hidden, clip))
    }
}

// MARK: - The encoder

/// The Gemma 3n audio encoder: the convolutional front end, the Conformer stack, and the trailing
/// reduction that keeps one frame in every `reductionFactor`.
public final class NFKMLXGemma3nAudioNet: Module {
    @ModuleInfo(key: "subsample_conv_projection") var subsample: NFKGemma3nAudioSubSample
    @ModuleInfo(key: "conformer") var conformer: [NFKGemma3nAudioConformerBlock]

    public let configuration: NFKMLXGemma3nAudioConfiguration

    public init(_ c: NFKMLXGemma3nAudioConfiguration) {
        configuration = c
        _subsample.wrappedValue = NFKGemma3nAudioSubSample(c)
        _conformer.wrappedValue = (0 ..< c.layerCount).map { _ in NFKGemma3nAudioConformerBlock(c) }
        super.init()
    }

    /// Encodes a mel spectrogram.
    ///
    /// - Parameters:
    ///   - mel: `[batch, frames, bands]`.
    ///   - valid: `[batch, frames]` marking the frames that carry audio, or nil for all of them.
    /// - Returns: the encoded sequence and the validity mask at the encoder's own frame rate.
    public func callAsFunction(_ mel: MLXArray, valid: MLXArray? = nil)
        -> (encoded: MLXArray, valid: MLXArray?) {
        let c = configuration
        var hidden = subsample(mel)
        let frames = hidden.shape[1]

        // The frame mask is subsampled by taking the frame each output frame's receptive field opens
        // at, which is the reference's own rule rather than an `all` over the window.
        var current: MLXArray?
        if let valid {
            let stride = c.convStrides.reduce(1) { $0 * $1.time }
            let indices = (0 ..< frames).map { Int32(Swift.min($0 * stride, valid.shape[1] - 1)) }
            current = valid.take(MLXArray(indices), axis: 1)
        }

        for block in conformer {
            hidden = block(hidden, valid: current)
        }

        if c.reductionFactor > 1 {
            hidden = hidden[0..., .stride(by: c.reductionFactor)]
            current = current.map { $0[0..., .stride(by: c.reductionFactor)] }
        }
        if let current {
            hidden = hidden * current.expandedDimensions(axis: -1).asType(hidden.dtype)
        }
        return (hidden, current)
    }
}

/// The activation clamp Gemma 3n's audio blocks apply, at inference as well as in training.
private func clipped(_ x: MLXArray, _ bound: Float) -> MLXArray {
    MLX.clip(x, min: MLXArray(-bound), max: MLXArray(bound))
}

// MARK: - Building from a release

/// Building a Gemma 3n audio encoder, reading a release's configuration, and loading its weights.
@objc(NFKMLXGemma3nAudio)
public final class NFKMLXGemma3nAudio: NSObject {

    static func makeNet(_ configuration: NFKMLXGemma3nAudioConfiguration = .released) -> NFKMLXGemma3nAudioNet {
        NFKMLXGemma3nAudioNet(configuration)
    }

    /// Reads a released `config.json`, whose audio encoder sits under `audio_config`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXGemma3nAudioConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        return try configuration(fromJSON: json)
    }

    static func configuration(fromJSON json: [String: Any]) throws -> NFKMLXGemma3nAudioConfiguration {
        let audio = (json["audio_config"] as? [String: Any]) ?? json
        let kind = (audio["model_type"] as? String) ?? ""
        guard kind.isEmpty || kind == "gemma3n_audio" else {
            throw NFKMLXError.unsupportedConfiguration("this reads a gemma3n audio encoder, not \(kind)")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (audio[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (audio[key] as? NSNumber)?.floatValue ?? fallback }
        func pairs(_ key: String, _ fallback: [(time: Int, frequency: Int)]) -> [(time: Int, frequency: Int)] {
            guard let list = audio[key] as? [[NSNumber]] else { return fallback }
            return list.map { (time: $0[0].intValue, frequency: $0[1].intValue) }
        }

        return NFKMLXGemma3nAudioConfiguration(
            hiddenSize: integer("hidden_size", 1536),
            layerCount: integer("conf_num_hidden_layers", 12),
            headCount: integer("conf_num_attention_heads", 8),
            inputFeatureSize: integer("input_feat_size", 128),
            convChannels: (audio["sscp_conv_channel_size"] as? [NSNumber])?.map(\.intValue) ?? [128, 32],
            convKernels: pairs("sscp_conv_kernel_size", [(3, 3), (3, 3)]),
            convStrides: pairs("sscp_conv_stride_size", [(2, 2), (2, 2)]),
            convGroupNormEpsilon: real("sscp_conv_group_norm_eps", 1e-3),
            attentionChunkSize: integer("conf_attention_chunk_size", 12),
            attentionContextLeft: integer("conf_attention_context_left", 13),
            attentionContextRight: integer("conf_attention_context_right", 0),
            attentionLogitCap: real("conf_attention_logit_cap", 50),
            convolutionKernelSize: integer("conf_conv_kernel_size", 5),
            residualWeight: real("conf_residual_weight", 0.5),
            gradientClipping: real("gradient_clipping", 1e10),
            reductionFactor: integer("conf_reduction_factor", 4),
            rmsEpsilon: real("rms_norm_eps", 1e-6))
    }

    /// The encoder's module key for a checkpoint key, or nil for a tensor that is not the encoder's.
    static func encoderName(of key: String) -> String? {
        guard let name = stripped(key, prefixes: ["model.audio_tower.", "audio_tower."]) else { return nil }
        return name
    }

    /// `key` with the first matching prefix removed, or nil when none matches.
    static func stripped(_ key: String, prefixes: [String]) -> String? {
        for prefix in prefixes where key.hasPrefix(prefix) {
            return String(key.dropFirst(prefix.count))
        }
        return nil
    }

    /// A checkpoint tensor in MLX's layout: PyTorch stores a convolution's channels first, MLX last.
    ///
    /// @discussion The depthwise convolution is 3-D (`[channels, 1, kernel]`) and the front end's are
    /// 4-D; both take the same treatment, moving the input-channel axis to the end.
    static func converted(_ name: String, _ value: MLXArray) -> MLXArray {
        guard name.hasSuffix(".weight") else { return value }
        if value.ndim == 4, name.contains(".conv.") || name.hasSuffix("conv.weight") {
            return value.transposed(0, 2, 3, 1)
        }
        if value.ndim == 3, name.contains("depthwise_conv1d") {
            return value.transposed(0, 2, 1)
        }
        return value
    }

    /// Loads the encoder from a released directory, taking only the audio tower's tensors.
    static func loadWeights(into net: NFKMLXGemma3nAudioNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision,
                                                     remap: encoderName(of:))
        try NFKMLXWeights.apply(mapped.map { ($0.0, converted($0.0, $0.1)) }, to: net)
    }
}
