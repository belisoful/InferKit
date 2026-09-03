//
//  NFKMLXGemma4Vision.swift
//  InferKitMLX
//
//  The Gemma 4 vision encoder (`Gemma4VisionModel`), the image tower of the tri-modal release. It
//  embeds flattened image patches through one linear projection, adds a learned 2-D position embedding
//  (an x-table and a y-table summed per patch), and runs the same sandwich-normed block the text
//  decoder does — but bidirectional and with NO rotary, since the positions are the learned
//  embeddings. The projections are `Gemma4ClippableLinear`, a linear under a `.linear` key whose
//  clamps are identity by default.
//
//  Scope: the patch embedder and the transformer encoder, at reference parity. The pooler that reduces
//  the patch grid to soft tokens, the optional standardization, and the image processor that extracts
//  patches and their positions are the remaining integration.
//

import Foundation
import MLX
import MLXNN

/// The geometry of a Gemma 4 vision encoder.
public struct NFKMLXGemma4VisionConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int
    public var intermediateSize: Int
    public var patchSize: Int
    public var positionEmbeddingSize: Int
    public var poolingKernelSize: Int
    public var rmsEpsilon: Float
    /// The rotary base for the 2-D position rope the attention applies (`rope_theta`, 100 in the release).
    public var ropeTheta: Float
    /// Whether the projections clamp their inputs and outputs to the release's trained bounds.
    public var useClippedLinears: Bool

    public init(hiddenSize: Int = 768, layerCount: Int = 16, headCount: Int = 12,
                keyValueHeadCount: Int = 12, headDimensions: Int = 64, intermediateSize: Int = 3072,
                patchSize: Int = 16, positionEmbeddingSize: Int = 10240, poolingKernelSize: Int = 3,
                rmsEpsilon: Float = 1e-6, ropeTheta: Float = 100, useClippedLinears: Bool = false) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.positionEmbeddingSize = positionEmbeddingSize
        self.poolingKernelSize = poolingKernelSize
        self.rmsEpsilon = rmsEpsilon
        self.ropeTheta = ropeTheta
        self.useClippedLinears = useClippedLinears
    }

    /// A tiny geometry, matching the `gemma4_vision` reference record.
    public static let tiny = NFKMLXGemma4VisionConfiguration(
        hiddenSize: 32, layerCount: 2, headCount: 4, keyValueHeadCount: 4, headDimensions: 8,
        intermediateSize: 48, patchSize: 4, positionEmbeddingSize: 16, poolingKernelSize: 2)
}

/// A `Gemma4ClippableLinear`: a bias-free linear under a `.linear` key, optionally with input and
/// output clamps. A release trained with `use_clipped_linears` carries FINITE per-linear clamp bounds
/// (a quantization-aware-training artifact), and they are load-bearing at inference — ignoring them
/// cost the released vision tower a cosine of 0.88. When `clipped` is off the clamps are absent and the
/// linear is plain.
final class NFKGemmaClippableLinear: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ParameterInfo(key: "input_min") var inputMinimum: MLXArray?
    @ParameterInfo(key: "input_max") var inputMaximum: MLXArray?
    @ParameterInfo(key: "output_min") var outputMinimum: MLXArray?
    @ParameterInfo(key: "output_max") var outputMaximum: MLXArray?

    init(_ inputSize: Int, _ outputSize: Int, clipped: Bool = false) {
        _linear.wrappedValue = Linear(inputSize, outputSize, bias: false)
        if clipped {
            _inputMinimum.wrappedValue = MLXArray(-Float.greatestFiniteMagnitude)
            _inputMaximum.wrappedValue = MLXArray(Float.greatestFiniteMagnitude)
            _outputMinimum.wrappedValue = MLXArray(-Float.greatestFiniteMagnitude)
            _outputMaximum.wrappedValue = MLXArray(Float.greatestFiniteMagnitude)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x
        if let inputMinimum, let inputMaximum { input = clip(input, min: inputMinimum, max: inputMaximum) }
        var output = linear(input)
        if let outputMinimum, let outputMaximum { output = clip(output, min: outputMinimum, max: outputMaximum) }
        return output
    }
}

/// The vision encoder's attention: the same query/key/value norms and scale-1 attention as the text
/// decoder, but bidirectional and with no rotary. The projections are clippable linears.
final class NFKGemma4VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: NFKGemmaClippableLinear
    @ModuleInfo(key: "k_proj") var keyProjection: NFKGemmaClippableLinear
    @ModuleInfo(key: "v_proj") var valueProjection: NFKGemmaClippableLinear
    @ModuleInfo(key: "o_proj") var outputProjection: NFKGemmaClippableLinear
    @ModuleInfo(key: "q_norm") var queryNorm: NFKGemmaNorm
    @ModuleInfo(key: "k_norm") var keyNorm: NFKGemmaNorm

    let headCount: Int
    let keyValueHeadCount: Int
    let width: Int
    let valueEpsilon: Float

    init(_ c: NFKMLXGemma4VisionConfiguration) {
        headCount = c.headCount
        keyValueHeadCount = c.keyValueHeadCount
        width = c.headDimensions
        valueEpsilon = c.rmsEpsilon
        _queryProjection.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, c.headCount * width, clipped: c.useClippedLinears)
        _keyProjection.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, c.keyValueHeadCount * width, clipped: c.useClippedLinears)
        _valueProjection.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, c.keyValueHeadCount * width, clipped: c.useClippedLinears)
        _outputProjection.wrappedValue = NFKGemmaClippableLinear(c.headCount * width, c.hiddenSize, clipped: c.useClippedLinears)
        _queryNorm.wrappedValue = NFKGemmaNorm(dimensions: width, eps: c.rmsEpsilon)
        _keyNorm.wrappedValue = NFKGemmaNorm(dimensions: width, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        // The queries and keys carry a 2-D rope (one head half per spatial axis); the values do not.
        var queries = queryNorm(queryProjection(x).reshaped([batch, length, headCount, width]))
        queries = NFKGemma4VisionAttention.applyRope(queries, cosine: cosine, sine: sine).transposed(0, 2, 1, 3)
        var keys = keyNorm(keyProjection(x).reshaped([batch, length, keyValueHeadCount, width]))
        keys = NFKGemma4VisionAttention.applyRope(keys, cosine: cosine, sine: sine).transposed(0, 2, 1, 3)
        var v = valueProjection(x).reshaped([batch, length, keyValueHeadCount, width])
        v = v * rsqrt((v * v).mean(axis: -1, keepDims: true) + valueEpsilon)
        let values = v.transposed(0, 2, 1, 3)
        // Bidirectional (no mask), and the queries are already per-head normalized, so scale is 1.
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1, mask: nil)
        return outputProjection(attended.transposed(0, 2, 1, 3).reshaped([batch, length, headCount * width]))
    }

    /// Applies the 2-D rope to `x` `[batch, length, heads, headDim]`: the head is split into two blocks,
    /// each rotated (rotate-half) by the cosine and sine for its own spatial axis. `cosine`/`sine` are
    /// `[batch, length, headDim]`, the x-axis block followed by the y-axis block.
    static func applyRope(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let width = x.shape[x.ndim - 1]
        let block = width / 2
        var parts = [MLXArray]()
        for index in 0 ..< 2 {
            let range = (index * block) ..< ((index + 1) * block)
            let part = x[.ellipsis, range]
            let cosinePart = cosine[.ellipsis, range].expandedDimensions(axis: 2)
            let sinePart = sine[.ellipsis, range].expandedDimensions(axis: 2)
            parts.append(part * cosinePart + rotateHalf(part) * sinePart)
        }
        return concatenated(parts, axis: -1)
    }

    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.shape[x.ndim - 1] / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, 0 ..< half]], axis: -1)
    }
}

/// The vision feed-forward: a SwiGLU over clippable linears with Gemma's activation.
final class NFKGemma4VisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: NFKGemmaClippableLinear
    @ModuleInfo(key: "up_proj") var up: NFKGemmaClippableLinear
    @ModuleInfo(key: "down_proj") var down: NFKGemmaClippableLinear

    init(_ c: NFKMLXGemma4VisionConfiguration) {
        _gate.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, c.intermediateSize, clipped: c.useClippedLinears)
        _up.wrappedValue = NFKGemmaClippableLinear(c.hiddenSize, c.intermediateSize, clipped: c.useClippedLinears)
        _down.wrappedValue = NFKGemmaClippableLinear(c.intermediateSize, c.hiddenSize, clipped: c.useClippedLinears)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(geluApproximate(gate(x)) * up(x))
    }
}

/// One vision encoder layer: the sandwich block, bidirectional, with no per-layer scalar.
final class NFKGemma4VisionBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKGemma4VisionAttention
    @ModuleInfo(key: "mlp") var feedForward: NFKGemma4VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: NFKGemmaNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: NFKGemmaNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: NFKGemmaNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: NFKGemmaNorm

    init(_ c: NFKMLXGemma4VisionConfiguration) {
        _attention.wrappedValue = NFKGemma4VisionAttention(c)
        _feedForward.wrappedValue = NFKGemma4VisionMLP(c)
        _inputNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postAttentionNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _preFeedForwardNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postFeedForwardNorm.wrappedValue = NFKGemmaNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let attended = x + postAttentionNorm(attention(inputNorm(x), cosine: cosine, sine: sine))
        return attended + postFeedForwardNorm(feedForward(preFeedForwardNorm(attended)))
    }
}

/// The patch embedder: one linear projection over flattened patches plus a learned 2-D position
/// embedding. Gemma applies no normalization, scaling the pixels to `-1 … 1` in the model instead.
final class NFKGemma4VisionPatchEmbedder: Module {
    @ModuleInfo(key: "input_proj") var inputProjection: Linear
    @ParameterInfo(key: "position_embedding_table") var positionTable: MLXArray

    init(_ c: NFKMLXGemma4VisionConfiguration) {
        _inputProjection.wrappedValue = Linear(3 * c.patchSize * c.patchSize, c.hiddenSize, bias: false)
        _positionTable.wrappedValue = MLXArray.ones([2, c.positionEmbeddingSize, c.hiddenSize])
        super.init()
    }

    /// - Parameters:
    ///   - pixelValues: the flattened patches `[batch, patches, 3·patch²]` in `0 … 1`.
    ///   - positionIds: the `(x, y)` grid `[batch, patches, 2]`; a padding patch is `(-1, -1)`.
    func callAsFunction(_ pixelValues: MLXArray, positionIds: MLXArray) -> MLXArray {
        let hidden = inputProjection(2 * (pixelValues - 0.5))
        // A padding position is negative; it is clamped for the lookup and its embedding zeroed.
        let clamped = maximum(positionIds, MLXArray(Int32(0)))
        let xEmbedding = take(positionTable[0], clamped[.ellipsis, 0], axis: 0)
        let yEmbedding = take(positionTable[1], clamped[.ellipsis, 1], axis: 0)
        let padding = (positionIds[.ellipsis, 0] .< 0).expandedDimensions(axis: -1)
        let positions = MLX.where(padding, MLXArray(Float(0)), xEmbedding + yEmbedding)
        return hidden + positions
    }
}

/// The Gemma 4 vision encoder: patch embedder followed by the bidirectional transformer. It returns
/// the last hidden state, one vector per patch, before the pooler.
public final class NFKMLXGemma4VisionNet: Module {
    @ModuleInfo(key: "patch_embedder") var patchEmbedder: NFKGemma4VisionPatchEmbedder
    @ModuleInfo(key: "encoder_layers") var layers: [NFKGemma4VisionBlock]

    let configuration: NFKMLXGemma4VisionConfiguration

    init(_ c: NFKMLXGemma4VisionConfiguration) {
        configuration = c
        _patchEmbedder.wrappedValue = NFKGemma4VisionPatchEmbedder(c)
        _layers.wrappedValue = (0 ..< c.layerCount).map { _ in NFKGemma4VisionBlock(c) }
        super.init()
    }

    public func callAsFunction(_ pixelValues: MLXArray, positionIds: MLXArray) -> MLXArray {
        var hidden = patchEmbedder(pixelValues, positionIds: positionIds)
        let (cosine, sine) = rope(positionIds: positionIds)
        for layer in layers { hidden = layer(hidden, cosine: cosine, sine: sine) }
        return hidden
    }

    /// The 2-D rope tables `[batch, patches, headDim]`: the head is split per spatial axis, and each
    /// axis's positions drive a rotate-half rotary over its block, matching the reference's
    /// `Gemma4VisionRotaryEmbedding`.
    private func rope(positionIds: MLXArray) -> (cosine: MLXArray, sine: MLXArray) {
        let c = configuration
        let spatialDimension = c.headDimensions / 2
        let count = spatialDimension / 2
        let inverseFrequencies = (0 ..< count).map {
            1 / powf(c.ropeTheta, Float(2 * $0) / Float(spatialDimension))
        }
        let frequencyTable = MLXArray(inverseFrequencies).reshaped([1, 1, count])
        var cosines = [MLXArray](), sines = [MLXArray]()
        for axis in 0 ..< 2 {
            let positions = positionIds[.ellipsis, axis].asType(.float32).expandedDimensions(axis: -1)
            let angles = concatenated([positions * frequencyTable, positions * frequencyTable], axis: -1)
            cosines.append(cos(angles)); sines.append(sin(angles))
        }
        return (concatenated(cosines, axis: -1), concatenated(sines, axis: -1))
    }

    /// The full tower: the encoder, then the position-based average pooler and the `√hidden` scaling,
    /// producing the soft tokens a language model reads. Standardization (a released model can enable
    /// it) is not modeled; the released defaults leave it off.
    public func softTokens(_ pixelValues: MLXArray, positionIds: MLXArray) -> MLXArray {
        let encoded = self(pixelValues, positionIds: positionIds)
        let outputLength = pixelValues.shape[1] / (configuration.poolingKernelSize * configuration.poolingKernelSize)
        return pool(encoded, positionIds: positionIds, outputLength: outputLength)
    }

    /// Averages the patches falling into each `k × k` grid cell — where `k` is the ratio of the input
    /// patch count to the output token count — and scales by `√hidden`. The cell a patch belongs to is
    /// read from its `(x, y)` position, so a padded grid pools correctly.
    private func pool(_ hidden: MLXArray, positionIds: MLXArray, outputLength: Int) -> MLXArray {
        let (batch, sequence, width) = (hidden.shape[0], hidden.shape[1], hidden.shape[2])
        let k = Int((Double(sequence / outputLength)).squareRoot())
        let clamped = maximum(positionIds, MLXArray(Int32(0))).asType(.float32)
        let xs = clamped[.ellipsis, 0], ys = clamped[.ellipsis, 1]
        let columnsPerRow = floor((xs.max(axis: -1, keepDims: true) + 1) / Float(k))
        let cell = floor(xs / Float(k)) + columnsPerRow * floor(ys / Float(k))     // [batch, sequence]
        let range = MLXArray((0 ..< outputLength).map(Float.init)).reshaped([1, 1, outputLength])
        let oneHot = (cell.expandedDimensions(axis: -1) .== range).asType(hidden.dtype) / Float(k * k)
        let pooled = matmul(oneHot.transposed(0, 2, 1), hidden)                    // [batch, length, width]
        _ = batch
        return pooled * sqrt(Float(width))
    }
}

public extension NFKMLXGemmaLanguage {
    /// Builds a Gemma 4 vision encoder network.
    static func makeVisionNet(_ configuration: NFKMLXGemma4VisionConfiguration) -> NFKMLXGemma4VisionNet {
        NFKMLXGemma4VisionNet(configuration)
    }
}
