//
//  NFKMLXGraniteSpeech.swift
//  InferKitMLX
//
//  Granite Speech 3.3-2B (`GraniteSpeechForConditionalGeneration`, IBM, Apache-2.0): a speech
//  language model. A Conformer CTC encoder turns log-mel features into acoustic embeddings, a BLIP-2
//  Q-former projector windows them into a fixed number of query embeddings per window and projects
//  them to the decoder's width, and a dense Granite decoder generates text with the projected audio
//  embeddings scattered into the prompt at the audio-token positions.
//
//  Three components, built bottom-up and measured seam by seam:
//    - `NFKGraniteSpeechEncoder` — a Conformer with Shaw relative-position, block-local attention, a
//      macaron feed-forward pair, a GLU convolution module, and a mid-stack CTC skip.
//    - `NFKGraniteSpeechProjector` — a two-layer BLIP-2 Q-former (self-attention then cross-attention
//      to the acoustic embeddings, learned query embeddings, windowed) and a linear projection.
//    - the dense Granite decoder (RoPE grouped-query attention, SwiGLU, Granite's four scalar
//      multipliers), built here so the package need not run the standalone dense Granite LLM.
//
//  Reference: HF `transformers` `GraniteSpeechForConditionalGeneration` (transformers 4.57.6, llmvenv).
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

/// The Conformer encoder's geometry.
public struct NFKMLXGraniteSpeechEncoderConfiguration: Sendable {
    public var inputDim: Int
    public var hiddenDim: Int
    public var outputDim: Int
    public var layerCount: Int
    public var headCount: Int
    public var headDimensions: Int
    public var feedForwardMultiplier: Int
    public var convolutionExpansionFactor: Int
    public var convolutionKernel: Int
    public var contextSize: Int
    public var maxPositionEmbeddings: Int

    public init(inputDim: Int = 160, hiddenDim: Int = 1024, outputDim: Int = 256,
                layerCount: Int = 16, headCount: Int = 8, headDimensions: Int = 128,
                feedForwardMultiplier: Int = 4, convolutionExpansionFactor: Int = 2,
                convolutionKernel: Int = 15, contextSize: Int = 200, maxPositionEmbeddings: Int = 512) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.outputDim = outputDim
        self.layerCount = layerCount
        self.headCount = headCount
        self.headDimensions = headDimensions
        self.feedForwardMultiplier = feedForwardMultiplier
        self.convolutionExpansionFactor = convolutionExpansionFactor
        self.convolutionKernel = convolutionKernel
        self.contextSize = contextSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
    }
}

/// The BLIP-2 Q-former projector's geometry.
public struct NFKMLXGraniteSpeechProjectorConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var intermediateSize: Int
    public var encoderHiddenSize: Int
    public var layerNormEpsilon: Float
    /// A cross-attention block sits on every layer whose index is a multiple of this (BLIP-2's default
    /// is 2; the released Granite Speech projector sets 1, so every layer cross-attends).
    public var crossAttentionFrequency: Int
    public var windowSize: Int
    public var downsampleRate: Int

    /// One query per `downsampleRate` frames of a window.
    public var queryCount: Int { windowSize / downsampleRate }

    public init(hiddenSize: Int = 1024, layerCount: Int = 2, headCount: Int = 16,
                intermediateSize: Int = 4096, encoderHiddenSize: Int = 1024,
                layerNormEpsilon: Float = 1e-12, crossAttentionFrequency: Int = 2,
                windowSize: Int = 15, downsampleRate: Int = 5) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.intermediateSize = intermediateSize
        self.encoderHiddenSize = encoderHiddenSize
        self.layerNormEpsilon = layerNormEpsilon
        self.crossAttentionFrequency = crossAttentionFrequency
        self.windowSize = windowSize
        self.downsampleRate = downsampleRate
    }
}

// MARK: - Conformer encoder

/// A macaron feed-forward: `pre_norm → up → silu → down`, added back at half weight.
final class NFKGraniteSpeechConformerFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(_ config: NFKMLXGraniteSpeechEncoderConfiguration) {
        _preNorm.wrappedValue = NFKLayerNorm(dimensions: config.hiddenDim)
        _upProjection.wrappedValue = Linear(config.hiddenDim, config.hiddenDim * config.feedForwardMultiplier)
        _downProjection.wrappedValue = Linear(config.hiddenDim * config.feedForwardMultiplier, config.hiddenDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProjection(NFKReferenceRounding.silu(upProjection(preNorm(x))))
    }
}

/// Conformer attention with Shaw's relative-position bias, computed block-local over `contextSize`
/// frames per block. Each block attends within itself; the relative-position embedding adds a learned
/// bias per (query, key) offset.
final class NFKGraniteSpeechConformerAttention: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "to_q") var toQuery: Linear
    @ModuleInfo(key: "to_kv") var toKeyValue: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "rel_pos_emb") var relativePositionEmbedding: Embedding

    let headCount: Int
    let headDimensions: Int
    let contextSize: Int
    let maxPositionEmbeddings: Int
    let scale: Float

    init(_ config: NFKMLXGraniteSpeechEncoderConfiguration) {
        headCount = config.headCount
        headDimensions = config.headDimensions
        contextSize = config.contextSize
        maxPositionEmbeddings = config.maxPositionEmbeddings
        scale = 1.0 / Float(config.headDimensions).squareRoot()
        let inner = config.headCount * config.headDimensions
        _preNorm.wrappedValue = NFKLayerNorm(dimensions: config.hiddenDim)
        _toQuery.wrappedValue = Linear(config.hiddenDim, inner, bias: false)
        _toKeyValue.wrappedValue = Linear(config.hiddenDim, inner * 2, bias: false)
        _toOut.wrappedValue = Linear(inner, config.hiddenDim)
        _relativePositionEmbedding.wrappedValue = Embedding(
            embeddingCount: 2 * config.maxPositionEmbeddings + 1, dimensions: config.headDimensions)
        super.init()
    }

    /// The clamped relative-position indices for one block, `[contextSize, contextSize]`.
    private func attentionDistances() -> MLXArray {
        let seq = MLXArray(Int32(0) ..< Int32(contextSize))
        let dist = seq.reshaped([contextSize, 1]) - seq.reshaped([1, contextSize])
        let clamped = clip(dist, min: -contextSize, max: contextSize) + maxPositionEmbeddings
        return clamped
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let normed = preNorm(hidden)
        let batch = normed.dim(0), numFeatures = normed.dim(1)
        let numBlocks = (numFeatures + contextSize - 1) / contextSize
        let remainder = numFeatures % contextSize
        var x = normed
        if remainder > 0 {
            x = padded(x, widths: [.init((0, 0)), .init((0, contextSize - remainder)), .init((0, 0))])
        }
        let padded = numBlocks * contextSize

        func heads(_ projected: MLXArray) -> MLXArray {
            // [B, padded, inner] → [B, numBlocks, headCount, contextSize, headDim]
            projected.reshaped([batch, numBlocks, contextSize, headCount, headDimensions])
                .transposed(0, 1, 3, 2, 4)
        }
        let q = heads(toQuery(x))
        let kv = split(toKeyValue(x), parts: 2, axis: -1)
        let k = heads(kv[0]), v = heads(kv[1])

        let reduced = NFKReferenceRounding.isReduced(q)
        // Content scores: [B, blocks, heads, ctx, ctx]. In half precision the reference forms them inside
        // its MATH attention, so only the float32 path adds them here.
        let scores = reduced ? nil : matmul(q, k.swappedAxes(-1, -2)) as MLXArray?

        // Shaw relative-position bias: einsum("b m h c d, c r d -> b m h c r", q, relPos).
        let relPos = relativePositionEmbedding(attentionDistances())          // [ctx, ctx, headDim]
        // For each query position c: q[...,c,:] · relPos[c]ᵀ. Move c to front and batch-matmul.
        let qByContext = q.transposed(3, 0, 1, 2, 4)                          // [ctx, B, blocks, heads, headDim]
            .reshaped([contextSize, batch * numBlocks * headCount, headDimensions])
        let relByContext = relPos.swappedAxes(-1, -2)                         // [ctx, headDim, ctx]
        let posByContext = matmul(qByContext, relByContext)                  // [ctx, B*blocks*heads, ctx]
        let posBias = posByContext.reshaped([contextSize, batch, numBlocks, headCount, contextSize])
            .transposed(1, 2, 3, 0, 4)                                       // [B, blocks, heads, ctx, ctx]
        // The position bias: in half precision rounded and scaled on its own, as the reference's einsum and
        // its Python-float scale do; in float32 added to the content scores before one scale.
        var bias = reduced ? NFKReferenceRounding.scaled(posBias, by: scale) : (scores! + posBias) * scale

        if remainder > 0 {
            // Mask the padded query/key positions in the last block.
            var maskRow = [Float](repeating: 0, count: contextSize)
            for i in remainder ..< contextSize { maskRow[i] = 1 }
            let colMask = MLXArray(maskRow).reshaped([1, contextSize])
            let rowMask = MLXArray(maskRow).reshaped([contextSize, 1])
            let block = maximum(colMask, rowMask)                            // 1 where either index is padding
            // The reference fills with its type's most negative finite value, in that type; a float32
            // fill would promote a half-precision layer to float32 from here on.
            let maskValue = reduced ? -Float(bitPattern: 0x7F7F_0000) : -Float.greatestFiniteMagnitude
            // Apply only to the last block.
            var blocks = [MLXArray]()
            for b in 0 ..< numBlocks {
                let s = bias[0..., b]
                blocks.append(b == numBlocks - 1
                              ? MLX.where(block .> 0, MLXArray(maskValue).asType(s.dtype), s) : s)
            }
            bias = stacked(blocks, axis: 1)
        }

        var out: MLXArray                                                    // [B, blocks, heads, ctx, headDim]
        if reduced {
            out = NFKReferenceRounding.mathAttention(queries: q, keys: k, values: v, scale: scale, mask: bias)
        } else {
            out = matmul(softmax(bias, axis: -1, precise: true), v)
        }
        out = out.transposed(0, 1, 3, 2, 4).reshaped([batch, padded, headCount * headDimensions])
        return toOut(out[0..., 0 ..< numFeatures, 0...])
    }
}

/// The depthwise convolution, wrapped so the checkpoint's `depth_conv.conv.weight` routes into a real
/// `conv` submodule (a dotted `@ModuleInfo` key would flatten to the right name yet never load).
final class NFKGraniteSpeechDepthwiseConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, kernel: Int) {
        _conv.wrappedValue = NFKConv1d(inputChannels: channels, outputChannels: channels,
                                    kernelSize: kernel, groups: channels, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// The Conformer convolution module: `LayerNorm → pointwise up (×2) → GLU → depthwise → BatchNorm →
/// silu → pointwise down`. Operated channels-last (MLX `Conv1d` is NLC).
final class NFKGraniteSpeechConformerConvolution: Module, UnaryLayer {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "up_conv") var upConvolution: Conv1d
    @ModuleInfo(key: "depth_conv") var depthConvolution: NFKGraniteSpeechDepthwiseConv
    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm
    @ModuleInfo(key: "down_conv") var downConvolution: Conv1d
    let leftPad: Int
    let rightPad: Int

    init(_ config: NFKMLXGraniteSpeechEncoderConfiguration) {
        let inner = config.hiddenDim * config.convolutionExpansionFactor
        let kernel = config.convolutionKernel
        leftPad = kernel / 2
        rightPad = kernel / 2 - (kernel + 1) % 2
        _norm.wrappedValue = NFKLayerNorm(dimensions: config.hiddenDim)
        _upConvolution.wrappedValue = NFKConv1d(inputChannels: config.hiddenDim, outputChannels: inner * 2, kernelSize: 1)
        _depthConvolution.wrappedValue = NFKGraniteSpeechDepthwiseConv(channels: inner, kernel: kernel)
        _batchNorm.wrappedValue = NFKBatchNorm(featureCount: inner)
        _downConvolution.wrappedValue = NFKConv1d(inputChannels: inner, outputChannels: config.hiddenDim, kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm(x)                                                       // [B, L, hidden]
        h = upConvolution(h)                                                  // [B, L, 2·inner]
        let parts = split(h, parts: 2, axis: -1)
        h = NFKReferenceRounding.isReduced(h)                                // GLU over channels
            ? (parts[0].asType(.float32) * sigmoid(parts[1].asType(.float32))).asType(h.dtype)
            : parts[0] * sigmoid(parts[1])
        h = padded(h, widths: [.init((0, 0)), .init((leftPad, rightPad)), .init((0, 0))])
        h = depthConvolution(h)                                              // [B, L, inner]
        h = NFKReferenceRounding.silu(batchNorm(h))
        return downConvolution(h)                                            // [B, L, hidden]
    }
}

/// One Conformer block: macaron feed-forward, attention, convolution, macaron feed-forward, post-norm.
final class NFKGraniteSpeechConformerBlock: Module {
    @ModuleInfo(key: "ff1") var feedForward1: NFKGraniteSpeechConformerFeedForward
    @ModuleInfo(key: "attn") var attention: NFKGraniteSpeechConformerAttention
    @ModuleInfo(key: "conv") var convolution: NFKGraniteSpeechConformerConvolution
    @ModuleInfo(key: "ff2") var feedForward2: NFKGraniteSpeechConformerFeedForward
    @ModuleInfo(key: "post_norm") var postNorm: LayerNorm

    init(_ config: NFKMLXGraniteSpeechEncoderConfiguration) {
        _feedForward1.wrappedValue = NFKGraniteSpeechConformerFeedForward(config)
        _attention.wrappedValue = NFKGraniteSpeechConformerAttention(config)
        _convolution.wrappedValue = NFKGraniteSpeechConformerConvolution(config)
        _feedForward2.wrappedValue = NFKGraniteSpeechConformerFeedForward(config)
        _postNorm.wrappedValue = NFKLayerNorm(dimensions: config.hiddenDim)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var h = 0.5 * feedForward1(hidden) + hidden
        h = attention(h) + h
        h = convolution(h) + h
        h = 0.5 * feedForward2(h) + h
        return postNorm(h)
    }
}

/// The Conformer CTC encoder: an input projection, the block stack with a mid-stack CTC skip, and the
/// output projections. The mid skip adds `out_mid(softmax(out(h)))` at the half-way layer.
public final class NFKMLXGraniteSpeechEncoder: Module {
    let config: NFKMLXGraniteSpeechEncoderConfiguration

    @ModuleInfo(key: "input_linear") var inputLinear: Linear
    @ModuleInfo(key: "layers") var layers: [NFKGraniteSpeechConformerBlock]
    @ModuleInfo(key: "out") var out: Linear
    @ModuleInfo(key: "out_mid") var outMid: Linear

    public init(_ config: NFKMLXGraniteSpeechEncoderConfiguration) {
        self.config = config
        _inputLinear.wrappedValue = Linear(config.inputDim, config.hiddenDim)
        _layers.wrappedValue = (0 ..< config.layerCount).map { _ in NFKGraniteSpeechConformerBlock(config) }
        _out.wrappedValue = Linear(config.hiddenDim, config.outputDim)
        _outMid.wrappedValue = Linear(config.outputDim, config.hiddenDim)
        super.init()
    }

    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        var h = inputLinear(features)
        let half = config.layerCount / 2
        for (index, layer) in layers.enumerated() {
            h = layer(h)
            if index + 1 == half {
                h = h + outMid(softmax(out(h), axis: -1, precise: true))
            }
        }
        return h
    }
}

// MARK: - BLIP-2 Q-former projector

/// The Q-former's multi-head projections, `[query, key, value]` — a real submodule so the checkpoint's
/// `…attention.attention.query` routes in. Cross-attention keys and values come from the encoder.
final class NFKGraniteSpeechQFormerMHA: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    let headCount: Int
    let headDimensions: Int
    let scale: Float

    init(_ config: NFKMLXGraniteSpeechProjectorConfiguration, isCrossAttention: Bool) {
        headCount = config.headCount
        headDimensions = config.hiddenSize / config.headCount
        scale = 1.0 / Float(headDimensions).squareRoot()
        let keyValueInput = isCrossAttention ? config.encoderHiddenSize : config.hiddenSize
        _query.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _key.wrappedValue = Linear(keyValueInput, config.hiddenSize)
        _value.wrappedValue = Linear(keyValueInput, config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, context: MLXArray) -> MLXArray {
        let batch = hidden.dim(0), queryLength = hidden.dim(1), keyLength = context.dim(1)
        func heads(_ x: MLXArray, _ length: Int) -> MLXArray {
            x.reshaped([batch, length, headCount, headDimensions]).transposed(0, 2, 1, 3)
        }
        let q = heads(query(hidden), queryLength)
        let k = heads(key(context), keyLength)
        let v = heads(value(context), keyLength)
        // The reference divides by the root of the head width, a Python float.
        let scores = NFKReferenceRounding.isReduced(q)
            ? NFKReferenceRounding.divided(matmul(q, k.swappedAxes(-1, -2)), by: Float(headDimensions).squareRoot())
            : matmul(q, k.swappedAxes(-1, -2)) * scale
        let weights = softmax(scores, axis: -1, precise: true)
        return matmul(weights, v).transposed(0, 2, 1, 3).reshaped([batch, queryLength, headCount * headDimensions])
    }
}

/// The residual output of a Q-former attention: `dense → NFKLayerNorm(x + input)`.
final class NFKGraniteSpeechQFormerSelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ config: NFKMLXGraniteSpeechProjectorConfiguration) {
        _dense.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _norm.wrappedValue = NFKLayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, residual: MLXArray) -> MLXArray {
        norm(dense(hidden) + residual)
    }
}

/// A Q-former attention block: the multi-head projections and the residual output.
final class NFKGraniteSpeechQFormerAttention: Module {
    @ModuleInfo(key: "attention") var attention: NFKGraniteSpeechQFormerMHA
    @ModuleInfo(key: "output") var output: NFKGraniteSpeechQFormerSelfOutput

    init(_ config: NFKMLXGraniteSpeechProjectorConfiguration, isCrossAttention: Bool) {
        _attention.wrappedValue = NFKGraniteSpeechQFormerMHA(config, isCrossAttention: isCrossAttention)
        _output.wrappedValue = NFKGraniteSpeechQFormerSelfOutput(config)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, context: MLXArray) -> MLXArray {
        output(attention(hidden, context: context), residual: hidden)
    }
}

/// The gelu feed-forward's up projection.
final class NFKGraniteSpeechQFormerIntermediate: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(_ config: NFKMLXGraniteSpeechProjectorConfiguration) {
        _dense.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { NFKReferenceRounding.wide(dense(x)) { gelu($0) } }
}

/// The gelu feed-forward's residual down projection.
final class NFKGraniteSpeechQFormerOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ config: NFKMLXGraniteSpeechProjectorConfiguration) {
        _dense.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        _norm.wrappedValue = NFKLayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, residual: MLXArray) -> MLXArray {
        norm(dense(hidden) + residual)
    }
}

/// One Q-former layer: self-attention over the queries, cross-attention into the encoder embeddings,
/// and a gelu feed-forward, each residual through a LayerNorm. The query feed-forward path
/// (`intermediate_query` / `output_query`) is the one the audio-only projector uses.
final class NFKGraniteSpeechQFormerLayer: Module {
    @ModuleInfo(key: "attention") var selfAttention: NFKGraniteSpeechQFormerAttention
    @ModuleInfo(key: "crossattention") var crossAttention: NFKGraniteSpeechQFormerAttention?
    @ModuleInfo(key: "intermediate_query") var intermediateQuery: NFKGraniteSpeechQFormerIntermediate
    @ModuleInfo(key: "output_query") var outputQuery: NFKGraniteSpeechQFormerOutput

    init(_ config: NFKMLXGraniteSpeechProjectorConfiguration, hasCrossAttention: Bool) {
        _selfAttention.wrappedValue = NFKGraniteSpeechQFormerAttention(config, isCrossAttention: false)
        _crossAttention.wrappedValue = hasCrossAttention
            ? NFKGraniteSpeechQFormerAttention(config, isCrossAttention: true) : nil
        _intermediateQuery.wrappedValue = NFKGraniteSpeechQFormerIntermediate(config)
        _outputQuery.wrappedValue = NFKGraniteSpeechQFormerOutput(config)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, context: MLXArray) -> MLXArray {
        var h = selfAttention(hidden, context: hidden)
        if let crossAttention { h = crossAttention(h, context: context) }
        return outputQuery(intermediateQuery(h), residual: h)
    }
}

/// The Q-former encoder: the layer stack. A cross-attention block sits on every layer whose index is a
/// multiple of `crossAttentionFrequency`. A real submodule so `qformer.encoder.layer.N` routes in.
final class NFKGraniteSpeechQFormerEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [NFKGraniteSpeechQFormerLayer]

    init(_ config: NFKMLXGraniteSpeechProjectorConfiguration) {
        _layer.wrappedValue = (0 ..< config.layerCount).map {
            NFKGraniteSpeechQFormerLayer(config, hasCrossAttention: $0 % config.crossAttentionFrequency == 0)
        }
        super.init()
    }
}

/// The Q-former: the embedding LayerNorm and the encoder. A real submodule so `qformer.*` routes in.
final class NFKGraniteSpeechQFormer: Module {
    @ModuleInfo(key: "layernorm") var layerNorm: LayerNorm
    @ModuleInfo(key: "encoder") var encoder: NFKGraniteSpeechQFormerEncoder

    init(_ config: NFKMLXGraniteSpeechProjectorConfiguration) {
        _layerNorm.wrappedValue = NFKLayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        _encoder.wrappedValue = NFKGraniteSpeechQFormerEncoder(config)
        super.init()
    }

    func callAsFunction(_ queries: MLXArray, context: MLXArray) -> MLXArray {
        var h = layerNorm(queries)
        for layer in encoder.layer { h = layer(h, context: context) }
        return h
    }
}

/// The BLIP-2 Q-former projector: learned query embeddings cross-attend, per window, to the acoustic
/// embeddings; the result projects to the decoder's width. The encoder output is windowed into blocks
/// of `windowSize` frames, each block yielding `queryCount` query embeddings.
public final class NFKMLXGraniteSpeechProjector: Module {
    let config: NFKMLXGraniteSpeechProjectorConfiguration

    @ParameterInfo(key: "query") var query: MLXArray
    @ModuleInfo(key: "qformer") var qformer: NFKGraniteSpeechQFormer
    @ModuleInfo(key: "linear") var linear: Linear

    public init(_ config: NFKMLXGraniteSpeechProjectorConfiguration, textHiddenSize: Int) {
        self.config = config
        _query.wrappedValue = MLXArray.zeros([1, config.queryCount, config.hiddenSize])
        _qformer.wrappedValue = NFKGraniteSpeechQFormer(config)
        _linear.wrappedValue = Linear(config.hiddenSize, textHiddenSize)
        super.init()
    }

    public func callAsFunction(_ encoderEmbeddings: MLXArray) -> MLXArray {
        let batch = encoderEmbeddings.dim(0), sequence = encoderEmbeddings.dim(1)
        let dim = encoderEmbeddings.dim(2)
        let window = config.windowSize
        let blocks = (sequence + window - 1) / window
        let pad = blocks * window - sequence
        var context = encoderEmbeddings
        if pad > 0 {
            context = padded(context, widths: [.init((0, 0)), .init((0, pad)), .init((0, 0))])
        }
        context = context.reshaped([batch * blocks, window, dim])

        let queries = broadcast(query, to: [batch * blocks, config.queryCount, config.hiddenSize])
        let h = qformer(queries, context: context)
        return linear(h.reshaped([batch, blocks * config.queryCount, config.hiddenSize]))
    }
}

// MARK: - Dense Granite text decoder

/// The dense Granite decoder's geometry (a Llama-family decoder with Granite's four scalar multipliers).
public struct NFKMLXGraniteTextConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    public var ropeTheta: Float
    public var rmsEpsilon: Float
    public var embeddingMultiplier: Float
    public var residualMultiplier: Float
    public var attentionMultiplier: Float
    public var logitsScaling: Float
    public var tiesWordEmbeddings: Bool

    public init(hiddenSize: Int = 2048, layerCount: Int = 40, headCount: Int = 32,
                keyValueHeadCount: Int = 8, headDimensions: Int = 64, intermediateSize: Int = 8192,
                vocabularySize: Int = 49160, ropeTheta: Float = 10_000_000, rmsEpsilon: Float = 1e-5,
                embeddingMultiplier: Float = 12, residualMultiplier: Float = 0.22,
                attentionMultiplier: Float = 0.015625, logitsScaling: Float = 8,
                tiesWordEmbeddings: Bool = true) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.rmsEpsilon = rmsEpsilon
        self.embeddingMultiplier = embeddingMultiplier
        self.residualMultiplier = residualMultiplier
        self.attentionMultiplier = attentionMultiplier
        self.logitsScaling = logitsScaling
        self.tiesWordEmbeddings = tiesWordEmbeddings
    }
}

/// Grouped-query attention with rotary embeddings and Granite's `attention_multiplier` scale.
final class NFKGraniteTextAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    let headCount: Int
    let keyValueHeadCount: Int
    let headDimensions: Int
    let scale: Float
    let ropeBase: Float

    init(_ config: NFKMLXGraniteTextConfiguration) {
        headCount = config.headCount
        keyValueHeadCount = config.keyValueHeadCount
        headDimensions = config.headDimensions
        scale = config.attentionMultiplier
        ropeBase = config.ropeTheta
        _queryProjection.wrappedValue = Linear(config.hiddenSize, headCount * headDimensions, bias: false)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, keyValueHeadCount * headDimensions, bias: false)
        _valueProjection.wrappedValue = Linear(config.hiddenSize, keyValueHeadCount * headDimensions, bias: false)
        _outputProjection.wrappedValue = Linear(headCount * headDimensions, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let batch = hidden.dim(0), length = hidden.dim(1)
        var q = queryProjection(hidden).reshaped([batch, length, headCount, headDimensions]).transposed(0, 2, 1, 3)
        var k = keyProjection(hidden).reshaped([batch, length, keyValueHeadCount, headDimensions]).transposed(0, 2, 1, 3)
        let v = valueProjection(hidden).reshaped([batch, length, keyValueHeadCount, headDimensions]).transposed(0, 2, 1, 3)
        q = NFKReferenceRounding.rotary(q, dimensions: headDimensions, base: ropeBase, offset: 0)
        k = NFKReferenceRounding.rotary(k, dimensions: headDimensions, base: ropeBase, offset: 0)
        let out = NFKReferenceRounding.isReduced(q)
            ? NFKReferenceRounding.attention(queries: q, keys: k, values: v, scale: scale,
                                           mask: NFKMLXLanguageNet.causalMask(length, offset: 0))
            : MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .causal)
        return outputProjection(out.transposed(0, 2, 1, 3).reshaped([batch, length, headCount * headDimensions]))
    }
}

/// The SwiGLU feed-forward, `down(silu(gate(x)) · up(x))`.
final class NFKGraniteTextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: NFKMLXGraniteTextConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(NFKReferenceRounding.silu(gate(x)) * up(x)) }
}

/// One dense Granite block: attention and feed-forward, each added back through the residual multiplier.
final class NFKGraniteTextBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: NFKGraniteRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: NFKGraniteRMSNorm
    @ModuleInfo(key: "self_attn") var attention: NFKGraniteTextAttention
    @ModuleInfo(key: "mlp") var mlp: NFKGraniteTextMLP
    let residualMultiplier: Float

    init(_ config: NFKMLXGraniteTextConfiguration) {
        residualMultiplier = config.residualMultiplier
        _inputNorm.wrappedValue = NFKGraniteRMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
        _postNorm.wrappedValue = NFKGraniteRMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
        _attention.wrappedValue = NFKGraniteTextAttention(config)
        _mlp.wrappedValue = NFKGraniteTextMLP(config)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var h = hidden + NFKReferenceRounding.scaled(attention(inputNorm(hidden)), by: residualMultiplier)
        h = h + NFKReferenceRounding.scaled(mlp(postNorm(h)), by: residualMultiplier)
        return h
    }
}

/// The dense Granite backbone: token embedding, block stack, and final normalization.
final class NFKGraniteTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKGraniteTextBlock]
    @ModuleInfo(key: "norm") var norm: NFKGraniteRMSNorm

    init(_ config: NFKMLXGraniteTextConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.layerCount).map { _ in NFKGraniteTextBlock(config) }
        _norm.wrappedValue = NFKGraniteRMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
        super.init()
    }
}

/// The dense Granite decoder: the embedding scaled by `embedding_multiplier`, the block stack, the final
/// normalization, and the tied output projection scaled down by `logits_scaling`. Takes either token ids
/// or pre-fused input embeddings (the speech path scatters audio features before the multiplier applies).
public final class NFKMLXGraniteTextNet: Module {
    let config: NFKMLXGraniteTextConfiguration

    @ModuleInfo(key: "model") var model: NFKGraniteTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: NFKMLXGraniteTextConfiguration) {
        self.config = config
        _model.wrappedValue = NFKGraniteTextModel(config)
        _lmHead.wrappedValue = config.tiesWordEmbeddings ? nil : Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    /// Raw token embeddings, before the embedding multiplier (the point the speech path fuses audio in).
    func tokenEmbeddings(_ tokens: MLXArray) -> MLXArray { model.embedTokens(tokens) }

    func hiddenStates(fromEmbeddings embeddings: MLXArray) -> MLXArray {
        var h = embeddings * config.embeddingMultiplier
        for layer in model.layers { h = layer(h) }
        return model.norm(h)
    }

    func logits(fromHidden hidden: MLXArray) -> MLXArray {
        let raw = lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
        return raw / config.logitsScaling
    }

    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        logits(fromHidden: hiddenStates(fromEmbeddings: tokenEmbeddings(tokens)))
    }
}

// MARK: - The full Granite Speech model

/// Granite Speech end to end: the Conformer encoder and Q-former projector turn log-mel features into
/// audio embeddings, which scatter into the dense Granite decoder's prompt at the audio-token positions.
public final class NFKMLXGraniteSpeechNet: Module {
    public let encoderConfiguration: NFKMLXGraniteSpeechEncoderConfiguration
    public let projectorConfiguration: NFKMLXGraniteSpeechProjectorConfiguration
    public let textConfiguration: NFKMLXGraniteTextConfiguration
    public let audioTokenId: Int

    @ModuleInfo(key: "encoder") var encoder: NFKMLXGraniteSpeechEncoder
    @ModuleInfo(key: "projector") var projector: NFKMLXGraniteSpeechProjector
    @ModuleInfo(key: "language_model") var languageModel: NFKMLXGraniteTextNet

    public init(encoder: NFKMLXGraniteSpeechEncoderConfiguration,
                projector: NFKMLXGraniteSpeechProjectorConfiguration,
                text: NFKMLXGraniteTextConfiguration,
                audioTokenId: Int) {
        encoderConfiguration = encoder
        projectorConfiguration = projector
        textConfiguration = text
        self.audioTokenId = audioTokenId
        _encoder.wrappedValue = NFKMLXGraniteSpeechEncoder(encoder)
        _projector.wrappedValue = NFKMLXGraniteSpeechProjector(projector, textHiddenSize: text.hiddenSize)
        _languageModel.wrappedValue = NFKMLXGraniteTextNet(text)
        super.init()
    }

    /// The projected audio embeddings for a batch of log-mel features `[B, frames, inputDim]`.
    public func audioEmbeddings(_ features: MLXArray) -> MLXArray {
        projector(encoder(features))
    }

    /// Fuses audio embeddings into the token embeddings at the audio-token positions, then runs the
    /// decoder. `tokens` `[1, L]` carries `audioTokenId` at each position an audio frame fills; the
    /// flattened `audioEmbeddings` supply those positions in order.
    public func callAsFunction(_ tokens: MLXArray, features: MLXArray) -> MLXArray {
        logits(tokens: tokens, audioEmbeddings: audioEmbeddings(features))
    }

    /// The same fusion and decode, from already-computed audio embeddings — so a generation loop runs
    /// the encoder and projector once and only re-decodes the growing token sequence.
    public func logits(tokens: MLXArray, audioEmbeddings audio: MLXArray) -> MLXArray {
        // Audio positions embed token 0 (they are overwritten with audio features below); this also
        // keeps the audio token id, which may sit outside the base vocabulary, from indexing the table.
        let safeTokens = MLX.where(tokens .== MLXArray(Int32(audioTokenId)), MLXArray(Int32(0)), tokens)
        var embeddings = languageModel.tokenEmbeddings(safeTokens)          // raw, before the embedding multiplier
        let ids = tokens.asArray(Int32.self)
        let hidden = embeddings.dim(2)
        let flatAudio = audio.reshaped([-1, hidden])
        var audioIndex = 0
        var rows = [MLXArray]()
        rows.reserveCapacity(ids.count)
        for (position, id) in ids.enumerated() {
            if Int(id) == audioTokenId {
                rows.append(flatAudio[audioIndex].reshaped([1, 1, hidden]))
                audioIndex += 1
            } else {
                rows.append(embeddings[0..., position, 0...].reshaped([1, 1, hidden]))
            }
        }
        embeddings = concatenated(rows, axis: 1)
        return languageModel.logits(fromHidden: languageModel.hiddenStates(fromEmbeddings: embeddings))
    }
}

// MARK: - Builders and the released-weight loader

/// Builders and the released-weight loader for Granite Speech 3.3-2B.
@objc(NFKMLXGraniteSpeech)
public final class NFKMLXGraniteSpeech: NSObject {
    public static func makeNet(encoder: NFKMLXGraniteSpeechEncoderConfiguration,
                               projector: NFKMLXGraniteSpeechProjectorConfiguration,
                               text: NFKMLXGraniteTextConfiguration,
                               audioTokenId: Int) -> NFKMLXGraniteSpeechNet {
        NFKMLXGraniteSpeechNet(encoder: encoder, projector: projector, text: text, audioTokenId: audioTokenId)
    }

    /// Reads a Granite Speech geometry from a release directory's nested `config.json`.
    public static func net(fromDirectory directory: URL) throws -> NFKMLXGraniteSpeechNet {
        let url = directory.appendingPathComponent("config.json")
        let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] ?? [:]
        guard (json["model_type"] as? String) == "granite_speech" else {
            throw NFKMLXError.unsupportedConfiguration(
                "expected model_type granite_speech, found \(json["model_type"] as? String ?? "nil")")
        }
        func intIn(_ d: [String: Any], _ key: String, _ fallback: Int) -> Int { (d[key] as? Int) ?? fallback }
        func floatIn(_ d: [String: Any], _ key: String, _ fallback: Float) -> Float {
            if let v = d[key] as? Double { return Float(v) }
            if let v = d[key] as? Int { return Float(v) }
            return fallback
        }
        let ec = json["encoder_config"] as? [String: Any] ?? [:]
        let pc = json["projector_config"] as? [String: Any] ?? [:]
        let tc = json["text_config"] as? [String: Any] ?? [:]

        let encoder = NFKMLXGraniteSpeechEncoderConfiguration(
            inputDim: intIn(ec, "input_dim", 160), hiddenDim: intIn(ec, "hidden_dim", 1024),
            outputDim: intIn(ec, "output_dim", 256), layerCount: intIn(ec, "num_layers", 16),
            headCount: intIn(ec, "num_heads", 8), headDimensions: intIn(ec, "dim_head", 128),
            feedForwardMultiplier: intIn(ec, "feedforward_mult", 4),
            convolutionExpansionFactor: intIn(ec, "conv_expansion_factor", 2),
            convolutionKernel: intIn(ec, "conv_kernel_size", 15), contextSize: intIn(ec, "context_size", 200),
            maxPositionEmbeddings: intIn(ec, "max_pos_emb", 512))
        let projector = NFKMLXGraniteSpeechProjectorConfiguration(
            hiddenSize: intIn(pc, "hidden_size", 1024), layerCount: intIn(pc, "num_hidden_layers", 2),
            headCount: intIn(pc, "num_attention_heads", 16), intermediateSize: intIn(pc, "intermediate_size", 4096),
            encoderHiddenSize: intIn(pc, "encoder_hidden_size", 1024),
            layerNormEpsilon: floatIn(pc, "layer_norm_eps", 1e-12),
            crossAttentionFrequency: intIn(pc, "cross_attention_frequency", 2),
            windowSize: intIn(json, "window_size", 15), downsampleRate: intIn(json, "downsample_rate", 5))
        let heads = intIn(tc, "num_attention_heads", 32)
        let text = NFKMLXGraniteTextConfiguration(
            hiddenSize: intIn(tc, "hidden_size", 2048), layerCount: intIn(tc, "num_hidden_layers", 40),
            headCount: heads, keyValueHeadCount: intIn(tc, "num_key_value_heads", 8),
            headDimensions: (tc["head_dim"] as? Int) ?? (intIn(tc, "hidden_size", 2048) / heads),
            intermediateSize: intIn(tc, "intermediate_size", 8192), vocabularySize: intIn(tc, "vocab_size", 49160),
            ropeTheta: floatIn(tc, "rope_theta", 10_000_000), rmsEpsilon: floatIn(tc, "rms_norm_eps", 1e-5),
            embeddingMultiplier: floatIn(tc, "embedding_multiplier", 12),
            residualMultiplier: floatIn(tc, "residual_multiplier", 0.22),
            attentionMultiplier: floatIn(tc, "attention_multiplier", 0.015625),
            logitsScaling: floatIn(tc, "logits_scaling", 8),
            tiesWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? true)
        let audioTokenId = intIn(json, "audio_token_id", intIn(json, "audio_token_index", 49159))
        return NFKMLXGraniteSpeechNet(encoder: encoder, projector: projector, text: text, audioTokenId: audioTokenId)
    }

    /// Loads a released Granite Speech decoder from its directory. Conv1d weights are transposed from
    /// PyTorch's `[out, in, kernel]` to MLX's channels-last `[out, kernel, in]`, `num_batches_tracked`
    /// counters are dropped, a tied release ships no `lm_head`, and the BatchNorm running statistics are
    /// switched on with `train(false)`. The audio LoRA adapter (`adapter_model.safetensors`), which
    /// Granite Speech enables for audio inputs, is folded into the decoder's query and value projections
    /// when present, so the loaded weights are the audio-active model.
    public static func loadWeights(into net: NFKMLXGraniteSpeechNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .checkpoint,
                                   mergeAudioAdapter: Bool = true) throws {
        let tied = net.languageModel.lmHead == nil
        let read = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision) { key in
            if key.hasSuffix("num_batches_tracked") { return nil }
            if tied && key.hasPrefix("language_model.lm_head.") { return nil }
            return key
        }
        var base = Dictionary(uniqueKeysWithValues: read.map { name, value in
            // Conv1d weights are the only 3-D `.weight` tensors; move them to channels-last. The learned
            // query is also 3-D but ends `.query`, so gating on `.weight` leaves it untouched.
            (name, value.ndim == 3 && name.hasSuffix(".weight") ? value.transposed(0, 2, 1) : value)
        })
        let adapterURL = directory.appendingPathComponent("adapter_model.safetensors")
        if mergeAudioAdapter, FileManager.default.fileExists(atPath: adapterURL.path) {
            try foldAudioAdapter(into: &base, directory: directory, adapterURL: adapterURL)
        }
        try NFKMLXWeights.apply(Array(base), to: net)
        net.train(false)
    }

    /// Folds the peft LoRA adapter into the decoder's query and value projections: `W += (α/r)·B·A`.
    /// The adapter keys mirror the module's own decoder paths, so no renaming is needed.
    private static func foldAudioAdapter(into base: inout [String: MLXArray], directory: URL,
                                         adapterURL: URL) throws {
        var scale: Float = 1
        let configURL = directory.appendingPathComponent("adapter_config.json")
        if let data = try? Data(contentsOf: configURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let rank = (json["r"] as? Int) ?? 1
            let alpha = (json["lora_alpha"] as? Double).map(Float.init) ?? Float((json["lora_alpha"] as? Int) ?? rank)
            scale = alpha / Float(rank)
        }
        let adapter = try NFKMLXWeights.loadCheckpoint(url: adapterURL).arrays
        for (key, loraA) in adapter where key.hasSuffix(".lora_A.weight") {
            let stem = String(key.dropLast(".lora_A.weight".count))
            guard let loraB = adapter[stem + ".lora_B.weight"], let weight = base[stem + ".weight"] else { continue }
            let delta = matmul(loraB.asType(.float32), loraA.asType(.float32)) * scale
            base[stem + ".weight"] = (weight.asType(.float32) + delta).asType(weight.dtype)
        }
    }
}
