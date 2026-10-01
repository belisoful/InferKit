//
//  NFKMLXWav2Vec2Bert.swift
//  InferKitMLX
//
//  W2V-BERT 2.0 (`facebook/w2v-bert-2.0`, MIT), Meta's 600M-parameter self-supervised speech encoder
//  (the Seamless encoder), ported into MLXNN from transformers' `Wav2Vec2BertModel`. It reads 80-bin
//  Kaldi filterbanks stacked in pairs (SeamlessM4T's feature extractor) rather than the waveform, and its
//  24 Conformer layers attend with a clipped relative-key position table (64 frames left, 8 right).
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

/// The geometry of a W2V-BERT release, read from its `config.json`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWav2Vec2BertConfiguration: Sendable, Equatable {
    public var hiddenSize: Int = 1024
    public var numHiddenLayers: Int = 24
    public var numAttentionHeads: Int = 16
    public var intermediateSize: Int = 4096
    /// The stacked filterbank width (`feature_projection_input_dim`): 80 bins × a stride of 2.
    public var featureDimensions: Int = 160
    public var depthwiseKernel: Int = 31
    public var leftPositions: Int = 64
    public var rightPositions: Int = 8
    public var layerNormEps: Float = 1e-5
    /// The CTC head's vocabulary size, or nil for the pretraining release.
    public var vocabularySize: Int?
    public var padTokenID: Int = 0
    /// Whether the encoder ends in the output adapter (`add_adapter`): strided convolutional layers that
    /// halve the frame rate, which the CTC fine-tuning recipe adds.
    public var addsAdapter: Bool = false
    public var adapterLayers: Int = 1
    public var adapterKernel: Int = 3
    public var adapterStride: Int = 2
    /// The adapter's width (`output_hidden_size`); a width other than the encoder's adds a projection.
    public var outputHiddenSize: Int = 1024
    /// SpecAugment's time-mask probability and span in stacked frames, which fine-tuning applies.
    public var maskTimeProbability: Float = 0.05
    public var maskTimeLength: Int = 10
    public var maskTimeMinimumMasks: Int = 2

    /// `facebook/w2v-bert-2.0`.
    public static let v2 = NFKMLXWav2Vec2BertConfiguration()

    public init() {}

    /// Reads a Hugging Face `config.json` for a `wav2vec2-bert` model.
    public init(configurationURL: URL) throws {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: configurationURL)) as? [String: Any] ?? [:]
        guard (json["model_type"] as? String ?? "wav2vec2-bert") == "wav2vec2-bert" else {
            throw NFKMLXError.unsupportedConfiguration("\(configurationURL.lastPathComponent) is not a wav2vec2-bert model")
        }
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        hiddenSize = int("hidden_size", hiddenSize)
        numHiddenLayers = int("num_hidden_layers", numHiddenLayers)
        numAttentionHeads = int("num_attention_heads", numAttentionHeads)
        intermediateSize = int("intermediate_size", intermediateSize)
        featureDimensions = int("feature_projection_input_dim", featureDimensions)
        depthwiseKernel = int("conv_depthwise_kernel_size", depthwiseKernel)
        leftPositions = int("left_max_position_embeddings", leftPositions)
        rightPositions = int("right_max_position_embeddings", rightPositions)
        layerNormEps = Float((json["layer_norm_eps"] as? NSNumber)?.doubleValue ?? Double(layerNormEps))
        padTokenID = int("pad_token_id", padTokenID)
        if (json["architectures"] as? [String] ?? []).contains(where: { $0.hasSuffix("ForCTC") }) {
            vocabularySize = int("vocab_size", 0)
        }
        for (key, expected) in [("position_embeddings_type", "relative_key"), ("hidden_act", "swish")] {
            let value = json[key] as? String ?? expected
            guard value == expected else {
                throw NFKMLXError.unsupportedConfiguration("\(key) \(value) is not the \(expected) the release uses")
            }
        }
        addsAdapter = (json["add_adapter"] as? NSNumber)?.boolValue ?? false
        adapterLayers = int("num_adapter_layers", adapterLayers)
        adapterKernel = int("adapter_kernel_size", adapterKernel)
        adapterStride = int("adapter_stride", adapterStride)
        outputHiddenSize = int("output_hidden_size", hiddenSize)
        maskTimeProbability = Float((json["mask_time_prob"] as? NSNumber)?.doubleValue ?? Double(maskTimeProbability))
        maskTimeLength = int("mask_time_length", maskTimeLength)
        maskTimeMinimumMasks = int("mask_time_min_masks", maskTimeMinimumMasks)
        if (json["use_intermediate_ffn_before_adapter"] as? NSNumber)?.boolValue == true {
            throw NFKMLXError.unsupportedConfiguration("\(configurationURL.lastPathComponent) adds a feed-forward block "
                                                      + "before the adapter, which neither the release nor its recipe uses")
        }
        if addsAdapter, (json["adapter_act"] as? String ?? "relu") != "relu" {
            throw NFKMLXError.unsupportedConfiguration("the adapter activation is not the relu the recipe uses")
        }
    }
}

// MARK: - Feature extractor

/// SeamlessM4T's feature extractor, as W2V-BERT reads it: the waveform scaled to 16-bit range, 80-bin
/// Kaldi filterbanks (25 ms Povey frames every 10 ms, pre-emphasis 0.97), each bin normalized over the
/// utterance to zero mean and unit sample variance, padded to an even frame count with the release's
/// padding value, and consecutive frames stacked in pairs. A stacked frame whose second half is padding
/// is masked, as the extractor's attention mask marks it.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXWav2Vec2BertProcessor {
    public static let sampleRate = 16000

    /// The log-mel energies `[frames, 80]` before normalization, computed in double precision as the
    /// extractor's NumPy transform computes them.
    public static func filterbank(_ samples: [Float]) -> MLXArray {
        NFKKaldiFbank.logEnergiesInDoublePrecision(samples.map { $0 * 32768 })
    }

    /// The model input `[1, ⌈frames / 2⌉, 160]` from 16 kHz samples in `[-1, 1)`, and its mask `[1, ⌈frames / 2⌉]`
    /// (true where the frame is real).
    public static func inputFeatures(_ samples: [Float], stride: Int = 2,
                                     paddingValue: Float = 1) -> (features: MLXArray, mask: MLXArray) {
        let bank = filterbank(samples)
        let frames = bank.dim(0)
        let mean = bank.mean(axis: 0, keepDims: true)
        let variance = ((bank - mean) * (bank - mean)).sum(axis: 0, keepDims: true) / Float(max(frames - 1, 1))
        var normalized = (bank - mean) / sqrt(variance + 1e-7)
        let padding = (stride - frames % stride) % stride
        if padding > 0 {
            normalized = concatenated([normalized, MLXArray.full([padding, bank.dim(1)], values: MLXArray(paddingValue))], axis: 0)
        }
        let stacked = (frames + padding) / stride
        let mask = MLXArray((0 ..< stacked).map { $0 * stride + stride - 1 < frames }).reshaped([1, stacked])
        return (normalized.reshaped([1, stacked, bank.dim(1) * stride]), mask)
    }
}

// MARK: - Layers

final class NFKW2VBertFeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediate: Linear
    @ModuleInfo(key: "output_dense") var output: Linear
    let rectified: Bool

    /// A swish feed-forward of the encoder's width, or the adapter's relu one of `width`.
    init(_ c: NFKMLXWav2Vec2BertConfiguration, width: Int? = nil, rectified: Bool = false) {
        self.rectified = rectified
        _intermediate.wrappedValue = Linear(width ?? c.hiddenSize, c.intermediateSize)
        _output.wrappedValue = Linear(c.intermediateSize, width ?? c.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = intermediate(x)
        return output(rectified ? relu(hidden) : silu(hidden))
    }
}

/// The adapter's self-attention: `Wav2Vec2BertSelfAttention` with no position term.
final class NFKW2VBertPlainAttention: Module {
    @ModuleInfo(key: "linear_q") var q: Linear
    @ModuleInfo(key: "linear_k") var k: Linear
    @ModuleInfo(key: "linear_v") var v: Linear
    @ModuleInfo(key: "linear_out") var out: Linear
    let heads: Int

    init(width: Int, heads: Int) {
        self.heads = heads
        _q.wrappedValue = Linear(width, width)
        _k.wrappedValue = Linear(width, width)
        _v.wrappedValue = Linear(width, width)
        _out.wrappedValue = Linear(width, width)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        let d = width / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([batch, length, heads, d]).transposed(0, 2, 1, 3) }
        var scores = matmul(split(q(x)), split(k(x)).transposed(0, 1, 3, 2)) * (1 / Float(d).squareRoot())
        if let mask {
            scores = scores + MLX.where(mask.reshaped([batch, 1, 1, length]), MLXArray(Float(0)),
                                        MLXArray(-Float.greatestFiniteMagnitude)).asType(scores.dtype)
        }
        let attended = matmul(softmax(scores, axis: -1, precise: true), split(v(x)))
        return out(attended.transposed(0, 2, 1, 3).reshaped([batch, length, width]))
    }
}

/// One adapter layer: a strided convolution and GLU pool the residual and the attention input to half the
/// frames, then self-attention and a relu feed-forward, each with a pre-norm.
final class NFKW2VBertAdapterLayer: Module {
    @ModuleInfo(key: "residual_layer_norm") var residualNorm: NFKLayerNorm
    @ModuleInfo(key: "residual_conv") var residualConv: Conv1d
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: NFKLayerNorm
    @ModuleInfo(key: "self_attn_conv") var attentionConv: Conv1d
    @ModuleInfo(key: "self_attn") var attention: NFKW2VBertPlainAttention
    @ModuleInfo(key: "ffn_layer_norm") var ffnNorm: NFKLayerNorm
    @ModuleInfo(key: "ffn") var ffn: NFKW2VBertFeedForward
    let rates: NFKW2VBertDropoutRates

    init(_ c: NFKMLXWav2Vec2BertConfiguration, rates: NFKW2VBertDropoutRates = NFKW2VBertDropoutRates()) {
        self.rates = rates
        let width = c.outputHiddenSize
        _residualNorm.wrappedValue = NFKLayerNorm(dimensions: width, eps: c.layerNormEps)
        _residualConv.wrappedValue = Conv1d(inputChannels: width, outputChannels: 2 * width, kernelSize: c.adapterKernel,
                                            stride: c.adapterStride, padding: c.adapterStride / 2)
        _attentionNorm.wrappedValue = NFKLayerNorm(dimensions: width, eps: c.layerNormEps)
        _attentionConv.wrappedValue = Conv1d(inputChannels: width, outputChannels: 2 * width, kernelSize: c.adapterKernel,
                                             stride: c.adapterStride, padding: c.adapterStride / 2)
        _attention.wrappedValue = NFKW2VBertPlainAttention(width: width, heads: c.numAttentionHeads)
        _ffnNorm.wrappedValue = NFKLayerNorm(dimensions: width, eps: c.layerNormEps)
        _ffn.wrappedValue = NFKW2VBertFeedForward(c, width: width, rectified: true)
        super.init()
    }

    static func glu(_ x: MLXArray) -> MLXArray {
        let halves = x.split(parts: 2, axis: -1)
        return halves[0] * sigmoid(halves[1])
    }

    /// `mask` marks the real frames after this layer's pooling.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let residual = Self.glu(residualConv(residualNorm(x)))
        let pooled = Self.glu(attentionConv(attentionNorm(x)))
        let attended = NFKDropout.apply(attention(pooled, mask: mask), rate: rates.values.convolution, active: training)
            + residual
        return ffn(ffnNorm(attended)) + attended
    }
}

/// `Wav2Vec2BertAdapter`: an optional projection to the adapter's width, then the adapter layers.
final class NFKW2VBertAdapter: Module {
    @ModuleInfo(key: "proj") var projection: Linear?
    @ModuleInfo(key: "proj_layer_norm") var projectionNorm: NFKLayerNorm?
    @ModuleInfo(key: "layers") var layers: [NFKW2VBertAdapterLayer]
    let kernel: Int
    let stride: Int

    init(_ c: NFKMLXWav2Vec2BertConfiguration, rates: NFKW2VBertDropoutRates = NFKW2VBertDropoutRates()) {
        kernel = c.adapterKernel
        stride = c.adapterStride
        let projects = c.outputHiddenSize != c.hiddenSize
        _projection.wrappedValue = projects ? Linear(c.hiddenSize, c.outputHiddenSize) : nil
        _projectionNorm.wrappedValue = projects ? NFKLayerNorm(dimensions: c.outputHiddenSize, eps: c.layerNormEps) : nil
        _layers.wrappedValue = (0 ..< c.adapterLayers).map { _ in NFKW2VBertAdapterLayer(c, rates: rates) }
        super.init()
    }

    /// The frame count after one layer's pooling: `⌊(n + 2⌊kernel / 2⌋ − kernel) / stride⌋ + 1`.
    func pooledLength(_ length: Int) -> Int {
        (length + 2 * (kernel / 2) - kernel) / stride + 1
    }

    /// The adapter's output and, when `mask` is given, the mask of its real frames.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> (MLXArray, MLXArray?) {
        var hidden = x
        if let projection, let projectionNorm { hidden = projectionNorm(projection(hidden)) }
        var lengths = mask.map { $0.asType(.int32).sum(axis: -1).asArray(Int32.self).map(Int.init) }
        var pooledMask: MLXArray?
        for layer in layers {
            lengths = lengths?.map(pooledLength)
            let frames = pooledLength(hidden.dim(1))
            pooledMask = lengths.map { counts in
                MLX.stacked(counts.map { count in MLXArray((0 ..< frames).map { $0 < count }) })
            }
            hidden = layer(hidden, mask: pooledMask)
        }
        return (hidden, pooledMask ?? mask)
    }
}

/// Self-attention with the relative-key term: `q · E[clamp(j − i, −left, right)]` added to the scores,
/// both scaled by `1 / √d`.
final class NFKW2VBertAttention: Module {
    @ModuleInfo(key: "linear_q") var q: Linear
    @ModuleInfo(key: "linear_k") var k: Linear
    @ModuleInfo(key: "linear_v") var v: Linear
    @ModuleInfo(key: "linear_out") var out: Linear
    @ModuleInfo(key: "distance_embedding") var distance: Embedding
    let heads: Int
    let left: Int
    let right: Int

    init(_ c: NFKMLXWav2Vec2BertConfiguration) {
        heads = c.numAttentionHeads
        left = c.leftPositions
        right = c.rightPositions
        _q.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _k.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _v.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _out.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _distance.wrappedValue = Embedding(embeddingCount: c.leftPositions + c.rightPositions + 1,
                                           dimensions: c.hiddenSize / c.numAttentionHeads)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        let d = width / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([batch, length, heads, d]).transposed(0, 2, 1, 3) }
        let query = split(q(x)), key = split(k(x)), value = split(v(x))
        let scale = 1 / Float(d).squareRoot()
        let offsets = MLXArray(0 ..< length).reshaped([1, length]) - MLXArray(0 ..< length).reshaped([length, 1])
        let index = clip(offsets, min: -left, max: right) + left                         // [L, L]
        let relative = matmul(query, distance.weight.T)                                  // [B, H, L, positions]
        let bias = takeAlong(relative, broadcast(index.reshaped([1, 1, length, length]),
                                                 to: [batch, heads, length, length]), axis: -1)
        var scores = matmul(query, key.transposed(0, 1, 3, 2)) * scale + bias * scale
        if let mask {
            let keyMask = mask.reshaped([batch, 1, 1, length])
            scores = scores + MLX.where(keyMask, MLXArray(Float(0)), MLXArray(-Float.greatestFiniteMagnitude)).asType(scores.dtype)
        }
        let attended = matmul(softmax(scores, axis: -1, precise: true), value)
        return out(attended.transposed(0, 2, 1, 3).reshaped([batch, length, width]))
    }
}

/// The Conformer convolution block: a pointwise projection to twice the width, a GLU, a causal depthwise
/// convolution (padded on the left only), a layer norm, a swish, and a pointwise projection back.
final class NFKW2VBertConvolution: Module {
    @ModuleInfo(key: "layer_norm") var norm: NFKLayerNorm
    @ModuleInfo(key: "pointwise_conv1") var pointwise1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwise: Conv1d
    @ModuleInfo(key: "depthwise_layer_norm") var depthwiseNorm: NFKLayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwise2: Conv1d
    let kernel: Int
    let rates: NFKW2VBertDropoutRates

    init(_ c: NFKMLXWav2Vec2BertConfiguration, rates: NFKW2VBertDropoutRates = NFKW2VBertDropoutRates()) {
        kernel = c.depthwiseKernel
        self.rates = rates
        _norm.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _pointwise1.wrappedValue = Conv1d(inputChannels: c.hiddenSize, outputChannels: 2 * c.hiddenSize, kernelSize: 1, bias: false)
        _depthwise.wrappedValue = Conv1d(inputChannels: c.hiddenSize, outputChannels: c.hiddenSize, kernelSize: c.depthwiseKernel,
                                         groups: c.hiddenSize, bias: false)
        _depthwiseNorm.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _pointwise2.wrappedValue = Conv1d(inputChannels: c.hiddenSize, outputChannels: c.hiddenSize, kernelSize: 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var normed = norm(x)
        if let mask {
            normed = MLX.where(mask.expandedDimensions(axis: -1), normed, MLXArray(Float(0)).asType(normed.dtype))
        }
        let projected = pointwise1(normed)
        let halves = projected.split(parts: 2, axis: -1)
        let gated = halves[0] * sigmoid(halves[1])
        let padded = MLX.padded(gated, widths: [.init(0), .init((kernel - 1, 0)), .init(0)])
        return NFKDropout.apply(pointwise2(silu(depthwiseNorm(depthwise(padded)))), rate: rates.values.convolution,
                                active: training)
    }
}

/// One Conformer layer: half-step feed-forward, self-attention, convolution, half-step feed-forward, and
/// a final layer norm.
final class NFKW2VBertLayer: Module {
    @ModuleInfo(key: "ffn1_layer_norm") var ffn1Norm: NFKLayerNorm
    @ModuleInfo(key: "ffn1") var ffn1: NFKW2VBertFeedForward
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: NFKLayerNorm
    @ModuleInfo(key: "self_attn") var attention: NFKW2VBertAttention
    @ModuleInfo(key: "conv_module") var convolution: NFKW2VBertConvolution
    @ModuleInfo(key: "ffn2_layer_norm") var ffn2Norm: NFKLayerNorm
    @ModuleInfo(key: "ffn2") var ffn2: NFKW2VBertFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalNorm: NFKLayerNorm

    init(_ c: NFKMLXWav2Vec2BertConfiguration, rates: NFKW2VBertDropoutRates = NFKW2VBertDropoutRates()) {
        _ffn1Norm.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _ffn1.wrappedValue = NFKW2VBertFeedForward(c)
        _attentionNorm.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _attention.wrappedValue = NFKW2VBertAttention(c)
        _convolution.wrappedValue = NFKW2VBertConvolution(c, rates: rates)
        _ffn2Norm.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _ffn2.wrappedValue = NFKW2VBertFeedForward(c)
        _finalNorm.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + ffn1(ffn1Norm(x)) * 0.5
        h = h + attention(attentionNorm(h), mask: mask)
        h = h + convolution(h, mask: mask)
        h = h + ffn2(ffn2Norm(h)) * 0.5
        return finalNorm(h)
    }
}

final class NFKW2VBertEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKW2VBertLayer]

    init(_ c: NFKMLXWav2Vec2BertConfiguration, rates: NFKW2VBertDropoutRates = NFKW2VBertDropoutRates()) {
        _layers.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in NFKW2VBertLayer(c, rates: rates) }
        super.init()
    }
}

/// `Wav2Vec2BertFeatureProjection`: a layer norm over the stacked filterbanks, then a linear map.
final class NFKW2VBertFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var norm: NFKLayerNorm
    @ModuleInfo(key: "projection") var projection: Linear

    init(_ c: NFKMLXWav2Vec2BertConfiguration) {
        _norm.wrappedValue = NFKLayerNorm(dimensions: c.featureDimensions, eps: c.layerNormEps)
        _projection.wrappedValue = Linear(c.featureDimensions, c.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(norm(x)) }
}

// MARK: - Dropout

/// The dropouts Hugging Face's W2V-BERT fine-tuning recipe leaves at the release's rates; it sets every
/// other dropout, layer drop, and the time mask to zero.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWav2Vec2BertDropout: Sendable, Equatable {
    /// `conformer_conv_dropout`: the end of each Conformer convolution module and each adapter layer's
    /// attention output.
    public var convolution: Float
    /// `final_dropout`: the features the CTC head reads.
    public var final: Float

    public init(convolution: Float = 0, final: Float = 0) {
        self.convolution = convolution
        self.final = final
    }

    /// No dropout.
    public static let none = NFKMLXWav2Vec2BertDropout()

    /// The two rates from a release's `config.json`; 0.1 each in `facebook/w2v-bert-2.0`.
    public init(configurationURL url: URL) throws {
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("config.json is not a JSON object")
        }
        self.init(convolution: (json["conformer_conv_dropout"] as? NSNumber)?.floatValue ?? 0,
                  final: (json["final_dropout"] as? NSNumber)?.floatValue ?? 0)
    }
}

/// The rates every module of one network reads when it runs, so setting them once reaches them all.
final class NFKW2VBertDropoutRates {
    var values = NFKMLXWav2Vec2BertDropout.none
}

// MARK: - Network

/// The W2V-BERT network: stacked filterbanks `[B, frames, 160]` in, contextual features
/// `[B, frames, 1024]` out (one frame per 20 ms). Module keys mirror the transformers checkpoint.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXWav2Vec2BertNet: Module {
    @ModuleInfo(key: "feature_projection") var featureProjection: NFKW2VBertFeatureProjection
    @ModuleInfo(key: "encoder") var encoder: NFKW2VBertEncoder
    @ParameterInfo(key: "masked_spec_embed") var maskedSpecEmbed: MLXArray
    @ModuleInfo(key: "adapter") var adapter: NFKW2VBertAdapter?
    @ModuleInfo(key: "lm_head") var head: Linear?

    public let configuration: NFKMLXWav2Vec2BertConfiguration
    let rates = NFKW2VBertDropoutRates()

    /// The dropout while the network trains; none by default. Set it to
    /// `NFKMLXWav2Vec2BertDropout(configurationURL:)` to train at a release's rates, which Hugging Face's
    /// fine-tuning recipe leaves on.
    ///
    /// Introduced in InferKit 0.4.0.
    public var dropout: NFKMLXWav2Vec2BertDropout {
        get { rates.values }
        set { rates.values = newValue }
    }

    public init(_ configuration: NFKMLXWav2Vec2BertConfiguration = .v2) {
        self.configuration = configuration
        _featureProjection.wrappedValue = NFKW2VBertFeatureProjection(configuration)
        _encoder.wrappedValue = NFKW2VBertEncoder(configuration, rates: rates)
        _maskedSpecEmbed.wrappedValue = MLXArray.zeros([configuration.hiddenSize])
        _adapter.wrappedValue = configuration.addsAdapter ? NFKW2VBertAdapter(configuration, rates: rates) : nil
        let headWidth = configuration.addsAdapter ? configuration.outputHiddenSize : configuration.hiddenSize
        _head.wrappedValue = configuration.vocabularySize.map { Linear(headWidth, $0) }
        super.init()
        train(false)
    }

    /// The encoder's features, through the adapter when the network carries one, and the mask of their
    /// real frames. `timeMask` (`[B, frames]`, boolean) replaces SpecAugment's masked frames with
    /// `masked_spec_embed` after the feature projection. While the network trains, `dropout.final`
    /// applies to the features, which is where `Wav2Vec2BertForCTC` applies it before its head.
    public func encode(_ features: MLXArray, mask: MLXArray?, timeMask: MLXArray? = nil) -> (hidden: MLXArray, mask: MLXArray?) {
        var projected = featureProjection(features)
        if let timeMask {
            projected = MLX.where(timeMask.expandedDimensions(axis: -1), maskedSpecEmbed.asType(projected.dtype), projected)
        }
        let output = run(projected, mask: mask, collecting: false).output
        let (hidden, hiddenMask) = adapter.map { $0(output, mask: mask) } ?? (output, mask)
        return (NFKDropout.apply(hidden, rate: rates.values.final, active: training), hiddenMask)
    }

    /// The CTC logits and the real frame count of each utterance, or nil without a head.
    public func logits(_ features: MLXArray, mask: MLXArray?) -> (logits: MLXArray, frames: [Int])? {
        guard let head else { return nil }
        let (hidden, outputMask) = encode(features, mask: mask)
        let frames = outputMask.map { $0.asType(.int32).sum(axis: -1).asArray(Int32.self).map(Int.init) }
            ?? [Int](repeating: hidden.dim(1), count: hidden.dim(0))
        return (head(hidden), frames)
    }

    public convenience init(configurationURL: URL) throws {
        self.init(try NFKMLXWav2Vec2BertConfiguration(configurationURL: configurationURL))
    }

    /// The last hidden state from stacked filterbanks and their frame mask (nil when every frame is real).
    public func callAsFunction(_ features: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        seams(features, mask: mask, collecting: false).output
    }

    /// The projected features, each layer's output when `collecting`, and the last hidden state. Masked
    /// frames are zeroed before the encoder, excluded as attention keys, and zeroed before each depthwise
    /// convolution, as `Wav2Vec2BertEncoder` treats them.
    func seams(_ features: MLXArray, mask: MLXArray? = nil, collecting: Bool = true)
        -> (projected: MLXArray, layers: [MLXArray], output: MLXArray) {
        let projected = featureProjection(features)
        let (layers, output) = run(projected, mask: mask, collecting: collecting)
        return (projected, layers, output)
    }

    private func run(_ projected: MLXArray, mask: MLXArray?, collecting: Bool) -> (layers: [MLXArray], output: MLXArray) {
        var hidden = projected
        if let mask {
            hidden = MLX.where(mask.expandedDimensions(axis: -1), hidden, MLXArray(Float(0)).asType(hidden.dtype))
        }
        var outputs = [MLXArray]()
        for layer in encoder.layers {
            hidden = layer(hidden, mask: mask)
            if collecting { outputs.append(hidden) }
        }
        return (outputs, hidden)
    }

    /// Loads a release's `model.safetensors` (or a fine-tuned directory's, with its adapter and head),
    /// moving convolutions to MLX's channels-last layout. Every parameter must be supplied.
    public func loadWeights(fromDirectory directory: URL) throws {
        try loadWeights(fromDirectory: directory, leavingFresh: [])
    }
}

// MARK: - Factories and backend

/// W2V-BERT 2.0 (`facebook/w2v-bert-2.0`, MIT), Meta's multilingual self-supervised speech encoder,
/// ported into `MLXNN` at reference parity. The release is the pretrained encoder: it produces contextual
/// frame features and a mean-pooled utterance embedding; a directory carrying a CTC head (a fine-tune)
/// also transcribes.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXWav2Vec2Bert)
public final class NFKMLXWav2Vec2Bert: NSObject {
    @objc public static let modelName = "w2v-bert-2.0"
    static let requiredFiles = ["config.json"]
    static let optionalFiles = ["preprocessor_config.json", "vocab.json", "tokenizer_config.json"]
    static let weightFiles = ["model.safetensors"]

    /// Builds a backend from a release directory (`config.json`, `model.safetensors`, and for a CTC
    /// fine-tune `vocab.json`). Run inference off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> NFKMLXWav2Vec2BertBackend {
        let net = try NFKMLXWav2Vec2BertNet(configurationURL: directoryURL.appendingPathComponent("config.json"))
        try net.loadWeights(fromDirectory: directoryURL)
        let tokenizer = net.head != nil ? try NFKMLXWav2Vec2Tokenizer(directoryURL: directoryURL) : nil
        return NFKMLXWav2Vec2BertBackend(net: net, tokenizer: tokenizer)
    }

    /// The asynchronous form of the directory factory, at user-initiated quality of service.
    @objc(backendWithDirectoryURL:completionHandler:)
    public static func backend(directoryURL: URL, completionHandler: @escaping (NFKMLXWav2Vec2BertBackend?, Error?) -> Void) {
        Task.detached(priority: .userInitiated) {
            do { completionHandler(try backend(directoryURL: directoryURL), nil) }
            catch { completionHandler(nil, error) }
        }
    }

    /// Downloads a release into the hub cache and builds the backend. The public release is
    /// `facebook/w2v-bert-2.0` (ungated); the download fetches `config.json`, `preprocessor_config.json`,
    /// and `model.safetensors`, not the repo's `conformer_shaw.pt`. Blocking on the network.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXWav2Vec2BertBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXWav2Vec2BertBackend?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do { completionHandler(try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil) }
            catch { completionHandler(nil, error) }
        }
    }
}

/// The W2V-BERT inference backend. Audio under `NFKInputAudio` (any sample rate, resampled to 16 kHz)
/// comes back as `NFKOutputEmbedding` (`[NSNumber]`, 1024 wide), the mean of the real frames' features,
/// and as `NFKOutputText` from a directory with a CTC head.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXWav2Vec2BertBackend)
public final class NFKMLXWav2Vec2BertBackend: NSObject, NFKInferenceBackend {
    final class Holder: @unchecked Sendable {
        let net: NFKMLXWav2Vec2BertNet
        let tokenizer: NFKMLXWav2Vec2Tokenizer?
        init(net: NFKMLXWav2Vec2BertNet, tokenizer: NFKMLXWav2Vec2Tokenizer?) {
            self.net = net
            self.tokenizer = tokenizer
        }
    }

    private let holder: Holder

    init(net: NFKMLXWav2Vec2BertNet, tokenizer: NFKMLXWav2Vec2Tokenizer?) {
        holder = Holder(net: net, tokenizer: tokenizer)
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { NFKMLXWav2Vec2Bert.modelName }
    @objc public var supportedParameterKeys: Set<String> { [] }
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result { return result }
        if let error = job.error { throw error }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let holder = self.holder
        Task.detached(priority: .userInitiated) {
            do {
                guard let value = request.input(forKey: NFKInputAudio) else { throw NFKMLXError.unsupportedInput }
                let data: Data?
                if let asset = value as? NFKAudioAsset, let url = asset.fileURL {
                    data = try? Data(contentsOf: url)
                } else {
                    data = value as? Data
                }
                guard let data, let (samples, rate) = NFKMLXWaveFile.read(data) else { throw NFKMLXError.unsupportedInput }
                let matched = NFKMLXAudioRate.matched(samples, from: rate, to: NFKMLXWav2Vec2BertProcessor.sampleRate)
                guard matched.count >= 2 * 400 else { throw NFKMLXError.unsupportedInput }
                let (features, mask) = NFKMLXWav2Vec2BertProcessor.inputFeatures(matched)
                let hidden = holder.net(features, mask: mask)
                let weights = mask[0].asType(.float32).expandedDimensions(axis: -1)
                let pooled = ((hidden[0].asType(.float32) * weights).sum(axis: 0) / weights.sum())
                var outputs: [String: Any] = [:]
                if let tokenizer = holder.tokenizer, let (logits, frames) = holder.net.logits(features, mask: mask) {
                    let frameTokens = argMax(logits[0, ..<frames[0]], axis: -1).asType(.int32)
                    eval(frameTokens, pooled)
                    outputs[NFKOutputText] = tokenizer.text(forFrameTokens: frameTokens.asArray(Int32.self).map(Int.init))
                } else {
                    eval(pooled)
                }
                outputs[NFKOutputEmbedding] = pooled.asArray(Float.self).map { NSNumber(value: $0) }
                job.finish(with: NFKInferenceResult(outputs: outputs))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}
