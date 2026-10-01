//
//  NFKMLXWav2Vec2.swift
//  InferKitMLX
//
//  Wav2Vec2 and HuBERT (`facebook/wav2vec2-*`, `facebook/hubert-*`, Apache-2.0), self-supervised speech
//  encoders ported into MLXNN from transformers' `Wav2Vec2Model` / `HubertModel`, with the `…ForCTC`
//  character head. One configuration-driven network serves both families: a strided convolutional
//  feature encoder over the raw 16 kHz waveform, a feature projection, a grouped convolutional position
//  embedding, and a transformer encoder that is post-norm in the base releases and pre-norm ("stable
//  layer norm") in the large ones.
//

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// The geometry of a Wav2Vec2 or HuBERT release, read from its `config.json`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWav2Vec2Configuration: Sendable, Equatable {
    /// How the feature encoder normalizes its convolutions (`feat_extract_norm`).
    public enum FeatureNorm: String, Sendable {
        /// A per-channel group norm on the first convolution only (the base releases).
        case group
        /// A layer norm after every convolution (the large releases).
        case layer
    }

    /// The release's family (`model_type`): `wav2vec2` or `hubert`, which share one architecture.
    public var modelType: String = "wav2vec2"
    public var hiddenSize: Int = 768
    public var numHiddenLayers: Int = 12
    public var numAttentionHeads: Int = 12
    public var intermediateSize: Int = 3072
    public var convDimensions: [Int] = [512, 512, 512, 512, 512, 512, 512]
    public var convKernels: [Int] = [10, 3, 3, 3, 3, 2, 2]
    public var convStrides: [Int] = [5, 2, 2, 2, 2, 2, 2]
    public var convBias: Bool = false
    public var featureNorm: FeatureNorm = .group
    /// Pre-norm encoder layers and a final layer norm after the stack (`do_stable_layer_norm`).
    public var stableLayerNorm: Bool = false
    /// Whether the feature projection normalizes before projecting. Always on in Wav2Vec2; HuBERT's
    /// `feat_proj_layer_norm`.
    public var featureProjectionLayerNorm: Bool = true
    public var positionalConvKernel: Int = 128
    public var positionalConvGroups: Int = 16
    public var layerNormEps: Float = 1e-5
    /// The CTC head's vocabulary size, or nil for a pretraining release without a head.
    public var vocabularySize: Int?
    /// The CTC blank (`pad_token_id`).
    public var padTokenID: Int = 0
    /// Whether the feature extractor normalizes each utterance to zero mean and unit variance
    /// (`preprocessor_config.json`'s `do_normalize`).
    public var normalizesInput: Bool = true
    /// SpecAugment's time-mask probability and span in feature frames (`mask_time_prob`,
    /// `mask_time_length`), which fine-tuning applies.
    public var maskTimeProbability: Float = 0.05
    public var maskTimeLength: Int = 10
    public var maskTimeMinimumMasks: Int = 2

    /// `facebook/wav2vec2-base-960h`: the base encoder with its 32-character LibriSpeech head.
    public static let base960h: NFKMLXWav2Vec2Configuration = {
        var configuration = NFKMLXWav2Vec2Configuration()
        configuration.vocabularySize = 32
        return configuration
    }()

    public init() {}

    /// Reads a Hugging Face `config.json` for a `wav2vec2` or `hubert` model, and the directory's
    /// `preprocessor_config.json` when it sits beside it.
    public init(configurationURL: URL) throws {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: configurationURL)) as? [String: Any] ?? [:]
        let modelType = json["model_type"] as? String ?? "wav2vec2"
        guard ["wav2vec2", "hubert"].contains(modelType) else {
            throw NFKMLXError.unsupportedConfiguration("\(configurationURL.lastPathComponent) describes a \(modelType) "
                                                      + "model, not wav2vec2 or hubert")
        }
        self.modelType = modelType
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func ints(_ key: String, _ fallback: [Int]) -> [Int] { (json[key] as? [NSNumber])?.map(\.intValue) ?? fallback }
        func bool(_ key: String, _ fallback: Bool) -> Bool { (json[key] as? NSNumber)?.boolValue ?? fallback }
        hiddenSize = int("hidden_size", hiddenSize)
        numHiddenLayers = int("num_hidden_layers", numHiddenLayers)
        numAttentionHeads = int("num_attention_heads", numAttentionHeads)
        intermediateSize = int("intermediate_size", intermediateSize)
        convDimensions = ints("conv_dim", convDimensions)
        convKernels = ints("conv_kernel", convKernels)
        convStrides = ints("conv_stride", convStrides)
        convBias = bool("conv_bias", convBias)
        featureNorm = FeatureNorm(rawValue: json["feat_extract_norm"] as? String ?? "group") ?? .group
        stableLayerNorm = bool("do_stable_layer_norm", stableLayerNorm)
        featureProjectionLayerNorm = modelType == "hubert" ? bool("feat_proj_layer_norm", true) : true
        positionalConvKernel = int("num_conv_pos_embeddings", positionalConvKernel)
        positionalConvGroups = int("num_conv_pos_embedding_groups", positionalConvGroups)
        layerNormEps = Float((json["layer_norm_eps"] as? NSNumber)?.doubleValue ?? Double(layerNormEps))
        padTokenID = int("pad_token_id", padTokenID)
        maskTimeProbability = Float((json["mask_time_prob"] as? NSNumber)?.doubleValue ?? Double(maskTimeProbability))
        maskTimeLength = int("mask_time_length", maskTimeLength)
        maskTimeMinimumMasks = int("mask_time_min_masks", maskTimeMinimumMasks)
        let architectures = json["architectures"] as? [String] ?? []
        if architectures.contains(where: { $0.hasSuffix("ForCTC") }) {
            vocabularySize = int("vocab_size", 32)
        }
        for (key, value) in [("hidden_act", "gelu"), ("feat_extract_activation", "gelu")] {
            let named = json[key] as? String ?? value
            guard named == value else {
                throw NFKMLXError.unsupportedConfiguration("\(key) \(named) is not the gelu every release uses")
            }
        }
        if bool("conv_pos_batch_norm", false) || json["adapter_attn_dim"] is NSNumber || bool("add_adapter", false) {
            throw NFKMLXError.unsupportedConfiguration("\(configurationURL.lastPathComponent) sets an adapter or a "
                                                      + "position batch norm, which no Wav2Vec2 or HuBERT release uses")
        }
        let processor = configurationURL.deletingLastPathComponent().appendingPathComponent("preprocessor_config.json")
        if let data = try? Data(contentsOf: processor),
           let settings = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let normalize = settings["do_normalize"] as? NSNumber {
            normalizesInput = normalize.boolValue
        }
    }

    /// The number of feature frames the encoder produces from `samples` waveform samples.
    public func frameCount(samples: Int) -> Int {
        zip(convKernels, convStrides).reduce(samples) { length, layer in (length - layer.0) / layer.1 + 1 }
    }
}

// MARK: - Feature encoder

/// One strided convolution of the feature encoder, with its norm (`layer_norm` in both variants) and a GELU.
final class NFKWav2Vec2ConvLayer: Module {
    @ModuleInfo(key: "conv") var conv: NFKConv1d
    @ModuleInfo(key: "layer_norm") var norm: Module?

    init(_ c: NFKMLXWav2Vec2Configuration, index: Int) {
        let input = index == 0 ? 1 : c.convDimensions[index - 1]
        let output = c.convDimensions[index]
        _conv.wrappedValue = NFKConv1d(inputChannels: input, outputChannels: output, kernelSize: c.convKernels[index],
                                       stride: c.convStrides[index], bias: c.convBias)
        switch c.featureNorm {
        case .layer:
            _norm.wrappedValue = NFKLayerNorm(dimensions: output, eps: 1e-5)
        case .group where index == 0:
            _norm.wrappedValue = NFKGroupNorm(groupCount: output, dimensions: output, eps: 1e-5, affine: true,
                                              pytorchCompatible: true)
        case .group:
            _norm.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv(x)
        if let layerNorm = norm as? NFKLayerNorm {
            y = layerNorm(y)
        } else if let groupNorm = norm as? NFKGroupNorm {
            y = groupNorm(y)
        }
        return gelu(y)
    }
}

/// The convolutional feature encoder: `[B, samples, 1]` to `[B, frames, conv_dim[-1]]`.
final class NFKWav2Vec2FeatureEncoder: Module {
    @ModuleInfo(key: "conv_layers") var layers: [NFKWav2Vec2ConvLayer]

    init(_ c: NFKMLXWav2Vec2Configuration) {
        _layers.wrappedValue = (0 ..< c.convDimensions.count).map { NFKWav2Vec2ConvLayer(c, index: $0) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers.reduce(x) { $1($0) }
    }
}

/// The feature projection: an optional layer norm, then a linear map to the encoder width.
final class NFKWav2Vec2FeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var norm: NFKLayerNorm?
    @ModuleInfo(key: "projection") var projection: Linear

    init(_ c: NFKMLXWav2Vec2Configuration) {
        let features = c.convDimensions.last ?? 512
        _norm.wrappedValue = c.featureProjectionLayerNorm ? NFKLayerNorm(dimensions: features, eps: c.layerNormEps) : nil
        _projection.wrappedValue = Linear(features, c.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(norm.map { $0(x) } ?? x)
    }
}

// MARK: - Encoder

/// The position convolution's weight-normalized kernel (`weight_norm(conv, dim=2)`): the weight is
/// `weight_g · weight_v / ‖weight_v‖`, the norm taken per kernel tap over the output and input channels.
/// The gain and direction stay separate parameters, as the reference trains them.
final class NFKWav2Vec2WeightNormConv: Module {
    @ParameterInfo(key: "weight_g") var gain: MLXArray
    @ParameterInfo(key: "weight_v") var direction: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let padding: Int
    let groups: Int

    init(channels: Int, kernel: Int, groups: Int) {
        self.padding = kernel / 2
        self.groups = groups
        let scale = 1 / Float(channels / groups * kernel).squareRoot()
        _direction.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [channels, kernel, channels / groups])
        _gain.wrappedValue = MLXArray.ones([1, kernel, 1])
        _bias.wrappedValue = MLXArray.zeros([channels])
        super.init()
    }

    /// The kernel `[out, taps, in / groups]` the gain and direction resolve to.
    var weight: MLXArray {
        let v = direction.asType(.float32)
        return (gain.asType(.float32) * v / sqrt((v * v).sum(axes: [0, 2], keepDims: true))).asType(direction.dtype)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let reduced = NFKReferenceRounding.isReduced(x)
        let input = reduced ? x.asType(.float32) : x
        let kernel = reduced ? weight.asType(.float32) : weight
        return (Self.groupedConvolution(input, kernel, padding: padding, groups: groups) + bias.asType(input.dtype))
            .asType(x.dtype)
    }

    /// The widest kernel whose input gradient MLX's GPU convolution computes correctly. At 17 taps and
    /// more, with a channel count that is a multiple of 16, the Metal backward of `conv1d` and `conv2d`
    /// returns a wrong input gradient (mlx 0.32.2) while the forward and the weight gradient stay exact.
    static let exactGradientTaps = 16

    /// The grouped convolution over explicitly zero-padded input, summed from kernel slices of at most
    /// ``exactGradientTaps`` taps, each over the input window it reaches. The forward equals
    /// `conv1d(…, padding:, groups:)` up to summation order; the input gradient is PyTorch's.
    static func groupedConvolution(_ x: MLXArray, _ kernel: MLXArray, padding: Int, groups: Int) -> MLXArray {
        let padded = padding > 0 ? MLX.padded(x, widths: [.init(0), .init((padding, padding)), .init(0)]) : x
        let taps = kernel.dim(1)
        let outputLength = padded.dim(1) - taps + 1
        var output: MLXArray?
        for start in stride(from: 0, to: taps, by: exactGradientTaps) {
            let width = min(exactGradientTaps, taps - start)
            let window = padded[0..., start ..< (start + outputLength + width - 1), 0...]
            let part = conv1d(window, kernel[0..., start ..< (start + width), 0...], groups: groups)
            output = output.map { $0 + part } ?? part
        }
        return output!
    }
}

/// The grouped convolutional position embedding: a same-padded, weight-normalized convolution whose
/// trailing frame is dropped for an even kernel, then a GELU.
final class NFKWav2Vec2PositionalConv: Module {
    @ModuleInfo(key: "conv") var conv: NFKWav2Vec2WeightNormConv
    let dropsTrailingFrame: Bool

    init(_ c: NFKMLXWav2Vec2Configuration) {
        _conv.wrappedValue = NFKWav2Vec2WeightNormConv(channels: c.hiddenSize, kernel: c.positionalConvKernel,
                                                       groups: c.positionalConvGroups)
        dropsTrailingFrame = c.positionalConvKernel % 2 == 0
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv(x)
        if dropsTrailingFrame {
            y = y[0..., ..<(y.dim(1) - 1), 0...]
        }
        return gelu(y)
    }
}

final class NFKWav2Vec2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let heads: Int

    init(_ c: NFKMLXWav2Vec2Configuration) {
        heads = c.numAttentionHeads
        _qProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _kProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _vProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _outProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        let headDimension = width / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([batch, length, heads, headDimension]).transposed(0, 2, 1, 3) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(qProj(x)), keys: split(kProj(x)), values: split(vProj(x)),
            scale: 1 / Float(headDimension).squareRoot(), mask: .none)
        return outProj(attended.transposed(0, 2, 1, 3).reshaped([batch, length, width]))
    }
}

final class NFKWav2Vec2FeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediate: Linear
    @ModuleInfo(key: "output_dense") var output: Linear

    init(_ c: NFKMLXWav2Vec2Configuration) {
        _intermediate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize)
        _output.wrappedValue = Linear(c.intermediateSize, c.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { output(gelu(intermediate(x))) }
}

/// One encoder layer. Post-norm (`Wav2Vec2EncoderLayer`) normalizes after each residual sum; pre-norm
/// (`Wav2Vec2EncoderLayerStableLayerNorm`) normalizes each branch's input.
final class NFKWav2Vec2EncoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: NFKWav2Vec2Attention
    @ModuleInfo(key: "layer_norm") var layerNorm: NFKLayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: NFKWav2Vec2FeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: NFKLayerNorm
    let preNorm: Bool

    init(_ c: NFKMLXWav2Vec2Configuration) {
        preNorm = c.stableLayerNorm
        _attention.wrappedValue = NFKWav2Vec2Attention(c)
        _layerNorm.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _feedForward.wrappedValue = NFKWav2Vec2FeedForward(c)
        _finalLayerNorm.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if preNorm {
            let attended = x + attention(layerNorm(x))
            return attended + feedForward(finalLayerNorm(attended))
        }
        let attended = layerNorm(x + attention(x))
        return finalLayerNorm(attended + feedForward(attended))
    }
}

final class NFKWav2Vec2Encoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var positionalConv: NFKWav2Vec2PositionalConv
    @ModuleInfo(key: "layer_norm") var layerNorm: NFKLayerNorm
    @ModuleInfo(key: "layers") var layers: [NFKWav2Vec2EncoderLayer]
    let stable: Bool

    init(_ c: NFKMLXWav2Vec2Configuration) {
        stable = c.stableLayerNorm
        _positionalConv.wrappedValue = NFKWav2Vec2PositionalConv(c)
        _layerNorm.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _layers.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in NFKWav2Vec2EncoderLayer(c) }
        super.init()
    }

    /// The encoder's output, and each layer's output when `collecting`.
    func callAsFunction(_ x: MLXArray, collecting: Bool = false) -> (output: MLXArray, layers: [MLXArray]) {
        var hidden = x + positionalConv(x)
        if !stable {
            hidden = layerNorm(hidden)
        }
        var outputs = [MLXArray]()
        for layer in layers {
            hidden = layer(hidden)
            if collecting { outputs.append(hidden) }
        }
        if stable {
            hidden = layerNorm(hidden)
        }
        return (hidden, outputs)
    }
}

// MARK: - Network

/// The Wav2Vec2 / HuBERT network: waveform in, contextual frame features out, and character logits from
/// a CTC release. One feature frame covers 20 ms of 16 kHz audio (a 400-sample receptive field).
///
/// Module keys mirror the transformers checkpoint below its `wav2vec2.` or `hubert.` prefix, with the
/// CTC head at `lm_head`.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXWav2Vec2Net: Module {
    @ModuleInfo(key: "feature_extractor") var featureEncoder: NFKWav2Vec2FeatureEncoder
    @ModuleInfo(key: "feature_projection") var featureProjection: NFKWav2Vec2FeatureProjection
    @ModuleInfo(key: "encoder") var encoder: NFKWav2Vec2Encoder
    /// The embedding SpecAugment writes into masked frames during fine-tuning (`masked_spec_embed`).
    @ParameterInfo(key: "masked_spec_embed") var maskedSpecEmbed: MLXArray
    @ModuleInfo(key: "lm_head") var head: Linear?

    public let configuration: NFKMLXWav2Vec2Configuration

    public init(_ configuration: NFKMLXWav2Vec2Configuration) {
        self.configuration = configuration
        _featureEncoder.wrappedValue = NFKWav2Vec2FeatureEncoder(configuration)
        _featureProjection.wrappedValue = NFKWav2Vec2FeatureProjection(configuration)
        _encoder.wrappedValue = NFKWav2Vec2Encoder(configuration)
        _maskedSpecEmbed.wrappedValue = MLXRandom.uniform(low: 0, high: 1, [configuration.hiddenSize])
        _head.wrappedValue = configuration.vocabularySize.map { Linear(configuration.hiddenSize, $0) }
        super.init()
    }

    public convenience init(configurationURL: URL) throws {
        self.init(try NFKMLXWav2Vec2Configuration(configurationURL: configurationURL))
    }

    /// Whether the network carries a CTC character head.
    public var transcribes: Bool { head != nil }

    /// The feature encoder's output `[B, frames, conv_dim[-1]]` from a normalized waveform `[B, samples]`.
    public func extractFeatures(_ waveform: MLXArray) -> MLXArray {
        featureEncoder(waveform.expandedDimensions(axis: -1))
    }

    /// The encoder's last hidden state `[B, frames, hidden]` from a normalized waveform `[B, samples]`.
    public func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        encode(featureProjection(extractFeatures(waveform))).output
    }

    /// Runs the encoder over projected features, optionally with SpecAugment's masked frames replaced.
    func encode(_ projected: MLXArray, timeMask: MLXArray? = nil, collecting: Bool = false)
        -> (output: MLXArray, layers: [MLXArray]) {
        var hidden = projected
        if let timeMask {
            hidden = MLX.where(timeMask.expandedDimensions(axis: -1), maskedSpecEmbed.asType(hidden.dtype), hidden)
        }
        return encoder(hidden, collecting: collecting)
    }

    /// Every seam of one forward, for parity measurement: the feature encoder's output, the projected
    /// features, each encoder layer's output, the last hidden state, and the CTC logits.
    func seams(_ waveform: MLXArray) -> (features: MLXArray, projected: MLXArray, layers: [MLXArray],
                                        output: MLXArray, logits: MLXArray?) {
        let features = extractFeatures(waveform)
        let projected = featureProjection(features)
        let (output, layers) = encode(projected, collecting: true)
        return (features, projected, layers, output, head.map { $0(output) })
    }

    /// The CTC logits `[B, frames, vocabulary]` from a normalized waveform, or nil for a release
    /// without a head.
    public func logits(_ waveform: MLXArray) -> MLXArray? {
        head.map { $0(self(waveform)) }
    }
}

// MARK: - Processor and CTC decoding

/// The Wav2Vec2 feature extractor: a mono 16 kHz waveform, normalized per utterance to zero mean and
/// unit variance (`(x − mean) / sqrt(var + 1e-7)`) when the release asks for it.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXWav2Vec2Processor {
    public static let sampleRate = 16000

    /// The model input `[1, samples]` from 16 kHz samples.
    public static func inputValues(_ samples: [Float], normalize: Bool) -> MLXArray {
        guard normalize, !samples.isEmpty else { return MLXArray(samples).reshaped([1, samples.count]) }
        let count = Double(samples.count)
        let mean = samples.reduce(0.0) { $0 + Double($1) } / count
        let variance = samples.reduce(0.0) { $0 + (Double($1) - mean) * (Double($1) - mean) } / count
        let scale = 1 / (variance + 1e-7).squareRoot()
        return MLXArray(samples.map { Float((Double($0) - mean) * scale) }).reshaped([1, samples.count])
    }
}

/// The character tokenizer of a CTC release (`Wav2Vec2CTCTokenizer`): `vocab.json`'s characters, the
/// pad token as the CTC blank, and `|` as the word delimiter.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXWav2Vec2Tokenizer: NSObject, @unchecked Sendable {
    let tokens: [String]
    let ids: [String: Int]
    let padToken: String
    let wordDelimiter: String
    let unknownToken: String

    /// Reads `vocab.json` and, when present, `tokenizer_config.json`'s pad, unknown, and delimiter tokens.
    public init(directoryURL: URL) throws {
        let data = try Data(contentsOf: directoryURL.appendingPathComponent("vocab.json"))
        guard let vocabulary = try JSONSerialization.jsonObject(with: data) as? [String: NSNumber] else {
            throw NFKMLXError.unsupportedConfiguration("vocab.json is not a token-to-id map")
        }
        var settings = [String: Any]()
        if let config = try? Data(contentsOf: directoryURL.appendingPathComponent("tokenizer_config.json")),
           let json = try? JSONSerialization.jsonObject(with: config) as? [String: Any] {
            settings = json
        }
        let ids = vocabulary.mapValues(\.intValue)
        var tokens = [String](repeating: "", count: (ids.values.max() ?? -1) + 1)
        for (token, id) in ids { tokens[id] = token }
        self.ids = ids
        self.tokens = tokens
        padToken = settings["pad_token"] as? String ?? "<pad>"
        wordDelimiter = settings["word_delimiter_token"] as? String ?? "|"
        unknownToken = settings["unk_token"] as? String ?? "<unk>"
        super.init()
    }

    /// Greedy CTC decoding of per-frame token ids: consecutive repeats collapse, the blank drops, and the
    /// word delimiter becomes a space.
    public func text(forFrameTokens frameTokens: [Int]) -> String {
        var collapsed = [String]()
        var previous: Int?
        for id in frameTokens where id != previous {
            previous = id
            collapsed.append(id < tokens.count ? tokens[id] : unknownToken)
        }
        let characters = collapsed.filter { $0 != padToken }.map { $0 == wordDelimiter ? " " : $0 }
        return characters.joined().trimmingCharacters(in: .whitespaces)
    }

    /// The label ids of a transcript for CTC training: each character's id, the word delimiter for a
    /// space, and the unknown token for a character outside the vocabulary. The releases' vocabularies
    /// are upper case; the transcript is matched as given.
    public func labels(for transcript: String) -> [Int] {
        transcript.map { character -> Int in
            let token = character == " " ? wordDelimiter : String(character)
            return ids[token] ?? ids[unknownToken] ?? 0
        }
    }
}
