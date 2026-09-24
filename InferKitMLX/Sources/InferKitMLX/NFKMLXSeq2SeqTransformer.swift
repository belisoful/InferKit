//
//  NFKMLXSeq2SeqTransformer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The BART-family encoder-decoder transformer (`MarianMTModel`, `M2M100ForConditionalGeneration`,
// `BartForConditionalGeneration`): a token embedding shared by both stacks and the output projection,
// sinusoidal or learned absolute positions, post- or pre-normalized blocks, a decoder whose blocks add a
// cross-attention over the encoder output, and an optional bias on the logits. The variants differ by
// configuration flags, so one network covers them, and the module keys follow the transformers layout
// with the `model.` prefix dropped, so a checkpoint loads by prefix strip.

/// The geometry and flags of a BART-family encoder-decoder.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXSeq2SeqConfiguration: Sendable {

    /// How positions enter the embeddings.
    public enum Positions: Sendable {
        /// Marian: `sin(pos / 10000^(2i/d))` in the first half, `cos` in the second, positions from 0.
        case marianSinusoidal
        /// fairseq (M2M100, NLLB): `10000^(-i/(half-1))` frequencies, positions from `pad + 1`.
        case fairseqSinusoidal
        /// BART, mBART: a learned table read at `position + 2`.
        case learned
    }

    public enum Activation: Sendable {
        case swish
        case relu
        case gelu
    }

    public var vocabularySize: Int
    public var dModel: Int
    public var encoderLayers: Int
    public var decoderLayers: Int
    public var heads: Int
    public var encoderFFDim: Int
    public var decoderFFDim: Int
    public var maxPositions: Int
    public var activation: Activation
    public var positions: Positions
    /// Pre-normalized blocks (`normalize_before`), as M2M100 and mBART; off is the post-normalized
    /// Marian and BART layout.
    public var normalizeBefore: Bool
    /// A layer norm on each stack's output (`encoder.layer_norm`), as M2M100 and mBART.
    public var finalLayerNorm: Bool
    /// A layer norm on the summed embeddings (`layernorm_embedding`), as BART and mBART.
    public var layerNormEmbedding: Bool
    /// Multiplies the token embedding by `sqrt(dModel)`.
    public var scaleEmbedding: Bool
    /// A `final_logits_bias` added to the logits, as Marian and BART carry.
    public var finalLogitsBias: Bool
    public var layerNormEps: Float
    public var padTokenId: Int
    public var eosTokenId: Int
    public var decoderStartTokenId: Int
    /// The width of the memory the decoder's cross-attention reads, when it differs from `dModel`
    /// (TrOCR's 768-wide ViT features under a 1024-wide decoder). Nil is `dModel`.
    public var crossAttentionWidth: Int?
    /// An `lm_head` of its own rather than the shared embedding (`tie_word_embeddings` off).
    public var untiedOutputProjection: Bool

    public init(vocabularySize: Int, dModel: Int, encoderLayers: Int, decoderLayers: Int, heads: Int,
                encoderFFDim: Int, decoderFFDim: Int, maxPositions: Int = 1024,
                activation: Activation, positions: Positions, normalizeBefore: Bool, finalLayerNorm: Bool,
                layerNormEmbedding: Bool = false, scaleEmbedding: Bool, finalLogitsBias: Bool,
                layerNormEps: Float = 1e-5, padTokenId: Int, eosTokenId: Int, decoderStartTokenId: Int,
                crossAttentionWidth: Int? = nil, untiedOutputProjection: Bool = false) {
        self.vocabularySize = vocabularySize
        self.dModel = dModel
        self.encoderLayers = encoderLayers
        self.decoderLayers = decoderLayers
        self.heads = heads
        self.encoderFFDim = encoderFFDim
        self.decoderFFDim = decoderFFDim
        self.maxPositions = maxPositions
        self.activation = activation
        self.positions = positions
        self.normalizeBefore = normalizeBefore
        self.finalLayerNorm = finalLayerNorm
        self.layerNormEmbedding = layerNormEmbedding
        self.scaleEmbedding = scaleEmbedding
        self.finalLogitsBias = finalLogitsBias
        self.layerNormEps = layerNormEps
        self.padTokenId = padTokenId
        self.eosTokenId = eosTokenId
        self.decoderStartTokenId = decoderStartTokenId
        self.crossAttentionWidth = crossAttentionWidth
        self.untiedOutputProjection = untiedOutputProjection
    }

    /// The geometry a transformers `config.json` declares. `model_type` picks the family's flags:
    /// `marian`, `m2m_100`, `nllb` (M2M100 layout), `bart`, `mbart`, and `trocr` (a decoder alone, with
    /// no encoder layers, read against an outside memory).
    public init(huggingFaceConfigURL url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        try self.init(huggingFaceConfig: json)
    }

    public init(huggingFaceConfig json: [String: Any]) throws {
        func int(_ key: String, _ fallback: Int? = nil) throws -> Int {
            if let value = json[key] as? NSNumber { return value.intValue }
            guard let fallback else { throw NFKMLXError.unsupportedConfiguration("config.json lacks \(key)") }
            return fallback
        }
        let modelType = (json["model_type"] as? String) ?? "bart"
        let family: (positions: Positions, normalizeBefore: Bool, finalLayerNorm: Bool,
                     layerNormEmbedding: Bool, finalLogitsBias: Bool)
        switch modelType {
        case "marian":
            family = (.marianSinusoidal, false, false, false, true)
        case "m2m_100", "nllb":
            family = (.fairseqSinusoidal, true, true, false, false)
        case "mbart":
            family = (.learned, true, true, true, true)
        case "bart", "florence2_language":
            family = (.learned, false, false, true, true)
        case "trocr":
            let learned = (json["use_learned_position_embeddings"] as? Bool) ?? true
            family = (learned ? .learned : .fairseqSinusoidal, false, false, true, false)
        default:
            throw NFKMLXError.unsupportedConfiguration("\(modelType) is not a BART-family model type")
        }
        let activationName = (json["activation_function"] as? String) ?? "gelu"
        let activation: Activation
        switch activationName {
        case "swish", "silu": activation = .swish
        case "relu": activation = .relu
        case "gelu", "gelu_new": activation = .gelu
        default: throw NFKMLXError.unsupportedConfiguration("\(activationName) is not a supported activation")
        }
        let normalizeBefore = (json["normalize_before"] as? Bool) ?? family.normalizeBefore
        let finalLayerNorm = (json["add_final_layer_norm"] as? Bool) ?? family.finalLayerNorm
        let layerNormEmbedding = (json["normalize_embedding"] as? Bool)
            ?? (json["layernorm_embedding"] as? Bool) ?? family.layerNormEmbedding
        let dModel = try int("d_model")
        let decoderOnly = modelType == "trocr"
        let heads = try int(decoderOnly ? "decoder_attention_heads" : "encoder_attention_heads")
        let decoderFFDim = try int("decoder_ffn_dim")
        let pad = try int("pad_token_id")
        let eos = try int("eos_token_id")
        let crossWidth = (json["cross_attention_hidden_size"] as? NSNumber)?.intValue
        self.init(vocabularySize: try int("vocab_size"), dModel: dModel,
                  encoderLayers: try int("encoder_layers", decoderOnly ? 0 : nil), decoderLayers: try int("decoder_layers"),
                  heads: heads, encoderFFDim: try int("encoder_ffn_dim", decoderOnly ? decoderFFDim : nil),
                  decoderFFDim: decoderFFDim,
                  maxPositions: try int("max_position_embeddings", 1024),
                  activation: activation, positions: family.positions,
                  normalizeBefore: normalizeBefore, finalLayerNorm: finalLayerNorm,
                  layerNormEmbedding: layerNormEmbedding,
                  scaleEmbedding: (json["scale_embedding"] as? Bool) ?? false,
                  finalLogitsBias: family.finalLogitsBias,
                  padTokenId: pad, eosTokenId: eos,
                  decoderStartTokenId: try int("decoder_start_token_id", eos),
                  crossAttentionWidth: crossWidth.flatMap { $0 == dModel ? nil : $0 },
                  untiedOutputProjection: !((json["tie_word_embeddings"] as? Bool) ?? true))
    }

    /// A small Marian-layout geometry for tests.
    public static let tinyMarian = NFKMLXSeq2SeqConfiguration(
        vocabularySize: 64, dModel: 16, encoderLayers: 2, decoderLayers: 2, heads: 2,
        encoderFFDim: 32, decoderFFDim: 32, maxPositions: 32, activation: .swish, positions: .marianSinusoidal,
        normalizeBefore: false, finalLayerNorm: false, scaleEmbedding: true, finalLogitsBias: true,
        padTokenId: 63, eosTokenId: 0, decoderStartTokenId: 63)

    /// A small M2M100-layout geometry for tests.
    public static let tinyM2M100 = NFKMLXSeq2SeqConfiguration(
        vocabularySize: 64, dModel: 16, encoderLayers: 2, decoderLayers: 2, heads: 2,
        encoderFFDim: 32, decoderFFDim: 32, maxPositions: 32, activation: .relu, positions: .fairseqSinusoidal,
        normalizeBefore: true, finalLayerNorm: true, scaleEmbedding: true, finalLogitsBias: false,
        padTokenId: 1, eosTokenId: 2, decoderStartTokenId: 2)

    var headDim: Int { dModel / heads }
}

// MARK: - Cache

/// The decoder's per-step state: each block's self-attention keys and values so far, and the
/// encoder projections its cross-attention computed once.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXSeq2SeqCache {
    var selfKeys: [MLXArray?]
    var selfValues: [MLXArray?]
    var crossKeys: [MLXArray?]
    var crossValues: [MLXArray?]

    public init(layers: Int) {
        selfKeys = Array(repeating: nil, count: layers)
        selfValues = Array(repeating: nil, count: layers)
        crossKeys = Array(repeating: nil, count: layers)
        crossValues = Array(repeating: nil, count: layers)
    }

    /// The number of decoder positions cached so far.
    public var length: Int { selfKeys.first??.dim(2) ?? 0 }

    /// Reorders the batch rows, which is how a beam search carries the surviving beams forward.
    public func reorder(_ rows: MLXArray) {
        for layer in 0 ..< selfKeys.count {
            selfKeys[layer] = selfKeys[layer].map { $0.take(rows, axis: 0) }
            selfValues[layer] = selfValues[layer].map { $0.take(rows, axis: 0) }
            crossKeys[layer] = crossKeys[layer].map { $0.take(rows, axis: 0) }
            crossValues[layer] = crossValues[layer].map { $0.take(rows, axis: 0) }
        }
    }
}

/// A sinusoid table kept outside the parameter tree.
final class NFKSeq2SeqSinusoids {
    let table: MLXArray
    init(_ table: MLXArray) { self.table = table }
}

// MARK: - Modules

final class NFKSeq2SeqAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let heads: Int
    let headDim: Int

    init(_ c: NFKMLXSeq2SeqConfiguration, keyValueWidth: Int? = nil) {
        heads = c.heads
        headDim = c.headDim
        _qProj.wrappedValue = Linear(c.dModel, c.dModel)
        _kProj.wrappedValue = Linear(keyValueWidth ?? c.dModel, c.dModel)
        _vProj.wrappedValue = Linear(keyValueWidth ?? c.dModel, c.dModel)
        _outProj.wrappedValue = Linear(c.dModel, c.dModel)
    }

    private func split(_ t: MLXArray) -> MLXArray {
        t.reshaped([t.dim(0), t.dim(1), heads, headDim]).transposed(0, 2, 1, 3)
    }

    /// Self-attention over `x`, appending to the cached keys and values when a cache is given.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cachedKeys: inout MLXArray?, cachedValues: inout MLXArray?) -> MLXArray {
        var keys = split(kProj(x))
        var values = split(vProj(x))
        if let previousKeys = cachedKeys, let previousValues = cachedValues {
            keys = concatenated([previousKeys, keys], axis: 2)
            values = concatenated([previousValues, values], axis: 2)
        }
        cachedKeys = keys
        cachedValues = values
        return attend(split(qProj(x)), keys, values, mask: mask)
    }

    /// Cross-attention from `x` to `memory`, projecting the memory once per cache.
    func callAsFunction(_ x: MLXArray, memory: MLXArray, cachedKeys: inout MLXArray?, cachedValues: inout MLXArray?) -> MLXArray {
        if cachedKeys == nil || cachedValues == nil {
            cachedKeys = split(kProj(memory))
            cachedValues = split(vProj(memory))
        }
        return attend(split(qProj(x)), cachedKeys!, cachedValues!, mask: nil)
    }

    private func attend(_ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, mask: MLXArray?) -> MLXArray {
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1 / sqrt(Float(headDim)), mask: mask)
        return outProj(attended.transposed(0, 2, 1, 3).reshaped([queries.dim(0), queries.dim(2), heads * headDim]))
    }
}

final class NFKSeq2SeqEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: NFKSeq2SeqAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
    let configuration: NFKMLXSeq2SeqConfiguration

    init(_ c: NFKMLXSeq2SeqConfiguration) {
        configuration = c
        _selfAttn.wrappedValue = NFKSeq2SeqAttention(c)
        _selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: c.dModel, eps: c.layerNormEps)
        _fc1.wrappedValue = Linear(c.dModel, c.encoderFFDim)
        _fc2.wrappedValue = Linear(c.encoderFFDim, c.dModel)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: c.dModel, eps: c.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var none: MLXArray?
        var noneValues: MLXArray?
        var h = x
        if configuration.normalizeBefore {
            h = h + selfAttn(selfAttnLayerNorm(h), mask: mask, cachedKeys: &none, cachedValues: &noneValues)
            h = h + fc2(configuration.activate(fc1(finalLayerNorm(h))))
        } else {
            h = selfAttnLayerNorm(h + selfAttn(h, mask: mask, cachedKeys: &none, cachedValues: &noneValues))
            h = finalLayerNorm(h + fc2(configuration.activate(fc1(h))))
        }
        return h
    }
}

final class NFKSeq2SeqDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: NFKSeq2SeqAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var encoderAttn: NFKSeq2SeqAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
    let configuration: NFKMLXSeq2SeqConfiguration

    init(_ c: NFKMLXSeq2SeqConfiguration) {
        configuration = c
        _selfAttn.wrappedValue = NFKSeq2SeqAttention(c)
        _selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: c.dModel, eps: c.layerNormEps)
        _encoderAttn.wrappedValue = NFKSeq2SeqAttention(c, keyValueWidth: c.crossAttentionWidth)
        _encoderAttnLayerNorm.wrappedValue = LayerNorm(dimensions: c.dModel, eps: c.layerNormEps)
        _fc1.wrappedValue = Linear(c.dModel, c.decoderFFDim)
        _fc2.wrappedValue = Linear(c.decoderFFDim, c.dModel)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: c.dModel, eps: c.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray, memory: MLXArray, mask: MLXArray?, cache: NFKMLXSeq2SeqCache, layer: Int) -> MLXArray {
        var h = x
        if configuration.normalizeBefore {
            h = h + selfAttn(selfAttnLayerNorm(h), mask: mask, cachedKeys: &cache.selfKeys[layer], cachedValues: &cache.selfValues[layer])
            h = h + encoderAttn(encoderAttnLayerNorm(h), memory: memory, cachedKeys: &cache.crossKeys[layer], cachedValues: &cache.crossValues[layer])
            h = h + fc2(configuration.activate(fc1(finalLayerNorm(h))))
        } else {
            h = selfAttnLayerNorm(h + selfAttn(h, mask: mask, cachedKeys: &cache.selfKeys[layer], cachedValues: &cache.selfValues[layer]))
            h = encoderAttnLayerNorm(h + encoderAttn(h, memory: memory, cachedKeys: &cache.crossKeys[layer], cachedValues: &cache.crossValues[layer]))
            h = finalLayerNorm(h + fc2(configuration.activate(fc1(h))))
        }
        return h
    }
}

/// The encoder stack: `layers`, an optional `layernorm_embedding`, and an optional `layer_norm`.
final class NFKSeq2SeqEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKSeq2SeqEncoderLayer]
    @ModuleInfo(key: "layernorm_embedding") var layerNormEmbedding: LayerNorm?
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm?
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding?

    init(_ c: NFKMLXSeq2SeqConfiguration) {
        _layers.wrappedValue = (0 ..< c.encoderLayers).map { _ in NFKSeq2SeqEncoderLayer(c) }
        if c.layerNormEmbedding { _layerNormEmbedding.wrappedValue = LayerNorm(dimensions: c.dModel, eps: c.layerNormEps) }
        if c.finalLayerNorm { _layerNorm.wrappedValue = LayerNorm(dimensions: c.dModel, eps: c.layerNormEps) }
        if c.positions == .learned { _embedPositions.wrappedValue = Embedding(embeddingCount: c.maxPositions + 2, dimensions: c.dModel) }
    }
}

/// The decoder stack, laid out like the encoder plus cross-attention in each block.
final class NFKSeq2SeqDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKSeq2SeqDecoderLayer]
    @ModuleInfo(key: "layernorm_embedding") var layerNormEmbedding: LayerNorm?
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm?
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding?

    init(_ c: NFKMLXSeq2SeqConfiguration) {
        _layers.wrappedValue = (0 ..< c.decoderLayers).map { _ in NFKSeq2SeqDecoderLayer(c) }
        if c.layerNormEmbedding { _layerNormEmbedding.wrappedValue = LayerNorm(dimensions: c.dModel, eps: c.layerNormEps) }
        if c.finalLayerNorm { _layerNorm.wrappedValue = LayerNorm(dimensions: c.dModel, eps: c.layerNormEps) }
        if c.positions == .learned { _embedPositions.wrappedValue = Embedding(embeddingCount: c.maxPositions + 2, dimensions: c.dModel) }
    }
}

// MARK: - Network

/// A BART-family encoder-decoder: Marian, M2M100 and NLLB, BART and mBART, by configuration.
///
/// @discussion Module keys mirror the transformers checkpoint with its `model.` prefix removed:
/// `shared`, `encoder.layers.N.self_attn.{q,k,v,out}_proj`, `encoder.layers.N.{self_attn_layer_norm,
/// fc1,fc2,final_layer_norm}`, `encoder.layer_norm`, `encoder.layernorm_embedding`, the decoder's
/// counterparts with `encoder_attn` and `encoder_attn_layer_norm` for the cross-attention, a learned
/// `embed_positions` where the family has one, and `final_logits_bias`. The output projection is the
/// shared embedding, as every family ties it. ``loadWeights(from:)`` performs the prefix strip and drops
/// the duplicated `embed_tokens`, `lm_head`, and stored sinusoid tables.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXSeq2SeqNet: Module {
    @ModuleInfo(key: "shared") var shared: Embedding
    /// Absent for a decoder-only configuration (`encoderLayers` 0, TrOCR), whose memory comes from outside.
    @ModuleInfo(key: "encoder") var encoder: NFKSeq2SeqEncoder?
    @ModuleInfo(key: "decoder") var decoder: NFKSeq2SeqDecoder
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    @ParameterInfo(key: "final_logits_bias") var finalLogitsBias: MLXArray?

    public let configuration: NFKMLXSeq2SeqConfiguration
    // Held behind a plain class so the module's parameter walk does not count the table as a weight.
    private let sinusoids: NFKSeq2SeqSinusoids?

    public init(_ configuration: NFKMLXSeq2SeqConfiguration) {
        self.configuration = configuration
        _shared.wrappedValue = Embedding(embeddingCount: configuration.vocabularySize, dimensions: configuration.dModel)
        if configuration.encoderLayers > 0 {
            _encoder.wrappedValue = NFKSeq2SeqEncoder(configuration)
        }
        _decoder.wrappedValue = NFKSeq2SeqDecoder(configuration)
        if configuration.untiedOutputProjection {
            _lmHead.wrappedValue = Linear(configuration.dModel, configuration.vocabularySize, bias: false)
        }
        if configuration.finalLogitsBias {
            _finalLogitsBias.wrappedValue = MLXArray.zeros([1, configuration.vocabularySize])
        }
        switch configuration.positions {
        case .marianSinusoidal:
            sinusoids = NFKSeq2SeqSinusoids(Self.marianSinusoids(length: configuration.maxPositions, channels: configuration.dModel))
        case .fairseqSinusoidal:
            sinusoids = NFKSeq2SeqSinusoids(Self.fairseqSinusoids(
                length: configuration.maxPositions + configuration.padTokenId + 2,
                channels: configuration.dModel, paddingIndex: configuration.padTokenId))
        case .learned:
            sinusoids = nil
        }
        super.init()
    }

    /// Marian's table: `pos / 10000^(2i/d)`, sines in the first half and cosines in the second.
    static func marianSinusoids(length: Int, channels: Int) -> MLXArray {
        let half = channels / 2
        var values = [Float](repeating: 0, count: length * channels)
        for position in 0 ..< length {
            for i in 0 ..< half {
                let angle = Double(position) / pow(10000, Double(2 * i) / Double(channels))
                values[position * channels + i] = Float(sin(angle))
                values[position * channels + half + i] = Float(cos(angle))
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [length, channels]) }
    }

    /// fairseq's table: frequencies `exp(-i · ln(10000) / (half - 1))`, the padding row zero.
    static func fairseqSinusoids(length: Int, channels: Int, paddingIndex: Int) -> MLXArray {
        let half = channels / 2
        var values = [Float](repeating: 0, count: length * channels)
        let step = log(10000.0) / Double(max(half - 1, 1))
        for position in 0 ..< length where position != paddingIndex {
            for i in 0 ..< half {
                let angle = Double(position) * exp(-Double(i) * step)
                values[position * channels + i] = Float(sin(angle))
                values[position * channels + half + i] = Float(cos(angle))
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [length, channels]) }
    }

    /// The positional term for `length` positions starting at `offset` (positions already consumed).
    private func positions(length: Int, offset: Int, table: Embedding?) -> MLXArray {
        switch configuration.positions {
        case .marianSinusoidal:
            return sinusoids!.table[offset ..< offset + length]
        case .fairseqSinusoidal:
            let start = configuration.padTokenId + 1 + offset
            return sinusoids!.table[start ..< start + length]
        case .learned:
            let ids = MLXArray((offset + 2 ..< offset + 2 + length).map { Int32($0) })
            return table!(ids)
        }
    }

    private var embedScale: Float { configuration.scaleEmbedding ? sqrt(Float(configuration.dModel)) : 1 }

    static func causalMask(_ length: Int, offset: Int = 0) -> MLXArray? {
        guard length > 1 else { return nil }
        var values = [Float](repeating: 0, count: length * (length + offset))
        for i in 0 ..< length {
            for j in (offset + i + 1) ..< (length + offset) {
                values[i * (length + offset) + j] = -1e9
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [length, length + offset]) }
    }

    // MARK: Encoder

    /// Token ids `[B, S]` → the encoder output `[B, S, dModel]`.
    public func encode(_ tokens: MLXArray) -> MLXArray {
        encode(embeddings: shared(tokens) * embedScale)
    }

    /// Already-embedded (and scaled) inputs `[B, S, dModel]` → the encoder output. A multimodal
    /// consumer that concatenates image features with text embeddings enters here.
    public func encode(embeddings: MLXArray) -> MLXArray {
        guard let encoder else {
            preconditionFailure("a decoder-only configuration has no encoder; pass the memory to decode(_:memory:cache:)")
        }
        var h = embeddings + positions(length: embeddings.dim(1), offset: 0, table: encoder.embedPositions)
        if let norm = encoder.layerNormEmbedding { h = norm(h) }
        for layer in encoder.layers { h = layer(h, mask: nil) }
        if let norm = encoder.layerNorm { h = norm(h) }
        return h
    }

    // MARK: Decoder

    /// Decoder token ids `[B, T]` against `memory` → logits `[B, T, vocabulary]`, extending `cache`.
    ///
    /// With a fresh cache `tokens` is the whole prefix; a later call carries only the new tokens, and
    /// the positions continue from the cache's length.
    public func decode(_ tokens: MLXArray, memory: MLXArray, cache: NFKMLXSeq2SeqCache) -> MLXArray {
        let offset = cache.length
        let length = tokens.dim(1)
        var h = shared(tokens) * embedScale + positions(length: length, offset: offset, table: decoder.embedPositions)
        if let norm = decoder.layerNormEmbedding { h = norm(h) }
        let mask = Self.causalMask(length, offset: offset)
        for (index, layer) in decoder.layers.enumerated() {
            h = layer(h, memory: memory, mask: mask, cache: cache, layer: index)
        }
        if let norm = decoder.layerNorm { h = norm(h) }
        var logits = lmHead.map { $0(h) } ?? h.matmul(shared.weight.transposed(1, 0))
        if let bias = finalLogitsBias { logits = logits + bias }
        return logits
    }

    /// Teacher-forced logits for a whole decoder sequence, without a cache.
    public func callAsFunction(source: MLXArray, target: MLXArray) -> MLXArray {
        decode(target, memory: encode(source), cache: NFKMLXSeq2SeqCache(layers: configuration.decoderLayers))
    }

    public func makeCache() -> NFKMLXSeq2SeqCache { NFKMLXSeq2SeqCache(layers: configuration.decoderLayers) }

    // MARK: Weights

    /// Loads a transformers checkpoint (`pytorch_model.bin` or `model.safetensors`).
    public func loadWeights(from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        try load(checkpoint.arrays)
    }

    /// Loads a release directory holding one or more weight files.
    public func loadWeights(fromDirectory directory: URL) throws {
        var arrays = [String: MLXArray]()
        for url in try Self.weightFiles(in: directory) {
            for (key, value) in try NFKMLXWeights.loadCheckpoint(url: url).arrays { arrays[key] = value }
        }
        try load(arrays)
    }

    static func weightFiles(in directory: URL) throws -> [URL] {
        if let files = try? NFKMLXReleaseWeights.files(inDirectory: directory) { return files }
        let bin = directory.appendingPathComponent("pytorch_model.bin")
        guard FileManager.default.fileExists(atPath: bin.path) else {
            throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) holds no model.safetensors or pytorch_model.bin")
        }
        return [bin]
    }

    private func load(_ arrays: [String: MLXArray]) throws {
        var mapped = [String: MLXArray]()
        let hasShared = arrays.keys.contains { Self.moduleKey(for: $0, configuration: configuration, hasShared: true) == "shared.weight" && $0.hasSuffix("shared.weight") }
        for (key, value) in arrays {
            guard let name = Self.moduleKey(for: key, configuration: configuration, hasShared: hasShared) else { continue }
            if mapped[name] != nil { continue }
            mapped[name] = value.asType(.float32)
        }
        try NFKMLXWeights.apply(Array(mapped), to: self)
    }

    /// The module key for a family whose checkpoint carries `shared` and ties its head (Florence-2's
    /// BART), by the positions kind alone.
    static func moduleKey(for key: String, learnedPositions: Bool) -> String? {
        var name = key
        if name.hasPrefix("model.") { name.removeFirst("model.".count) }
        if name.hasSuffix("embed_tokens.weight") || name.hasPrefix("lm_head.") { return nil }
        if name.hasSuffix("embed_positions.weight"), !learnedPositions { return nil }
        return name
    }

    /// The module key a checkpoint key maps to, or nil for a tensor the network derives: the tied
    /// `embed_tokens` and `lm_head` copies, and a stored sinusoid table or its placeholder. A checkpoint with no `shared`
    /// (TrOCR's `decoder.model.decoder.embed_tokens`) supplies `shared` from its first `embed_tokens`;
    /// an untied `output_projection` becomes `lm_head`.
    static func moduleKey(for key: String, configuration: NFKMLXSeq2SeqConfiguration, hasShared: Bool) -> String? {
        var name = key
        for prefix in ["decoder.model.", "model."] where name.hasPrefix(prefix) {
            name.removeFirst(prefix.count)
            break
        }
        if name.hasSuffix("embed_tokens.weight") { return hasShared ? nil : "shared.weight" }
        if name.hasPrefix("lm_head.") || name.hasPrefix("output_projection.") || name.hasPrefix("decoder.output_projection.") {
            return configuration.untiedOutputProjection ? "lm_head.weight" : nil
        }
        if name.hasSuffix("embed_positions.weight"), configuration.positions != .learned { return nil }
        // A sinusoidal release (TrOCR's stage-1 and large-printed) stores fairseq's device-tracking
        // `_float_tensor` placeholder; the table itself is computed.
        if name.hasSuffix("embed_positions._float_tensor") { return nil }
        return name
    }
}

extension NFKMLXSeq2SeqConfiguration {
    func activate(_ x: MLXArray) -> MLXArray {
        switch activation {
        case .swish: return silu(x)
        case .relu: return relu(x)
        case .gelu: return gelu(x)
        }
    }
}
