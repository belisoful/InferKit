//
//  NFKMLXDeBERTaV2.swift
//  InferKitMLX
//
//  The DeBERTa-v2 / v3 encoder (`DebertaV2Model`, Microsoft, MIT). Attention is disentangled: beside
//  the content-to-content scores, each head adds a content-to-position term (a query against the
//  relative-position embedding of each key's offset) and a position-to-content term (each key against
//  the embedding of the query's offset). Offsets are log-bucketed, so nearby tokens keep exact
//  distances and distant ones share buckets, and the position embeddings pass through the same query
//  and key projections as the content (`share_att_key`). No absolute position is added to the input.
//
//  The released v3 geometry is supported: relative attention with both terms, shared projections,
//  a LayerNorm over the relative embeddings, and no token types, no convolution layer, and no
//  embedding projection. A configuration outside that is refused rather than approximated.
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// The geometry of a DeBERTa-v2 / v3 encoder.
public struct NFKMLXDeBERTaV2Configuration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    public var layerNormEpsilon: Float
    /// How many log buckets each side of zero the relative offsets fall into; 0 keeps raw offsets.
    public var positionBuckets: Int
    /// The largest offset the bucketing is scaled to.
    public var maxRelativePositions: Int

    public init(hiddenSize: Int = 1024, layerCount: Int = 24, headCount: Int = 16, intermediateSize: Int = 4096,
                vocabularySize: Int = 128_100, layerNormEpsilon: Float = 1e-7, positionBuckets: Int = 256,
                maxRelativePositions: Int = 512) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.layerNormEpsilon = layerNormEpsilon
        self.positionBuckets = positionBuckets
        self.maxRelativePositions = maxRelativePositions
    }

    /// `microsoft/deberta-v3-large`: 24 layers, 1024 wide, 16 heads.
    public static let v3Large = NFKMLXDeBERTaV2Configuration()

    /// A small configuration for tests, with few enough buckets that the logarithmic range is reached.
    public static let tiny = NFKMLXDeBERTaV2Configuration(hiddenSize: 64, layerCount: 2, headCount: 4,
                                                          intermediateSize: 128, vocabularySize: 512,
                                                          positionBuckets: 8, maxRelativePositions: 32)

    var headDimensions: Int { hiddenSize / headCount }

    /// How many offsets each side of zero the position embeddings cover.
    var attentionSpan: Int { positionBuckets > 0 ? positionBuckets : maxRelativePositions }

    /// Reads a transformers `config.json`, refusing the options this port does not implement.
    public init(configURL: URL) throws {
        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(configURL.lastPathComponent) is not a JSON object")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        let attentionTypes = Set((json["pos_att_type"] as? [String]) ?? [])
        let refusals: [(Bool, String)] = [
            ((json["relative_attention"] as? NSNumber)?.boolValue != true, "relative_attention is off"),
            ((json["share_att_key"] as? NSNumber)?.boolValue != true, "share_att_key is off"),
            (attentionTypes != ["c2p", "p2c"], "pos_att_type is not [c2p, p2c]"),
            ((json["position_biased_input"] as? NSNumber)?.boolValue != false, "position_biased_input is on"),
            (integer("type_vocab_size", 0) != 0, "type_vocab_size is not 0"),
            (integer("conv_kernel_size", 0) != 0, "conv_kernel_size is not 0"),
            ((json["norm_rel_ebd"] as? String)?.contains("layer_norm") != true, "norm_rel_ebd is not layer_norm"),
            (integer("embedding_size", integer("hidden_size", 1024)) != integer("hidden_size", 1024),
             "embedding_size differs from hidden_size"),
        ]
        if let refusal = refusals.first(where: \.0) {
            throw NFKMLXError.unsupportedConfiguration("this port reads the DeBERTa-v3 geometry; \(refusal.1)")
        }
        let maxPositions = integer("max_position_embeddings", 512)
        let relative = integer("max_relative_positions", -1)
        self.init(hiddenSize: integer("hidden_size", 1024), layerCount: integer("num_hidden_layers", 24),
                  headCount: integer("num_attention_heads", 16), intermediateSize: integer("intermediate_size", 4096),
                  vocabularySize: integer("vocab_size", 128_100),
                  layerNormEpsilon: (json["layer_norm_eps"] as? NSNumber)?.floatValue ?? 1e-7,
                  positionBuckets: integer("position_buckets", -1),
                  maxRelativePositions: relative < 1 ? maxPositions : relative)
    }

    /// The bucketed offset `query - key` for every pair of positions, row-major by query. Nearby
    /// offsets stay exact; beyond half the bucket count they grow logarithmically up to the span.
    ///
    /// The arithmetic is float32 throughout, as the reference computes it, because the ceiling is
    /// taken on a float32 logarithm and a wider type moves a few offsets to the neighboring bucket.
    public func relativePositions(length: Int) -> [Int32] {
        var table = [Int32](repeating: 0, count: length * length)
        let middle = positionBuckets / 2
        let buckets = positionBuckets > 0 && maxRelativePositions > 0
        let logRange = Foundation.log(Float(maxRelativePositions - 1) / Float(middle))
        for query in 0 ..< length {
            for key in 0 ..< length {
                let offset = query - key
                table[query * length + key] = Int32(buckets ? Self.bucket(offset, middle: middle, logRange: logRange) : offset)
            }
        }
        return table
    }

    private static func bucket(_ offset: Int, middle: Int, logRange: Float) -> Int {
        let magnitude = offset < middle && offset > -middle ? middle - 1 : abs(offset)
        if magnitude <= middle { return offset }
        let scaled = Foundation.log(Float(magnitude) / Float(middle)) / logRange * Float(middle - 1)
        return (Int(scaled.rounded(.up)) + middle) * (offset < 0 ? -1 : 1)
    }
}

/// Where each pair of positions reads the relative-position scores, computed once per length.
struct NFKDeBERTaPositions {
    /// `[length, length]`: the column of a query's content-to-position scores each key reads.
    let contentToPosition: MLXArray
    /// `[length, length]`: the column of a key's position-to-content scores each query reads.
    let positionToContent: MLXArray

    init(configuration c: NFKMLXDeBERTaV2Configuration, length: Int) {
        let span = c.attentionSpan
        let relative = c.relativePositions(length: length)
        let clamp = { (value: Int32) in Int32(Swift.min(Swift.max(Int(value), 0), 2 * span - 1)) }
        contentToPosition = MLXArray(relative.map { clamp($0 + Int32(span)) }, [length, length])
        positionToContent = MLXArray(relative.map { clamp(Int32(span) - $0) }, [length, length])
    }
}

final class NFKDeBERTaSelfAttention: Module {
    @ModuleInfo(key: "query_proj") var query: Linear
    @ModuleInfo(key: "key_proj") var key: Linear
    @ModuleInfo(key: "value_proj") var value: Linear

    let configuration: NFKMLXDeBERTaV2Configuration

    init(_ c: NFKMLXDeBERTaV2Configuration) {
        configuration = c
        _query.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _key.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _value.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, positions: NFKDeBERTaPositions,
                        relativeEmbeddings: MLXArray) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.shape[0], x.shape[1])
        let heads = c.headCount
        let width = c.headDimensions
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([batch, length, heads, width]).transposed(0, 2, 1, 3) }
        func splitPositions(_ t: MLXArray) -> MLXArray {
            t.reshaped([-1, heads, width]).transposed(1, 0, 2)                 // [heads, 2·span, width]
        }

        let queries = split(query(x))
        let keys = split(key(x))
        let values = split(value(x))

        // Three score terms share one scale: content-to-content, content-to-position, position-to-content.
        let scale = MLXArray(sqrt(Float(width) * 3))
        var scores = matmul(queries, (keys / scale).transposed(0, 1, 3, 2))

        let positionKeys = splitPositions(key(relativeEmbeddings))
        let positionQueries = splitPositions(query(relativeEmbeddings))
        let span = [batch, heads, length, length]

        let contentToPosition = takeAlong(matmul(queries, positionKeys.transposed(0, 2, 1)),
                                          broadcast(positions.contentToPosition, to: span), axis: -1)
        // Gathered per key along its own offsets, then transposed so the rows are queries again.
        let positionToContent = takeAlong(matmul(keys, positionQueries.transposed(0, 2, 1)),
                                          broadcast(positions.positionToContent, to: span), axis: -1)
            .transposed(0, 1, 3, 2)
        scores = scores + (contentToPosition / scale + positionToContent / scale)

        if let mask {
            scores = which(mask, scores, MLXArray(-Float.greatestFiniteMagnitude))
        }
        let weights = softmax(scores, axis: -1, precise: true)
        return matmul(weights, values).transposed(0, 2, 1, 3).reshaped([batch, length, c.hiddenSize])
    }
}

/// A dense projection added back to its input, then normalized.
final class NFKDeBERTaResidualOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(inputs: Int, _ c: NFKMLXDeBERTaV2Configuration) {
        _dense.wrappedValue = Linear(inputs, c.hiddenSize)
        _norm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, residual: MLXArray) -> MLXArray { norm(dense(x) + residual) }
}

final class NFKDeBERTaAttention: Module {
    @ModuleInfo(key: "self") var attention: NFKDeBERTaSelfAttention
    @ModuleInfo(key: "output") var output: NFKDeBERTaResidualOutput

    init(_ c: NFKMLXDeBERTaV2Configuration) {
        _attention.wrappedValue = NFKDeBERTaSelfAttention(c)
        _output.wrappedValue = NFKDeBERTaResidualOutput(inputs: c.hiddenSize, c)
        super.init()
    }
}

final class NFKDeBERTaIntermediate: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(_ c: NFKMLXDeBERTaV2Configuration) {
        _dense.wrappedValue = Linear(c.hiddenSize, c.intermediateSize)
        super.init()
    }
}

final class NFKDeBERTaLayer: Module {
    @ModuleInfo(key: "attention") var attention: NFKDeBERTaAttention
    @ModuleInfo(key: "intermediate") var intermediate: NFKDeBERTaIntermediate
    @ModuleInfo(key: "output") var output: NFKDeBERTaResidualOutput

    init(_ c: NFKMLXDeBERTaV2Configuration) {
        _attention.wrappedValue = NFKDeBERTaAttention(c)
        _intermediate.wrappedValue = NFKDeBERTaIntermediate(c)
        _output.wrappedValue = NFKDeBERTaResidualOutput(inputs: c.intermediateSize, c)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, positions: NFKDeBERTaPositions,
                        relativeEmbeddings: MLXArray) -> MLXArray {
        let attended = attention.output(attention.attention(x, mask: mask, positions: positions,
                                                            relativeEmbeddings: relativeEmbeddings),
                                        residual: x)
        return output(gelu(intermediate.dense(attended)), residual: attended)
    }
}

final class NFKDeBERTaEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var words: Embedding
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ c: NFKMLXDeBERTaV2Configuration) {
        _words.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _norm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        super.init()
    }
}

final class NFKDeBERTaEncoder: Module {
    @ModuleInfo(key: "layer") var layers: [NFKDeBERTaLayer]
    @ModuleInfo(key: "rel_embeddings") var relativeEmbeddings: Embedding
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ c: NFKMLXDeBERTaV2Configuration) {
        _layers.wrappedValue = (0 ..< c.layerCount).map { _ in NFKDeBERTaLayer(c) }
        _relativeEmbeddings.wrappedValue = Embedding(embeddingCount: 2 * c.attentionSpan, dimensions: c.hiddenSize)
        _norm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        super.init()
    }
}

/// The DeBERTa-v2 / v3 encoder, under the keys `DebertaV2Model` saves.
public final class NFKMLXDeBERTaV2Net: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKDeBERTaEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKDeBERTaEncoder

    public let configuration: NFKMLXDeBERTaV2Configuration

    public init(_ c: NFKMLXDeBERTaV2Configuration) {
        configuration = c
        _embeddings.wrappedValue = NFKDeBERTaEmbeddings(c)
        _encoder.wrappedValue = NFKDeBERTaEncoder(c)
        super.init()
    }

    /// The last layer's states, `[batch, length, hidden]`. `attentionMask` is `[batch, length]`, 1 for
    /// a token and 0 for padding; nil reads every token.
    public func callAsFunction(_ tokens: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        hiddenStates(tokens, attentionMask: attentionMask).last!
    }

    /// The embedding output, then every layer's output: the reference's `hidden_states`.
    public func hiddenStates(_ tokens: MLXArray, attentionMask: MLXArray? = nil) -> [MLXArray] {
        let length = tokens.shape[1]
        var hidden = embeddings.norm(embeddings.words(tokens))
        var pairMask: MLXArray?
        if let attentionMask {
            // Padding is zeroed after the embedding norm, and a pair attends only when both are tokens.
            let valid = attentionMask.asType(.float32)
            hidden = hidden * valid.expandedDimensions(axis: -1)
            let pairs = valid.expandedDimensions(axes: [1, 2]) * valid.expandedDimensions(axes: [1, 3])
            pairMask = pairs .> 0
        }
        let positions = NFKDeBERTaPositions(configuration: configuration, length: length)
        let relative = encoder.norm(encoder.relativeEmbeddings.weight)
        var states = [hidden]
        for layer in encoder.layers {
            hidden = layer(hidden, mask: pairMask, positions: positions, relativeEmbeddings: relative)
            states.append(hidden)
        }
        return states
    }
}
