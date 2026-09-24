//
//  NFKMLXMADLAD.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// MADLAD-400 MT (Google, Apache-2.0): a T5 encoder-decoder over a 256k SentencePiece vocabulary that
// translates between 400+ languages, the target named by a `<2xx>` token at the front of the source.
// The encoder is the T5 stack the LTX text encoder already ports; this file adds the decoder (a causal
// relative-position bias, a cross-attention sublayer, and a cache), the untied output head, and the
// tokenizer rules of the release's fast tokenizer.

/// T5 geometry for the MADLAD-400 releases.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXMADLADConfiguration: Sendable {
    public var encoder: NFKMLXT5Configuration
    public var decoderLayers: Int
    /// Reads `lm_head` from the checkpoint rather than the shared embedding.
    public var untiedHead: Bool
    public var padTokenId: Int
    public var eosTokenId: Int
    public var decoderStartTokenId: Int

    public init(encoder: NFKMLXT5Configuration, decoderLayers: Int, untiedHead: Bool = true,
                padTokenId: Int = 1, eosTokenId: Int = 2, decoderStartTokenId: Int = 0) {
        self.encoder = encoder
        self.decoderLayers = decoderLayers
        self.untiedHead = untiedHead
        self.padTokenId = padTokenId
        self.eosTokenId = eosTokenId
        self.decoderStartTokenId = decoderStartTokenId
    }

    /// `google/madlad400-3b-mt`: 32 + 32 layers at 1024, 16 heads of 128, 8192-wide gated FFN.
    public static let mt3B = NFKMLXMADLADConfiguration(
        encoder: NFKMLXT5Configuration(dModel: 1024, layers: 32, heads: 16, keyDim: 128, ffDim: 8192,
                                       vocabularySize: 256000), decoderLayers: 32)

    /// `google/madlad400-7b-mt`: 48 + 48 layers at 2048, 16 heads of 128, 8192-wide gated FFN.
    public static let mt7B = NFKMLXMADLADConfiguration(
        encoder: NFKMLXT5Configuration(dModel: 2048, layers: 48, heads: 16, keyDim: 128, ffDim: 8192,
                                       vocabularySize: 256000), decoderLayers: 48)

    /// A small geometry for tests.
    public static let tiny = NFKMLXMADLADConfiguration(encoder: .tiny, decoderLayers: 2)

    /// The geometry a release's `config.json` declares.
    public init(huggingFaceConfigURL url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        let encoder = NFKMLXT5Configuration(
            dModel: int("d_model", 1024), layers: int("num_layers", 32), heads: int("num_heads", 16),
            keyDim: int("d_kv", 128), ffDim: int("d_ff", 8192), vocabularySize: int("vocab_size", 256000),
            relativeBuckets: int("relative_attention_num_buckets", 32),
            relativeMaxDistance: int("relative_attention_max_distance", 128),
            layerNormEps: (json["layer_norm_epsilon"] as? NSNumber)?.floatValue ?? 1e-6)
        self.init(encoder: encoder, decoderLayers: int("num_decoder_layers", int("num_layers", 32)),
                  untiedHead: !((json["tie_word_embeddings"] as? Bool) ?? false),
                  padTokenId: int("pad_token_id", 1), eosTokenId: int("eos_token_id", 2),
                  decoderStartTokenId: int("decoder_start_token_id", 0))
    }
}

// MARK: - Decoder modules

/// The decoder's self-attention: T5 attention with a causal relative-position bias and a cache.
final class NFKT5CachedAttention: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding?
    let configuration: NFKMLXT5Configuration

    init(_ c: NFKMLXT5Configuration, hasBias: Bool) {
        configuration = c
        _q.wrappedValue = Linear(c.dModel, c.inner, bias: false)
        _k.wrappedValue = Linear(c.dModel, c.inner, bias: false)
        _v.wrappedValue = Linear(c.dModel, c.inner, bias: false)
        _o.wrappedValue = Linear(c.inner, c.dModel, bias: false)
        if hasBias {
            _relativeAttentionBias.wrappedValue = Embedding(embeddingCount: c.relativeBuckets, dimensions: c.heads)
        }
    }

    private func split(_ t: MLXArray) -> MLXArray {
        t.reshaped([t.dim(0), t.dim(1), configuration.heads, configuration.keyDim]).transposed(0, 2, 1, 3)
    }

    private func merge(_ t: MLXArray) -> MLXArray {
        o(t.transposed(0, 2, 1, 3).reshaped([t.dim(0), t.dim(2), configuration.inner]))
    }

    /// Self-attention over `x` at positions `offset...`, extending the cached keys and values.
    func callAsFunction(_ x: MLXArray, bias: MLXArray, cachedKeys: inout MLXArray?, cachedValues: inout MLXArray?) -> MLXArray {
        var keys = split(k(x))
        var values = split(v(x))
        if let previousKeys = cachedKeys, let previousValues = cachedValues {
            keys = concatenated([previousKeys, keys], axis: 2)
            values = concatenated([previousValues, values], axis: 2)
        }
        cachedKeys = keys
        cachedValues = values
        return merge(MLXFast.scaledDotProductAttention(queries: split(q(x)), keys: keys, values: values, scale: 1, mask: bias))
    }

    /// Cross-attention to `memory`, whose projections are computed once per cache.
    func callAsFunction(_ x: MLXArray, memory: MLXArray, cachedKeys: inout MLXArray?, cachedValues: inout MLXArray?) -> MLXArray {
        if cachedKeys == nil || cachedValues == nil {
            cachedKeys = split(k(memory))
            cachedValues = split(v(memory))
        }
        return merge(MLXFast.scaledDotProductAttention(queries: split(q(x)), keys: cachedKeys!, values: cachedValues!, scale: 1, mask: nil))
    }

    /// The causal bias `[1, heads, T, offset + T]` for queries at `offset ..< offset + T` over every
    /// key so far: the unidirectional bucketing of the reference's decoder, and `-inf` ahead of each query.
    func causalBias(queryLength: Int, offset: Int) -> MLXArray {
        let keyLength = offset + queryLength
        var buckets = [Int32](repeating: 0, count: queryLength * keyLength)
        var mask = [Float](repeating: 0, count: queryLength * keyLength)
        for i in 0 ..< queryLength {
            for j in 0 ..< keyLength {
                let relative = j - (offset + i)
                buckets[i * keyLength + j] = Int32(Self.causalBucket(relative, numBuckets: configuration.relativeBuckets,
                                                                     maxDistance: configuration.relativeMaxDistance))
                if relative > 0 { mask[i * keyLength + j] = -1e9 }
            }
        }
        let indices = buckets.withUnsafeBufferPointer { MLXArray($0, [queryLength, keyLength]) }
        let additive = mask.withUnsafeBufferPointer { MLXArray($0, [queryLength, keyLength]) }
        let values = relativeAttentionBias!(indices).transposed(2, 0, 1).expandedDimensions(axis: 0)
        return values + additive
    }

    /// The Mesh-TensorFlow bucketing with `bidirectional=False`: only distances into the past count,
    /// and every bucket serves them.
    static func causalBucket(_ relativePosition: Int, numBuckets: Int, maxDistance: Int) -> Int {
        let distance = max(-relativePosition, 0)
        let maxExact = numBuckets / 2
        if distance < maxExact {
            return distance
        }
        let large = maxExact + Int(log(Double(distance) / Double(maxExact))
            / log(Double(maxDistance) / Double(maxExact)) * Double(numBuckets - maxExact))
        return min(large, numBuckets - 1)
    }
}

final class NFKT5DecoderSelfAttentionLayer: Module {
    @ModuleInfo(key: "SelfAttention") var attention: NFKT5CachedAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: NFKT5LayerNorm

    init(_ c: NFKMLXT5Configuration, hasBias: Bool) {
        _attention.wrappedValue = NFKT5CachedAttention(c, hasBias: hasBias)
        _layerNorm.wrappedValue = NFKT5LayerNorm(c.dModel, eps: c.layerNormEps)
    }
}

final class NFKT5CrossAttentionLayer: Module {
    @ModuleInfo(key: "EncDecAttention") var attention: NFKT5CachedAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: NFKT5LayerNorm

    init(_ c: NFKMLXT5Configuration) {
        _attention.wrappedValue = NFKT5CachedAttention(c, hasBias: false)
        _layerNorm.wrappedValue = NFKT5LayerNorm(c.dModel, eps: c.layerNormEps)
    }
}

/// A decoder block: `layer = [self-attention, cross-attention, feed-forward]`.
final class NFKT5DecoderBlock: Module {
    @ModuleInfo(key: "layer") var layer: [Module]

    init(_ c: NFKMLXT5Configuration, hasBias: Bool) {
        _layer.wrappedValue = [NFKT5DecoderSelfAttentionLayer(c, hasBias: hasBias),
                               NFKT5CrossAttentionLayer(c), NFKT5FeedForwardLayer(c)]
    }

    var selfAttention: NFKT5DecoderSelfAttentionLayer { layer[0] as! NFKT5DecoderSelfAttentionLayer }
    var crossAttention: NFKT5CrossAttentionLayer { layer[1] as! NFKT5CrossAttentionLayer }
    var feedForward: NFKT5FeedForwardLayer { layer[2] as! NFKT5FeedForwardLayer }

    func callAsFunction(_ x: MLXArray, memory: MLXArray, bias: MLXArray, cache: NFKMLXSeq2SeqCache, index: Int) -> MLXArray {
        var h = x
        h = h + selfAttention.attention(selfAttention.layerNorm(h), bias: bias,
                                        cachedKeys: &cache.selfKeys[index], cachedValues: &cache.selfValues[index])
        h = h + crossAttention.attention(crossAttention.layerNorm(h), memory: memory,
                                         cachedKeys: &cache.crossKeys[index], cachedValues: &cache.crossValues[index])
        return feedForward(h)
    }
}

final class NFKT5DecoderStack: Module {
    @ModuleInfo(key: "block") var block: [NFKT5DecoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: NFKT5LayerNorm

    init(_ c: NFKMLXT5Configuration, layers: Int) {
        _block.wrappedValue = (0 ..< layers).map { NFKT5DecoderBlock(c, hasBias: $0 == 0) }
        _finalLayerNorm.wrappedValue = NFKT5LayerNorm(c.dModel, eps: c.layerNormEps)
    }
}

// MARK: - Network

/// The T5 encoder-decoder (`T5ForConditionalGeneration`) as MADLAD-400 uses it.
///
/// @discussion Module keys mirror the checkpoint: `shared`, `encoder.block.N.layer.{0,1}`,
/// `encoder.final_layer_norm`, `decoder.block.N.layer.{0,1,2}` (self-attention, `EncDecAttention`,
/// feed-forward), `decoder.final_layer_norm`, and `lm_head`. A tied release omits `lm_head` and
/// projects through the embedding scaled by `dModel^-0.5`, as the reference does.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXT5Seq2SeqNet: Module {
    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "encoder") var encoder: NFKT5Stack
    @ModuleInfo(key: "decoder") var decoder: NFKT5DecoderStack
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let configuration: NFKMLXMADLADConfiguration

    public init(_ configuration: NFKMLXMADLADConfiguration) {
        self.configuration = configuration
        let c = configuration.encoder
        _shared.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.dModel)
        _encoder.wrappedValue = NFKT5Stack(c)
        _decoder.wrappedValue = NFKT5DecoderStack(c, layers: configuration.decoderLayers)
        if configuration.untiedHead {
            _lmHead.wrappedValue = Linear(c.dModel, c.vocabularySize, bias: false)
        }
        super.init()
    }

    /// Token ids `[B, S]` → the encoder output `[B, S, dModel]`.
    public func encode(_ tokens: MLXArray) -> MLXArray {
        var hidden = shared(tokens)
        let bias = encoder.block[0].selfAttention.attention.computeBias(tokens.dim(1))
        for block in encoder.block {
            hidden = block(hidden, bias: bias)
        }
        return encoder.finalLayerNorm(hidden)
    }

    /// Decoder ids `[B, T]` against `memory` → logits `[B, T, vocabulary]`, extending `cache`.
    public func decode(_ tokens: MLXArray, memory: MLXArray, cache: NFKMLXSeq2SeqCache) -> MLXArray {
        let offset = cache.length
        var hidden = shared(tokens)
        let bias = decoder.block[0].selfAttention.attention.causalBias(queryLength: tokens.dim(1), offset: offset)
        for (index, block) in decoder.block.enumerated() {
            hidden = block(hidden, memory: memory, bias: bias, cache: cache, index: index)
        }
        hidden = decoder.finalLayerNorm(hidden)
        if let lmHead {
            return lmHead(hidden)
        }
        return (hidden * pow(Float(configuration.encoder.dModel), -0.5)).matmul(shared.weight.transposed(1, 0))
    }

    /// Teacher-forced logits for a whole decoder sequence.
    public func callAsFunction(source: MLXArray, target: MLXArray) -> MLXArray {
        decode(target, memory: encode(source), cache: makeCache())
    }

    public func makeCache() -> NFKMLXSeq2SeqCache { NFKMLXSeq2SeqCache(layers: configuration.decoderLayers) }

    /// Loads a release directory (`model.safetensors` or shards). `half` casts the weights to
    /// `bfloat16` once loaded, which halves what the 3B release holds resident; the reference runs
    /// float32, which is what a parity measurement uses.
    public func loadWeights(fromDirectory directory: URL, half: Bool = false) throws {
        try load(try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: .checkpoint), half: half)
    }

    /// Loads one checkpoint file, such as one ``NFKMLXWeights/save(_:to:)`` wrote after a fine-tune.
    public func loadWeights(from url: URL, half: Bool = false) throws {
        try load(Array(try NFKMLXWeights.loadCheckpoint(url: url).arrays), half: half)
    }

    /// A release stores the shared embedding under `shared.weight`, or only as an `embed_tokens`
    /// copy on one or both stacks (the 3B release keeps the decoder's alone); the first one seen
    /// becomes `shared` and the rest are dropped.
    private func load(_ arrays: [(String, MLXArray)], half: Bool) throws {
        var mapped = [String: MLXArray]()
        for (key, value) in arrays {
            let name = key.hasSuffix("embed_tokens.weight") ? "shared.weight" : key
            if name == "shared.weight", mapped[name] != nil { continue }
            mapped[name] = half ? value.asType(.bfloat16) : value.asType(.float32)
        }
        try NFKMLXWeights.apply(Array(mapped), to: self)
    }
}

extension NFKMLXT5Seq2SeqNet: NFKMLXSeq2SeqDecodable {
    public func encodeSource(_ tokens: MLXArray) -> MLXArray { encode(tokens) }
    public func makeDecodingCache() -> NFKMLXSeq2SeqCache { makeCache() }
    public func decodeStep(_ tokens: MLXArray, memory: MLXArray, cache: NFKMLXSeq2SeqCache) -> MLXArray {
        decode(tokens, memory: memory, cache: cache)
    }
    public func reorderCache(_ cache: NFKMLXSeq2SeqCache, rows: MLXArray) { cache.reorder(rows) }
}

// MARK: - Translator

/// A loaded MADLAD-400 release.
///
/// @discussion Tokenization follows the release's fast tokenizer: runs of two or more spaces collapse
/// to one, the whole `<2xx> text` string is prefixed with `▁` and segmented by the unigram model with
/// the `<2xx>` markers as pieces, and the end token closes it. No byte fallback, as that tokenizer
/// declares none.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXMADLADTranslator: NFKMLXTranslator {
    public let net: NFKMLXT5Seq2SeqNet
    public let segmenter: NFKMLXSentencePieceSegmenter
    /// The `<2xx>` marker ids by the code inside the brackets.
    public let targetCodes: [String: Int]
    public let identifier: String
    public var defaultDecoding: NFKMLXSeq2SeqDecoding

    init(net: NFKMLXT5Seq2SeqNet, segmenter: NFKMLXSentencePieceSegmenter, identifier: String) {
        self.net = net
        self.segmenter = segmenter
        self.identifier = identifier
        var codes = [String: Int]()
        for (index, piece) in segmenter.model.pieces.enumerated()
        where piece.type == .userDefined && piece.text.hasPrefix("<2") && piece.text.hasSuffix(">") {
            codes[String(piece.text.dropFirst(2).dropLast())] = index
        }
        targetCodes = codes
        defaultDecoding = NFKMLXSeq2SeqDecoding(beams: 1, maxTokens: 512, startToken: net.configuration.decoderStartTokenId,
                                                endToken: net.configuration.eosTokenId)
    }

    public var fixedSourceLanguage: String? { nil }
    public var fixedTargetLanguage: String? { nil }

    /// The marker id for a BCP-47 tag: the primary subtag with its script where the release
    /// distinguishes one (`zh-Hant` → `<2zh_Hant>`), or nil when the release has no such target.
    public func targetCode(for language: String) -> Int? {
        let primary = NFKMLXTranslationBackend.primary(language)
        let aliases = ["nb": "no", "nn": "no", "tl": "fil", "iw": "he", "jw": "jv"]
        var candidates = [String]()
        if let script = NFKMLXTranslationBackend.script(language) {
            candidates.append("\(primary)_\(script)")
        }
        candidates.append(primary)
        if let alias = aliases[primary] { candidates.append(alias) }
        if let three = NFKMLXLanguageCodes.iso639_3[primary] { candidates.append(three) }
        return candidates.lazy.compactMap { self.targetCodes[$0] }.first
    }

    /// The source reads any language; the target must have a marker.
    public func supports(language: String) -> Bool { targetCode(for: language) != nil }

    /// The source ids: `▁`, the target marker, the pieces of the text, and the end token.
    public func sourceIds(for text: String, target: String) -> [Int]? {
        guard let code = targetCode(for: target), let marker = segmenter.piece(at: code) else { return nil }
        let collapsed = (marker + " " + text).replacingOccurrences(of: " {2,}", with: " ", options: .regularExpression)
        return segmenter.encode(collapsed, dummyPrefix: true) + [net.configuration.eosTokenId]
    }

    public func translate(_ text: String, from source: String?, to target: String,
                          decoding: NFKMLXSeq2SeqDecoding) throws -> String {
        guard let ids = sourceIds(for: text, target: target) else {
            throw NFKMLXTranslationBackend.error(.error_InferenceUnsupported, "\(identifier) does not translate into \(target)")
        }
        let generated = NFKMLXSeq2SeqDecoder.generate(net, source: ids, decoding: decoding)
        return segmenter.decode(generated)
    }
}

/// Registration, download, and construction of MADLAD-400 translation backends.
///
/// @discussion A release directory holds `config.json`, `spiece.model`, and `model.safetensors`.
/// The 3B release is 11.8 GB of float32 on disk; `half` loads it as bfloat16.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXMADLAD)
public final class NFKMLXMADLAD: NSObject {
    /// The registered name.
    @objc public static let modelName = "madlad400-3b-mt"
    static let requiredFiles = ["config.json", "spiece.model"]
    static let optionalFiles = ["generation_config.json"]
    static let weightFiles = ["model.safetensors"]

    /// Loads a release directory as a translator.
    public static func translator(directoryURL directory: URL, half: Bool = false) throws -> NFKMLXMADLADTranslator {
        let configuration = try NFKMLXMADLADConfiguration(huggingFaceConfigURL: directory.appendingPathComponent("config.json"))
        let net = try network(directoryURL: directory, configuration: configuration, half: half)
        return try translator(net: net, directoryURL: directory)
    }

    /// Wraps a network with the release's tokenizer.
    public static func translator(net: NFKMLXT5Seq2SeqNet, directoryURL directory: URL) throws -> NFKMLXMADLADTranslator {
        var model = try NFKMLXSentencePieceModel(contentsOf: directory.appendingPathComponent("spiece.model"))
        model.byteFallback = false
        return NFKMLXMADLADTranslator(net: net, segmenter: NFKMLXSentencePieceSegmenter(model: model), identifier: modelName)
    }

    /// Builds the network alone, ready to adapt.
    public static func network(directoryURL directory: URL?, configuration: NFKMLXMADLADConfiguration = .tiny,
                               half: Bool = false) throws -> NFKMLXT5Seq2SeqNet {
        let net = NFKMLXT5Seq2SeqNet(configuration)
        if let directory {
            try net.loadWeights(fromDirectory: directory, half: half)
        }
        return net
    }

    /// Builds the backend from a release directory at float32.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, half: false)
    }

    /// Builds the backend from a release directory, at bfloat16 when `half` is set.
    @objc(backendWithDirectoryURL:halfPrecision:error:)
    public static func backend(directoryURL: URL, half: Bool) throws -> any NFKInferenceBackend {
        NFKMLXTranslationBackend(translator: try translator(directoryURL: directoryURL, half: half))
    }

    /// Downloads a release (`google/madlad400-3b-mt`) and builds the backend. Blocking on the network.
    @objc(backendWithRepo:revision:cacheDirectoryURL:halfPrecision:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?, half: Bool) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles), half: half)
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:half:)``.
    @objc(backendWithRepo:revision:cacheDirectoryURL:halfPrecision:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?, half: Bool,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) { try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, half: half) }
    }

    /// Registers `madlad400-3b-mt` with `NFKMLXModelRegistry`; the registry's URL is the release
    /// directory, loaded at bfloat16.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { url in
            guard let url else { throw NFKMLXError.unsupportedConfiguration("madlad400-3b-mt builds from a release directory, not without weights") }
            return try backend(directoryURL: url, half: true)
        }
    }
}
