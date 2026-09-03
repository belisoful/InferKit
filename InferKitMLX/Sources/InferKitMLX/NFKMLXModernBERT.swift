//
//  NFKMLXModernBERT.swift
//  InferKitMLX
//
//  A ModernBERT cross-encoder reranker: the third piece of the retrieval story after the two embedders.
//  An embedder scores a query and a document independently and compares the vectors; a cross-encoder
//  reads the pair together — `[CLS] query [SEP] document [SEP]` — through a bidirectional encoder and
//  predicts one relevance score, which is more accurate and is what reranks an embedder's shortlist.
//
//  ModernBERT is a modernized BERT encoder: rotary position embeddings (a global base every third
//  layer, a smaller local base with a sliding window elsewhere), a GeGLU feed-forward, LayerNorm
//  throughout with no biases, and no absolute position embeddings. The reranker adds a prediction head
//  and a single-logit classifier over the mean-pooled sequence.
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// The geometry of a ModernBERT encoder and its reranker head.
public struct NFKMLXModernBertConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    /// Every `globalAttentionEvery`th layer (counting from 0) attends globally; the rest are local.
    public var globalAttentionEvery: Int
    /// Rotary base for the global-attention layers.
    public var globalRopeTheta: Float
    /// Rotary base for the local-attention layers.
    public var localRopeTheta: Float
    /// The local layers' attention window, total width; a token attends to `localAttention / 2` either side.
    public var localAttention: Int
    public var normEpsilon: Float
    /// The classification tokens the pair is wrapped in.
    public var clsToken: Int
    public var sepToken: Int

    public init(hiddenSize: Int = 768, layerCount: Int = 22, headCount: Int = 12,
                intermediateSize: Int = 1152, vocabularySize: Int = 50_368,
                globalAttentionEvery: Int = 3, globalRopeTheta: Float = 160_000,
                localRopeTheta: Float = 10_000, localAttention: Int = 128, normEpsilon: Float = 1e-5,
                clsToken: Int = 50_281, sepToken: Int = 50_282) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.globalAttentionEvery = globalAttentionEvery
        self.globalRopeTheta = globalRopeTheta
        self.localRopeTheta = localRopeTheta
        self.localAttention = localAttention
        self.normEpsilon = normEpsilon
        self.clsToken = clsToken
        self.sepToken = sepToken
    }

    var headDimensions: Int { hiddenSize / headCount }

    /// The released `gte-reranker-modernbert-base` geometry.
    public static let gteReranker = NFKMLXModernBertConfiguration()

    /// A small configuration for tests. Keeps the every-third-layer global pattern and a tiny window so
    /// the sliding path is exercised.
    public static let tiny = NFKMLXModernBertConfiguration(
        hiddenSize: 64, layerCount: 4, headCount: 2, intermediateSize: 128, vocabularySize: 512,
        globalAttentionEvery: 3, globalRopeTheta: 160_000, localRopeTheta: 10_000, localAttention: 4)

    func isGlobal(layer index: Int) -> Bool { index % globalAttentionEvery == 0 }
}

/// LayerNorm with a learned scale and no bias, which is how ModernBERT normalizes (`norm_bias` false).
final class NFKModernBertNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        epsilon = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let centered = x - mean
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        return centered * rsqrt(variance + epsilon) * weight
    }
}

/// ModernBERT attention: one fused query-key-value projection, rotary embeddings at the layer's own
/// base, and a bidirectional window on the local layers. Bias-free.
final class NFKModernBertAttention: Module {
    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    let heads: Int
    let headDimensions: Int
    let scale: Float
    let base: Float
    let global: Bool

    init(_ c: NFKMLXModernBertConfiguration, global: Bool) {
        heads = c.headCount
        headDimensions = c.headDimensions
        scale = 1 / sqrt(Float(c.headDimensions))
        base = global ? c.globalRopeTheta : c.localRopeTheta
        self.global = global
        _wqkv.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: false)
        _wo.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, globalMask: MLXArray?, localMask: MLXArray?) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        // Wqkv → [batch, length, 3, heads, headDim] → [batch, heads, 3, length, headDim], then unbind.
        let qkv = wqkv(x).reshaped([batch, length, 3, heads, headDimensions]).transposed(0, 3, 2, 1, 4)
        var queries = qkv[0..., 0..., 0]
        var keys = qkv[0..., 0..., 1]
        let values = qkv[0..., 0..., 2]

        queries = MLXFast.RoPE(queries, dimensions: headDimensions, traditional: false, base: base, scale: 1, offset: 0)
        keys = MLXFast.RoPE(keys, dimensions: headDimensions, traditional: false, base: base, scale: 1, offset: 0)

        let mask = global ? globalMask : localMask
        let attention = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: mask.map { $0.asType(queries.dtype) })
        return wo(attention.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDimensions]))
    }
}

/// ModernBERT's GeGLU feed-forward: one projection to twice the intermediate width, split into an input
/// and a gate, the input's exact GELU times the gate, projected back down. Bias-free.
final class NFKModernBertMLP: Module {
    @ModuleInfo(key: "Wi") var wi: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(_ c: NFKMLXModernBertConfiguration) {
        _wi.wrappedValue = Linear(c.hiddenSize, 2 * c.intermediateSize, bias: false)
        _wo.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = wi(x)
        let parts = projected.split(parts: 2, axis: -1)
        return wo(gelu(parts[0]) * parts[1])
    }
}

/// One ModernBERT block: pre-normalized attention and feed-forward, each added back. The first layer's
/// `attn_norm` is the identity, because the embeddings are already normalized — the checkpoint carries
/// no weight for it, so it is absent here too.
final class NFKModernBertBlock: Module {
    @ModuleInfo(key: "attn_norm") var attentionNorm: NFKModernBertNorm?
    @ModuleInfo(key: "attn") var attention: NFKModernBertAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: NFKModernBertNorm
    @ModuleInfo(key: "mlp") var mlp: NFKModernBertMLP

    init(_ c: NFKMLXModernBertConfiguration, layer index: Int) {
        _attentionNorm.wrappedValue = index == 0 ? nil : NFKModernBertNorm(dimensions: c.hiddenSize, eps: c.normEpsilon)
        _attention.wrappedValue = NFKModernBertAttention(c, global: c.isGlobal(layer: index))
        _mlpNorm.wrappedValue = NFKModernBertNorm(dimensions: c.hiddenSize, eps: c.normEpsilon)
        _mlp.wrappedValue = NFKModernBertMLP(c)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, globalMask: MLXArray?, localMask: MLXArray?) -> MLXArray {
        let normed = attentionNorm.map { $0(x) } ?? x
        let attended = x + attention(normed, globalMask: globalMask, localMask: localMask)
        return attended + mlp(mlpNorm(attended))
    }
}

/// The token embedding and its normalization. ModernBERT carries no positional embedding; the rotary
/// embeddings supply position.
final class NFKModernBertEmbeddings: Module {
    @ModuleInfo(key: "tok_embeddings") var tokenEmbeddings: Embedding
    @ModuleInfo(key: "norm") var norm: NFKModernBertNorm

    init(_ c: NFKMLXModernBertConfiguration) {
        _tokenEmbeddings.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _norm.wrappedValue = NFKModernBertNorm(dimensions: c.hiddenSize, eps: c.normEpsilon)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray { norm(tokenEmbeddings(tokens)) }
}

/// The ModernBERT encoder, under the `model.` prefix the reranker checkpoint uses.
final class NFKModernBertModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKModernBertEmbeddings
    @ModuleInfo(key: "layers") var layers: [NFKModernBertBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: NFKModernBertNorm

    let configuration: NFKMLXModernBertConfiguration

    init(_ c: NFKMLXModernBertConfiguration) {
        configuration = c
        _embeddings.wrappedValue = NFKModernBertEmbeddings(c)
        _layers.wrappedValue = (0 ..< c.layerCount).map { NFKModernBertBlock(c, layer: $0) }
        _finalNorm.wrappedValue = NFKModernBertNorm(dimensions: c.hiddenSize, eps: c.normEpsilon)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let hidden = embeddings(tokens)
        let localMask = slidingWindowMask(length: tokens.shape[1])
        return finalNorm(reduce(hidden, localMask: localMask))
    }

    /// The state entering the stack and the RAW state each layer produces, for the isolation harness.
    /// ModernBERT's `output_hidden_states` does not apply the final norm to its last entry (the final
    /// norm goes only into `last_hidden_state`), so this matches that convention; the score test
    /// exercises the final norm.
    func layerStates(_ tokens: MLXArray) -> [MLXArray] {
        var hidden = embeddings(tokens)
        var states = [hidden]
        let localMask = slidingWindowMask(length: tokens.shape[1])
        for layer in layers {
            hidden = layer(hidden, globalMask: nil, localMask: localMask)
            states.append(hidden)
        }
        return states
    }

    private func reduce(_ input: MLXArray, localMask: MLXArray?) -> MLXArray {
        var hidden = input
        for layer in layers {
            hidden = layer(hidden, globalMask: nil, localMask: localMask)
        }
        return hidden
    }

    /// The bidirectional sliding-window mask a local layer uses, or nil when the sequence fits inside
    /// the window. A token attends to any other within `localAttention / 2` positions either side. The
    /// global layers pass nil (a single unpadded sequence is fully visible).
    private func slidingWindowMask(length: Int) -> MLXArray? {
        let half = configuration.localAttention / 2
        guard length > half + 1 else { return nil }
        let rows = MLXArray(0 ..< length).reshaped([length, 1])
        let columns = MLXArray(0 ..< length).reshaped([1, length])
        let within = abs(rows - columns) .<= half
        return MLX.where(within, MLXArray(Float(0)), MLXArray(Float(-1e9)))
    }
}

/// ModernBERT's prediction head: a dense projection, an exact GELU, and a LayerNorm.
final class NFKModernBertHead: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "norm") var norm: NFKModernBertNorm

    init(_ c: NFKMLXModernBertConfiguration) {
        _dense.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _norm.wrappedValue = NFKModernBertNorm(dimensions: c.hiddenSize, eps: c.normEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(gelu(dense(x))) }
}

/// The cross-encoder reranker: the ModernBERT encoder, a mean pool over the pair, the prediction head,
/// and a single-logit classifier that gives the relevance score.
public final class NFKMLXModernBertRerankerNet: Module {
    @ModuleInfo(key: "model") var model: NFKModernBertModel
    @ModuleInfo(key: "head") var head: NFKModernBertHead
    @ModuleInfo(key: "classifier") var classifier: Linear

    let configuration: NFKMLXModernBertConfiguration

    init(_ c: NFKMLXModernBertConfiguration) {
        configuration = c
        _model.wrappedValue = NFKModernBertModel(c)
        _head.wrappedValue = NFKModernBertHead(c)
        _classifier.wrappedValue = Linear(c.hiddenSize, 1, bias: true)
        super.init()
    }

    /// The relevance logit for one `[CLS] query [SEP] document [SEP]` token sequence.
    func score(tokens: [Int]) -> MLXArray {
        let input = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
        let hidden = model(input)                    // [1, length, hidden]
        let pooled = hidden[0].mean(axis: 0)         // mean pooling over the pair
        return classifier(head(pooled))              // [1]
    }
}

/// Holds the reranker's network and tokenizer for capture off the render thread.
private final class NFKRerankerHolder: @unchecked Sendable {
    let net: NFKMLXModernBertRerankerNet
    let tokenizer: NFKTokenizer?
    let configuration: NFKMLXModernBertConfiguration
    init(_ net: NFKMLXModernBertRerankerNet, _ tokenizer: NFKTokenizer?,
         _ configuration: NFKMLXModernBertConfiguration) {
        self.net = net
        self.tokenizer = tokenizer
        self.configuration = configuration
    }
}

/// A ModernBERT cross-encoder reranker.
///
/// @discussion A reranker scores a query against each candidate document and orders them, which is what
/// turns an embedder's approximate shortlist into a precise ranking. It reads the query and document
/// together as one `[CLS] query [SEP] document [SEP]` sequence and returns a single relevance logit; a
/// higher score is more relevant. Because it takes a query and a LIST of documents rather than one input,
/// it is a scoring object rather than an `NFKInferenceBackend`. Run scoring off the render thread.
@objc(NFKMLXModernBERTReranker)
public final class NFKMLXModernBERTReranker: NSObject {

    /// A name for the reranker the factories produce.
    @objc public static let modelName = "gte-reranker-modernbert-base"

    private let holder: NFKRerankerHolder

    init(net: NFKMLXModernBertRerankerNet, tokenizer: NFKTokenizer?,
         configuration: NFKMLXModernBertConfiguration) {
        holder = NFKRerankerHolder(net, tokenizer, configuration)
        super.init()
    }

    /// The relevance score of `document` for `query`. A higher score is more relevant; the value is an
    /// unbounded logit, so compare scores rather than reading one in isolation.
    @objc(scoreForQuery:document:)
    public func score(query: String, document: String) -> Double {
        Double(scoreValue(query: query, document: document))
    }

    /// The relevance score of each document for `query`, in the documents' order.
    @objc(scoresForQuery:documents:)
    public func scores(query: String, documents: [String]) -> [NSNumber] {
        documents.map { NSNumber(value: scoreValue(query: query, document: $0)) }
    }

    /// The documents' indices ordered from most relevant to least, which is the reranking.
    @objc(rankedIndicesForQuery:documents:)
    public func rankedIndices(query: String, documents: [String]) -> [NSNumber] {
        let scored = documents.enumerated().map { ($0.offset, scoreValue(query: query, document: $0.element)) }
        return scored.sorted { $0.1 > $1.1 }.map { NSNumber(value: $0.0) }
    }

    /// The token ids of a `[CLS] query [SEP] document [SEP]` pair, for a caller that scores ids directly.
    func tokens(query: String, document: String) -> [Int]? {
        guard let tokenizer = holder.tokenizer else { return nil }
        let c = holder.configuration
        return [c.clsToken] + tokenizer.encode(query).map(\.intValue) + [c.sepToken]
            + tokenizer.encode(document).map(\.intValue) + [c.sepToken]
    }

    private func scoreValue(query: String, document: String) -> Float {
        guard let tokens = tokens(query: query, document: document) else { return 0 }
        let logit = holder.net.score(tokens: tokens)
        eval(logit)
        return logit.asArray(Float.self)[0]
    }

    /// Builds a reranker from a downloaded release directory holding the weights, `config.json`, and the
    /// tokenizer files. Run scoring off the render thread.
    @objc(rerankerWithDirectoryURL:error:)
    public static func reranker(directoryURL: URL) throws -> NFKMLXModernBERTReranker {
        let net = NFKMLXModernBertRerankerNet(.gteReranker)
        try loadWeights(into: net, fromDirectory: directoryURL)
        let tokenizer = byteLevelTokenizer(inDirectory: directoryURL)
        return NFKMLXModernBERTReranker(net: net, tokenizer: tokenizer, configuration: .gteReranker)
    }

    /// Builds a reranker from optional local weights and a tokenizer, for a caller not loading a whole
    /// release directory. A nil `weightsURL` builds random weights (scoring runs, the numbers are
    /// meaningless), which is what a smoke test uses.
    public static func reranker(weightsURL: URL?, tokenizer: NFKTokenizer?,
                                configuration: NFKMLXModernBertConfiguration = .gteReranker)
        throws -> NFKMLXModernBERTReranker {
        let net = NFKMLXModernBertRerankerNet(configuration)
        if let weightsURL {
            let mapped = try NFKMLXWeights.loadCheckpoint(url: weightsURL).arrays.map { ($0, $1) }
            try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
        }
        return NFKMLXModernBERTReranker(net: net, tokenizer: tokenizer, configuration: configuration)
    }

    /// Loads the released checkpoint. Every key is the module's, so nothing is remapped.
    static func loadWeights(into net: NFKMLXModernBertRerankerNet, fromDirectory directory: URL) throws {
        try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: .float32)
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: .float32)
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    /// The byte-level BPE tokenizer a release describes.
    ///
    /// @discussion ModernBERT's tokenizer is GPT-2-family byte-level BPE, which the core
    /// `NFKByteLevelBPETokenizer` reads. The release ships only `tokenizer.json`, so its vocabulary and
    /// merges are extracted into the `vocab.json`/`merges.txt` the core reader takes.
    static func byteLevelTokenizer(inDirectory directory: URL) -> NFKTokenizer? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any],
              let vocabulary = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [Any] else { return nil }

        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        guard (try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)) != nil,
              let vocabularyData = try? JSONSerialization.data(withJSONObject: vocabulary),
              (try? vocabularyData.write(to: scratch.appendingPathComponent("vocab.json"))) != nil else {
            return nil
        }
        var mergesText = "#version: 0.2\n"
        for entry in merges {
            if let pair = entry as? [String], pair.count == 2 {
                mergesText += pair[0] + " " + pair[1] + "\n"
            } else if let text = entry as? String {
                mergesText += text + "\n"
            }
        }
        guard (try? mergesText.write(to: scratch.appendingPathComponent("merges.txt"),
                                     atomically: true, encoding: .utf8)) != nil else { return nil }
        let manifest: [String: Any] = ["tokenizer": ["type": "bpe-bytelevel", "pretokenizer": "gpt2"]]
        return try? NFKTokenizer(forManifest: manifest, directory: scratch)
    }
}
