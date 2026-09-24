//
//  NFKMLXQwen3VLRetrievalTraining.swift
//  InferKitMLX
//
//  Customizing the Qwen3-VL retrieval pair on a consumer's own corpus. The released models are
//  general retrievers; a consumer's corpus has its own vocabulary and its own notion of relevance,
//  and the released similarity ranks it the way the pretraining data ranked its own.
//
//  The level is a probe: the backbone stays frozen and produces its embeddings once, and a small
//  module over those embeddings is what trains. A full fine-tune of a 2B backbone needs the optimizer
//  state of 2 billion parameters and a batch of negatives large enough for the contrastive objective
//  to mean anything, which is an offline run rather than a device one.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/// A linear adapter over a frozen Qwen3-VL embedding.
///
/// The adapter starts as the identity, so an untrained one reproduces the released embedding space
/// exactly and training moves away from it. The result is re-normalized, so a dot product between two
/// adapted embeddings stays a cosine similarity.
public final class NFKMLXQwen3VLEmbeddingAdapter: Module {

    /// How wide the embeddings it adapts are.
    public let dimensions: Int

    @ModuleInfo(key: "projection") public var projection: Linear

    /// Builds an identity adapter over `dimensions`-wide embeddings.
    public init(dimensions: Int) {
        self.dimensions = dimensions
        self._projection.wrappedValue = Linear(dimensions, dimensions, bias: false)
        super.init()
        projection.update(parameters: ModuleParameters.unflattened(
            ["weight": MLXArray.eye(dimensions)]))
    }

    /// The adapted embeddings, `[batch, dimensions]` in and out, normalized.
    public func callAsFunction(_ embeddings: MLXArray) -> MLXArray {
        let projected = projection(embeddings)
        return projected / sqrt((projected * projected).sum(axis: -1, keepDims: true))
    }
}

/// The pair scorer a Qwen3-VL reranker fine-tune trains.
///
/// The released scorer is one direction through the output projection: the difference of its "yes"
/// and "no" rows, read at the last position. A retarget starts from that direction and moves it, so
/// an untrained head reproduces the released score.
public final class NFKMLXQwen3VLRerankerHead: Module {

    /// How wide the hidden states it scores are.
    public let dimensions: Int

    @ModuleInfo(key: "score") public var score: Linear

    /// Builds a head from the released scoring direction, `[dimensions]`.
    public init(direction: MLXArray) {
        self.dimensions = direction.dim(0)
        self._score.wrappedValue = Linear(direction.dim(0), 1, bias: false)
        super.init()
        score.update(parameters: ModuleParameters.unflattened(
            ["weight": direction.reshaped([1, direction.dim(0)])]))
    }

    /// Builds a zero-initialized head, which scores every pair at 0.5 until it is trained.
    public convenience init(dimensions: Int) {
        self.init(direction: MLXArray.zeros([dimensions]))
    }

    /// The pair logits, `[batch, dimensions]` in and `[batch]` out. A relevance between 0 and 1 is
    /// this through a sigmoid, which is what the reference's `LogitScore` module reports.
    public func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        score(hidden).reshaped([-1])
    }
}

/// The contrastive objective the embedder is trained with.
///
/// This is sentence-transformers' `MultipleNegativesRankingLoss` at its defaults, which is the loss
/// the release is packaged for: cosine similarity between each query and every document in the batch,
/// scaled by 20, read as a classification over the batch whose right answer is the query's own
/// positive. Every other document is a negative, so a batch of `n` pairs carries `n - 1` negatives per
/// query at no extra cost, and explicit hard negatives add to them.
public struct NFKMLXQwen3VLEmbeddingObjective: Sendable {

    /// The inverse temperature the similarities are multiplied by before the softmax.
    public var scale: Float

    public init(scale: Float = 20) {
        self.scale = scale
    }

    /// The loss over a batch of queries and their document groups.
    ///
    /// - Parameters:
    ///   - queries: `[batch, dimensions]`, one query embedding per row.
    ///   - documents: one `[batch, dimensions]` group per document column: the positives first, then
    ///     a group per hard negative. Row `i` of every group belongs to query `i`.
    public func loss(queries: MLXArray, documents: [MLXArray]) -> MLXArray {
        let batch = queries.dim(0)
        let candidates = concatenated(documents.map { Self.normalized($0) }, axis: 0)
        let scores = matmul(Self.normalized(queries), candidates.transposed(1, 0)) * scale
        let positions = MLXArray((0 ..< batch).map { Int32($0) }).reshaped([batch, 1])
        let positive = takeAlong(scores, positions, axis: 1).reshaped([batch])
        return (logSumExp(scores, axis: 1) - positive).mean()
    }

    /// The loss of an adapter over frozen embeddings: the queries are the batch's input and the
    /// document groups its target, stacked `[groups, batch, dimensions]`.
    public func callAsFunction(_ adapter: NFKMLXQwen3VLEmbeddingAdapter, queries: MLXArray,
                               documents: MLXArray) -> MLXArray {
        loss(queries: adapter(queries),
             documents: (0 ..< documents.dim(0)).map { adapter(documents[$0]) })
    }

    static func normalized(_ values: MLXArray) -> MLXArray {
        values / sqrt((values * values).sum(axis: -1, keepDims: true))
    }
}

/// The pointwise objective the reranker is trained with.
///
/// This is sentence-transformers' `BinaryCrossEntropyLoss` at its defaults, which is the loss the
/// release's cross-encoder packaging names: binary cross entropy over the raw pair logit, with no
/// activation in front of it and no positive-class weight.
public struct NFKMLXQwen3VLRerankerObjective: Sendable {

    public init() {}

    /// The loss of `logits` against labels that are 1 for a relevant pair and 0 for an irrelevant one.
    ///
    /// The stable form of binary cross entropy with logits, which is the reference's: it never
    /// exponentiates a positive number, so a confident wrong pair cannot overflow.
    public func loss(logits: MLXArray, labels: MLXArray) -> MLXArray {
        (maximum(logits, 0) - logits * labels + log1p(exp(-abs(logits)))).mean()
    }

    /// The loss of a head over frozen hidden states.
    public func callAsFunction(_ head: NFKMLXQwen3VLRerankerHead, hidden: MLXArray,
                               labels: MLXArray) -> MLXArray {
        loss(logits: head(hidden), labels: labels)
    }
}

extension NFKMLXQwen3VLEmbedder {

    /// Builds an adapter over this model's embeddings, loading `weightsURL` when one is given.
    ///
    /// A nil `weightsURL` is the identity adapter, which is where a fine-tune starts. A file written
    /// by ``NFKMLXWeights/save(_:to:)`` after a run loads here and reproduces that run's embeddings.
    public func makeAdapter(weightsURL: URL? = nil) throws -> NFKMLXQwen3VLEmbeddingAdapter {
        let adapter = NFKMLXQwen3VLEmbeddingAdapter(dimensions: embeddingDimensions)
        if let weightsURL {
            let checkpoint = try NFKMLXWeights.loadCheckpoint(url: weightsURL)
            try NFKMLXWeights.apply(checkpoint.arrays.map { ($0.key, $0.value) }, to: adapter,
                                    verifyShapes: true)
        }
        return adapter
    }

    /// Installs a fine-tuned adapter from a file, so every later embedding is the adapted one.
    ///
    /// This is the Objective-C reach into a fine-tune: a consumer trains through ``fineTune(adapter:queries:documents:steps:learningRate:objective:observer:)``
    /// in Swift, saves, and an app installs the result here.
    @objc(loadAdapterFromURL:error:)
    public func loadAdapter(from url: URL) throws {
        adapter = try makeAdapter(weightsURL: url)
    }

    /// Trains an adapter over frozen embeddings the caller has already computed.
    ///
    /// - Parameters:
    ///   - adapter: the module to train, from ``makeAdapter(weightsURL:)``.
    ///   - queries: `[batch, dimensions]` query embeddings, from this model.
    ///   - documents: one `[batch, dimensions]` group per document column, positives first.
    ///   - steps: how many optimizer steps to take over the batch.
    ///   - learningRate: the learning rate. The optimizer is the one sentence-transformers' trainer
    ///     defaults to, `torch.optim.AdamW` with no weight decay, which is a bias-corrected Adam.
    ///   - objective: the loss, the release's in-batch contrastive objective by default.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The backbone is not in the graph: the embeddings are values, so nothing about the 2B model is
    /// differentiated, which is what makes the run fit on a device. Run it off the main thread.
    @discardableResult
    public func fineTune(adapter: NFKMLXQwen3VLEmbeddingAdapter, queries: MLXArray,
                         documents: [MLXArray], steps: Int, learningRate: Float = 1e-3,
                         objective: NFKMLXQwen3VLEmbeddingObjective = NFKMLXQwen3VLEmbeddingObjective(),
                         observer: NFKMLXTrainer.Observer? = nil) throws -> [Float] {
        let stacked = stacked(documents, axis: 0)
        return try NFKMLXTrainer.train(
            adapter, optimizer: Adam(learningRate: learningRate, biasCorrection: true), steps: steps,
            batch: { _ in (queries, stacked) },
            loss: { model, queries, documents in objective(model, queries: queries, documents: documents) },
            clipGradientNorm: 1, observer: observer)
    }
}

extension NFKMLXQwen3VLReranker {

    /// Builds a scoring head, starting from the release's own direction unless `weightsURL` names a
    /// fine-tuned one.
    public func makeHead(weightsURL: URL? = nil) throws -> NFKMLXQwen3VLRerankerHead {
        let head = NFKMLXQwen3VLRerankerHead(direction: scoringDirection())
        if let weightsURL {
            let checkpoint = try NFKMLXWeights.loadCheckpoint(url: weightsURL)
            try NFKMLXWeights.apply(checkpoint.arrays.map { ($0.key, $0.value) }, to: head,
                                    verifyShapes: true)
        }
        return head
    }

    /// Installs a fine-tuned head from a file, so every later score is the retargeted one.
    @objc(loadHeadFromURL:error:)
    public func loadHead(from url: URL) throws {
        head = try makeHead(weightsURL: url)
    }

    /// Trains a head over frozen last-position hidden states the caller has already computed.
    ///
    /// - Parameters:
    ///   - head: the module to train, from ``makeHead(weightsURL:)``.
    ///   - hidden: `[batch, dimensions]`, one last-position hidden state per pair.
    ///   - labels: `[batch]`, 1 for a relevant pair and 0 for an irrelevant one.
    ///   - steps: how many optimizer steps to take over the batch.
    ///   - learningRate: the learning rate. The optimizer is the one sentence-transformers' trainer
    ///     defaults to, `torch.optim.AdamW` with no weight decay, which is a bias-corrected Adam.
    ///   - objective: the loss, the release's binary cross entropy over the pair logit by default.
    ///   - observer: receives each step and can end the run early.
    @discardableResult
    public func fineTune(head: NFKMLXQwen3VLRerankerHead, hidden: MLXArray, labels: MLXArray,
                         steps: Int, learningRate: Float = 1e-3,
                         objective: NFKMLXQwen3VLRerankerObjective = NFKMLXQwen3VLRerankerObjective(),
                         observer: NFKMLXTrainer.Observer? = nil) throws -> [Float] {
        try NFKMLXTrainer.train(
            head, optimizer: Adam(learningRate: learningRate, biasCorrection: true), steps: steps,
            batch: { _ in (hidden, labels) },
            loss: { model, hidden, labels in objective(model, hidden: hidden, labels: labels) },
            clipGradientNorm: 1, observer: observer)
    }
}
