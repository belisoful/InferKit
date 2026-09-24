//
//  NFKMLXEmbeddingAdapter.swift
//  InferKitMLX
//
//  Adapting a retrieval embedder to a consumer's own corpus without touching the embedder.
//
//  A released embedder ranks a corpus the way its pretraining data ranked its own. A consumer's corpus
//  has its own vocabulary and its own notion of relevance. The adapter is a linear map over the frozen
//  embedding, trained on the consumer's query-document pairs under the contrastive objective the
//  sentence-transformers releases are trained with. The embedder encodes each text once, so the run
//  differentiates only the adapter and fits on a device. Qwen3-VL-Embedding, Qwen3-Embedding, and
//  EmbeddingGemma share it.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/*!
 @abstract A linear adapter over a frozen embedding.
 @discussion The adapter starts as the identity, so an untrained one reproduces the released embedding
 space exactly and training moves away from it. The result is re-normalized, so a dot product between
 two adapted embeddings stays a cosine similarity. Introduced in InferKit 0.5.0.
 */
public final class NFKMLXEmbeddingAdapter: Module {

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

    /// Builds an adapter over `dimensions`-wide embeddings from a file `NFKMLXWeights.save` wrote.
    public convenience init(dimensions: Int, weightsURL: URL) throws {
        self.init(dimensions: dimensions)
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: weightsURL)
        try NFKMLXWeights.apply(checkpoint.arrays.map { ($0.key, $0.value) }, to: self, verifyShapes: true)
    }

    /// The adapted embeddings, `[batch, dimensions]` in and out, normalized.
    public func callAsFunction(_ embeddings: MLXArray) -> MLXArray {
        let projected = projection(embeddings)
        return projected / sqrt((projected * projected).sum(axis: -1, keepDims: true))
    }

    /// Trains an adapter over frozen embeddings the caller has already computed, returning the loss
    /// from each step.
    ///
    /// - Parameters:
    ///   - adapter: the module to train.
    ///   - queries: `[batch, dimensions]` query embeddings.
    ///   - documents: one `[batch, dimensions]` group per document column, positives first.
    ///   - steps: how many optimizer steps to take over the batch.
    ///   - learningRate: the learning rate. The optimizer is the one sentence-transformers' trainer
    ///     defaults to, `torch.optim.AdamW` with no weight decay, which is a bias-corrected Adam.
    ///   - objective: the loss, sentence-transformers' in-batch contrastive objective by default.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The embedder is not in the graph: the embeddings are values, so nothing about it is
    /// differentiated. Run it off the main thread.
    @discardableResult
    public static func train(_ adapter: NFKMLXEmbeddingAdapter, queries: MLXArray, documents: [MLXArray],
                             steps: Int, learningRate: Float = 1e-3,
                             objective: NFKMLXEmbeddingRankingObjective = NFKMLXEmbeddingRankingObjective(),
                             observer: NFKMLXTrainer.Observer? = nil) throws -> [Float] {
        guard !documents.isEmpty else {
            throw NFKMLXError.trainingDataMismatch("an adapter needs at least one document group, the positives")
        }
        let widths = [queries] + documents
        guard widths.allSatisfy({ $0.ndim == 2 && $0.dim(1) == adapter.dimensions && $0.dim(0) == queries.dim(0) }) else {
            throw NFKMLXError.trainingDataMismatch(
                "queries and every document group must be [batch, \(adapter.dimensions)] with one row per query")
        }
        let stacked = stacked(documents, axis: 0)
        return try NFKMLXFineTune.run(
            adapter, freezing: {}, optimizer: nil,
            reference: { Adam(learningRate: learningRate, biasCorrection: true) },
            referenceSchedule: { .constant }, steps: steps,
            batch: { _ in (queries, stacked) },
            loss: { model, queries, documents in objective(model, queries: queries, documents: documents) },
            clipGradientNorm: 1, observer: observer)
    }
}

/*!
 @abstract The contrastive objective an embedding adapter trains with.
 @discussion This is sentence-transformers' `MultipleNegativesRankingLoss` at its defaults: cosine
 similarity between each query and every document in the batch, scaled by 20, read as a classification
 over the batch whose right answer is the query's own positive. Every other document is a negative, so
 a batch of `n` pairs carries `n - 1` negatives per query at no extra cost, and explicit hard negatives
 add to them. Introduced in InferKit 0.5.0.
 */
public struct NFKMLXEmbeddingRankingObjective: Sendable {

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
    public func callAsFunction(_ adapter: NFKMLXEmbeddingAdapter, queries: MLXArray,
                               documents: MLXArray) -> MLXArray {
        loss(queries: adapter(queries),
             documents: (0 ..< documents.dim(0)).map { adapter(documents[$0]) })
    }

    static func normalized(_ values: MLXArray) -> MLXArray {
        values / sqrt((values * values).sum(axis: -1, keepDims: true))
    }
}
