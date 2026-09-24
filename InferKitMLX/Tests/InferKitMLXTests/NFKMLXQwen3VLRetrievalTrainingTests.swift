//
//  NFKMLXQwen3VLRetrievalTrainingTests.swift
//  InferKitMLXTests
//
//  Customizing the Qwen3-VL retrieval pair on a consumer's own corpus. Three things here are
//  load-bearing beyond the loop: both objectives are the ones sentence-transformers trains these
//  releases with, an untrained probe reproduces the released behavior, and a trained one reloads
//  through the model's own installer.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXQwen3VLRetrievalTrainingTests: XCTestCase {

    private var config: [String: String] { NFKMLXValidationConfig.environment }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_921)
    }

    // MARK: - The objectives

    // Both objectives against sentence-transformers, the library the releases are packaged for, on
    // identical tensors: the contrastive ranking loss with in-batch negatives alone and with an
    // explicit hard-negative group, and the pointwise binary cross entropy the reranker trains with.
    func testTheRetrievalObjectivesMatchTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_QWEN3_VL_RETRIEVAL_LOSS"] else {
            throw XCTSkip("set IK_PARITY_QWEN3_VL_RETRIEVAL_LOSS (run_reference.py qwen3vl_retrieval_loss)")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let queries = try XCTUnwrap(arrays["queries"])
        let positives = try XCTUnwrap(arrays["positives"])
        let negatives = try XCTUnwrap(arrays["negatives"])

        let ranking = NFKMLXQwen3VLEmbeddingObjective()
        let paired = ranking.loss(queries: queries, documents: [positives])
        let withNegatives = ranking.loss(queries: queries, documents: [positives, negatives])
        let binary = NFKMLXQwen3VLRerankerObjective().loss(
            logits: try XCTUnwrap(arrays["logits"]), labels: try XCTUnwrap(arrays["labels"]))
        eval(paired, withNegatives, binary)

        let referencePaired = try XCTUnwrap(arrays["ranking_loss"]).item(Float.self)
        let referenceNegatives = try XCTUnwrap(arrays["ranking_loss_with_negatives"]).item(Float.self)
        let referenceBinary = try XCTUnwrap(arrays["binary_loss"]).item(Float.self)
        print("VALIDATION PARITY qwen3vl_retrieval_loss: ranking \(paired.item(Float.self)) vs "
              + "\(referencePaired), with negatives \(withNegatives.item(Float.self)) vs "
              + "\(referenceNegatives), binary \(binary.item(Float.self)) vs \(referenceBinary)")
        XCTAssertEqual(paired.item(Float.self), referencePaired, accuracy: 1e-4)
        XCTAssertEqual(withNegatives.item(Float.self), referenceNegatives, accuracy: 1e-4)
        XCTAssertEqual(binary.item(Float.self), referenceBinary, accuracy: 1e-5)
    }

    // An in-batch negative is every other document, so a batch whose documents are permuted away from
    // their queries scores worse than one where they line up. This is what the objective is for.
    func testTheRankingObjectivePrefersTheMatchingPairing() throws {
        try requireMLXRuntime()
        let queries = MLXRandom.normal([4, 16])
        let aligned = queries + MLXRandom.normal([4, 16]) * 0.01
        let shuffled = aligned[MLXArray([Int32(1), 2, 3, 0])]
        let objective = NFKMLXQwen3VLEmbeddingObjective()
        let matched = objective.loss(queries: queries, documents: [aligned])
        let mismatched = objective.loss(queries: queries, documents: [shuffled])
        eval(matched, mismatched)
        XCTAssertLessThan(matched.item(Float.self), mismatched.item(Float.self))
    }

    // MARK: - The probes

    // The adapter starts as the identity, so an untrained one leaves the released embedding space
    // exactly where it was. A fine-tune that moved a randomly initialized adapter instead would be
    // training away from the released model rather than from it.
    func testAnUntrainedAdapterIsTheIdentity() throws {
        try requireMLXRuntime()
        let adapter = NFKMLXQwen3VLEmbeddingAdapter(dimensions: 8)
        let embeddings = NFKMLXQwen3VLEmbeddingObjective.normalized(MLXRandom.normal([3, 8]))
        let adapted = adapter(embeddings)
        eval(adapted, embeddings)
        let difference = abs(adapted - embeddings).max().item(Float.self)
        XCTAssertLessThan(difference, 1e-6, "the identity adapter reproduces its input")
    }

    // The head starts from the release's own scoring direction, so an untrained one reproduces the
    // released score rather than a random one.
    func testAnUntrainedHeadReproducesItsDirection() throws {
        try requireMLXRuntime()
        let direction = MLXRandom.normal([12])
        let head = NFKMLXQwen3VLRerankerHead(direction: direction)
        let hidden = MLXRandom.normal([5, 12])
        let mine = head(hidden)
        let reference = matmul(hidden, direction.reshaped([12, 1])).reshaped([-1])
        eval(mine, reference)
        XCTAssertLessThan(abs(mine - reference).max().item(Float.self), 1e-5)
    }

    // A run over frozen embeddings lowers the contrastive loss and moves the adapter away from the
    // identity it started at. The backbone is not in the graph at all: the embeddings are values.
    func testAnAdapterFineTuneLowersTheRankingLoss() throws {
        try requireMLXRuntime()
        let width = 16
        let queries = MLXRandom.normal([6, width])
        // Each query's positive is a rotation of it, which no identity adapter ranks first.
        let rotation = MLXArray((0 ..< width).map { Int32(($0 + 5) % width) })
        let positives = queries.take(rotation, axis: 1) + MLXRandom.normal([6, width]) * 0.01
        eval(queries, positives)

        let embedder = NFKMLXQwen3VLEmbeddingAdapter(dimensions: width)
        // The parameter is replaced in place by the update, so the comparison holds VALUES rather
        // than the array the module carried before the run.
        eval(embedder.projection.weight)
        let before = embedder.projection.weight.asArray(Float.self)
        let objective = NFKMLXQwen3VLEmbeddingObjective()
        let losses = try NFKMLXTrainer.train(
            embedder, optimizer: Adam(learningRate: 5e-2), steps: 40,
            batch: { _ in (queries, positives.reshaped([1, 6, width])) },
            loss: { model, q, d in objective(model, queries: q, documents: d) },
            clipGradientNorm: 1)
        XCTAssertEqual(losses.count, 40)
        XCTAssertLessThan(losses[losses.count - 1], losses[0], "the ranking loss falls")
        eval(embedder.projection.weight)
        let after = embedder.projection.weight.asArray(Float.self)
        let moved = zip(before, after).map { Swift.abs($0 - $1) }.max() ?? 0
        XCTAssertGreaterThan(moved, 1e-4, "the adapter moves")
    }

    // A run over frozen hidden states lowers the binary loss and separates the labeled pairs.
    func testAHeadFineTuneLowersTheBinaryLoss() throws {
        try requireMLXRuntime()
        let width = 12
        let hidden = MLXRandom.normal([8, width])
        let labels = MLXArray([Float(1), 0, 1, 0, 1, 0, 1, 0])
        let head = NFKMLXQwen3VLRerankerHead(dimensions: width)
        let objective = NFKMLXQwen3VLRerankerObjective()
        let losses = try NFKMLXTrainer.train(
            head, optimizer: Adam(learningRate: 5e-2), steps: 60,
            batch: { _ in (hidden, labels) },
            loss: { model, h, l in objective(model, hidden: h, labels: l) },
            clipGradientNorm: 1)
        XCTAssertLessThan(losses[losses.count - 1], losses[0], "the binary loss falls")

        let scored = head(hidden)
        eval(scored)
        let values = scored.asArray(Float.self)
        let relevant = stride(from: 0, to: 8, by: 2).map { values[$0] }.min() ?? 0
        let irrelevant = stride(from: 1, to: 8, by: 2).map { values[$0] }.max() ?? 0
        XCTAssertGreaterThan(relevant, irrelevant, "every relevant pair outscores every irrelevant one")
    }

    // MARK: - The round trip

    // A trained probe saves and reloads, reproducing its output. Without this a fine-tune is a
    // number in a log rather than a model a consumer can ship.
    func testAProbeRoundTripsThroughItsFile() throws {
        try requireMLXRuntime()
        let width = 10
        let adapter = NFKMLXQwen3VLEmbeddingAdapter(dimensions: width)
        adapter.update(parameters: ModuleParameters.unflattened(
            ["projection.weight": MLXRandom.normal([width, width])]))
        let embeddings = MLXRandom.normal([4, width])
        let before = adapter(embeddings)
        eval(before)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3vl-adapter-\(UUID().uuidString).safetensors")
        try NFKMLXWeights.save(adapter, to: url)
        let reloaded = NFKMLXQwen3VLEmbeddingAdapter(dimensions: width)
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        try NFKMLXWeights.apply(checkpoint.arrays.map { ($0.key, $0.value) }, to: reloaded,
                                verifyShapes: true)
        let after = reloaded(embeddings)
        eval(after)
        XCTAssertLessThan(abs(after - before).max().item(Float.self), 1e-6,
                          "the reloaded adapter reproduces the embeddings")

        let head = NFKMLXQwen3VLRerankerHead(direction: MLXRandom.normal([width]))
        let hidden = MLXRandom.normal([3, width])
        let scoresBefore = head(hidden)
        eval(scoresBefore)
        let headURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3vl-head-\(UUID().uuidString).safetensors")
        try NFKMLXWeights.save(head, to: headURL)
        let reloadedHead = NFKMLXQwen3VLRerankerHead(dimensions: width)
        let headCheckpoint = try NFKMLXWeights.loadCheckpoint(url: headURL)
        try NFKMLXWeights.apply(headCheckpoint.arrays.map { ($0.key, $0.value) }, to: reloadedHead,
                                verifyShapes: true)
        let scoresAfter = reloadedHead(hidden)
        eval(scoresAfter)
        XCTAssertLessThan(abs(scoresAfter - scoresBefore).max().item(Float.self), 1e-6,
                          "the reloaded head reproduces the scores")
    }

    // The released models install a probe and go on scoring: an identity adapter leaves the released
    // embedding unchanged, and a head built from the release's own direction reproduces its score.
    // This is the seam a consumer's fine-tune lands on, measured on the released weights.
    func testAProbeInstallsOnTheReleasedModels() throws {
        try requireMLXRuntime()
        guard let embeddingDirectory = config["IK_VAL_QWEN3_VL_EMBEDDING"],
              let rerankerDirectory = config["IK_VAL_QWEN3_VL_RERANKER"] else {
            throw XCTSkip("set IK_VAL_QWEN3_VL_EMBEDDING and IK_VAL_QWEN3_VL_RERANKER")
        }
        let embedder = try NFKMLXQwen3VLEmbedder.embedder(
            directoryURL: URL(fileURLWithPath: embeddingDirectory))
        let text = "A photograph of a red bicycle."
        let released = embedder.embedding(forText: text).map(\.doubleValue)
        embedder.adapter = try embedder.makeAdapter()
        let adapted = embedder.embedding(forText: text).map(\.doubleValue)
        let worst = zip(released, adapted).map { abs($0 - $1) }.max() ?? 1
        XCTAssertLessThan(worst, 1e-5, "the identity adapter leaves the released embedding alone")

        let reranker = try NFKMLXQwen3VLReranker.reranker(
            directoryURL: URL(fileURLWithPath: rerankerDirectory))
        let query = "How tall is the Eiffel Tower?"
        let document = "The Eiffel Tower stands 330 metres tall, including its antennas."
        let releasedScore = reranker.score(query: query, document: document)
        reranker.head = try reranker.makeHead()
        let headScore = reranker.score(query: query, document: document)
        print("VALIDATION probe qwen3vl-reranker: released \(releasedScore) vs head \(headScore)")
        XCTAssertEqual(headScore, releasedScore, accuracy: 1e-4,
                       "the head starts at the release's own direction")
    }
}
