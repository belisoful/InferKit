//
//  NFKMLXTextEmbeddingAdapterTests.swift
//  InferKitMLXTests
//
//  The adapter fine-tune every text-embedding backend carries, over the two embedder families it
//  serves: Qwen3-Embedding's last-token pooled decoder and EmbeddingGemma's mean-pooled encoder. The
//  claims: an untrained adapter changes nothing, a run lowers the ranking loss over frozen embeddings,
//  and a saved adapter installs into a backend and reproduces the run's embeddings.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXTextEmbeddingAdapterTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        // Clearing the cache reaches MLX's runtime, which needs a Metal library it can find.
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func qwen3EmbeddingBackend() -> NFKMLXTextEmbeddingBackend {
        NFKMLXRandom.seed(3)
        let embedder = NFKMLXTextEmbedder(net: NFKMLXLanguage.makeNet(.tiny), configuration:
            NFKMLXTextEmbedderConfiguration(pooling: .lastToken, appendedToken: 1, normalizes: true))
        return NFKMLXTextEmbeddingBackend(embedder: embedder, tokenize: nil, identifier: "tiny-qwen3-embedding")
    }

    private func embeddingGemmaBackend() -> NFKMLXTextEmbeddingBackend {
        NFKMLXRandom.seed(5)
        let width = NFKMLXGemma3EncoderConfiguration.tiny.hiddenSize
        // Random projections, so the Dense head carries the pooled vector rather than mapping every
        // input to the same constant.
        let embedder = NFKMLXEmbeddingGemmaEmbedder(
            net: NFKMLXGemma3EncoderNet(.tiny),
            dense2: MLXRandom.normal([4 * width, width]) * 0.05,
            dense3: MLXRandom.normal([width, 4 * width]) * 0.05,
            configuration: NFKMLXEmbeddingGemmaConfiguration(prependedToken: nil, appendedToken: nil, dimensions: nil))
        return NFKMLXTextEmbeddingBackend(embedder: embedder, tokenize: nil, identifier: "tiny-embeddinggemma")
    }

    /// Six query texts and six documents, each document its query's positive.
    private let queries: [[Int]] = [[3, 17, 42], [5, 9, 11, 2], [21, 8], [30, 31, 4], [12, 6, 7], [40, 2, 19]]
    private let documents: [[Int]] = [[44, 13], [27, 3, 3], [9, 50, 18], [14, 14], [36, 25, 1], [10, 29]]

    private func backends() -> [NFKMLXTextEmbeddingBackend] {
        [qwen3EmbeddingBackend(), embeddingGemmaBackend()]
    }

    func testAnUntrainedAdapterReportsTheReleasedEmbedding() throws {
        try requireMLXRuntime()
        for backend in backends() {
            let released = backend.embedding(forTokens: [3, 17, 42, 5]).map(\.floatValue)
            backend.adapter = try backend.makeAdapter()
            let adapted = backend.embedding(forTokens: [3, 17, 42, 5]).map(\.floatValue)
            XCTAssertEqual(adapted.count, released.count)
            for (a, r) in zip(adapted, released) {
                XCTAssertEqual(a, r, accuracy: 1e-5, backend.backendIdentifier)
            }
        }
    }

    func testTheCachedEmbeddingsAreTheModelsOwnWithoutTheAdapter() throws {
        try requireMLXRuntime()
        for backend in backends() {
            let cached = try backend.embeddings(forTokenSequences: queries)
            XCTAssertEqual(cached.shape, [queries.count, backend.embeddingDimensions])
            backend.adapter = NFKMLXEmbeddingAdapter(dimensions: backend.embeddingDimensions)
            backend.adapter?.projection.update(parameters: ModuleParameters.unflattened(
                ["weight": MLXRandom.normal([backend.embeddingDimensions, backend.embeddingDimensions])]))
            let again = try backend.embeddings(forTokenSequences: queries)
            XCTAssertEqual(again.asArray(Float.self), cached.asArray(Float.self), backend.backendIdentifier)
        }
    }

    func testEncodingNoTextsFails() throws {
        try requireMLXRuntime()
        XCTAssertThrowsError(try qwen3EmbeddingBackend().embeddings(forTokenSequences: []))
    }

    func testTheTextPathNeedsATokenizer() throws {
        try requireMLXRuntime()
        XCTAssertThrowsError(try qwen3EmbeddingBackend().embeddings(for: ["a query"]))
    }

    func testAFineTuneLowersTheRankingLoss() throws {
        try requireMLXRuntime()
        for backend in backends() {
            let queryEmbeddings = try backend.embeddings(forTokenSequences: queries)
            let documentEmbeddings = try backend.embeddings(forTokenSequences: documents)
            let adapter = try backend.makeAdapter()
            let losses = try backend.fineTune(adapter: adapter, queries: queryEmbeddings,
                                              documents: [documentEmbeddings], steps: 40, learningRate: 5e-2)
            XCTAssertEqual(losses.count, 40)
            XCTAssertLessThan(losses.last!, losses.first!, backend.backendIdentifier)
        }
    }

    func testMismatchedWidthsAreRefused() throws {
        try requireMLXRuntime()
        let adapter = NFKMLXEmbeddingAdapter(dimensions: 8)
        XCTAssertThrowsError(try NFKMLXEmbeddingAdapter.train(adapter, queries: MLXArray.zeros([4, 8]),
                                                              documents: [MLXArray.zeros([4, 9])], steps: 1))
        XCTAssertThrowsError(try NFKMLXEmbeddingAdapter.train(adapter, queries: MLXArray.zeros([4, 8]),
                                                              documents: [], steps: 1))
    }

    // A trained adapter saves, installs into a fresh backend from the file, and reproduces the run's
    // embeddings; removing it restores the released ones.
    func testASavedAdapterInstallsIntoABackendAndReproducesTheRun() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("text-embedding-adapter-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let trainer = embeddingGemmaBackend()
        let queryEmbeddings = try trainer.embeddings(forTokenSequences: queries)
        let adapter = try trainer.makeAdapter()
        try trainer.fineTune(adapter: adapter, queries: queryEmbeddings,
                             documents: [try trainer.embeddings(forTokenSequences: documents)],
                             steps: 20, learningRate: 5e-2)
        try NFKMLXWeights.save(adapter, to: url)
        let expected = adapter(queryEmbeddings[0 ..< 1]).reshaped([-1]).asArray(Float.self)

        let deployed = embeddingGemmaBackend()
        let released = deployed.embedding(forTokens: queries[0].map { NSNumber(value: $0) }).map(\.floatValue)
        try deployed.loadAdapter(from: url)
        let adapted = deployed.embedding(forTokens: queries[0].map { NSNumber(value: $0) }).map(\.floatValue)
        XCTAssertEqual(adapted.count, expected.count)
        for (a, e) in zip(adapted, expected) {
            XCTAssertEqual(a, e, accuracy: 1e-5)
        }
        deployed.removeAdapter()
        let restored = deployed.embedding(forTokens: queries[0].map { NSNumber(value: $0) }).map(\.floatValue)
        XCTAssertEqual(restored, released)
    }
}
