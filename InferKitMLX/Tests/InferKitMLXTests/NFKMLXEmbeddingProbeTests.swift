//
//  NFKMLXEmbeddingProbeTests.swift
//  InferKitMLXTests
//
//  The shared linear probe, apart from any embedder: what it checks before a run and how a saved probe
//  reloads.
//

import XCTest
import MLX
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXEmbeddingProbeTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_924)
    }

    func testEmbeddingsOfTheWrongWidthAreRefused() throws {
        try requireMLXRuntime()
        let probe = NFKMLXEmbeddingProbe(embedDimensions: 8, classCount: 2)
        XCTAssertThrowsError(try NFKMLXEmbeddingProbe.train(probe, embeddings: MLXArray.zeros([4, 9]),
                                                            labels: MLXArray([Int32(0), 1, 0, 1]), steps: 1))
    }

    func testMismatchedEmbeddingsAndLabelsAreRefused() throws {
        try requireMLXRuntime()
        let probe = NFKMLXEmbeddingProbe(embedDimensions: 8, classCount: 2)
        XCTAssertThrowsError(try NFKMLXEmbeddingProbe.train(probe, embeddings: MLXArray.zeros([4, 8]),
                                                            labels: MLXArray([Int32(0), 1]), steps: 1))
    }

    func testASavedProbeReloadsItsWidthClassCountAndWeights() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("embedding-probe-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let embeddings = MLXRandom.uniform(low: -1, high: 1, [6, 8])
        let probe = NFKMLXEmbeddingProbe(embedDimensions: 8, classCount: 3)
        try NFKMLXEmbeddingProbe.train(probe, embeddings: embeddings, labels: MLXArray([Int32(0), 1, 2, 0, 1, 2]),
                                       optimizer: AdamW(learningRate: 5e-2), steps: 20)
        try NFKMLXWeights.save(probe, to: url)

        let reloaded = try NFKMLXEmbeddingProbe(weightsURL: url)
        XCTAssertEqual(reloaded.embedDimensions, 8)
        XCTAssertEqual(reloaded.classCount, 3)
        XCTAssertEqual(reloaded(embeddings).asArray(Float.self), probe(embeddings).asArray(Float.self))
    }

    func testAFileWithNoClassifierIsRefused() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("not-a-probe-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: ["other.weight": MLXArray.zeros([2, 2])], url: url)
        XCTAssertThrowsError(try NFKMLXEmbeddingProbe(weightsURL: url))
    }
}
