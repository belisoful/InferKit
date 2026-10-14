//
//  NFKMLXModernBERTBackendTests.swift
//  InferKitMLXTests
//
//  The ModernBERT reranker behind the InferKit contract: the query and the documents reach the reranker,
//  and the scores and the ranking come back under NFKOutputStructured as the object computes them.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXModernBERTBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// The tiny encoder with its wrapping tokens inside its 512-entry vocabulary.
    private func tinyReranker() throws -> NFKMLXModernBERTReranker {
        NFKMLXRandom.seed(8)
        var configuration = NFKMLXModernBertConfiguration.tiny
        configuration.clsToken = 510
        configuration.sepToken = 511
        return try NFKMLXModernBERTReranker.reranker(weightsURL: nil,
                                                     tokenizer: try NFKMLXBackendTestSupport.tokenizer(),
                                                     configuration: configuration)
    }

    private let documents = ["the cat sat on the mat", "stock prices fell", "a kitten naps", "rain tomorrow"]

    func testTheDocumentsComeBackScoredAndRanked() throws {
        try requireMLXRuntime()
        let reranker = try tinyReranker()
        let backend = reranker.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXModernBERTReranker.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputDocuments])
        XCTAssertEqual(backend.supportedParameterKeys, [])

        let result = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "where is the cat", NFKInputDocuments: documents], parameters: [:]))
        let fields = try XCTUnwrap(result.output(forKey: NFKOutputStructured) as? [String: Any])
        let scores = try XCTUnwrap(fields[NFKMLXRerankKey.scores] as? [NSNumber]).map(\.doubleValue)
        let ranked = try XCTUnwrap(fields[NFKMLXRerankKey.rankedIndices] as? [NSNumber]).map(\.intValue)
        let expected = reranker.scores(query: "where is the cat", documents: documents).map(\.doubleValue)
        XCTAssertEqual(scores.count, documents.count)
        for (score, reference) in zip(scores, expected) {
            XCTAssertEqual(score, reference, accuracy: 1e-6)
        }
        XCTAssertGreaterThan(Set(scores).count, 1, "different documents score differently")
        XCTAssertEqual(Set(ranked), Set(documents.indices))
        XCTAssertEqual(ranked.map { scores[$0] }, scores.sorted(by: >), "the ranking runs from the highest score")
    }

    func testNoDocumentsScoreToEmptyLists() throws {
        try requireMLXRuntime()
        let result = try tinyReranker().inferenceBackend().runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "where is the cat", NFKInputDocuments: [String]()], parameters: [:]))
        let fields = try XCTUnwrap(result.output(forKey: NFKOutputStructured) as? [String: Any])
        XCTAssertEqual((fields[NFKMLXRerankKey.scores] as? [NSNumber])?.count, 0)
        XCTAssertEqual((fields[NFKMLXRerankKey.rankedIndices] as? [NSNumber])?.count, 0)
    }

    func testARequestWithoutAQueryOrTextDocumentsIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyReranker().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputDocuments: documents], parameters: [:])))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "where is the cat"], parameters: [:])))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "where is the cat", NFKInputDocuments: ["text", 7] as [Any]], parameters: [:])))
    }
}
