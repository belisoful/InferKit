//
//  NFKMLXQwen3VLRerankerBackendTests.swift
//  InferKitMLXTests
//
//  Qwen3-VL-Reranker behind the InferKit contract: a text or image query, text and image documents, and
//  the request's instruction reach the reranker, and the scores and the ranking come back under
//  NFKOutputStructured as the object computes them.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXQwen3VLRerankerBackendTests: XCTestCase {

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

    /// The Qwen3-VL backend test's tiny tower and decoder at random weights, scoring tokens 1 and 2.
    private func tinyReranker() throws -> NFKMLXQwen3VLReranker {
        NFKMLXRandom.seed(9)
        return NFKMLXQwen3VLReranker(visionNet: NFKMLXQwen3VLVisionNet(NFKMLXQwen3VLBackendTests.vision),
                                     decoder: NFKMLXLanguage.makeNet(NFKMLXQwen3VLBackendTests.decoder),
                                     tokenizer: try NFKMLXQwen3VLBackendTests.tokenizer(),
                                     processor: NFKMLXQwen3VLBackendTests.processor, yesToken: 1, noToken: 2)
    }

    private func scores(_ result: NFKInferenceResult) throws -> (scores: [Double], ranked: [Int]) {
        let fields = try XCTUnwrap(result.output(forKey: NFKOutputStructured) as? [String: Any])
        return (try XCTUnwrap(fields[NFKMLXRerankKey.scores] as? [NSNumber]).map(\.doubleValue),
                try XCTUnwrap(fields[NFKMLXRerankKey.rankedIndices] as? [NSNumber]).map(\.intValue))
    }

    func testTextAndImageDocumentsComeBackScoredAndRanked() throws {
        try requireMLXRuntime()
        let reranker = try tinyReranker()
        let backend = reranker.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXQwen3VLReranker.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputImage, NFKInputDocuments])
        XCTAssertEqual(backend.supportedParameterKeys, [NFKMLXRerankKey.instruction])

        let image = NFKMLXBackendTestSupport.solidImage(width: 8, height: 8, gray: 40)
        let documents: [Any] = ["a red barn", image, "stock prices fell"]
        let (scores, ranked) = try scores(backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a farm", NFKInputDocuments: documents], parameters: [:])))
        let expected = [reranker.score(query: "a farm", document: "a red barn"),
                        reranker.score(query: "a farm", queryImage: nil, document: nil, documentImage: image),
                        reranker.score(query: "a farm", document: "stock prices fell")]
        XCTAssertEqual(scores.count, 3)
        for (score, reference) in zip(scores, expected) {
            XCTAssertEqual(score, reference, accuracy: 1e-6)
            XCTAssertTrue(score > 0 && score < 1, "a relevance lies between 0 and 1")
        }
        XCTAssertEqual(ranked.map { scores[$0] }, scores.sorted(by: >), "the ranking runs from the highest score")
    }

    func testTheRequestsInstructionAndQueryImageReachTheReranker() throws {
        try requireMLXRuntime()
        let reranker = try tinyReranker()
        let backend = reranker.inferenceBackend()
        let image = NFKMLXBackendTestSupport.solidImage(width: 8, height: 8, gray: 200)
        let instruction = "Find matching pictures."

        let (instructed, _) = try scores(backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a farm", NFKInputDocuments: ["a red barn"]],
            parameters: [NFKMLXRerankKey.instruction: instruction])))
        let reference = reranker.score(query: "a farm", document: "a red barn", instruction: instruction)
        XCTAssertEqual(instructed[0], reference, accuracy: 1e-6)
        XCTAssertNotEqual(reference, reranker.score(query: "a farm", document: "a red barn"),
                          "the instruction changes the score, so the comparison discriminates")

        let (imageQuery, _) = try scores(backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: image, NFKInputDocuments: ["a red barn"]], parameters: [:])))
        XCTAssertEqual(imageQuery[0], reranker.score(query: nil, queryImage: image, document: "a red barn"),
                       accuracy: 1e-6)
    }

    func testARequestWithoutAQueryOrDocumentsIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyReranker().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputDocuments: ["a red barn"]], parameters: [:])))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a farm"], parameters: [:])))
    }
}
