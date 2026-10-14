//
//  NFKMLXSmolVLMBackendTests.swift
//  InferKitMLXTests
//
//  SmolVLM2 behind the InferKit contract: the request's image, question, and token limit reach the model,
//  the answer comes back under NFKOutputText, and a request's token limit does not outlive it.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXSmolVLMBackendTests: XCTestCase {

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

    /// A 64-pixel tile cut 2 × 2 from a 128-pixel frame plus the global tile, four `<image>` tokens a tile,
    /// and a decoder that emits "!" at every step and never stops on its own.
    private func tinyModel() throws -> NFKMLXSmolVLM {
        NFKMLXRandom.seed(4)
        let imageToken = 500
        let net = NFKMLXSmolVLMNet(vision: NFKMLXSigLIPNet(.tiny),
                                   connector: NFKMLXSmolVLMConnector(visionHidden: 32, decoderHidden: 64, scaleFactor: 2),
                                   decoder: try NFKMLXBackendTestSupport.constantDecoder(.tiny),
                                   imageTokenId: imageToken)
        let tokenizer = try NFKMLXBackendTestSupport.tokenizer(specials: [
            "<image>": imageToken, "<end_of_utterance>": 501, "<fake_token_around_image>": 502, "<global-img>": 503])
        let release = NFKMLXSmolVLM.Release(vision: .tiny, decoder: .tiny, scaleFactor: 2, imageTokenId: imageToken,
                                            tileSize: 64, longestEdge: 128)
        return NFKMLXSmolVLM(net: net, tokenizer: tokenizer, endToken: 1_000_000, release: release)
    }

    func testAQuestionAboutAnImageBecomesTextBoundedByTheRequest() throws {
        try requireMLXRuntime()
        let model = try tinyModel()
        let backend = model.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXSmolVLM.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputImage, NFKInputPrompt])
        XCTAssertEqual(backend.supportedParameterKeys, [NFKParameterMaxTokens])

        let image = NFKMLXBackendTestSupport.solidImage(width: 64, height: 64)
        let bounded = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: image, NFKInputPrompt: "What is this?"], parameters: [NFKParameterMaxTokens: 3]))
        XCTAssertEqual(bounded.output(forKey: NFKOutputText) as? String, "!!!")
        XCTAssertEqual(model.answer(image: image, question: "What is this?", maxTokens: 3), "!!!")

        let unbounded = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: image, NFKInputPrompt: "What is this?"], parameters: [:]))
        XCTAssertEqual(unbounded.output(forKey: NFKOutputText) as? String, String(repeating: "!", count: 128),
                       "a request without a token limit takes the default, not the last request's")
    }

    func testARequestWithoutAnImageOrAQuestionIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyModel().inferenceBackend()
        let image = NFKMLXBackendTestSupport.solidImage(width: 64, height: 64)
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "What is this?"], parameters: [:])))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: image], parameters: [:])))
    }
}
