//
//  NFKMLXPixtralBackendTests.swift
//  InferKitMLXTests
//
//  Pixtral behind the InferKit contract: the request's image, question, and token limit reach the model,
//  the answer comes back under NFKOutputText, and a request's token limit does not outlive it.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXPixtralBackendTests: XCTestCase {

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

    /// A 2-pixel-patch tower over an 8 × 8 image (a 4 × 4 grid of `[IMG]` tokens) and a decoder that emits
    /// "!" at every step and never stops on its own.
    private func tinyModel() throws -> NFKMLXPixtral {
        NFKMLXRandom.seed(6)
        let vision = NFKMLXPixtralVisionConfiguration(hiddenSize: 32, depth: 2, headCount: 2, intermediateSize: 64,
                                                      patchSize: 2, imageSize: 16)
        let tokenizer = try NFKMLXBackendTestSupport.tokenizer(specials: [
            "[IMG]": NFKMLXPixtral.imageTokenId, "<s>": 500, "[INST]": 501, "[/INST]": 502,
            "[IMG_BREAK]": 503, "[IMG_END]": 504])
        return NFKMLXPixtral(visionNet: NFKMLXPixtralVisionNet(vision),
                             connector: NFKMLXPixtralConnector(visionSize: 32, textSize: 64),
                             decoder: try NFKMLXBackendTestSupport.constantDecoder(.tiny), tokenizer: tokenizer,
                             endTokens: [], processor: NFKMLXPixtralImageProcessor(patchSize: 2, longestEdge: 8))
    }

    func testAQuestionAboutAnImageBecomesTextBoundedByTheRequest() throws {
        try requireMLXRuntime()
        let model = try tinyModel()
        let backend = model.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXPixtral.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputImage, NFKInputPrompt])
        XCTAssertEqual(backend.supportedParameterKeys, [NFKParameterMaxTokens])

        let image = NFKMLXBackendTestSupport.solidImage(width: 8, height: 8)
        let bounded = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: image, NFKInputPrompt: "Describe it."], parameters: [NFKParameterMaxTokens: 3]))
        XCTAssertEqual(bounded.output(forKey: NFKOutputText) as? String, "!!!")
        XCTAssertEqual(try model.answer(image: image, question: "Describe it.", maxTokens: 3), "!!!")

        let unbounded = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: image, NFKInputPrompt: "Describe it."], parameters: [:]))
        XCTAssertEqual(unbounded.output(forKey: NFKOutputText) as? String, String(repeating: "!", count: 128),
                       "a request without a token limit takes the default, not the last request's")
    }

    func testARequestWithoutAnImageOrAQuestionIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyModel().inferenceBackend()
        let image = NFKMLXBackendTestSupport.solidImage(width: 8, height: 8)
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "Describe it."], parameters: [:])))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: image], parameters: [:])))
    }
}
