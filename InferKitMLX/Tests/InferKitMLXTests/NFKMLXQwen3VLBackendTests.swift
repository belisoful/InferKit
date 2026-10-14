//
//  NFKMLXQwen3VLBackendTests.swift
//  InferKitMLXTests
//
//  Qwen3-VL behind the InferKit contract: the request's image, question, and token limit reach the model,
//  the answer comes back under NFKOutputText, and a request's token limit does not outlive it.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXQwen3VLBackendTests: XCTestCase {

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

    /// The tiny vision tower, its 32-wide output matched by the decoder, which must reach the fixed
    /// `<|image_pad|>` id.
    static let vision = NFKMLXQwen3VLVisionConfiguration(
        hiddenSize: 32, depth: 4, headCount: 2, intermediateSize: 64, patchSize: 2, temporalPatchSize: 2,
        spatialMergeSize: 2, outHiddenSize: 32, positionGridSide: 4, deepstackLayers: [1, 2])

    static var decoder: NFKMLXLanguageConfiguration {
        var configuration = NFKMLXLanguageConfiguration.tiny
        configuration.hiddenSize = 32
        configuration.headDimensions = 8
        configuration.intermediateSize = 64
        configuration.vocabularySize = NFKMLXQwen3VL.imageTokenId + 1
        return configuration
    }

    /// An 8 × 8 image becomes a 4 × 4 patch grid and four merged tokens.
    static let processor = NFKMLXQwen3VLImageProcessor(patchSize: 2, temporalPatchSize: 2, mergeSize: 2,
                                                       minPixels: 16, maxPixels: 64)

    static func tokenizer() throws -> NFKTokenizer {
        try NFKMLXBackendTestSupport.tokenizer(specials: [
            "<|image_pad|>": NFKMLXQwen3VL.imageTokenId, "<|im_start|>": 500, "<|im_end|>": 501,
            "<|vision_start|>": 502, "<|vision_end|>": 503])
    }

    /// A model whose decoder emits "!" at every step and never stops on its own.
    private func tinyModel() throws -> NFKMLXQwen3VL {
        NFKMLXRandom.seed(6)
        return NFKMLXQwen3VL(visionNet: NFKMLXQwen3VLVisionNet(Self.vision),
                             decoder: try NFKMLXBackendTestSupport.constantDecoder(Self.decoder),
                             tokenizer: try Self.tokenizer(), endTokens: [], processor: Self.processor)
    }

    func testAQuestionAboutAnImageBecomesTextBoundedByTheRequest() throws {
        try requireMLXRuntime()
        let model = try tinyModel()
        let backend = model.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXQwen3VL.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputImage, NFKInputPrompt])
        XCTAssertEqual(backend.supportedParameterKeys, [NFKParameterMaxTokens])

        let image = NFKMLXBackendTestSupport.solidImage(width: 8, height: 8)
        let bounded = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: image, NFKInputPrompt: "Describe it."], parameters: [NFKParameterMaxTokens: 3]))
        XCTAssertEqual(bounded.output(forKey: NFKOutputText) as? String, "!!!")
        XCTAssertEqual(model.answer(image: image, question: "Describe it.", maxTokens: 3), "!!!")

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
