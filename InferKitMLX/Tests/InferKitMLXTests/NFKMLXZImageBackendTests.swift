//
//  NFKMLXZImageBackendTests.swift
//  InferKitMLXTests
//
//  Z-Image behind the InferKit contract: the request's prompt and parameters reach the generator, the
//  image comes back under NFKOutputImage, and a request's settings do not outlive it.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXZImageBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tinyGenerator() throws -> NFKMLXZImageGenerator {
        var vae = NFKMLXSDVAEConfiguration()
        vae.latentChannels = 4
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4
        vae.useQuantConv = false
        vae.scaleFactor = 0.3611
        vae.shiftFactor = 0.1159
        // The transformer's `.tiny` geometry reads 24-wide caption features.
        let text = NFKMLXLanguageConfiguration(hiddenSize: 24, layerCount: 3, headCount: 2, keyValueHeadCount: 1,
                                               headDimensions: 12, intermediateSize: 48, vocabularySize: 512)
        MLXRandom.seed(11)
        let generator = NFKMLXZImageGenerator(
            resident: true, tokenizer: NFKMLXScalarTestTokenizer(vocabularySize: 512),
            loadTextEncoder: { NFKMLXLanguage.makeNet(text) },
            loadPipeline: {
                NFKMLXZImagePipeline(transformer: NFKMLXZImageTransformerNet(.tiny),
                                     vae: NFKMLXSDAutoencoder(configuration: vae), schedule: .zImageTurbo)
            })
        try generator.loadResident()
        return generator
    }

    func testARequestBecomesAnImageAtTheRequestedSize() throws {
        try requireMLXRuntime()
        let generator = try tinyGenerator()
        let backend = generator.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXZImageGenerator.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputNegativePrompt])

        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox", NFKInputNegativePrompt: "blurry"],
                                          parameters: [NFKParameterWidth: 32, NFKParameterHeight: 16,
                                                       NFKParameterSteps: 2, NFKParameterGuidanceScale: 4,
                                                       NFKParameterSeed: 3])
        let result = try backend.runInference(for: request)
        let image = try XCTUnwrap(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(CFGetTypeID(image as CFTypeRef), CGImage.typeID)
        let cgImage = image as! CGImage
        XCTAssertEqual(cgImage.width, 32)
        XCTAssertEqual(cgImage.height, 16)
        XCTAssertEqual(generator.steps, 9, "the request's steps apply to that request only")
        XCTAssertEqual(generator.guidance, 0)
    }

    func testARequestWithoutAPromptIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyGenerator().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }
}
