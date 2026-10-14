//
//  NFKMLXQwenImageBackendTests.swift
//  InferKitMLXTests
//
//  Qwen-Image 2.1 behind the InferKit contract: the request's prompt and parameters reach the generator,
//  the RGBA image comes back under NFKOutputImage, and a request's settings do not outlive it.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXQwenImageBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tinyGenerator() throws -> NFKMLXQwenImageGenerator {
        // The M-RoPE sections the encoder reads cover 64 frequency pairs, so the head is 128 wide.
        var language = NFKMLXLanguageConfiguration.tiny
        language.headCount = 2
        language.keyValueHeadCount = 1
        language.headDimensions = 128
        var geometry = NFKMLXQwenImageConfiguration.tiny
        geometry.inChannels = 4
        geometry.outChannels = 4
        geometry.contextInDimensions = language.hiddenSize
        MLXRandom.seed(13)
        let generator = NFKMLXQwenImageGenerator(
            resident: true, tokenizer: NFKMLXScalarTestTokenizer(vocabularySize: language.vocabularySize),
            loadTextEncoder: { NFKMLXLanguageNet(language) },
            loadPipeline: {
                NFKMLXQwenImagePipeline(transformer: NFKMLXQwenImage.makeNet(geometry),
                                        vae: NFKMLXQwenImageVAE.makeNet(.qwenImage21Tiny))
            })
        try generator.loadResident()
        return generator
    }

    func testARequestBecomesAnImageAtTheRequestedSize() throws {
        try requireMLXRuntime()
        let generator = try tinyGenerator()
        let backend = generator.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXQwenImageGenerator.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputNegativePrompt])

        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox", NFKInputNegativePrompt: "blurry"],
                                          parameters: [NFKParameterWidth: 64, NFKParameterHeight: 32,
                                                       NFKParameterSteps: 2, NFKParameterGuidanceScale: 3,
                                                       NFKParameterSeed: 5])
        let result = try backend.runInference(for: request)
        let image = try XCTUnwrap(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(CFGetTypeID(image as CFTypeRef), CGImage.typeID)
        let cgImage = image as! CGImage
        // The tiny autoencoder upsamples less than a release's, so the expected size is the generator's own.
        let direct = try generator.image(forPrompt: "a red fox", width: 64, height: 32, seed: 5)
        XCTAssertEqual(cgImage.width, direct.dim(1))
        XCTAssertEqual(cgImage.height, direct.dim(0))
        XCTAssertEqual(generator.steps, 40, "the request's steps apply to that request only")
        XCTAssertEqual(generator.guidance, 1)
    }

    func testARequestWithoutAPromptIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyGenerator().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }
}
