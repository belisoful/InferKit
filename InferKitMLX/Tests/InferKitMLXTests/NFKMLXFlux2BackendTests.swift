//
//  NFKMLXFlux2BackendTests.swift
//  InferKitMLXTests
//
//  FLUX.2 [klein] behind the InferKit contract: text-to-image, reference conditioning through
//  NFKInputImages, and inpainting through NFKInputImage with NFKInputMask, each coming back under
//  NFKOutputImage, with a request's settings not outliving it.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXFlux2BackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tinyFlux() throws -> NFKMLXFlux2 {
        let language = NFKMLXLanguageConfiguration.tiny
        var geometry = NFKMLXFlux2Configuration.tiny
        geometry.inChannels = 4 * 2 * 2
        geometry.jointAttentionDim = 2 * language.hiddenSize
        geometry.guidanceEmbeds = false
        var vae = NFKMLXSDVAEConfiguration.flux2
        vae.latentChannels = 4
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4
        MLXRandom.seed(23)
        // Not distilled, so the model guides and reads the negative prompt.
        let flux = NFKMLXFlux2(
            resident: true,
            loadTextEncoder: {
                NFKMLXFlux2TextEncoder(decoder: NFKMLXLanguage.makeNet(language), layers: [1, 2], contextLength: 16,
                                       padToken: 0)
            },
            loadPipeline: {
                NFKMLXFlux2Pipeline(transformer: NFKMLXFlux2TransformerNet(geometry),
                                    autoencoder: NFKMLXSDAutoencoder(configuration: vae),
                                    codec: NFKMLXFlux2LatentCodec(patchedChannels: 16, epsilon: 1e-4, patch: 2))
            },
            tokenizer: NFKMLXScalarTestTokenizer(vocabularySize: language.vocabularySize), chatTemplate: nil,
            isDistilled: false, encodesInFloat32: true)
        try flux.loadResident()
        return flux
    }

    private func outputImage(_ result: NFKInferenceResult) throws -> CGImage {
        let image = try XCTUnwrap(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(CFGetTypeID(image as CFTypeRef), CGImage.typeID)
        return image as! CGImage
    }

    func testARequestBecomesAnImageAtTheRequestedSize() throws {
        try requireMLXRuntime()
        let flux = try tinyFlux()
        let backend = flux.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXFlux2.modelName)
        XCTAssertEqual(backend.supportedInputKeys,
                       [NFKInputPrompt, NFKInputNegativePrompt, NFKInputImages, NFKInputImage, NFKInputMask])
        XCTAssertEqual(backend.supportedParameterKeys?.contains(NFKMLXFlux2ParameterKey.cachesReferences), true)

        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox", NFKInputNegativePrompt: "blurry"],
                                          parameters: [NFKParameterWidth: 48, NFKParameterHeight: 32,
                                                       NFKParameterSteps: 2, NFKParameterGuidanceScale: 3,
                                                       NFKParameterSeed: 5])
        let image = try outputImage(try backend.runInference(for: request))
        // The tiny autoencoder upsamples less than a release's, so the expected size is the generator's own.
        let direct = try flux.image(forPrompt: "a red fox", width: 48, height: 32, seed: 5)
        XCTAssertEqual(image.width, direct.dim(2))
        XCTAssertEqual(image.height, direct.dim(1))
        XCTAssertEqual(flux.steps, 28, "the request's steps apply to that request only")
        XCTAssertEqual(flux.guidance, 4)
    }

    func testReferenceImagesConditionTheGeneration() throws {
        try requireMLXRuntime()
        let flux = try tinyFlux()
        let backend = flux.inferenceBackend()
        let parameters: [String: Any] = [NFKParameterWidth: 32, NFKParameterHeight: 32, NFKParameterSteps: 2,
                                         NFKParameterSeed: 5, NFKMLXFlux2ParameterKey.cachesReferences: true]
        let plain = try outputImage(try backend.runInference(
            for: NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox"], parameters: parameters)))
        let reference = try NFKMLXGeneratorBackendTestSupport.image(width: 40, height: 24)
        let conditioned = try outputImage(try backend.runInference(
            for: NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox", NFKInputImages: [reference]],
                                     parameters: parameters)))
        XCTAssertEqual(conditioned.width, plain.width, "only the generated tokens are decoded")
        XCTAssertEqual(conditioned.height, plain.height)
        XCTAssertNotEqual(conditioned.dataProvider?.data as Data?, plain.dataProvider?.data as Data?,
                          "a reference image changes the result")
        XCTAssertFalse(flux.cachesReferences, "the request's reference caching applies to that request only")
    }

    func testAnImageWithAMaskIsInpaintedAtTheImagesSize() throws {
        try requireMLXRuntime()
        let backend = try tinyFlux().inferenceBackend()
        let source = try NFKMLXGeneratorBackendTestSupport.image(width: 36, height: 20)
        let mask = try NFKMLXGeneratorBackendTestSupport.image(width: 36, height: 20)
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox", NFKInputImage: source,
                                                   NFKInputMask: mask],
                                          parameters: [NFKParameterSteps: 2, NFKParameterStrength: 1,
                                                       NFKParameterSeed: 5])
        let image = try outputImage(try backend.runInference(for: request))
        XCTAssertEqual(image.width, 32, "the image's sides round down to multiples of 16")
        XCTAssertEqual(image.height, 16)
    }

    func testAnImageWithoutAMaskIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyFlux().inferenceBackend()
        let source = try NFKMLXGeneratorBackendTestSupport.image(width: 32, height: 32)
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a red fox", NFKInputImage: source], parameters: [:])))
    }

    func testARequestWithoutAPromptIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyFlux().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }
}
