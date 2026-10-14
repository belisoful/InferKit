//
//  NFKMLXSD3BackendTests.swift
//  InferKitMLXTests
//
//  Stable Diffusion 3 behind the InferKit contract: the request's prompt and parameters reach the
//  generator, the image comes back under NFKOutputImage, and a request's settings do not outlive it.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSD3BackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tower(projection: Int, activation: NFKCLIPActivation) -> NFKMLXSDTextEncoderConfiguration {
        var configuration = NFKMLXSDTextEncoderConfiguration()
        configuration.width = 8
        configuration.layers = 3
        configuration.heads = 2
        configuration.intermediate = 16
        configuration.activation = activation
        configuration.output = .penultimateHiddenState
        configuration.projectionDimensions = projection
        return configuration
    }

    private func tinyGenerator() throws -> NFKMLXSD3Generator {
        let clipL = tower(projection: 8, activation: .quickGELU)
        let clipG = tower(projection: 12, activation: .gelu)
        let t5 = NFKMLXT5Configuration(dModel: 24, layers: 2, heads: 2, keyDim: 12, ffDim: 48,
                                       vocabularySize: 32128, relativeBuckets: 16, relativeMaxDistance: 32)
        var vae = NFKMLXSDVAEConfiguration()
        vae.latentChannels = 4
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4
        vae.useQuantConv = false
        vae.scaleFactor = 1.5305
        vae.shiftFactor = 0.0609
        let tokenizer = try NFKMLXGeneratorBackendTestSupport.clipTokenizer(self)
        MLXRandom.seed(17)
        let generator = NFKMLXSD3Generator(
            resident: true, tokenizerL: tokenizer, tokenizerG: tokenizer,
            segmenter: try NFKMLXGeneratorBackendTestSupport.t5Segmenter(), t5Length: 32,
            loadTextEncoders: {
                NFKSD3TextStage(clipL: NFKMLXSDTextEncoderNet(configuration: clipL),
                                clipG: NFKMLXSDTextEncoderNet(configuration: clipG),
                                t5: NFKMLXT5Encoder.makeNet(t5), jointDimensions: 24, t5Length: 32)
            },
            loadPipeline: {
                NFKMLXSD3Pipeline(transformer: NFKMLXSD3TransformerNet(.tiny),
                                  vae: NFKMLXSDAutoencoder(configuration: vae))
            })
        try generator.loadResident()
        return generator
    }

    func testARequestBecomesAnImageAtTheRequestedSize() throws {
        try requireMLXRuntime()
        let generator = try tinyGenerator()
        let backend = generator.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXSD3Generator.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputNegativePrompt])

        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox", NFKInputNegativePrompt: "a fox"],
                                          parameters: [NFKParameterWidth: 32, NFKParameterHeight: 16,
                                                       NFKParameterSteps: 2, NFKParameterGuidanceScale: 5,
                                                       NFKParameterSeed: 3])
        let result = try backend.runInference(for: request)
        let image = try XCTUnwrap(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(CFGetTypeID(image as CFTypeRef), CGImage.typeID)
        let cgImage = image as! CGImage
        XCTAssertEqual(cgImage.width, 32)
        XCTAssertEqual(cgImage.height, 16)
        XCTAssertEqual(generator.steps, 28, "the request's steps apply to that request only")
        XCTAssertEqual(generator.guidance, 7)
    }

    func testARequestWithoutAPromptIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyGenerator().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }
}
