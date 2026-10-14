//
//  NFKMLXSD3ControlNetBackendTests.swift
//  InferKitMLXTests
//
//  SD3 ControlNet behind the InferKit contract, at a tiny random geometry without T5: a prompt and a
//  control image become an image at the requested size, the conditioning scale is a request key, and a
//  request's settings do not outlive it.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSD3ControlNetBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tower(projection: Int, activation: NFKCLIPActivation) -> NFKMLXSDTextEncoderConfiguration {
        var configuration = NFKMLXSDTextEncoderConfiguration()
        configuration.width = 8
        configuration.layers = 2
        configuration.heads = 2
        configuration.intermediate = 16
        configuration.activation = activation
        configuration.output = .penultimateHiddenState
        configuration.projectionDimensions = projection
        return configuration
    }

    /// The tiny MMDiT (joint width 24, pooled 20) under a tiny dual-stream ControlNet.
    private func tinyGenerator() throws -> NFKMLXSD3ControlNetGenerator {
        MLXRandom.seed(29)
        let clipL = tower(projection: 8, activation: .quickGELU)
        let clipG = tower(projection: 12, activation: .gelu)
        var vae = NFKMLXSDVAEConfiguration()
        vae.latentChannels = 4
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4
        vae.useQuantConv = false
        let tokenizer = try NFKMLXFluxControlNetBackendTests.clipTokenizer(self)
        let generator = NFKMLXSD3ControlNetGenerator(
            resident: true, tokenizerL: tokenizer, tokenizerG: tokenizer, segmenter: nil, t5Length: 8,
            pixelsPerLatent: 1 << (vae.blockChannels.count - 1),
            loadTextEncoders: {
                NFKSD3TextStage(clipL: NFKMLXSDTextEncoderNet(configuration: clipL),
                                clipG: NFKMLXSDTextEncoderNet(configuration: clipG), t5: nil,
                                jointDimensions: NFKMLXSD3Configuration.tiny.jointAttentionDim, t5Length: 8)
            },
            loadPipeline: {
                NFKMLXSD3ControlNetPipeline(transformer: NFKMLXSD3TransformerNet(.tiny),
                                            controlnet: NFKMLXSD3ControlNetNet(.tiny),
                                            vae: NFKMLXSDAutoencoder(configuration: vae))
            })
        try generator.loadResident()
        return generator
    }

    func testAPromptAndAControlImageBecomeAnImageAtTheRequestedSize() throws {
        try requireMLXRuntime()
        let generator = try tinyGenerator()
        let backend = generator.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXSD3ControlNetGenerator.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputNegativePrompt, NFKInputControl])
        XCTAssertFalse(backend.supportedParameterKeys?.contains(NFKMLXControlNetParameterKey.controlMode) ?? false)

        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a red fox", NFKInputNegativePrompt: "a fox",
                     NFKInputControl: NFKMLXFluxControlNetBackendTests.plate(width: 30, height: 20)],
            parameters: [NFKParameterWidth: 16, NFKParameterHeight: 12, NFKParameterSteps: 2,
                         NFKParameterGuidanceScale: 3, NFKParameterSeed: 2,
                         NFKMLXControlNetParameterKey.conditioningScale: 0.7])
        let image = try NFKMLXFluxControlNetBackendTests.cgImage(
            backend.runInference(for: request).output(forKey: NFKOutputImage))
        XCTAssertEqual(image.width, 16)
        XCTAssertEqual(image.height, 12)
        XCTAssertEqual(generator.steps, 28, "the request's steps apply to that request only")
        XCTAssertEqual(generator.guidance, 7)
        XCTAssertEqual(generator.conditioningScale, 1)
    }

    func testASideOffThePatchGridIsRefused() throws {
        try requireMLXRuntime()
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a fox", NFKInputControl: NFKMLXFluxControlNetBackendTests.plate(width: 16, height: 16)],
            parameters: [NFKParameterWidth: 14, NFKParameterHeight: 16])
        XCTAssertThrowsError(try tinyGenerator().inferenceBackend().runInference(for: request))
    }

    func testARequestWithoutAPromptOrAControlImageIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyGenerator().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a fox"], parameters: [:])))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputControl: NFKMLXFluxControlNetBackendTests.plate(width: 16, height: 16)], parameters: [:])))
    }
}
