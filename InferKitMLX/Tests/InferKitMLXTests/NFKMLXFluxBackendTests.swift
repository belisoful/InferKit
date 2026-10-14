//
//  NFKMLXFluxBackendTests.swift
//  InferKitMLXTests
//
//  FLUX.1 behind the InferKit contract: the request's prompt and parameters reach the model, the image
//  comes back under NFKOutputImage, and a request's settings do not outlive it.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXFluxBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tinyFlux() throws -> NFKMLXFlux {
        var clip = NFKMLXSDTextEncoderConfiguration.tiny
        clip.contextLength = 77
        let t5 = NFKMLXT5Configuration.tiny
        let geometry = NFKMLXFluxConfiguration(
            inChannels: 8, outChannels: 8, numLayers: 1, numSingleLayers: 1, attentionHeadDim: 6,
            numAttentionHeads: 2, jointAttentionDim: t5.dModel, pooledProjectionDim: clip.width,
            guidanceEmbeds: false, axesDimsRope: [2, 2, 2])
        var vae = NFKMLXSDVAEConfiguration()
        vae.latentChannels = 2
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4
        vae.useQuantConv = false
        let tokenizer = try NFKMLXGeneratorBackendTestSupport.clipTokenizer(self)
        let segmenter = try NFKMLXGeneratorBackendTestSupport.t5Segmenter()
        MLXRandom.seed(19)
        let flux = NFKMLXFlux(
            resident: true,
            loadTextEncoder: {
                NFKMLXFluxTextEncoder(clip: NFKMLXSDTextEncoderNet(configuration: clip), clipTokenizer: tokenizer,
                                      t5: NFKMLXT5Encoder(net: NFKMLXT5Encoder.makeNet(t5)), t5Segmenter: segmenter,
                                      t5Context: 8)
            },
            loadPipeline: {
                NFKMLXFluxPipeline(transformer: NFKMLXFluxTransformerNet(geometry),
                                   vae: NFKMLXSDAutoencoder(configuration: vae))
            })
        try flux.loadResident()
        return flux
    }

    func testARequestBecomesAnImageAtTheRequestedSize() throws {
        try requireMLXRuntime()
        let flux = try tinyFlux()
        let backend = flux.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXFlux.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt])

        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox"],
                                          parameters: [NFKParameterWidth: 32, NFKParameterHeight: 16,
                                                       NFKParameterSteps: 1, NFKParameterGuidanceScale: 2,
                                                       NFKParameterSeed: 5])
        let result = try backend.runInference(for: request)
        let image = try XCTUnwrap(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(CFGetTypeID(image as CFTypeRef), CGImage.typeID)
        let cgImage = image as! CGImage
        // The tiny autoencoder upsamples less than a release's, so the expected size is the generator's own.
        let direct = try flux.image(forPrompt: "a red fox", width: 32, height: 16, seed: 5)
        XCTAssertEqual(cgImage.width, direct.dim(2))
        XCTAssertEqual(cgImage.height, direct.dim(1))
        XCTAssertEqual(flux.steps, 4, "the request's steps apply to that request only")
        XCTAssertEqual(flux.guidance, 3.5)
    }

    func testARequestWithoutAPromptIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyFlux().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }
}
