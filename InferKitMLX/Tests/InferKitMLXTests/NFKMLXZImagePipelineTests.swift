//
//  NFKMLXZImagePipelineTests.swift
//  InferKitMLXTests
//
//  The Z-Image text-to-image pipeline glue (S3-DiT + flow loop → Flux VAE), on matching tiny
//  configurations so the chaining — the guided sampling loop, the timestep/latent conventions, the
//  centered-latent decode — is exercised with random weights. Runs under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXZImagePipelineTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func pipeline() -> NFKMLXZImagePipeline {
        let transformer = NFKMLXZImageTransformerNet(.tiny)               // inChannels 4, captionFeatureDim 24
        var vaeConfig = NFKMLXSDVAEConfiguration()
        vaeConfig.latentChannels = 4
        vaeConfig.blockChannels = [8, 16]
        vaeConfig.layersPerBlock = 1
        vaeConfig.normalizationGroups = 4
        vaeConfig.useQuantConv = false
        vaeConfig.scaleFactor = 0.3611
        vaeConfig.shiftFactor = 0.1159
        let vae = NFKMLXSDAutoencoder(configuration: vaeConfig)
        return NFKMLXZImagePipeline(transformer: transformer, vae: vae)
    }

    func testThePipelineGeneratesAnImageOfTheExpectedShape() throws {
        try requireMLXRuntime()
        let prompt = MLXRandom.normal([6, 24])
        let image = pipeline().generate(promptEmbeds: prompt, negativeEmbeds: nil,
                                        latentHeight: 4, latentWidth: 4, steps: 3, guidance: 1)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.ndim, 4, "a [B, H, W, 3] image")
        XCTAssertEqual(image.shape[3], 3, "RGB output")
        XCTAssertEqual(image.shape[1], 8, "the two VAE levels upsample the 4×4 latent to 8×8")
    }

    func testClassifierFreeGuidanceRunsBothPrompts() throws {
        try requireMLXRuntime()
        let prompt = MLXRandom.normal([6, 24])
        let negative = MLXRandom.normal([6, 24])
        let image = pipeline().generate(promptEmbeds: prompt, negativeEmbeds: negative,
                                        latentHeight: 4, latentWidth: 4, steps: 2, guidance: 4)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.shape[3], 3)
    }

    func testImageToImageGeneratesFromASource() throws {
        try requireMLXRuntime()
        let source = MLXRandom.normal([1, 8, 8, 3])                        // encodes to a 4×4 latent
        let prompt = MLXRandom.normal([6, 24])
        let image = pipeline().generate(image: source, promptEmbeds: prompt, negativeEmbeds: nil,
                                        strength: 0.6, steps: 3, guidance: 1)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.ndim, 4, "a [B, H, W, 3] image")
        XCTAssertEqual(image.shape[3], 3, "RGB output")
    }
}
