//
//  NFKMLXSANAPipelineTests.swift
//  InferKitMLXTests
//
//  The SANA text-to-image pipeline glue (linear-attention DiT + flow loop → DC-AE), on matching tiny
//  configurations so the chaining is exercised with random weights. Runs under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSANAPipelineTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func pipeline() -> NFKMLXSANAPipeline {
        let transformer = NFKMLXSANATransformerNet(.tiny)                 // inChannels 8, captionChannels 12
        let vae = NFKMLXDCAutoencoderNet(.tiny)                          // latentChannels 8
        return NFKMLXSANAPipeline(transformer: transformer, vae: vae)
    }

    func testThePipelineGeneratesAnImageOfTheExpectedShape() throws {
        try requireMLXRuntime()
        let prompt = MLXRandom.normal([6, 12])
        let image = pipeline().generate(promptEmbeds: prompt, negativeEmbeds: nil,
                                        latentHeight: 4, latentWidth: 4, steps: 3, guidance: 1)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.ndim, 4, "a [B, H, W, 3] image")
        XCTAssertEqual(image.shape[3], 3, "RGB output")
        XCTAssertEqual(image.shape[1], 8, "the DC-AE stage upsamples the 4×4 latent to 8×8")
    }

    func testClassifierFreeGuidanceRunsBothPrompts() throws {
        try requireMLXRuntime()
        let prompt = MLXRandom.normal([6, 12])
        let negative = MLXRandom.normal([6, 12])
        let image = pipeline().generate(promptEmbeds: prompt, negativeEmbeds: negative,
                                        latentHeight: 4, latentWidth: 4, steps: 2, guidance: 4.5)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.shape[3], 3)
    }
}
