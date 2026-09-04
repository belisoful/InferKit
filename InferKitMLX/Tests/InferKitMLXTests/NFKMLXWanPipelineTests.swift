//
//  NFKMLXWanPipelineTests.swift
//  InferKitMLXTests
//
//  The Wan text-to-video pipeline glue (DiT + flow loop → 3D causal VAE), on matching tiny
//  configurations so the chaining — the guided sampling loop, the DiT↔VAE bridge, the streaming decode —
//  is exercised with random weights. Runs under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXWanPipelineTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func pipeline() -> NFKMLXWanPipeline {
        let transformer = NFKMLXWanTransformerNet(.tiny)                 // inChannels 4, textDim 10
        let vae = NFKMLXWanVideoVAENet(.tiny)                          // zDim 4
        return NFKMLXWanPipeline(transformer: transformer, vae: vae)
    }

    func testThePipelineGeneratesAVideoOfTheExpectedShape() throws {
        try requireMLXRuntime()
        let text = MLXRandom.normal([6, 10])
        let video = pipeline().generate(textEmbeds: text, negativeEmbeds: nil,
                                        frames: 2, height: 4, width: 4, steps: 3, guidance: 1)
        eval(video)
        XCTAssertEqual(video.shape[0], 1)
        XCTAssertEqual(video.ndim, 5, "a [B, T, H, W, 3] clip")
        XCTAssertEqual(video.shape[4], 3, "RGB output")
        XCTAssertEqual(video.shape[2], 16, "the VAE upsamples the 4-wide latent by 4× spatially")
    }

    func testClassifierFreeGuidanceRunsBothPrompts() throws {
        try requireMLXRuntime()
        let text = MLXRandom.normal([6, 10])
        let negative = MLXRandom.normal([6, 10])
        let video = pipeline().generate(textEmbeds: text, negativeEmbeds: negative,
                                        frames: 2, height: 4, width: 4, steps: 2, guidance: 5)
        eval(video)
        XCTAssertEqual(video.shape[0], 1)
        XCTAssertEqual(video.shape[4], 3)
    }
}
