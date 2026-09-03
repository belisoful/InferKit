//
//  NFKMLXLTXPipelineTests.swift
//  InferKitMLXTests
//
//  The LTX text-to-video pipeline glue (T5 → DiT + flow loop → VAE), on matching tiny configurations so
//  the chaining — latent packing, the guided sampling loop, unpacking, decode — is exercised with random
//  weights. Runs under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLTXPipelineTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func pipeline() -> NFKMLXLTXPipeline {
        let text = NFKMLXT5Encoder.makeNet(NFKMLXT5Configuration(
            dModel: 32, layers: 2, heads: 2, keyDim: 16, ffDim: 64, vocabularySize: 128,
            relativeBuckets: 16, relativeMaxDistance: 32))
        let transformer = NFKMLXLTXTransformer.makeNet(NFKMLXLTXTransformerConfiguration(
            inChannels: 8, heads: 2, headDim: 8, layers: 2, crossAttentionDim: 16, captionChannels: 32))
        let vae = NFKMLXLTXVideoVAE.makeNet(NFKMLXLTXVAEConfiguration(
            latentChannels: 8, blockOutChannels: [8, 16, 16, 16], layersPerBlock: [1, 1, 1, 1, 1]))
        return NFKMLXLTXPipeline(textEncoder: text, transformer: transformer, vae: vae, latentChannels: 8)
    }

    func testThePipelineGeneratesAVideoOfTheExpectedShape() throws {
        try requireMLXRuntime()
        let prompt = MLXArray((0 ..< 4).map { Int32($0 + 1) }, [1, 4])
        let video = pipeline().generate(promptTokens: prompt, negativeTokens: nil,
                                        frames: 2, height: 2, width: 2, steps: 3, guidance: 1)
        // The VAE unpatchifies (×4) and upsamples the three scaling stages (×8) from the latent grid.
        XCTAssertEqual(video.shape[0], 1)
        XCTAssertEqual(video.ndim, 5, "a [B, T, H, W, 3] clip")
        XCTAssertEqual(video.shape[4], 3, "RGB output")
    }

    func testClassifierFreeGuidanceRunsBothPrompts() throws {
        try requireMLXRuntime()
        let prompt = MLXArray((0 ..< 4).map { Int32($0 + 1) }, [1, 4])
        let negative = MLXArray([Int32](repeating: 0, count: 4), [1, 4])
        let video = pipeline().generate(promptTokens: prompt, negativeTokens: negative,
                                        frames: 2, height: 2, width: 2, steps: 2, guidance: 3)
        XCTAssertEqual(video.shape[0], 1)
        XCTAssertEqual(video.shape[4], 3)
    }
}
