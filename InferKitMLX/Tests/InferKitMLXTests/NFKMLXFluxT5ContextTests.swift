//
//  NFKMLXFluxT5ContextTests.swift
//  InferKitMLXTests
//
//  The T5 sequence length a FLUX.1 release encodes at follows its transformer, as diffusers'
//  FluxPipeline runs it: 512 tokens for a guidance-embedding ([dev]) transformer, 256 for [schnell].
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXFluxT5ContextTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tinyTransformer(guidanceEmbeds: Bool) -> NFKMLXFluxConfiguration {
        NFKMLXFluxConfiguration(
            inChannels: 8, outChannels: 8, numLayers: 1, numSingleLayers: 1, attentionHeadDim: 6,
            numAttentionHeads: 2, jointAttentionDim: NFKMLXT5Configuration.tiny.dModel,
            pooledProjectionDim: NFKMLXSDTextEncoderConfiguration.tiny.width,
            guidanceEmbeds: guidanceEmbeds, axesDimsRope: [2, 2, 2])
    }

    private func encodedShape(for transformer: NFKMLXFluxConfiguration) throws -> [Int] {
        var clip = NFKMLXSDTextEncoderConfiguration.tiny
        clip.contextLength = 77
        MLXRandom.seed(23)
        let encoder = NFKMLXFluxTextEncoder(
            clip: NFKMLXSDTextEncoderNet(configuration: clip),
            clipTokenizer: try NFKMLXGeneratorBackendTestSupport.clipTokenizer(self),
            t5: NFKMLXT5Encoder(net: NFKMLXT5Encoder.makeNet(.tiny)),
            t5Segmenter: try NFKMLXGeneratorBackendTestSupport.t5Segmenter(),
            t5Context: NFKMLXFluxTextEncoder.t5Context(for: transformer))
        return encoder.encode(prompt: "a red fox").promptEmbeds.shape
    }

    func testTheReleasedTransformersPickTheirT5Length() throws {
        try requireMLXRuntime()
        XCTAssertEqual(NFKMLXFluxTextEncoder.t5Context(for: .dev), 512)
        XCTAssertEqual(NFKMLXFluxTextEncoder.t5Context(for: .schnell), 256)
    }

    func testADevGeometryEncodesFiveHundredTwelveTokens() throws {
        try requireMLXRuntime()
        XCTAssertEqual(try encodedShape(for: tinyTransformer(guidanceEmbeds: true)),
                       [512, NFKMLXT5Configuration.tiny.dModel])
    }

    func testASchnellGeometryEncodesTwoHundredFiftySixTokens() throws {
        try requireMLXRuntime()
        XCTAssertEqual(try encodedShape(for: tinyTransformer(guidanceEmbeds: false)),
                       [256, NFKMLXT5Configuration.tiny.dModel])
    }
}
