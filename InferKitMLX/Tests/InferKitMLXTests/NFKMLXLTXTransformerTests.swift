//
//  NFKMLXLTXTransformerTests.swift
//  InferKitMLXTests
//
//  The LTX-Video DiT. The forward evaluates MLX arrays, so it skips under `swift test` and runs under
//  `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLTXTransformerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    func testParameterNamesFollowTheReferenceLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXLTXTransformer.makeNet(.tiny).parameters().flattened().map(\.0))
        for expected in ["proj_in.weight", "time_embed.emb.timestep_embedder.linear_1.weight",
                         "time_embed.linear.weight", "caption_projection.linear_1.weight", "scale_shift_table",
                         "transformer_blocks.0.attn1.to_q.weight", "transformer_blocks.0.attn1.norm_q.weight",
                         "transformer_blocks.0.attn2.to_k.weight", "transformer_blocks.0.ff.net.0.proj.weight",
                         "transformer_blocks.0.ff.net.2.weight", "transformer_blocks.0.scale_shift_table",
                         "proj_out.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testForwardPredictsAVelocityPerToken() throws {
        try requireMLXRuntime()
        let c = NFKMLXLTXTransformerConfiguration.tiny
        let net = NFKMLXLTXTransformer.makeNet(c)
        let grid = (2, 2, 2), tokens = 8
        let latent = MLXRandom.normal([1, tokens, c.inChannels])
        let text = MLXRandom.normal([1, 4, c.captionChannels])
        let velocity = net(latent, text: text, timestep: MLXArray([Float(500)]), grid: grid, ropeScale: (1, 1, 1))
        XCTAssertEqual(velocity.shape, [1, tokens, c.inChannels], "one velocity vector per latent token")
    }

    func testTheRotaryEmbeddingHasTheTransformerWidth() throws {
        try requireMLXRuntime()
        let net = NFKMLXLTXTransformer.makeNet(.tiny)
        let (cos, sin) = net.rotary.embedding(frames: 2, height: 2, width: 2, scale: (1, 1, 1))
        XCTAssertEqual(cos.shape, [1, 8, NFKMLXLTXTransformerConfiguration.tiny.innerDim])
        XCTAssertEqual(sin.shape, cos.shape)
    }
}
