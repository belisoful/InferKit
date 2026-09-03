//
//  NFKMLXLTXVideoVAETests.swift
//  InferKitMLXTests
//
//  The LTX-Video causal 3D VAE. Encode/decode evaluate MLX arrays, so they skip under `swift test` and
//  run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLTXVideoVAETests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    func testParameterNamesFollowTheReferenceLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXLTXVideoVAE.makeNet(.tiny).parameters().flattened().map(\.0))
        for expected in ["encoder.conv_in.conv.weight", "encoder.down_blocks.0.resnets.0.conv1.conv.weight",
                         "encoder.down_blocks.0.downsamplers.0.conv.weight", "encoder.down_blocks.0.conv_out.norm3.weight",
                         "encoder.mid_block.resnets.0.conv1.conv.weight", "encoder.conv_out.conv.weight",
                         "decoder.conv_in.conv.weight", "decoder.up_blocks.0.resnets.0.conv1.conv.weight",
                         "decoder.conv_out.conv.weight", "latents_mean", "latents_std"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testEncodeCompressesAndDecodeRestoresTheShape() throws {
        try requireMLXRuntime()
        let net = NFKMLXLTXVideoVAE.makeNet(.tiny)
        // 9 frames, 32×32; patch 4 + three spatiotemporal downsamples.
        let video = MLXRandom.uniform(low: -1, high: 1, [1, 9, 32, 32, 3])
        let latent = net.encode(video)
        XCTAssertEqual(latent.shape, [1, 2, 1, 1, 8], "spatiotemporal compression to the latent")
        let reconstruction = net.decode(latent)
        XCTAssertEqual(reconstruction.shape, [1, 9, 32, 32, 3], "the decoder restores the video shape")
    }

    func testTheEncoderIsCausalInTime() throws {
        try requireMLXRuntime()
        // A causal encoder's early latents cannot depend on later frames: two clips that share their
        // first eight frames and differ only in the last must produce the same first latent frame.
        let net = NFKMLXLTXVideoVAE.makeNet(.tiny)
        let shared = MLXRandom.uniform(low: -1, high: 1, [1, 8, 32, 32, 3])
        let videoA = concatenated([shared, MLXRandom.uniform(low: -1, high: 1, [1, 1, 32, 32, 3])], axis: 1)
        let videoB = concatenated([shared, MLXRandom.uniform(low: -1, high: 1, [1, 1, 32, 32, 3])], axis: 1)
        let firstA = net.encode(videoA)[0, 0], firstB = net.encode(videoB)[0, 0]
        eval(firstA, firstB)
        XCTAssertEqual(firstA.reshaped([-1]).asArray(Float.self), firstB.reshaped([-1]).asArray(Float.self),
                       "the first latent frame is independent of the last input frame")
    }

    func testTheObjectEncodeDecodeRoundTripsShape() throws {
        try requireMLXRuntime()
        let vae = try NFKMLXLTXVideoVAE.vae(configuration: .tiny, weightsURL: nil)
        let video = MLXRandom.uniform(low: -1, high: 1, [9, 32, 32, 3])       // unbatched
        let reconstruction = vae.decode(vae.encode(video))
        XCTAssertEqual(reconstruction.shape, [1, 9, 32, 32, 3])
    }
}
