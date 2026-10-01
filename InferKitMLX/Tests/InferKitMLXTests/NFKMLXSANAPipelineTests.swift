//
//  NFKMLXSANAPipelineTests.swift
//  InferKitMLXTests
//
//  The SANA text-to-image pipeline glue (linear-attention DiT + flow loop → DC-AE), on matching
//  tiny configurations so the chaining is exercised with random weights. Runs where MLX has a Metal
//  library (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSANAPipelineTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
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

    // A diffusers transformer/ directory: its config.json read into the geometry, and its weights,
    // saved in the release's [out, in, kH, kW] layout, reproducing the network they came from.
    func testATransformerDirectoryLoadsAtItsOwnGeometry() throws {
        try requireMLXRuntime()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config: [String: Any] = [
            "_class_name": "SanaTransformer2DModel", "in_channels": 8, "out_channels": 8,
            "num_attention_heads": 2, "attention_head_dim": 8, "num_layers": 3,
            "num_cross_attention_heads": 2, "cross_attention_head_dim": 8, "cross_attention_dim": 16,
            "caption_channels": 12, "mlp_ratio": 2.0, "patch_size": 1, "norm_eps": 1e-6,
        ]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        let geometry = try NFKMLXSANATransformerNet.configuration(
            fromHuggingFace: directory.appendingPathComponent("config.json"))
        XCTAssertEqual(geometry.layers, 3)
        XCTAssertEqual(geometry.captionChannels, 12)

        let trained = NFKMLXSANATransformerNet(geometry)
        let released = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        try save(arrays: released, url: directory.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        let loaded = NFKMLXSANATransformerNet(geometry)
        try NFKMLXSANATransformerNet.loadWeights(into: loaded, fromDirectory: directory)

        let latent = MLXRandom.normal([8, 4, 4])
        let caption = MLXRandom.normal([5, 12])
        let timestep = MLXArray([Float(500)])
        let difference = abs(trained(latent, capFeats: caption, t: timestep)
                             - loaded(latent, capFeats: caption, t: timestep)).max().item(Float.self)
        XCTAssertEqual(difference, 0, "the loaded transformer is the one saved")
    }

    func testAReleaseWithAPartThePortLacksIsRefused() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        for extra in [["guidance_embeds": true], ["qk_norm": "rms_norm_across_heads"], ["interpolation_scale": 1]]
            as [[String: Any]] {
            try JSONSerialization.data(withJSONObject: extra.merging(["num_layers": 2]) { a, _ in a }).write(to: url)
            XCTAssertThrowsError(try NFKMLXSANATransformerNet.configuration(fromHuggingFace: url), "\(extra)")
        }
    }
}
