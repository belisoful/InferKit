//
//  NFKMLXStableDiffusionModelsTests.swift
//  InferKitMLXTests
//
//  The shared Stable Diffusion UNet and autoencoder at sizes a test can run, plus the pure key
//  remaps. The parity records prove the numbers against diffusers; these prove the geometry, the
//  remaps, and the configuration switches with no external files. The MLX forwards skip under
//  `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXStableDiffusionModelsTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    // MARK: Remaps (pure)

    func testTheRemapTranslatesTheUNetKeys() {
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapUNetKey(
            "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj.weight"),
            "down_blocks.0.attentions.1.transformer_blocks.0.ff.proj.weight")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapUNetKey(
            "mid_block.attentions.0.transformer_blocks.0.ff.net.2.bias"),
            "mid_block.attentions.0.transformer_blocks.0.ff.out.bias")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapUNetKey(
            "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.weight"),
            "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.weight")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapUNetKey(
            "down_blocks.1.attentions.0.proj_in.weight"),
            "down_blocks.1.attentions.0.proj_in_conv.weight")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapUNetKey("conv_in.weight"), "conv_in.weight")
    }

    func testTheRemapTranslatesTheVAEKeys() {
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapVAEKey(
            "decoder.mid_block.attentions.0.to_out.0.weight"),
            "decoder.mid_block.attentions.0.to_out.weight")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapVAEKey(
            "encoder.down_blocks.1.resnets.0.conv_shortcut.weight"),
            "encoder.down_blocks.1.resnets.0.block.conv_shortcut.weight")
        // The SD 1.5 autoencoders predate the diffusers attention rename.
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapVAEKey(
            "encoder.mid_block.attentions.0.query.weight"),
            "encoder.mid_block.attentions.0.to_q.weight")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapVAEKey(
            "decoder.mid_block.attentions.0.proj_attn.bias"),
            "decoder.mid_block.attentions.0.to_out.bias")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapVAEKey("quant_conv.weight"), "quant_conv.weight")
    }

    func testTheReleasedConfigurationsCarryTheirLoadBearingSwitches() {
        XCTAssertEqual(NFKMLXSDUNetConfiguration.inpainting.inputChannels, 9,
                       "noisy latent + mask + masked-image latent")
        XCTAssertTrue(NFKMLXSDUNetConfiguration.marigold.usesLinearProjection)
        XCTAssertEqual(NFKMLXSDUNetConfiguration.upscaler.attends, [false, true, true, true],
                       "the upscaler's plain level is its FIRST")
        XCTAssertEqual(NFKMLXSDUNetConfiguration.upscaler.onlyCrossAttention, [true, true, true, false],
                       "three levels put the context's width on their first attention too")
        XCTAssertEqual(NFKMLXSDUNetConfiguration.upscaler.classEmbeddingCount, 1000)
        XCTAssertEqual(NFKMLXSDVAEConfiguration.upscaler.blockChannels.count, 3,
                       "one level shallower is where the upscaler's ×4 comes from")
    }

    // MARK: Geometry (needs MLX)

    /// The released structure at a size a test can run, over two narrow levels. Widths stay multiples
    /// of the normalization's group count.
    static func tinyUNet(inputChannels: Int = 4, onlyCross: Bool = false,
                         classes: Int? = nil) -> NFKMLXSDUNetConfiguration {
        var configuration = NFKMLXSDUNetConfiguration()
        configuration.inputChannels = inputChannels
        configuration.blockChannels = [32, 64]
        configuration.attends = [true, false]
        configuration.attentionHeads = [2, 2]
        configuration.onlyCrossAttention = [onlyCross, false]
        configuration.crossAttentionDimensions = 16
        configuration.classEmbeddingCount = classes
        return configuration
    }

    func testUNetPredictsAtTheLatentShapeThroughEveryConditioningPath() throws {
        try requireMLXRuntime()
        let latent = MLXRandom.uniform(low: -1, high: 1, [1, 8, 8, 4])
        let context = MLXRandom.uniform(low: -1, high: 1, [1, 7, 16])
        for (name, configuration) in [("plain", Self.tinyUNet()),
                                      ("only-cross", Self.tinyUNet(onlyCross: true)),
                                      ("class-embedded", Self.tinyUNet(classes: 10))] {
            let net = NFKMLXSDUNet(configuration: configuration)
            net.train(false)
            let out = net(latent, timestep: MLXArray([Int32(201)]), context: context,
                          classLabel: configuration.classEmbeddingCount.map { _ in MLXArray([Int32(3)]) })
            eval(out)
            XCTAssertEqual(out.shape, [1, 8, 8, 4], "\(name) UNet keeps the latent shape")
        }
    }

    func testAutoencoderRoundTripsAnImageAtTheConfiguredStride() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXSDVAEConfiguration()
        configuration.blockChannels = [32, 64, 64]                // three levels → stride 4
        let net = NFKMLXSDAutoencoder(configuration: configuration)
        net.train(false)
        let image = MLXRandom.uniform(low: -1, high: 1, [1, 32, 32, 3])
        let (mean, logVariance) = net.encode(image)
        eval(mean, logVariance)
        XCTAssertEqual(mean.shape, [1, 8, 8, 4], "two downsampling levels give a stride of 4")
        XCTAssertEqual(logVariance.shape, mean.shape)
        let decoded = net.decode(mean)
        eval(decoded)
        XCTAssertEqual(decoded.shape, image.shape)
    }

    // MARK: Round trip (needs MLX)

    func testACheckpointRoundTripReproducesTheUNetPrediction() throws {
        try requireMLXRuntime()
        let trained = NFKMLXSDUNet(configuration: Self.tinyUNet())
        // Write the diffusers layout: the remapped names inverted, 4-D tensors rotated back.
        let referenceLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened()
            .map { key, value -> (String, MLXArray) in
                var name = key
                name = name.replacingOccurrences(of: ".ff.proj.", with: ".ff.net.0.proj.")
                name = name.replacingOccurrences(of: ".ff.out.", with: ".ff.net.2.")
                name = name.replacingOccurrences(of: ".to_out.", with: ".to_out.0.")
                name = name.replacingOccurrences(of: ".proj_in_conv.", with: ".proj_in.")
                name = name.replacingOccurrences(of: ".proj_out_conv.", with: ".proj_out.")
                return (name, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
            })
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sd-unet-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: referenceLayout, url: url)

        let loaded = NFKMLXSDUNet(configuration: Self.tinyUNet())
        try NFKMLXStableDiffusionModels.loadUNetWeights(into: loaded, from: url)

        let latent = MLXRandom.uniform(low: -1, high: 1, [1, 8, 8, 4], key: MLXRandom.key(5))
        let context = MLXRandom.uniform(low: -1, high: 1, [1, 7, 16], key: MLXRandom.key(6))
        let expected = trained(latent, timestep: MLXArray([Int32(101)]), context: context)
        let actual = loaded(latent, timestep: MLXArray([Int32(101)]), context: context)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }
}
