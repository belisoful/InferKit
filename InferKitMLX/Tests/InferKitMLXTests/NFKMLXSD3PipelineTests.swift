//
//  NFKMLXSD3PipelineTests.swift
//  InferKitMLXTests
//
//  The SD3 / SD3.5 and FLUX.1 text-to-image pipeline glue (MMDiT / FLUX transformer + flow loop → VAE),
//  on matching tiny configurations so the chaining — the guided sampling loop, the timestep/latent
//  conventions, the latent packing (FLUX), the decode — is exercised with random weights. The DiTs are
//  validated at reference parity separately (NFKMLXReferenceParityTests). Runs where MLX has a Metal
//  library (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSD3PipelineTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func sd3VAE() -> NFKMLXSDAutoencoder {
        var config = NFKMLXSDVAEConfiguration()
        config.latentChannels = 4                                          // matches the tiny MMDiT's in_channels
        config.blockChannels = [8, 16]
        config.layersPerBlock = 1
        config.normalizationGroups = 4
        config.useQuantConv = true                                         // SD3's VAE keeps the quant convs
        config.scaleFactor = 1.5305
        config.shiftFactor = 0.0609
        return NFKMLXSDAutoencoder(configuration: config)
    }

    func testSD3PipelineGeneratesAnImageOfTheExpectedShape() throws {
        try requireMLXRuntime()
        let pipeline = NFKMLXSD3Pipeline(transformer: NFKMLXSD3TransformerNet(.tiny), vae: sd3VAE())
        let prompt = MLXRandom.normal([7, 24])                             // [L, jointDim]
        let pooled = MLXRandom.normal([20])                                // [pooledDim]
        let image = pipeline.generate(promptEmbeds: prompt, pooled: pooled, negativeEmbeds: nil,
                                      negativePooled: nil, latentHeight: 4, latentWidth: 4, steps: 3, guidance: 1)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.ndim, 4, "a [B, H, W, 3] image")
        XCTAssertEqual(image.shape[3], 3, "RGB output")
        XCTAssertEqual(image.shape[1], 8, "the two VAE levels upsample the 4×4 latent to 8×8")
    }

    func testSD3ClassifierFreeGuidanceRunsBothPrompts() throws {
        try requireMLXRuntime()
        let pipeline = NFKMLXSD3Pipeline(transformer: NFKMLXSD3TransformerNet(.tiny), vae: sd3VAE())
        let prompt = MLXRandom.normal([7, 24]), pooled = MLXRandom.normal([20])
        let negative = MLXRandom.normal([7, 24]), negativePooled = MLXRandom.normal([20])
        let image = pipeline.generate(promptEmbeds: prompt, pooled: pooled, negativeEmbeds: negative,
                                      negativePooled: negativePooled, latentHeight: 4, latentWidth: 4,
                                      steps: 2, guidance: 7)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.shape[3], 3)
    }

    private func fluxVAE() -> NFKMLXSDAutoencoder {
        var config = NFKMLXSDVAEConfiguration()
        config.latentChannels = 2                                          // packed to 8 = the tiny FLUX in_channels
        config.blockChannels = [8, 16]
        config.layersPerBlock = 1
        config.normalizationGroups = 4
        config.useQuantConv = false                                        // FLUX's VAE drops the quant convs
        config.scaleFactor = 0.3611
        config.shiftFactor = 0.1159
        return NFKMLXSDAutoencoder(configuration: config)
    }

    func testFluxPipelineGeneratesAnImageOfTheExpectedShape() throws {
        try requireMLXRuntime()
        let pipeline = NFKMLXFluxPipeline(transformer: NFKMLXFluxTransformerNet(.tiny), vae: fluxVAE())
        let prompt = MLXRandom.normal([5, 24])                             // [L, jointDim]
        let pooled = MLXRandom.normal([10])                                // [pooledDim]
        let image = pipeline.generate(promptEmbeds: prompt, pooled: pooled, latentHeight: 4, latentWidth: 4,
                                      steps: 3, guidance: 3.5)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.ndim, 4, "a [B, H, W, 3] image")
        XCTAssertEqual(image.shape[3], 3, "RGB output")
        XCTAssertEqual(image.shape[1], 8, "the two VAE levels upsample the 4×4 latent to 8×8")
    }

    func testFluxPackRoundTrips() throws {
        try requireMLXRuntime()
        let latent = MLXRandom.normal([1, 2, 6, 8])
        let packed = NFKMLXFluxPipeline.pack(latent)
        XCTAssertEqual(packed.shape, [1, 3 * 4, 8], "each 2×2 block folds into the channel axis")
        let restored = NFKMLXFluxPipeline.unpack(packed, channels: 2, height: 6, width: 8)
        eval(restored)
        let difference = abs(restored - latent).max().item(Float.self)
        XCTAssertLessThan(difference, 1e-6, "unpack is the inverse of pack")
    }

    // MARK: ControlNet pipelines

    func testSD3ControlNetPipelineGeneratesAnImageOfTheExpectedShape() throws {
        try requireMLXRuntime()
        let pipeline = NFKMLXSD3ControlNetPipeline(transformer: NFKMLXSD3TransformerNet(.tiny),
                                                   controlnet: NFKMLXSD3ControlNetNet(.tiny), vae: sd3VAE())
        let prompt = MLXRandom.normal([7, 24]), pooled = MLXRandom.normal([20])
        let control = MLXRandom.normal([1, 8, 8, 3])                        // the two VAE levels → a 4×4 latent
        let image = pipeline.generate(promptEmbeds: prompt, pooled: pooled, negativeEmbeds: nil,
                                      negativePooled: nil, controlImage: control, controlnetScale: 0.8,
                                      latentHeight: 4, latentWidth: 4, steps: 2, guidance: 1)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.shape[3], 3, "RGB output")
        XCTAssertEqual(image.shape[1], 8, "the two VAE levels upsample the 4×4 latent to 8×8")
    }

    func testFluxControlNetPipelineGeneratesAnImageOfTheExpectedShape() throws {
        try requireMLXRuntime()
        let pipeline = NFKMLXFluxControlNetPipeline(transformer: NFKMLXFluxTransformerNet(.tiny),
                                                    controlnet: NFKMLXFluxControlNetNet(.tiny), vae: fluxVAE())
        let prompt = MLXRandom.normal([5, 24]), pooled = MLXRandom.normal([10])
        let control = MLXRandom.normal([1, 8, 8, 3])
        let image = pipeline.generate(promptEmbeds: prompt, pooled: pooled, controlImage: control,
                                      controlnetScale: 0.7, latentHeight: 4, latentWidth: 4, steps: 2,
                                      guidance: 3.5)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.shape[3], 3, "RGB output")
        XCTAssertEqual(image.shape[1], 8, "the two VAE levels upsample the 4×4 latent to 8×8")
    }

    func testSD3ControlNetConfigurationReaderMatchesTheReleasedFields() throws {
        // The InstantX dual-stream shape (a joint_attention_dim present → context embedder).
        let instantX = try write("""
        {"_class_name":"SD3ControlNetModel","sample_size":128,"patch_size":2,"in_channels":16,
        "num_layers":12,"attention_head_dim":64,"num_attention_heads":24,"joint_attention_dim":4096,
        "caption_projection_dim":1536,"pooled_projection_dim":2048,"pos_embed_max_size":192,
        "extra_conditioning_channels":0}
        """)
        let a = try NFKMLXSD3ControlNetNet.configuration(fromHuggingFace: instantX)
        XCTAssertEqual(a.numLayers, 12)
        XCTAssertTrue(a.useContextEmbedder)
        XCTAssertTrue(a.usePosEmbed)
        XCTAssertEqual(a.extraConditioningChannels, 0)

        // The Stability 8B single-stream shape (no joint_attention_dim, use_pos_embed false).
        let stability = try write("""
        {"_class_name":"SD3ControlNetModel","sample_size":128,"patch_size":2,"in_channels":16,
        "num_layers":18,"attention_head_dim":64,"num_attention_heads":38,"pooled_projection_dim":2048,
        "pos_embed_max_size":192,"extra_conditioning_channels":1,"use_pos_embed":false,"qk_norm":"rms_norm"}
        """)
        let b = try NFKMLXSD3ControlNetNet.configuration(fromHuggingFace: stability)
        XCTAssertEqual(b.numLayers, 18)
        XCTAssertFalse(b.useContextEmbedder)
        XCTAssertFalse(b.usePosEmbed)
        XCTAssertEqual(b.extraConditioningChannels, 1)
        XCTAssertTrue(b.qkNorm)
    }

    func testFluxControlNetConfigurationReaderMatchesTheReleasedFields() throws {
        let url = try write("""
        {"_class_name":"FluxControlNetModel","patch_size":1,"in_channels":64,"num_layers":5,
        "num_single_layers":0,"attention_head_dim":128,"num_attention_heads":24,"joint_attention_dim":4096,
        "pooled_projection_dim":768,"guidance_embeds":true,"axes_dims_rope":[16,56,56],"num_mode":10}
        """)
        let configuration = try NFKMLXFluxControlNetNet.configuration(fromHuggingFace: url)
        XCTAssertEqual(configuration.numLayers, 5)
        XCTAssertEqual(configuration.numSingleLayers, 0)
        XCTAssertEqual(configuration.numMode, 10)
        XCTAssertTrue(configuration.guidanceEmbeds)
        XCTAssertEqual(configuration.axesDimsRope, [16, 56, 56])
    }

    // The configuration readers against the `transformer/config.json` field names. The SD3 case is a
    // synthetic config exercising every field the reader reads — the wide 2432 caption projection (SD3.5
    // large's), RMS qk-norm, and a dual_attention_layers list (SD3.5 medium's MMDiT-X path) — so the
    // list parsing and the qk-norm flag are both covered. FLUX.1 [dev]: 19 double blocks, 38 single
    // blocks, the guidance embedding, the (16,56,56) rope axes.
    private func write(_ json: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        try json.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    func testSD3ConfigurationReaderMatchesTheReleasedFields() throws {
        let url = try write("""
        {"_class_name":"SD3Transformer2DModel","sample_size":128,"patch_size":2,"in_channels":16,
        "num_layers":38,"attention_head_dim":64,"num_attention_heads":38,"joint_attention_dim":4096,
        "caption_projection_dim":2432,"pooled_projection_dim":2048,"out_channels":16,
        "pos_embed_max_size":192,"dual_attention_layers":[0,1,2,3,4,5,6,7,8,9,10,11,12],"qk_norm":"rms_norm"}
        """)
        let configuration = try NFKMLXSD3TransformerNet.configuration(fromHuggingFace: url)
        XCTAssertEqual(configuration.numLayers, 38)
        XCTAssertEqual(configuration.numAttentionHeads, 38)
        XCTAssertEqual(configuration.innerDim, 2432)
        XCTAssertEqual(configuration.captionProjectionDim, 2432)
        XCTAssertEqual(configuration.posEmbedMaxSize, 192)
        XCTAssertEqual(configuration.dualAttentionLayers, Array(0 ... 12))
        XCTAssertTrue(configuration.qkNorm)
    }

    func testFluxConfigurationReaderMatchesTheReleasedFields() throws {
        let url = try write("""
        {"_class_name":"FluxTransformer2DModel","patch_size":1,"in_channels":64,"num_layers":19,
        "num_single_layers":38,"attention_head_dim":128,"num_attention_heads":24,"joint_attention_dim":4096,
        "pooled_projection_dim":768,"guidance_embeds":true,"axes_dims_rope":[16,56,56]}
        """)
        let configuration = try NFKMLXFluxTransformerNet.configuration(fromHuggingFace: url)
        XCTAssertEqual(configuration.numLayers, 19)
        XCTAssertEqual(configuration.numSingleLayers, 38)
        XCTAssertEqual(configuration.innerDim, 3072)
        XCTAssertEqual(configuration.pooledProjectionDim, 768)
        XCTAssertTrue(configuration.guidanceEmbeds)
        XCTAssertEqual(configuration.axesDimsRope, [16, 56, 56])
    }
}
