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
import MLXNN
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

    // The FLUX.2 pipeline at a tiny random geometry: the denoising loop, the latent codec's round
    // trip, and the autoencoder's decode chained. Weight-free, so it asserts the contract rather than
    // a picture: the shape the caller asked for, a range the image bridge accepts, and determinism
    // under a seed. The arithmetic of each stage is measured separately in the parity tests.
    // The residency plan, from the stored sizes of the released klein 4B and base 9B and the budget a
    // 32 GB machine gets (0.85 of Metal's recommended 25 GiB working set).
    func testFlux2PlansResidencyAgainstTheWorkingSet() throws {
        func bytes(_ gib: Double) -> Int { Int(gib * 1_073_741_824) }
        let budget = bytes(21.25)
        func plan(_ encoder: Double, _ pipeline: Double, _ budget: Int,
                  _ residency: NFKMLXResidency) throws -> (Bool, Bool) {
            let placement = try NFKMLXFlux2.plan(encoderStoredBytes: bytes(encoder),
                                                 pipelineStoredBytes: bytes(pipeline), budget: budget,
                                                 residency: residency)
            return (placement.resident, placement.encoderFloat32)
        }
        // klein 4B: a 7.5 GiB encoder (15.0 at float32) and a 7.4 GiB transformer and autoencoder.
        XCTAssertTrue(try plan(7.5, 7.37, budget, .automatic) == (true, false),
                      "klein 4B stays resident with its encoder as stored; at float32 it would not fit beside the transformer")
        XCTAssertTrue(try plan(7.5, 7.37, budget, .staged) == (false, true),
                      "staged, the encoder alone fits at float32")
        // base 9B: a 15.3 GiB encoder (30.5 at float32) and a 17.07 GiB transformer and autoencoder.
        XCTAssertTrue(try plan(15.26, 17.07, budget, .automatic) == (false, false),
                      "klein 9B is staged, its encoder as stored")
        XCTAssertThrowsError(try plan(15.26, 17.07, budget, .resident), "klein 9B cannot be held whole here")
        XCTAssertTrue(try plan(15.26, 17.07, bytes(40.8), .automatic) == (true, false),
                      "a 64 GB machine holds klein 9B resident, its encoder as stored")
        XCTAssertTrue(try plan(15.26, 17.07, 0, .automatic) == (false, false),
                      "a machine that reports no budget stages, its encoder as stored, rather than gambling")
        XCTAssertTrue(try plan(15.26, 17.07, 0, .resident) == (true, false),
                      "asked to hold it resident, such a machine is not refused")
        XCTAssertThrowsError(try plan(7.5, 20, budget, .automatic),
                             "a transformer that does not fit on its own fails in every placement")
    }

    // Staging changes when the stages are loaded, not what they compute: a staged facade and a resident
    // one over the same weights produce the same image, and a staged one holds neither stage after it.
    func testFlux2StagingReleasesEachStageAndChangesNothing() throws {
        try requireMLXRuntime()
        guard let tokenizerPath = NFKMLXValidationConfig.environment["IK_VAL_FLUX2_TOKENIZER"],
              let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: URL(fileURLWithPath: tokenizerPath)) else {
            throw XCTSkip("set IK_VAL_FLUX2_TOKENIZER (the klein 4B tokenizer directory)")
        }
        let template = try? String(contentsOf: URL(fileURLWithPath: tokenizerPath)
                                    .appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        var language = NFKMLXLanguageConfiguration.tiny
        language.vocabularySize = 151_936                                // the Qwen3 tokenizer's id range
        var geometry = NFKMLXFlux2Configuration.tiny
        geometry.inChannels = 4 * 2 * 2
        geometry.jointAttentionDim = 2 * language.hiddenSize             // two layers read
        geometry.guidanceEmbeds = false
        var vae = NFKMLXSDVAEConfiguration.flux2
        vae.latentChannels = 4
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4

        // One set of weights, applied to a fresh module on every load, so a staged reload reads the
        // same network a resident load holds.
        func captured(_ module: Module) -> [(String, MLXArray)] {
            let parameters = module.parameters().flattened()
            eval(parameters.map(\.1))
            return parameters
        }
        let decoderWeights = captured(NFKMLXLanguage.makeNet(language))
        let transformerWeights = captured(NFKMLXFlux2TransformerNet(geometry))
        let autoencoderWeights = captured(NFKMLXSDAutoencoder(configuration: vae))
        var loads = (encoder: 0, pipeline: 0)
        func facade(resident: Bool) throws -> NFKMLXFlux2 {
            // Not distilled, so the release guides and both prompts go through one encoder stage.
            let flux = NFKMLXFlux2(
                resident: resident,
                loadTextEncoder: {
                    loads.encoder += 1
                    let decoder = NFKMLXLanguage.makeNet(language)
                    try NFKMLXWeights.apply(decoderWeights, to: decoder)
                    return NFKMLXFlux2TextEncoder(decoder: decoder, layers: [1, 2], contextLength: 32)
                },
                loadPipeline: {
                    loads.pipeline += 1
                    let transformer = NFKMLXFlux2TransformerNet(geometry)
                    try NFKMLXWeights.apply(transformerWeights, to: transformer)
                    let autoencoder = NFKMLXSDAutoencoder(configuration: vae)
                    try NFKMLXWeights.apply(autoencoderWeights, to: autoencoder)
                    return NFKMLXFlux2Pipeline(transformer: transformer, autoencoder: autoencoder,
                                               codec: NFKMLXFlux2LatentCodec(patchedChannels: 16,
                                                                             epsilon: 1e-4, patch: 2))
                },
                tokenizer: tokenizer, chatTemplate: template, isDistilled: false, encodesInFloat32: true)
            if resident {
                try flux.loadResident()
            }
            flux.steps = 2
            return flux
        }

        let resident = try facade(resident: true)
        let residentImages = try (0 ..< 2).map { _ in
            try resident.image(forPrompt: "a red fox in the snow", width: 32, height: 32, seed: 5)
        }
        XCTAssertEqual(loads.encoder, 1, "a resident facade loads the encoder once")
        XCTAssertEqual(loads.pipeline, 1, "and the pipeline once")
        XCTAssertTrue(resident.isHoldingTextEncoder && resident.isHoldingPipeline)

        loads = (0, 0)
        let staged = try facade(resident: false)
        XCTAssertFalse(staged.isHoldingTextEncoder || staged.isHoldingPipeline, "a staged facade loads nothing up front")
        let stagedImages = try (0 ..< 2).map { _ in
            try staged.image(forPrompt: "a red fox in the snow", width: 32, height: 32, seed: 5)
        }
        XCTAssertEqual(loads.encoder, 2, "a staged facade loads the encoder once per image, both prompts together")
        XCTAssertEqual(loads.pipeline, 2, "and the pipeline once per image")
        XCTAssertFalse(staged.isHoldingTextEncoder || staged.isHoldingPipeline, "and holds neither after")

        for (residentImage, stagedImage) in zip(residentImages, stagedImages) {
            XCTAssertEqual(stagedImage.reshaped([-1]).asArray(Float.self),
                           residentImage.reshaped([-1]).asArray(Float.self),
                           "staging does not change the image")
        }
    }

    func testFlux2PipelineGeneratesAnImageOfTheExpectedShape() throws {
        try requireMLXRuntime()
        var vae = NFKMLXSDVAEConfiguration.flux2
        vae.latentChannels = 4
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4

        var geometry = NFKMLXFlux2Configuration.tiny
        geometry.inChannels = 4 * 2 * 2                                  // the codec's 2×2 patch
        let pipeline = NFKMLXFlux2Pipeline(
            transformer: NFKMLXFlux2TransformerNet(geometry),
            autoencoder: NFKMLXSDAutoencoder(configuration: vae),
            codec: NFKMLXFlux2LatentCodec(patchedChannels: 16, epsilon: 1e-4, patch: 2))

        let embeds = MLXArray.zeros([1, 5, geometry.jointAttentionDim])
        let image = pipeline.generate(promptEmbeds: embeds, latentHeight: 2, latentWidth: 3,
                                      steps: 2, seed: 7)
        eval(image)
        // The expected size is DERIVED from the configuration rather than written down: the codec
        // unpacks its `patch`×`patch` fold, and the autoencoder upsamples once per level after the
        // first, so a two-level autoencoder strides by 2 and the released four-level one by 8.
        let stride = 1 << (vae.blockChannels.count - 1)
        XCTAssertEqual(image.shape, [1, 2 * 2 * stride, 3 * 2 * stride, 3],
                       "the decoded image covers the requested grid")
        let values = image.reshaped([-1]).asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { $0 >= 0 && $0 <= 1 }, "the pipeline returns [0, 1] pixels")

        let again = pipeline.generate(promptEmbeds: embeds, latentHeight: 2, latentWidth: 3,
                                      steps: 2, seed: 7)
        eval(again)
        XCTAssertEqual(again.reshaped([-1]).asArray(Float.self), values,
                       "the same seed reproduces the same image")

        let other = pipeline.generate(promptEmbeds: embeds, latentHeight: 2, latentWidth: 3,
                                      steps: 2, seed: 8)
        eval(other)
        XCTAssertNotEqual(other.reshaped([-1]).asArray(Float.self), values,
                          "a different seed starts from different noise")
    }

    // Reference-image conditioning: FLUX.2 edits by appending an encoded image's latent to the token
    // sequence under its own TIME coordinate, and keeping only the generated tokens from what the
    // transformer returns. Without the time offset a reference patch and a generated patch at the
    // same row and column would carry the same rotary position.
    func testFlux2ReferenceImagesConditionTheGeneration() throws {
        try requireMLXRuntime()
        var vae = NFKMLXSDVAEConfiguration.flux2
        vae.latentChannels = 4
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4
        var geometry = NFKMLXFlux2Configuration.tiny
        geometry.inChannels = 16
        let pipeline = NFKMLXFlux2Pipeline(
            transformer: NFKMLXFlux2TransformerNet(geometry),
            autoencoder: NFKMLXSDAutoencoder(configuration: vae),
            codec: NFKMLXFlux2LatentCodec(patchedChannels: 16, epsilon: 1e-4, patch: 2))

        let embeds = MLXArray.zeros([1, 5, geometry.jointAttentionDim])
        let plain = pipeline.generate(promptEmbeds: embeds, latentHeight: 2, latentWidth: 3,
                                      steps: 2, seed: 5)
        // A 4×6 latent is the codec's 2×2 patch over the pipeline's 2×3 grid.
        let reference = MLXRandom.normal([1, 4, 6, 4])
        let conditioned = pipeline.generate(promptEmbeds: embeds, latentHeight: 2, latentWidth: 3,
                                            steps: 2, seed: 5, references: [reference])
        eval(plain, conditioned)
        XCTAssertEqual(conditioned.shape, plain.shape,
                       "only the generated tokens are decoded, whatever is appended")
        XCTAssertNotEqual(conditioned.reshaped([-1]).asArray(Float.self),
                          plain.reshaped([-1]).asArray(Float.self),
                          "a reference image changes the result")

        // The time coordinate is what separates them: image ids sit at t = 0, reference N at
        // t = 10·(N+1).
        let generatedIds = NFKMLXFlux2TransformerNet.imageIds(height: 2, width: 3)
        let firstReference = NFKMLXFlux2TransformerNet.referenceImageIds(height: 2, width: 3, index: 0)
        let secondReference = NFKMLXFlux2TransformerNet.referenceImageIds(height: 2, width: 3, index: 1)
        XCTAssertEqual(generatedIds[0, 0].item(Float.self), 0)
        XCTAssertEqual(firstReference[0, 0].item(Float.self), 10)
        XCTAssertEqual(secondReference[0, 0].item(Float.self), 20)
        // The row and column axes are unchanged, so only time distinguishes them.
        XCTAssertEqual(firstReference[0..., 1...].asArray(Float.self),
                       generatedIds[0..., 1...].asArray(Float.self))
    }

    // Classifier-free guidance runs both prompts and mixes them, which the klein releases do not use
    // but the pipeline supports for a caller who wants it.
    func testFlux2GuidanceMixesTheTwoPrompts() throws {
        try requireMLXRuntime()
        var vae = NFKMLXSDVAEConfiguration.flux2
        vae.latentChannels = 4
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4
        var geometry = NFKMLXFlux2Configuration.tiny
        geometry.inChannels = 16
        let pipeline = NFKMLXFlux2Pipeline(
            transformer: NFKMLXFlux2TransformerNet(geometry),
            autoencoder: NFKMLXSDAutoencoder(configuration: vae),
            codec: NFKMLXFlux2LatentCodec(patchedChannels: 16, epsilon: 1e-4, patch: 2))

        let positive = MLXArray.zeros([1, 5, geometry.jointAttentionDim])
        let negative = MLXArray.ones([1, 5, geometry.jointAttentionDim])
        let plain = pipeline.generate(promptEmbeds: positive, latentHeight: 2, latentWidth: 3,
                                      steps: 2, seed: 3)
        let guided = pipeline.generate(promptEmbeds: positive, negativeEmbeds: negative,
                                       latentHeight: 2, latentWidth: 3, steps: 2,
                                       guidanceScale: 4, seed: 3)
        eval(plain, guided)
        XCTAssertNotEqual(guided.reshaped([-1]).asArray(Float.self),
                          plain.reshaped([-1]).asArray(Float.self),
                          "a negative prompt changes the result")
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
