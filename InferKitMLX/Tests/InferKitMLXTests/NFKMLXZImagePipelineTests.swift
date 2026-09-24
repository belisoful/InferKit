//
//  NFKMLXZImagePipelineTests.swift
//  InferKitMLXTests
//
//  The Z-Image text-to-image pipeline glue (S3-DiT + flow loop → Flux VAE), on matching tiny
//  configurations so the chaining — the guided sampling loop, the timestep/latent conventions, the
//  centered-latent decode — is exercised with random weights. Runs where MLX has a Metal library
//  (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXZImagePipelineTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private let vaeConfiguration: NFKMLXSDVAEConfiguration = {
        var configuration = NFKMLXSDVAEConfiguration()
        configuration.latentChannels = 4
        configuration.blockChannels = [8, 16]
        configuration.layersPerBlock = 1
        configuration.normalizationGroups = 4
        configuration.useQuantConv = false
        configuration.scaleFactor = 0.3611
        configuration.shiftFactor = 0.1159
        return configuration
    }()

    private func textConfiguration(vocabulary: Int) -> NFKMLXLanguageConfiguration {
        NFKMLXLanguageConfiguration(hiddenSize: 24, layerCount: 3, headCount: 2, keyValueHeadCount: 1,
                                    headDimensions: 12, intermediateSize: 48, vocabularySize: vocabulary)
    }

    private func tokenizer() throws -> NFKTokenizer {
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_QWEN3_4B"],
              let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: URL(fileURLWithPath: path)) else {
            throw XCTSkip("set IK_VAL_QWEN3_4B (a Qwen3 release directory, for its tokenizer)")
        }
        return tokenizer
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
        let y = b.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
        let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
        return dot / (sqrt(x.reduce(0) { $0 + $1 * $1 }) * sqrt(y.reduce(0) { $0 + $1 * $1 }))
    }

    private func pipeline() -> NFKMLXZImagePipeline {
        let transformer = NFKMLXZImageTransformerNet(.tiny)               // inChannels 4, captionFeatureDim 24
        var vaeConfig = NFKMLXSDVAEConfiguration()
        vaeConfig.latentChannels = 4
        vaeConfig.blockChannels = [8, 16]
        vaeConfig.layersPerBlock = 1
        vaeConfig.normalizationGroups = 4
        vaeConfig.useQuantConv = false
        vaeConfig.scaleFactor = 0.3611
        vaeConfig.shiftFactor = 0.1159
        let vae = NFKMLXSDAutoencoder(configuration: vaeConfig)
        return NFKMLXZImagePipeline(transformer: transformer, vae: vae)
    }

    func testThePipelineGeneratesAnImageOfTheExpectedShape() throws {
        try requireMLXRuntime()
        let prompt = MLXRandom.normal([6, 24])
        let image = pipeline().generate(promptEmbeds: prompt, negativeEmbeds: nil,
                                        latentHeight: 4, latentWidth: 4, steps: 3, guidance: 1)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.ndim, 4, "a [B, H, W, 3] image")
        XCTAssertEqual(image.shape[3], 3, "RGB output")
        XCTAssertEqual(image.shape[1], 8, "the two VAE levels upsample the 4×4 latent to 8×8")
    }

    func testClassifierFreeGuidanceRunsBothPrompts() throws {
        try requireMLXRuntime()
        let prompt = MLXRandom.normal([6, 24])
        let negative = MLXRandom.normal([6, 24])
        let image = pipeline().generate(promptEmbeds: prompt, negativeEmbeds: negative,
                                        latentHeight: 4, latentWidth: 4, steps: 2, guidance: 4)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.shape[3], 3)
    }

    func testImageToImageGeneratesFromASource() throws {
        try requireMLXRuntime()
        let source = MLXRandom.normal([1, 8, 8, 3])                        // encodes to a 4×4 latent
        let prompt = MLXRandom.normal([6, 24])
        let image = pipeline().generate(image: source, promptEmbeds: prompt, negativeEmbeds: nil,
                                        strength: 0.6, steps: 3, guidance: 1)
        eval(image)
        XCTAssertEqual(image.shape[0], 1)
        XCTAssertEqual(image.ndim, 4, "a [B, H, W, 3] image")
        XCTAssertEqual(image.shape[3], 3, "RGB output")
    }

    // The glue against diffusers' ZImagePipeline at a tiny random geometry with the real Qwen3
    // tokenizer: the templated prompt's ids, the penultimate-state features cut to the prompt, the
    // static-shift schedule ending at sigma 0, the guided and unguided loops, and the decode.
    // `run_reference.py z_image_pipeline` records the reference.
    func testThePipelineMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_Z_IMAGE_PIPELINE"] else {
            throw XCTSkip("set IK_PARITY_Z_IMAGE_PIPELINE (run_reference.py z_image_pipeline)")
        }
        let tokenizer = try tokenizer()
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        func weights(_ prefix: String) -> [(String, MLXArray)] {
            arrays.compactMap { key, value in key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil }
        }
        let embedding = try XCTUnwrap(arrays["te::model.embed_tokens.weight"])
        let decoder = NFKMLXLanguage.makeNet(textConfiguration(vocabulary: embedding.dim(0)))
        try NFKMLXWeights.apply(weights("te::"), to: decoder, verifyShapes: true)
        let transformer = NFKMLXZImageTransformerNet(.tiny)
        try NFKMLXWeights.apply(weights("t::"), to: transformer, verifyShapes: true)
        let vae = NFKMLXSDAutoencoder(configuration: vaeConfiguration)
        try NFKMLXWeights.apply(weights("v::").map { key, value in
            (NFKMLXStableDiffusionModels.remapVAEKey(key), value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }, to: vae, verifyShapes: true)

        let ids = NFKMLXZImageGenerator.promptIds("A red fox walking through fresh snow, cinematic", tokenizer: tokenizer)
        XCTAssertEqual(ids, try XCTUnwrap(arrays["input_ids"]).asArray(Int32.self).map(Int.init),
                       "the prompt tokenizes through the chat template as the reference's")
        let text = NFKZImageTextStage(decoder)
        let features = text.features(ids)
        let negative = text.features(NFKMLXZImageGenerator.promptIds("blurry, low quality", tokenizer: tokenizer))
        let textCosine = cosine(features, try XCTUnwrap(arrays["prompt_embeds"]))
        XCTAssertEqual(features.shape, try XCTUnwrap(arrays["prompt_embeds"]).shape)
        XCTAssertGreaterThan(textCosine, 0.999999, "the penultimate state at the prompt's tokens is the reference's")
        XCTAssertGreaterThan(cosine(negative, try XCTUnwrap(arrays["negative_embeds"])), 0.999999)

        let pipeline = NFKMLXZImagePipeline(transformer: transformer, vae: vae, schedule: .zImageTurbo)
        let start = try XCTUnwrap(arrays["latents"]).expandedDimensions(axis: 1)
        let guided = pipeline.denoise(start, promptEmbeds: features, negativeEmbeds: negative, steps: 5, guidance: 4)
        let unguided = pipeline.denoise(start, promptEmbeds: features, negativeEmbeds: negative, steps: 5, guidance: 0)
        let guidedCosine = cosine(guided, try XCTUnwrap(arrays["final_latents"]))
        let unguidedCosine = cosine(unguided, try XCTUnwrap(arrays["final_latents_unguided"]))
        let control = cosine(unguided, try XCTUnwrap(arrays["final_latents"]))
        let image = clip(pipeline.decode(guided)[0] / 2 + 0.5, min: 0, max: 1)
        let reference = try XCTUnwrap(arrays["image"])
        XCTAssertEqual(image.shape, reference.shape)
        let imageCosine = cosine(image, reference)
        print("VALIDATION PARITY z-image-pipeline: prompt features \(textCosine), final latents \(guidedCosine) "
              + "(unguided \(unguidedCosine), control \(control)), image \(imageCosine)")
        XCTAssertGreaterThan(guidedCosine, 0.99999, "the guided loop matches the reference")
        XCTAssertGreaterThan(unguidedCosine, 0.99999, "and the unguided one")
        XCTAssertLessThan(control, 0.9999, "the record separates guided from unguided")
        XCTAssertGreaterThan(imageCosine, 0.99999, "the centered-latent decode matches the reference")
    }

    // A staged generator and a resident one over the same weights produce the same image, and the
    // staged one loads the text encoder once per image, both prompts together.
    func testStagingReleasesEachStageAndChangesNothing() throws {
        try requireMLXRuntime()
        let tokenizer = try tokenizer()
        MLXRandom.seed(45)
        func captured(_ module: Module) -> [(String, MLXArray)] {
            let parameters = module.parameters().flattened()
            eval(parameters.map(\.1))
            return parameters
        }
        let textConfiguration = textConfiguration(vocabulary: 151_936)
        let decoderWeights = captured(NFKMLXLanguage.makeNet(textConfiguration))
        let transformerWeights = captured(NFKMLXZImageTransformerNet(.tiny))
        let vaeWeights = captured(NFKMLXSDAutoencoder(configuration: vaeConfiguration))
        var loads = (text: 0, pipeline: 0)
        func generator(resident: Bool) throws -> NFKMLXZImageGenerator {
            let generator = NFKMLXZImageGenerator(
                resident: resident, tokenizer: tokenizer,
                loadTextEncoder: {
                    loads.text += 1
                    let decoder = NFKMLXLanguage.makeNet(textConfiguration)
                    try NFKMLXWeights.apply(decoderWeights, to: decoder)
                    return decoder
                },
                loadPipeline: {
                    loads.pipeline += 1
                    let transformer = NFKMLXZImageTransformerNet(.tiny)
                    try NFKMLXWeights.apply(transformerWeights, to: transformer)
                    let vae = NFKMLXSDAutoencoder(configuration: self.vaeConfiguration)
                    try NFKMLXWeights.apply(vaeWeights, to: vae)
                    return NFKMLXZImagePipeline(transformer: transformer, vae: vae, schedule: .zImageTurbo)
                })
            if resident {
                try generator.loadResident()
            }
            generator.steps = 3
            generator.guidance = 4
            return generator
        }
        let resident = try generator(resident: true)
        let residentImages = try (0 ..< 2).map { _ in try resident.image(forPrompt: "a red fox", width: 16, height: 16, seed: 3) }
        XCTAssertEqual((loads.text, loads.pipeline) == (1, 1), true)
        loads = (0, 0)
        let staged = try generator(resident: false)
        let stagedImages = try (0 ..< 2).map { _ in try staged.image(forPrompt: "a red fox", width: 16, height: 16, seed: 3) }
        XCTAssertEqual(loads.text, 2, "a staged generator loads the text encoder once per image")
        XCTAssertEqual(loads.pipeline, 2)
        XCTAssertFalse(staged.isHoldingTextEncoder || staged.isHoldingPipeline, "and holds neither after")
        XCTAssertEqual(stagedImages[0].shape, [16, 16, 3])
        for (a, b) in zip(residentImages, stagedImages) {
            XCTAssertEqual(b.reshaped([-1]).asArray(Float.self), a.reshaped([-1]).asArray(Float.self),
                           "staging does not change the image")
        }
        XCTAssertThrowsError(try staged.image(forPrompt: "a red fox", width: 20, height: 16),
                             "a side that is not a multiple of 16 is refused, as the reference refuses it")
    }

    // The released scheduler configs: Turbo's static shift of 3 and the base release's 6, both over
    // the ramp to sigma 0, so the last of nine Turbo steps moves nothing.
    func testTheReleaseScheduleEndsAtSigmaZero() throws {
        var turbo = NFKMLXFlowMatchScheduler(.zImageTurbo)
        turbo.setTimesteps(5, sequenceLength: 4096)
        let expected: [Float] = [1, 0.9, 0.75, 0.5, 0, 0]
        for (mine, reference) in zip(turbo.sigmas, expected) {
            XCTAssertEqual(mine, reference, accuracy: 1e-6, "diffusers' sigmas at shift 3")
        }
        var base = NFKMLXFlowMatchScheduler(.zImage)
        base.setTimesteps(5, sequenceLength: 4096)
        XCTAssertEqual(base.sigmas[1], 0.94736844, accuracy: 1e-6, "diffusers' second sigma at shift 6")
        XCTAssertEqual(base.sigmas[4], 0)
    }

    // A float32 transformer release read back through the config reader and the converting load: the
    // geometry round-trips, every parameter arrives in bfloat16 at the stored value, and the stage is
    // weighed at half its stored bytes.
    func testAFloat32TransformerReleaseLoadsAtBfloat16() throws {
        try requireMLXRuntime()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("zimage-release-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        MLXRandom.seed(46)
        let stored = NFKMLXZImageTransformerNet(.tiny).parameters().flattened()
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: stored),
                     url: directory.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        let config: [String: Any] = [
            "all_patch_size": [2], "all_f_patch_size": [1], "in_channels": 4, "dim": 32, "n_layers": 2,
            "n_refiner_layers": 1, "n_heads": 2, "n_kv_heads": 2, "norm_eps": 1e-5, "qk_norm": true,
            "cap_feat_dim": 24, "rope_theta": 256.0, "t_scale": 1000.0, "axes_dims": [4, 6, 6]]
        let configURL = directory.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: config).write(to: configURL)

        let configuration = try NFKMLXZImageRelease.transformerConfiguration(fromHuggingFace: configURL)
        XCTAssertEqual(configuration.dim, 32)
        XCTAssertEqual(configuration.captionFeatureDim, 24)
        XCTAssertEqual(configuration.axesDims, [4, 6, 6])
        let net = NFKMLXZImageTransformerNet(configuration)
        try NFKMLXZImageRelease.loadTransformer(into: net, fromDirectory: directory, dtype: .bfloat16)
        let loaded = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        for (key, value) in stored {
            let mine = try XCTUnwrap(loaded[key], key)
            XCTAssertEqual(mine.dtype, .bfloat16, key)
            XCTAssertEqual(mine.asType(.float32).reshaped([-1]).asArray(Float.self),
                           value.asType(.bfloat16).asType(.float32).reshaped([-1]).asArray(Float.self), key)
        }
        let storedBytes = stored.reduce(0) { $0 + $1.1.nbytes }
        XCTAssertEqual(try NFKMLXStageWeights.bytes(inDirectory: directory, holding: .bfloat16), storedBytes / 2)

        var grouped = config
        grouped["n_kv_heads"] = 1
        try JSONSerialization.data(withJSONObject: grouped).write(to: configURL)
        XCTAssertThrowsError(try NFKMLXZImageRelease.transformerConfiguration(fromHuggingFace: configURL),
                             "grouped key-value heads are refused")
    }
}
