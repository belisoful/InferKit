//
//  NFKMLXSDLoRATests.swift
//  InferKitMLXTests
//
//  Released Stable Diffusion LoRA adapters merged into a tiny UNet, the latent consistency schedule, and
//  the guidance projection an LCM distillation conditions on. Weight-free: the released adapters are
//  measured in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXSDLoRATests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private var tinyUNet: NFKMLXSDUNet {
        let unet = NFKMLXSDUNet(configuration: NFKMLXSDTextToImageConfiguration.tiny.unet)
        unet.train(false)
        return unet
    }

    /// The layer's weight as a new array. `Module.update(parameters:)` writes into the existing array
    /// object, so a snapshot taken by reference would follow the merge.
    private func weight(_ module: Module, _ path: String) -> MLXArray {
        let layer = module.leafModules().flattened().first { $0.0 == path }!.1
        let value = layer.parameters().flattened().first { $0.0 == "weight" }!.1 + 0
        eval(value)
        return value
    }

    private func temporaryAdapter(_ arrays: [String: MLXArray]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXSDLoRATests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("pytorch_lora_weights.safetensors")
        try save(arrays: arrays, url: url)
        return url
    }

    // MARK: Merging

    // The kohya layout: a linear target, a 3×3 convolution target, and a 1×1 convolution target (the
    // transformer's projection), each with its alpha. The expected deltas are formed by hand.
    func testKohyaAdapterMergesIntoLinearAndConvolutionTargets() throws {
        try requireMLXRuntime()
        let unet = tinyUNet
        let rank = 2
        let linearPath = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
        let convPath = "down_blocks.0.resnets.0.conv1"
        let projectionPath = "down_blocks.0.attentions.0.proj_in_conv"
        let beforeLinear = weight(unet, linearPath)
        let beforeConv = weight(unet, convPath)
        let beforeProjection = weight(unet, projectionPath)
        let (linearOut, linearIn) = (beforeLinear.dim(0), beforeLinear.dim(1))
        let (convOut, convIn) = (beforeConv.dim(0), beforeConv.dim(3))
        let projectionOut = beforeProjection.dim(0), projectionIn = beforeProjection.dim(3)

        let linearDown = MLXRandom.normal([rank, linearIn], key: MLXRandom.key(1))
        let linearUp = MLXRandom.normal([linearOut, rank], key: MLXRandom.key(2))
        let convDown = MLXRandom.normal([rank, convIn, 3, 3], key: MLXRandom.key(3))
        let convUp = MLXRandom.normal([convOut, rank, 1, 1], key: MLXRandom.key(4))
        let projectionDown = MLXRandom.normal([rank, projectionIn, 1, 1], key: MLXRandom.key(5))
        let projectionUp = MLXRandom.normal([projectionOut, rank, 1, 1], key: MLXRandom.key(6))
        let alpha: Float = 1                                       // alpha / rank = 0.5
        let arrays: [String: MLXArray] = [
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight": linearDown.asType(.float16),
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight": linearUp.asType(.float16),
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha": MLXArray(alpha),
            "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight": convDown.asType(.float16),
            "lora_unet_down_blocks_0_resnets_0_conv1.lora_up.weight": convUp.asType(.float16),
            "lora_unet_down_blocks_0_resnets_0_conv1.alpha": MLXArray(alpha),
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight": projectionDown.asType(.float16),
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight": projectionUp.asType(.float16),
            "lora_unet_down_blocks_0_attentions_0_proj_in.alpha": MLXArray(alpha),
        ]
        let scale: Float = 0.8
        let report = try NFKMLXSDLoRA.merge(adapterURL: temporaryAdapter(arrays), into: unet, scale: scale)
        XCTAssertEqual(report.mergedLayers, 3)
        XCTAssertEqual(report.skippedTextEncoderKeys, 0)

        let strength = scale * alpha / Float(rank)
        let expectedLinear = beforeLinear
            + matmul(linearUp.asType(.float16).asType(.float32), linearDown.asType(.float16).asType(.float32)) * strength
        let convDelta = matmul(convUp.asType(.float16).asType(.float32).reshaped([convOut, rank]),
                               convDown.asType(.float16).asType(.float32).reshaped([rank, -1]))
            .reshaped([convOut, convIn, 3, 3]).transposed(0, 2, 3, 1)
        let expectedConv = beforeConv + convDelta * strength
        let projectionDelta = matmul(projectionUp.asType(.float16).asType(.float32).reshaped([projectionOut, rank]),
                                     projectionDown.asType(.float16).asType(.float32).reshaped([rank, -1]))
            .reshaped([projectionOut, projectionIn, 1, 1]).transposed(0, 2, 3, 1)
        let expectedProjection = beforeProjection + projectionDelta * strength
        eval(expectedLinear, expectedConv, expectedProjection)

        XCTAssertTrue(allClose(weight(unet, linearPath), expectedLinear, atol: 1e-6).item(Bool.self))
        XCTAssertTrue(allClose(weight(unet, convPath), expectedConv, atol: 1e-6).item(Bool.self))
        XCTAssertTrue(allClose(weight(unet, projectionPath), expectedProjection, atol: 1e-6).item(Bool.self))
    }

    // The PEFT layout diffusers writes: the module path with dots, `lora_A` / `lora_B`, and no alpha,
    // whose scaling is already in the weights. The reference's `to_out.0` and `ff.net.2` spellings map
    // onto this module's `to_out` and `ff.out`.
    func testPEFTAdapterMergesWithoutAnAlpha() throws {
        try requireMLXRuntime()
        let unet = tinyUNet
        let outPath = "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out"
        let ffPath = "down_blocks.0.attentions.0.transformer_blocks.0.ff.out"
        let beforeOut = weight(unet, outPath)
        let beforeFF = weight(unet, ffPath)
        let rank = 3
        let outDown = MLXRandom.normal([rank, beforeOut.dim(1)], key: MLXRandom.key(7))
        let outUp = MLXRandom.normal([beforeOut.dim(0), rank], key: MLXRandom.key(8))
        let ffDown = MLXRandom.normal([rank, beforeFF.dim(1)], key: MLXRandom.key(9))
        let ffUp = MLXRandom.normal([beforeFF.dim(0), rank], key: MLXRandom.key(10))
        let arrays: [String: MLXArray] = [
            "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_A.weight": outDown,
            "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_B.weight": outUp,
            "unet.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.lora_A.weight": ffDown,
            "unet.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.lora_B.weight": ffUp,
            "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight": MLXArray.zeros([rank, 8]),
            "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight": MLXArray.zeros([8, rank]),
        ]
        let url = try temporaryAdapter(arrays)
        XCTAssertThrowsError(try NFKMLXSDLoRA.merge(adapterURL: url, into: unet),
                             "an adapter that also adapts the text towers is refused without the opt-in")
        let report = try NFKMLXSDLoRA.merge(adapterURL: url, into: unet, ignoringTextEncoderKeys: true)
        XCTAssertEqual(report.mergedLayers, 2)
        XCTAssertEqual(report.skippedTextEncoderKeys, 2, "text-tower keys are left alone and counted")

        let expectedOut = beforeOut + matmul(outUp, outDown)
        let expectedFF = beforeFF + matmul(ffUp, ffDown)
        eval(expectedOut, expectedFF)
        XCTAssertTrue(allClose(weight(unet, outPath), expectedOut, atol: 1e-6).item(Bool.self))
        XCTAssertTrue(allClose(weight(unet, ffPath), expectedFF, atol: 1e-6).item(Bool.self))
    }

    func testAnAdapterForAnotherGeometryIsRefused() throws {
        try requireMLXRuntime()
        let unet = tinyUNet
        let arrays: [String: MLXArray] = [
            "lora_unet_down_blocks_3_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight": MLXArray.zeros([2, 32]),
            "lora_unet_down_blocks_3_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight": MLXArray.zeros([32, 2]),
            "lora_unet_down_blocks_3_attentions_0_transformer_blocks_0_attn1_to_q.alpha": MLXArray(Float(2)),
        ]
        XCTAssertThrowsError(try NFKMLXSDLoRA.merge(adapterURL: temporaryAdapter(arrays), into: unet)) { error in
            XCTAssertTrue("\(error)".contains("down_blocks_3"), "the refusal names the target: \(error)")
        }
        let mismatched: [String: MLXArray] = [
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight": MLXArray.zeros([2, 16]),
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight": MLXArray.zeros([32, 2]),
        ]
        XCTAssertThrowsError(try NFKMLXSDLoRA.merge(adapterURL: temporaryAdapter(mismatched), into: unet),
                             "a delta of the wrong shape is refused")
    }

    func testDiffusersPathInvertsTheCheckpointRemap() {
        for path in ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.weight",
                     "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
                     "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.weight",
                     "mid_block.attentions.0.proj_in.weight",
                     "up_blocks.1.resnets.0.time_emb_proj.weight"] {
            let module = NFKMLXStableDiffusionModels.remapUNetKey(path)
            XCTAssertEqual(NFKMLXSDLoRA.diffusersPath(module), path)
        }
    }

    // MARK: The latent consistency schedule

    // diffusers' LCMScheduler over 50 distillation steps: 4 steps visit 999, 759, 499, 259; 8 steps 999,
    // 879, 759, 639, 499, 379, 259, 139; 2 steps 999, 499; 1 step 999. Each step's "previous" signal
    // ratio is the next step's, and the final step's is 1, which makes the step return the clean
    // estimate as the reference's last step does.
    func testLCMScheduleMatchesTheReference() throws {
        try requireMLXRuntime()
        let scheduler = NFKLCMScheduler()
        XCTAssertEqual(scheduler.steps(4).map(\.train), [999, 759, 499, 259])
        XCTAssertEqual(scheduler.steps(8).map(\.train), [999, 879, 759, 639, 499, 379, 259, 139])
        XCTAssertEqual(scheduler.steps(2).map(\.train), [999, 499])
        XCTAssertEqual(scheduler.steps(1).map(\.train), [999])
        XCTAssertEqual(scheduler.steps(50).map(\.train).prefix(3).map { $0 }, [999, 979, 959])
        XCTAssertEqual(scheduler.steps(50).last?.train, 19)

        let four = scheduler.steps(4)
        let ddim = NFKDDIMScheduler()
        XCTAssertEqual(four[0].alphaBarPrev, four[1].alphaBar)
        XCTAssertEqual(four[2].alphaBarPrev, four[3].alphaBar)
        XCTAssertEqual(four[3].alphaBarPrev, 1, "the final step returns the clean estimate")
        XCTAssertEqual(four[1].alphaBar, ddim.alphaBar(at: 759))
    }

    // The backend seeds the sampler from the request, so two seeds draw two different re-noisings and
    // one seed draws the same twice.
    func testLCMSamplerTakesTheRequestSeed() throws {
        try requireMLXRuntime()
        let scheduler = NFKLCMScheduler()
        let step = scheduler.steps(4)[1]
        let latent = MLXRandom.normal([4, 4, 4], key: MLXRandom.key(31))
        let prediction = MLXRandom.normal([4, 4, 4], key: MLXRandom.key(32))
        let a = scheduler.seeded(7).step(prediction: prediction, timestep: step, latent: latent)
        let b = scheduler.seeded(7).step(prediction: prediction, timestep: step, latent: latent)
        let c = scheduler.seeded(8).step(prediction: prediction, timestep: step, latent: latent)
        eval(a, b, c)
        XCTAssertTrue(arrayEqual(a, b).item(Bool.self))
        XCTAssertFalse(arrayEqual(a, c).item(Bool.self))
        XCTAssertTrue(NFKDDIMScheduler().seeded(3) is NFKDDIMScheduler, "a deterministic sampler returns itself")
    }

    // The reference's draw replaces the seeded stream when a caller supplies one.
    func testLCMStepTakesTheCallersNoise() throws {
        try requireMLXRuntime()
        var scheduler = NFKLCMScheduler()
        let step = scheduler.steps(4)[1]
        let latent = MLXRandom.normal([1, 4, 4, 4], key: MLXRandom.key(11))
        let prediction = MLXRandom.normal([1, 4, 4, 4], key: MLXRandom.key(12))
        let draw = MLXRandom.normal([4, 4, 4], key: MLXRandom.key(13))
        let box = NFKMLXArrayBox(draw)
        scheduler.noise = { _ in box.array }
        let next = scheduler.step(prediction: prediction, timestep: step, latent: latent)

        let sqrtBar = sqrtf(step.alphaBar)
        let clean = (latent - prediction * sqrtf(1 - step.alphaBar)) / sqrtBar
        let scaled = Float(step.train) * 10
        let denominator = scaled * scaled + 0.25
        let denoised = clean * (scaled / sqrtf(denominator)) + latent * (0.25 / denominator)
        let expected = denoised * sqrtf(step.alphaBarPrev) + draw.reshaped([1, 4, 4, 4]) * sqrtf(1 - step.alphaBarPrev)
        eval(next, expected)
        XCTAssertTrue(allClose(next, expected, atol: 1e-6).item(Bool.self))
    }

    // MARK: The trajectory consistency schedule

    // diffusers' TCDScheduler shares the latent consistency schedule, but its "previous" timestep after
    // the last step is 0, not -1: the final step's signal ratio is training step 0's, just under 1.
    func testTCDScheduleMatchesTheReference() throws {
        try requireMLXRuntime()
        let scheduler = NFKTCDScheduler()
        let ddim = NFKDDIMScheduler()
        XCTAssertEqual(scheduler.steps(4).map(\.train), [999, 759, 499, 259])
        XCTAssertEqual(scheduler.steps(1).map(\.train), [999])
        let four = scheduler.steps(4)
        XCTAssertEqual(four[0].alphaBarPrev, four[1].alphaBar)
        XCTAssertEqual(four[3].alphaBarPrev, ddim.alphaBar(at: 0))
        XCTAssertEqual(scheduler.steps(1)[0].alphaBarPrev, ddim.alphaBar(at: 0))
        XCTAssertEqual(scheduler.transitionTrain(for: four[0]), 0, "eta 1 re-noises from the clean estimate")
        XCTAssertEqual(NFKTCDScheduler(eta: 0.3).transitionTrain(for: four[0]), Int((0.7 * 759.0).rounded(.down)))
        XCTAssertEqual(NFKTCDScheduler(eta: 0).transitionTrain(for: four[0]), 759)
        XCTAssertEqual(ddim.train(withAlphaBar: ddim.alphaBar(at: 759)), 759)
    }

    // One step at eta 1 on the caller's draw: the estimate re-noised to training step 0, then carried to
    // the next step's level with the draw. The final step returns the estimate alone, and eta 0 never
    // draws.
    func testTCDStepMatchesTheReference() throws {
        try requireMLXRuntime()
        let ddim = NFKDDIMScheduler()
        var scheduler = NFKTCDScheduler()
        let steps = scheduler.steps(2)
        let latent = MLXRandom.normal([1, 4, 4, 4], key: MLXRandom.key(21))
        let prediction = MLXRandom.normal([1, 4, 4, 4], key: MLXRandom.key(22))
        let draw = MLXRandom.normal([4, 4, 4], key: MLXRandom.key(23))
        let box = NFKMLXArrayBox(draw)
        scheduler.noise = { _ in box.array }

        let step = steps[0]
        let clean = (latent - prediction * sqrtf(1 - step.alphaBar)) / sqrtf(step.alphaBar)
        let alphaBarS = ddim.alphaBar(at: 0)
        let estimate = clean * sqrtf(alphaBarS) + prediction * sqrtf(1 - alphaBarS)
        let ratio = step.alphaBarPrev / alphaBarS
        let expected = estimate * sqrtf(ratio) + draw.reshaped([1, 4, 4, 4]) * sqrtf(1 - ratio)
        let next = scheduler.step(prediction: prediction, timestep: step, latent: latent)
        eval(next, expected)
        XCTAssertTrue(allClose(next, expected, atol: 1e-6).item(Bool.self))

        let final = steps[1]
        let finalClean = (latent - prediction * sqrtf(1 - final.alphaBar)) / sqrtf(final.alphaBar)
        let finalExpected = finalClean * sqrtf(alphaBarS) + prediction * sqrtf(1 - alphaBarS)
        let last = scheduler.step(prediction: prediction, timestep: final, latent: latent)
        eval(last, finalExpected)
        XCTAssertTrue(allClose(last, finalExpected, atol: 1e-6).item(Bool.self))

        let deterministic = NFKTCDScheduler(eta: 0)
        let a = deterministic.seeded(1).step(prediction: prediction, timestep: step, latent: latent)
        let b = deterministic.seeded(2).step(prediction: prediction, timestep: step, latent: latent)
        eval(a, b)
        XCTAssertTrue(arrayEqual(a, b).item(Bool.self), "eta 0 draws no noise")
        let seededA = NFKTCDScheduler().seeded(1).step(prediction: prediction, timestep: step, latent: latent)
        let seededB = NFKTCDScheduler().seeded(2).step(prediction: prediction, timestep: step, latent: latent)
        eval(seededA, seededB)
        XCTAssertFalse(arrayEqual(seededA, seededB).item(Bool.self), "the request's seed changes the draw")
    }

    // MARK: The guidance projection

    // diffusers' `get_guidance_scale_embedding` at guidance 8 (w = 7) over 256 channels: sines first,
    // frequencies spaced over `half - 1`.
    func testGuidanceEmbeddingMatchesTheReference() throws {
        try requireMLXRuntime()
        let embedding = NFKSDTimesteps.guidanceEmbedding(8, channels: 256)
        XCTAssertEqual(embedding.shape, [1, 256])
        let values = embedding.asArray(Float.self)
        for (index, expected) in [(0, Float(0.5068854)), (1, 0.8042871), (2, -0.8604797),
                                  (125, 0.7237797), (126, 0.6835765), (127, 0.6442176),
                                  (128, 0.8620135), (129, 0.5942408), (130, -0.5094848),
                                  (253, 0.6900312), (254, 0.7298789), (255, 0.7648422)] {
            XCTAssertEqual(values[index], expected, accuracy: 2e-4, "channel \(index)")
        }
    }

    func testALatentConsistencyUNetConditionsOnTheGuidanceEmbedding() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXSDTextToImageConfiguration.tiny.unet
        configuration.timeConditionProjectionDimensions = 8
        let unet = NFKMLXSDUNet(configuration: configuration)
        unet.train(false)
        XCTAssertNotNil(unet.timeEmbedding.conditionProjection)
        XCTAssertEqual(unet.timeEmbedding.conditionProjection?.weight.shape, [32, 8])
        XCTAssertEqual(NFKMLXSDUNet.guidanceEmbedding(scale: 8, channels: 8).shape, [1, 8])
        XCTAssertTrue(unet.parameters().flattened().contains { $0.0 == "time_embedding.cond_proj.weight" },
                      "the projection loads from the reference's key")

        let latent = MLXRandom.normal([1, 8, 8, 4], key: MLXRandom.key(21))
        let context = MLXRandom.normal([1, 5, 32], key: MLXRandom.key(22))
        let timestep = MLXArray([Int32(499)])
        let plain = unet(latent, timestep: timestep, context: context)
        let guided = unet(latent, timestep: timestep, context: context,
                          guidanceEmbedding: NFKSDTimesteps.guidanceEmbedding(8, channels: 8))
        eval(plain, guided)
        XCTAssertEqual(guided.shape, latent.shape)
        XCTAssertFalse(arrayEqual(plain, guided).item(Bool.self), "the guidance scale reaches the time embedding")
    }

    // MARK: Configurations

    func testWithSamplerSetsTheAdapterDefaults() {
        let lcm = NFKMLXSDTextToImageConfiguration.stableDiffusion15.withSampler(.lcm)
        XCTAssertEqual(lcm.sampler, .lcm)
        XCTAssertEqual(lcm.steps, 4)
        XCTAssertEqual(lcm.guidanceScale, 1)
        XCTAssertTrue(lcm.makeScheduler() is NFKLCMScheduler)

        let lightning = NFKMLXSDTextToImageConfiguration.sdxlLightning
        XCTAssertEqual(lightning.sampler, .ddimTrailing)
        XCTAssertEqual(lightning.timestepSpacing, .trailing)
        XCTAssertEqual(lightning.steps, 4)
        XCTAssertEqual(lightning.guidanceScale, 0)
        XCTAssertEqual(lightning.unet.transformerLayers, [1, 2, 10])
        XCTAssertNotNil(lightning.secondaryTextEncoder)
        XCTAssertTrue(lightning.makeScheduler() is NFKDDIMScheduler)

        let dreamshaper = NFKMLXSDTextToImageConfiguration.latentConsistency15
        XCTAssertEqual(dreamshaper.unet.timeConditionProjectionDimensions, 256)
        XCTAssertEqual(dreamshaper.sampler, .lcm)
        XCTAssertEqual(dreamshaper.guidanceScale, 8)
        XCTAssertEqual(dreamshaper.steps, 4)
        XCTAssertEqual(dreamshaper.withSampler(.lcm).guidanceScale, 8,
                       "asking for the sampler a configuration already runs keeps its own guidance")
        XCTAssertEqual(NFKMLXStableDiffusionRelease(.lcmDreamshaperV7).configuration.withSampler(.lcm).guidanceScale, 8)

        let unified = NFKMLXSDTextToImageConfiguration.sdxl.withSampler(.tcd)
        XCTAssertEqual(unified.sampler, .tcd)
        XCTAssertEqual(unified.steps, 1)
        XCTAssertEqual(unified.guidanceScale, 0)
        XCTAssertEqual(unified.trajectoryConsistencyEta, 1)
        XCTAssertTrue(unified.makeScheduler() is NFKTCDScheduler)
        XCTAssertEqual((unified.makeScheduler() as? NFKTCDScheduler)?.eta, 1)

        let overridden = NFKMLXSDTextToImageConfiguration.sdxl.withSampler(.ddimTrailing, steps: 8, guidanceScale: 0)
        XCTAssertEqual(overridden.steps, 8)
        XCTAssertEqual(NFKMLXSDTextToImageConfiguration.sdxl.steps, 50, "the base release keeps its own schedule")
        XCTAssertEqual(NFKMLXSDTextToImageConfiguration.sdxl.timestepSpacing, .leading)
    }

    func testBundledReleasesNameTheNewRepositories() {
        XCTAssertEqual(NFKMLXStableDiffusionRelease(.sdxlBase).repository, "stabilityai/stable-diffusion-xl-base-1.0")
        XCTAssertEqual(NFKMLXStableDiffusionRelease(.lcmDreamshaperV7).repository, "SimianLuo/LCM_Dreamshaper_v7")
        XCTAssertEqual(NFKMLXStableDiffusionRelease(.lcmDreamshaperV7).configuration.unet.timeConditionProjectionDimensions, 256)
        XCTAssertEqual(NFKMLXStableDiffusionRelease(.sdxlBase).configuration.steps, 50)
    }

    // A whole run on random weights under each sampler: the loop, the guidance projection, and the
    // adapter plumbing hold together without a release.
    func testTinyTextToImageRunsUnderEachSampler() throws {
        try requireMLXRuntime()
        var lcm = NFKMLXSDTextToImageConfiguration.tiny.withSampler(.lcm, steps: 2, guidanceScale: 8)
        lcm.unet.timeConditionProjectionDimensions = 8
        for configuration in [NFKMLXSDTextToImageConfiguration.tiny.withSampler(.ddimTrailing, steps: 2), lcm,
                              NFKMLXSDTextToImageConfiguration.tiny.withSampler(.tcd, steps: 2)] {
            let backend = NFKMLXTextToImage.backend(configuration: configuration)
            let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a tiny test"], parameters: [:])
            let result = try backend.runInference(for: request)
            XCTAssertNotNil(result.output(forKey: NFKOutputImage))
        }
    }
}
