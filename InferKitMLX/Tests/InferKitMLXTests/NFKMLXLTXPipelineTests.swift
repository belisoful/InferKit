//
//  NFKMLXLTXPipelineTests.swift
//  InferKitMLXTests
//
//  The LTX-Video text-to-video glue: the prompt padded and masked, the DiT denoised over the pipeline's
//  own flow ramp with classifier-free guidance, the latent denormalization, and the decode. Measured
//  against diffusers' `LTXPipeline` at a tiny random geometry (`run_reference.py ltx_pipeline`); each
//  model is measured against its own reference elsewhere.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLTXPipelineTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private let t5 = NFKMLXT5Configuration(dModel: 32, layers: 2, heads: 2, keyDim: 16, ffDim: 64,
                                           vocabularySize: 32128, relativeBuckets: 16, relativeMaxDistance: 32)
    private let transformer = NFKMLXLTXTransformerConfiguration(
        inChannels: 16, heads: 2, headDim: 8, layers: 2, crossAttentionDim: 16, captionChannels: 32)
    private let vae = NFKMLXLTXVAEConfiguration(latentChannels: 16, blockOutChannels: [8, 16, 16, 16],
                                                layersPerBlock: [1, 1, 1, 1, 1])

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
        let y = b.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
        var dot = 0.0, xx = 0.0, yy = 0.0
        for (p, q) in zip(x, y) {
            dot += p * q
            xx += p * p
            yy += q * q
        }
        return dot / (xx.squareRoot() * yy.squareRoot())
    }

    func testThePipelineGeneratesAClipOfTheExpectedShape() throws {
        try requireMLXRuntime()
        let pipeline = NFKMLXLTXPipeline(transformer: NFKMLXLTXTransformer.makeNet(transformer),
                                         vae: NFKMLXLTXVideoVAE.makeNet(vae))
        let text = MLXRandom.normal([1, 6, 32])
        let mask = MLXArray([Int32(1), 1, 1, 0, 0, 0]).reshaped([1, 6])
        let latents = pipeline.denoise(text: text, textMask: mask, negativeText: MLXRandom.normal([1, 6, 32]),
                                       negativeMask: mask, frames: 2, height: 2, width: 2, steps: 2)
        let video = pipeline.decode(latents, frames: 2, height: 2, width: 2)
        XCTAssertEqual(video.shape, [1, 9, 64, 64, 3], "8 frames per latent frame after the first, 32 pixels per cell")
    }

    // The padding a masked caption carries does not reach the transformer: two captions that differ only
    // past the mask give the same velocity.
    func testTheCaptionMaskHidesThePadding() throws {
        try requireMLXRuntime()
        MLXRandom.seed(9)
        let net = NFKMLXLTXTransformer.makeNet(transformer)
        let latent = MLXRandom.normal([1, 8, 16])
        let caption = MLXRandom.normal([1, 6, 32])
        let other = concatenated([caption[0..., 0 ..< 3], MLXRandom.normal([1, 3, 32])], axis: 1)
        let mask = MLXArray([Int32(1), 1, 1, 0, 0, 0]).reshaped([1, 6])
        let a = net(latent, text: caption, timestep: MLXArray([Float(500)]), grid: (2, 2, 2), ropeScale: (1, 1, 1),
                    textMask: mask)
        let b = net(latent, text: other, timestep: MLXArray([Float(500)]), grid: (2, 2, 2), ropeScale: (1, 1, 1),
                    textMask: mask)
        XCTAssertGreaterThan(cosine(a, b), 0.999999, "the padded positions carry an additive bias of −10000")
    }

    func testThePipelineMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_LTX_PIPELINE"],
              let tokenizer = NFKMLXValidationConfig.environment["IK_VAL_T5_TOKENIZER"] else {
            throw XCTSkip("set IK_PARITY_LTX_PIPELINE (run_reference.py ltx_pipeline) and IK_VAL_T5_TOKENIZER")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        func weights(_ prefix: String) -> [(String, MLXArray)] {
            arrays.compactMap { key, value in key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil }
        }
        let encoder = NFKMLXT5Encoder.makeNet(t5)
        try NFKMLXWeights.apply(weights("t5::"), to: encoder, verifyShapes: true)
        let net = NFKMLXLTXTransformer.makeNet(transformer)
        try NFKMLXWeights.apply(weights("t::"), to: net, verifyShapes: true)
        let autoencoder = NFKMLXLTXVideoVAE.makeNet(vae)
        try NFKMLXWeights.apply(weights("v::").map { key, value in
            (key, value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value)
        }, to: autoencoder, verifyShapes: true)
        let segmenter = NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: URL(fileURLWithPath: tokenizer).appendingPathComponent("spiece.model")))

        let prompt = NFKMLXT5Prompt("A red fox walking through fresh snow, cinematic", segmenter: segmenter, length: 128)
        let negative = NFKMLXT5Prompt("worst quality, blurry", segmenter: segmenter, length: 128)
        XCTAssertEqual(prompt.ids, try XCTUnwrap(arrays["input_ids"]).asArray(Int32.self).map(Int.init),
                       "the prompt tokenizes, ends and pads as the reference's")
        XCTAssertEqual(prompt.mask, try XCTUnwrap(arrays["mask"]).asArray(Int32.self).map(Int.init))
        XCTAssertEqual(negative.mask, try XCTUnwrap(arrays["negative_mask"]).asArray(Int32.self).map(Int.init))

        let text = encoder(prompt.tokens), negativeText = encoder(negative.tokens)
        let textCosine = cosine(text[0], try XCTUnwrap(arrays["prompt_embeds"]))
        XCTAssertGreaterThan(textCosine, 0.999999, "the prompt features are the reference's")

        var scheduler = NFKMLXFlowMatchScheduler(.ltxVideoPipeline)
        scheduler.setTimesteps(4, sequenceLength: 8)
        for (mine, theirs) in zip(scheduler.sigmas, try XCTUnwrap(arrays["sigmas"]).asArray(Float.self)) {
            XCTAssertEqual(mine, theirs, accuracy: 1e-5, "the pipeline's own ramp, shifted by the sequence length")
        }

        let pipeline = NFKMLXLTXPipeline(transformer: net, vae: autoencoder)
        let latents = pipeline.denoise(text: text, textMask: prompt.maskArray, negativeText: negativeText,
                                       negativeMask: negative.maskArray, frames: 2, height: 2, width: 2, steps: 4,
                                       guidance: 3, latents: try XCTUnwrap(arrays["latents"]), frameRate: 25)
        let latentCosine = cosine(latents[0], try XCTUnwrap(arrays["final_latents"]))
        let video = clip(pipeline.decode(latents, frames: 2, height: 2, width: 2)[0] / 2 + 0.5, min: 0, max: 1)
        let reference = try XCTUnwrap(arrays["video"])
        XCTAssertEqual(video.shape, reference.shape)
        let videoCosine = cosine(video, reference)
        print("VALIDATION PARITY ltx-pipeline: prompt features \(textCosine), final latents \(latentCosine), "
              + "video \(videoCosine)")
        XCTAssertGreaterThan(latentCosine, 0.99999, "the loop matches the reference")
        XCTAssertGreaterThan(videoCosine, 0.99999, "the denormalized decode matches the reference")
    }

    // A staged generator and a resident one over the same weights produce the same clip, and the staged
    // one holds neither stage between clips.
    func testStagingReleasesEachStageAndChangesNothing() throws {
        try requireMLXRuntime()
        guard let tokenizer = NFKMLXValidationConfig.environment["IK_VAL_T5_TOKENIZER"] else {
            throw XCTSkip("set IK_VAL_T5_TOKENIZER (a T5 SentencePiece tokenizer directory)")
        }
        let segmenter = NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: URL(fileURLWithPath: tokenizer).appendingPathComponent("spiece.model")))
        MLXRandom.seed(33)
        func captured(_ module: Module) -> [(String, MLXArray)] {
            let parameters = module.parameters().flattened()
            eval(parameters.map(\.1))
            return parameters
        }
        let encoderWeights = captured(NFKMLXT5Encoder.makeNet(t5))
        let transformerWeights = captured(NFKMLXLTXTransformer.makeNet(transformer))
        let vaeWeights = captured(NFKMLXLTXVideoVAE.makeNet(vae))
        var loads = (text: 0, pipeline: 0)
        func generator(resident: Bool) throws -> NFKMLXLTXVideoGenerator {
            let generator = NFKMLXLTXVideoGenerator(
                resident: resident, segmenter: segmenter,
                loadTextEncoder: {
                    loads.text += 1
                    let encoder = NFKMLXT5Encoder.makeNet(self.t5)
                    try NFKMLXWeights.apply(encoderWeights, to: encoder)
                    return encoder
                },
                loadPipeline: {
                    loads.pipeline += 1
                    let net = NFKMLXLTXTransformer.makeNet(self.transformer)
                    try NFKMLXWeights.apply(transformerWeights, to: net)
                    let vae = NFKMLXLTXVideoVAE.makeNet(self.vae)
                    try NFKMLXWeights.apply(vaeWeights, to: vae)
                    return NFKMLXLTXPipeline(transformer: net, vae: vae)
                })
            if resident {
                try generator.loadResident()
            }
            generator.steps = 2
            return generator
        }

        let resident = try generator(resident: true)
        let residentClips = try (0 ..< 2).map { _ in
            try resident.video(forPrompt: "a red fox", frames: 9, width: 64, height: 64, seed: 3)
        }
        XCTAssertEqual(loads.text, 1)
        XCTAssertEqual(loads.pipeline, 1)
        loads = (0, 0)
        let staged = try generator(resident: false)
        let stagedClips = try (0 ..< 2).map { _ in
            try staged.video(forPrompt: "a red fox", frames: 9, width: 64, height: 64, seed: 3)
        }
        XCTAssertEqual(loads.text, 2, "a staged generator loads T5 once per clip, both prompts together")
        XCTAssertEqual(loads.pipeline, 2)
        XCTAssertFalse(staged.isHoldingTextEncoder || staged.isHoldingPipeline, "and holds neither after")
        XCTAssertEqual(stagedClips[0].shape, [9, 64, 64, 3])
        for (a, b) in zip(residentClips, stagedClips) {
            XCTAssertEqual(b.reshaped([-1]).asArray(Float.self), a.reshaped([-1]).asArray(Float.self),
                           "staging does not change the clip")
        }
    }
}
