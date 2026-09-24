//
//  NFKMLXWanPipelineTests.swift
//  InferKitMLXTests
//
//  The Wan text-to-video pipeline glue (DiT + flow loop → 3D causal VAE), on matching tiny
//  configurations so the chaining — the guided sampling loop, the DiT↔VAE bridge, the streaming
//  decode — is exercised with random weights. Runs where MLX has a Metal library (see
//  Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXWanPipelineTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func pipeline() -> NFKMLXWanPipeline {
        let transformer = NFKMLXWanTransformerNet(.tiny)                 // inChannels 4, textDim 10
        let vae = NFKMLXWanVideoVAENet(.tiny)                          // zDim 4
        return NFKMLXWanPipeline(transformer: transformer, vae: vae)
    }

    func testThePipelineGeneratesAVideoOfTheExpectedShape() throws {
        try requireMLXRuntime()
        let text = MLXRandom.normal([6, 10])
        let video = pipeline().generate(textEmbeds: text, negativeEmbeds: nil,
                                        frames: 2, height: 4, width: 4, steps: 3, guidance: 1)
        eval(video)
        XCTAssertEqual(video.shape[0], 1)
        XCTAssertEqual(video.ndim, 5, "a [B, T, H, W, 3] clip")
        XCTAssertEqual(video.shape[4], 3, "RGB output")
        XCTAssertEqual(video.shape[2], 16, "the VAE upsamples the 4-wide latent by 4× spatially")
    }

    func testClassifierFreeGuidanceRunsBothPrompts() throws {
        try requireMLXRuntime()
        let text = MLXRandom.normal([6, 10])
        let negative = MLXRandom.normal([6, 10])
        let video = pipeline().generate(textEmbeds: text, negativeEmbeds: negative,
                                        frames: 2, height: 4, width: 4, steps: 2, guidance: 5)
        eval(video)
        XCTAssertEqual(video.shape[0], 1)
        XCTAssertEqual(video.shape[4], 3)
    }

    // MARK: Against diffusers' WanPipeline

    private let umt5 = NFKMLXT5Configuration(dModel: 32, layers: 2, heads: 2, keyDim: 16, ffDim: 64,
                                             vocabularySize: 32128, relativeBuckets: 16, relativeMaxDistance: 32,
                                             perLayerBias: true)
    private let transformer = NFKMLXWanConfiguration(inChannels: 4, heads: 2, headDim: 16, layers: 2, ffnDim: 48,
                                                     textDim: 32, patchSize: [1, 2, 2])
    private let mean = MLXArray([Float(0.1), -0.2, 0.3, -0.4])
    private let std = MLXArray([Float(1.1), 0.9, 1.3, 0.8])

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

    private func segmenter() throws -> NFKMLXSentencePieceSegmenter {
        guard let tokenizer = NFKMLXValidationConfig.environment["IK_VAL_T5_TOKENIZER"] else {
            throw XCTSkip("set IK_VAL_T5_TOKENIZER (a T5 SentencePiece tokenizer directory)")
        }
        return NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: URL(fileURLWithPath: tokenizer).appendingPathComponent("spiece.model")))
    }

    // The glue at a tiny random geometry: the cleaned prompt padded and masked, the masked umT5 encode
    // cut at the prompt's length and zero-padded, the UniPC flow loop with guidance, the latent
    // statistics, and the Wan 2.1 decode.
    func testThePipelineMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN_PIPELINE"] else {
            throw XCTSkip("set IK_PARITY_WAN_PIPELINE (run_reference.py wan_pipeline)")
        }
        let segmenter = try segmenter()
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        func weights(_ prefix: String) -> [(String, MLXArray)] {
            arrays.compactMap { key, value in key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil }
        }
        let encoder = NFKMLXT5Encoder.makeNet(umt5)
        try NFKMLXWeights.apply(weights("t5::"), to: encoder, verifyShapes: true)
        let net = NFKMLXWanTransformerNet(transformer)
        try NFKMLXWeights.apply(weights("t::").map { ($0.0, $0.1.ndim == 5 ? $0.1.transposed(0, 2, 3, 4, 1) : $0.1) },
                                to: net, verifyShapes: true)
        let vae = NFKMLXWanVideoVAENet(.tiny21)
        try NFKMLXWeights.apply(weights("v::").map(NFKMLXWanRelease.adaptedVAE), to: vae, verifyShapes: true)

        let text = NFKWanTextStage(encoder)
        let prompt = NFKMLXT5Prompt(NFKMLXWanVideoGenerator.cleaned("  A red fox   walking through fresh snow, cinematic "),
                                    segmenter: segmenter, length: 64)
        let negative = NFKMLXT5Prompt(NFKMLXWanVideoGenerator.cleaned("worst quality, blurry"),
                                      segmenter: segmenter, length: 64)
        XCTAssertEqual(prompt.ids, try XCTUnwrap(arrays["input_ids"]).asArray(Int32.self).map(Int.init),
                       "the cleaned prompt tokenizes, ends and pads as the reference's")
        XCTAssertEqual(prompt.mask, try XCTUnwrap(arrays["mask"]).asArray(Int32.self).map(Int.init))
        let features = text.features(prompt), negativeFeatures = text.features(negative)
        let textCosine = cosine(features, try XCTUnwrap(arrays["prompt_embeds"]))
        XCTAssertGreaterThan(textCosine, 0.999999, "the masked encode, cut and zero-padded, is the reference's")
        XCTAssertGreaterThan(cosine(negativeFeatures, try XCTUnwrap(arrays["negative_embeds"])), 0.999999)

        let pipeline = NFKMLXWanPipeline(transformer: net, vae: vae, latentsMean: mean, latentsStd: std,
                                         schedule: NFKMLXUniPCConfiguration(flowShift: 3))
        let latent = pipeline.denoise(textEmbeds: features, negativeEmbeds: negativeFeatures, frames: 3, height: 8,
                                      width: 8, steps: 4, guidance: 5, latents: try XCTUnwrap(arrays["latents"]))
        let latentCosine = cosine(latent, try XCTUnwrap(arrays["final_latents"]))
        let expandedCosine = cosine(latent, try XCTUnwrap(arrays["final_latents_expanded"]))
        let video = clip(pipeline.decode(latent)[0] / 2 + 0.5, min: 0, max: 1)
        let reference = try XCTUnwrap(arrays["video"])
        XCTAssertEqual(video.shape, reference.shape)
        let videoCosine = cosine(video, reference)
        print("VALIDATION PARITY wan-pipeline: prompt features \(textCosine), final latents \(latentCosine) "
              + "(expanded timesteps \(expandedCosine)), video \(videoCosine)")
        XCTAssertGreaterThan(latentCosine, 0.99999, "the loop matches the reference")
        XCTAssertGreaterThan(expandedCosine, 0.99999, "and the per-token timestep form of it")
        XCTAssertGreaterThan(videoCosine, 0.99999, "the de-normalized decode matches the reference")
    }

    // A staged generator and a resident one over the same weights produce the same clip.
    func testStagingReleasesEachStageAndChangesNothing() throws {
        try requireMLXRuntime()
        let segmenter = try segmenter()
        MLXRandom.seed(35)
        func captured(_ module: Module) -> [(String, MLXArray)] {
            let parameters = module.parameters().flattened()
            eval(parameters.map(\.1))
            return parameters
        }
        let encoderWeights = captured(NFKMLXT5Encoder.makeNet(umt5))
        let transformerWeights = captured(NFKMLXWanTransformerNet(transformer))
        let vaeWeights = captured(NFKMLXWanVideoVAENet(.tiny21))
        var loads = (text: 0, pipeline: 0)
        func generator(resident: Bool) throws -> NFKMLXWanVideoGenerator {
            let generator = NFKMLXWanVideoGenerator(
                resident: resident, segmenter: segmenter,
                compression: NFKMLXWanVideoGenerator.compression(of: .tiny21), sequenceLength: 64,
                loadTextEncoder: {
                    loads.text += 1
                    let encoder = NFKMLXT5Encoder.makeNet(self.umt5)
                    try NFKMLXWeights.apply(encoderWeights, to: encoder)
                    return encoder
                },
                loadPipeline: {
                    loads.pipeline += 1
                    let net = NFKMLXWanTransformerNet(self.transformer)
                    try NFKMLXWeights.apply(transformerWeights, to: net)
                    let vae = NFKMLXWanVideoVAENet(.tiny21)
                    try NFKMLXWeights.apply(vaeWeights, to: vae)
                    return NFKMLXWanPipeline(transformer: net, vae: vae, latentsMean: self.mean, latentsStd: self.std)
                })
            if resident {
                try generator.loadResident()
            }
            generator.steps = 2
            return generator
        }
        let resident = try generator(resident: true)
        let residentClips = try (0 ..< 2).map { _ in
            try resident.video(forPrompt: "a red fox", frames: 5, width: 16, height: 16, seed: 3)
        }
        XCTAssertEqual((loads.text, loads.pipeline) == (1, 1), true)
        loads = (0, 0)
        let staged = try generator(resident: false)
        let stagedClips = try (0 ..< 2).map { _ in
            try staged.video(forPrompt: "a red fox", frames: 5, width: 16, height: 16, seed: 3)
        }
        XCTAssertEqual(loads.text, 2, "a staged generator loads umT5 once per clip, both prompts together")
        XCTAssertEqual(loads.pipeline, 2)
        XCTAssertFalse(staged.isHoldingTextEncoder || staged.isHoldingPipeline, "and holds neither after")
        XCTAssertEqual(stagedClips[0].shape, [5, 16, 16, 3])
        for (a, b) in zip(residentClips, stagedClips) {
            XCTAssertEqual(b.reshaped([-1]).asArray(Float.self), a.reshaped([-1]).asArray(Float.self),
                           "staging does not change the clip")
        }
    }

    // The release's own umT5 tokenizer (a 256k-piece SentencePiece model) against the fast tokenizer
    // Wan's pipeline names, over prompts in several scripts: the cleanup, the pieces, the end token,
    // the padding to 512, and the mask. `run_reference.py umt5_tokenizer` records the reference.
    func testTheUMT5TokenizerMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let directory = NFKMLXValidationConfig.environment["IK_VAL_UMT5_TOKENIZER"],
              let path = NFKMLXValidationConfig.environment["IK_PARITY_UMT5_TOKENIZER"] else {
            throw XCTSkip("set IK_VAL_UMT5_TOKENIZER (Wan's tokenizer/) and IK_PARITY_UMT5_TOKENIZER")
        }
        let prompts = [
            "A red fox walking through fresh snow, cinematic",
            "  Two   cats\tplaying\nin the sun  ",
            "一只红色的狐狸在雪地里行走，电影感",
            "Ein roter Fuchs läuft durch frischen Schnee, 4K, HDR!",
            "キツネが雪の中を歩く 🦊❄️",
            "Un zorro rojo — cámara lenta, 24 fps, f/1.8",
            "Лиса бежит по снегу. 1234567890",
            "naïve café résumé coöperate",
            "",
        ]
        let segmenter = NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: URL(fileURLWithPath: directory).appendingPathComponent("spiece.model")))
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        var exact = 0
        for (index, text) in prompts.enumerated() {
            let prompt = NFKMLXT5Prompt(NFKMLXWanVideoGenerator.cleaned(text), segmenter: segmenter,
                                        length: NFKMLXWanVideoGenerator.maximumSequenceLength)
            let ids = try XCTUnwrap(arrays["ids_\(index)"]).asArray(Int32.self).map(Int.init)
            let mask = try XCTUnwrap(arrays["mask_\(index)"]).asArray(Int32.self).map(Int.init)
            XCTAssertEqual(prompt.ids, ids, "prompt \(index): \(text)")
            XCTAssertEqual(prompt.mask, mask, "prompt \(index)")
            exact += prompt.ids == ids ? 1 : 0
        }
        print("VALIDATION PARITY umt5-tokenizer: \(exact) of \(prompts.count) prompts token-exact")
    }
}
