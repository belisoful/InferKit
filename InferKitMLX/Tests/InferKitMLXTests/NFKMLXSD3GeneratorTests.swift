//
//  NFKMLXSD3GeneratorTests.swift
//  InferKitMLXTests
//
//  The Stable Diffusion 3 generator's glue against diffusers' StableDiffusion3Pipeline at a tiny random
//  geometry (the three text towers, the joint sequence, the guided loop, the decode), its staging, and
//  the release config readers it builds from. Runs where MLX has a Metal library (see
//  Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSD3GeneratorTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tower(width: Int, projection: Int, activation: NFKCLIPActivation) -> NFKMLXSDTextEncoderConfiguration {
        var configuration = NFKMLXSDTextEncoderConfiguration()
        configuration.width = width
        configuration.layers = 3
        configuration.heads = 2
        configuration.intermediate = 2 * width
        configuration.activation = activation
        configuration.output = .penultimateHiddenState
        configuration.projectionDimensions = projection
        return configuration
    }

    private lazy var clipL = tower(width: 8, projection: 8, activation: .quickGELU)
    private lazy var clipG = tower(width: 8, projection: 12, activation: .gelu)
    private let t5 = NFKMLXT5Configuration(dModel: 24, layers: 2, heads: 2, keyDim: 12, ffDim: 48,
                                           vocabularySize: 32128, relativeBuckets: 16, relativeMaxDistance: 32)
    private let vaeConfiguration: NFKMLXSDVAEConfiguration = {
        var configuration = NFKMLXSDVAEConfiguration()
        configuration.latentChannels = 4
        configuration.blockChannels = [8, 16]
        configuration.layersPerBlock = 1
        configuration.normalizationGroups = 4
        configuration.useQuantConv = false
        configuration.scaleFactor = 1.5305
        configuration.shiftFactor = 0.0609
        return configuration
    }()

    /// CLIP-L's tokenizer (end-marker padding), a `!`-padding CLIP tokenizer in bigG's place, and T5's
    /// SentencePiece model.
    private func tokenizers() throws -> (NFKMLXSDPromptTokenizer, NFKMLXSDPromptTokenizer, NFKMLXSentencePieceSegmenter) {
        let environment = NFKMLXValidationConfig.environment
        guard let l = environment["IK_VAL_CLIP_TOKENIZER"], let g = environment["IK_VAL_CLIP_BANG_TOKENIZER"],
              let t5 = environment["IK_VAL_T5_TOKENIZER"] else {
            throw XCTSkip("set IK_VAL_CLIP_TOKENIZER, IK_VAL_CLIP_BANG_TOKENIZER and IK_VAL_T5_TOKENIZER")
        }
        let segmenter = NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: URL(fileURLWithPath: t5).appendingPathComponent("spiece.model")))
        return (try NFKMLXSDPromptTokenizer(directoryURL: URL(fileURLWithPath: l)),
                try NFKMLXSDPromptTokenizer(directoryURL: URL(fileURLWithPath: g)), segmenter)
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
        let y = b.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
        let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
        return dot / (sqrt(x.reduce(0) { $0 + $1 * $1 }) * sqrt(y.reduce(0) { $0 + $1 * $1 }))
    }

    private func generator(resident: Bool, loadTextEncoders: @escaping () throws -> NFKSD3TextStage,
                           loadPipeline: @escaping () throws -> NFKMLXSD3Pipeline) throws -> NFKMLXSD3Generator {
        let (l, g, segmenter) = try tokenizers()
        return NFKMLXSD3Generator(resident: resident, tokenizerL: l, tokenizerG: g, segmenter: segmenter, t5Length: 32,
                                  loadTextEncoders: loadTextEncoders, loadPipeline: loadPipeline)
    }

    // The glue against diffusers' StableDiffusion3Pipeline: each tower's ids, the joint sequence and the
    // pooled projection, the guided and unguided loops, and the decode. `run_reference.py sd3_pipeline`
    // records the reference.
    func testThePipelineMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_SD3_PIPELINE"] else {
            throw XCTSkip("set IK_PARITY_SD3_PIPELINE (run_reference.py sd3_pipeline)")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        func weights(_ prefix: String) -> [(String, MLXArray)] {
            arrays.compactMap { key, value in key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil }
        }
        func tower(_ configuration: NFKMLXSDTextEncoderConfiguration, _ prefix: String) throws -> NFKMLXSDTextEncoderNet {
            let net = NFKMLXSDTextEncoderNet(configuration: configuration)
            try NFKMLXWeights.apply(NFKMLXSDTextEncoder.remap(Dictionary(uniqueKeysWithValues: weights(prefix))), to: net,
                                    verifyShapes: true)
            return net
        }
        let encoder3 = NFKMLXT5Encoder.makeNet(t5)
        try NFKMLXWeights.apply(weights("t5::"), to: encoder3, verifyShapes: true)
        let stage = NFKSD3TextStage(clipL: try tower(clipL, "l::"), clipG: try tower(clipG, "g::"), t5: encoder3,
                                    jointDimensions: 24, t5Length: 32)
        let transformer = NFKMLXSD3TransformerNet(.tiny)
        try NFKMLXWeights.apply(weights("t::").map { ($0.0, $0.1.ndim == 4 ? $0.1.transposed(0, 2, 3, 1) : $0.1) },
                                to: transformer, verifyShapes: true)
        let vae = NFKMLXSDAutoencoder(configuration: vaeConfiguration)
        try NFKMLXWeights.apply(weights("v::").map { key, value in
            (NFKMLXStableDiffusionModels.remapVAEKey(key), value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }, to: vae, verifyShapes: true)
        let generator = try generator(resident: true, loadTextEncoders: { stage }, loadPipeline: {
            NFKMLXSD3Pipeline(transformer: transformer, vae: vae)
        })

        let prompt = generator.prompt("A red fox walking through fresh snow, cinematic")
        let ids = { (key: String) in try XCTUnwrap(arrays[key]).asArray(Int32.self).map(Int.init) }
        XCTAssertEqual(prompt.clipL, try ids("ids_l"), "CLIP-L's ids, padded with the end marker")
        XCTAssertEqual(prompt.clipG, try ids("ids_g"), "bigG's ids, padded with `!`")
        XCTAssertEqual(prompt.t5?.ids, try ids("ids_t5"), "T5's ids, ended and padded")
        let (sequence, pooled) = stage.embeddings(prompt)
        let (negativeSequence, negativePooled) = stage.embeddings(generator.prompt("blurry, low quality"))
        let sequenceCosine = cosine(sequence, try XCTUnwrap(arrays["prompt_embeds"]))
        let pooledCosine = cosine(pooled, try XCTUnwrap(arrays["pooled"]))
        XCTAssertEqual(sequence.shape, try XCTUnwrap(arrays["prompt_embeds"]).shape)
        XCTAssertGreaterThan(sequenceCosine, 0.999999, "the joint sequence is the reference's")
        XCTAssertGreaterThan(pooledCosine, 0.999999, "the pooled projection is the reference's")
        XCTAssertGreaterThan(cosine(negativeSequence, try XCTUnwrap(arrays["negative_embeds"])), 0.999999)
        XCTAssertGreaterThan(cosine(negativePooled, try XCTUnwrap(arrays["negative_pooled"])), 0.999999)

        let pipeline = NFKMLXSD3Pipeline(transformer: transformer, vae: vae)
        let start = try XCTUnwrap(arrays["latents"]).expandedDimensions(axis: 0)
        let guided = pipeline.denoise(start, promptEmbeds: sequence, pooled: pooled, negativeEmbeds: negativeSequence,
                                      negativePooled: negativePooled, steps: 5, guidance: 5)
        let unguided = pipeline.denoise(start, promptEmbeds: sequence, pooled: pooled, negativeEmbeds: negativeSequence,
                                        negativePooled: negativePooled, steps: 5, guidance: 1)
        let guidedCosine = cosine(guided, try XCTUnwrap(arrays["final_latents"]))
        let unguidedCosine = cosine(unguided, try XCTUnwrap(arrays["final_latents_unguided"]))
        let control = cosine(unguided, try XCTUnwrap(arrays["final_latents"]))
        let image = clip(pipeline.decode(guided)[0] / 2 + 0.5, min: 0, max: 1)
        let reference = try XCTUnwrap(arrays["image"])
        XCTAssertEqual(image.shape, reference.shape)
        let imageCosine = cosine(image, reference)
        print("VALIDATION PARITY sd3-pipeline: joint sequence \(sequenceCosine), pooled \(pooledCosine), "
              + "final latents \(guidedCosine) (unguided \(unguidedCosine), control \(control)), image \(imageCosine)")
        XCTAssertGreaterThan(guidedCosine, 0.99999, "the guided loop matches the reference")
        XCTAssertGreaterThan(unguidedCosine, 0.99999, "and the unguided one")
        XCTAssertLessThan(control, 0.99, "the record separates guided from unguided")
        XCTAssertGreaterThan(imageCosine, 0.99999, "the decode matches the reference")
    }

    // A staged generator and a resident one over the same weights produce the same image, and the
    // staged one loads the three towers once per image, both prompts together.
    func testStagingReleasesEachStageAndChangesNothing() throws {
        try requireMLXRuntime()
        MLXRandom.seed(57)
        func captured(_ module: Module) -> [(String, MLXArray)] {
            let parameters = module.parameters().flattened()
            eval(parameters.map(\.1))
            return parameters
        }
        let lWeights = captured(NFKMLXSDTextEncoderNet(configuration: clipL))
        let gWeights = captured(NFKMLXSDTextEncoderNet(configuration: clipG))
        let t5Weights = captured(NFKMLXT5Encoder.makeNet(t5))
        let transformerWeights = captured(NFKMLXSD3TransformerNet(.tiny))
        let vaeWeights = captured(NFKMLXSDAutoencoder(configuration: vaeConfiguration))
        var loads = (text: 0, pipeline: 0)
        func make(resident: Bool) throws -> NFKMLXSD3Generator {
            let generator = try self.generator(resident: resident, loadTextEncoders: {
                loads.text += 1
                let l = NFKMLXSDTextEncoderNet(configuration: self.clipL), g = NFKMLXSDTextEncoderNet(configuration: self.clipG)
                let encoder3 = NFKMLXT5Encoder.makeNet(self.t5)
                try NFKMLXWeights.apply(lWeights, to: l)
                try NFKMLXWeights.apply(gWeights, to: g)
                try NFKMLXWeights.apply(t5Weights, to: encoder3)
                return NFKSD3TextStage(clipL: l, clipG: g, t5: encoder3, jointDimensions: 24, t5Length: 32)
            }, loadPipeline: {
                loads.pipeline += 1
                let transformer = NFKMLXSD3TransformerNet(.tiny)
                try NFKMLXWeights.apply(transformerWeights, to: transformer)
                let vae = NFKMLXSDAutoencoder(configuration: self.vaeConfiguration)
                try NFKMLXWeights.apply(vaeWeights, to: vae)
                return NFKMLXSD3Pipeline(transformer: transformer, vae: vae)
            })
            if resident {
                try generator.loadResident()
            }
            generator.steps = 3
            generator.guidance = 5
            return generator
        }
        let resident = try make(resident: true)
        let residentImages = try (0 ..< 2).map { _ in try resident.image(forPrompt: "a red fox", width: 16, height: 16, seed: 3) }
        XCTAssertEqual((loads.text, loads.pipeline) == (1, 1), true)
        loads = (0, 0)
        let staged = try make(resident: false)
        let stagedImages = try (0 ..< 2).map { _ in try staged.image(forPrompt: "a red fox", width: 16, height: 16, seed: 3) }
        XCTAssertEqual(loads.text, 2, "a staged generator loads the towers once per image")
        XCTAssertEqual(loads.pipeline, 2)
        XCTAssertFalse(staged.isHoldingTextEncoders || staged.isHoldingPipeline, "and holds neither after")
        XCTAssertEqual(stagedImages[0].shape, [16, 16, 3])
        for (a, b) in zip(residentImages, stagedImages) {
            XCTAssertEqual(b.reshaped([-1]).asArray(Float.self), a.reshaped([-1]).asArray(Float.self),
                           "staging does not change the image")
        }
    }

    private func write(_ json: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sd3-config-\(UUID().uuidString).json")
        try Data(json.utf8).write(to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    // The released component configs: the schedule's static shift over the scheduler's own ramp, the
    // autoencoder without its quantization convolutions, and CLIP-L as a projecting tower.
    func testTheReleaseConfigReadersReadTheComponents() throws {
        var schedule = NFKMLXFlowMatchScheduler(try NFKMLXSD3Release.schedule(fromHuggingFace: write("""
        {"_class_name":"FlowMatchEulerDiscreteScheduler","num_train_timesteps":1000,"shift":3.0}
        """)))
        var reference = NFKMLXFlowMatchScheduler(.sd3)
        schedule.setTimesteps(28, sequenceLength: 4096)
        reference.setTimesteps(28, sequenceLength: 4096)
        XCTAssertEqual(schedule.sigmas, reference.sigmas, "a static shift of 3 is the SD3 preset")

        let vae = try NFKMLXStableDiffusionModels.vaeConfiguration(fromHuggingFace: write("""
        {"_class_name":"AutoencoderKL","block_out_channels":[128,256,512,512],"latent_channels":16,
        "layers_per_block":2,"norm_num_groups":32,"scaling_factor":1.5305,"shift_factor":0.0609,
        "use_quant_conv":false,"use_post_quant_conv":false}
        """))
        XCTAssertEqual(vae.latentChannels, 16)
        XCTAssertEqual(vae.scaleFactor, 1.5305)
        XCTAssertEqual(vae.shiftFactor, 0.0609)
        XCTAssertFalse(vae.useQuantConv)
        XCTAssertThrowsError(try NFKMLXStableDiffusionModels.vaeConfiguration(fromHuggingFace: write("""
        {"use_quant_conv":true,"use_post_quant_conv":false}
        """)), "one quantization convolution of two is refused")

        let tower = try NFKMLXSDTextEncoder.configuration(fromHuggingFace: write("""
        {"architectures":["CLIPTextModelWithProjection"],"hidden_size":768,"num_hidden_layers":12,
        "num_attention_heads":12,"intermediate_size":3072,"hidden_act":"quick_gelu","projection_dim":768,
        "vocab_size":49408,"max_position_embeddings":77}
        """), output: .penultimateHiddenState)
        XCTAssertEqual(tower.width, 768)
        XCTAssertEqual(tower.projectionDimensions, 768)
        XCTAssertEqual(tower.activation, .quickGELU)
        XCTAssertEqual(tower.output, .penultimateHiddenState)
    }
}
