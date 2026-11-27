//
//  NFKMLXWanVACEFunTests.swift
//  InferKitMLXTests
//
//  Wan 2.2 VACE-Fun A14B (alibaba-pai, VideoX-Fun): the VACE blocks' names, and, where the validation store holds
//  the test-dev release and its record, the whole sampler against VideoX-Fun's own `Wan2_2VaceFunPipeline`
//  (`run_reference.py wan_vace_fun_pipeline`). Runs where MLX has a Metal library (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXWanVACEFunTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The cosine of two arrays, summed in double precision on the host.
    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).asArray(Float.self), y = b.asType(.float32).asArray(Float.self)
        precondition(x.count == y.count, "shapes \(a.shape) and \(b.shape)")
        var dot = 0.0, xx = 0.0, yy = 0.0
        for i in x.indices {
            dot += Double(x[i]) * Double(y[i])
            xx += Double(x[i]) * Double(x[i])
            yy += Double(y[i]) * Double(y[i])
        }
        return dot / (xx.squareRoot() * yy.squareRoot())
    }

    func testTheVACEBlocksNamesMapToTheModule() {
        XCTAssertEqual(NFKMLXWanOriginalRelease.transformerModuleKey(forRelease: "vace_blocks.3.self_attn.q.weight"),
                       "vace_blocks.3.attn1.to_q.weight")
        XCTAssertEqual(NFKMLXWanOriginalRelease.transformerModuleKey(forRelease: "vace_blocks.0.before_proj.weight"),
                       "vace_blocks.0.before_proj.weight")
        XCTAssertEqual(NFKMLXWanOriginalRelease.transformerModuleKey(forRelease: "vace_blocks.7.norm3.bias"),
                       "vace_blocks.7.norm2.bias")
        XCTAssertEqual(NFKMLXWanOriginalRelease.transformerModuleKey(forRelease: "vace_patch_embedding.weight"),
                       "vace_patch_embedding.weight")
    }

    private let prompt = "A red fox  walking through fresh snow, cinematic"
    private let negative = "worst quality, blurry"

    func testVACEFunMatchesVideoXFun() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_WAN22_VACE_FUN_TESTDEV"] else {
            throw XCTSkip("set IK_VAL_WAN22_VACE_FUN_TESTDEV (Tools/wan-fun-testdev/make_testdev.py)")
        }
        guard let recordPath = NFKMLXValidationConfig.environment["IK_PARITY_WAN22_VACE_FUN_PIPELINE"] else {
            throw XCTSkip("set IK_PARITY_WAN22_VACE_FUN_PIPELINE (run_reference.py wan_vace_fun_pipeline --checkpoint <testdev>)")
        }
        let directory = URL(fileURLWithPath: path)
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: recordPath))
        let loaded = try NFKMLXWanFunRelease.pipeline(directoryURL: directory, boundary: NFKMLXWanFunPipeline.vaceFunBoundary)
        let pipeline = NFKMLXWanFunPipeline(transformer: loaded.transformer, highNoiseTransformer: loaded.highNoiseTransformer,
                                            vae: loaded.vae, latentsMean: loaded.latentsMean, latentsStd: loaded.latentsStd,
                                            boundary: NFKMLXWanFunPipeline.vaceFunBoundary, schedule: .wanVACEFun)
        let encoder = try NFKMLXWanOriginalRelease.textEncoder(
            from: directory.appendingPathComponent(NFKMLXWanOriginalRelease.textEncoderFile))
        let segmenter = try NFKMLXWanFunRelease.segmenter(directory)
        for (name, text) in [("prompt", prompt), ("negative", negative)] {
            let ids = NFKMLXWanFunRelease.prompt(text, segmenter: segmenter).ids
            XCTAssertEqual(ids.map(Int32.init), record["\(name)_ids"]!.asArray(Int32.self), "\(name) token ids")
        }
        let text = NFKMLXWanFunRelease.features(prompt, encoder: encoder, segmenter: segmenter)
        let negativeText = NFKMLXWanFunRelease.features(negative, encoder: encoder, segmenter: segmenter)
        var measured = [String: Double]()
        measured["prompt features"] = cosine(text[0 ..< record["prompt_embeds"]!.dim(0)], record["prompt_embeds"]!)
        let control = record["control_video"]!
        let frames = control.dim(0), height = control.dim(1), width = control.dim(2)
        let runs: [(String, NFKMLXWanVACEConditions, MLXArray)] = [
            ("", NFKMLXWanVACEConditions(controlVideo: control, subjectImages: [record["subject_image"]!]), record["latents"]!),
            ("mask.", NFKMLXWanVACEConditions(video: record["inpaint_video"]!, mask: record["region_mask"]!),
             record["latents"]![0..., 1...]),
        ]
        for (prefix, conditions, noise) in runs {
            let (prepared, references) = try pipeline.prepareVACE(conditions, frames: frames, height: height, width: width)
            measured["\(prefix)vace context"] = cosine(prepared.vaceContext!, record[prefix + "input.vace_context"]!)
            var steps = [MLXArray]()
            let latents = try pipeline.denoise(text: text, negative: negativeText, prepared: prepared, noise: noise,
                                               steps: 3, guidance: 5, onStep: { steps.append($1) })
            for (index, step) in steps.enumerated() {
                measured["\(prefix)step \(index)"] = cosine(step, record["\(prefix)step.\(index)"]!)
            }
            var scheduler = NFKMLXFlowMatchScheduler(pipeline.schedule)
            scheduler.setTimesteps(3, sequenceLength: 0)
            XCTAssertEqual(scheduler.timesteps.map { Int32($0 >= pipeline.boundary * 1000 ? 1 : 0) },
                           record[prefix + "experts"]!.asArray(Int32.self), "\(prefix)experts")
            let final = latents[0..., references...]
            measured["\(prefix)final latents"] = cosine(final, record[prefix + "final_latents"]!)
            measured["\(prefix)clip"] = cosine(pipeline.decode(final), record[prefix + "video"]!)
        }
        for (name, value) in measured.sorted(by: { $0.key < $1.key }) {
            print("wan vace fun \(name): \(String(format: "%.16f", value))")
            XCTAssertGreaterThan(value, 0.9999999, name)
        }
    }

    // MARK: The generator

    private func testDev() throws -> (URL, [String: MLXArray]) {
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_WAN22_VACE_FUN_TESTDEV"] else {
            throw XCTSkip("set IK_VAL_WAN22_VACE_FUN_TESTDEV (Tools/wan-fun-testdev/make_testdev.py)")
        }
        guard let recordPath = NFKMLXValidationConfig.environment["IK_PARITY_WAN22_VACE_FUN_PIPELINE"] else {
            throw XCTSkip("set IK_PARITY_WAN22_VACE_FUN_PIPELINE (run_reference.py wan_vace_fun_pipeline --checkpoint <testdev>)")
        }
        return (URL(fileURLWithPath: path), try MLX.loadArrays(url: URL(fileURLWithPath: recordPath)))
    }

    /// The generator's final latents from the record's control video, subject image, and noise.
    private func generatorRun(_ generator: NFKMLXWanFunGenerator, _ record: [String: MLXArray]) throws -> MLXArray {
        generator.steps = 3
        let control = record["control_video"]!
        return try generator.generate(
            prompt: prompt, negativePrompt: negative, conditions: NFKMLXWanFunConditions(),
            vace: NFKMLXWanVACEConditions(controlVideo: control, subjectImages: [record["subject_image"]!]),
            frames: control.dim(0), width: control.dim(2), height: control.dim(1), seed: 0, noise: record["latents"]!).latent
    }

    func testTheGeneratorMatchesVideoXFunThroughEveryResidency() throws {
        try requireMLXRuntime()
        let (directory, record) = try testDev()
        let resident = try NFKMLXWanFunGenerator.generator(directoryURL: directory, residency: .resident,
                                                           precision: .float32, quantizationBits: 0)
        XCTAssertEqual(resident.variant, .vaceA14B)
        XCTAssertEqual(resident.guidance, 5)
        let held = try generatorRun(resident, record)
        let measured = cosine(held, record["final_latents"]!)
        print("wan vace fun generator: \(String(format: "%.16f", measured))")
        XCTAssertGreaterThan(measured, 0.9999999)

        let staged = try NFKMLXWanFunGenerator.generator(directoryURL: directory, residency: .staged,
                                                         precision: .float32, quantizationBits: 0)
        XCTAssertEqual((try generatorRun(staged, record) - held).abs().max().item(Float.self), 0, "staged")
        let streamed = try NFKMLXWanFunGenerator.generator(directoryURL: directory, residency: .resident,
                                                           precision: .float32, bits: nil, streamedBlocks: [1])
        XCTAssertGreaterThan(streamed.streamedLayerCount, 0)
        XCTAssertEqual((try generatorRun(streamed, record) - held).abs().max().item(Float.self), 0, "streamed")
        let packed = try NFKMLXWanFunGenerator.generator(directoryURL: directory, residency: .resident,
                                                         precision: .float32, quantizationBits: 8)
        XCTAssertGreaterThan(cosine(try generatorRun(packed, record), held), 0.999, "8-bit")
    }

    func testTheBackendReturnsAClipAndTheRegistryReachesIt() throws {
        try requireMLXRuntime()
        let (directory, record) = try testDev()
        let generator = try NFKMLXWanFunGenerator.generator(directoryURL: directory)
        let subject = try NFKMLXImageBridge.cgImage(from: record["subject_image"]!, options: NFKMLXImageOptions())
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: prompt, NFKMLXVideoInputKey.referenceImage: subject],
                                          parameters: [NFKParameterFrameCount: 5, NFKParameterWidth: 48,
                                                       NFKParameterHeight: 32, NFKParameterSteps: 2])
        let result = try generator.inferenceBackend().runInference(for: request)
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        XCTAssertEqual(asset.dimensions, CGSize(width: 48, height: 32))
        NFKMLXWanFunGenerator.register()
        XCTAssertNoThrow(try NFKMLXModelRegistry.backend(named: "wan2.2-vace-fun-a14b", weightsURL: directory))
        XCTAssertThrowsError(try NFKMLXModelRegistry.backend(named: "wan2.2-fun-a14b-control", weightsURL: directory))
    }

    // MARK: Training

    func testVACEFunLoRAMatchesDiffSynth() throws {
        try requireMLXRuntime()
        let (directory, _) = try testDev()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN22_VACE_FUN_LORA_DIFFSYNTH"] else {
            throw XCTSkip("set IK_PARITY_WAN22_VACE_FUN_LORA_DIFFSYNTH (run_reference.py wan_vace_fun_lora_diffsynth --checkpoint <testdev>)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let preset = NFKMLXWanFunGenerator.vaceDiffSynthPreset(expert: .lowNoise)
        let draw = preset.objective.sampler.draw(record["index"]!.item(Float.self), weighting: preset.objective.weighting)
        XCTAssertEqual(draw.timestep, record["timestep"]!.item(Float.self))
        XCTAssertEqual(draw.sigma, record["sigma"]!.item(Float.self))
        let net = try NFKMLXWanFunGenerator.network(directoryURL: directory, expert: .lowNoise, precision: .float32)
        let adapted = try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 8) { path, _ in preset.targets.contains(path) }
        let adapters = record.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("lora::") else { return nil }
            let name = String(key.dropFirst("lora::".count)), layer = String(name.split(separator: ".").dropLast().joined(separator: "."))
            return NFKMLXWanLoRA.textToVideoPath(original: layer).map { ($0 + "." + name.split(separator: ".").last!, value) }
        }
        XCTAssertEqual(adapted * 2, adapters.count, "the recipe adapts DiffSynth's VACE targets alone")
        try net.update(parameters: ModuleParameters.unflattened(adapters), verify: .noUnusedKeys)
        let example = NFKMLXWanTrainingExample(latent: record["latent"]!, text: record["text"]!,
                                               vaceContext: record["vace_context"]!)
        let noise = record["noise"]!
        let predictionCosine = cosine(preset.objective.prediction(net, example: example, draw: draw, noise: noise, epsilon: nil),
                                      record["prediction"]!)
        let loss = preset.objective.score(net, example: example, draw: draw, noise: noise, epsilon: nil).item(Float.self)
        let reference = record["loss"]!.item(Float.self)
        XCTAssertEqual(loss, reference, accuracy: 1e-5 * reference)
        net.freeze()
        for (_, layer) in net.leafModules().flattened() where layer is NFKMLXLoRALinear {
            layer.unfreeze(recursive: false, keys: ["lora_a", "lora_b"])
        }
        let gradients = Dictionary(uniqueKeysWithValues: valueAndGrad(model: net) { net, _ in
            [preset.objective.score(net, example: example, draw: draw, noise: noise, epsilon: nil)]
        }(net, []).1.flattened())
        var lowest = 1.0, matched = 0
        for (key, value) in record where key.hasPrefix("grad::") {
            let name = String(key.dropFirst("grad::".count))
            let layer = String(name.split(separator: ".").dropLast().joined(separator: "."))
            guard let path = NFKMLXWanLoRA.textToVideoPath(original: layer),
                  let ours = gradients[path + "." + name.split(separator: ".").last!] else {
                XCTFail("no gradient for \(name)")
                continue
            }
            matched += 1
            lowest = min(lowest, cosine(ours, value))
        }
        XCTAssertEqual(matched, gradients.count, "a gradient for every adapter")
        print("wan vace fun lora diffsynth: prediction \(String(format: "%.16f", predictionCosine)), loss \(loss) vs \(reference), "
              + "lowest gradient cosine \(String(format: "%.12f", lowest))")
        XCTAssertGreaterThan(predictionCosine, 0.999999)
        XCTAssertGreaterThan(lowest, 0.999999)
    }

    func testAVACEFunExpertTrainsSavesAndReloadsThroughItsFactory() throws {
        try requireMLXRuntime()
        let (directory, record) = try testDev()
        let generator = try NFKMLXWanFunGenerator.generator(directoryURL: directory, residency: .resident,
                                                            precision: .float32, quantizationBits: 0)
        let control = record["control_video"]!
        let conditions = NFKMLXWanVACEConditions(controlVideo: control, subjectImages: [record["subject_image"]!])
        let example = try generator.trainingExample(video: record["inpaint_video"]!, prompt: prompt, vace: conditions)
        XCTAssertEqual(example.latent.shape, [16, 4, 4, 6], "the subject image's latent leads the clip's")
        XCTAssertEqual(example.vaceContext?.shape, [96, 4, 4, 6])
        var preset = NFKMLXWanFunGenerator.referencePreset(variant: .vaceA14B, expert: .lowNoise)
        XCTAssertEqual(preset.targets, .vaceAttentionAndFeedForward)
        preset.learningRate = 1e-2
        preset.warmupSteps = 0
        let net = try NFKMLXWanFunGenerator.network(directoryURL: directory, expert: .lowNoise, precision: .float32)
        let losses = try NFKMLXWanFunGenerator.fineTune(net, examples: { _ in [example] }, preset: preset, steps: 3, seed: 7)
        XCTAssertEqual(losses.count, 3)

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wan-vace-lora-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWanFunGenerator.saveAdapters(of: net, to: url, preset: preset)
        let fresh = try NFKMLXWanFunGenerator.network(directoryURL: directory, expert: .lowNoise, precision: .float32)
        let report = try NFKMLXWanLoRA.merge(adapterURL: url, into: fresh)
        XCTAssertEqual(report.mergedLayers, 2 * 10)
        let draw = NFKMLXWanTrainingDraw(sigma: 0.5, timestep: 500, weight: 1)
        let noise = MLXRandom.normal(example.latent.shape, key: MLXRandom.key(9))
        let trained = preset.objective.prediction(net, example: example, draw: draw, noise: noise, epsilon: nil)
        let merged = preset.objective.prediction(fresh, example: example, draw: draw, noise: noise, epsilon: nil)
        XCTAssertLessThan((trained - merged).abs().max().item(Float.self), 1e-4, "the saved adapters merge to the trained network")

        let before = try generatorRun(generator, record)
        try generator.addAdapter(at: url, scale: 1, expert: .lowNoise)
        XCTAssertGreaterThan((try generatorRun(generator, record) - before).abs().max().item(Float.self), 0,
                             "the low-noise adapter changes the clip")
    }
}
