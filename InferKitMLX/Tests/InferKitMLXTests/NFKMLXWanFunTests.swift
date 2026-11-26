//
//  NFKMLXWanFunTests.swift
//  InferKitMLXTests
//
//  The Wan 2.2 Fun releases (alibaba-pai, VideoX-Fun): the original Wan layout's names, the conditions'
//  regrouping and resizes, and, where the validation store holds a test-dev release and its record, the
//  whole sampler against VideoX-Fun's own `Wan2_2FunControlPipeline` / `Wan2_2FunInpaintPipeline`
//  (`run_reference.py wan_fun_pipeline`). Runs where MLX has a Metal library (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXWanFunTests: XCTestCase {

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

    // MARK: Weight-free

    func testOriginalTransformerNamesMapToTheModule() {
        let expected = [
            "blocks.3.self_attn.q.weight": "blocks.3.attn1.to_q.weight",
            "blocks.3.self_attn.o.bias": "blocks.3.attn1.to_out.0.bias",
            "blocks.3.self_attn.norm_k.weight": "blocks.3.attn1.norm_k.weight",
            "blocks.3.cross_attn.v.weight": "blocks.3.attn2.to_v.weight",
            "blocks.3.norm3.bias": "blocks.3.norm2.bias",
            "blocks.3.modulation": "blocks.3.scale_shift_table",
            "blocks.3.ffn.0.weight": "blocks.3.ffn.net.0.proj.weight",
            "blocks.3.ffn.2.bias": "blocks.3.ffn.net.2.bias",
            "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
            "time_projection.1.bias": "condition_embedder.time_proj.bias",
            "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
            "head.head.weight": "proj_out.weight",
            "head.modulation": "scale_shift_table",
            "patch_embedding.weight": "patch_embedding.weight",
            "ref_conv.weight": "ref_conv.weight",
            "control_adapter.residual_blocks.0.conv2.bias": "control_adapter.residual_blocks.0.conv2.bias",
        ]
        for (released, module) in expected {
            XCTAssertEqual(NFKMLXWanOriginalRelease.transformerModuleKey(forRelease: released), module, released)
        }
    }

    func testOriginalAutoencoderNamesMapToTheModule() {
        let wan22 = NFKMLXWanVAEConfiguration.wan22, wan21 = NFKMLXWanVAEConfiguration.wan21
        let expected22 = [
            "conv1.weight": "quant_conv.weight",
            "encoder.conv1.bias": "encoder.conv_in.bias",
            "encoder.downsamples.1.downsamples.0.residual.2.weight": "encoder.down_blocks.1.resnets.0.conv1.weight",
            "encoder.downsamples.1.downsamples.1.shortcut.bias": "encoder.down_blocks.1.resnets.1.conv_shortcut.bias",
            "encoder.downsamples.1.downsamples.2.resample.1.weight": "encoder.down_blocks.1.downsampler.resample.1.weight",
            "encoder.downsamples.2.downsamples.2.time_conv.weight": "encoder.down_blocks.2.downsampler.time_conv.weight",
            "decoder.upsamples.0.upsamples.3.resample.1.bias": "decoder.up_blocks.0.upsampler.resample.1.bias",
            "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
            "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
            "decoder.head.0.gamma": "decoder.norm_out.gamma",
            "decoder.head.2.weight": "decoder.conv_out.weight",
        ]
        for (released, module) in expected22 {
            XCTAssertEqual(NFKMLXWanOriginalRelease.vaeModuleKey(forRelease: released, configuration: wan22), module, released)
        }
        let expected21 = [
            "encoder.downsamples.2.resample.1.weight": "encoder.down_blocks.2.resample.1.weight",
            "encoder.downsamples.0.residual.6.bias": "encoder.down_blocks.0.conv2.bias",
            "decoder.upsamples.2.residual.0.gamma": "decoder.up_blocks.0.resnets.2.norm1.gamma",
            "decoder.upsamples.3.time_conv.weight": "decoder.up_blocks.0.upsamplers.0.time_conv.weight",
            "decoder.upsamples.4.shortcut.weight": "decoder.up_blocks.1.resnets.0.conv_shortcut.weight",
            "decoder.upsamples.11.resample.1.bias": "decoder.up_blocks.2.upsamplers.0.resample.1.bias",
            "decoder.upsamples.14.residual.6.weight": "decoder.up_blocks.3.resnets.2.conv2.weight",
        ]
        for (released, module) in expected21 {
            XCTAssertEqual(NFKMLXWanOriginalRelease.vaeModuleKey(forRelease: released, configuration: wan21), module, released)
        }
    }

    func testOriginalTextEncoderNamesMapToTheModule() {
        let expected = [
            "token_embedding.weight": "shared.weight",
            "norm.weight": "encoder.final_layer_norm.weight",
            "blocks.5.attn.o.weight": "encoder.block.5.layer.0.SelfAttention.o.weight",
            "blocks.5.pos_embedding.embedding.weight": "encoder.block.5.layer.0.SelfAttention.relative_attention_bias.weight",
            "blocks.5.ffn.gate.0.weight": "encoder.block.5.layer.1.DenseReluDense.wi_0.weight",
            "blocks.5.ffn.fc1.weight": "encoder.block.5.layer.1.DenseReluDense.wi_1.weight",
            "blocks.5.norm2.weight": "encoder.block.5.layer.1.layer_norm.weight",
        ]
        for (released, module) in expected {
            XCTAssertEqual(NFKMLXWanOriginalRelease.textEncoderModuleKey(forRelease: released), module, released)
        }
    }

    func testAMaskRegroupsByLatentFrame() throws {
        try requireMLXRuntime()
        // The reference repeats pixel frame 0 four times, then views every four frames as one latent frame.
        let frames = MLXArray((0 ..< 9).map(Float.init)).reshaped([1, 9, 1, 1])
        let grouped = NFKMLXWanFunPipeline.groupedByLatentFrame(frames)
        XCTAssertEqual(grouped.shape, [4, 3, 1, 1])
        XCTAssertEqual(grouped.transposed(1, 0, 2, 3).reshaped([-1]).asArray(Float.self),
                       [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    }

    func testTheLinearResizeFollowsTorch() throws {
        try requireMLXRuntime()
        // torch: F.interpolate(arange(8).view(1, 1, 1, 1, 8), size=(1, 1, 2), mode="trilinear").
        let row = MLXArray((0 ..< 8).map(Float.init)).reshaped([1, 1, 1, 8])
        XCTAssertEqual(NFKMLXWanFunPipeline.linearResized(row, size: [1, 1, 2], alignCorners: false).reshaped([-1])
            .asArray(Float.self), [1.5, 5.5])
        XCTAssertEqual(NFKMLXWanFunPipeline.linearResized(row, size: [1, 1, 3], alignCorners: true).reshaped([-1])
            .asArray(Float.self), [0, 3.5, 7])
    }

    // MARK: The released transformers, held by shape

    /// Every released Fun transformer's header (`Tools/validation-assets/shapes.py`) against the module its
    /// `config.json` builds, the released names mapped as the loader maps them: none missing, none left
    /// over, none at another shape.
    func testEveryFunReleaseMatchesItsLoader() throws {
        try requireMLXRuntime()
        guard let root = NFKMLXValidationConfig.environment["IK_SHAPES_ROOT"] else { throw XCTSkip("set IK_SHAPES_ROOT") }
        let names = ["wan22-fun-5b-control", "wan22-fun-5b-inp", "wan22-fun-5b-control-camera"]
            + ["control", "inp", "control-camera"].flatMap { ["wan22-fun-a14b-\($0)-high", "wan22-fun-a14b-\($0)-low"] }
        for name in names {
            let directory = URL(fileURLWithPath: root).appendingPathComponent(name)
            guard let data = try? Data(contentsOf: directory.appendingPathComponent("shapes.json")) else {
                throw XCTSkip("fetch \(name) with Tools/validation-assets/shapes.py")
            }
            let released = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: [Int]])
            let configuration = try NFKMLXWanOriginalRelease.transformerConfiguration(
                fromConfigURL: directory.appendingPathComponent("config.json"))
            let net = NFKMLXWanTransformerNet(configuration)
            let built = Dictionary(net.parameters().flattened().map { ($0.0, $0.1.shape) }, uniquingKeysWith: { a, _ in a })
            let mapped = Dictionary(released.map { key, shape -> (String, [Int]) in
                let name = NFKMLXWanOriginalRelease.transformerModuleKey(forRelease: key)
                return (name, shape.count >= 4 ? [shape[0]] + shape.dropFirst(2) + [shape[1]] : shape)
            }, uniquingKeysWith: { a, _ in a })
            let missing = built.keys.filter { mapped[$0] == nil }.sorted()
            let unconsumed = mapped.keys.filter { built[$0] == nil }.sorted()
            let mismatched = built.compactMap { key, shape in mapped[key].flatMap { $0 == shape ? nil : "\(key) \(shape) vs \($0)" } }
            print("VALIDATION structure \(name): \(built.count) built, \(released.count) released, \(missing.count) missing, "
                  + "\(unconsumed.count) unconsumed, \(mismatched.count) mismatched")
            XCTAssertTrue(missing.isEmpty, "\(name) lacks \(missing.prefix(5))")
            XCTAssertTrue(unconsumed.isEmpty, "\(name) ships \(unconsumed.prefix(5)) nothing reads")
            XCTAssertTrue(mismatched.isEmpty, "\(name): \(mismatched.prefix(5))")
        }
    }

    // MARK: Against VideoX-Fun

    private func release(_ key: String, record recordKey: String) throws -> (URL, [String: MLXArray]) {
        guard let directory = NFKMLXValidationConfig.environment[key] else {
            throw XCTSkip("set \(key) (Tools/wan-fun-testdev/make_testdev.py)")
        }
        guard let record = NFKMLXValidationConfig.environment[recordKey] else {
            throw XCTSkip("set \(recordKey) (run_reference.py wan_fun_pipeline --checkpoint <\(key)>)")
        }
        return (URL(fileURLWithPath: directory), try MLX.loadArrays(url: URL(fileURLWithPath: record)))
    }

    private let prompt = "A red fox  walking through fresh snow, cinematic"
    private let negative = "worst quality, blurry"

    /// Every recorded seam of one release against the sampler's own, the measured cosines printed.
    private func assertMatchesVideoXFun(testDevKey: String, recordKey: String) throws {
        let (directory, record) = try release(testDevKey, record: recordKey)
        let pipeline = try NFKMLXWanFunRelease.pipeline(directoryURL: directory)
        let encoder = try NFKMLXWanOriginalRelease.textEncoder(
            from: directory.appendingPathComponent(NFKMLXWanOriginalRelease.textEncoderFile))
        let segmenter = try NFKMLXWanFunRelease.segmenter(directory)
        for (name, text) in [("prompt", prompt), ("negative", negative)] {
            let ids = NFKMLXWanFunRelease.prompt(text, segmenter: segmenter).ids
            XCTAssertEqual(ids.map(Int32.init), record["\(name)_ids"]!.asArray(Int32.self), "\(name) token ids")
        }
        let text = NFKMLXWanFunRelease.features(prompt, encoder: encoder, segmenter: segmenter)
        let negativeText = NFKMLXWanFunRelease.features(negative, encoder: encoder, segmenter: segmenter)
        let length = record["prompt_embeds"]!.dim(0), negativeLength = record["negative_embeds"]!.dim(0)
        var measured = [String: Double]()
        measured["prompt features"] = cosine(text[0 ..< length], record["prompt_embeds"]!)
        measured["negative features"] = cosine(negativeText[0 ..< negativeLength], record["negative_embeds"]!)

        let kind = record["kind"]!.asArray(Int32.self)
        let inpaint = kind[1] == 1, camera = kind[2] == 1, reference = kind[3] == 1
        let frames = record["control_video"]!.dim(0), height = record["control_video"]!.dim(1),
            width = record["control_video"]!.dim(2)
        let start = NFKMLXWanFunConditions.startImage(record["start_image"]!, endImage: record["end_image"]!, frames: frames)
        var cameraEmbedding = record["camera"]
        if camera, let bytes = record["pose_file"] {
            let text = String(decoding: bytes.asArray(UInt8.self), as: UTF8.self)
            let ours = try NFKMLXWanCameraTrajectory(poseFileContents: text).plucker(width: width, height: height, frames: frames)
            measured["camera embedding"] = cosine(ours, record["camera"]!)
            XCTAssertLessThan((ours - record["camera"]!).abs().max().item(Float.self), 1e-5, "camera embedding")
            cameraEmbedding = ours
        }
        for prefix in ["", "start.", "mask."] {
            var conditions = NFKMLXWanFunConditions()
            if !inpaint {
                if camera {
                    conditions.cameraEmbedding = cameraEmbedding
                } else {
                    conditions.controlVideo = record["control_video"]
                }
                if reference {
                    conditions.referenceImage = record["reference_image"]
                }
            }
            if prefix == "start." {
                conditions.inpaintVideo = start.video
                conditions.inpaintMask = start.mask
            } else if prefix == "mask." {
                conditions.inpaintVideo = record["control_video"]![.stride(by: -1)]
                conditions.inpaintMask = record["region_mask"]
            }
            let prepared = try pipeline.prepare(conditions, frames: frames, height: height, width: width)
            if let y = record[prefix + "input.y"] {
                if y[1].abs().max().item(Float.self) == 0 {
                    XCTAssertEqual(prepared.channels.abs().max().item(Float.self), 0, "\(prefix)condition channels")
                } else {
                    measured["\(prefix)condition channels"] = cosine(prepared.channels, y[1])
                }
            }
            if let fullReference = record[prefix + "input.full_ref"], let ours = prepared.reference {
                measured["\(prefix)reference latent"] = cosine(ours, fullReference[1])
            }
            var steps = [MLXArray]()
            let final = try pipeline.denoise(text: text, negative: negativeText, prepared: prepared,
                                             noise: record["latents"]!, steps: 3, guidance: 4, onStep: { steps.append($1) })
            for (index, latents) in steps.enumerated() {
                measured["\(prefix)step \(index)"] = cosine(latents, record["\(prefix)step.\(index)"]!)
            }
            if pipeline.highNoiseTransformer != nil {
                var scheduler = NFKMLXFlowMatchScheduler(pipeline.schedule)
                scheduler.setTimesteps(3, sequenceLength: 0)
                let ours = scheduler.timesteps.map { Int32($0 >= pipeline.boundary * 1000 ? 1 : 0) }
                XCTAssertEqual(ours, record[prefix + "experts"]!.asArray(Int32.self), "\(prefix)experts")
            }
            measured["\(prefix)final latents"] = cosine(final, record[prefix + "final_latents"]!)
            measured["\(prefix)clip"] = cosine(pipeline.decode(final), record[prefix + "video"]!)
        }
        for (name, value) in measured.sorted(by: { $0.key < $1.key }) {
            print("wan fun \(testDevKey) \(name): \(String(format: "%.16f", value))")
            XCTAssertGreaterThan(value, 0.9999999, name)
        }
    }

    func testFun5BControlMatchesVideoXFun() throws {
        try requireMLXRuntime()
        try assertMatchesVideoXFun(testDevKey: "IK_VAL_WAN22_FUN_5B_CONTROL_TESTDEV",
                                   recordKey: "IK_PARITY_WAN22_FUN_5B_CONTROL_PIPELINE")
    }

    func testFun5BInPMatchesVideoXFun() throws {
        try requireMLXRuntime()
        try assertMatchesVideoXFun(testDevKey: "IK_VAL_WAN22_FUN_5B_INP_TESTDEV",
                                   recordKey: "IK_PARITY_WAN22_FUN_5B_INP_PIPELINE")
    }

    func testFun5BControlCameraMatchesVideoXFun() throws {
        try requireMLXRuntime()
        try assertMatchesVideoXFun(testDevKey: "IK_VAL_WAN22_FUN_5B_CAMERA_TESTDEV",
                                   recordKey: "IK_PARITY_WAN22_FUN_5B_CAMERA_PIPELINE")
    }

    func testFunA14BControlMatchesVideoXFun() throws {
        try requireMLXRuntime()
        try assertMatchesVideoXFun(testDevKey: "IK_VAL_WAN22_FUN_A14B_CONTROL_TESTDEV",
                                   recordKey: "IK_PARITY_WAN22_FUN_A14B_CONTROL_PIPELINE")
    }

    func testFunA14BInPMatchesVideoXFun() throws {
        try requireMLXRuntime()
        try assertMatchesVideoXFun(testDevKey: "IK_VAL_WAN22_FUN_A14B_INP_TESTDEV",
                                   recordKey: "IK_PARITY_WAN22_FUN_A14B_INP_PIPELINE")
    }

    func testFunA14BControlCameraMatchesVideoXFun() throws {
        try requireMLXRuntime()
        try assertMatchesVideoXFun(testDevKey: "IK_VAL_WAN22_FUN_A14B_CAMERA_TESTDEV",
                                   recordKey: "IK_PARITY_WAN22_FUN_A14B_CAMERA_PIPELINE")
    }

    // MARK: The generator

    /// The generator over a test-dev release, its run from the record's noise and conditions against the
    /// record's final latents.
    private func generatorRun(_ generator: NFKMLXWanFunGenerator, _ record: [String: MLXArray]) throws -> MLXArray {
        generator.steps = 3
        generator.guidance = 4
        let control = record["control_video"]!
        let frames = control.dim(0), height = control.dim(1), width = control.dim(2)
        var conditions = NFKMLXWanFunConditions()
        if generator.variant.rawValue % 3 == 0 {
            conditions.controlVideo = control
            conditions.referenceImage = record["reference_image"]
        }
        if generator.variant.rawValue % 3 == 2 {
            conditions.cameraEmbedding = record["camera"]
        }
        return try generator.generate(prompt: prompt, negativePrompt: negative, conditions: conditions, frames: frames,
                                      width: width, height: height, seed: 0, noise: record["latents"]!).latent
    }

    func testTheGeneratorMatchesVideoXFunThroughEveryResidency() throws {
        try requireMLXRuntime()
        for (key, recordKey) in [("IK_VAL_WAN22_FUN_5B_CONTROL_TESTDEV", "IK_PARITY_WAN22_FUN_5B_CONTROL_PIPELINE"),
                                 ("IK_VAL_WAN22_FUN_A14B_CONTROL_TESTDEV", "IK_PARITY_WAN22_FUN_A14B_CONTROL_PIPELINE")] {
            let (directory, record) = try release(key, record: recordKey)
            let resident = try NFKMLXWanFunGenerator.generator(directoryURL: directory, residency: .resident,
                                                               precision: .float32, quantizationBits: 0)
            XCTAssertTrue(resident.holdsStagesResident)
            let held = try generatorRun(resident, record)
            let measured = cosine(held, record["final_latents"]!)
            print("wan fun generator \(key): \(String(format: "%.16f", measured))")
            XCTAssertGreaterThan(measured, 0.9999999, key)

            let staged = try NFKMLXWanFunGenerator.generator(directoryURL: directory, residency: .staged,
                                                             precision: .float32, quantizationBits: 0)
            XCTAssertFalse(staged.holdsStagesResident)
            XCTAssertEqual((try generatorRun(staged, record) - held).abs().max().item(Float.self), 0, "\(key) staged")

            let streamed = try NFKMLXWanFunGenerator.generator(directoryURL: directory, residency: .resident,
                                                              precision: .float32, bits: nil, streamedBlocks: [1])
            XCTAssertGreaterThan(streamed.streamedLayerCount, 0)
            XCTAssertEqual((try generatorRun(streamed, record) - held).abs().max().item(Float.self), 0, "\(key) streamed")

            let packed = try NFKMLXWanFunGenerator.generator(directoryURL: directory, residency: .resident,
                                                            precision: .float32, quantizationBits: 8)
            XCTAssertEqual(packed.quantizationBits, 8)
            XCTAssertGreaterThan(cosine(try generatorRun(packed, record), held), 0.999, "\(key) 8-bit")
        }
    }

    func testTheBackendReturnsAClipAndTheRegistryRefusesAnotherVariant() throws {
        try requireMLXRuntime()
        let (directory, record) = try release("IK_VAL_WAN22_FUN_5B_CONTROL_TESTDEV",
                                              record: "IK_PARITY_WAN22_FUN_5B_CONTROL_PIPELINE")
        let generator = try NFKMLXWanFunGenerator.generator(directoryURL: directory)
        XCTAssertEqual(generator.variant, .control5B)
        let reference = try NFKMLXImageBridge.cgImage(from: record["reference_image"]!, options: NFKMLXImageOptions())
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: prompt, NFKMLXVideoInputKey.referenceImage: reference],
                                          parameters: [NFKParameterFrameCount: 5, NFKParameterWidth: 64,
                                                       NFKParameterHeight: 64, NFKParameterSteps: 2])
        let result = try generator.inferenceBackend().runInference(for: request)
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        XCTAssertEqual(asset.dimensions, CGSize(width: 64, height: 64))
        NFKMLXWanFunGenerator.register()
        XCTAssertThrowsError(try NFKMLXModelRegistry.backend(named: "wan2.2-fun-5b-inp", weightsURL: directory))
        XCTAssertNoThrow(try NFKMLXModelRegistry.backend(named: "wan2.2-fun-5b-control", weightsURL: directory))
    }

    // MARK: Training

    /// The network a DiffSynth Fun record's case describes, its adapters (rank 4, alpha 8) at the recorded values.
    private func diffSynthNetwork(_ record: [String: MLXArray], case name: String) throws -> NFKMLXWanTransformerNet {
        let section = { (prefix: String) in
            Dictionary(uniqueKeysWithValues: record.compactMap { key, value in
                key.hasPrefix("\(name).\(prefix)") ? (String(key.dropFirst(name.count + 1 + prefix.count)), value) : nil
            })
        }
        let weights = section("w::")
        let inChannels = weights["patch_embedding.weight"]!.dim(1)
        let configuration = NFKMLXWanConfiguration(
            inChannels: inChannels, heads: 2, headDim: 24, layers: 2, ffnDim: 48, textDim: 32, patchSize: [1, 2, 2],
            outChannels: 16, referenceChannels: weights["ref_conv.weight"] == nil ? nil : 16,
            controlAdapterChannels: weights["control_adapter.conv.weight"] == nil ? nil : 24,
            formsSinusoidsInFloat64: true)
        let net = NFKMLXWanTransformerNet(configuration)
        let adapted = try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 8) { path, _ in
            NFKMLXWanLoRATargets.attentionAndFeedForward.contains(path)
        }
        let adapters = section("lora::")
        XCTAssertEqual(adapted * 2, adapters.count, "the oracle adapts DiffSynth's targets")
        let base = weights.map { key, value -> (String, MLXArray) in
            NFKMLXWanOriginalRelease.adaptedTransformer(key: key, value: value)
        }
        try NFKMLXWeights.apply(base + adapters.map { ($0.key, $0.value) }, to: net, verifyShapes: true)
        return net
    }

    func testFunA14BLoRAMatchesDiffSynth() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN22_FUN_LORA_DIFFSYNTH"] else {
            throw XCTSkip("set IK_PARITY_WAN22_FUN_LORA_DIFFSYNTH (run_reference.py wan_fun_lora_diffsynth)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let preset = NFKMLXWanFunGenerator.diffSynthPreset(expert: .lowNoise)
        for name in ["control", "camera"] {
            let draw = preset.objective.sampler.draw(record["\(name).index"]!.item(Float.self), weighting: preset.objective.weighting)
            XCTAssertEqual(draw.timestep, record["\(name).timestep"]!.item(Float.self), name)
            XCTAssertEqual(draw.sigma, record["\(name).sigma"]!.item(Float.self), name)
            let net = try diffSynthNetwork(record, case: name)
            let example = NFKMLXWanTrainingExample(latent: record["\(name).latent"]!, text: record["\(name).text"]!,
                                                   condition: record["\(name).condition"], reference: record["\(name).reference"],
                                                   camera: record["\(name).camera"])
            let noise = record["\(name).noise"]!
            let noisy = (1 - draw.sigma) * example.latent + draw.sigma * noise
            let prediction = net(concatenated([noisy, example.condition!], axis: 0), text: example.text,
                                 t: Double(draw.timestep),
                                 conditioning: NFKWanConditioning(reference: example.reference, cameraControl: example.camera))
            let predictionCosine = cosine(prediction, record["\(name).prediction"]!)
            let loss = preset.objective.score(net, example: example, draw: draw, noise: noise, epsilon: nil).item(Float.self)
            let reference = record["\(name).loss"]!.item(Float.self)
            XCTAssertEqual(loss, reference, accuracy: 1e-5 * reference, name)
            net.freeze()
            for (_, layer) in net.leafModules().flattened() where layer is NFKMLXLoRALinear {
                layer.unfreeze(recursive: false, keys: ["lora_a", "lora_b"])
            }
            let gradients = Dictionary(uniqueKeysWithValues: valueAndGrad(model: net) { net, _ in
                [preset.objective.score(net, example: example, draw: draw, noise: noise, epsilon: nil)]
            }(net, []).1.flattened())
            let recorded = Dictionary(uniqueKeysWithValues: record.compactMap { key, value in
                key.hasPrefix("\(name).grad::") ? (String(key.dropFirst(name.count + 7)), value) : nil
            })
            XCTAssertEqual(Set(gradients.keys), Set(recorded.keys), "\(name): a gradient for every adapter")
            let lowest = recorded.map { key, value in gradients[key].map { cosine($0, value) } ?? 0 }.min() ?? 0
            print("wan fun lora diffsynth \(name): prediction \(String(format: "%.16f", predictionCosine)), loss \(loss) "
                  + "vs \(reference), lowest gradient cosine \(String(format: "%.10f", lowest))")
            XCTAssertGreaterThan(predictionCosine, 0.999999, name)
            XCTAssertGreaterThan(lowest, 0.999999, name)
        }
    }

    func testAFunExpertTrainsSavesAndReloadsThroughItsFactory() throws {
        try requireMLXRuntime()
        let (directory, record) = try release("IK_VAL_WAN22_FUN_A14B_CONTROL_TESTDEV",
                                              record: "IK_PARITY_WAN22_FUN_A14B_CONTROL_PIPELINE")
        let generator = try NFKMLXWanFunGenerator.generator(directoryURL: directory, residency: .resident,
                                                            precision: .float32, quantizationBits: 0)
        let control = record["control_video"]!
        let video = control[.stride(by: -1)]
        let example = try generator.trainingExample(
            video: video, prompt: prompt,
            conditions: NFKMLXWanFunConditions(controlVideo: control, referenceImage: record["reference_image"]))
        XCTAssertEqual(example.condition?.dim(0), 16 + 4 + 16)
        XCTAssertEqual(example.reference?.shape, [16, 4, 6])
        var preset = NFKMLXWanFunGenerator.diffSynthPreset(expert: .lowNoise)
        preset.learningRate = 1e-2
        preset.warmupSteps = 0
        let net = try NFKMLXWanFunGenerator.network(directoryURL: directory, expert: .lowNoise)
        let losses = try NFKMLXWanFunGenerator.fineTune(net, examples: { _ in [example] }, preset: preset, steps: 3, seed: 7)
        XCTAssertEqual(losses.count, 3)

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wan-fun-lora-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWanFunGenerator.saveAdapters(of: net, to: url, preset: preset)
        let fresh = try NFKMLXWanFunGenerator.network(directoryURL: directory, expert: .lowNoise)
        let report = try NFKMLXWanLoRA.merge(adapterURL: url, into: fresh)
        XCTAssertEqual(report.mergedLayers, 2 * 10)
        let input = concatenated([example.latent, example.condition!], axis: 0)
        let conditioning = NFKWanConditioning(reference: example.reference)
        let trained = net(input, text: example.text, t: 500, conditioning: conditioning)
        let merged = fresh(input, text: example.text, t: 500, conditioning: conditioning)
        XCTAssertLessThan((trained - merged).abs().max().item(Float.self), 1e-3, "the saved adapters merge to the trained network")

        let before = try generatorRun(generator, record)
        try generator.addAdapter(at: url, scale: 1, expert: .lowNoise)
        XCTAssertEqual(generator.adapterCount, 1)
        XCTAssertGreaterThan((try generatorRun(generator, record) - before).abs().max().item(Float.self), 0,
                             "the low-noise adapter changes the clip")
    }

    func testA5BReleaseTrainsUnderVideoXFunsRecipeAndReloads() throws {
        try requireMLXRuntime()
        let (directory, record) = try release("IK_VAL_WAN22_FUN_5B_CONTROL_TESTDEV",
                                              record: "IK_PARITY_WAN22_FUN_5B_CONTROL_PIPELINE")
        let generator = try NFKMLXWanFunGenerator.generator(directoryURL: directory, residency: .resident,
                                                            precision: .float32, quantizationBits: 0)
        let control = record["control_video"]!
        let start = NFKMLXWanFunConditions.startImage(record["start_image"]!, frames: control.dim(0))
        let example = try generator.trainingExample(
            video: control[.stride(by: -1)], prompt: prompt,
            conditions: NFKMLXWanFunConditions(controlVideo: control, inpaintVideo: start.video, inpaintMask: start.mask,
                                               referenceImage: record["reference_image"]))
        XCTAssertNotNil(example.latentMask, "a 5B release's example carries its latent mask")
        var preset = NFKMLXWanFunGenerator.referencePreset(variant: .control5B, expert: .highNoise)
        XCTAssertEqual(preset.format, .musubi)
        preset.learningRate = 1e-2
        preset.clipGradientNorm = nil
        let net = try NFKMLXWanFunGenerator.network(directoryURL: directory, precision: .float32)
        XCTAssertEqual(try NFKMLXWanFunGenerator.fineTune(net, examples: { _ in [example] }, preset: preset, steps: 2,
                                                          seed: 3).count, 2)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wan-fun-5b-lora-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWanFunGenerator.saveAdapters(of: net, to: url, preset: preset)
        let fresh = try NFKMLXWanFunGenerator.network(directoryURL: directory, precision: .float32)
        XCTAssertEqual(try NFKMLXWanLoRA.merge(adapterURL: url, into: fresh).mergedLayers, 2 * 8)
        let draw = preset.objective.sampler.draw(500)
        let noise = MLXArray.zeros(example.latent.shape)
        let trained = preset.objective.prediction(net, example: example, draw: draw, noise: noise, epsilon: nil)
        let merged = preset.objective.prediction(fresh, example: example, draw: draw, noise: noise, epsilon: nil)
        XCTAssertLessThan((trained - merged).abs().max().item(Float.self), 1e-4, "the saved adapters merge to the trained network")
        try generator.addAdapter(at: url, scale: 1, expert: .highNoise)
        XCTAssertEqual(generator.adapterCount, 1)
    }

    func testFunLoRAMatchesVideoXFun() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN22_FUN_LORA_VIDEOXFUN"] else {
            throw XCTSkip("set IK_PARITY_WAN22_FUN_LORA_VIDEOXFUN (run_reference.py wan_fun_lora_videoxfun)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let table = NFKMLXWanSigmaSampler.videoXFunTable(shift: 5)
        XCTAssertEqual(table.timesteps, record["sampler.timestep"]!.asArray(Float.self), "the scheduler's table")
        XCTAssertEqual(table.sigmas, record["sampler.sigma"]!.asArray(Float.self))
        for (name, expert, heads, width) in [("5b", NFKMLXWanExpert?.none, 24, 3072), ("a14b", .lowNoise, 2, 48)] {
            let preset = NFKMLXWanTrainingPreset.videoXFun(expert: expert)
            guard case let .videoXFun(_, low, high) = preset.objective.sampler else { return XCTFail("\(name) sampler") }
            XCTAssertEqual([low, high], record["\(name).range"]!.asArray(Int32.self).map(Int.init), name)
            let draw = preset.objective.sampler.draw(record["\(name).index"]!.item(Float.self))
            XCTAssertEqual(draw.timestep, record["\(name).timestep"]!.item(Float.self), name)
            XCTAssertEqual(draw.sigma, record["\(name).sigma"]!.item(Float.self), name)
            let weights = Dictionary(uniqueKeysWithValues: record.compactMap { key, value in
                key.hasPrefix("\(name).w::") ? (String(key.dropFirst(name.count + 4)), value) : nil
            })
            let latent = record["\(name).latent"]!
            let configuration = NFKMLXWanConfiguration(
                inChannels: weights["patch_embedding.weight"]!.dim(1), heads: heads, headDim: width / heads, layers: 2,
                ffnDim: name == "5b" ? 96 : 48, textDim: 32, patchSize: [1, 2, 2], outChannels: latent.dim(0),
                referenceChannels: latent.dim(0), formsSinusoidsInFloat64: true)
            let net = NFKMLXWanTransformerNet(configuration)
            let adapted = try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 8) { path, _ in preset.targets.contains(path) }
            let adapters = record.filter { $0.key.hasPrefix("\(name).lora::") }
            XCTAssertEqual(adapted * 2, adapters.count, "\(name): the oracle adapts the recipe's targets")
            try NFKMLXWeights.apply(weights.map { NFKMLXWanOriginalRelease.adaptedTransformer(key: $0.key, value: $0.value) }
                                    + adapters.map { (String($0.key.dropFirst(name.count + 7)), $0.value) },
                                    to: net, verifyShapes: true)
            // VideoX-Fun's forward pads the prompt to its 512 tokens with zeros, as the generator's features are padded.
            let prompt = record["\(name).text"]!
            let text = concatenated([prompt, MLXArray.zeros([512 - prompt.dim(0), prompt.dim(1)])], axis: 0)
            let example = NFKMLXWanTrainingExample(latent: latent, text: text,
                                                   condition: record["\(name).condition"], reference: record["\(name).reference"],
                                                   latentMask: record["\(name).latent_mask"])
            let noise = record["\(name).noise"]!
            let predictionCosine = cosine(preset.objective.prediction(net, example: example, draw: draw, noise: noise, epsilon: nil),
                                          record["\(name).prediction"]!)
            let loss = preset.objective.score(net, example: example, draw: draw, noise: noise, epsilon: nil).item(Float.self)
            let reference = record["\(name).loss"]!.item(Float.self)
            XCTAssertEqual(loss, reference, accuracy: 1e-5 * reference, name)
            net.freeze()
            for (_, layer) in net.leafModules().flattened() where layer is NFKMLXLoRALinear {
                layer.unfreeze(recursive: false, keys: ["lora_a", "lora_b"])
            }
            let gradients = Dictionary(uniqueKeysWithValues: valueAndGrad(model: net) { net, _ in
                [preset.objective.score(net, example: example, draw: draw, noise: noise, epsilon: nil)]
            }(net, []).1.flattened())
            let recorded = Dictionary(uniqueKeysWithValues: record.compactMap { key, value in
                key.hasPrefix("\(name).grad::") ? (String(key.dropFirst(name.count + 7)), value) : nil
            })
            XCTAssertEqual(Set(gradients.keys), Set(recorded.keys), "\(name): a gradient for every adapter")
            let lowest = recorded.map { key, value in gradients[key].map { cosine($0, value) } ?? 0 }.min() ?? 0
            for (key, value) in recorded.sorted(by: { $0.key < $1.key }) {
                if let ours = gradients[key], cosine(ours, value) < 0.9999999 {
                    print("wan fun lora videoxfun \(name) gradient \(key): cosine \(cosine(ours, value)), norm "
                          + "\(sqrt((value * value).sum()).item(Float.self)), largest \(value.abs().max().item(Float.self))")
                }
            }
            print("wan fun lora videoxfun \(name): prediction \(String(format: "%.16f", predictionCosine)), loss \(loss) "
                  + "vs \(reference), lowest gradient cosine \(String(format: "%.12f", lowest))")
            XCTAssertGreaterThan(predictionCosine, 0.999999, name)
            XCTAssertGreaterThan(lowest, 0.999999, name)
        }
    }
}
