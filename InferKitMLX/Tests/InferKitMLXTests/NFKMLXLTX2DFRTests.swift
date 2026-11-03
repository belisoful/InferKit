//
//  NFKMLXLTX2DFRTests.swift
//  InferKitMLXTests
//
//  LTX-2.5's Diffusion Fidelity Rendering recipe against diffusers at tiny random configurations of every
//  stage (`Tools/reference-parity/run_reference.py ltx2_dfr`): the layout, the DFR passes, the IC-LoRA, the
//  temporal refine round, and the third spatial stage, each from the reference's own inputs and then chained,
//  with every noise the reference drew. Each test skips without its key.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLTX2DFRTests: XCTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    private func record() throws -> [String: MLXArray] {
        guard let path = config["IK_PARITY_LTX2_DFR"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_LTX2_DFR (run_reference.py ltx2_dfr)")
        }
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Double {
        let a = mine.asType(.float32).reshaped([-1]).asArray(Float.self)
        let b = reference.asType(.float32).reshaped([-1]).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0
        var i = 0
        while i < min(a.count, b.count) {
            let x = Double(a[i]), y = Double(b[i])
            dot += x * y; na += x * x; nb += y * y
            i += 1
        }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    @discardableResult
    private func compare(_ label: String, _ mine: MLXArray, _ reference: MLXArray?,
                         file: StaticString = #filePath, line: UInt = #line) -> Double {
        guard let reference else {
            XCTFail("the record has no \(label)", file: file, line: line)
            return 0
        }
        eval(mine)
        XCTAssertEqual(mine.shape, reference.shape, "\(label) shape", file: file, line: line)
        let similarity = cosine(mine, reference)
        print("SEAM \(label): cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99999, "\(label) matches the reference", file: file, line: line)
        return similarity
    }

    private func weights(_ arrays: [String: MLXArray], prefix: String) -> [(String, MLXArray)] {
        arrays.compactMap { key, value in
            key.hasPrefix("w::\(prefix).") ? (String(key.dropFirst(3 + prefix.count + 1)), value) : nil
        }
    }

    /// NDHWC → NCDHW and back.
    private func channelsFirst(_ x: MLXArray) -> MLXArray { x.transposed(0, 4, 1, 2, 3) }
    private func channelsLast(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 4, 1) }

    /// Raw packed audio `[1, L, C·M]` ↔ the reference's public `[1, C, L, M]`.
    private func publicAudio(_ packed: MLXArray) -> MLXArray {
        packed.reshaped([1, packed.dim(1), 2, 4]).transposed(0, 2, 1, 3)
    }
    private func packedAudio(_ unpacked: MLXArray) -> MLXArray {
        unpacked.transposed(0, 2, 1, 3).reshaped([1, unpacked.dim(2), -1])
    }

    /// The record's noise draws in order, each tagged with the module that drew it.
    private struct Draws {
        var queue: [(tag: String, value: MLXArray)]
        init(_ arrays: [String: MLXArray]) {
            queue = arrays.keys.filter { $0.hasPrefix("draw") && $0 != "draws" }.sorted().map { key in
                (String(key.split(separator: ".")[1]), arrays[key]!)
            }
        }
        mutating func next(_ tag: String, shape: [Int]) -> MLXArray {
            guard let index = queue.firstIndex(where: { $0.tag == tag }) else {
                XCTFail("no \(tag) draw left")
                return MLXArray.zeros(shape)
            }
            let value = queue.remove(at: index).value
            XCTAssertEqual(value.shape, shape, "\(tag) draw shape")
            return value
        }
    }

    private struct Stages {
        let pipeline: NFKMLXLTX2Pipeline
        let text: MLXArray, audioText: MLXArray
        let spatial: NFKMLXLTX2LatentUpsamplerNet, temporal: NFKMLXLTX2LatentUpsamplerNet
        let encoder: NFKMLXLTX2VideoEncoderNet
        let schedule: NFKMLXFlowMatchConfiguration
    }

    private func stages(_ arrays: [String: MLXArray]) throws -> Stages {
        let gemma = NFKMLXGemma3Net(try NFKMLXGemma3Language.configuration(fromJSON: [
            "model_type": "gemma3_text", "vocab_size": 64, "hidden_size": 12, "intermediate_size": 24,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
            "query_pre_attn_scalar": 8, "sliding_window": 4, "layer_types": ["sliding_attention", "full_attention"],
            "rope_theta": 10000, "rope_local_base_freq": 10000, "rms_norm_eps": 1e-6]))
        try NFKMLXWeights.apply(weights(arrays, prefix: "text").compactMap { key, value in
            NFKMLXGemma3Language.decoderName(of: key).map { ($0, value) }
        }, to: gemma)
        var connectorConfiguration = NFKMLXLTX2ConnectorConfiguration.tiny
        connectorConfiguration.videoHeadDim = 16
        connectorConfiguration.videoWidth = 32
        connectorConfiguration.audioHeadDim = 8
        connectorConfiguration.audioWidth = 16
        let connectors = NFKMLXLTX2ConnectorsNet(connectorConfiguration)
        try NFKMLXWeights.apply(weights(arrays, prefix: "connectors"), to: connectors)
        let ids = try XCTUnwrap(arrays["ids"]).asArray(Int32.self)
        let mask = try XCTUnwrap(arrays["mask"]).asArray(Int32.self)
        let valid = (0 ..< 8).filter { mask[8 + $0] != 0 }
        let (text, audioText) = connectors(
            validStates: NFKMLXLTX2Pipeline.textStates(gemma, tokens: valid.map { Int(ids[8 + $0]) }), length: 8)

        var transformerConfiguration = NFKMLXLTX2Configuration.tiny
        transformerConfiguration.audioInChannels = 8
        transformerConfiguration.audioOutChannels = 8
        transformerConfiguration.feedForwardBias = false
        transformerConfiguration.usesKeyframeEmbedding = true
        let transformer = NFKMLXLTX2TransformerNet(transformerConfiguration)
        try NFKMLXWeights.apply(weights(arrays, prefix: "transformer"), to: transformer)
        let videoDecoder = NFKMLXLTX2VideoDecoderNet(.tiny)
        let vae = weights(arrays, prefix: "vae").map { key, value in
            (key, value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value)
        }
        let videoStatistics = try NFKMLXLTX2VideoVAE.apply(vae, to: videoDecoder)
        let audioDecoder = NFKMLXLTX2AudioDecoderNet(.tiny)
        let audioStatistics = try NFKMLXLTX2AudioVAE.apply(weights(arrays, prefix: "audio_vae").map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }, to: audioDecoder)
        var vocoderConfiguration = NFKMLXLTX2VocoderConfiguration.tiny
        vocoderConfiguration.vocoder.inChannels = 32
        let vocoder = NFKMLXLTX2VocoderNet(vocoderConfiguration)
        try NFKMLXWeights.apply(weights(arrays, prefix: "vocoder").compactMap(NFKMLXLTX2Vocoder.mapped), to: vocoder)
        let pipeline = NFKMLXLTX2Pipeline(transformer: transformer, videoDecoder: videoDecoder,
                                          videoStatistics: videoStatistics, audioDecoder: audioDecoder,
                                          audioStatistics: audioStatistics, vocoder: vocoder)

        func upsampler(_ prefix: String, _ configuration: NFKMLXLTX2LatentUpsamplerConfiguration) throws
            -> NFKMLXLTX2LatentUpsamplerNet {
            let net = NFKMLXLTX2LatentUpsamplerNet(configuration)
            try NFKMLXLTX2LatentUpsamplers.apply(
                weights(arrays, prefix: prefix).map { ($0, NFKMLXLTX2LatentUpsamplers.laid($1)) }, to: net)
            return net
        }
        let spatial = try upsampler("upsampler", .init(inChannels: 8, midChannels: 32, blocksPerStage: 2))
        let temporal = try upsampler("temporal", .init(inChannels: 8, midChannels: 32, blocksPerStage: 2,
                                                      upsamplesSpace: false, upsamplesTime: true,
                                                      usesRationalResampler: true))

        let encoderURL = FileManager.default.temporaryDirectory.appendingPathComponent("ltx2-dfr-vae-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: encoderURL) }
        try JSONSerialization.data(withJSONObject: [
            "_class_name": "AutoencoderKLLTX2Video", "in_channels": 3, "latent_channels": 8,
            "block_out_channels": [8, 16, 32, 32], "layers_per_block": [1, 1, 1, 1, 1],
            "spatio_temporal_scaling": [true, true, true, true],
            "downsample_type": ["spatial", "temporal", "spatiotemporal", "spatiotemporal"], "patch_size": 4,
            "resnet_norm_eps": 1e-6, "encoder_causal": true, "encoder_spatial_padding_mode": "zeros"]).write(to: encoderURL)
        let encoder = try NFKMLXLTX2VideoEncoderNet(try NFKMLXLTX2VideoVAE.encoderConfiguration(fromHuggingFace: encoderURL))
        try NFKMLXWeights.apply(vae.compactMap { key, value in
            key.hasPrefix("encoder.") ? (String(key.dropFirst("encoder.".count)), value) : nil
        }, to: encoder, verifyShapes: true)

        var schedule = NFKMLXFlowMatchConfiguration(baseShift: 1, shiftTerminal: nil, useDynamicShifting: false)
        schedule.rampEndsAtStepFraction = true
        return Stages(pipeline: pipeline, text: text, audioText: audioText, spatial: spatial, temporal: temporal,
                      encoder: encoder, schedule: schedule)
    }

    private func loraArrays(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: arrays.compactMap { key, value in
            key.hasPrefix("lora::") ? (String(key.dropFirst("lora::".count)), value) : nil
        })
    }

    private func coordinates(_ record: MLXArray) -> NFKMLXLTX2VideoCoordinates {
        let values = record.asType(.float32)
        return NFKMLXLTX2VideoCoordinates(
            starts: (0 ..< 3).map { values[0, $0, 0..., 0].asArray(Float.self) },
            ends: (0 ..< 3).map { values[0, $0, 0..., 1].asArray(Float.self) })
    }

    // MARK: - The layout

    // The segment grid picks the length that pads the request least, the longer on a tie; the round's seam
    // tiles and the epilogue's blend windows are the reference's.
    func testDFRLayoutMatchesTheReference() throws {
        XCTAssertEqual(try NFKMLXLTX2DFRLayout.canvas(frames: 121).positions, [24, 48, 72, 96, 120])
        XCTAssertEqual(try NFKMLXLTX2DFRLayout.canvas(frames: 97).positions, [32, 64, 96], "a tie keeps the longer segment")
        let padded = try NFKMLXLTX2DFRLayout.canvas(frames: 81)
        XCTAssertEqual([padded.frames, padded.segment], [97, 32])
        XCTAssertEqual(try NFKMLXLTX2DFRLayout.canvas(frames: 73).positions, [24, 48, 72])
        XCTAssertThrowsError(try NFKMLXLTX2DFRLayout.canvas(frames: 72))

        let tiles = try NFKMLXLTX2DFRLayout.temporalTiles(seamPositions: [48, 96, 144], frames: 145, tiles: 2)
        XCTAssertEqual(tiles.map { [$0.interval.start, $0.interval.end, $0.interval.leftRamp] }, [[0, 13, 0], [6, 19, 7]])
        XCTAssertEqual(tiles.map(\.slots), [[24, 72], [72, 120]])
        XCTAssertEqual(tiles.map(\.anchors), [[48, 96], [48, 96, 144]])

        let arrays = try record()
        let epilogue = try NFKMLXLTX2DFRLayout.epilogueTiles(latentShape: (19, 8, 12), frameTiles: 2, frameSeams: [6, 12, 18])
        XCTAssertEqual(epilogue.count, 8)
        for (index, tile) in epilogue.enumerated() {
            let extent = try XCTUnwrap(arrays["epilogue.tile\(index).extent"]).asArray(Int32.self).map(Int.init)
            XCTAssertEqual([tile.frames.lowerBound, tile.frames.upperBound, tile.heights.lowerBound, tile.heights.upperBound,
                            tile.widths.lowerBound, tile.widths.upperBound], extent, "tile \(index)")
            let reference = try XCTUnwrap(arrays["epilogue.tile\(index).weight"])
            XCTAssertEqual(tile.weight.shape, reference.shape)
            XCTAssertEqual(abs(tile.weight - reference).max().item(Float.self), 0, "tile \(index)'s window is exact")
        }
    }

    // The ancestral sampler's arithmetic on its own, against a step computed here from its definition.
    func testAncestralSamplerStepsDownThenRenoises() {
        let scheduler = NFKMLXLTX2AncestralScheduler(sigmas: [0.9, 0.5])
        XCTAssertEqual(scheduler.sigmas, [0.9, 0.5, 0])
        let sample = MLXArray([Float(1), -2]), velocity = MLXArray([Float(0.5), 0.25]), noise = MLXArray([Float(0.3), -0.1])
        let stepped = scheduler.step(velocity: velocity, sample: sample, index: 0) { _ in noise }
        let (s, n, eta): (Float, Float, Float) = (0.9, 0.5, 0.5)
        let down = n * (1 + (n / s - 1) * eta)
        let denoised = sample - s * velocity
        let deterministic = (down / s) * sample + (1 - down / s) * denoised
        let renoise = (n * n - down * down * (1 - n) * (1 - n) / ((1 - down) * (1 - down) + 1e-12)).squareRoot()
        let expected = ((1 - n) / ((1 - down) + 1e-12)) * deterministic + noise * renoise
        XCTAssertEqual(abs(stepped - expected).max().item(Float.self), 0, accuracy: 1e-7)
        XCTAssertEqual(scheduler.step(velocity: velocity, sample: sample, index: 1) { _ in noise }.asArray(Float.self),
                       (sample - 0.5 * velocity).asArray(Float.self), "the last step returns x0")
    }

    // The detailing IC-LoRA's every tensor lands on a layer of LTX-2.5's released transformer, its `A`
    // `[rank, inputs]` and `B` `[outputs, rank]` matching that layer, with no tensor left over.
    func testDetailingLoRAMatchesTheReleasedTransformer() throws {
        guard let root = config["IK_SHAPES_ROOT"] else { throw XCTSkip("set IK_SHAPES_ROOT") }
        let url = URL(fileURLWithPath: root).appendingPathComponent("ltx2-25-ic-lora/shapes.json")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no ltx2-25-ic-lora header inventory") }
        let shapes = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: [Int]])
        XCTAssertEqual(shapes.count, 960)
        let layers = Dictionary(uniqueKeysWithValues: NFKMLXLTX2TransformerNet(.ltx25).leafModules().flattened()
            .compactMap { path, module in (module as? Linear).map { (path, $0) } })
        var matched = Set<String>()
        for (key, shape) in shapes {
            let suffix = key.hasSuffix(".lora_A.weight") ? ".lora_A.weight" : ".lora_B.weight"
            let path = try XCTUnwrap(NFKMLXLTX2LoRA.modulePath(String(key.dropLast(suffix.count))), key)
            let layer = try XCTUnwrap(layers[path], "\(key) names \(path), which the transformer lacks")
            let (outputs, inputs) = layer.shape
            XCTAssertEqual(shape, suffix == ".lora_A.weight" ? [32, inputs] : [outputs, 32], key)
            matched.insert(path)
        }
        XCTAssertEqual(matched.count, 480, "48 blocks, ten layers each")
    }

    // The released IC-LoRA through the public loader onto LTX-2.5's transformer geometry: 480 adapters at the
    // recipe's weight, each holding the file's `A` and `B` transposed. The base weights are never evaluated.
    func testDetailingLoRALoadsOntoTheReleasedTransformer() throws {
        guard let path = config["IK_VAL_LTX2_5_IC_LORA"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_LTX2_5_IC_LORA (\(NFKMLXLTX2Generator.detailingLoRARepo))")
        }
        let arrays = try NFKMLXLTX2LoRA.arrays(at: URL(fileURLWithPath: path))
        let transformer = NFKMLXLTX2TransformerNet(.ltx25)
        XCTAssertEqual(try NFKMLXLTX2LoRA.apply(arrays, to: transformer, strength: 0.5), 480)
        let adapters = transformer.leafModules().flattened().compactMap { path, module in
            (module as? NFKMLXLoRAAdapter).map { (path, $0) }
        }
        XCTAssertEqual(adapters.count, 480)
        XCTAssertTrue(adapters.allSatisfy { $0.1.strength == 0.5 && $0.1.scale == 1 }, "alpha = rank: the weight alone scales it")
        let adapter = try XCTUnwrap(adapters.first { $0.0 == "transformer_blocks.47.ff.net.2" }?.1)
        let a = try XCTUnwrap(arrays["diffusion_model.transformer_blocks.47.ff.net.2.lora_A.weight"])
        let b = try XCTUnwrap(arrays["diffusion_model.transformer_blocks.47.ff.net.2.lora_B.weight"])
        XCTAssertEqual(abs(adapter.loraA - a.transposed().asType(.float32)).max().item(Float.self), 0)
        XCTAssertEqual(abs(adapter.loraB - b.transposed().asType(.float32)).max().item(Float.self), 0)
        NFKMLXLTX2LoRA.setStrength(0, in: transformer)
        XCTAssertTrue(adapters.allSatisfy { $0.1.strength == 0 })
    }

    // The epilogue's keyframe rebuild rounds each decoded frame to bytes, so its floor is the decode's
    // last-ulp differences landing on the other side of a half: the stretch is Pillow's to the byte on the
    // reference's own bytes, the bytes differ from the reference's by one level where they differ at all, and
    // encoding the reference's stretched bytes gives its keyframes to the encoder's own precision.
    func testEpilogueKeyframeRebuildFloorIsTheByteRounding() throws {
        let arrays = try record()
        let s = try stages(arrays)
        let keyframes = channelsLast(try XCTUnwrap(arrays["round.keyframes"]))
        let reference = channelsLast(try XCTUnwrap(arrays["epilogue.rebuilt"]))
        var flipped = 0, total = 0
        for index in 0 ..< keyframes.dim(1) {
            let rgb = try XCTUnwrap(arrays["epilogue.rgb\(index)"]).asArray(Int32.self).map(UInt8.init)
            let stretched = try XCTUnwrap(arrays["epilogue.stretched\(index)"]).asArray(Int32.self).map(UInt8.init)
            XCTAssertEqual(NFKMLXPillowLanczos.resize(rgb, width: 192, height: 128, outWidth: 384, outHeight: 256),
                           stretched, "plane \(index): Pillow's Lanczos to the byte")
            let mine = s.pipeline.keyframeBytes(keyframes[0..., index ..< index + 1]).rgb
            let differences = zip(mine, rgb).map { abs(Int($0) - Int($1)) }
            XCTAssertLessThanOrEqual(differences.max() ?? 0, 1, "plane \(index): one level at most")
            flipped += differences.filter { $0 > 0 }.count
            total += rgb.count
            compare("dfr rebuild plane \(index) from the reference's bytes",
                    s.pipeline.encodedBytes(stretched, width: 384, height: 256, encoder: s.encoder),
                    reference[0..., index ..< index + 1])
        }
        print("VALIDATION PARITY ltx2.5-dfr-rebuild-floor: \(flipped) of \(total) bytes one level apart")
    }

    // MARK: - The stages, each from the reference's inputs

    /// The record's image conditions under `prefix` (none for text-to-video), as the reference gave them: a still
    /// at latent 0 at strength 1 and one at latent −1 at strength 0.8.
    private func imageConditions(_ arrays: [String: MLXArray], prefix: String) -> [NFKMLXLTX2ImageCondition] {
        guard let first = arrays["\(prefix)image0"], let second = arrays["\(prefix)image1"] else { return [] }
        return [NFKMLXLTX2ImageCondition(pixels: first, latentIndex: 0, strength: 1),
                NFKMLXLTX2ImageCondition(pixels: second, latentIndex: -1, strength: 0.8)]
    }

    /// The draws of one chain in order: stage 1's two, stage 2's two, each round tile's starting draw and its
    /// renoises, and the epilogue's two.
    private struct ChainDraws {
        var passes: [MLXArray] = []
        var tileStarts: [MLXArray] = []
        var renoises: [[MLXArray]] = []
        init(_ arrays: [String: MLXArray], prefix: String) {
            let keys = arrays.keys.filter { $0.hasPrefix("\(prefix)draw") && !$0.hasSuffix("draws") }.sorted()
            for key in keys {
                let value = arrays[key]!
                switch key.split(separator: ".").last! {
                case "dfr": passes.append(value)
                case "refine": tileStarts.append(value); renoises.append([])
                default: renoises[renoises.count - 1].append(value)
                }
            }
        }
    }

    /// Every DFR stage under `prefix`, each from the reference's own inputs, against the reference.
    private func assertStages(prefix: String) throws {
        let arrays = try record()
        func ref(_ key: String) -> MLXArray? { arrays[prefix + key] }
        func refLast(_ key: String) throws -> MLXArray { channelsLast(try XCTUnwrap(ref(key), key)) }
        let s = try stages(arrays)
        let pipeline = s.pipeline
        let images = imageConditions(arrays, prefix: prefix)
        let draws = ChainDraws(arrays, prefix: prefix)
        XCTAssertEqual([draws.passes.count, draws.tileStarts.count] + draws.renoises.map(\.count), [6, 2, 3, 3])
        func conditions(width: Int, height: Int, frames: Int) -> [NFKMLXLTX2DFRCondition] {
            pipeline.encodedConditions(images, width: width, height: height, frames: frames, encoder: s.encoder)
        }

        // Stage 1: the first call's inputs and prediction, then the pass.
        let stageOneConditions = conditions(width: 96, height: 64, frames: 73)
        let sequence = NFKMLXLTX2DFR.sequence(
            configuration: pipeline.transformer.config, latentFrames: 10, latentHeight: 2, latentWidth: 3, channels: 8,
            conditions: stageOneConditions, slots: [24, 48, 72], fps: 24, noiseScale: 1, noise: { _ in draws.passes[0] })
        compare("\(prefix)dfr stage1 call0 video", sequence.latents, ref("stage1.call0.video"))
        compare("\(prefix)dfr stage1 call0 keyframes", sequence.keyframes.expandedDimensions(axis: -1),
                ref("stage1.call0.keyframes"))
        let coordinates = coordinates(try XCTUnwrap(ref("stage1.call0.coords")))
        XCTAssertEqual(sequence.coordinates.starts, coordinates.starts, "\(prefix)stage 1 coordinates' starts")
        XCTAssertEqual(sequence.coordinates.ends, coordinates.ends, "\(prefix)stage 1 coordinates' ends")
        let timestep = 1000 * (1 - sequence.conditioning.squeezed(axis: -1))
        compare("\(prefix)dfr stage1 call0 timestep", timestep, ref("stage1.call0.timestep"))
        let pass = NFKMLXLTX2Pass(video: sequence.latents, audio: draws.passes[1], text: s.text, audioText: s.audioText,
                                  timestep: timestep, audioTimestep: MLXArray([Float(1000)]), sigma: MLXArray([Float(1000)]),
                                  audioSigma: MLXArray([Float(1000)]), keyframes: sequence.keyframes)
        compare("\(prefix)dfr stage1 call0 prediction", pipeline.transformer.predictions(
            [pass], frames: 10, height: 2, width: 3, audioFrames: 76, fps: 24, crossTimestep: true,
            videoCoordinates: sequence.coordinates)[0].video, ref("stage1.call0.prediction"))
        let first = try pipeline.dfrPass(
            text: s.text, audioText: s.audioText, frames: 73, latentHeight: 2, latentWidth: 3, frameRate: 24,
            sigmas: [1.0, 0.909375, 0.421875], conditions: stageOneConditions, schedule: s.schedule,
            sequenceNoise: { _ in draws.passes[0] }, audioNoise: { _ in draws.passes[1] })
        XCTAssertEqual(first.positions, [24, 48, 72])
        let stageOne = compare("\(prefix)dfr stage1 video", channelsFirst(first.video), ref("stage1.video"))
        compare("\(prefix)dfr stage1 slots", channelsFirst(try XCTUnwrap(first.keyframes)), ref("stage1.keyframes"))
        compare("\(prefix)dfr stage1 audio", publicAudio(first.audio), ref("stage1.audio"))

        // The spatial upsampler, then stage 2 with the LoRA at 0.5 and stage 1's video as the reference.
        let stageOneVideo = try refLast("stage1.video")
        compare("\(prefix)dfr upsampled video", channelsFirst(NFKMLXLTX2Pipeline.upsampled(stageOneVideo, by: s.spatial)),
                ref("up.video"))
        compare("\(prefix)dfr upsampled slots", channelsFirst(NFKMLXLTX2Pipeline.upsampled(try refLast("stage1.keyframes"),
                                                                                         by: s.spatial)), ref("up.keyframes"))
        XCTAssertEqual(try NFKMLXLTX2LoRA.apply(loraArrays(arrays), to: pipeline.transformer, strength: 0.5), 14)
        let second = try pipeline.dfrPass(
            text: s.text, audioText: s.audioText, frames: 73, latentHeight: 4, latentWidth: 6, frameRate: 24,
            sigmas: NFKMLXLTX2Pipeline.refinementSigmas, latents: try refLast("up.video"),
            audio: packedAudio(try XCTUnwrap(ref("stage1.audio"))), slotInitials: try refLast("up.keyframes"),
            slotPositions: [24, 48, 72], reference: stageOneVideo,
            conditions: conditions(width: 192, height: 128, frames: 73), schedule: s.schedule,
            sequenceNoise: { _ in draws.passes[2] }, audioNoise: { _ in draws.passes[3] })
        let stageTwo = compare("\(prefix)dfr stage2 video", channelsFirst(second.video), ref("stage2.video"))
        compare("\(prefix)dfr stage2 slots", channelsFirst(try XCTUnwrap(second.keyframes)), ref("stage2.keyframes"))
        compare("\(prefix)dfr stage2 audio", publicAudio(second.audio), ref("stage2.audio"))

        // One temporal round with the LoRA off.
        NFKMLXLTX2LoRA.setStrength(0, in: pipeline.transformer)
        var consumed = [0, 0]
        let round = try pipeline.temporalRound(
            video: try refLast("stage2.video"), keyframes: try refLast("stage2.keyframes"), positions: [24, 48, 72],
            audio: packedAudio(try XCTUnwrap(ref("stage1.audio"))), frames: 73, frameRate: 24, sourceSeconds: 73.0 / 24.0,
            round: 1, upsampler: s.temporal, text: s.text, audioText: s.audioText,
            conditions: conditions(width: 192, height: 128, frames: 73),
            tileNoise: { index, shape in
                XCTAssertEqual(draws.tileStarts[index].shape, shape)
                return draws.tileStarts[index]
            },
            ancestralNoise: { index in
                { _ in
                    defer { consumed[index] += 1 }
                    return draws.renoises[index][consumed[index]]
                }
            })
        XCTAssertEqual(consumed, [3, 3], "three renoises a tile: every step but the last")
        XCTAssertEqual(round.positions, try XCTUnwrap(ref("round.positions")).asArray(Int32.self).map(Int.init))
        let roundVideo = compare("\(prefix)dfr round video", channelsFirst(round.video), ref("round.video"))
        compare("\(prefix)dfr round keyframes", channelsFirst(try XCTUnwrap(round.keyframes)), ref("round.keyframes"))
        compare("\(prefix)dfr round audio", publicAudio(round.audio), ref("round.audio"))

        // The third spatial stage: keyframes rebuilt through RGB, then the tiled pass with the LoRA back on.
        let rebuilt = pipeline.rebuiltKeyframes(try refLast("round.keyframes"), encoder: s.encoder)
        compare("\(prefix)dfr epilogue rebuilt keyframes", channelsFirst(rebuilt), ref("epilogue.rebuilt"))
        compare("\(prefix)dfr epilogue upsampled", channelsFirst(NFKMLXLTX2Pipeline.upsampled(try refLast("round.video"),
                                                                                            by: s.spatial)),
                ref("epilogue.upsampled"))
        NFKMLXLTX2LoRA.setStrength(0.5, in: pipeline.transformer)
        let tiles = try NFKMLXLTX2DFRLayout.epilogueTiles(latentShape: (19, 8, 12), frameTiles: 2,
                                                          frameSeams: [24, 48, 72].map { 2 * $0 / 8 })
        let epilogue = try pipeline.dfrPass(
            text: s.text, audioText: s.audioText, frames: 145, latentHeight: 8, latentWidth: 12, frameRate: 48,
            sigmas: NFKMLXLTX2Pipeline.refinementSigmas, latents: try refLast("epilogue.upsampled"),
            audio: packedAudio(try XCTUnwrap(ref("stage1.audio"))), freezesAudio: true, generatesSlots: false,
            reference: try refLast("round.video"),
            guidanceKeyframes: (try refLast("epilogue.rebuilt"), round.positions, 1),
            conditions: conditions(width: 384, height: 256, frames: 145), tiles: tiles, schedule: s.schedule,
            sequenceNoise: { _ in draws.passes[4] }, audioNoise: { _ in draws.passes[5] })
        let epilogueVideo = compare("\(prefix)dfr epilogue video", channelsFirst(epilogue.video), ref("epilogue.video"))
        compare("\(prefix)dfr epilogue audio", publicAudio(epilogue.audio), ref("epilogue.audio"))
        print("VALIDATION PARITY ltx2.5-dfr-\(prefix.isEmpty ? "t2v" : "i2v")-stages: stage 1 video latent cosine \(stageOne), "
              + "stage 2 \(stageTwo), round \(roundVideo), epilogue \(epilogueVideo)")
    }

    // Text-to-video: stage 1 with its slots, the spatial upsampler, stage 2 with the IC-LoRA and the reference,
    // one temporal round, and the third spatial stage.
    func testDFRStagesMatchTheReference() throws {
        try assertStages(prefix: "")
    }

    // The same stages conditioned on two stills: one replacing the first frame, one appended at the last latent
    // frame at 0.8, carried into every pass at that pass's size and into the round's tiles at twice its frame.
    func testDFRStagesWithImageConditionsMatchTheReference() throws {
        try assertStages(prefix: "i2v.")
    }

    // The whole recipe on its own outputs, stage after stage, with the reference's draws.
    func testDFRRecipeChainsToTheReference() throws {
        let arrays = try record()
        let s = try stages(arrays)
        let pipeline = s.pipeline
        let draws = ChainDraws(arrays, prefix: "")
        let first = try pipeline.dfrPass(
            text: s.text, audioText: s.audioText, frames: 73, latentHeight: 2, latentWidth: 3, frameRate: 24,
            sigmas: [1.0, 0.909375, 0.421875], schedule: s.schedule,
            sequenceNoise: { _ in draws.passes[0] }, audioNoise: { _ in draws.passes[1] })
        try NFKMLXLTX2LoRA.apply(loraArrays(arrays), to: pipeline.transformer, strength: 0.5)
        let second = try pipeline.dfrPass(
            text: s.text, audioText: s.audioText, frames: 73, latentHeight: 4, latentWidth: 6, frameRate: 24,
            sigmas: NFKMLXLTX2Pipeline.refinementSigmas,
            latents: NFKMLXLTX2Pipeline.upsampled(first.video, by: s.spatial), audio: first.audio,
            slotInitials: NFKMLXLTX2Pipeline.upsampled(try XCTUnwrap(first.keyframes), by: s.spatial),
            slotPositions: first.positions, reference: first.video, schedule: s.schedule,
            sequenceNoise: { _ in draws.passes[2] }, audioNoise: { _ in draws.passes[3] })
        NFKMLXLTX2LoRA.setStrength(0, in: pipeline.transformer)
        var consumed = [0, 0]
        let round = try pipeline.temporalRound(
            video: second.video, keyframes: try XCTUnwrap(second.keyframes), positions: second.positions,
            audio: first.audio, frames: second.canvasFrames, frameRate: 24,
            sourceSeconds: Double(second.canvasFrames) / 24, round: 1, upsampler: s.temporal, text: s.text,
            audioText: s.audioText, tileNoise: { index, _ in draws.tileStarts[index] },
            ancestralNoise: { index in
                { _ in
                    defer { consumed[index] += 1 }
                    return draws.renoises[index][consumed[index]]
                }
            })
        let rebuilt = pipeline.rebuiltKeyframes(try XCTUnwrap(round.keyframes), encoder: s.encoder)
        NFKMLXLTX2LoRA.setStrength(0.5, in: pipeline.transformer)
        let epilogue = try pipeline.dfrPass(
            text: s.text, audioText: s.audioText, frames: round.canvasFrames, latentHeight: 8, latentWidth: 12,
            frameRate: 48, sigmas: NFKMLXLTX2Pipeline.refinementSigmas,
            latents: NFKMLXLTX2Pipeline.upsampled(round.video, by: s.spatial), audio: first.audio, freezesAudio: true,
            generatesSlots: false, reference: round.video, guidanceKeyframes: (rebuilt, round.positions, 1),
            tiles: try NFKMLXLTX2DFRLayout.epilogueTiles(latentShape: (19, 8, 12), frameTiles: 2,
                                                         frameSeams: second.positions.map { 2 * $0 / 8 }),
            schedule: s.schedule, sequenceNoise: { _ in draws.passes[4] }, audioNoise: { _ in draws.passes[5] })
        compare("dfr chained round video", channelsFirst(round.video), arrays["round.video"])
        let video = compare("dfr chained epilogue video", channelsFirst(epilogue.video), arrays["epilogue.video"])
        print("VALIDATION PARITY ltx2.5-dfr-chain: epilogue video latent cosine \(video)")
    }
}

// MARK: - A tiny LTX-2.5 release

extension NFKMLXLTX2DFRTests {

    /// A release directory in the diffusers layout holding the record's tiny LTX-2.5 stages, its autoencoder with
    /// the encoder, both upsamplers, and its LoRA as a file beside it, in the release's own tensor layout.
    private func tinyRelease(_ arrays: [String: MLXArray]) throws -> (directory: URL, lora: URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("ltx25-dfr-tiny-\(UUID().uuidString)")
        func write(_ json: [String: Any], _ path: String) throws {
            let url = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys]).write(to: url)
        }
        func weights(_ prefix: String, _ path: String, rename: (String) -> String? = { $0 }) throws {
            var selected = [String: MLXArray]()
            for (key, value) in arrays where key.hasPrefix("w::\(prefix).") {
                if let name = rename(String(key.dropFirst(3 + prefix.count + 1))) { selected[name] = value }
            }
            try MLX.save(arrays: selected, url: root.appendingPathComponent(path))
        }
        try write(["_class_name": "LTX2Pipeline"], "model_index.json")
        try write(["_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 1024, "base_shift": 0.95,
                   "max_image_seq_len": 4096, "max_shift": 2.05, "shift": 1.0, "use_dynamic_shifting": false],
                  "scheduler/scheduler_config.json")
        try write(["model_type": "gemma3_text", "vocab_size": 64, "hidden_size": 12, "intermediate_size": 24,
                   "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
                   "query_pre_attn_scalar": 8, "sliding_window": 4,
                   "layer_types": ["sliding_attention", "full_attention"], "rope_theta": 10000,
                   "rope_local_base_freq": 10000, "rms_norm_eps": 1e-6], "text_encoder/config.json")
        try weights("text", "text_encoder/model.safetensors") { $0.hasSuffix("lm_head.weight") ? nil : $0 }
        var vocabulary: [String: Int] = ["<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "\u{2581}": 4]
        for (offset, scalar) in "abcdefghijklmnopqrstuvwxyz".unicodeScalars.enumerated() {
            vocabulary[String(scalar)] = 5 + offset
        }
        try write(["model": ["type": "BPE", "vocab": vocabulary, "merges": [String](), "unk_token": "<unk>"],
                   "added_tokens": [["id": 2, "content": "<bos>", "special": true]]], "tokenizer/tokenizer.json")
        try write(["_class_name": "LTX2TextConnectors", "caption_channels": 12, "text_proj_in_factor": 3,
                   "video_connector_num_attention_heads": 2, "video_connector_attention_head_dim": 16,
                   "video_connector_num_layers": 2, "video_connector_num_learnable_registers": 4,
                   "audio_connector_num_attention_heads": 2, "audio_connector_attention_head_dim": 8,
                   "audio_connector_num_layers": 2, "audio_connector_num_learnable_registers": 4,
                   "per_modality_projections": true, "proj_bias": true, "rope_type": "split",
                   "video_gated_attn": true, "audio_gated_attn": true, "video_hidden_dim": 32,
                   "audio_hidden_dim": 16], "connectors/config.json")
        try weights("connectors", "connectors/diffusion_pytorch_model.safetensors")
        try write(["_class_name": "LTX2VideoTransformer3DModel", "in_channels": 8, "out_channels": 8,
                   "num_attention_heads": 2, "attention_head_dim": 16, "cross_attention_dim": 32, "base_height": 64,
                   "base_width": 64, "audio_in_channels": 8, "audio_out_channels": 8, "audio_num_attention_heads": 2,
                   "audio_attention_head_dim": 8, "audio_cross_attention_dim": 16, "num_layers": 2,
                   "caption_channels": 14, "ff_bias": false, "use_keyframes_abs_pos_embedding": true],
                  "transformer/config.json")
        try weights("transformer", "transformer/diffusion_pytorch_model.safetensors")
        try write(["_class_name": "AutoencoderKLLTX2Video", "in_channels": 3, "latent_channels": 8,
                   "block_out_channels": [8, 16, 32, 32], "layers_per_block": [1, 1, 1, 1, 1],
                   "spatio_temporal_scaling": [true, true, true, true],
                   "downsample_type": ["spatial", "temporal", "spatiotemporal", "spatiotemporal"],
                   "encoder_causal": true, "encoder_spatial_padding_mode": "zeros",
                   "decoder_block_out_channels": [8, 16, 16, 32], "decoder_layers_per_block": [1, 2, 1, 1, 1],
                   "decoder_spatio_temporal_scaling": [true, true, true, true],
                   "upsample_type": ["spatiotemporal", "spatiotemporal", "temporal", "spatial"],
                   "upsample_residual": [false, false, false, false], "upsample_factor": [2, 2, 1, 2],
                   "decoder_causal": false, "decoder_inject_noise": [false, false, false, false, false],
                   "timestep_conditioning": false], "vae/config.json")
        try weights("vae", "vae/diffusion_pytorch_model.safetensors")
        try write(["_class_name": "AutoencoderKLLTX2Audio", "base_channels": 8, "latent_channels": 2, "mel_bins": 16,
                   "norm_type": "pixel", "causality_axis": "height"], "audio_vae/config.json")
        try weights("audio_vae", "audio_vae/diffusion_pytorch_model.safetensors")
        try write(["_class_name": "LTX2VocoderWithBWE", "in_channels": 32, "hidden_channels": 32,
                   "upsample_kernel_sizes": [11, 4], "upsample_factors": [5, 2], "bwe_in_channels": 16,
                   "bwe_hidden_channels": 16, "bwe_upsample_kernel_sizes": [12, 11], "bwe_upsample_factors": [6, 5],
                   "filter_length": 32, "hop_length": 10, "window_length": 32, "num_mel_channels": 8],
                  "vocoder/config.json")
        try weights("vocoder", "vocoder/diffusion_pytorch_model.safetensors")
        try write(["_class_name": "LTX2LatentUpsamplerModel", "in_channels": 8, "mid_channels": 32,
                   "num_blocks_per_stage": 2, "dims": 3, "spatial_upsample": true, "temporal_upsample": false,
                   "use_rational_resampler": false], "latent_upsampler/config.json")
        try weights("upsampler", "latent_upsampler/diffusion_pytorch_model.safetensors")
        try write(["_class_name": "LTX2LatentUpsamplerModel", "in_channels": 8, "mid_channels": 32,
                   "num_blocks_per_stage": 2, "dims": 3, "spatial_upsample": false, "temporal_upsample": true,
                   "use_rational_resampler": true], "temporal_latent_upsampler/config.json")
        try weights("temporal", "temporal_latent_upsampler/diffusion_pytorch_model.safetensors")
        let lora = root.appendingPathComponent("detailing.safetensors")
        try MLX.save(arrays: loraArrays(arrays), url: lora)
        return (root, lora)
    }

    // The public factory on a tiny LTX-2.5 release: DFR with one temporal round, the epilogue, the IC-LoRA, and an
    // image condition through the generator's switches, then through the backend's request keys and its
    // NFKInputImage; DFR refuses the other decode-side paths and an image without DFR.
    func testATinyLTX25ReleaseRendersWithDFR() throws {
        let arrays = try record()
        let (release, lora) = try tinyRelease(arrays)
        defer { try? FileManager.default.removeItem(at: release) }

        let generator = try NFKMLXLTX2Generator.generator(directoryURL: release, residency: .staged)
        XCTAssertEqual(generator.variant, .ltx25)
        XCTAssertFalse(generator.rendersWithDFR || generator.dfrEpilogue || generator.hasDetailingLoRA)
        XCTAssertEqual(generator.dfrTemporalRounds, 1)
        XCTAssertEqual(generator.detailingStrength, 0.5)
        try generator.loadDetailingLoRA(at: lora)
        XCTAssertTrue(generator.hasDetailingLoRA)
        generator.sigmas = [1.0, 0.909375, 0.421875].map { NSNumber(value: $0) }

        generator.imageConditions = [NFKMLXLTX2ImageCondition(pixels: MLXRandom.uniform(0 ..< 255, [50, 70, 3]))]
        XCTAssertThrowsError(try generator.clip(forPrompt: "a red fox", frames: 73, width: 192, height: 128),
                             "an image condition needs DFR")
        generator.rendersWithDFR = true
        generator.dfrEpilogue = true
        generator.refinesUpsampled = true
        XCTAssertThrowsError(try generator.clip(forPrompt: "a red fox", frames: 73, width: 192, height: 128),
                             "DFR runs its own spatial stage")
        generator.refinesUpsampled = false
        XCTAssertThrowsError(try generator.clip(forPrompt: "a red fox", frames: 73, width: 160, height: 128),
                             "DFR halves the clip, so its sides are multiples of 64")
        var reported = 0
        let clip = try generator.clip(forPrompt: "a red fox", frames: 73, width: 192, height: 128, seed: 5) { _ in
            reported += 1
        }
        XCTAssertEqual(clip.video.shape, [145, 256, 384, 3], "one round doubles the frames; the epilogue the size")
        XCTAssertEqual(generator.outputFrameRate, 48)
        XCTAssertEqual(reported, 3 + 3 + 2 * 4 + 3, "stage 1, stage 2, two round tiles, the epilogue")
        let values = clip.video.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
        // The tiny vocoder makes far fewer samples a second than the 48 kHz its configuration names, so the cut
        // is measured on a waveform long enough to need it.
        let cut = Int((145.0 / 48 * Double(generator.audioSampleRate)).rounded(.toNearestOrEven))
        XCTAssertEqual(generator.trimmedSoundtrack(MLXArray.zeros([cut + 1000, 2]), frames: 73).dim(0), cut,
                       "the soundtrack is cut to the clip")
        XCTAssertLessThanOrEqual(clip.audio.dim(0), cut)

        generator.imageConditions = []
        generator.rendersWithDFR = false
        generator.dfrEpilogue = false
        let picture = try NFKMLXImageBridge.cgImage(from: MLXRandom.uniform(0 ..< 1, [40, 60, 3]), options: NFKMLXImageOptions())
        let backend = generator.inferenceBackend()
        XCTAssertTrue(backend.supportedInputKeys?.contains(NFKInputImage) == true)
        let result = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a red fox", NFKInputImage: picture],
            parameters: [NFKParameterFrameCount: 73, NFKParameterWidth: 128, NFKParameterHeight: 64,
                         NFKMLXLTX2ParameterKey.dfr: true, NFKMLXLTX2ParameterKey.dfrTemporalRounds: 0]))
        let movie = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let wave = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        defer {
            movie.fileURL.map { try? FileManager.default.removeItem(at: $0) }
            wave.fileURL.map { try? FileManager.default.removeItem(at: $0) }
        }
        XCTAssertEqual(movie.dimensions, CGSize(width: 128, height: 64))
        XCTAssertEqual(movie.framesPerSecond, 24)
        XCTAssertEqual(try NFKMLXVideoFile.read(try XCTUnwrap(movie.fileURL)).frames.count, 73)
        XCTAssertFalse(generator.rendersWithDFR, "a request's switches apply to that request only")
        XCTAssertTrue(generator.imageConditions.isEmpty, "a request's image applies to that request only")
    }
}
