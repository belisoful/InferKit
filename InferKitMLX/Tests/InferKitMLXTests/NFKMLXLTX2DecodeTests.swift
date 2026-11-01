//
//  NFKMLXLTX2DecodeTests.swift
//  InferKitMLXTests
//
//  LTX-2.5's decode-side components against diffusers: the diffusion video decoder, the spatial and
//  temporal latent upsamplers, and the two-stage distilled recipe with both decodes, at tiny random
//  configurations (`Tools/reference-parity/run_reference.py ltx2_diffusion_decoder`, `ltx2_latent_upsampler`,
//  `ltx2_two_stage`), on the released weights (`ltx2_real_diffusion_decoder`), and against the released
//  headers. Each test skips without its key.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLTX2DecodeTests: XCTestCase {

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

    private func record(_ key: String, oracle: String) throws -> [String: MLXArray] {
        guard let path = config[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key) (run_reference.py \(oracle))")
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

    /// The `w::<prefix>.` weights of a record, without the prefix.
    private func weights(_ arrays: [String: MLXArray], prefix: String) -> [(String, MLXArray)] {
        arrays.compactMap { key, value in
            key.hasPrefix("w::\(prefix).") ? (String(key.dropFirst(3 + prefix.count + 1)), value) : nil
        }
    }

    /// NDHWC → the reference's NCDHW, and back.
    private func channelsFirst(_ x: MLXArray) -> MLXArray { x.transposed(0, 4, 1, 2, 3) }
    private func channelsLast(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 4, 1) }

    private func diffusionDecoder(_ arrays: [String: MLXArray], prefix: String,
                                  configuration: NFKMLXLTX2DiffusionDecoderConfiguration = .tiny) throws
        -> NFKMLXLTX2DiffusionDecoder {
        try NFKMLXLTX2DiffusionDecoders.apply(weights(arrays, prefix: prefix),
                                              to: NFKMLXLTX2DiffusionDecoderNet(configuration))
    }

    private func upsampler(_ arrays: [String: MLXArray], prefix: String,
                           configuration: NFKMLXLTX2LatentUpsamplerConfiguration) throws -> NFKMLXLTX2LatentUpsamplerNet {
        let net = NFKMLXLTX2LatentUpsamplerNet(configuration)
        try NFKMLXLTX2LatentUpsamplers.apply(
            weights(arrays, prefix: prefix).map { ($0, NFKMLXLTX2LatentUpsamplers.laid($1)) }, to: net)
        return net
    }

    private func tinyUpsampler(_ change: (inout NFKMLXLTX2LatentUpsamplerConfiguration) -> Void = { _ in })
        -> NFKMLXLTX2LatentUpsamplerConfiguration {
        var configuration = NFKMLXLTX2LatentUpsamplerConfiguration(inChannels: 8, midChannels: 32, blocksPerStage: 2)
        change(&configuration)
        return configuration
    }

    // MARK: - The diffusion decoder

    // The release's one-step x0 decode, three x0 steps (reverse Euler through `(x − x0) / t`), and two
    // velocity steps, every seam from `conv_in` through the stage-5 blocks against the reference's own
    // pieces, with the reference's starting noise.
    func testLTX25DiffusionDecoderMatchesTheReference() throws {
        let arrays = try record("IK_PARITY_LTX2_DIFFUSION_DECODER", oracle: "ltx2_diffusion_decoder")
        let latent = channelsLast(try XCTUnwrap(arrays["latent"]))
        for (prefix, steps) in [("x0", 1), ("x0s3", 3), ("v2", 2)] {
            var configuration = NFKMLXLTX2DiffusionDecoderConfiguration.tiny
            configuration.predictsSample = prefix != "v2"
            let model = try diffusionDecoder(arrays, prefix: prefix, configuration: configuration)
            let net = model.decoder
            compare("ltx2.5-decoder \(prefix) latents_mean", model.statistics.mean, arrays["w::\(prefix).latents_mean"])

            let pad = configuration.trailingPadFrames
            let last = latent[0..., (latent.dim(1) - 1)...]
            var h = net.convIn(concatenated([latent, broadcast(last, to: [1, pad, 3, 4, 8])], axis: 1))
            compare("ltx2.5-decoder \(prefix) conv_in", h, arrays["\(prefix).seam.conv_in"])
            for stage in 0 ..< net.stages.count - 1 {
                h = net.stages[stage].reduce(h) { $1($0) }
                compare("ltx2.5-decoder \(prefix) stage\(stage)", h, arrays["\(prefix).seam.stage\(stage)"])
                h = net.upsamples[stage](h)
                compare("ltx2.5-decoder \(prefix) up\(stage)", h, arrays["\(prefix).seam.up\(stage)"])
            }
            compare("ltx2.5-decoder \(prefix) features", net.earlyStages(latent), arrays["\(prefix).seam.features"])
            let context = net.context(latent)
            compare("ltx2.5-decoder \(prefix) context", context, arrays["\(prefix).seam.context"])

            let noise = channelsLast(try XCTUnwrap(arrays["\(prefix).seam.noise"]))
            let embedded = net.timeEmbedder(MLXArray([Float(1)]) * configuration.timestepScale)
            compare("ltx2.5-decoder \(prefix) t_emb", embedded, arrays["\(prefix).seam.t_emb"])
            let modulation = net.sharedAdaptiveNorm(embedded)
            compare("ltx2.5-decoder \(prefix) modulation", modulation, arrays["\(prefix).seam.modulation"])
            var sample = net.convInSample(ltx2Patchify(noise, patch: configuration.patchSize))
            for (index, block) in net.diffusionBlocks.enumerated() {
                sample = block(sample, context: context, modulation: modulation)
                compare("ltx2.5-decoder \(prefix) diff\(index)", sample, arrays["\(prefix).seam.diff\(index)"])
            }
            compare("ltx2.5-decoder \(prefix) prediction",
                    channelsFirst(net.prediction(context: context, sample: noise, timestep: MLXArray([Float(1)]))),
                    arrays["\(prefix).seam.prediction"])
            let decoded = model.decode(latent, steps: steps) { shape in
                XCTAssertEqual(shape, noise.shape)
                return noise
            }
            let similarity = compare("ltx2.5-decoder \(prefix) decoded", channelsFirst(decoded), arrays["\(prefix).decoded"])
            print("VALIDATION PARITY ltx2.5-diffusion-decoder-\(prefix): decoded video cosine \(similarity)")
        }
        XCTAssertGreaterThan(try XCTUnwrap(arrays["byrows.cosine"]).item(Double.self), 0.9999999,
                             "the released-weight oracle's row-blocked attention is the reference's")
    }

    // The tiled decode in the release's one-step arrangement, where each tile draws its own noise, and at
    // three steps, where the tiles slice one canvas: tiles small enough to cut every axis, the remnant
    // merges and the blends included, each draw in the reference's order.
    func testLTX25DiffusionDecoderTilesAsTheReferenceDoes() throws {
        let arrays = try record("IK_PARITY_LTX2_DIFFUSION_DECODER", oracle: "ltx2_diffusion_decoder")
        let model = try diffusionDecoder(arrays, prefix: "x0")
        let tiling = NFKMLXLTX2DecodeTiling(height: 64, width: 64, frames: 16, heightStride: 48, widthStride: 48,
                                            frameStride: 8)
        for (prefix, steps) in [("tiled", 1), ("tiled3", 3)] {
            let draws = Int(try XCTUnwrap(arrays["\(prefix).draws"]).item(Int32.self))
            var drawn = 0
            let decoded = model.decode(channelsLast(try XCTUnwrap(arrays["\(prefix).latent"])), steps: steps,
                                       tiling: tiling) { shape in
                defer { drawn += 1 }
                let noise = channelsLast(arrays["\(prefix).noise\(drawn)"] ?? MLXArray.zeros([1, 1, 1, 1, 1]))
                XCTAssertEqual(shape, noise.shape, "\(prefix) draw \(drawn)")
                return noise
            }
            XCTAssertEqual(drawn, draws, "\(prefix): the reference's draw count")
            let similarity = compare("ltx2.5-decoder \(prefix) decoded", channelsFirst(decoded), arrays["\(prefix).decoded"])
            print("VALIDATION PARITY ltx2.5-diffusion-decoder-\(prefix): decoded video cosine \(similarity)")
        }
    }

    // The window arithmetic on its own: every query reads exactly `kernel` keys per axis, the window
    // shifted inward at the borders, and the blocks cover each position once.
    func testNeighborhoodWindowsShiftInwardAtTheBorders() {
        let axis = NFKLTX2NeighborhoodAxis(length: 17, kernel: 5)
        XCTAssertEqual(axis.starts, [0, 8, 9])
        XCTAssertEqual((0 ..< 17).map(axis.windowStart), [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12])
        XCTAssertEqual(axis.starts.indices.map { axis.block - axis.skip($0) }.reduce(0, +), 17)
        for start in axis.starts {
            let mask = axis.mask(start).asType(.int32)
            XCTAssertEqual(mask.sum(axis: 1).asArray(Int32.self), [Int32](repeating: 5, count: axis.block))
        }
        let short = NFKLTX2NeighborhoodAxis(length: 3, kernel: 7)
        XCTAssertEqual([short.kernel, short.block, short.box], [3, 3, 3], "a window wider than the grid reads all of it")
    }

    // MARK: - The latent upsamplers

    // LTX-2.5's spatial ×2 (a per-frame convolution and a 2×2 shuffle) and temporal ×2 (the first frame
    // dropped), the rational resampler at 1.5 with its blur, and the spatiotemporal ×2.
    func testLTX2LatentUpsamplersMatchTheReference() throws {
        let arrays = try record("IK_PARITY_LTX2_LATENT_UPSAMPLER", oracle: "ltx2_latent_upsampler")
        let latent = channelsLast(try XCTUnwrap(arrays["latent"]))
        let arrangements: [(String, NFKMLXLTX2LatentUpsamplerConfiguration)] = [
            ("spatial", tinyUpsampler()),
            ("temporal", tinyUpsampler { $0.upsamplesSpace = false; $0.upsamplesTime = true; $0.usesRationalResampler = true }),
            ("rational", tinyUpsampler { $0.usesRationalResampler = true; $0.rationalScale = 1.5 }),
            ("both", tinyUpsampler { $0.upsamplesTime = true }),
        ]
        for (prefix, configuration) in arrangements {
            let net = try upsampler(arrays, prefix: prefix, configuration: configuration)
            let head = net.head(latent)
            compare("ltx2-upsampler \(prefix) head", channelsFirst(head), arrays["\(prefix).seam.head"])
            let upsampled = net.upsample(head)
            let seam = try XCTUnwrap(arrays["\(prefix).seam.upsampler"])
            if configuration.upsamplesTime {
                compare("ltx2-upsampler \(prefix) upsampler", channelsFirst(upsampled), seam[0..., 0..., 1...])
            } else {
                let frames = upsampled.reshaped([-1, upsampled.dim(2), upsampled.dim(3), upsampled.dim(4)])
                compare("ltx2-upsampler \(prefix) upsampler", frames.transposed(0, 3, 1, 2), seam)
            }
            let similarity = compare("ltx2-upsampler \(prefix) upsampled", channelsFirst(net(latent)),
                                     arrays["\(prefix).upsampled"])
            print("VALIDATION PARITY ltx2-latent-upsampler-\(prefix): upsampled latent cosine \(similarity)")
        }
    }

    // MARK: - The two-stage recipe

    // LTX-2.5's two-stage distilled recipe through the pipeline's own glue: stage 1 at the half size,
    // the spatial upsampler on its denormalized latent, stage 2 from the renoised upsampled video latent and
    // stage 1's renoised audio latent over the stage-2 sigmas, then the convolutional decode, the diffusion
    // decode, and the temporal upsampler, every noise the reference's.
    func testLTX25TwoStageRecipeMatchesTheReference() throws {
        let arrays = try record("IK_PARITY_LTX2_TWO_STAGE", oracle: "ltx2_two_stage")
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
        let videoStatistics = try NFKMLXLTX2VideoVAE.apply(weights(arrays, prefix: "vae").map { key, value in
            (key, value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value)
        }, to: videoDecoder)
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

        let unguided = NFKMLXLTX2Guidance(textScale: 1, spatioTemporalScale: 0, modalityScale: 1, rescale: 0.7)
        var schedule = NFKMLXFlowMatchConfiguration(baseShift: 1, shiftTerminal: nil, useDynamicShifting: false)
        schedule.rampEndsAtStepFraction = true
        func unpackedAudio(_ packed: MLXArray) -> MLXArray {
            (packed * audioStatistics.std + audioStatistics.mean).reshaped([1, packed.dim(1), 2, 4]).transposed(0, 2, 1, 3)
        }
        let (stage1Video, stage1Audio) = try pipeline.denoise(
            video: try XCTUnwrap(arrays["latents"]), audio: pipeline.normalizedAudio(try XCTUnwrap(arrays["audio_latents"])),
            text: text, audioText: audioText, frames: 2, height: 2, width: 3, audioFrames: 9, frameRate: 24, steps: 3,
            videoGuidance: unguided, audioGuidance: unguided, perturbedBlocks: [1], schedule: schedule,
            sigmas: [1.0, 0.909375, 0.421875])
        compare("ltx2.5-two-stage stage 1 video latent",
                channelsFirst(pipeline.videoLatent(stage1Video, frames: 2, height: 2, width: 3)), arrays["stage1.video"])
        compare("ltx2.5-two-stage stage 1 audio latent", unpackedAudio(stage1Audio), arrays["stage1.audio"])

        let spatial = try upsampler(arrays, prefix: "upsampler", configuration: tinyUpsampler())
        compare("ltx2.5-two-stage upsampled latent",
                channelsFirst(NFKMLXLTX2Pipeline.upsampled(
                    pipeline.videoLatent(stage1Video, frames: 2, height: 2, width: 3), by: spatial)),
                arrays["upsampled"])
        let refined = try pipeline.refine(
            video: stage1Video, audio: stage1Audio, frames: 2, height: 2, width: 3, upsampler: spatial,
            videoNoise: try XCTUnwrap(arrays["stage2.video_noise"]), audioNoise: try XCTUnwrap(arrays["stage2.audio_noise"]),
            text: text, audioText: audioText, audioFrames: 9, frameRate: 24, videoGuidance: unguided,
            audioGuidance: unguided, perturbedBlocks: [1], schedule: schedule)
        XCTAssertEqual([refined.height, refined.width], [4, 6])
        let stage2 = pipeline.videoLatent(refined.video, frames: 2, height: 4, width: 6)
        let latentSimilarity = compare("ltx2.5-two-stage stage 2 video latent", channelsFirst(stage2), arrays["stage2.video"])
        compare("ltx2.5-two-stage stage 2 audio latent", unpackedAudio(refined.audio), arrays["stage2.audio"])

        func clip(_ decoded: MLXArray) -> MLXArray { MLX.clip(decoded / 2 + 0.5, min: 0, max: 1).transposed(0, 1, 4, 2, 3) }
        let videoSimilarity = compare("ltx2.5-two-stage video", clip(pipeline.decodeLatent(stage2)), arrays["video"])
        let waveform = compare("ltx2.5-two-stage waveform", pipeline.decodeAudio(refined.audio).transposed(0, 2, 1),
                               arrays["waveform"])

        let decoder = try diffusionDecoder(arrays, prefix: "decoder")
        let decoderNoise = channelsLast(try XCTUnwrap(arrays["diffusion.noise"]))
        let diffusion = compare("ltx2.5-two-stage diffusion decode",
                                clip(decoder.decodeLatent(stage2, tiling: nil) { _ in decoderNoise }),
                                arrays["diffusion.video"])

        let temporal = try upsampler(arrays, prefix: "temporal", configuration: tinyUpsampler {
            $0.upsamplesSpace = false; $0.upsamplesTime = true; $0.usesRationalResampler = true
        })
        let doubled = NFKMLXLTX2Pipeline.upsampled(stage2, by: temporal)
        compare("ltx2.5-two-stage temporal latent", channelsFirst(doubled), arrays["temporal.upsampled"])
        let temporalVideo = compare("ltx2.5-two-stage temporal video", clip(pipeline.decodeLatent(doubled)),
                                    arrays["temporal.video"])
        print("VALIDATION PARITY ltx2.5-two-stage: stage 2 video latent cosine \(latentSimilarity), clip \(videoSimilarity), "
              + "waveform \(waveform), diffusion decode \(diffusion), temporal clip \(temporalVideo)")
    }

    // MARK: - The released headers and weights

    /// A release component's `config.json` and its `name → shape` inventory from `IK_SHAPES_ROOT/<name>`.
    private func releasedShapes(_ name: String) throws -> (config: URL, shapes: [String: [Int]]) {
        guard let root = config["IK_SHAPES_ROOT"] else { throw XCTSkip("set IK_SHAPES_ROOT (shapes.py)") }
        let directory = URL(fileURLWithPath: root).appendingPathComponent(name)
        let url = directory.appendingPathComponent("shapes.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("run shapes.py Lightricks/LTX-2.5-Diffusers \(directory.path) <component>")
        }
        let shapes = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: [Int]])
        return (directory.appendingPathComponent("config.json"), shapes)
    }

    /// Every parameter of `module` against the released inventory under `prefix`, in the release's layout:
    /// nothing missing, nothing mismatched, and every released tensor under `prefix` accounted for except the
    /// ones `unread` names.
    private func assertMatches(_ label: String, _ module: Module, released: [String: [Int]], prefix: String = "",
                               rename: (String) -> String = { $0 }, unread: (String) -> Bool = { _ in false }) {
        var built = [String: [Int]]()
        for (key, value) in module.parameters().flattened() {
            let shape = value.shape
            built[prefix + rename(key)] = shape.count == 5 ? [shape[0], shape[4], shape[1], shape[2], shape[3]]
                : shape.count == 4 ? [shape[0], shape[3], shape[1], shape[2]] : shape
        }
        let missing = built.keys.filter { released[$0] == nil }.sorted()
        let mismatched = built.keys.filter { released[$0] != nil && released[$0] != built[$0] }.sorted()
        let unaccounted = released.keys.filter { $0.hasPrefix(prefix) && built[$0] == nil && !unread($0) }.sorted()
        print("STRUCTURE \(label): \(built.count) built, \(missing.count) missing, \(mismatched.count) mismatched, "
              + "\(unaccounted.count) unaccounted")
        XCTAssertEqual(missing, [], "\(label): built tensors the release lacks")
        XCTAssertEqual(mismatched.map { "\($0) \(built[$0]!) vs \(released[$0]!)" }, [], "\(label): shapes")
        XCTAssertEqual(unaccounted, [], "\(label): released tensors nothing reads")
    }

    // The diffusion decoder and both upsamplers built from the released configurations and held to the
    // released headers. The decoder's `type_emb`, which the reference never reads, is the one unread tensor.
    func testLTX25DecodeComponentsMatchTheReleasedShapes() throws {
        let decoder = try releasedShapes("ltx2-25-diffusion_decoder")
        let configuration = try NFKMLXLTX2DiffusionDecoders.configuration(fromHuggingFace: decoder.config)
        XCTAssertEqual(configuration, .ltx25)
        assertMatches("ltx2-25-diffusion_decoder", NFKMLXLTX2DiffusionDecoderNet(configuration),
                      released: decoder.shapes.filter { !$0.key.hasPrefix("latents_") }, prefix: "decoder.",
                      unread: NFKMLXLTX2DiffusionDecoders.isUnread)
        XCTAssertEqual(decoder.shapes["latents_mean"], [128])
        XCTAssertEqual(decoder.shapes.count, 408)

        for (name, expected) in [("ltx2-25-latent_upsampler", NFKMLXLTX2LatentUpsamplerConfiguration.ltx25Spatial),
                                 ("ltx2-25-temporal_latent_upsampler", .ltx25Temporal)] {
            let component = try releasedShapes(name)
            let configuration = try NFKMLXLTX2LatentUpsamplers.configuration(fromHuggingFace: component.config)
            XCTAssertEqual(configuration, expected, name)
            assertMatches(name, NFKMLXLTX2LatentUpsamplerNet(configuration), released: component.shapes)
            XCTAssertEqual(component.shapes.count, 72, name)
        }
    }

    private func decodeRelease() throws -> URL {
        guard let path = config["IK_VAL_LTX2_5_DECODE"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_LTX2_5_DECODE (Lightricks/LTX-2.5-Diffusers' diffusion_decoder/, latent_upsampler/, "
                          + "and temporal_latent_upsampler/)")
        }
        return URL(fileURLWithPath: path)
    }

    // The released diffusion decoder and both upsamplers through their public loaders at float32, against
    // diffusers at float32 over the same stored bfloat16: a latent drawn in the decoder's own statistics at
    // the smallest grid its 7×7 first window admits.
    func testLTX25DiffusionDecoderAndUpsamplersMatchTheReferenceOnReleasedWeights() throws {
        let arrays = try record("IK_PARITY_LTX2_REAL_DIFFUSION_DECODER",
                                oracle: "ltx2_real_diffusion_decoder --checkpoint <release>")
        let directory = try decodeRelease()
        let model = try NFKMLXLTX2DiffusionDecoders.load(from: directory.appendingPathComponent("diffusion_decoder"))
        compare("ltx2.5-real latents_mean", model.statistics.mean, arrays["latents_mean"])
        compare("ltx2.5-real latents_std", model.statistics.std, arrays["latents_std"])
        let latent = channelsLast(try XCTUnwrap(arrays["latent"]))
        let context = model.decoder.context(latent)
        let contextSimilarity = compare("ltx2.5-real context", context, arrays["context"])
        let noise = channelsLast(try XCTUnwrap(arrays["noise"]))
        let video = compare("ltx2.5-real diffusion decode",
                            channelsFirst(model.decoder.denoise(context: context, sample: noise, steps: 1)), arrays["video"])

        let small = channelsLast(try XCTUnwrap(arrays["upsampler_latent"]))
        var figures = [String]()
        for name in ["latent_upsampler", "temporal_latent_upsampler"] {
            let net = try NFKMLXLTX2LatentUpsamplers.load(from: directory.appendingPathComponent(name))
            figures.append("\(name) \(compare("ltx2.5-real \(name)", channelsFirst(net(small)), arrays[name]))")
        }
        print("VALIDATION PARITY ltx2.5-real-decode: context cosine \(contextSimilarity), diffusion decode \(video), "
              + figures.joined(separator: ", "))
    }
}

// MARK: - A tiny LTX-2.5 release directory

extension NFKMLXLTX2DecodeTests {

    /// A release directory in the diffusers layout holding the two-stage record's tiny LTX-2.5 stages, its
    /// diffusion decoder, and both upsamplers, in the release's own tensor layout.
    private func tinyRelease(_ arrays: [String: MLXArray]) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("ltx25-tiny-\(UUID().uuidString)")
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
        try write(["_class_name": "AutoencoderKLLTX2Video", "latent_channels": 8,
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
        try write(["_class_name": "LTX2VideoDiffusionDecoderModel", "latent_channels": 8, "decoder_head_dim": 16,
                   "decoder_stage_channels": [128, 64, 32, 32, 32], "decoder_stage_depths": [1, 2, 1, 1, 2],
                   "decoder_stage_kernels": [[3, 3, 3], [3, 5, 5], [3, 3, 3], [3, 5, 5]],
                   "decoder_upsample_channel_reductions": [2, 2, 1, 1], "decoder_stage5_kernel": [5, 7, 7],
                   "decoder_t_emb_dim": 16], "diffusion_decoder/config.json")
        try weights("decoder", "diffusion_decoder/diffusion_pytorch_model.safetensors")
        try write(["_class_name": "LTX2LatentUpsamplerModel", "in_channels": 8, "mid_channels": 32,
                   "num_blocks_per_stage": 2, "dims": 3, "spatial_upsample": true, "temporal_upsample": false,
                   "use_rational_resampler": false], "latent_upsampler/config.json")
        try weights("upsampler", "latent_upsampler/diffusion_pytorch_model.safetensors")
        try write(["_class_name": "LTX2LatentUpsamplerModel", "in_channels": 8, "mid_channels": 32,
                   "num_blocks_per_stage": 2, "dims": 3, "spatial_upsample": false, "temporal_upsample": true,
                   "use_rational_resampler": true], "temporal_latent_upsampler/config.json")
        try weights("temporal", "temporal_latent_upsampler/diffusion_pytorch_model.safetensors")
        return root
    }

    // The public directory factory on a tiny LTX-2.5 release: every decode-side path runs from the
    // generator's switches and from the backend's request keys, each refuses where its folder is missing,
    // and the temporal path doubles the frames and the frame rate.
    func testATinyLTX25ReleaseRunsEveryDecodePath() throws {
        let arrays = try record("IK_PARITY_LTX2_TWO_STAGE", oracle: "ltx2_two_stage")
        let release = try tinyRelease(arrays)
        defer { try? FileManager.default.removeItem(at: release) }

        let generator = try NFKMLXLTX2Generator.generator(directoryURL: release, residency: .staged)
        XCTAssertEqual(generator.variant, .ltx25)
        XCTAssertTrue(generator.hasDiffusionDecoder && generator.hasSpatialUpsampler && generator.hasTemporalUpsampler)
        XCTAssertFalse(generator.decodesWithDiffusionDecoder || generator.refinesUpsampled || generator.upsamplesTemporally)
        XCTAssertEqual(generator.refinementSigmas.map(\.floatValue), [0.909375, 0.725, 0.421875])
        generator.sigmas = [1.0, 0.909375, 0.421875].map { NSNumber(value: $0) }

        generator.refinesUpsampled = true
        var reported = [Int]()
        let refined = try generator.clip(forPrompt: "a red fox", frames: 9, width: 192, height: 128, seed: 3) {
            reported.append($0)
        }
        XCTAssertEqual(refined.video.shape, [9, 128, 192, 3], "stage 1 at 96×64, refined at 192×128")
        XCTAssertEqual(reported, Array(0 ..< 6), "three stage-1 steps, then three refinement steps")

        generator.decodesWithDiffusionDecoder = true
        generator.upsamplesTemporally = true
        let full = try generator.clip(forPrompt: "a red fox", frames: 9, width: 192, height: 128, seed: 3)
        XCTAssertEqual(full.video.shape, [17, 128, 192, 3], "the temporal upsampler doubles the frames")
        XCTAssertEqual(full.audio.shape, refined.audio.shape)
        XCTAssertEqual(generator.outputFrameRate, 48)
        let values = full.video.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
        (generator.refinesUpsampled, generator.decodesWithDiffusionDecoder, generator.upsamplesTemporally) = (false, false, false)

        let backend = generator.inferenceBackend()
        let result = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a red fox"],
            parameters: [NFKParameterFrameCount: 9, NFKParameterWidth: 96, NFKParameterHeight: 64,
                         NFKMLXLTX2ParameterKey.diffusionDecode: true, NFKMLXLTX2ParameterKey.temporalUpsampling: true]))
        let movie = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let wave = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        defer {
            movie.fileURL.map { try? FileManager.default.removeItem(at: $0) }
            wave.fileURL.map { try? FileManager.default.removeItem(at: $0) }
        }
        XCTAssertEqual(movie.dimensions, CGSize(width: 96, height: 64))
        XCTAssertEqual(movie.framesPerSecond, 48)
        XCTAssertEqual(try NFKMLXVideoFile.read(try XCTUnwrap(movie.fileURL)).frames.count, 17)
        XCTAssertFalse(generator.decodesWithDiffusionDecoder || generator.upsamplesTemporally,
                       "a request's switches apply to that request only")

        generator.decodePaths = NFKLTX2DecodePaths(directory: nil)
        XCTAssertFalse(generator.hasDiffusionDecoder)
        generator.decodesWithDiffusionDecoder = true
        XCTAssertThrowsError(try generator.clip(forPrompt: "a red fox", frames: 9, width: 96, height: 64),
                             "a release without diffusion_decoder/ refuses the diffusion decode")
        XCTAssertEqual(NFKMLXLTX2Generator.releaseComponents(for: .ltx25).count,
                       NFKMLXLTX2Generator.releaseComponents(for: .ltx23).count + 3)
    }
}
