//
//  NFKMLXLTX2StageTests.swift
//  InferKitMLXTests
//
//  The LTX-2 stages beside the transformer, each against diffusers at a tiny random configuration:
//  the video decoder, the audio decoder, the vocoder with bandwidth extension, the text connectors,
//  and the transformer's sampling-time paths. Each record comes from
//  `Tools/reference-parity/run_reference.py ltx2_<stage>`; each test skips without its key.
//

import AVFoundation
import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLTX2StageTests: XCTestCase {

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

    /// NDHWC → the reference's NCDHW.
    private func channelsFirst(_ x: MLXArray) -> MLXArray { x.transposed(0, 4, 1, 2, 3) }

    // The release's decoder arrangement, and the residual-upsampler arrangement with reflected spatial
    // padding the class also carries, so neither upsampler path or padding mode is unmeasured.
    func testLTX2VideoDecoderMatchesTheReference() throws {
        let arrays = try record("IK_PARITY_LTX2_VIDEO_VAE", oracle: "ltx2_video_vae")
        let latent = try XCTUnwrap(arrays["latent"]).transposed(0, 2, 3, 4, 1)
        var residual = NFKMLXLTX2VideoVAEConfiguration.tiny
        residual.upsampleResidual = [true, true, true, true]
        residual.reflectsSpatialPadding = true
        for (prefix, configuration) in [("rel", NFKMLXLTX2VideoVAEConfiguration.tiny), ("res", residual)] {
            let decoder = NFKMLXLTX2VideoDecoderNet(configuration)
            let laid = weights(arrays, prefix: prefix).map { key, value in
                (key, value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value)
            }
            try NFKMLXLTX2VideoVAE.apply(laid, to: decoder)

            let causal = configuration.isCausal
            var h = decoder.convIn(latent, causal: causal)
            compare("ltx2-video-vae \(prefix) conv_in", channelsFirst(h), arrays["\(prefix).seam.conv_in"])
            h = decoder.midBlock(h, causal: causal)
            compare("ltx2-video-vae \(prefix) mid", channelsFirst(h), arrays["\(prefix).seam.mid"])
            for (index, block) in decoder.upBlocks.enumerated() {
                h = block(h, causal: causal)
                compare("ltx2-video-vae \(prefix) up\(index)", channelsFirst(h), arrays["\(prefix).seam.up\(index)"])
            }
            compare("ltx2-video-vae \(prefix) norm_out", channelsFirst(decoder.features(latent)),
                    arrays["\(prefix).seam.norm_out"])
            let similarity = compare("ltx2-video-vae \(prefix) decoded", channelsFirst(decoder(latent)),
                                     arrays["\(prefix).decoded"])
            print("VALIDATION PARITY ltx2-video-vae-\(prefix): decoded video cosine \(similarity)")
        }
    }

    func testLTX2AudioDecoderMatchesTheReference() throws {
        let arrays = try record("IK_PARITY_LTX2_AUDIO_VAE", oracle: "ltx2_audio_vae")
        let decoder = NFKMLXLTX2AudioDecoderNet(.tiny)
        let laid = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            return (String(key.dropFirst(3)), value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXLTX2AudioVAE.apply(laid, to: decoder)
        func channelsFirst(_ x: MLXArray) -> MLXArray { x.transposed(0, 3, 1, 2) }

        let latent = try XCTUnwrap(arrays["latent"]).transposed(0, 2, 3, 1)
        var h = decoder.convIn(latent)
        compare("ltx2-audio-vae conv_in", channelsFirst(h), arrays["seam.conv_in"])
        h = decoder.mid.block2(decoder.mid.block1(h))
        compare("ltx2-audio-vae mid", channelsFirst(h), arrays["seam.mid"])
        for level in decoder.up.indices.reversed() {
            let stage = decoder.up[level]
            for block in stage.block { h = block(h) }
            if let upsample = stage.upsample {
                h = upsample(h)
                compare("ltx2-audio-vae up\(level)", channelsFirst(h), arrays["seam.up\(level)"])
            }
        }
        let similarity = compare("ltx2-audio-vae decoded", channelsFirst(decoder(latent)), arrays["decoded"])
        print("VALIDATION PARITY ltx2-audio-vae: decoded mel cosine \(similarity)")
    }

    func testLTX2VocoderMatchesTheReference() throws {
        let arrays = try record("IK_PARITY_LTX2_VOCODER", oracle: "ltx2_vocoder")
        let net = NFKMLXLTX2VocoderNet(.tiny)
        let laid = arrays.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("w::") ? NFKMLXLTX2Vocoder.mapped(String(key.dropFirst(3)), value) : nil
        }
        try NFKMLXWeights.apply(laid, to: net)

        compare("ltx2-vocoder resampler filter", net.resamplerFilter, arrays["resampler_filter"].map { $0.reshaped([-1]) })
        let mel = try XCTUnwrap(arrays["mel"]).transposed(0, 2, 3, 1)
        let base = net.baseWaveform(mel)
        compare("ltx2-vocoder base waveform", base.transposed(0, 2, 1), arrays["seam.vocoder"])
        // The reference's log-mel is `[B · channels, melChannels, frames]`.
        let logMel = net.extensionMel(base)
        compare("ltx2-vocoder log-mel",
                logMel.transposed(0, 3, 2, 1).reshaped([-1, logMel.dim(2), logMel.dim(1)]), arrays["seam.log_mel"])
        let waveform = net(mel)
        let similarity = compare("ltx2-vocoder waveform", waveform.transposed(0, 2, 1), arrays["waveform"])
        print("VALIDATION PARITY ltx2-vocoder: 48 kHz waveform cosine \(similarity)")
    }

    // Row 0 of the record is left-padded and row 1 is full, so both the front alignment of the valid
    // tokens and the register fill are measured.
    func testLTX2ConnectorsMatchTheReference() throws {
        let arrays = try record("IK_PARITY_LTX2_CONNECTORS", oracle: "ltx2_connectors")
        let net = NFKMLXLTX2ConnectorsNet(.tiny)
        try NFKMLXWeights.apply(arrays.compactMap { key, value in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value) : nil
        }, to: net)
        let hidden = try XCTUnwrap(arrays["hidden"])
        let mask = try XCTUnwrap(arrays["mask"])
        let (video, audio) = net(hidden, mask: mask)
        let videoSimilarity = compare("ltx2-connectors video", video, arrays["video"])
        let audioSimilarity = compare("ltx2-connectors audio", audio, arrays["audio"])
        XCTAssertEqual(try XCTUnwrap(arrays["out_mask"]).asArray(Int32.self), [Int32](repeating: 1, count: 16),
                       "the registers leave every text token valid")
        print("VALIDATION PARITY ltx2-connectors: video text cosine \(videoSimilarity), audio \(audioSimilarity)")
    }

    // The four paths the pipeline's default guidance runs that `testLTX2TransformerMatchesTheReference`
    // does not: a text mask, the cross timestep, spatio-temporal guidance, and modality isolation.
    func testLTX2TransformerGuidancePathsMatchTheReference() throws {
        let arrays = try record("IK_PARITY_LTX2_GUIDANCE", oracle: "ltx2_guidance")
        let net = NFKMLXLTX2TransformerNet(.tiny)
        try NFKMLXWeights.apply(arrays.compactMap { key, value in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value) : nil
        }, to: net)
        let video = try XCTUnwrap(arrays["hidden"]), audio = try XCTUnwrap(arrays["audio_hidden"])
        let text = try XCTUnwrap(arrays["encoder"]), audioText = try XCTUnwrap(arrays["audio_encoder"])
        let timestep = try XCTUnwrap(arrays["timestep"]), audioTimestep = try XCTUnwrap(arrays["audio_timestep"])
        let sigma = try XCTUnwrap(arrays["sigma"]), audioSigma = try XCTUnwrap(arrays["audio_sigma"])
        let mask = try XCTUnwrap(arrays["mask"])

        func run(_ name: String, textMask: MLXArray? = nil, crossTimestep: Bool = false, perturbed: Set<Int> = [],
                 isolates: Bool = false) {
            let (predictedVideo, predictedAudio) = net(
                video: video, audio: audio, text: text, audioText: audioText, timestep: timestep,
                audioTimestep: audioTimestep, sigma: sigma, frames: 2, height: 2, width: 3,
                audioFrames: audio.dim(1), textMask: textMask, audioSigma: crossTimestep ? audioSigma : nil,
                crossTimestep: crossTimestep, perturbedBlocks: perturbed, isolatesModalities: isolates)
            let videoSimilarity = compare("ltx2-guidance \(name) video", predictedVideo, arrays["\(name).video"])
            compare("ltx2-guidance \(name) audio", predictedAudio, arrays["\(name).audio"])
            print("VALIDATION PARITY ltx2-guidance-\(name): video velocity cosine \(videoSimilarity)")
        }
        run("mask", textMask: mask)
        run("cross", crossTimestep: true)
        run("stg", perturbed: [1])
        run("isolate", isolates: true)
    }
}

extension NFKMLXLTX2StageTests {

    // Spatio-temporal guidance over part of a batch: the mask `[0, 1, 0]` perturbs the outer examples and leaves
    // the middle one attended, against the all-perturbed control the middle example must differ from.
    func testLTX2PartialSpatioTemporalGuidanceMatchesTheReference() throws {
        let arrays = try record("IK_PARITY_LTX2_STG_MASK", oracle: "ltx2_stg_mask")
        let net = NFKMLXLTX2TransformerNet(.tiny)
        try NFKMLXWeights.apply(arrays.compactMap { key, value in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value) : nil
        }, to: net)
        func run(_ mask: MLXArray?) throws -> (video: MLXArray, audio: MLXArray) {
            net(video: try XCTUnwrap(arrays["hidden"]), audio: try XCTUnwrap(arrays["audio_hidden"]),
                text: try XCTUnwrap(arrays["encoder"]), audioText: try XCTUnwrap(arrays["audio_encoder"]),
                timestep: try XCTUnwrap(arrays["timestep"]), audioTimestep: try XCTUnwrap(arrays["audio_timestep"]),
                sigma: try XCTUnwrap(arrays["sigma"]), frames: 2, height: 2, width: 3, audioFrames: 4,
                perturbedBlocks: [1], perturbationMask: mask)
        }
        let masked = try run(try XCTUnwrap(arrays["mask"]))
        let videoSimilarity = compare("ltx2-stg-mask video", masked.video, arrays["masked.video"])
        compare("ltx2-stg-mask audio", masked.audio, arrays["masked.audio"])
        let all = try run(nil)
        compare("ltx2-stg-all video", all.video, arrays["all.video"])
        let referenceAll = try XCTUnwrap(arrays["all.video"]), referenceMasked = try XCTUnwrap(arrays["masked.video"])
        XCTAssertLessThan(cosine(referenceAll[1], referenceMasked[1]), 0.9999, "the control differs in the attended example")
        XCTAssertGreaterThan(cosine(masked.video[1], referenceMasked[1]), 0.99999)
        print("VALIDATION PARITY ltx2-stg-mask: video velocity cosine \(videoSimilarity)")
    }
}

extension NFKMLXLTX2StageTests {

    // LTX-2.5's duration head: the seconds it predicts from both connector streams and from each alone, and the
    // frame count the pipeline derives from a duration, over the reference's own sweep of windows.
    func testLTX2DurationHeadMatchesTheReference() throws {
        let arrays = try record("IK_PARITY_LTX2_DURATION_HEAD", oracle: "ltx2_duration_head")
        let head = NFKMLXLTX2DurationHeadNet(.tiny)
        try NFKMLXWeights.apply(arrays.compactMap { key, value in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value) : nil
        }, to: head, verifyShapes: true)
        let video = try XCTUnwrap(arrays["video"]), audio = try XCTUnwrap(arrays["audio"])
        let similarity = compare("ltx2-duration both", head(video: video, audio: audio), arrays["both"])
        compare("ltx2-duration video", head(video: video, audio: nil), arrays["video_only"])
        compare("ltx2-duration audio", head(video: nil, audio: audio), arrays["audio_only"])
        let table = try XCTUnwrap(arrays["frames.table"]).asArray(Float.self)
        for row in stride(from: 0, to: table.count, by: 5) {
            let (seconds, fps, low, high) = (table[row], table[row + 1], table[row + 2], table[row + 3])
            // The record stores the sweep at float32; the reference computed it from the doubles these name.
            func exact(_ value: Float) -> Double { Double(String(format: "%.4g", value))! }
            XCTAssertEqual(NFKMLXLTX2DurationHeadNet.frames(forSeconds: exact(seconds), frameRate: exact(fps),
                                                            minimumSeconds: exact(low), maximumSeconds: exact(high)),
                           Int(table[row + 4]), "\(seconds) s at \(fps) fps in [\(low), \(high)]")
        }
        print("VALIDATION PARITY ltx2-duration-head: seconds cosine \(similarity), \(table.count / 5) frame counts")
    }

    // The whole pipeline at tiny configurations of every stage, from the record's starting noise: the
    // text states from token ids, the connectors, three steps of the guided loop (the guided pair,
    // spatio-temporal guidance, modality isolation, and the rescale), both decodes, and the vocoder.
    // A disagreement localizes to the text front end, the loop, or a decode.
    func testLTX2PipelineMatchesTheReference() throws {
        try assertPipelineParity(key: "IK_PARITY_LTX2_PIPELINE", oracle: "ltx2_pipeline", label: "ltx2-pipeline",
                                 transformerSwitches: { _ in }, schedule: .ltxVideoPipeline, sigmas: nil,
                                 videoGuidance: .video, audioGuidance: .audio)
    }

    // LTX-2.5's arrangement as its release ships it: the transformer without the video feed-forward's bias
    // and with the keyframe embedding, the static schedule without the terminal stretch, a caller's own
    // sigmas, and no guidance pass, so each step runs the transformer once.
    func testLTX25PipelineMatchesTheReference() throws {
        let unguided = NFKMLXLTX2Guidance(textScale: 1, spatioTemporalScale: 0, modalityScale: 1, rescale: 0.7)
        var schedule = NFKMLXFlowMatchConfiguration(baseShift: 1, shiftTerminal: nil, useDynamicShifting: false)
        schedule.rampEndsAtStepFraction = true
        try assertPipelineParity(key: "IK_PARITY_LTX2_PIPELINE25", oracle: "ltx2_pipeline25", label: "ltx2.5-pipeline",
                                 transformerSwitches: { $0.feedForwardBias = false; $0.usesKeyframeEmbedding = true },
                                 schedule: schedule, sigmas: [1.0, 0.909375, 0.421875], videoGuidance: unguided,
                                 audioGuidance: unguided)
    }

    private func assertPipelineParity(key: String, oracle: String, label: String,
                                      transformerSwitches: (inout NFKMLXLTX2Configuration) -> Void,
                                      schedule: NFKMLXFlowMatchConfiguration, sigmas: [Float]?,
                                      videoGuidance: NFKMLXLTX2Guidance, audioGuidance: NFKMLXLTX2Guidance) throws {
        let arrays = try record(key, oracle: oracle)
        let guided = videoGuidance.textScale > 1 || audioGuidance.textScale > 1

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
        let embeds = try XCTUnwrap(arrays["embeds"])
        let length = 8
        var videoStreams = [MLXArray](), audioStreams = [MLXArray]()
        for row in 0 ..< 2 {
            let valid = (0 ..< length).filter { mask[row * length + $0] != 0 }
            let states = NFKMLXLTX2Pipeline.textStates(gemma, tokens: valid.map { Int(ids[row * length + $0]) })
            compare("\(label) text states row \(row)", states,
                    embeds[row].take(MLXArray(valid.map(Int32.init)), axis: 0).expandedDimensions(axis: 0))
            let (video, audio) = connectors(validStates: states, length: length)
            videoStreams.append(video)
            audioStreams.append(audio)
        }
        let text = concatenated(videoStreams, axis: 0), audioText = concatenated(audioStreams, axis: 0)
        compare("\(label) connector video", guided ? text : text[1 ..< 2], arrays["seam.connector_video"])
        compare("\(label) connector audio", guided ? audioText : audioText[1 ..< 2], arrays["seam.connector_audio"])

        var transformerConfiguration = NFKMLXLTX2Configuration.tiny
        transformerConfiguration.audioInChannels = 8
        transformerConfiguration.audioOutChannels = 8
        transformerSwitches(&transformerConfiguration)
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

        let grid = NFKMLXLTX2Pipeline.grid(frames: 9, width: 96, height: 64, frameRate: 24, video: .tiny,
                                           audio: .tiny)
        XCTAssertEqual([grid.frames, grid.height, grid.width, grid.audioFrames], [2, 2, 3, 9])
        var scheduler = NFKMLXFlowMatchScheduler(schedule)
        if let sigmas {
            scheduler.setTimesteps(sigmas: sigmas, sequenceLength: 12)
        } else {
            scheduler.setTimesteps(3, sequenceLength: 12)
        }
        compare("\(label) sigmas", MLXArray(scheduler.sigmas), arrays["sigmas"])

        // Unguided, the reference conditions on the positive prompt alone.
        let (video, audio) = try pipeline.denoise(
            video: try XCTUnwrap(arrays["latents"]), audio: pipeline.normalizedAudio(try XCTUnwrap(arrays["audio_latents"])),
            text: guided ? text : text[1 ..< 2], audioText: guided ? audioText : audioText[1 ..< 2], frames: grid.frames,
            height: grid.height, width: grid.width, audioFrames: grid.audioFrames, frameRate: 24, steps: 3,
            videoGuidance: videoGuidance, audioGuidance: audioGuidance, perturbedBlocks: [1], schedule: schedule,
            sigmas: sigmas)
        let unpackedVideo = video.reshaped([1, 2, 2, 3, 8]) * videoStatistics.std + videoStatistics.mean
        let videoLatent = compare("\(label) final video latent", channelsFirst(unpackedVideo),
                                  arrays["seam.video_latent"])
        let unpackedAudio = (audio * audioStatistics.std + audioStatistics.mean).reshaped([1, 9, 2, 4])
        compare("\(label) final audio latent", unpackedAudio.transposed(0, 2, 1, 3), arrays["seam.audio_latent"])

        let clip = clip(pipeline.decodeVideo(video, frames: 2, height: 2, width: 3) / 2 + 0.5, min: 0, max: 1)
        let videoSimilarity = compare("\(label) video", clip.transposed(0, 1, 4, 2, 3), arrays["video"])
        compare("\(label) mel", pipeline.decodeMel(audio).transposed(0, 3, 1, 2), arrays["seam.mel"])
        let audioSimilarity = compare("\(label) audio", pipeline.decodeAudio(audio).transposed(0, 2, 1),
                                      arrays["audio"])
        print("VALIDATION PARITY \(label): final video latent cosine \(videoLatent), clip \(videoSimilarity), "
              + "waveform \(audioSimilarity)")
    }
}

// MARK: - A tiny release directory

extension NFKMLXLTX2StageTests {

    /// A release directory in the diffusers layout holding the pipeline record's tiny stages, written in
    /// the release's own tensor layout so the factories' readers and loaders run as they do on a download.
    private func tinyRelease(_ arrays: [String: MLXArray]) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("ltx2-tiny-\(UUID().uuidString)")
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
                   "max_image_seq_len": 4096, "max_shift": 2.05, "shift": 1.0, "shift_terminal": 0.1,
                   "use_dynamic_shifting": true], "scheduler/scheduler_config.json")
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
                   "caption_channels": 14], "transformer/config.json")
        try weights("transformer", "transformer/diffusion_pytorch_model.safetensors")
        try write(["_class_name": "AutoencoderKLLTX2Video", "latent_channels": 8,
                   "block_out_channels": [8, 16, 32, 32], "layers_per_block": [1, 1, 1, 1, 1],
                   "downsample_type": ["spatial", "temporal", "spatiotemporal", "spatiotemporal"],
                   "spatio_temporal_scaling": [true, true, true, true], "encoder_causal": true,
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
        return root
    }

    // A release that ships a duration head: a clip without a frame count takes the length the head predicts
    // for its prompt, which is the head's own prediction over the prompt's connector streams.
    func testAReleaseWithADurationHeadPredictsTheClipLength() throws {
        let arrays = try record("IK_PARITY_LTX2_PIPELINE", oracle: "ltx2_pipeline")
        let headRecord = try record("IK_PARITY_LTX2_DURATION_HEAD", oracle: "ltx2_duration_head")
        let release = try tinyRelease(arrays)
        defer { try? FileManager.default.removeItem(at: release) }
        let headDirectory = release.appendingPathComponent("duration_head")
        try FileManager.default.createDirectory(at: headDirectory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: [
            "_class_name": "LTX2DurationHead", "video_cross_attention_dim": 32, "audio_cross_attention_dim": 16,
            "pooler_hidden_dim": 16, "num_queries": 2, "num_pooler_heads": 2, "mlp_hidden_dim": 16,
        ]).write(to: headDirectory.appendingPathComponent("config.json"))
        try MLX.save(arrays: headRecord.reduce(into: [String: MLXArray]()) { stored, entry in
            if entry.key.hasPrefix("w::") { stored[String(entry.key.dropFirst(3))] = entry.value }
        }, url: headDirectory.appendingPathComponent("diffusion_pytorch_model.safetensors"))

        let generator = try NFKMLXLTX2Generator.generator(directoryURL: release, residency: .resident)
        XCTAssertTrue(generator.predictsDuration)
        generator.frameRate = 24
        generator.minimumSeconds = 0.3
        generator.maximumSeconds = 2
        let text = try generator.connectorStates(generator.tokens("a red fox"))
        let head = NFKMLXLTX2DurationHeadNet(.tiny)
        try NFKMLXLTX2DurationHead.loadWeights(into: head, from: headDirectory)
        let seconds = Double(head(video: text.video, audio: text.audio).item(Float.self))
        let expected = NFKMLXLTX2DurationHeadNet.frames(forSeconds: seconds, frameRate: 24, minimumSeconds: 0.3,
                                                        maximumSeconds: 2)
        XCTAssertEqual(try generator.predictedFrameCount(forPrompt: "a red fox").intValue, expected)
        generator.steps = 1
        XCTAssertEqual(try generator.clip(forPrompt: "a red fox", width: 96, height: 64, seed: 4).video.dim(0), expected)
        XCTAssertEqual(try generator.clip(forPrompt: "a red fox", frames: 9, width: 96, height: 64, seed: 4).video.dim(0),
                       9, "a given frame count wins")

        let headless = try tinyRelease(arrays)
        defer { try? FileManager.default.removeItem(at: headless) }
        let plain = try NFKMLXLTX2Generator.generator(directoryURL: headless, residency: .staged)
        XCTAssertFalse(plain.predictsDuration)
        XCTAssertEqual(try plain.predictedFrameCount(forPrompt: "a red fox").intValue, 121)
    }

    // LTX-2.5's full transformer: the variant factory reads `transformer_full/`, samples it guided over the
    // dynamically shifted schedule the distilled release's static scheduler turns off, refuses the directory
    // under another release's name, and downloads the full folder in place of the distilled one.
    func testTheFullTransformerVariantReadsItsOwnFolder() throws {
        let arrays = try record("IK_PARITY_LTX2_PIPELINE25", oracle: "ltx2_pipeline25")
        let release = try tinyRelease(arrays)
        defer { try? FileManager.default.removeItem(at: release) }
        let transformerConfig = release.appendingPathComponent("transformer/config.json")
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: transformerConfig)) as? [String: Any])
        json["ff_bias"] = false
        json["use_keyframes_abs_pos_embedding"] = true
        try JSONSerialization.data(withJSONObject: json).write(to: transformerConfig)
        try JSONSerialization.data(withJSONObject: [
            "_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 1024, "base_shift": 0.95,
            "max_image_seq_len": 4096, "max_shift": 2.05, "shift": 1.0, "use_dynamic_shifting": false,
        ]).write(to: release.appendingPathComponent("scheduler/scheduler_config.json"))
        try FileManager.default.copyItem(at: release.appendingPathComponent("transformer"),
                                         to: release.appendingPathComponent("transformer_full"))

        let distilled = try NFKMLXLTX2Generator.generator(directoryURL: release, residency: .staged)
        XCTAssertEqual(distilled.variant, .ltx25)
        XCTAssertNotNil(distilled.sigmas)
        XCTAssertFalse(distilled.schedule.useDynamicShifting)

        let full = try NFKMLXLTX2Generator.generator(directoryURL: release, variant: .ltx25Full, residency: .staged)
        XCTAssertEqual(full.variant, .ltx25Full)
        XCTAssertEqual(full.modelName, "ltx-2.5-full")
        XCTAssertNil(full.sigmas, "the full transformer samples the schedule rather than the distilled sigmas")
        XCTAssertEqual(full.guidance, 3)
        XCTAssertEqual(full.spatioTemporalGuidance, 1)
        XCTAssertTrue(full.schedule.useDynamicShifting)
        XCTAssertEqual(full.schedule.baseShift, 0.95)
        XCTAssertEqual(full.schedule.shiftTerminal, 0.1)
        full.steps = 1
        XCTAssertEqual(try full.clip(forPrompt: "a red fox", frames: 9, width: 96, height: 64, seed: 4).video.shape,
                       [9, 64, 96, 3])

        XCTAssertThrowsError(try NFKMLXLTX2Generator.generator(directoryURL: release, variant: .ltx23, residency: .staged))
        let folders = NFKMLXLTX2Generator.releaseComponents(for: .ltx25Full).flatMap(\.required)
        XCTAssertTrue(folders.contains("transformer_full/config.json"))
        XCTAssertFalse(folders.contains("transformer/config.json"))
        let network = try NFKMLXLTX2Generator.network(directoryURL: release, variant: .ltx25Full)
        XCTAssertTrue(network.config.usesKeyframeEmbedding)
    }

    // The customization path on a release written to disk: the data adapter encodes a clip, its soundtrack,
    // and its prompt, the training builder reads the release, a short fine-tune moves only the adapters, the saved adapters reapply through the generator's own factory and
    // give the trained transformer's velocities, and an 8-bit build trains its adapters over the packed base.
    func testAFineTuneSavesAdaptersTheGeneratorApplies() throws {
        let arrays = try record("IK_PARITY_LTX2_PIPELINE", oracle: "ltx2_pipeline")
        let release = try tinyRelease(arrays)
        defer { try? FileManager.default.removeItem(at: release) }
        let generator = try NFKMLXLTX2Generator.generator(directoryURL: release, residency: .resident)
        let example = try generator.trainingExample(
            video: MLXRandom.uniform(low: 0, high: 1, [9, 64, 96, 3], key: MLXRandom.key(5)), frameRate: 24,
            audio: MLXRandom.uniform(low: -0.5, high: 0.5, [9000, 1], key: MLXRandom.key(6)), sampleRate: 24000,
            prompt: "a red fox")
        XCTAssertEqual(example.video.shape, [2 * 2 * 3, 8], "two latent frames of 2×3, eight channels")
        XCTAssertEqual(example.audio.shape, [10, 8], "a quarter of the 38 mel frames, packed two by four")
        XCTAssertEqual(example.text.shape, [1024, 32])
        XCTAssertEqual(example.audioText.shape, [1024, 16])
        XCTAssertEqual([example.frames, example.height, example.width], [2, 2, 3])
        XCTAssertThrowsError(try generator.trainingExample(
            video: MLXArray.zeros([8, 64, 96, 3]), frameRate: 24, audio: MLXArray.zeros([100, 2]), sampleRate: 16000,
            prompt: "x"), "eight frames is not 8k + 1")

        let net = try NFKMLXLTX2Generator.network(directoryURL: release)
        let frozen = net.projIn.weight
        let losses = try NFKMLXLTX2Generator.fineTune(net, examples: { _ in [example] }, trainable: .lora(rank: 4, alpha: 4),
                                                      learningRate: 1e-2, steps: 3, seed: 1)
        XCTAssertEqual(losses.count, 3)
        XCTAssertTrue(losses.allSatisfy(\.isFinite))
        XCTAssertTrue(arrayEqual(net.projIn.weight, frozen).item(Bool.self), "the base stays frozen")
        let adapters = NFKMLXLoRA.adapterArrays(of: net)
        XCTAssertEqual(adapters.count, 2 * 2 * 6 * 4)
        XCTAssertTrue(adapters.filter { $0.key.hasSuffix(".lora_b") }.contains { $0.value.abs().max().item(Float.self) > 0 },
                      "the adapters moved")

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ltx2-lora-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXLTX2Generator.saveAdapters(of: net, to: url)
        try generator.applyAdapters(at: url)
        net.train(false)
        let inputs = NFKMLXLTX2Objective.inputs(
            video: example.video[.newAxis], audio: example.audio[.newAxis], sigma: MLXArray([Float(0.6)]),
            audioSigma: MLXArray([Float(0.4)]), noise: MLXRandom.normal([1, 12, 8], key: MLXRandom.key(9)),
            audioNoise: MLXRandom.normal([1, 10, 8], key: MLXRandom.key(10)), conditioned: MLXArray([Float(1)]),
            tokensPerFrame: 6)
        func velocities(_ transformer: NFKMLXLTX2TransformerNet) -> (video: MLXArray, audio: MLXArray) {
            NFKMLXLTX2Objective.predictions(transformer, inputs: inputs, text: example.text[.newAxis],
                                            audioText: example.audioText[.newAxis], frames: 2, height: 2, width: 3,
                                            frameRate: 24)
        }
        let trained = velocities(net)
        let reloaded = try generator.withTransformer { velocities($0) }
        XCTAssertEqual(cosine(reloaded.video, trained.video), 1, accuracy: 1e-12)
        XCTAssertEqual(cosine(reloaded.audio, trained.audio), 1, accuracy: 1e-12)

        let packed = try NFKMLXLTX2Generator.network(directoryURL: release, quantizationBits: 8, quantizationGroupSize: 32)
        XCTAssertTrue(packed.transformerBlocks[0].attn1.toQ is QuantizedLinear, "the blocks load packed")
        XCTAssertFalse(packed.projIn is QuantizedLinear, "the layers outside the blocks keep the stored type")
        let packedLosses = try NFKMLXLTX2Generator.fineTune(packed, examples: { _ in [example] },
                                                            trainable: .lora(rank: 4, alpha: 4), steps: 2, seed: 1)
        XCTAssertTrue(packedLosses.allSatisfy(\.isFinite))
        XCTAssertTrue(packed.transformerBlocks[0].attn1.toQ is NFKMLXQuantizedLoRALinear, "QLoRA over the packed base")
    }

    // The public directory factory on a release written to disk: the clip and its soundtrack come back at
    // their shapes, a transformer whose blocks stream from the release gives the held one's velocities, the
    // backend writes the movie with its audio track and the soundtrack beside it, and the registry refuses
    // the directory under the other release's name.
    func testATinyReleaseGeneratesAClipWithItsSoundtrack() throws {
        let arrays = try record("IK_PARITY_LTX2_PIPELINE", oracle: "ltx2_pipeline")
        let release = try tinyRelease(arrays)
        defer { try? FileManager.default.removeItem(at: release) }

        let resident = try NFKMLXLTX2Generator.generator(directoryURL: release, residency: .resident)
        XCTAssertEqual(resident.variant, .ltx23)
        XCTAssertTrue(resident.holdsStagesResident)
        XCTAssertEqual(resident.perturbedBlocks, [1], "spatio-temporal guidance perturbs the last block of two")
        resident.steps = 2
        let (video, audio) = try resident.clip(forPrompt: "a red fox", frames: 9, width: 96, height: 64, seed: 4)
        XCTAssertEqual(video.shape, [9, 64, 96, 3])
        XCTAssertEqual(audio.shape, [990, 2])

        // Both blocks read from the release in their turn give the held transformer's velocities.
        let configuration = try NFKMLXLTX2TransformerNet.configuration(
            fromHuggingFace: release.appendingPathComponent("transformer/config.json"))
        let held = NFKMLXLTX2TransformerNet(configuration), streamed = NFKMLXLTX2TransformerNet(configuration)
        try NFKMLXLTX2TransformerNet.loadWeights(into: held, from: release.appendingPathComponent("transformer"),
                                                 precision: .checkpoint, streamed: [])
        try NFKMLXLTX2TransformerNet.loadWeights(into: streamed, from: release.appendingPathComponent("transformer"),
                                                 precision: .checkpoint, streamed: [0, 1])
        let text = MLXRandom.normal([1, 8, 32], key: MLXRandom.key(1))
        let audioText = MLXRandom.normal([1, 8, 16], key: MLXRandom.key(2))
        let pass = NFKMLXLTX2Pass(video: MLXRandom.normal([1, 12, 8], key: MLXRandom.key(3)),
                                  audio: MLXRandom.normal([1, 9, 8], key: MLXRandom.key(4)), text: text,
                                  audioText: audioText, timestep: MLXArray([Float(700)]),
                                  audioTimestep: MLXArray([Float(700)]), sigma: MLXArray([Float(700)]))
        let perturbedPass = NFKMLXLTX2Pass(video: pass.video, audio: pass.audio, text: text, audioText: audioText,
                                           timestep: pass.timestep, audioTimestep: pass.timestep, sigma: pass.sigma,
                                           perturbedBlocks: [1])
        let heldOutputs = held.predictions([pass, perturbedPass], frames: 2, height: 2, width: 3, audioFrames: 9,
                                           crossTimestep: true)
        let streamedOutputs = streamed.predictions([pass, perturbedPass], frames: 2, height: 2, width: 3,
                                                   audioFrames: 9, crossTimestep: true)
        try streamed.verifyStream()
        for (heldOutput, streamedOutput) in zip(heldOutputs, streamedOutputs) {
            XCTAssertEqual(cosine(streamedOutput.video, heldOutput.video), 1, accuracy: 1e-12)
            XCTAssertEqual(cosine(streamedOutput.audio, heldOutput.audio), 1, accuracy: 1e-12)
        }

        let backend = resident.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, "ltx-2.3")
        let result = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a red fox"],
            parameters: [NFKParameterFrameCount: 9, NFKParameterWidth: 96, NFKParameterHeight: 64, NFKParameterSteps: 1,
                         NFKMLXLTX2ParameterKey.spatioTemporalGuidanceScale: 0]))
        let movie = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let wave = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        defer {
            movie.fileURL.map { try? FileManager.default.removeItem(at: $0) }
            wave.fileURL.map { try? FileManager.default.removeItem(at: $0) }
        }
        XCTAssertEqual(movie.dimensions, CGSize(width: 96, height: 64))
        XCTAssertEqual(try NFKMLXVideoFile.read(try XCTUnwrap(movie.fileURL)).frames.count, 9)
        let movieURL = try XCTUnwrap(movie.fileURL)
        let audioTracks = try NFKMLXVideoFile.blocking {
            try await AVURLAsset(url: movieURL).loadTracks(withMediaType: .audio).count
        }
        XCTAssertEqual(audioTracks, 1, "the movie carries the soundtrack")
        XCTAssertEqual(wave.sampleRate, 48000)
        XCTAssertEqual(wave.channelCount, 2)
        XCTAssertEqual(resident.steps, 2, "the request's steps apply to that request only")
        XCTAssertEqual(resident.spatioTemporalGuidance, 1)

        NFKMLXLTX2Generator.register()
        XCTAssertThrowsError(try NFKMLXModelRegistry.backend(named: "ltx-2.5", weightsURL: release),
                             "the LTX-2.5 name refuses an LTX-2.3 directory")
        XCTAssertThrowsError(try NFKMLXModelRegistry.backend(named: "ltx-2.3", weightsURL: nil))
    }
}

// MARK: - The released headers

extension NFKMLXLTX2StageTests {

    /// A release component's `config.json` and its `name → shape` inventory from `IK_SHAPES_ROOT/<name>`.
    private func releasedShapes(_ name: String) throws -> (config: URL, shapes: [String: [Int]]) {
        guard let root = config["IK_SHAPES_ROOT"] else { throw XCTSkip("set IK_SHAPES_ROOT (shapes.py)") }
        let directory = URL(fileURLWithPath: root).appendingPathComponent(name)
        let url = directory.appendingPathComponent("shapes.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("run shapes.py diffusers/LTX-2.3-Diffusers \(directory.path) <component>")
        }
        let shapes = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: [Int]])
        return (directory.appendingPathComponent("config.json"), shapes)
    }

    /// Every parameter of `module` against the released inventory under `prefix`, in the release's layout:
    /// nothing missing, nothing mismatched, and every released tensor under `prefix` accounted for except
    /// the ones `unread` names.
    private func assertMatches(_ label: String, _ module: Module, released: [String: [Int]], prefix: String = "",
                               unread: (String) -> Bool = { _ in false },
                               layout: (String, [Int]) -> [Int]) {
        var built = [String: [Int]]()
        for (key, value) in module.parameters().flattened() { built[prefix + key] = layout(prefix + key, value.shape) }
        let missing = built.keys.filter { released[$0] == nil }.sorted()
        let mismatched = built.keys.filter { released[$0] != nil && released[$0] != built[$0] }.sorted()
        let unaccounted = released.keys.filter { $0.hasPrefix(prefix) && built[$0] == nil && !unread($0) }.sorted()
        print("STRUCTURE \(label): \(built.count) built, \(missing.count) missing, \(mismatched.count) mismatched, "
              + "\(unaccounted.count) unaccounted")
        XCTAssertEqual(missing, [], "\(label): built tensors the release lacks")
        XCTAssertEqual(mismatched.map { "\($0) \(built[$0]!) vs \(released[$0]!)" }, [], "\(label): shapes")
        XCTAssertEqual(unaccounted, [], "\(label): released tensors nothing reads")
    }

    // The four LTX-2.3 stages beside the transformer, built from the released configurations and held to
    // the released headers. The autoencoders' encoders are not built: generation decodes only.
    func testLTX2StagesMatchTheReleasedShapes() throws {
        let vae = try releasedShapes("ltx2-23-vae")
        let videoConfiguration = try NFKMLXLTX2VideoVAE.configuration(fromHuggingFace: vae.config)
        assertMatches("ltx2-23-vae", NFKMLXLTX2VideoDecoderNet(videoConfiguration), released: vae.shapes,
                      prefix: "decoder.") { _, shape in
            shape.count == 5 ? [shape[0], shape[4], shape[1], shape[2], shape[3]] : shape
        }
        XCTAssertEqual(vae.shapes["latents_mean"], [videoConfiguration.latentChannels])

        let audioVAE = try releasedShapes("ltx2-23-audio_vae")
        let audioConfiguration = try NFKMLXLTX2AudioVAE.configuration(fromHuggingFace: audioVAE.config)
        assertMatches("ltx2-23-audio_vae", NFKMLXLTX2AudioDecoderNet(audioConfiguration), released: audioVAE.shapes,
                      prefix: "decoder.") { _, shape in
            shape.count == 4 ? [shape[0], shape[3], shape[1], shape[2]] : shape
        }
        XCTAssertEqual(audioVAE.shapes["latents_std"],
                       [audioConfiguration.latentChannels * audioConfiguration.latentMelBins])

        let vocoder = try releasedShapes("ltx2-23-vocoder")
        let vocoderConfiguration = try NFKMLXLTX2Vocoder.configuration(fromHuggingFace: vocoder.config)
        assertMatches("ltx2-23-vocoder", NFKMLXLTX2VocoderNet(vocoderConfiguration), released: vocoder.shapes,
                      unread: { $0.hasSuffix("inverse_basis") }) { key, shape in
            guard shape.count == 3, !key.hasSuffix(".filter") else { return shape }
            // MLX holds a convolution `[out, kernel, in]`; the release a convolution `[out, in, kernel]` and a
            // transposed convolution `[in, out, kernel]`.
            return key.contains(".upsamplers.") ? [shape[2], shape[0], shape[1]] : [shape[0], shape[2], shape[1]]
        }

        let connectors = try releasedShapes("ltx2-23-connectors")
        assertMatches("ltx2-23-connectors",
                      NFKMLXLTX2ConnectorsNet(try NFKMLXLTX2Connectors.configuration(fromHuggingFace: connectors.config)),
                      released: connectors.shapes) { _, shape in shape }
    }
}

// MARK: - The released weights

extension NFKMLXLTX2StageTests {

    private func release() throws -> URL {
        guard let path = config["IK_VAL_LTX2_3"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_LTX2_3 (diffusers/LTX-2.3-Diffusers without its transformer and text encoder weights)")
        }
        return URL(fileURLWithPath: path)
    }

    // The released LTX-2.3 decoders and vocoder through their public loaders at float32, against
    // diffusers at float32 over the same stored bfloat16 weights: the inputs are drawn in the latents' own
    // distribution, and the vocoder vocodes the audio decoder's own mel spectrogram.
    func testLTX2DecodersAndVocoderMatchTheReferenceOnReleasedWeights() throws {
        let arrays = try record("IK_PARITY_LTX2_REAL_DECODERS", oracle: "ltx2_real_decoders --checkpoint <release>")
        let directory = try release()

        let videoDecoder = NFKMLXLTX2VideoDecoderNet(try NFKMLXLTX2VideoVAE.configuration(
            fromHuggingFace: directory.appendingPathComponent("vae/config.json")))
        try NFKMLXLTX2VideoVAE.loadWeights(into: videoDecoder, from: directory.appendingPathComponent("vae"))
        let latent = try XCTUnwrap(arrays["video_latent"]).transposed(0, 2, 3, 4, 1)
        let video = compare("ltx2-23-real video decode", channelsFirst(videoDecoder(latent)), arrays["video"])

        let audioDecoder = NFKMLXLTX2AudioDecoderNet(try NFKMLXLTX2AudioVAE.configuration(
            fromHuggingFace: directory.appendingPathComponent("audio_vae/config.json")))
        try NFKMLXLTX2AudioVAE.loadWeights(into: audioDecoder, from: directory.appendingPathComponent("audio_vae"))
        let mel = audioDecoder(try XCTUnwrap(arrays["audio_latent"]).transposed(0, 2, 3, 1))
        let melSimilarity = compare("ltx2-23-real mel", mel.transposed(0, 3, 1, 2), arrays["mel"])

        let vocoder = NFKMLXLTX2VocoderNet(try NFKMLXLTX2Vocoder.configuration(
            fromHuggingFace: directory.appendingPathComponent("vocoder/config.json")))
        try NFKMLXLTX2Vocoder.loadWeights(into: vocoder, from: directory.appendingPathComponent("vocoder"))
        let referenceMel = try XCTUnwrap(arrays["mel"]).transposed(0, 2, 3, 1)
        let waveform = compare("ltx2-23-real waveform", vocoder(referenceMel).transposed(0, 2, 1), arrays["waveform"])
        print("VALIDATION PARITY ltx2-23-real: video decode cosine \(video), mel \(melSimilarity), waveform \(waveform)")
    }

    // The released LTX-2.3 connectors at float32 over 128 left-padded tokens, 32 of them real.
    func testLTX2ConnectorsMatchTheReferenceOnReleasedWeights() throws {
        let arrays = try record("IK_PARITY_LTX2_REAL_CONNECTORS", oracle: "ltx2_real_connectors --checkpoint <release>")
        let directory = try release()
        let net = NFKMLXLTX2ConnectorsNet(try NFKMLXLTX2Connectors.configuration(
            fromHuggingFace: directory.appendingPathComponent("connectors/config.json")))
        try NFKMLXLTX2Connectors.loadWeights(into: net, from: directory.appendingPathComponent("connectors"),
                                             precision: .float32)
        let (video, audio) = net(validStates: try XCTUnwrap(arrays["valid_hidden"]), length: 128)
        let videoSimilarity = compare("ltx2-23-real connectors video", video, arrays["video"])
        let audioSimilarity = compare("ltx2-23-real connectors audio", audio, arrays["audio"])
        print("VALIDATION PARITY ltx2-23-real-connectors: video text cosine \(videoSimilarity), audio \(audioSimilarity)")
    }

    // LTX-2.5's released prompt enhancer against diffusers' own `enhance_prompt`. The input ids equal the
    // chat template's (the reference left-pads to a multiple of eight under a zero attention mask, which changes
    // nothing it computes), and the cleaning matches on every character it maps. At float32, teacher-forced
    // along the reference's float32 reply, every step's processed scores (the repeated 5-grams at -inf) agree
    // with the reference's top five and the choice is the reference's. The bfloat16 greedy reply departs from
    // the float32 one no further than the reference's own bfloat16 reply does: torch's bfloat16 run departs at
    // a float32 gap past one bfloat16 spacing, so that gap is the bar.
    func testLTX25PromptEnhancerMatchesTheReferenceOnReleasedWeights() throws {
        let arrays = try record("IK_PARITY_LTX2_REAL_PROMPT_ENHANCER",
                                oracle: "ltx2_real_prompt_enhancer --checkpoint <prompt enhancer parent>")
        guard let path = config["IK_VAL_LTX2_5_PROMPT_ENHANCER"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_LTX2_5_PROMPT_ENHANCER to the directory holding prompt_enhancer/ and processor/")
        }
        func text(_ key: String) throws -> String {
            String(decoding: try XCTUnwrap(arrays[key]).asArray(UInt8.self), as: UTF8.self)
        }
        func ints(_ key: String) throws -> [Int] { try XCTUnwrap(arrays[key]).asArray(Int32.self).map(Int.init) }
        for index in 0 ..< 4 {
            XCTAssertEqual(NFKMLXLTX2PromptEnhancer.cleaned(try text("cleaning.\(index).in")),
                           try text("cleaning.\(index).out"), "cleaning case \(index)")
        }
        let requests = [("fox", "a red fox walking through a snowy forest at dawn"),
                        ("cafe", "Two friends laugh over coffee in a busy cafe; rain taps on the window.")]
        let root = URL(fileURLWithPath: path)

        // The f32 reference's first departure from itself at bf16, and the float32 gap it departs at.
        func departure(_ generated: [Int], name: String) throws -> (step: Int, gap: Float)? {
            let reference = try ints("\(name).f32.reply")
            let step = zip(generated, reference).prefix { $0 == $1 }.count
            guard step < min(generated.count, reference.count) else { return nil }
            let ids = try XCTUnwrap(arrays["\(name).f32.step_top_ids"])[step].asArray(Int32.self).map(Int.init)
            let scores = try XCTUnwrap(arrays["\(name).f32.step_top_scores"])[step].asArray(Float.self)
            let index = try XCTUnwrap(ids.firstIndex(of: generated[step]),
                                      "\(name) step \(step): \(generated[step]) is not among the reference's top five")
            return (step, scores[0] - scores[index])
        }

        // Float32, teacher-forced: one pass over the prompt and the reference's whole float32 reply.
        do {
            // The prompt's ids need only the tokenizer and template; the float32 decoder is held alone.
            let processor = root.appendingPathComponent("processor")
            let enhancer = NFKMLXLTX2PromptEnhancer(
                logits: { _ in MLXArray(0) }, tokenizer: try XCTUnwrap(NFKMLXGemmaTokenizer(directoryURL: processor)),
                template: try String(contentsOf: processor.appendingPathComponent("chat_template.jinja"), encoding: .utf8))
            // At float32 the decoder outgrows a 32 GB machine's budget, so its layers stream where they must;
            // one teacher-forced pass reads each once.
            let decoder = try NFKMLXGemmaLanguage.textDecoder(
                directoryURL: root.appendingPathComponent("prompt_enhancer"), precision: .float32,
                residency: .automatic, streams: true)
            for (name, request) in requests {
                let ids = try enhancer.tokens(for: request)
                let padding = try XCTUnwrap(arrays["\(name).pad"]).item(Int.self)
                XCTAssertEqual(ids, Array(try ints("\(name).input_ids").dropFirst(padding)), "\(name) prompt ids")
                let reply = try ints("\(name).f32.reply")
                let sequence = ids + reply
                let logits = decoder.logits(MLXArray(sequence.map(Int32.init)).reshaped([1, sequence.count]))[0]
                let topIds = try XCTUnwrap(arrays["\(name).f32.step_top_ids"])
                let topScores = try XCTUnwrap(arrays["\(name).f32.step_top_scores"])
                var worst: Float = 0, mismatched = 0
                for step in 0 ..< reply.count {
                    let banned = NFKMLXSeq2SeqDecoder.repeatedNgramCompletions(
                        Array(sequence.prefix(ids.count + step)), size: NFKMLXLTX2PromptEnhancer.repeatedNgramSize)
                    let row = logits[ids.count + step - 1].asType(.float32)
                    let referenceIds = topIds[step].asArray(Int32.self).map(Int.init)
                    let referenceScores = topScores[step].asArray(Float.self)
                    let mine = row.take(MLXArray(referenceIds.map(Int32.init))).asArray(Float.self)
                    for (k, id) in referenceIds.enumerated() where !banned.contains(id) && referenceScores[k].isFinite {
                        worst = max(worst, abs(mine[k] - referenceScores[k]))
                    }
                    var masked = row
                    if !banned.isEmpty {
                        var mask = [Float](repeating: 0, count: row.dim(0))
                        for token in banned { mask[token] = -Float.infinity }
                        masked = row + MLXArray(mask)
                    }
                    if masked.argMax().item(Int.self) != reply[step] { mismatched += 1 }
                }
                print("VALIDATION PARITY ltx2-25-real-prompt-enhancer-f32 \(name): \(reply.count - mismatched) of "
                      + "\(reply.count) teacher-forced choices equal, largest top-five score difference \(worst)")
                XCTAssertEqual(mismatched, 0, "\(name) float32 choices")
                XCTAssertLessThan(worst, 1e-2, "\(name) float32 scores")
            }
        }
        NFKMLXGPU.clearCache()

        // Bfloat16 greedy, against the reference's own bfloat16 departure.
        let enhancer = try NFKMLXLTX2PromptEnhancer.enhancer(directoryURL: root)
        for (name, request) in requests {
            let reply = enhancer.reply(to: try enhancer.tokens(for: request))
            let referenceBF16 = try ints("\(name).bf16.reply").filter { !NFKMLXLTX2PromptEnhancer.endTokens.contains($0) }
            let ours = try departure(reply, name: name)
            let torch = try departure(referenceBF16, name: name)
            print("VALIDATION BF16 ltx2-25-real-prompt-enhancer \(name): departs from the float32 reply at step "
                  + "\(ours.map { "\($0.step), gap \($0.gap)" } ?? "none"); torch's bfloat16 at "
                  + "\(torch.map { "\($0.step), gap \($0.gap)" } ?? "none")")
            if let ours {
                XCTAssertLessThanOrEqual(ours.gap, max(torch?.gap ?? 0, Float(pow(2.0, -4))),
                                         "\(name): the bfloat16 reply departs past the reference's own bfloat16 gap")
            }
        }
    }

    // LTX-2.5's released duration head at float32, over connector-shaped inputs.
    func testLTX25DurationHeadMatchesTheReferenceOnReleasedWeights() throws {
        let arrays = try record("IK_PARITY_LTX2_REAL_DURATION_HEAD",
                                oracle: "ltx2_real_duration_head --checkpoint <duration_head>")
        guard let path = config["IK_VAL_LTX2_5_DURATION_HEAD"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_LTX2_5_DURATION_HEAD to LTX-2.5's duration_head directory")
        }
        let directory = URL(fileURLWithPath: path)
        let head = NFKMLXLTX2DurationHeadNet(try NFKMLXLTX2DurationHead.configuration(
            fromHuggingFace: directory.appendingPathComponent("config.json")))
        try NFKMLXLTX2DurationHead.loadWeights(into: head, from: directory)
        let video = try XCTUnwrap(arrays["video"]), audio = try XCTUnwrap(arrays["audio"])
        for (name, mine, key) in [("both", head(video: video, audio: audio), "both"),
                                  ("video", head(video: video, audio: nil), "video_only"),
                                  ("audio", head(video: nil, audio: audio), "audio_only")] {
            let reference = try XCTUnwrap(arrays[key]).item(Float.self), seconds = mine.item(Float.self)
            print("VALIDATION PARITY ltx2-25-real-duration-head \(name): \(seconds) s against \(reference) s")
            XCTAssertEqual(seconds, reference, accuracy: 1e-5 * reference)
        }
    }
}

extension NFKMLXLTX2StageTests {

    // LTX-2.5 (`Lightricks/LTX-2.5-Diffusers`): the transformer built from the released configuration is
    // `.ltx25` and the generator reads it as LTX-2.5; it, the Gemma 4 unified text decoder, and the
    // connectors are held to the released headers. The autoencoders and the vocoder are LTX-2.3's files,
    // byte for byte.
    func testLTX25StagesMatchTheReleasedShapes() throws {
        let transformer = try releasedShapes("ltx2-25-transformer")
        let configuration = try NFKMLXLTX2TransformerNet.configuration(fromHuggingFace: transformer.config)
        XCTAssertEqual(NFKMLXLTX2Generator.variant(of: configuration), .ltx25)
        XCTAssertFalse(configuration.feedForwardBias)
        XCTAssertTrue(configuration.usesKeyframeEmbedding)
        XCTAssertTrue(configuration.usesPromptAdaptiveNorm, "LTX-2.5 keeps the timestep-dependent prompt modulation")
        for keyPath in [\NFKMLXLTX2Configuration.numLayers, \.numAttentionHeads, \.attentionHeadDim,
                        \.audioNumAttentionHeads, \.audioAttentionHeadDim, \.crossAttentionDim, \.inChannels] {
            XCTAssertEqual(configuration[keyPath: keyPath], NFKMLXLTX2Configuration.ltx25[keyPath: keyPath])
        }
        XCTAssertEqual(configuration.feedForwardBias, NFKMLXLTX2Configuration.ltx25.feedForwardBias)
        XCTAssertEqual(configuration.usesPromptAdaptiveNorm, NFKMLXLTX2Configuration.ltx25.usesPromptAdaptiveNorm)
        XCTAssertEqual(transformer.shapes.count, 4091, "LTX-2.3's 4186 less the 96 feed-forward biases, plus the keyframe embedding")
        assertMatches("ltx2-25-transformer", NFKMLXLTX2TransformerNet(configuration), released: transformer.shapes) { _, shape in shape }

        let text = try releasedShapes("ltx2-25-text_encoder")
        guard case .gemma4Unified(let textConfiguration) = try NFKLTX2TextEncoder.reading(text.config) else {
            return XCTFail("LTX-2.5's text encoder is Gemma 4 unified")
        }
        XCTAssertEqual(textConfiguration.layerCount + 1, 49, "the connectors read 49 hidden states")
        assertMatches("ltx2-25-text_encoder", NFKMLXGemma4UnifiedNet(textConfiguration), released: text.shapes,
                      prefix: "model.language_model.") { _, shape in shape }

        let connectors = try releasedShapes("ltx2-25-connectors")
        assertMatches("ltx2-25-connectors",
                      NFKMLXLTX2ConnectorsNet(try NFKMLXLTX2Connectors.configuration(fromHuggingFace: connectors.config)),
                      released: connectors.shapes) { _, shape in shape }
    }
}
