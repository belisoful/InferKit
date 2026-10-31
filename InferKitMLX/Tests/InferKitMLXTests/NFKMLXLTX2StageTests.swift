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

    // The whole pipeline at tiny configurations of every stage, from the record's starting noise: the
    // text states from token ids, the connectors, three steps of the guided loop (the guided pair,
    // spatio-temporal guidance, modality isolation, and the rescale), both decodes, and the vocoder.
    // A disagreement localizes to the text front end, the loop, or a decode.
    func testLTX2PipelineMatchesTheReference() throws {
        let arrays = try record("IK_PARITY_LTX2_PIPELINE", oracle: "ltx2_pipeline")

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
            compare("ltx2-pipeline text states row \(row)", states,
                    embeds[row].take(MLXArray(valid.map(Int32.init)), axis: 0).expandedDimensions(axis: 0))
            let (video, audio) = connectors(validStates: states, length: length)
            videoStreams.append(video)
            audioStreams.append(audio)
        }
        let text = concatenated(videoStreams, axis: 0), audioText = concatenated(audioStreams, axis: 0)
        compare("ltx2-pipeline connector video", text, arrays["seam.connector_video"])
        compare("ltx2-pipeline connector audio", audioText, arrays["seam.connector_audio"])

        var transformerConfiguration = NFKMLXLTX2Configuration.tiny
        transformerConfiguration.audioInChannels = 8
        transformerConfiguration.audioOutChannels = 8
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
        var scheduler = NFKMLXFlowMatchScheduler(.ltxVideoPipeline)
        scheduler.setTimesteps(3, sequenceLength: 12)
        compare("ltx2-pipeline sigmas", MLXArray(scheduler.sigmas), arrays["sigmas"])

        let (video, audio) = try pipeline.denoise(
            video: try XCTUnwrap(arrays["latents"]), audio: pipeline.normalizedAudio(try XCTUnwrap(arrays["audio_latents"])),
            text: text, audioText: audioText, frames: grid.frames, height: grid.height, width: grid.width,
            audioFrames: grid.audioFrames, frameRate: 24, steps: 3, perturbedBlocks: [1])
        let unpackedVideo = video.reshaped([1, 2, 2, 3, 8]) * videoStatistics.std + videoStatistics.mean
        let videoLatent = compare("ltx2-pipeline final video latent", channelsFirst(unpackedVideo),
                                  arrays["seam.video_latent"])
        let unpackedAudio = (audio * audioStatistics.std + audioStatistics.mean).reshaped([1, 9, 2, 4])
        compare("ltx2-pipeline final audio latent", unpackedAudio.transposed(0, 2, 1, 3), arrays["seam.audio_latent"])

        let clip = clip(pipeline.decodeVideo(video, frames: 2, height: 2, width: 3) / 2 + 0.5, min: 0, max: 1)
        let videoSimilarity = compare("ltx2-pipeline video", clip.transposed(0, 1, 4, 2, 3), arrays["video"])
        compare("ltx2-pipeline mel", pipeline.decodeMel(audio).transposed(0, 3, 1, 2), arrays["seam.mel"])
        let audioSimilarity = compare("ltx2-pipeline audio", pipeline.decodeAudio(audio).transposed(0, 2, 1),
                                      arrays["audio"])
        print("VALIDATION PARITY ltx2-pipeline: final video latent cosine \(videoLatent), clip \(videoSimilarity), "
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
        try write(["_class_name": "FlowMatchEulerDiscreteScheduler", "base_shift": 0.95, "max_shift": 2.05,
                   "shift_terminal": 0.1, "use_dynamic_shifting": true], "scheduler/scheduler_config.json")
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
