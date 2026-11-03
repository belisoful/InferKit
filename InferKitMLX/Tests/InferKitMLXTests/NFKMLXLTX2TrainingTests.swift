//
//  NFKMLXLTX2TrainingTests.swift
//  InferKitMLXTests
//
//  The LTX-2 LoRA objective against Lightricks' trainer (`run_reference.py ltx2_loss`, `ltx-trainer` at
//  Lightricks/LTX-2 e98f6d9 over `ltx_core`'s own transformer and peft's LoRA), and the fine-tune recipe on a
//  tiny transformer: the loss falls, only the adapters move, and the adapters save and reload.
//

import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLTX2TrainingTests: XCTestCase {

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
        guard let path = config["IK_PARITY_LTX2_LOSS"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_LTX2_LOSS (run_reference.py ltx2_loss)")
        }
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    /// The tiny transformer in LTX-2.5's arrangement, with the record's audio width.
    private static var tinyConfiguration: NFKMLXLTX2Configuration {
        var configuration = NFKMLXLTX2Configuration.tiny
        configuration.audioInChannels = 8
        configuration.audioOutChannels = 8
        configuration.feedForwardBias = false
        configuration.usesKeyframeEmbedding = true
        return configuration
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Double {
        let a = mine.asType(.float32).reshaped([-1]).asArray(Float.self)
        let b = reference.asType(.float32).reshaped([-1]).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) {
            dot += Double(x) * Double(y); na += Double(x) * Double(x); nb += Double(y) * Double(y)
        }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    private func maxDifference(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        (mine.asType(.float32) - reference.asType(.float32)).abs().max().item(Float.self)
    }

    private func required(_ arrays: [String: MLXArray], _ key: String) throws -> MLXArray {
        try XCTUnwrap(arrays[key], "the record has no \(key)")
    }

    // The sampler's arithmetic from the reference's own draws, either side of the shift's interpolation.
    func testLTX2SigmaSamplerMatchesTheTrainer() throws {
        let arrays = try record()
        let sampler = NFKMLXLTX2SigmaSampler()
        for tokens in [100, 6000] {
            let mine = sampler.sigmas(normal: try required(arrays, "sampler.\(tokens).normal"),
                                      uniform: try required(arrays, "sampler.\(tokens).uniform"),
                                      choice: try required(arrays, "sampler.\(tokens).choice"), tokens: tokens)
            let difference = maxDifference(mine, try required(arrays, "sampler.\(tokens).sigma"))
            print("VALIDATION PARITY ltx2-sampler-\(tokens): max sigma difference \(difference)")
            XCTAssertLessThan(difference, 1e-6, "\(tokens) tokens")
        }
    }

    // Both reference strategies, end to end: the noised inputs, the targets, the per-token timesteps and loss
    // mask, the predictions over `ltx_core`'s cross-modal modulation, the loss, and every adapter gradient.
    func testLTX2ObjectiveMatchesTheTrainer() throws {
        let arrays = try record()
        let net = NFKMLXLTX2TransformerNet(Self.tinyConfiguration)
        try NFKMLXWeights.apply(arrays.compactMap { key, value in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value) : nil
        }, to: net, verifyShapes: true)
        let adapted = try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 8) { path, _ in NFKMLXLTX2Trainable.targets(path) }
        let adapters = Dictionary(uniqueKeysWithValues: arrays.compactMap { key, value in
            key.hasPrefix("lora::") ? (String(key.dropFirst(6)), value) : nil
        })
        XCTAssertEqual(try NFKMLXLoRA.loadAdapters(adapters, into: net), adapted)
        XCTAssertEqual(adapted, 2 * 6 * 4, "four projections of six attentions in each of two blocks")

        let clean = try required(arrays, "clean"), audioClean = try required(arrays, "audio_clean")
        let text = try required(arrays, "video_text"), audioText = try required(arrays, "audio_text")
        let sampler = NFKMLXLTX2SigmaSampler()
        for prefix in ["t2v", "flex"] {
            let sigma = try required(arrays, "\(prefix).sigma"), audioSigma = try required(arrays, "\(prefix).audio_sigma")
            let normal = try required(arrays, "\(prefix).draw.normal")
            let sampled = sampler.sigmas(normal: normal, uniform: try required(arrays, "\(prefix).draw.uniform"),
                                         choice: try required(arrays, "\(prefix).draw.choice"), tokens: clean.dim(1))
            XCTAssertLessThan(maxDifference(sampled, sigma), 1e-6, "\(prefix) video sigma")
            if prefix == "flex" {
                let audioSampled = sampler.sigmas(
                    normal: try required(arrays, "flex.draw.audio_normal"),
                    uniform: try required(arrays, "flex.draw.audio_uniform"),
                    choice: try required(arrays, "flex.draw.audio_choice"), tokens: audioClean.dim(1))
                XCTAssertLessThan(maxDifference(audioSampled, audioSigma), 1e-6, "flex audio sigma")
            }
            let conditioned = prefix == "t2v" ? try required(arrays, "t2v.conditioned") : MLXArray.zeros([clean.dim(0)])
            let noise = try required(arrays, "\(prefix).draw.noise")
            let audioNoise = try required(arrays, "\(prefix).draw.audio_noise")
            let inputs = NFKMLXLTX2Objective.inputs(video: clean, audio: audioClean, sigma: sigma, audioSigma: audioSigma,
                                                    noise: noise, audioNoise: audioNoise, conditioned: conditioned,
                                                    tokensPerFrame: 2 * 3)
            for (label, mine, key) in [("noisy", inputs.video, "noisy"), ("audio noisy", inputs.audio, "audio_noisy"),
                                       ("targets", inputs.videoTargets, "targets"),
                                       ("audio targets", inputs.audioTargets, "audio_targets"),
                                       ("timesteps", inputs.timesteps, "timesteps"),
                                       ("audio timesteps", inputs.audioTimesteps, "audio_timesteps"),
                                       ("loss mask", inputs.lossMask, "loss_mask")] {
                let reference = try required(arrays, "\(prefix).\(key)")
                XCTAssertEqual(mine.shape, reference.shape, "\(prefix) \(label) shape")
                XCTAssertLessThan(maxDifference(mine, reference), 1e-6, "\(prefix) \(label)")
            }

            let (video, audio) = NFKMLXLTX2Objective.predictions(net, inputs: inputs, text: text, audioText: audioText,
                                                                 frames: 3, height: 2, width: 3, frameRate: 24)
            let videoSimilarity = cosine(video, try required(arrays, "\(prefix).prediction"))
            let audioSimilarity = cosine(audio, try required(arrays, "\(prefix).audio_prediction"))
            print("SEAM ltx2-loss \(prefix) prediction: video cosine \(videoSimilarity), audio \(audioSimilarity)")
            XCTAssertGreaterThan(videoSimilarity, 0.99999)
            XCTAssertGreaterThan(audioSimilarity, 0.99999)
            // The control: diffusers' cross-modal modulation, which the record must tell apart from `ltx_core`'s.
            let scale = net.config.timestepScaleMultiplier
            for crossTimestep in [false, true] {
                let control = net(video: inputs.video, audio: inputs.audio, text: text, audioText: audioText,
                                  timestep: inputs.timesteps * scale, audioTimestep: inputs.audioTimesteps * scale,
                                  sigma: inputs.sigma * scale, frames: 3, height: 2, width: 3,
                                  audioFrames: inputs.audio.dim(1), fps: 24, audioSigma: inputs.audioSigma * scale,
                                  crossTimestep: crossTimestep)
                let controlSimilarity = min(cosine(control.video, try required(arrays, "\(prefix).prediction")),
                                            cosine(control.audio, try required(arrays, "\(prefix).audio_prediction")))
                print("SEAM ltx2-loss \(prefix) control (diffusers, crossTimestep \(crossTimestep)): cosine \(controlSimilarity)")
                XCTAssertLessThan(controlSimilarity, 0.9999999, "the record discriminates the cross-modal modulation")
            }
            let losses = NFKMLXLTX2Objective.exampleLosses(video: video, audio: audio, inputs: inputs)
            let referenceLosses = try required(arrays, "\(prefix).per_example_loss")
            let lossError = (maxDifference(losses, referenceLosses) / referenceLosses.abs().max().item(Float.self))
            XCTAssertLessThan(lossError, 1e-5, "\(prefix) per-example loss")

            let step = valueAndGrad(model: net) { (net: NFKMLXLTX2TransformerNet, _: [MLXArray]) in
                [NFKMLXLTX2Objective.score(net, video: clean, audio: audioClean, text: text, audioText: audioText,
                                           sigma: sigma, audioSigma: audioSigma, noise: noise, audioNoise: audioNoise,
                                           conditioned: conditioned, frames: 3, height: 2, width: 3, frameRate: 24)]
            }
            let (values, gradients) = step(net, [])
            let loss = values[0].item(Float.self), referenceLoss = try required(arrays, "\(prefix).loss").item(Float.self)
            let flat = Dictionary(uniqueKeysWithValues: gradients.flattened())
            var worst = 1.0
            for path in adapters.keys.sorted() {
                let mine = try XCTUnwrap(flat[path], "no gradient for \(path)")
                let similarity = cosine(mine, try required(arrays, "\(prefix).grad::\(path)"))
                worst = min(worst, similarity)
                XCTAssertGreaterThan(similarity, 0.99999, "\(prefix) \(path) gradient")
            }
            print("VALIDATION PARITY ltx2-loss-\(prefix): loss \(loss) against \(referenceLoss), "
                  + "worst adapter-gradient cosine \(worst) over \(adapters.count) arrays")
            XCTAssertLessThan(abs(loss - referenceLoss) / referenceLoss, 1e-5, "\(prefix) loss")
        }
    }

    // The recipe on a tiny transformer: over one clip the loss falls, the adapters move, the base and the
    // layers outside the trainer's targets stay where they were.
    func testLTX2FineTuneLowersTheLossAndTrainsOnlyTheAdapters() throws {
        NFKMLXRandom.seed(3)
        let net = NFKMLXLTX2TransformerNet(Self.tinyConfiguration)
        eval(net)
        let example = NFKMLXLTX2TrainingExample(
            video: MLXRandom.normal([18, 8]), audio: MLXRandom.normal([5, 8]), text: MLXRandom.normal([4, 32]),
            audioText: MLXRandom.normal([4, 16]), frames: 3, height: 2, width: 3, frameRate: 24)
        let gate = net.transformerBlocks[0].attn1.toGateLogits?.weight
        let losses = try NFKMLXLTX2Generator.fineTune(net, examples: { _ in [example, example] },
                                                      trainable: .lora(rank: 8, alpha: 8),
                                                      objective: .textToVideo, learningRate: 3e-3, steps: 60, seed: 2)
        let first = losses.prefix(10).reduce(0, +) / 10, last = losses.suffix(10).reduce(0, +) / 10
        print("ltx2 fine-tune: mean loss \(first) over the first ten steps, \(last) over the last ten")
        XCTAssertLessThan(last, first * 0.9)
        XCTAssertTrue(net.transformerBlocks[0].attn1.toQ is NFKMLXLoRALinear)
        XCTAssertFalse(net.transformerBlocks[0].attn1.toGateLogits is NFKMLXLoRALinear, "the gate is not a target")
        XCTAssertFalse(net.projIn is NFKMLXLoRALinear, "projections outside the blocks are not targets")
        XCTAssertTrue(arrayEqual(try XCTUnwrap(net.transformerBlocks[0].attn1.toGateLogits?.weight), try XCTUnwrap(gate))
            .item(Bool.self), "a frozen layer stays put")
    }

    // The video encoder the data preparation runs, through its directory loader: the posterior's moments and
    // the normalized latents the trainer stores, for a square 9-frame clip and a wide 17-frame one.
    func testLTX2VideoEncoderMatchesTheReference() throws {
        guard let path = config["IK_PARITY_LTX2_VIDEO_ENCODER"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_LTX2_VIDEO_ENCODER (run_reference.py ltx2_video_encoder)")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("ltx2-vae-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try MLX.save(arrays: arrays.reduce(into: [String: MLXArray]()) { stored, entry in
            if entry.key.hasPrefix("w::") { stored[String(entry.key.dropFirst(3))] = entry.value }
        }, url: directory.appendingPathComponent("diffusion_pytorch_model.safetensors"))

        let encoder = try NFKMLXLTX2VideoEncoderNet(.tiny)
        let statistics = try NFKMLXLTX2VideoVAE.loadEncoderWeights(into: encoder, from: directory)
        for name in ["square", "wide"] {
            let video = try required(arrays, "\(name).video").transposed(0, 2, 3, 4, 1)
            let moments = encoder(video).transposed(0, 4, 1, 2, 3)
            let latents = encoder.latents(video, statistics: statistics).transposed(0, 4, 1, 2, 3)
            let reference = try required(arrays, "\(name).latents")
            XCTAssertEqual(latents.shape, reference.shape)
            let momentSimilarity = cosine(moments, try required(arrays, "\(name).moments"))
            let latentSimilarity = cosine(latents, reference)
            print("VALIDATION PARITY ltx2-video-encoder-\(name): moments cosine \(momentSimilarity), "
                  + "normalized latents \(latentSimilarity), max difference \(maxDifference(latents, reference))")
            XCTAssertGreaterThan(momentSimilarity, 0.99999)
            XCTAssertGreaterThan(latentSimilarity, 0.99999)
        }
    }

    // The trainer's audio front end and the audio encoder: the resample to 16 kHz from 24 and 44.1 kHz, the
    // release's 64-bin log-mel spectrogram, and the tiny encoder's moments and packed, normalized latents.
    func testLTX2AudioEncoderAndFrontEndMatchTheTrainer() throws {
        guard let path = config["IK_PARITY_LTX2_AUDIO_ENCODER"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_LTX2_AUDIO_ENCODER (run_reference.py ltx2_audio_encoder)")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let release = NFKMLXLTX2MelFrontEnd()
        for (name, rate) in [("24k", 24000), ("44k", 44100)] {
            let wave = try required(arrays, "front.\(name).wave")
            let resampled = release.resampled(wave, from: rate)
            let referenceResampled = try required(arrays, "front.\(name).resampled")
            XCTAssertEqual(resampled.shape, referenceResampled.shape)
            let resampleDifference = maxDifference(resampled, referenceResampled)
            // `[1, frames, mel, channels]` against the reference's `[1, channels, frames, mel]`.
            let mel = release.logMel(resampled).transposed(0, 3, 1, 2)
            let referenceMel = try required(arrays, "front.\(name).mel")
            XCTAssertEqual(mel.shape, referenceMel.shape)
            let melDifference = maxDifference(mel, referenceMel)
            print("VALIDATION PARITY ltx2-mel-\(name): resample max difference \(resampleDifference), "
                  + "log-mel cosine \(cosine(mel, referenceMel)), max difference \(melDifference)")
            XCTAssertLessThan(resampleDifference, 1e-5)
            XCTAssertLessThan(melDifference, 1e-3)
            XCTAssertGreaterThan(cosine(mel, referenceMel), 0.99999)
        }

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("ltx2-audio-vae-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try MLX.save(arrays: arrays.reduce(into: [String: MLXArray]()) { stored, entry in
            if entry.key.hasPrefix("w::") { stored[String(entry.key.dropFirst(3))] = entry.value }
        }, url: directory.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        let encoder = NFKMLXLTX2AudioEncoderNet(.tiny)
        let statistics = try NFKMLXLTX2AudioVAE.loadEncoderWeights(into: encoder, from: directory)
        let tiny = NFKMLXLTX2MelFrontEnd(.tiny)
        let mel = tiny.logMel(tiny.resampled(try required(arrays, "encoder.wave"), from: 24000))
        let referenceMel = try required(arrays, "encoder.mel")
        XCTAssertGreaterThan(cosine(mel.transposed(0, 3, 1, 2), referenceMel), 0.99999)
        // The encoder reads the reference's own spectrogram, so its figure is the encoder's alone.
        let input = referenceMel.transposed(0, 2, 3, 1)
        let moments = encoder(input).transposed(0, 3, 1, 2)
        let latents = encoder.latents(input, statistics: statistics)
        let referenceLatents = try required(arrays, "encoder.latents")
        XCTAssertEqual(latents.shape, referenceLatents.shape)
        let momentSimilarity = cosine(moments, try required(arrays, "encoder.moments"))
        let latentSimilarity = cosine(latents, referenceLatents)
        print("VALIDATION PARITY ltx2-audio-encoder: moments cosine \(momentSimilarity), "
              + "packed normalized latents \(latentSimilarity)")
        XCTAssertGreaterThan(momentSimilarity, 0.99999)
        XCTAssertGreaterThan(latentSimilarity, 0.99999)
    }

    // Recomputing each block in the backward pass changes no number: the loss and every adapter's gradient
    // equal the plain backward's, over a float base and an 8-bit one.
    func testRecomputingBlocksInBackwardKeepsTheGradients() throws {
        for bits in [nil, 8] as [Int?] {
            NFKMLXRandom.seed(5)
            let net = NFKMLXLTX2TransformerNet(Self.tinyConfiguration)
            if let bits { try NFKMLXQuantization.quantize(module: net, bits: bits, groupSize: 32) }
            try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 4) { path, _ in NFKMLXLTX2Trainable.targets(path) }
            for (path, layer) in net.leafModules().flattened() {
                guard let adapter = layer as? NFKMLXLoRAAdapter else { continue }
                _ = adapter.update(parameters: ModuleParameters.unflattened(
                    ["lora_b": MLXRandom.normal(adapter.loraB.shape, key: MLXRandom.key(UInt64(path.count))) * 0.1]))
            }
            let video = MLXRandom.normal([2, 18, 8]), audio = MLXRandom.normal([2, 5, 8])
            let text = MLXRandom.normal([2, 4, 32]), audioText = MLXRandom.normal([2, 4, 16])
            let draws = [MLXArray([Float(0.3), 0.8]), MLXArray([Float(0.6), 0.2]), MLXRandom.normal([2, 18, 8]),
                         MLXRandom.normal([2, 5, 8]), MLXArray([Float(1), 0])]
            func gradients() -> (Float, [String: MLXArray]) {
                let step = valueAndGrad(model: net) { (net: NFKMLXLTX2TransformerNet, _: [MLXArray]) in
                    [NFKMLXLTX2Objective.score(net, video: video, audio: audio, text: text, audioText: audioText,
                                               sigma: draws[0], audioSigma: draws[1], noise: draws[2],
                                               audioNoise: draws[3], conditioned: draws[4], frames: 3, height: 2,
                                               width: 3, frameRate: 24)]
                }
                let (values, grads) = step(net, [])
                return (values[0].item(Float.self), Dictionary(uniqueKeysWithValues: grads.flattened()))
            }
            net.recomputesBlocksInBackward = false
            let (plainLoss, plain) = gradients()
            net.recomputesBlocksInBackward = true
            let (recomputedLoss, recomputed) = gradients()
            XCTAssertEqual(recomputedLoss, plainLoss, "\(bits.map { "\($0)-bit" } ?? "float") loss")
            XCTAssertEqual(Set(recomputed.keys), Set(plain.keys))
            var worst: Float = 0
            for (key, value) in plain {
                worst = max(worst, maxDifference(try XCTUnwrap(recomputed[key]), value)
                    / max(value.abs().max().item(Float.self), 1e-12))
            }
            print("ltx2 recomputed backward (\(bits.map { "\($0)-bit" } ?? "float")): largest relative gradient "
                  + "difference \(worst) over \(plain.count) arrays")
            XCTAssertLessThan(worst, 1e-5)
        }
    }

    // One QLoRA run on LTX-2.5's released full transformer (22B) at 4 bits on this machine: the block-wise
    // quantized build, then three steps of the reference recipe (rank 32 on every attention projection) over a
    // 49-frame 512×320 clip's latent grid, its audio, and 1024 text tokens, with the build's memory, the peak,
    // and each step's time printed.
    func testLTX25QLoRAStepOnReleasedWeights() throws {
        guard let path = config["IK_VAL_LTX2_5_TRANSFORMER_FULL"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_LTX2_5_TRANSFORMER_FULL to the directory holding LTX-2.5's transformer_full/")
        }
        func gigabytes(_ bytes: Int) -> String { String(format: "%.2f", Double(bytes) / 1e9) }
        NFKMLXGPU.resetPeakMemory()
        let buildStart = Date()
        let net = try NFKMLXLTX2Generator.network(directoryURL: URL(fileURLWithPath: path), variant: .ltx25Full,
                                                  quantizationBits: 4)
        let buildSeconds = Date().timeIntervalSince(buildStart)
        XCTAssertTrue(net.transformerBlocks[0].attn1.toQ is QuantizedLinear)
        print("MEMORY ltx2.5-qlora build: \(String(format: "%.1f", buildSeconds)) s, active "
              + "\(gigabytes(NFKMLXGPU.activeMemory)) GB, peak \(gigabytes(NFKMLXGPU.peakMemory)) GB")

        let (frames, height, width) = (7, 10, 16)
        let example = NFKMLXLTX2TrainingExample(
            video: MLXRandom.normal([frames * height * width, 128], key: MLXRandom.key(1)),
            audio: MLXRandom.normal([51, 128], key: MLXRandom.key(2)),
            text: MLXRandom.normal([1024, 4096], key: MLXRandom.key(3)).asType(.bfloat16),
            audioText: MLXRandom.normal([1024, 2048], key: MLXRandom.key(4)).asType(.bfloat16),
            frames: frames, height: height, width: width, frameRate: 24)
        NFKMLXGPU.resetPeakMemory()
        var stepStart = Date()
        var stepSeconds = [Double]()
        let losses = try NFKMLXLTX2Generator.fineTune(net, examples: { _ in [example] }, steps: 3, seed: 1) { _ in
            stepSeconds.append(Date().timeIntervalSince(stepStart))
            stepStart = Date()
            return true
        }
        print("MEMORY ltx2.5-qlora steps: losses \(losses), seconds \(stepSeconds.map { String(format: "%.1f", $0) }), "
              + "peak \(gigabytes(NFKMLXGPU.peakMemory)) GB, trainable "
              + "\(NFKMLXLoRA.trainableParameterCount(of: net)) parameters")
        XCTAssertEqual(losses.count, 3)
        XCTAssertTrue(losses.allSatisfy(\.isFinite))
        XCTAssertTrue(net.transformerBlocks[0].attn1.toQ is NFKMLXQuantizedLoRALinear)
    }
}
