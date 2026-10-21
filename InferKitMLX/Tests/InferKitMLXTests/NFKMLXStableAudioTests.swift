//
//  NFKMLXStableAudioTests.swift
//  InferKitMLXTests
//
//  Stable Audio Open 1.0: the Oobleck autoencoder, the diffusion transformer, the seconds conditioners,
//  t5-base's ReLU feed-forward, and the cosine DPM-Solver++ SDE sampler, against diffusers' own classes
//  at tiny random configurations (`run_reference.py stable_audio_tiny`), on the released components
//  (`stable_audio_components`), and end to end through `StableAudioPipeline` with the reference's own
//  noise (`stable_audio_pipeline`).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXStableAudioTests: XCTestCase {

    private let config = NFKMLXValidationConfig.environment

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func values(_ a: MLXArray) -> [Double] {
        a.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = values(a), y = values(b)
        XCTAssertEqual(x.count, y.count, "the compared tensors hold the same number of elements")
        var dot = 0.0, nx = 0.0, ny = 0.0
        for i in 0 ..< min(x.count, y.count) {
            dot += x[i] * y[i]; nx += x[i] * x[i]; ny += y[i] * y[i]
        }
        return (nx > 0 && ny > 0) ? dot / (nx.squareRoot() * ny.squareRoot()) : 0
    }

    /// The largest absolute difference relative to the reference's largest magnitude.
    private func relativeError(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = values(a), y = values(b)
        let scale = y.map(abs).max() ?? 0
        let worst = zip(x, y).map { abs($0 - $1) }.max() ?? 0
        return scale > 0 ? worst / scale : worst
    }

    private func record(_ key: String, mode: String) throws -> [String: MLXArray] {
        guard let path = config[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key) (run_reference.py \(mode))")
        }
        return try loadArrays(url: URL(fileURLWithPath: path))
    }

    private func weights(_ arrays: [String: MLXArray], prefix: String) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: arrays.compactMap { key, value in
            key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil
        })
    }

    /// `[B, C, L]` → `[B, L, C]`.
    private func nlc(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 1) }

    // MARK: - Weight-free structure

    // The sampler's schedule runs from σ_max down to σ_min and then to zero, the model reads
    // atan(σ)·2/π, and the last step, which lands on zero noise, returns the denoised estimate itself.
    func testTheSamplerEndsOnTheDenoisedEstimate() throws {
        try requireMLXRuntime()
        let scheduler = NFKMLXCosineDPMSolverScheduler(steps: 4)
        XCTAssertEqual(scheduler.sigmas.count, 5)
        XCTAssertEqual(scheduler.sigmas[0], 500, accuracy: 1e-3)
        XCTAssertEqual(scheduler.sigmas[3], 0.3, accuracy: 1e-6)
        XCTAssertEqual(scheduler.sigmas[4], 0)
        XCTAssertEqual(scheduler.timesteps[0], Foundation.atan(Float(500)) / Float.pi * 2, accuracy: 1e-7)

        var sample = MLXArray([Float(1), -2])
        let output = MLXArray([Float(0.5), 0.25])
        for _ in 0 ..< 3 {
            sample = scheduler.step(modelOutput: output, sample: sample, noise: MLXArray.zeros([2]))
        }
        let estimate = scheduler.denoised(output, sample: sample)
        let last = scheduler.step(modelOutput: output, sample: sample, noise: MLXArray([Float(9), 9]))
        eval(estimate, last)
        XCTAssertEqual(last.asArray(Float.self), estimate.asArray(Float.self), "zero noise leaves the estimate")
    }

    // The rotary turns only the leading half of each head, and position 0 is the identity.
    func testTheRotaryTurnsOnlyTheLeadingChannelsOfEachHead() throws {
        try requireMLXRuntime()
        let x = MLXRandom.normal([1, 2, 3, 8], key: MLXRandom.key(3))
        let table = NFKMLXStableAudio.rotaryTable(length: 3, dimensions: 4)
        let turned = NFKMLXStableAudio.rotated(x, table)
        eval(turned)
        XCTAssertEqual(values(turned[.ellipsis, 4...]), values(x[.ellipsis, 4...]))
        XCTAssertEqual(values(turned[0..., 0..., 0 ..< 1]), values(x[0..., 0..., 0 ..< 1]))
        XCTAssertNotEqual(values(turned[0..., 0..., 1 ..< 2, 0 ..< 4]), values(x[0..., 0..., 1 ..< 2, 0 ..< 4]))
    }

    // The transformer drops its prepended global token, so the prediction keeps the latent's shape, and a
    // different timestep changes it.
    func testThePredictionKeepsTheLatentShapeAndReadsTheTimestep() throws {
        try requireMLXRuntime()
        let c = NFKMLXStableAudioDiTConfiguration.tiny
        let net = NFKMLXStableAudioDiTNet(c)
        let latents = MLXRandom.normal([1, 6, c.inChannels], key: MLXRandom.key(5))
        let context = MLXRandom.normal([1, 3, c.crossAttentionInputDim], key: MLXRandom.key(6))
        let global = MLXRandom.normal([1, 1, c.globalStatesInputDim], key: MLXRandom.key(7))
        let early = net(latents, timestep: MLXArray([Float(0.1)]), context: context, global: global)
        let late = net(latents, timestep: MLXArray([Float(0.9)]), context: context, global: global)
        eval(early, late)
        XCTAssertEqual(early.shape, latents.shape)
        XCTAssertGreaterThan((early - late).abs().max().item(Float.self), 0, "the timestep conditions the prediction")
    }

    // The encoder divides the length by the hop and the decoder restores it.
    func testTheAutoencoderDividesAndRestoresTheLength() throws {
        try requireMLXRuntime()
        let net = NFKMLXOobleckAutoencoderNet(.tiny)
        let waveform = MLXRandom.normal([1, 64, 2], key: MLXRandom.key(9))
        let posterior = net.encode(waveform)
        let decoded = net.decode(posterior.mean)
        eval(posterior.standardDeviation, decoded)
        XCTAssertEqual(posterior.mean.shape, [1, 64 / NFKMLXOobleckConfiguration.tiny.hopLength, 4])
        XCTAssertEqual(decoded.shape, waveform.shape)
        XCTAssertGreaterThan(posterior.standardDeviation.min().item(Float.self), 1e-4 - 1e-7)
    }

    // A T5 config without `feed_forward_proj` is the original T5's ReLU feed-forward, as transformers
    // reads it, and `gated-gelu` is the v1.1 form.
    func testAT5ConfigWithoutAFeedForwardKindReadsAsReLU() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("t5-config-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let plain = directory.appendingPathComponent("plain.json"), gated = directory.appendingPathComponent("gated.json")
        try Data(#"{"d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072}"#.utf8).write(to: plain)
        try Data(#"{"d_model": 768, "feed_forward_proj": "gated-gelu"}"#.utf8).write(to: gated)
        XCTAssertFalse(try NFKMLXT5Encoder.configuration(fromHuggingFace: plain).gatedFeedForward)
        XCTAssertTrue(try NFKMLXT5Encoder.configuration(fromHuggingFace: gated).gatedFeedForward)
        try Data(#"{"feed_forward_proj": "gated-silu"}"#.utf8).write(to: plain)
        XCTAssertThrowsError(try NFKMLXT5Encoder.configuration(fromHuggingFace: plain))
    }

    // The release states `sample_size` as 1024.0; an integer field stated as a float still reads, so a
    // geometry other than the default is not silently replaced by it.
    func testATransformerConfigStatingAFloatSizeReadsAsAnInteger() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("stable-audio-config-\(UUID().uuidString).json")
        try Data(#"{"sample_size": 512.0, "num_layers": 3, "attention_head_dim": 32}"#.utf8).write(to: url)
        let configuration = try NFKMLXStableAudioDiTConfiguration.configuration(fromHuggingFace: url)
        XCTAssertEqual(configuration.sampleSize, 512)
        XCTAssertEqual(configuration.layers, 3)
        XCTAssertEqual(configuration.headDim, 32)
    }

    // A stereo WAV reads back with its channels kept, and the mono reader still averages them.
    func testAStereoWaveReadsBackWithItsChannels() throws {
        let samples: [Float] = [0.5, -0.5, 0.25, 0.75]
        let data = NFKMLXWaveFile.data(samples: samples, sampleRate: 44100, channels: 2)
        let stereo = try XCTUnwrap(NFKMLXWaveFile.readChannels(data))
        XCTAssertEqual(stereo.channels, 2)
        XCTAssertEqual(stereo.sampleRate, 44100)
        for (read, written) in zip(stereo.samples, samples) {
            XCTAssertEqual(read, written, accuracy: 1 / 32767)
        }
        let mono = try XCTUnwrap(NFKMLXWaveFile.read(data))
        XCTAssertEqual(mono.samples.count, 2)
        XCTAssertEqual(mono.samples[0], 0, accuracy: 1 / 32767)
        XCTAssertEqual(mono.samples[1], 0.5, accuracy: 1 / 32767)
    }

    // MARK: - Customization

    private func tinyTrainingNet() -> NFKMLXStableAudioTrainingNet {
        NFKMLXStableAudioTrainingNet(transformer: NFKMLXStableAudioDiTNet(.tiny),
                                     projection: NFKMLXStableAudioProjectionNet(.tiny))
    }

    private func tinyExamples(count: Int, seed: UInt64) -> [NFKMLXStableAudioExample] {
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 2 * count)
        return (0 ..< count).map { index in
            NFKMLXStableAudioExample(latents: MLXRandom.normal([6, 4], key: keys[2 * index]), validFrames: 6 - index,
                                     text: MLXRandom.normal([3, 10], key: keys[2 * index + 1]),
                                     startSeconds: Float(index), totalSeconds: 10)
        }
    }

    // The loss reads only the frames that hold the clip: a padded frame's prediction does not move it.
    func testTheObjectiveLeavesThePaddedFramesOut() throws {
        try requireMLXRuntime()
        let objective = NFKMLXStableAudioObjective()
        let latents = MLXRandom.normal([1, 4, 2], key: MLXRandom.key(51))
        let noise = MLXRandom.normal([1, 4, 2], key: MLXRandom.key(52))
        let t = MLXArray([Float(0.4)]), mask = MLXArray([Float(1), 1, 1, 0]).reshaped([1, 4])
        let prediction = MLXRandom.normal([1, 4, 2], key: MLXRandom.key(53))
        let changed = concatenated([prediction[0..., 0 ..< 3], prediction[0..., 3...] + 100], axis: 1)
        let a = objective.loss(prediction: prediction, latents: latents, noise: noise, timesteps: t, mask: mask)
        let b = objective.loss(prediction: changed, latents: latents, noise: noise, timesteps: t, mask: mask)
        eval(a, b)
        XCTAssertEqual(a.item(Float.self), b.item(Float.self))
    }

    // The reference's LoRA trains: the objective on a fixed draw falls, adapters land on the transformer
    // and on the seconds conditioners, and no base weight moves.
    func testAFineTuneLowersTheLossAndMovesOnlyTheAdapters() throws {
        try requireMLXRuntime()
        MLXRandom.seed(31)
        let net = tinyTrainingNet()
        let examples = tinyExamples(count: 2, seed: 33)
        let latents = MLX.stacked(examples.map(\.latents))
        let mask = MLXArray([Float](repeating: 1, count: 12)).reshaped([2, 6])
        let text = MLX.stacked(examples.map(\.text))
        let start = MLXArray(examples.map(\.startSeconds)), total = MLXArray(examples.map(\.totalSeconds))
        let noise = MLXRandom.normal(latents.shape, key: MLXRandom.key(35))
        let timesteps = MLXArray([Float(0.3), 0.7])
        let objective = NFKMLXStableAudioObjective(conditioningDropout: 0)
        func evaluate() -> Float {
            let loss = objective(net, latents: latents, mask: mask, text: text, start: start, total: total, noise: noise,
                                 timesteps: timesteps, dropped: MLXArray.zeros([2]))
            eval(loss)
            return loss.item(Float.self)
        }
        let before = evaluate()
        let baseBefore = net.parameters().flattened().filter { !$0.0.contains("lora") }
        eval(baseBefore.map(\.1))
        let losses = try NFKMLXStableAudio.fineTune(net, examples: { _ in examples }, trainable: .lora(rank: 4, alpha: 4),
                                                    objective: objective, learningRate: 3e-3, weightDecay: 0,
                                                    steps: 40, seed: 37, learningRateSchedule: .constant)
        XCTAssertEqual(losses.count, 40)
        let after = evaluate()
        print("VALIDATION stable_audio fine-tune: fixed-draw loss \(before) -> \(after)")
        XCTAssertLessThan(after, before, "the adapters fit the clips")

        let adapters = net.trainableParameters().flattened().map(\.0)
        XCTAssertTrue(adapters.allSatisfy { $0.hasSuffix("lora_a") || $0.hasSuffix("lora_b") })
        XCTAssertTrue(adapters.contains { $0.hasPrefix("transformer.preprocess_conv.") }, "the 1x1 convolutions adapt")
        XCTAssertTrue(adapters.contains { $0.hasPrefix("transformer.transformer_blocks.0.attn2.to_k.") })
        XCTAssertTrue(adapters.contains { $0.hasPrefix("projection.end_number_conditioner.") }, "the conditioners adapt")
        let baseAfter = Dictionary(uniqueKeysWithValues: net.parameters().flattened().filter { !$0.0.contains("lora") })
        for (key, old) in baseBefore {
            XCTAssertEqual(values(old), values(try XCTUnwrap(baseAfter[key])), "\(key) is frozen")
        }
    }

    // A fine-tuned net saves as a release's `transformer/` and `projection_model/` and reloads through the
    // model's own builder to the same prediction and conditioning.
    func testAFineTunedNetRoundTripsThroughItsDirectories() throws {
        try requireMLXRuntime()
        MLXRandom.seed(41)
        let net = tinyTrainingNet()
        try NFKMLXStableAudio.fineTune(net, examples: { _ in self.tinyExamples(count: 1, seed: 43) },
                                       trainable: .lora(rank: 2, alpha: 4), learningRate: 1e-2, steps: 3,
                                       learningRateSchedule: .constant)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("stable-audio-\(UUID().uuidString)")
        try NFKMLXStableAudio.save(net, toDirectoryURL: directory)
        let reloaded = try NFKMLXStableAudio.network(directoryURL: directory)
        XCTAssertEqual(reloaded.transformer.configuration, .tiny)
        XCTAssertEqual(reloaded.projection.configuration, .tiny)
        let example = tinyExamples(count: 1, seed: 45)[0]
        func predict(_ net: NFKMLXStableAudioTrainingNet) -> MLXArray {
            let condition = net.conditioning(text: example.text.expandedDimensions(axis: 0), start: MLXArray([Float(1)]),
                                             total: MLXArray([Float(9)]))
            return net.transformer(example.latents.expandedDimensions(axis: 0), timestep: MLXArray([Float(0.5)]),
                                   context: condition.context, global: condition.global)
        }
        let mine = predict(net), theirs = predict(reloaded)
        eval(mine, theirs)
        XCTAssertLessThan((mine - theirs).abs().max().item(Float.self), 1e-6)
    }

    // The objective on identical tensors: stable-audio-tools' own loss functions over the tiny diffusers
    // transformer and projection model, its loss and three gradients (the input path, a cross-attention
    // key projection, and a seconds conditioner).
    func testTheObjectiveMatchesTheReferenceTrainingStep() throws {
        try requireMLXRuntime()
        let arrays = try record("IK_PARITY_STABLE_AUDIO_LOSS", mode: "stable_audio_loss")
        let net = tinyTrainingNet()
        try NFKMLXWeights.apply(NFKMLXStableAudio.transformerArrays(weights(arrays, prefix: "dit::").map { ($0, $1) }),
                                to: net.transformer, verifyShapes: true)
        try NFKMLXWeights.apply(weights(arrays, prefix: "proj::").map { ($0, $1) }, to: net.projection, verifyShapes: true)
        let objective = NFKMLXStableAudioObjective()
        let inputs = try ["latents", "mask", "text", "start", "total", "noise", "timesteps", "dropped"].map {
            try XCTUnwrap(arrays[$0], $0)
        }
        func loss(_ net: NFKMLXStableAudioTrainingNet, _ unused: [MLXArray]) -> [MLXArray] {
            [objective(net, latents: nlc(inputs[0]), mask: inputs[1], text: inputs[2], start: inputs[3], total: inputs[4],
                       noise: nlc(inputs[5]), timesteps: inputs[6], dropped: inputs[7])]
        }
        let (value, gradients) = valueAndGrad(model: net, loss)(net, [])
        eval(value, gradients)
        let reference = try XCTUnwrap(arrays["loss"]).item(Float.self)
        let mine = value[0].item(Float.self)
        print("VALIDATION PARITY stable_audio objective: loss \(mine) vs \(reference)")
        XCTAssertEqual(mine, reference, accuracy: abs(reference) * 1e-5)
        let flattened = Dictionary(uniqueKeysWithValues: gradients.flattened())
        for key in ["transformer.proj_in.weight", "transformer.transformer_blocks.0.attn2.to_k.weight",
                    "projection.start_number_conditioner.time_positional_embedding.1.weight"] {
            let similarity = cosine(try XCTUnwrap(flattened[key], key), try XCTUnwrap(arrays["grad::\(key)"], key))
            print("VALIDATION PARITY stable_audio objective gradient \(key): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.99999, key)
        }
    }

    // MARK: - Tiny configurations against diffusers

    // The transformer at a tiny configuration with grouped cross-attention heads (4 query, 2 key-value),
    // the half-head rotary, and the prepended global token, against diffusers' StableAudioDiTModel.
    func testTheTinyTransformerMatchesTheReference() throws {
        try requireMLXRuntime()
        let arrays = try record("IK_PARITY_STABLE_AUDIO_TINY", mode: "stable_audio_tiny")
        let net = NFKMLXStableAudioDiTNet(.tiny)
        try NFKMLXWeights.apply(NFKMLXStableAudio.transformerArrays(weights(arrays, prefix: "dit::").map { ($0, $1) }),
                                to: net, verifyShapes: true)
        let latents = nlc(try XCTUnwrap(arrays["dit_latents"]))
        let context = try XCTUnwrap(arrays["dit_context"]), global = try XCTUnwrap(arrays["dit_global"])
        let timestep = try XCTUnwrap(arrays["dit_timestep"])
        let prediction = net(latents, timestep: timestep, context: context, global: global)
        let reference = nlc(try XCTUnwrap(arrays["dit_velocity"]))
        let similarity = cosine(prediction, reference)
        print("VALIDATION PARITY stable_audio_tiny transformer: cosine \(similarity), relative \(relativeError(prediction, reference))")
        XCTAssertGreaterThan(similarity, 0.999999)
    }

    // The autoencoder's posterior (mean and softplus standard deviation) and the decode of its mean, with
    // the weight-normalized convolutions fused at load.
    func testTheTinyAutoencoderMatchesTheReference() throws {
        try requireMLXRuntime()
        let arrays = try record("IK_PARITY_STABLE_AUDIO_TINY", mode: "stable_audio_tiny")
        let net = NFKMLXOobleckAutoencoderNet(.tiny)
        try NFKMLXWeights.apply(NFKMLXStableAudio.autoencoderArrays(weights(arrays, prefix: "vae::")), to: net,
                                verifyShapes: true)
        let posterior = net.encode(nlc(try XCTUnwrap(arrays["vae_waveform"])))
        let decoded = net.decode(posterior.mean)
        let seams = [("mean", posterior.mean, nlc(try XCTUnwrap(arrays["vae_mean"]))),
                     ("std", posterior.standardDeviation, nlc(try XCTUnwrap(arrays["vae_std"]))),
                     ("decoded", decoded, nlc(try XCTUnwrap(arrays["vae_decoded"])))]
        for (name, mine, reference) in seams {
            let similarity = cosine(mine, reference)
            print("VALIDATION PARITY stable_audio_tiny autoencoder \(name): cosine \(similarity), relative \(relativeError(mine, reference))")
            XCTAssertGreaterThan(similarity, 0.999999, name)
        }
    }

    // The projection model with a text projection (the release's is the identity) and both seconds
    // conditioners, one value past the range so the clamp runs.
    func testTheTinyProjectionMatchesTheReference() throws {
        try requireMLXRuntime()
        let arrays = try record("IK_PARITY_STABLE_AUDIO_TINY", mode: "stable_audio_tiny")
        let net = NFKMLXStableAudioProjectionNet(.tiny)
        try NFKMLXWeights.apply(weights(arrays, prefix: "proj::").map { ($0, $1) }, to: net, verifyShapes: true)
        let seconds = net.seconds(start: try XCTUnwrap(arrays["proj_start"]), end: try XCTUnwrap(arrays["proj_end"]))
        let seams = [("text", net.projectedText(try XCTUnwrap(arrays["proj_text"])), try XCTUnwrap(arrays["proj_text_out"])),
                     ("start", seconds.start, try XCTUnwrap(arrays["proj_start_out"])),
                     ("end", seconds.end, try XCTUnwrap(arrays["proj_end_out"]))]
        for (name, mine, reference) in seams {
            XCTAssertEqual(mine.shape, reference.shape, name)
            let similarity = cosine(mine, reference)
            print("VALIDATION PARITY stable_audio_tiny projection \(name): cosine \(similarity), relative \(relativeError(mine, reference))")
            XCTAssertGreaterThan(similarity, 0.999999, name)
        }
    }

    // The original T5's ReLU feed-forward (`DenseReluDense.wi` → ReLU → `wo`) under a padding mask.
    func testTheTinyReLUT5EncoderMatchesTheReference() throws {
        try requireMLXRuntime()
        let arrays = try record("IK_PARITY_STABLE_AUDIO_TINY", mode: "stable_audio_tiny")
        let net = NFKMLXT5Encoder.makeNet(.tinyReLU)
        let tensors = weights(arrays, prefix: "t5::").filter { $0.key != "encoder.embed_tokens.weight" }
        try NFKMLXWeights.apply(tensors.map { ($0, $1) }, to: net, verifyShapes: true)
        let states = net(try XCTUnwrap(arrays["t5_ids"]), mask: try XCTUnwrap(arrays["t5_mask"]))
        let reference = try XCTUnwrap(arrays["t5_states"])
        let similarity = cosine(states, reference)
        print("VALIDATION PARITY stable_audio_tiny t5 relu: cosine \(similarity), relative \(relativeError(states, reference))")
        XCTAssertGreaterThan(similarity, 0.999999)
    }

    // The sampler's schedules at 8 steps (the lower-order final steps apply below 15) and 100, and an
    // 8-step trajectory over the reference's recorded model outputs and noise.
    func testTheSamplerMatchesTheReference() throws {
        try requireMLXRuntime()
        let arrays = try record("IK_PARITY_STABLE_AUDIO_TINY", mode: "stable_audio_tiny")
        for steps in [8, 100] {
            let scheduler = NFKMLXCosineDPMSolverScheduler(steps: steps)
            let sigmas = try XCTUnwrap(arrays["sched_sigmas_\(steps)"]).asArray(Float.self)
            let timesteps = try XCTUnwrap(arrays["sched_timesteps_\(steps)"]).asArray(Float.self)
            XCTAssertEqual(scheduler.sigmas.count, sigmas.count)
            for (mine, theirs) in zip(scheduler.sigmas, sigmas) {
                XCTAssertEqual(mine, theirs, accuracy: max(abs(theirs) * 1e-6, 1e-7), "σ at \(steps) steps")
            }
            for (mine, theirs) in zip(scheduler.timesteps, timesteps) {
                XCTAssertEqual(mine, theirs, accuracy: 1e-6, "timestep at \(steps) steps")
            }
        }
        let scheduler = NFKMLXCosineDPMSolverScheduler(steps: 8)
        var sample = nlc(try XCTUnwrap(arrays["sched_sample"]))
        var worst = 0.0
        for index in 0 ..< 8 {
            sample = scheduler.step(modelOutput: nlc(try XCTUnwrap(arrays["sched_output_\(index)"])), sample: sample,
                                    noise: nlc(try XCTUnwrap(arrays["sched_noise_\(index)"])))
            worst = max(worst, relativeError(sample, nlc(try XCTUnwrap(arrays["sched_prev_\(index)"]))))
        }
        print("VALIDATION PARITY stable_audio_tiny sampler: worst relative step error \(worst)")
        XCTAssertLessThan(worst, 1e-5)
    }

    // MARK: - Released components

    private func releaseDirectory() throws -> URL {
        guard let path = config["IK_VAL_STABLE_AUDIO_OPEN"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_STABLE_AUDIO_OPEN (the stabilityai/stable-audio-open-1.0 release)")
        }
        return URL(fileURLWithPath: path)
    }

    private func textStage(_ directory: URL) throws -> NFKStableAudioTextStage {
        let encoder = NFKMLXT5Encoder.makeNet(try NFKMLXT5Encoder.configuration(
            fromHuggingFace: directory.appendingPathComponent("text_encoder/config.json")))
        try NFKMLXT5Encoder.loadWeights(into: encoder, from: directory.appendingPathComponent("text_encoder"),
                                        dtype: .float32)
        let projection = NFKMLXStableAudioProjectionNet(try .configuration(
            fromHuggingFace: directory.appendingPathComponent("projection_model/config.json")))
        try NFKMLXStableAudio.loadProjectionWeights(
            into: projection, from: directory.appendingPathComponent("projection_model/diffusion_pytorch_model.safetensors"))
        return NFKStableAudioTextStage(encoder: encoder, projection: projection)
    }

    private func pipeline(_ directory: URL) throws -> NFKMLXStableAudioPipeline {
        let transformer = NFKMLXStableAudioDiTNet(try .configuration(
            fromHuggingFace: directory.appendingPathComponent("transformer/config.json")))
        try NFKMLXStableAudio.loadTransformerWeights(into: transformer,
                                                     fromDirectory: directory.appendingPathComponent("transformer"))
        let vae = NFKMLXOobleckAutoencoderNet(try .configuration(
            fromHuggingFace: directory.appendingPathComponent("vae/config.json")))
        try NFKMLXStableAudio.loadAutoencoderWeights(
            into: vae, from: directory.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"))
        let scheduler = try NFKMLXCosineDPMSolverScheduler.Configuration.configuration(
            fromHuggingFace: directory.appendingPathComponent("scheduler/scheduler_config.json"))
        return NFKMLXStableAudioPipeline(transformer: transformer, autoencoder: vae, schedulerConfiguration: scheduler)
    }

    private func segmenter(_ directory: URL) throws -> NFKMLXSentencePieceSegmenter {
        NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: directory.appendingPathComponent("tokenizer/spiece.model")))
    }

    // The release tokenizer's ids for every prompt in the record, token for token.
    func testTheReleaseTokenizationMatchesTheReference() throws {
        let directory = try releaseDirectory()
        let arrays = try record("IK_PARITY_STABLE_AUDIO_COMPONENTS", mode: "stable_audio_components")
        let length = try NFKMLXStableAudioGenerator.tokenizerLength(
            directory.appendingPathComponent("tokenizer/tokenizer_config.json"))
        XCTAssertEqual(length, Int(try XCTUnwrap(arrays["model_max_length"]).item(Int32.self)))
        let segmenter = try segmenter(directory)
        var index = 0
        while let ids = arrays["ids_\(index)"], let mask = arrays["mask_\(index)"] {
            let prompt = NFKMLXT5Prompt(Self.prompts[index], segmenter: segmenter, length: length)
            XCTAssertEqual(prompt.ids, ids.asArray(Int32.self).map(Int.init), "prompt \(index) ids")
            XCTAssertEqual(prompt.mask, mask.asArray(Int32.self).map(Int.init), "prompt \(index) mask")
            index += 1
        }
        XCTAssertEqual(index, Self.prompts.count)
    }

    /// `run_reference.py`'s `STABLE_AUDIO_PROMPTS`.
    static let prompts = [
        "The sound of a hammer hitting a wooden surface.",
        "A gentle piano melody with soft rain in the background, 90 BPM",
        "  Techno   drums\tand a\nwarm bass line  ",
        "Ambient synth pad, C# minor — slow, dreamy; café vibes!",
        "",
    ]

    // t5-base on a prompt, the seconds conditioners (one value past the range), the transformer at three
    // noise levels and on the zeroed unconditional branch, the autoencoder's posterior and decode, and the
    // release scheduler's schedules.
    func testTheReleasedComponentsMatchTheReference() throws {
        try requireMLXRuntime()
        let directory = try releaseDirectory()
        let arrays = try record("IK_PARITY_STABLE_AUDIO_COMPONENTS", mode: "stable_audio_components")
        func check(_ name: String, _ mine: MLXArray, _ reference: MLXArray, bar: Double = 0.999999) {
            XCTAssertEqual(mine.shape, reference.shape, name)
            let similarity = cosine(mine, reference)
            print("VALIDATION PARITY stable_audio \(name): cosine \(similarity), relative \(relativeError(mine, reference))")
            XCTAssertGreaterThan(similarity, bar, name)
        }

        try autoreleasepool {
            let text = try textStage(directory)
            let ids = try XCTUnwrap(arrays["ids_1"]).reshaped([1, -1]), mask = try XCTUnwrap(arrays["mask_1"]).reshaped([1, -1])
            check("t5 states", text.encoder(ids, mask: mask)[0], try XCTUnwrap(arrays["t5_states"]))
            let seconds = text.projection.seconds(start: try XCTUnwrap(arrays["seconds_start_in"]),
                                                  end: try XCTUnwrap(arrays["seconds_end_in"]))
            check("seconds start", seconds.start, try XCTUnwrap(arrays["seconds_start"]))
            check("seconds end", seconds.end, try XCTUnwrap(arrays["seconds_end"]))
        }
        NFKMLXGPU.clearCache()

        let pipeline = try pipeline(directory)
        XCTAssertEqual(pipeline.transformer.configuration, .release)
        XCTAssertEqual(pipeline.autoencoder.configuration, .release)
        let context = try XCTUnwrap(arrays["dit_context"]).expandedDimensions(axis: 0)
        let global = try XCTUnwrap(arrays["dit_global"]).expandedDimensions(axis: 0)
        for name in ["high", "mid", "low"] {
            let input = try XCTUnwrap(arrays["dit_input_\(name)"]).transposed(1, 0).expandedDimensions(axis: 0)
            let prediction = pipeline.transformer(input, timestep: try XCTUnwrap(arrays["dit_timestep_\(name)"]),
                                                  context: context, global: global)
            check("transformer \(name)", prediction[0], try XCTUnwrap(arrays["dit_velocity_\(name)"]).transposed(1, 0))
        }
        let input = try XCTUnwrap(arrays["dit_input_mid"]).transposed(1, 0).expandedDimensions(axis: 0)
        let unconditional = pipeline.transformer(input, timestep: try XCTUnwrap(arrays["dit_timestep_mid"]),
                                                 context: MLXArray.zeros(like: context), global: global)
        check("transformer unconditional", unconditional[0],
              try XCTUnwrap(arrays["dit_velocity_unconditional"]).transposed(1, 0))

        let waveform = try XCTUnwrap(arrays["vae_waveform"]).transposed(1, 0).expandedDimensions(axis: 0)
        let posterior = pipeline.autoencoder.encode(waveform)
        check("autoencoder mean", posterior.mean[0], try XCTUnwrap(arrays["vae_mean"]).transposed(1, 0))
        check("autoencoder std", posterior.standardDeviation[0], try XCTUnwrap(arrays["vae_std"]).transposed(1, 0))
        check("autoencoder decoded", pipeline.autoencoder.decode(posterior.mean)[0],
              try XCTUnwrap(arrays["vae_decoded"]).transposed(1, 0))

        for steps in [8, 100] {
            let scheduler = NFKMLXCosineDPMSolverScheduler(steps: steps, configuration: pipeline.schedulerConfiguration)
            let sigmas = try XCTUnwrap(arrays["sched_sigmas_\(steps)"]).asArray(Float.self)
            XCTAssertEqual(scheduler.sigmas.count, sigmas.count)
            for (mine, theirs) in zip(scheduler.sigmas, sigmas) {
                XCTAssertEqual(mine, theirs, accuracy: max(abs(theirs) * 1e-6, 1e-7), "release σ at \(steps) steps")
            }
        }
    }

    // MARK: - End to end

    // Both pipeline runs, each from the reference's own initial latents and per-step noise: the
    // conditioning this side builds from the prompt, the latents after every step, and the cropped
    // waveform. Run `b_` guides against a negative prompt and starts from an initial waveform, encoded and
    // sampled with the reference's own posterior draw.
    func testThePipelineMatchesTheReferenceStepByStep() throws {
        try requireMLXRuntime()
        let directory = try releaseDirectory()
        let arrays = try record("IK_PARITY_STABLE_AUDIO_PIPELINE", mode: "stable_audio_pipeline")
        let length = try NFKMLXStableAudioGenerator.tokenizerLength(
            directory.appendingPathComponent("tokenizer/tokenizer_config.json"))
        let segmenter = try segmenter(directory)
        let runs: [(prefix: String, prompt: String, negative: String?)] = [
            ("a_", "A gentle piano melody with soft rain in the background, 90 BPM", nil),
            ("b_", "Techno drums and a warm bass line", "low quality, distorted"),
        ]
        var conditioning = [String: (context: MLXArray, global: MLXArray, negative: MLXArray?)]()
        try autoreleasepool {
            let text = try textStage(directory)
            for run in runs {
                let seconds = try XCTUnwrap(arrays["\(run.prefix)seconds"]).asArray(Float.self)
                let positive = text.conditioning(NFKMLXT5Prompt(run.prompt, segmenter: segmenter, length: length),
                                                 start: seconds[0], end: seconds[1])
                let negative = run.negative.map {
                    text.conditioning(NFKMLXT5Prompt($0, segmenter: segmenter, length: length),
                                      start: seconds[0], end: seconds[1]).context
                }
                eval(positive.context, positive.global)
                conditioning[run.prefix] = (positive.context, positive.global, negative)
            }
        }
        NFKMLXGPU.clearCache()

        let pipeline = try pipeline(directory)
        for run in runs {
            let p = run.prefix
            let condition = try XCTUnwrap(conditioning[p])
            let referenceContext = try XCTUnwrap(arrays["\(p)context"])
            let pairedContext = concatenated([condition.negative ?? MLXArray.zeros(like: condition.context),
                                              condition.context], axis: 0)
            let contextSimilarity = cosine(pairedContext, referenceContext)
            print("VALIDATION PARITY stable_audio pipeline \(p)context: cosine \(contextSimilarity)")
            XCTAssertGreaterThan(contextSimilarity, 0.999999)
            XCTAssertGreaterThan(cosine(concatenated([condition.global, condition.global], axis: 0),
                                        try XCTUnwrap(arrays["\(p)global"])), 0.999999)

            let initial = try XCTUnwrap(arrays["\(p)initial"]).transposed(1, 0).expandedDimensions(axis: 0)
            var encoded: MLXArray?
            if let audio = arrays["\(p)initial_audio"] {
                let noise = try XCTUnwrap(arrays["\(p)posterior_noise"]).transposed(1, 0).expandedDimensions(axis: 0)
                encoded = pipeline.encodedLatents(audio.transposed(1, 0), noise: noise)
                let scale = NFKMLXCosineDPMSolverScheduler(steps: 8, configuration: pipeline.schedulerConfiguration).inputScale
                let firstInput = (encoded! + initial * pipeline.initialNoiseScale) * scale
                let reference = try XCTUnwrap(arrays["\(p)first_input"])[0].transposed(1, 0)
                let similarity = cosine(firstInput[0], reference)
                print("VALIDATION PARITY stable_audio pipeline \(p)initial-audio input: cosine \(similarity), relative \(relativeError(firstInput[0], reference))")
                XCTAssertGreaterThan(similarity, 0.999999)
            }
            var stepSimilarities = [Double]()
            let latents = try XCTUnwrap(pipeline.latents(
                context: condition.context, global: condition.global, negativeContext: condition.negative, steps: 8,
                guidance: 7, initial: initial, encoded: encoded,
                noise: { index in arrays["\(p)noise_\(index)"]!.transposed(1, 0).expandedDimensions(axis: 0) },
                progress: { index, sample in
                    if let reference = arrays["\(p)latents_\(index)"] {
                        stepSimilarities.append(self.cosine(sample[0], reference.transposed(1, 0)))
                    }
                    return true
                }))
            XCTAssertEqual(stepSimilarities.count, 8)
            print("VALIDATION PARITY stable_audio pipeline \(p)latents per step: \(stepSimilarities)")
            for (index, similarity) in stepSimilarities.enumerated() {
                XCTAssertGreaterThan(similarity, 0.99999, "\(p)step \(index)")
            }

            let seconds = try XCTUnwrap(arrays["\(p)seconds"]).asArray(Float.self)
            let waveform = pipeline.waveform(latents, start: Double(seconds[0]), end: Double(seconds[1]))
            let reference = try XCTUnwrap(arrays["\(p)audio"]).transposed(1, 0)
            XCTAssertEqual(waveform.shape, reference.shape)
            let audioSimilarity = cosine(waveform, reference)
            print("VALIDATION PARITY stable_audio pipeline \(p)waveform: cosine \(audioSimilarity), relative \(relativeError(waveform, reference))")
            XCTAssertGreaterThan(audioSimilarity, 0.99999)
            NFKMLXGPU.clearCache()
        }
    }
}
