//
//  NFKMLXWanTrainingTests.swift
//  InferKitMLXTests
//
//  Wan LoRA fine-tuning as finetrainers, DiffSynth-Studio, and musubi-tuner train it: each trainer's draws,
//  its objective's prediction, loss, and adapter gradients, and its saved file merged back, against the
//  trainer's own code where the validation store holds its record (`run_reference.py wan_lora_finetrainers`,
//  `wan_lora_diffsynth`, `wan_lora_musubi`); and, weight-free, the presets' schedules and names, a run
//  that moves only the adapters, and a save in each layout that merges back to the trained network.
//  Runs where MLX has a Metal library (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
@testable import InferKitMLX

final class NFKMLXWanTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    // MARK: Weight-free

    func testTorchLinspaceFusesEachEntryFromItsNearerEnd() {
        let values = NFKTorchLinspace.float32(from: 1, to: 0, count: 1001)
        XCTAssertEqual(values.first, 1)
        XCTAssertEqual(values.last, 0)
        // torch.linspace(1, 0, 1001)[30] is 0.96999997, the fused 1 + step·30; an unfused product and sum give 0.97.
        XCTAssertEqual(values[30], 0.96999997)
        XCTAssertEqual(values[997], Float(0).addingProduct(Float(1) / 1000, 3))
    }

    func testThePresetsWarmUpAsTheirTrainersDo() {
        let finetrainers = NFKMLXWanTrainingPreset.finetrainers.schedule
        XCTAssertEqual(finetrainers.multiplier(0), 0, "finetrainers' warm-up multiplies by step / warmup")
        XCTAssertEqual(finetrainers.multiplier(500), 0.5)
        XCTAssertEqual(finetrainers.multiplier(5000), 1)
        let diffSynth = NFKMLXWanTrainingPreset.diffSynth.schedule
        XCTAssertEqual(diffSynth.multiplier(0), 0.2, "DiffSynth-Studio's warm-up multiplies by (step + 1) / warmup")
        XCTAssertEqual(diffSynth.multiplier(4), 1)
        XCTAssertEqual(NFKMLXWanTrainingPreset.musubi.schedule.multiplier(0), 1)
    }

    func testTheTargetsNameTheTrainersLayers() {
        let attention = NFKMLXWanLoRATargets.attention, both = NFKMLXWanLoRATargets.attentionAndFeedForward
        XCTAssertTrue(attention.contains("blocks.0.attn1.to_q"))
        XCTAssertTrue(attention.contains("blocks.3.attn2.to_out.0"))
        XCTAssertFalse(attention.contains("blocks.0.ffn.net.0.proj"))
        XCTAssertTrue(both.contains("blocks.0.ffn.net.0.proj"))
        XCTAssertTrue(both.contains("blocks.0.ffn.net.2"))
        XCTAssertFalse(both.contains("proj_out"))
        XCTAssertFalse(both.contains("condition_embedder.text_embedder.linear_1"))
    }

    func testOriginalNamesInvertTheReadersMapping() throws {
        for path in ["blocks.1.attn1.to_q", "blocks.1.attn1.to_out.0", "blocks.0.attn2.to_v", "blocks.0.ffn.net.0.proj",
                     "blocks.2.ffn.net.2"] {
            let original = try NFKMLXWanVideoGenerator.originalName(path)
            XCTAssertEqual(NFKMLXWanLoRA.textToVideoPath(original: original), path)
        }
        XCTAssertThrowsError(try NFKMLXWanVideoGenerator.originalName("proj_out"))
    }

    func testDiffSynthTableWeightsAverageOne() {
        let table = NFKMLXWanSigmaSampler.diffSynthTable(shift: 5)
        XCTAssertEqual(table.sigmas.count, 1000)
        XCTAssertEqual(table.sigmas[0], 1)
        XCTAssertEqual(table.weights.reduce(0, +) / 1000, 1, accuracy: 1e-5)
        XCTAssertEqual(table.weights.min()!, 0)
    }

    private func tinyExamples(count: Int) -> [NFKMLXWanTrainingExample] {
        let config = NFKMLXWanConfiguration.tiny
        return (0 ..< count).map { index in
            let key = MLXRandom.key(UInt64(100 + index))
            let keys = MLXRandom.split(key: key, into: 3)
            return NFKMLXWanTrainingExample(
                latent: MLXRandom.normal([config.inChannels, 3, 4, 4], key: keys[0]),
                logVariance: MLXRandom.normal([config.inChannels, 3, 4, 4], key: keys[1]) * 0.5 - 2,
                text: MLXRandom.normal([8, config.textDim], key: keys[2]))
        }
    }

    func testFineTuningMovesOnlyTheAdaptersAndLowersTheLoss() throws {
        try requireMLXRuntime()
        for preset in [NFKMLXWanTrainingPreset.finetrainers, .diffSynth, .musubi] {
            MLXRandom.seed(7)
            let net = NFKMLXWanTransformerNet(.tiny)
            let before = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1 * 1) })
            eval(Array(before.values))
            let examples = tinyExamples(count: 2)
            var fast = preset
            fast.rank = 4
            fast.alpha = preset.alpha == 1 ? 1 : 4
            fast.objective.sampler = preset.objective.sampler
            let losses = try NFKMLXWanVideoGenerator.fineTune(
                net, examples: { _ in examples }, preset: fast, optimizer: Adam(learningRate: 1e-2), steps: 40, seed: 3,
                learningRateSchedule: .constant)
            XCTAssertEqual(losses.count, 40)
            let early = losses.prefix(5).reduce(0, +), late = losses.suffix(5).reduce(0, +)
            XCTAssertLessThan(late, early, "\(fast.format): the loss falls")
            for (key, value) in net.parameters().flattened() {
                let lora = key.hasSuffix(".lora_a") || key.hasSuffix(".lora_b")
                guard let original = before[key] else {
                    XCTAssertTrue(lora, "\(key) is new and not an adapter")
                    continue
                }
                XCTAssertEqual(abs(value - original).max().item(Float.self), 0, "\(fast.format): \(key) is frozen")
            }
            let adapted = net.leafModules().flattened().filter { $0.1 is NFKMLXLoRALinear }.map(\.0)
            XCTAssertEqual(adapted.count, fast.targets == .attention ? 16 : 20)
        }
    }

    func testEachLayoutSavesWhatMergesBackToTheTrainedNetwork() throws {
        try requireMLXRuntime()
        let examples = tinyExamples(count: 1)
        let input = MLXRandom.normal([4, 3, 4, 4], key: MLXRandom.key(9))
        for preset in [NFKMLXWanTrainingPreset.finetrainers, .diffSynth, .musubi] {
            MLXRandom.seed(11)
            let base = NFKMLXWanTransformerNet(.tiny)
            let weights = base.parameters().flattened().map { ($0.0, $0.1 * 1) }
            eval(weights.map(\.1))
            var fast = preset
            fast.rank = 4
            fast.alpha = preset.alpha == 1 ? 1 : 8
            try NFKMLXWanVideoGenerator.fineTune(base, examples: { _ in examples }, preset: fast,
                                                 optimizer: Adam(learningRate: 1e-2), steps: 3, seed: 5,
                                                 learningRateSchedule: .constant)
            let trained = base(input, text: examples[0].text, t: MLXArray([Float(500)]))

            let url = FileManager.default.temporaryDirectory.appendingPathComponent("wan-train-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            try NFKMLXWanVideoGenerator.saveAdapters(of: base, to: url, preset: fast)
            let fresh = NFKMLXWanTransformerNet(.tiny)
            try NFKMLXWeights.apply(weights, to: fresh, verifyShapes: true)
            let report = try NFKMLXWanLoRA.merge(adapterURL: url, into: fresh)
            XCTAssertEqual(report.mergedLayers, fast.targets == .attention ? 16 : 20, "\(fast.format)")
            XCTAssertTrue(report.droppedKeys.isEmpty)
            let merged = fresh(input, text: examples[0].text, t: MLXArray([Float(500)]))
            XCTAssertLessThan(abs(merged - trained).max().item(Float.self), 1e-5, "\(fast.format) merges to the trained network")
        }
    }

    // MARK: Against each trainer

    /// The oracles' geometry: width 48 over 2 heads, 2 blocks, a 48-wide feed-forward, 32-wide text.
    private let geometry = NFKMLXWanConfiguration(inChannels: 4, heads: 2, headDim: 24, layers: 2, ffnDim: 48,
                                                  textDim: 32, patchSize: [1, 2, 2])

    private struct Record {
        let arrays: [String: MLXArray]
        subscript(_ key: String) -> MLXArray { arrays[key]! }
        func section(_ prefix: String) -> [String: MLXArray] {
            Dictionary(uniqueKeysWithValues: arrays.compactMap { key, value in
                key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil
            })
        }
    }

    private func record(_ variable: String, _ mode: String) throws -> Record {
        guard let path = NFKMLXValidationConfig.environment[variable] else {
            throw XCTSkip("set \(variable) (run_reference.py \(mode))")
        }
        return Record(arrays: try MLX.loadArrays(url: URL(fileURLWithPath: path)))
    }

    private func baseWeights(_ record: Record) -> [(String, MLXArray)] {
        record.section("w::").map { ($0.key, $0.value.ndim == 5 ? $0.value.transposed(0, 2, 3, 4, 1) : $0.value) }
    }

    /// The oracle's network with its adapters (rank 4, alpha 8) at the recorded values.
    private func adaptedNetwork(_ record: Record, targets: NFKMLXWanLoRATargets) throws -> NFKMLXWanTransformerNet {
        let net = NFKMLXWanTransformerNet(geometry)
        let adapted = try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 8) { path, _ in targets.contains(path) }
        let adapters = record.section("lora::")
        XCTAssertEqual(adapted * 2, adapters.count, "the oracle adapts the preset's targets")
        try NFKMLXWeights.apply(baseWeights(record) + adapters.map { ($0.key, $0.value) }, to: net, verifyShapes: true)
        return net
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).flattened(), y = b.asType(.float32).flattened()
        let dot = Double((x * y).sum().item(Float.self))
        return dot / (Double(sqrt((x * x).sum()).item(Float.self)) * Double(sqrt((y * y).sum()).item(Float.self)))
    }

    /// The objective's gradient for each adapter against the recorded one; returns the lowest cosine.
    private func assertGradientsMatch(_ net: NFKMLXWanTransformerNet, _ record: Record,
                                      loss: @escaping (NFKMLXWanTransformerNet) -> MLXArray) -> Double {
        net.freeze()
        for (_, layer) in net.leafModules().flattened() where layer is NFKMLXLoRALinear {
            layer.unfreeze(recursive: false, keys: ["lora_a", "lora_b"])
        }
        let lossAndGradient = valueAndGrad(model: net) { net, _ in [loss(net)] }
        let gradients = Dictionary(uniqueKeysWithValues: lossAndGradient(net, []).1.flattened())
        let recorded = record.section("grad::")
        XCTAssertEqual(Set(gradients.keys), Set(recorded.keys), "a gradient for every adapter and nothing else")
        var lowest = 1.0
        for (key, reference) in recorded {
            guard let ours = gradients[key] else { continue }
            let c = cosine(ours, reference)
            lowest = min(lowest, c)
            XCTAssertGreaterThan(c, 0.999999, key)
        }
        return lowest
    }

    /// Writes the trainer's saved file (`file::` keys, `file_metadata::` strings) back as a safetensors file.
    private func trainerFile(_ record: Record) throws -> URL {
        var metadata = [String: String]()
        for (key, bytes) in record.section("file_metadata::") {
            metadata[key] = String(decoding: bytes.asArray(UInt8.self), as: UTF8.self)
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wan-trainer-\(UUID().uuidString).safetensors")
        try MLX.save(arrays: record.section("file::"), metadata: metadata, url: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    /// Our save of `net` in `preset`'s layout against the trainer's own file, key for key.
    private func assertSavesTheTrainersFile(_ net: NFKMLXWanTransformerNet, _ record: Record,
                                            preset: NFKMLXWanTrainingPreset) throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wan-ours-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        var preset = preset
        preset.alpha = 8
        try NFKMLXWanVideoGenerator.saveAdapters(of: net, to: url, preset: preset)
        let ours = try MLX.loadArrays(url: url)
        let theirs = record.section("file::")
        XCTAssertEqual(Set(ours.keys), Set(theirs.keys), "\(preset.format): the trainer's key set")
        for (key, value) in theirs {
            guard let mine = ours[key] else { continue }
            XCTAssertLessThan(abs(mine.asType(.float32).reshaped(value.shape) - value).max().item(Float.self), 1e-6, key)
        }
    }

    /// Every parameter of a fresh network merged with the trainer's file against the trainer's own merge.
    private func assertMergesAsTheTrainerDoes(_ record: Record) throws -> Float {
        let net = NFKMLXWanTransformerNet(geometry)
        let base = baseWeights(record)
        try NFKMLXWeights.apply(base, to: net, verifyShapes: true)
        let report = try NFKMLXWanLoRA.merge(adapterURL: try trainerFile(record), into: net)
        XCTAssertTrue(report.droppedKeys.isEmpty)
        let fused = record.section("fused::"), original = Dictionary(uniqueKeysWithValues: base)
        XCTAssertEqual(report.mergedLayers, fused.count)
        var worst: Float = 0
        for (key, value) in net.parameters().flattened() {
            let expected = fused[key] ?? original[key]!
            let gap = abs(value - expected).max().item(Float.self)
            worst = max(worst, gap)
            XCTAssertLessThan(gap, 1e-6, key)
        }
        return worst
    }

    func testFinetrainersObjectiveMatchesTheTrainer() throws {
        try requireMLXRuntime()
        let record = try record("IK_PARITY_WAN_LORA_FINETRAINERS", "wan_lora_finetrainers")
        let uniform = NFKMLXWanSigmaSampler.finetrainers(.uniform)
        let u = record["sampler.u"].asArray(Float.self)
        let sigmas = record["sampler.sigma"].asArray(Float.self), timesteps = record["sampler.timestep"].asArray(Float.self)
        for index in u.indices {
            let draw = uniform.draw(u[index])
            XCTAssertEqual(draw.sigma, sigmas[index], "draw \(index)")
            XCTAssertEqual(draw.timestep, timesteps[index], "draw \(index)")
        }
        let draw = uniform.draw(record["u"].item(Float.self))
        XCTAssertEqual(draw.sigma, record["sigma"].item(Float.self))
        XCTAssertEqual(draw.timestep, record["timestep"].item(Float.self))

        let net = try adaptedNetwork(record, targets: NFKMLXWanTrainingPreset.finetrainers.targets)
        let objective = NFKMLXWanTrainingPreset.finetrainers.objective
        let example = NFKMLXWanTrainingExample(latent: record["latent"], logVariance: record["log_variance"],
                                               text: record["text"])
        let clean = objective.clean(example, epsilon: record["epsilon"])
        let noisy = (1 - draw.sigma) * clean + draw.sigma * record["noise"]
        let noisyGap = abs(noisy - record["noisy"]).max().item(Float.self)
        XCTAssertLessThan(noisyGap, 1e-6)
        let prediction = net(record["noisy"], text: example.text, t: MLXArray([draw.timestep]))
        let predictionCosine = cosine(prediction, record["prediction"])
        XCTAssertGreaterThan(predictionCosine, 0.999999)
        let loss = objective.score(net, example: example, draw: draw, noise: record["noise"], epsilon: record["epsilon"])
        XCTAssertEqual(loss.item(Float.self), record["loss"].item(Float.self), accuracy: 1e-5 * record["loss"].item(Float.self))
        let gradientCosine = assertGradientsMatch(net, record) {
            objective.score($0, example: example, draw: draw, noise: record["noise"], epsilon: record["epsilon"])
        }

        try assertSavesTheTrainersFile(net, record, preset: .finetrainers)
        let fresh = NFKMLXWanTransformerNet(geometry)
        try NFKMLXWeights.apply(baseWeights(record), to: fresh, verifyShapes: true)
        try NFKMLXWanLoRA.merge(adapterURL: try trainerFile(record), into: fresh)
        let adaptedCosine = cosine(fresh(record["noisy"], text: example.text, t: MLXArray([draw.timestep])),
                                   record["adapted"])
        XCTAssertGreaterThan(adaptedCosine, 0.999999)
        print("VALIDATION PARITY wan-lora-finetrainers: noisy gap \(noisyGap), prediction cosine \(predictionCosine), "
              + "loss \(loss.item(Float.self)) vs \(record["loss"].item(Float.self)), lowest gradient cosine "
              + "\(gradientCosine), merged-file cosine \(adaptedCosine)")
    }

    func testDiffSynthObjectiveMatchesTheTrainer() throws {
        try requireMLXRuntime()
        let record = try record("IK_PARITY_WAN_LORA_DIFFSYNTH", "wan_lora_diffsynth")
        let table = NFKMLXWanSigmaSampler.diffSynthTable(shift: 5)
        XCTAssertEqual(table.sigmas, record["sampler.sigma"].asArray(Float.self))
        XCTAssertEqual(table.timesteps, record["sampler.timestep"].asArray(Float.self))
        let weightGap = zip(table.weights, record["sampler.weight"].asArray(Float.self)).map { abs($0 - $1) }.max()!
        XCTAssertLessThan(weightGap, 1e-6)

        let preset = NFKMLXWanTrainingPreset.diffSynth
        let draw = preset.objective.sampler.draw(record["index"].item(Float.self), weighting: preset.objective.weighting)
        XCTAssertEqual(draw.sigma, record["sigma"].item(Float.self))
        XCTAssertEqual(draw.timestep, record["timestep"].item(Float.self))
        XCTAssertEqual(draw.weight, record["weight"].item(Float.self), accuracy: 1e-7)

        let net = try adaptedNetwork(record, targets: preset.targets)
        let example = NFKMLXWanTrainingExample(latent: record["latent"], text: record["text"])
        let noisy = (1 - draw.sigma) * example.latent + draw.sigma * record["noise"]
        let predictionCosine = cosine(net(noisy, text: example.text, t: MLXArray([draw.timestep])), record["prediction"])
        XCTAssertGreaterThan(predictionCosine, 0.999999)
        let loss = preset.objective.score(net, example: example, draw: draw, noise: record["noise"], epsilon: nil)
        XCTAssertEqual(loss.item(Float.self), record["loss"].item(Float.self), accuracy: 1e-5 * record["loss"].item(Float.self))
        let gradientCosine = assertGradientsMatch(net, record) {
            preset.objective.score($0, example: example, draw: draw, noise: record["noise"], epsilon: nil)
        }

        try assertSavesTheTrainersFile(net, record, preset: preset)
        let mergeGap = try assertMergesAsTheTrainerDoes(record)
        print("VALIDATION PARITY wan-lora-diffsynth: table weight gap \(weightGap), prediction cosine \(predictionCosine), "
              + "loss \(loss.item(Float.self)) vs \(record["loss"].item(Float.self)), lowest gradient cosine "
              + "\(gradientCosine), merge gap \(mergeGap)")
    }

    func testMusubiObjectiveMatchesTheTrainer() throws {
        try requireMLXRuntime()
        let record = try record("IK_PARITY_WAN_LORA_MUSUBI", "wan_lora_musubi")
        let preset = NFKMLXWanTrainingPreset.musubi
        let z = record["sampler.z"].asArray(Float.self), shifted = record["sampler.timestep"].asArray(Float.self)
        var shiftGap: Float = 0
        for index in z.indices {
            shiftGap = max(shiftGap, abs(preset.objective.sampler.draw(z[index]).timestep - shifted[index]))
        }
        XCTAssertLessThan(shiftGap, 1e-4, "float32 timesteps near 1000 hold 6e-5")
        let table = NFKMLXWanSigmaSampler.musubiTable(.uniform, minimum: 0, maximum: 1000)
        let u = record["table.u"].asArray(Float.self)
        let tableTimesteps = record["table.timestep"].asArray(Float.self), tableSigmas = record["table.sigma"].asArray(Float.self)
        for index in u.indices {
            let draw = table.draw(u[index])
            XCTAssertEqual(draw.timestep, tableTimesteps[index], "table draw \(index)")
            XCTAssertEqual(draw.sigma, tableSigmas[index], accuracy: 1e-6, "table draw \(index)")
        }

        let draw = preset.objective.sampler.draw(record["z"].item(Float.self))
        XCTAssertEqual(draw.timestep, record["timestep"].item(Float.self), accuracy: 1e-4)
        let net = try adaptedNetwork(record, targets: preset.targets)
        let example = NFKMLXWanTrainingExample(latent: record["latent"], text: record["text"])
        let noisy = (1 - draw.sigma) * example.latent + draw.sigma * record["noise"]
        let noisyGap = abs(noisy - record["noisy"]).max().item(Float.self)
        XCTAssertLessThan(noisyGap, 1e-6)
        let prediction = net(noisy, text: example.text, t: MLXArray([draw.timestep]))
        let predictionCosine = cosine(prediction, record["prediction"])
        XCTAssertGreaterThan(predictionCosine, 0.999999)
        // musubi's own attention rounds its inputs to bfloat16; the record proper attends in float32.
        let asRunCosine = cosine(prediction, record["as_run.prediction"])
        XCTAssertGreaterThan(asRunCosine, 0.9999)
        let loss = preset.objective.score(net, example: example, draw: draw, noise: record["noise"], epsilon: nil)
        XCTAssertEqual(loss.item(Float.self), record["loss"].item(Float.self), accuracy: 1e-5 * record["loss"].item(Float.self))
        let gradientCosine = assertGradientsMatch(net, record) {
            preset.objective.score($0, example: example, draw: draw, noise: record["noise"], epsilon: nil)
        }

        try assertSavesTheTrainersFile(net, record, preset: preset)
        let mergeGap = try assertMergesAsTheTrainerDoes(record)
        print("VALIDATION PARITY wan-lora-musubi: shift timestep gap \(shiftGap), prediction cosine \(predictionCosine) "
              + "(\(asRunCosine) against musubi's bfloat16 attention), "
              + "loss \(loss.item(Float.self)) vs \(record["loss"].item(Float.self)), lowest gradient cosine "
              + "\(gradientCosine), merge gap \(mergeGap)")
    }
}
