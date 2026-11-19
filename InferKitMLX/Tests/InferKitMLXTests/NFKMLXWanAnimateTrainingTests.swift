//
//  NFKMLXWanAnimateTrainingTests.swift
//  InferKitMLXTests
//
//  Wan 2.2 Animate 2 LoRA fine-tuning as DiffSynth-Studio trains it: the reference and generation passes in
//  one forward, the noised stack with the reference image's latent in the loss, and the adapters' gradients,
//  against DiffSynth's own model and loss where the validation store holds the record
//  (`run_reference.py wan_animate_lora_diffsynth`); and, weight-free, a run that moves only the adapters and
//  a save that merges back. Runs where MLX has a Metal library (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
@testable import InferKitMLX

final class NFKMLXWanAnimateTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).flattened(), y = b.asType(.float32).flattened()
        let dot = Double((x * y).sum().item(Float.self))
        return dot / (Double(sqrt((x * x).sum()).item(Float.self)) * Double(sqrt((y * y).sum()).item(Float.self)))
    }

    /// The oracle's geometry: width 48 over 2 heads, 2 blocks, a 96-wide feed-forward, 32-wide text of 16 tokens.
    private let geometry = NFKMLXWanAnimateConfiguration(inChannels: 36, outChannels: 16, dim: 48, heads: 2, layers: 2,
                                                         ffnDim: 96, textDim: 32, textLength: 16)

    private func tinyExample(_ key: UInt64) -> NFKMLXWanAnimateTrainingExample {
        let keys = MLXRandom.split(key: MLXRandom.key(key), into: 6)
        let config = NFKMLXWanAnimateConfiguration.tiny
        let latentChannels = config.outChannels, conditionChannels = config.inChannels - config.outChannels
        return NFKMLXWanAnimateTrainingExample(
            latent: MLXRandom.normal([latentChannels, 3, 4, 4], key: keys[0]),
            condition: MLXRandom.normal([conditionChannels, 3, 4, 4], key: keys[1]),
            reference: MLXRandom.normal([latentChannels, 2, 4, 4], key: keys[2]),
            referenceCondition: MLXRandom.normal([conditionChannels, 2, 4, 4], key: keys[3]),
            text: MLXRandom.normal([config.textLength, config.textDim], key: keys[4]),
            referenceText: MLXRandom.normal([config.textLength, config.textDim], key: keys[5]),
            imageFeatures: MLXRandom.normal([257, config.imageDim], key: keys[0]),
            referenceImageFeatures: MLXRandom.normal([257, config.imageDim], key: keys[1]),
            videoFrames: 5, videoArea: 32 * 32)
    }

    func testFineTuningMovesOnlyTheSelfAttentionAdapters() throws {
        try requireMLXRuntime()
        MLXRandom.seed(3)
        let net = NFKMLXWanAnimate.makeNet(.tiny)
        let before = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1 * 1) })
        eval(Array(before.values))
        var preset = NFKMLXWanAnimateTrainingPreset.diffSynth
        preset.rank = 4
        preset.alpha = 4
        let example = tinyExample(7)
        let losses = try NFKMLXWanAnimate.fineTune(net, examples: { _ in [example] }, preset: preset,
                                                   optimizer: Adam(learningRate: 1e-2), steps: 12, seed: 1,
                                                   learningRateSchedule: .constant)
        XCTAssertEqual(losses.count, 12)
        XCTAssertTrue(losses.allSatisfy(\.isFinite))
        for (key, value) in net.parameters().flattened() {
            guard let original = before[key] else {
                XCTAssertTrue(key.hasSuffix(".lora_a") || key.hasSuffix(".lora_b"), "\(key) is new and not an adapter")
                continue
            }
            XCTAssertEqual(abs(value - original).max().item(Float.self), 0, "\(key) is frozen")
        }
        let adapted = net.leafModules().flattened().filter { $0.1 is NFKMLXLoRAAdapter }.map(\.0)
        XCTAssertEqual(Set(adapted), Set((0 ..< 2).flatMap { block in
            ["to_q", "to_k", "to_v"].map { "blocks.\(block).self_attn.\($0)" }
        }))

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wan-animate-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWanAnimate.saveAdapters(of: net, to: url)
        let saved = try MLX.loadArrays(url: url)
        XCTAssertTrue(saved.keys.contains("blocks.0.block.self_attn.q.lora_A.default.weight"))
        let fresh = NFKMLXWanAnimate.makeNet(.tiny)
        try NFKMLXWeights.apply(before.map { ($0.key, $0.value) }, to: fresh, verifyShapes: true)
        let report = try NFKMLXWanLoRA.merge(adapterURL: url, into: fresh)
        XCTAssertEqual(report.mergedLayers, 6)
        let noise = MLXRandom.normal(example.latent.shape, key: MLXRandom.key(2))
        let draw = NFKMLXWanTrainingDraw(sigma: 0.5, timestep: 500, weight: 1)
        let trained = try NFKMLXWanAnimate.score(net, example: example, objective: preset.objective, draw: draw, noise: noise)
        let merged = try NFKMLXWanAnimate.score(fresh, example: example, objective: preset.objective, draw: draw, noise: noise)
        XCTAssertEqual(trained.item(Float.self), merged.item(Float.self), accuracy: 1e-4, "the saved file merges back")
    }

    func testTheObjectiveMatchesDiffSynth() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN_ANIMATE_LORA_DIFFSYNTH"] else {
            throw XCTSkip("set IK_PARITY_WAN_ANIMATE_LORA_DIFFSYNTH (run_reference.py wan_animate_lora_diffsynth)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        func section(_ prefix: String) -> [(String, MLXArray)] {
            record.compactMap { key, value in
                key.hasPrefix(prefix) ? (NFKMLXWanAnimate.moduleKey(forRelease: String(key.dropFirst(prefix.count))), value) : nil
            }
        }
        let net = NFKMLXWanAnimateNet(geometry)
        let preset = NFKMLXWanAnimateTrainingPreset.diffSynth
        let adapted = try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 8) { path, _ in preset.adapts(path) }
        XCTAssertEqual(adapted, 6)
        let weights = section("w::").map { ($0.0, $0.1.ndim == 5 ? $0.1.transposed(0, 2, 3, 4, 1) : $0.1) }
        try NFKMLXWeights.apply(weights + section("lora::"), to: net, verifyShapes: true)

        let shape = record["shape"]!.asArray(Int32.self).map(Int.init)
        let example = NFKMLXWanAnimateTrainingExample(
            latent: record["latent"]!, condition: record["condition"]!, reference: record["reference"]!,
            referenceCondition: record["reference_condition"]!, text: record["text"]!,
            referenceText: record["reference_text"]!, imageFeatures: record["image_features"]!,
            referenceImageFeatures: record["reference_image_features"]!, videoFrames: shape[0],
            videoArea: shape[1] * shape[2])
        let draw = preset.objective.sampler.draw(record["index"]!.item(Float.self), weighting: preset.objective.weighting)
        XCTAssertEqual(draw.timestep, record["timestep"]!.item(Float.self))
        XCTAssertEqual(draw.sigma, record["sigma"]!.item(Float.self))
        let noise = record["noise"]!
        let noisy = (1 - draw.sigma) * example.latent + draw.sigma * noise
        let prediction = try NFKMLXWanAnimate.trainingPrediction(net, example: example, noisy: noisy, timestep: draw.timestep)
        let predictionCosine = cosine(prediction, record["prediction"]!)
        XCTAssertGreaterThan(predictionCosine, 0.999999)
        let loss = try NFKMLXWanAnimate.score(net, example: example, objective: preset.objective, draw: draw, noise: noise)
        XCTAssertEqual(loss.item(Float.self), record["loss"]!.item(Float.self), accuracy: 1e-5 * record["loss"]!.item(Float.self))

        net.freeze()
        for (_, layer) in net.leafModules().flattened() where layer is NFKMLXLoRAAdapter {
            layer.unfreeze(recursive: false, keys: ["lora_a", "lora_b"])
        }
        let lossAndGradient = valueAndGrad(model: net) { net, _ in
            [try! NFKMLXWanAnimate.score(net, example: example, objective: preset.objective, draw: draw, noise: noise)]
        }
        let gradients = Dictionary(uniqueKeysWithValues: lossAndGradient(net, []).1.flattened())
        let recorded = section("grad::")
        XCTAssertEqual(Set(gradients.keys), Set(recorded.map(\.0)))
        var lowest = 1.0
        for (key, reference) in recorded {
            guard let ours = gradients[key] else { continue }
            let c = cosine(ours, reference)
            lowest = min(lowest, c)
            XCTAssertGreaterThan(c, 0.999999, key)
        }
        print("VALIDATION PARITY wan-animate-lora-diffsynth: prediction cosine \(predictionCosine), loss "
              + "\(loss.item(Float.self)) vs \(record["loss"]!.item(Float.self)), lowest gradient cosine \(lowest)")
    }

    func testAClipBecomesAnExampleAndTrainsOverItsRelease() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_WAN_ANIMATE_TESTDEV"] else {
            throw XCTSkip("set IK_VAL_WAN_ANIMATE_TESTDEV (Tools/wan-animate-testdev/make_testdev.py)")
        }
        let directory = URL(fileURLWithPath: path)
        let generator = try NFKMLXWanAnimateGenerator.generator(directoryURL: directory, residency: .resident)
        let keys = MLXRandom.split(key: MLXRandom.key(5), into: 3)
        let video = MLXRandom.uniform(low: 0, high: 1, [5, 32, 48, 3], key: keys[0])
        let driving = MLXRandom.uniform(low: 0, high: 1, [5, 32, 48, 3], key: keys[1])
        let reference = MLXRandom.uniform(low: 0, high: 1, [32, 48, 3], key: keys[2])
        let example = try generator.trainingExample(referenceImage: reference, drivingVideo: driving, video: video,
                                                    prompt: "a dancer")
        XCTAssertEqual(example.latent.shape, [16, 3, 4, 6], "the reference image's latent, then the clip's two")
        XCTAssertEqual(example.condition.shape, [20, 3, 4, 6])
        XCTAssertEqual(example.reference.shape, [16, 2, 4, 6])
        XCTAssertEqual(example.condition[0 ..< 4, 0].min().item(Float.self), 1, "the reference image's slot is marked")
        XCTAssertEqual(example.condition[0 ..< 4, 1...].max().item(Float.self), 0, "the continuation slot is empty")
        XCTAssertEqual(example.referenceCondition[0 ..< 4].min().item(Float.self), 1, "every driving frame is marked")
        XCTAssertThrowsError(try generator.trainingExample(referenceImage: reference, drivingVideo: driving[0 ..< 4],
                                                           video: video, prompt: "a dancer"))

        let net = try NFKMLXWanAnimate.network(directoryURL: directory, quantizationBits: 8)
        var preset = NFKMLXWanAnimateTrainingPreset.diffSynth
        preset.rank = 4
        let losses = try NFKMLXWanAnimate.fineTune(net, examples: { _ in [example] }, preset: preset,
                                                   optimizer: Adam(learningRate: 1e-3), steps: 2, seed: 2,
                                                   learningRateSchedule: .constant)
        XCTAssertEqual(losses.count, 2)
        XCTAssertTrue(losses.allSatisfy(\.isFinite))
    }
}
