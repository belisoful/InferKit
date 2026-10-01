//
//  NFKMLXWav2Vec2BertTrainingTests.swift
//  InferKitMLXTests
//
//  W2V-BERT 2.0's CTC fine-tuning path against transformers' `Wav2Vec2BertForCTC` with Hugging Face's recipe
//  settings, `run_reference.py w2v_bert_loss` (`IK_VAL_W2V_BERT_2`, `IK_PARITY_W2V_BERT_LOSS`); the round
//  trip runs on a tiny configuration.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXWav2Vec2BertTrainingTests: XCTestCase {

    override func tearDown() {
        Memory.clearCache()
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func release() throws -> (directory: URL, record: [String: MLXArray]) {
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_W2V_BERT_2"], let path = env["IK_PARITY_W2V_BERT_LOSS"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_W2V_BERT_2 and IK_PARITY_W2V_BERT_LOSS (run_reference.py w2v_bert_loss)")
        }
        return (URL(fileURLWithPath: directory), try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays)
    }

    /// The release with the reference's own adapter and head initialization in place.
    private func referenceNetwork(_ directory: URL, _ record: [String: MLXArray]) throws -> NFKMLXWav2Vec2BertNet {
        let size = try XCTUnwrap(record["vocabulary_size"]).item(Int.self)
        let net = try NFKMLXWav2Vec2Bert.network(directoryURL: directory, vocabulary: (0 ..< size).map(String.init))
        let initial = record.filter { $0.key.hasPrefix("init.") }.map { key, value -> (String, MLXArray) in
            (String(key.dropFirst("init.".count)), value.ndim == 3 ? value.transposed(0, 2, 1) : value)
        }
        XCTAssertEqual(initial.count, net.parameters().flattened().filter { $0.0.hasPrefix("adapter.") || $0.0.hasPrefix("lm_head.") }.count)
        try NFKMLXWeights.apply(initial, to: net, strict: false)
        return net
    }

    /// The CTC loss over the adapter's real frames and its gradient, on identical logits; and the adapted
    /// network's logits from the reference's initialization, which measures the adapter's forward.
    func testTheObjectiveAndTheAdapterMatchTheReference() throws {
        try requireMLXRuntime()
        let (directory, record) = try release()
        let labels = try XCTUnwrap(record["labels"]).asArray(Int64.self).map(Int.init)
        let frames = try XCTUnwrap(record["frames"]).item(Int.self)
        let logits = try XCTUnwrap(record["logits"]).expandedDimensions(axis: 0)
        let objective = NFKMLXWav2Vec2Objective()
        let (values, gradients) = valueAndGrad { (x: [MLXArray]) in
            [objective.loss(logits: x[0], labels: [labels], frames: [frames])]
        }([logits])
        let reference = try XCTUnwrap(record["objective.loss"]).item(Float.self)
        let gradient = NFKMLXWav2Vec2Tests.cosine(gradients[0][0], try XCTUnwrap(record["objective.grad"]))
        XCTAssertEqual(values[0].item(Float.self), reference, accuracy: 1e-5 * reference)
        XCTAssertGreaterThan(gradient, 0.999999)

        let net = try referenceNetwork(directory, record)
        let waveform = try XCTUnwrap(record["waveform"]).asArray(Float.self)
        let (features, mask) = NFKMLXWav2Vec2BertProcessor.inputFeatures(waveform)
        let (mine, counts) = try XCTUnwrap(net.logits(features, mask: mask))
        XCTAssertEqual(counts, [frames], "real frames after the adapter's pooling")
        let similarity = NFKMLXWav2Vec2Tests.cosine(mine[0], record["logits"]!)
        print("VALIDATION PARITY w2v-bert objective: loss \(values[0].item(Float.self)) vs \(reference), gradient \(gradient); "
              + "adapted logits \(similarity) over \(frames) frames")
        XCTAssertGreaterThan(similarity, 0.99999, "the adapter and head")
    }

    /// Three steps of the recipe (every parameter, AdamW 5e-5, clipping at 1.0, a linear decay) from the
    /// reference's initialization: each step's loss, the first step's gradients, and every step's movement.
    /// The recipe's warm-up schedule matches the reference's at sample steps.
    func testThreeRecipeStepsFollowTheReference() throws {
        try requireMLXRuntime()
        let (directory, record) = try release()
        let schedule = NFKMLXLearningRateSchedule.linearWithWarmup(steps: 2000, warmupSteps: 500)
        let sampled = try XCTUnwrap(record["schedule.steps"]).asArray(Int64.self).map(Int.init)
        for (step, scale) in zip(sampled, try XCTUnwrap(record["schedule.scales"]).asType(.float32).asArray(Float.self)) {
            XCTAssertEqual(schedule.multiplier(step), scale, accuracy: 1e-6, "schedule at \(step)")
        }

        let net = try referenceNetwork(directory, record)
        let waveform = try XCTUnwrap(record["waveform"]).asArray(Float.self)
        let labels = try XCTUnwrap(record["labels"]).asArray(Int64.self).map(Int.init)
        let watched = record.keys.filter { $0.hasPrefix("step0_delta.") }.map { String($0.dropFirst("step0_delta.".count)) }.sorted()
        func layout(_ x: MLXArray) -> MLXArray { x.ndim == 3 ? x.transposed(0, 2, 1) : x }

        let (features, mask) = NFKMLXWav2Vec2BertProcessor.inputFeatures(waveform)
        net.unfreeze()
        let gradients = Dictionary(uniqueKeysWithValues: valueAndGrad(model: net) { (net: NFKMLXWav2Vec2BertNet, arrays: [MLXArray]) in
            let (hidden, outputMask) = net.encode(arrays[0], mask: arrays[1])
            let frames = outputMask!.asType(.int32).sum(axis: -1).asArray(Int32.self).map(Int.init)
            return [NFKMLXWav2Vec2Objective().loss(logits: net.head!(hidden), labels: [labels], frames: frames)]
        }(net, [features, mask]).1.flattened())
        var lines = [String]()
        for name in watched {
            let similarity = NFKMLXWav2Vec2Tests.cosine(try XCTUnwrap(gradients[name], name), layout(record["step_grad.\(name)"]!))
            lines.append("grad \(name) \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "\(name) first-step gradient")
        }

        let start = NFKMLXWav2Vec2TrainingTests.snapshot(net)
        var moved = [[String: MLXArray]]()
        let losses = try NFKMLXWav2Vec2Bert.fineTune(
            net, examples: { _ in (waveform, labels) }, steps: 3, learningRateSchedule: .poly(steps: 3),
            observer: { _ in
                let now = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
                moved.append(Dictionary(uniqueKeysWithValues: watched.map { ($0, now[$0]! - start[$0]!) }))
                return true
            })
        let referenceLosses = try XCTUnwrap(record["step_losses"]).asType(.float32).asArray(Float.self)
        for (index, (loss, reference)) in zip(losses, referenceLosses).enumerated() {
            lines.append("loss\(index) \(loss) vs \(reference)")
            XCTAssertEqual(loss, reference, accuracy: (index == 0 ? 1e-4 : 2e-3) * reference, "step \(index) loss")
        }
        for (index, deltas) in moved.enumerated() {
            for name in watched {
                let similarity = NFKMLXWav2Vec2Tests.cosine(deltas[name]!, layout(try XCTUnwrap(record["step\(index)_delta.\(name)"])))
                if index == 2 { lines.append("moved \(name) \(similarity)") }
                XCTAssertGreaterThan(similarity, 0.999, "step \(index) \(name) movement")
            }
        }
        print("VALIDATION PARITY w2v-bert steps: " + lines.joined(separator: ", "))
    }

    /// A tiny fine-tune lowers the loss; the saved directory reloads through the network builder with the
    /// same logits and transcribes through the `@objc` factory.
    func testTheFineTunedDirectoryRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        MLXRandom.seed(11)
        var c = NFKMLXWav2Vec2BertConfiguration()
        c.hiddenSize = 32
        c.outputHiddenSize = 32
        c.numHiddenLayers = 1
        c.numAttentionHeads = 2
        c.intermediateSize = 64
        let release = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let saved = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {
            try? FileManager.default.removeItem(at: release)
            try? FileManager.default.removeItem(at: saved)
        }
        try NFKMLXWav2Vec2Bert.save(NFKMLXWav2Vec2BertNet(c), tokenizer: nil, toDirectoryURL: release)
        let tokenizer = try NFKMLXWav2Vec2Tokenizer(characters: ["A", "B", "C"])
        let net = try NFKMLXWav2Vec2Bert.network(directoryURL: release, vocabulary: tokenizer.characters)
        XCTAssertNotNil(net.adapter)
        let samples = (0 ..< 16000).map { Float(sin(Double($0) * 0.05) * 0.3) }
        let losses = try NFKMLXWav2Vec2Bert.fineTune(net, examples: { _ in (samples, tokenizer.labels(for: "AB C")) },
                                                     learningRate: 1e-3, warmupSteps: 0, steps: 8)
        XCTAssertLessThan(losses.last!, losses.first!)
        try NFKMLXWav2Vec2Bert.save(net, tokenizer: tokenizer, toDirectoryURL: saved)

        let (features, mask) = NFKMLXWav2Vec2BertProcessor.inputFeatures(samples)
        let expected = try XCTUnwrap(net.logits(features, mask: mask)).logits
        let reloaded = try NFKMLXWav2Vec2Bert.network(directoryURL: saved)
        XCTAssertEqual(abs(try XCTUnwrap(reloaded.logits(features, mask: mask)).logits - expected).max().item(Float.self), 0,
                       accuracy: 1e-5)
        let backend = try NFKMLXWav2Vec2Bert.backend(directoryURL: saved)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: NFKMLXWaveFile.data(samples: samples, sampleRate: 16000)]))
        XCTAssertNotNil(result.output(forKey: NFKOutputText) as? String)
    }
}
