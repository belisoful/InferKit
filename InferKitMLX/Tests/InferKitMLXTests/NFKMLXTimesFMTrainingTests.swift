//
//  NFKMLXTimesFMTrainingTests.swift
//  InferKitMLXTests
//
//  TimesFM 2.5's LoRA fine-tuning path against transformers' `TimesFm2_5ModelForPrediction` and the
//  settings of google-research/timesfm's `finetune_lora.py`, `run_reference.py timesfm_loss`
//  (`IK_VAL_TIMESFM25`, `IK_PARITY_TIMESFM25_LOSS`); the round trip runs on a tiny configuration.
//

import XCTest
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
@testable import InferKitMLX

final class NFKMLXTimesFMTrainingTests: XCTestCase {

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
        guard let directory = env["IK_VAL_TIMESFM25"], let path = env["IK_PARITY_TIMESFM25_LOSS"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_TIMESFM25 and IK_PARITY_TIMESFM25_LOSS (run_reference.py timesfm_loss)")
        }
        return (URL(fileURLWithPath: directory), try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays)
    }

    private static func windows(_ record: [String: MLXArray]) -> [(context: [Float], target: [Float])] {
        let contexts = record["windows.context"]!, targets = record["windows.target"]!
        return (0 ..< contexts.dim(0)).map { (contexts[$0].asArray(Float.self), targets[$0].asArray(Float.self)) }
    }

    /// A transformers module path, as the oracle names an adapted layer, in this module's layout.
    static func modulePath(_ transformers: String) -> String {
        var parts = transformers.split(separator: ".").map(String.init)
        if parts.first == "model" { parts.removeFirst() }
        let joined = parts.joined(separator: ".")
        let renames = [("input_ff_layer.input_layer", "tokenizer.hidden_layer"), ("input_ff_layer.", "tokenizer."),
                       (".input_layer", ".hidden_layer"), ("layers.", "stacked_xf."), ("self_attn.q_proj", "attn.query"),
                       ("self_attn.k_proj", "attn.key"), ("self_attn.v_proj", "attn.value"), ("self_attn.o_proj", "attn.out"),
                       ("mlp.fc1", "ff0"), ("mlp.fc2", "ff1")]
        return renames.reduce(joined) { $0.replacingOccurrences(of: $1.0, with: $1.1) }
    }

    /// The objective on identical forecasts and targets, the reference's channel-to-level pairing included.
    func testTheObjectiveMatchesTheReference() throws {
        try requireMLXRuntime()
        let (_, record) = try release()
        let loss = NFKMLXTimesFMObjective().loss(forecast: try XCTUnwrap(record["objective.forecast"]),
                                                 targets: try XCTUnwrap(record["objective.target"]),
                                                 quantiles: NFKMLXTimesFMConfiguration.v2_5.quantiles)
        let reference = try XCTUnwrap(record["objective.loss"]).item(Float.self)
        print("VALIDATION PARITY timesfm objective: \(loss.item(Float.self)) vs \(reference)")
        XCTAssertEqual(loss.item(Float.self), reference, accuracy: 1e-6 * abs(reference))
    }

    /// The model's training loss on four windows, from the transformers release through this network.
    func testTheTrainingLossMatchesTheReference() throws {
        try requireMLXRuntime()
        let (directory, record) = try release()
        let net = try NFKMLXTimesFM.network(directoryURL: directory)
        let loss = NFKMLXTimesFMObjective()(net, windows: Self.windows(record)).item(Float.self)
        let reference = try XCTUnwrap(record["loss"]).item(Float.self)
        print("VALIDATION PARITY timesfm training loss: \(loss) vs \(reference) (rel \(abs(loss - reference) / reference))")
        XCTAssertEqual(loss, reference, accuracy: 1e-5 * reference)
    }

    /// Three LoRA steps from the reference's own adapter initialization: each step's loss, the first step's
    /// gradient on the watched B matrices, and every step's movement of the watched A and B matrices.
    func testThreeLoRAStepsFollowTheReference() throws {
        try requireMLXRuntime()
        let (directory, record) = try release()
        let net = try NFKMLXTimesFM.network(directoryURL: directory)
        let adapted = try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 8)
        XCTAssertEqual(adapted, try XCTUnwrap(record["adapted_count"]).item(Int.self), "PEFT's all-linear adapts every Linear")
        var initial = [(String, MLXArray)]()
        for (key, value) in record where key.hasPrefix("lora_a.") {
            initial.append((Self.modulePath(String(key.dropFirst("lora_a.".count))) + ".lora_a", value.transposed()))
        }
        XCTAssertEqual(initial.count, adapted)
        try NFKMLXWeights.apply(initial, to: net, strict: false)
        let windows = Self.windows(record)
        let watched = record.keys.filter { $0.hasPrefix("step_grad_b.") }.map { String($0.dropFirst("step_grad_b.".count)) }

        let gradients = Dictionary(uniqueKeysWithValues: valueAndGrad(model: net) { (net: NFKMLXTimesFMNet, _: [MLXArray]) in
            [NFKMLXTimesFMObjective()(net, windows: windows)]
        }(net, [MLXArray(0)]).1.flattened())
        var lines = [String]()
        for name in watched {
            let similarity = NFKMLXWav2Vec2Tests.cosine(try XCTUnwrap(gradients[Self.modulePath(name) + ".lora_b"], name),
                                                        record["step_grad_b.\(name)"]!.transposed())
            lines.append("grad \(Self.modulePath(name)) \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "\(name) first-step gradient")
        }

        let start = NFKMLXWav2Vec2TrainingTests.snapshot(net)
        var moved = [[String: MLXArray]]()
        let losses = try NFKMLXTimesFM.fineTune(net, windows: { _ in windows }, steps: 3, observer: { _ in
            let now = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
            moved.append(now.filter { $0.key.hasSuffix(".lora_a") || $0.key.hasSuffix(".lora_b") }
                .mapValues { $0 } .reduce(into: [:]) { $0[$1.key] = $1.value - start[$1.key]! })
            return true
        })
        let referenceLosses = try XCTUnwrap(record["step_losses"]).asType(.float32).asArray(Float.self)
        for (index, (loss, reference)) in zip(losses, referenceLosses).enumerated() {
            lines.append("loss\(index) \(loss) vs \(reference)")
            XCTAssertEqual(loss, reference, accuracy: 1e-4 * reference, "step \(index) loss")
        }
        XCTAssertEqual(moved.count, 3)
        for (index, deltas) in moved.enumerated() {
            for name in watched {
                for (matrix, suffix) in [("a", "lora_a"), ("b", "lora_b")] {
                    let reference = try XCTUnwrap(record["step\(index)_delta_\(matrix).\(name)"]).transposed()
                    let mine = try XCTUnwrap(deltas[Self.modulePath(name) + "." + suffix])
                    if abs(reference).max().item(Float.self) == 0 {
                        XCTAssertEqual(abs(mine).max().item(Float.self), 0, "step \(index) \(name) \(suffix) stays")
                        continue
                    }
                    let similarity = NFKMLXWav2Vec2Tests.cosine(mine, reference)
                    if index == 2 { lines.append("moved \(Self.modulePath(name)).\(suffix) \(similarity)") }
                    XCTAssertGreaterThan(similarity, 0.999, "step \(index) \(name) \(suffix) movement")
                }
            }
        }
        print("VALIDATION PARITY timesfm LoRA steps: " + lines.joined(separator: ", "))
    }

    /// A tiny LoRA fine-tune lowers the loss and leaves every base weight unchanged; the saved directory,
    /// with the adapters folded in, forecasts through the `@objc` factory exactly as the adapted network does.
    func testALoRAFineTuneRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        MLXRandom.seed(9)
        var c = NFKMLXTimesFMConfiguration()
        c.hiddenSize = 64
        c.intermediateSize = 64
        c.numLayers = 2
        c.numHeads = 4
        let net = NFKMLXTimesFMNet(c)
        let series = (0 ..< 400).map { Float(sin(Double($0) * 0.3) * 2 + Double($0) * 0.01) }
        let windows = (0 ..< 4).map { start in (context: Array(series[(start * 40) ..< (start * 40 + 64)]),
                                                target: Array(series[(start * 40 + 64) ..< (start * 40 + 77)])) }
        let before = NFKMLXWav2Vec2TrainingTests.snapshot(net)
        let losses = try NFKMLXTimesFM.fineTune(net, windows: { _ in windows }, learningRate: 3e-3, steps: 15)
        XCTAssertLessThan(losses.last!, losses.first!, "the loss falls")
        let after = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        for (name, value) in before {
            XCTAssertEqual(abs(after[name]! - value).max().item(Float.self), 0, "\(name) is frozen")
        }

        let context = Array(series.prefix(200))
        let adapted = try NFKMLXTimesFM(net: net).forecast(context: context, horizon: 20).values
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try NFKMLXTimesFM.save(net, toDirectoryURL: directory)
        let reloaded = try NFKMLXTimesFM.timesFM(directoryURL: directory).forecast(context: context, horizon: 20).values
        let difference = zip(adapted.joined(), reloaded.joined()).map { abs($0 - $1) }.max() ?? .infinity
        XCTAssertLessThan(difference, 1e-4, "the folded adapters forecast as the adapted network does")
    }
}
