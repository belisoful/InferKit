//
//  NFKMLXLoRATests.swift
//  InferKitMLXTests
//
//  Low-rank adaptation. Three properties carry the design: an adapted model starts out identical to the
//  one it wrapped, only the adapters train, and merging reproduces the adapted forward exactly — which
//  is what lets a customized model ship as one ordinary checkpoint.
//

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXLoRATests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func setUp() {
        super.setUp()
        NFKMLXRandom.seed(20_260_814)
    }

    /// A small stack with named layers, standing in for an attention block.
    private final class Block: Module {
        @ModuleInfo(key: "q") var q: Linear
        @ModuleInfo(key: "v") var v: Linear
        @ModuleInfo(key: "mlp") var mlp: Linear

        override init() {
            _q.wrappedValue = Linear(8, 8)
            _v.wrappedValue = Linear(8, 8)
            _mlp.wrappedValue = Linear(8, 8)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            mlp(q(x) + v(x))
        }
    }

    private func input() -> MLXArray {
        MLXArray((0 ..< 24).map { Float($0 % 7) / 7.0 }).reshaped([3, 8])
    }

    // MARK: - Adapting

    func testAnAdaptedModelStartsIdenticalToTheOneItWrapped() throws {
        try requireMLXRuntime()
        let block = Block()
        let x = input()
        let before = block(x)
        eval(before)

        try NFKMLXLoRA.apply(to: block, rank: 4)

        let after = block(x)
        eval(after)
        // B starts at zero, so the detour contributes nothing until it is trained. A fine-tune's
        // starting point is a checkpoint worth preserving.
        XCTAssertEqual(after.asArray(Float.self), before.asArray(Float.self))
    }

    func testThePredicateSelectsWhichLayersAreAdapted() throws {
        try requireMLXRuntime()
        let block = Block()
        let adapted = try NFKMLXLoRA.apply(to: block, rank: 4) { path, _ in
            path.hasSuffix("q") || path.hasSuffix("v")
        }
        XCTAssertEqual(adapted, 2, "targeting the attention projections is the usual choice")
        XCTAssertTrue(block.q is NFKMLXLoRALinear)
        XCTAssertTrue(block.v is NFKMLXLoRALinear)
        XCTAssertFalse(block.mlp is NFKMLXLoRALinear)
    }

    func testApplyingTwiceDoesNotStackDetours() throws {
        try requireMLXRuntime()
        let block = Block()
        XCTAssertEqual(try NFKMLXLoRA.apply(to: block, rank: 4), 3)
        XCTAssertEqual(try NFKMLXLoRA.apply(to: block, rank: 4), 0, "already-adapted layers are skipped")
    }

    // MARK: - What trains

    func testOnlyTheAdaptersAreTrainable() throws {
        try requireMLXRuntime()
        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)

        let trainable = block.trainableParameters().flattened().map(\.0)
        XCTAssertFalse(trainable.isEmpty)
        XCTAssertTrue(trainable.allSatisfy { $0.hasSuffix("lora_a") || $0.hasSuffix("lora_b") },
                      "a frozen base costs no gradient and no optimizer state: \(trainable)")
    }

    func testAdaptingCutsTheTrainableParameterCount() throws {
        try requireMLXRuntime()
        let full = Block()
        let fullCount = NFKMLXLoRA.trainableParameterCount(of: full)

        let adapted = Block()
        try NFKMLXLoRA.apply(to: adapted, rank: 2)
        let adaptedCount = NFKMLXLoRA.trainableParameterCount(of: adapted)

        XCTAssertLessThan(adaptedCount, fullCount,
                          "rank 2 over 8-wide layers trains less than the layers themselves: "
                          + "\(adaptedCount) vs \(fullCount)")
    }

    func testTrainingMovesOnlyTheAdapters() throws {
        try requireMLXRuntime()
        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)
        let baseWeight = block.q.weight.asArray(Float.self)

        let x = input()
        let target = MLXArray.zeros([3, 8])
        try NFKMLXTrainer.train(block, optimizer: Adam(learningRate: 1e-2), steps: 20,
                                batch: { _ in (x, target) },
                                loss: { model, input, target in (model(input) - target).square().mean() })

        XCTAssertEqual(block.q.weight.asArray(Float.self), baseWeight, "the frozen base did not move")
        let adapter = try XCTUnwrap(block.q as? NFKMLXLoRALinear)
        XCTAssertNotEqual(adapter.loraB.asArray(Float.self),
                          [Float](repeating: 0, count: adapter.loraB.size),
                          "the detour did")
    }

    func testTrainingAnAdaptedModelReducesTheLoss() throws {
        try requireMLXRuntime()
        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)
        let x = input()
        let target = MLXArray.zeros([3, 8])

        let history = try NFKMLXTrainer.train(block, optimizer: Adam(learningRate: 1e-2), steps: 40,
                                              batch: { _ in (x, target) },
                                              loss: { model, input, target in
                                                  (model(input) - target).square().mean()
                                              })
        XCTAssertLessThan(history.last!, history.first! * 0.5,
                          "a rank-4 detour is enough to move the output: "
                          + "\(history.first!) -> \(history.last!)")
    }

    // MARK: - Merging

    func testMergingReproducesTheAdaptedForward() throws {
        try requireMLXRuntime()
        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)
        let x = input()
        let target = MLXArray.zeros([3, 8])
        try NFKMLXTrainer.train(block, optimizer: Adam(learningRate: 1e-2), steps: 20,
                                batch: { _ in (x, target) },
                                loss: { model, input, target in (model(input) - target).square().mean() })

        let adaptedOutput = block(x)
        eval(adaptedOutput)

        XCTAssertEqual(try NFKMLXLoRA.merge(into: block), 3)
        XCTAssertFalse(block.q is NFKMLXLoRALinear, "plain layers again")

        let mergedOutput = block(x)
        eval(mergedOutput)
        for (merged, adapted) in zip(mergedOutput.asArray(Float.self), adaptedOutput.asArray(Float.self)) {
            XCTAssertEqual(merged, adapted, accuracy: 1e-5,
                           "the fold into the base weights is exact")
        }
    }

    func testAMergedModelSavesAndReloadsAsAnOrdinaryCheckpoint() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("lora-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)
        let x = input()
        try NFKMLXTrainer.train(block, optimizer: Adam(learningRate: 1e-2), steps: 20,
                                batch: { _ in (x, MLXArray.zeros([3, 8])) },
                                loss: { model, input, target in (model(input) - target).square().mean() })
        try NFKMLXLoRA.merge(into: block)
        try NFKMLXWeights.save(block, to: url)

        // The point of merging: what comes out carries no adapter keys, so an unmodified model loads it.
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        XCTAssertFalse(checkpoint.arrays.keys.contains { $0.contains("lora_") },
                       "no adapter format to carry around: \(checkpoint.arrays.keys.sorted())")

        let plain = Block()
        try NFKMLXWeights.apply(Array(checkpoint.arrays), to: plain)
        let expected = block(x)
        let actual = plain(x)
        eval(expected, actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    /// A model that stores a layer in a plain property cannot receive a replacement. MLX's own path
    /// aborts the process there, so the library has to report it instead.
    private final class Unwrapped: Module {
        let projection = Linear(8, 8)
    }

    func testAdaptingALayerThatCannotBeReplacedThrows() throws {
        try requireMLXRuntime()
        XCTAssertThrowsError(try NFKMLXLoRA.apply(to: Unwrapped(), rank: 4)) { error in
            guard case NFKMLXError.loRANotApplicable(let detail) = error else {
                return XCTFail("expected loRANotApplicable, got \(error)")
            }
            XCTAssertTrue(detail.contains("@ModuleInfo"), "names the fix: \(detail)")
        }
    }

    func testMergingUnfreezesTheModel() throws {
        try requireMLXRuntime()
        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)
        try NFKMLXLoRA.merge(into: block)
        XCTAssertEqual(block.trainableParameters().flattened().count,
                       block.parameters().flattened().count,
                       "a merged model is an ordinary model again")
    }
}
