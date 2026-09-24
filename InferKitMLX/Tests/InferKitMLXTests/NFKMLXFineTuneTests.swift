//
//  NFKMLXFineTuneTests.swift
//  InferKitMLXTests
//
//  The sequence `NFKMLXFineTune.run` hoists out of every recipe. Each test here pins one rule a
//  recipe has been written against by hand, and each rule has a way of being wrong that produces a
//  plausible loss curve rather than an error.
//

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXFineTuneTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_924)
    }

    /// A frozen first layer and a trainable second, which is the head-only shape every retarget uses.
    private final class HeadOverBackbone: Module {

        @ModuleInfo(key: "backbone") var backbone: Linear
        @ModuleInfo(key: "head") var head: Linear

        override init() {
            _backbone.wrappedValue = Linear(2, 2)
            _head.wrappedValue = Linear(2, 2)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            head(backbone(x))
        }

        func freezeBackbone() {
            unfreeze()
            backbone.freeze()
        }
    }

    private func fixedBatch() -> (input: MLXArray, target: MLXArray) {
        let input = MLXArray([1.0, 0.0, 0.0, 1.0, 0.5, -0.5] as [Float]).reshaped([3, 2])
        return (input, input * 2.0)
    }

    private func meanSquaredError(_ model: HeadOverBackbone, _ input: MLXArray,
                                  _ target: MLXArray) -> MLXArray {
        (model(input) - target).square().mean()
    }

    private func trainableCount(of model: Module) -> Int {
        model.trainableParameters().flattened().reduce(0) { $0 + $1.1.size }
    }

    // MARK: Ordering

    /// Freezing runs before the reference optimizer is built, so the optimizer never sees a parameter
    /// that will not move. Building the optimizer first gives it state for the whole model, which
    /// costs memory the run was frozen to save and reports nothing.
    func testFreezingRunsBeforeTheReferenceOptimizerIsBuilt() throws {
        try requireMLXRuntime()
        let net = HeadOverBackbone()
        let whole = trainableCount(of: net)
        var seenWhenOptimizerWasBuilt: Int?

        _ = try NFKMLXFineTune.run(
            net,
            freezing: { net.freezeBackbone() },
            optimizer: nil,
            reference: {
                seenWhenOptimizerWasBuilt = self.trainableCount(of: net)
                return SGD(learningRate: 0.01)
            },
            referenceSchedule: { .constant },
            steps: 1,
            batch: { _ in self.fixedBatch() },
            loss: self.meanSquaredError,
            clipGradientNorm: nil)

        XCTAssertEqual(seenWhenOptimizerWasBuilt, trainableCount(of: net),
                       "the optimizer is built against the frozen model")
        XCTAssertLessThan(seenWhenOptimizerWasBuilt ?? whole, whole,
                          "freezing actually removed parameters, so the check above can fail")
    }

    // MARK: Optimizer resolution

    /// A caller's optimizer is used and the reference's is never built.
    func testTheCallersOptimizerReplacesTheReferenceWithoutBuildingIt() throws {
        try requireMLXRuntime()
        let net = HeadOverBackbone()
        var builtReference = false

        _ = try NFKMLXFineTune.run(
            net,
            freezing: { net.unfreeze() },
            optimizer: SGD(learningRate: 0.01),
            reference: {
                builtReference = true
                return SGD(learningRate: 0.01)
            },
            referenceSchedule: { .constant },
            steps: 1,
            batch: { _ in self.fixedBatch() },
            loss: self.meanSquaredError,
            clipGradientNorm: nil)

        XCTAssertFalse(builtReference,
                       "a reference optimizer that walks the parameter tree is not paid for")
    }

    // MARK: Schedule resolution

    /// With no optimizer from the caller, the reference's schedule applies.
    func testTheReferenceScheduleAppliesWhenTheCallerSuppliesNoOptimizer() throws {
        try requireMLXRuntime()
        let net = HeadOverBackbone()
        let optimizer = SGD(learningRate: 1.0)
        var rates: [Float] = []

        _ = try NFKMLXFineTune.run(
            net,
            freezing: { net.unfreeze() },
            optimizer: nil,
            reference: { optimizer },
            referenceSchedule: { NFKMLXLearningRateSchedule { step in Float(step + 1) * 0.5 } },
            steps: 3,
            batch: { _ in self.fixedBatch() },
            loss: self.meanSquaredError,
            clipGradientNorm: nil,
            observer: { _ in rates.append(optimizer.learningRate); return true })

        XCTAssertEqual(rates, [0.5, 1.0, 1.5], "each step ran at the reference schedule's rate")
    }

    /// A caller who passes an optimizer has chosen its rate, so the reference's schedule is not
    /// applied over it. Resolving against the recipe's own optimizer instead of the caller's is what
    /// makes every run constant-rate, and nothing reports it.
    func testACallersOptimizerHoldsItsRateAgainstTheReferenceSchedule() throws {
        try requireMLXRuntime()
        let net = HeadOverBackbone()
        let optimizer = SGD(learningRate: 1.0)
        var rates: [Float] = []

        _ = try NFKMLXFineTune.run(
            net,
            freezing: { net.unfreeze() },
            optimizer: optimizer,
            reference: { SGD(learningRate: 1.0) },
            referenceSchedule: { NFKMLXLearningRateSchedule { step in Float(step + 1) * 0.5 } },
            steps: 3,
            batch: { _ in self.fixedBatch() },
            loss: self.meanSquaredError,
            clipGradientNorm: nil,
            observer: { _ in rates.append(optimizer.learningRate); return true })

        XCTAssertEqual(rates, [1.0, 1.0, 1.0], "the caller's rate is held")
    }

    /// A caller's optimizer that has no single rate to scale still runs when no schedule is given.
    /// Holding a caller's rate is done by applying no schedule, so the trainer never asks Adafactor
    /// for a rate it does not have.
    func testACallersOptimizerWithNoSingleRateRunsWithoutASchedule() throws {
        try requireMLXRuntime()
        let net = HeadOverBackbone()

        let history = try NFKMLXFineTune.run(
            net,
            freezing: { net.unfreeze() },
            optimizer: Adafactor(),
            reference: { SGD(learningRate: 0.01) },
            referenceSchedule: { NFKMLXLearningRateSchedule { _ in 0.5 } },
            steps: 2,
            batch: { _ in self.fixedBatch() },
            loss: self.meanSquaredError,
            clipGradientNorm: nil)

        XCTAssertEqual(history.count, 2)
    }

    /// An explicit schedule applies whoever supplied the optimizer.
    func testAnExplicitScheduleAppliesOverACallersOptimizer() throws {
        try requireMLXRuntime()
        let net = HeadOverBackbone()
        let optimizer = SGD(learningRate: 1.0)
        var rates: [Float] = []

        _ = try NFKMLXFineTune.run(
            net,
            freezing: { net.unfreeze() },
            optimizer: optimizer,
            reference: { SGD(learningRate: 1.0) },
            referenceSchedule: { .constant },
            steps: 2,
            batch: { _ in self.fixedBatch() },
            loss: self.meanSquaredError,
            clipGradientNorm: nil,
            learningRateSchedule: NFKMLXLearningRateSchedule { _ in 0.25 },
            observer: { _ in rates.append(optimizer.learningRate); return true })

        XCTAssertEqual(rates, [0.25, 0.25])
    }

    // MARK: The loop underneath

    /// The sequence still trains, so the wrapper is not a no-op over a loop that stopped working.
    func testTheWrappedRunStillDrivesTheLossDown() throws {
        try requireMLXRuntime()
        let net = HeadOverBackbone()

        let history = try NFKMLXFineTune.run(
            net,
            freezing: { net.unfreeze() },
            optimizer: nil,
            reference: { SGD(learningRate: 0.05) },
            referenceSchedule: { .constant },
            steps: 30,
            batch: { _ in self.fixedBatch() },
            loss: self.meanSquaredError,
            clipGradientNorm: 1)

        XCTAssertEqual(history.count, 30)
        XCTAssertLessThan(history.last!, history.first!)
    }

    /// A freezing policy that throws ends the run before the optimizer is built or a step is taken.
    /// A LoRA policy throws when its predicate matches no layer, and training on through that would
    /// report a loss curve for weights that never move.
    func testAFreezingPolicyThatThrowsEndsTheRunBeforeAnyStep() throws {
        try requireMLXRuntime()
        let net = HeadOverBackbone()
        struct PolicyFailed: Error {}
        var builtReference = false
        var steps = 0

        XCTAssertThrowsError(try NFKMLXFineTune.run(
            net,
            freezing: { throw PolicyFailed() },
            optimizer: nil,
            reference: {
                builtReference = true
                return SGD(learningRate: 0.01)
            },
            referenceSchedule: { .constant },
            steps: 3,
            batch: { _ in self.fixedBatch() },
            loss: self.meanSquaredError,
            clipGradientNorm: nil,
            observer: { _ in steps += 1; return true })) { error in
                XCTAssertTrue(error is PolicyFailed, "the policy's own error reaches the caller")
            }
        XCTAssertFalse(builtReference)
        XCTAssertEqual(steps, 0)
    }

    /// A freezing policy that leaves nothing trainable is reported rather than run.
    func testAFreezingPolicyThatLeavesNothingTrainableThrows() throws {
        try requireMLXRuntime()
        let net = HeadOverBackbone()

        XCTAssertThrowsError(try NFKMLXFineTune.run(
            net,
            freezing: { net.freeze() },
            optimizer: nil,
            reference: { SGD(learningRate: 0.01) },
            referenceSchedule: { .constant },
            steps: 1,
            batch: { _ in self.fixedBatch() },
            loss: self.meanSquaredError,
            clipGradientNorm: nil)) { error in
                guard case NFKMLXError.nothingToTrain = error else {
                    return XCTFail("expected nothingToTrain, got \(error)")
                }
            }
    }
}
