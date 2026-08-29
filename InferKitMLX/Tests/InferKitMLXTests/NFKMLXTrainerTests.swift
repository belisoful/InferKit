//
//  NFKMLXTrainerTests.swift
//  InferKitMLXTests
//
//  The training loop. Fitting a known linear map is the smallest problem that proves the whole chain —
//  gradients reach the parameters, the optimizer applies them, and the loss falls — so a failure here
//  is in the loop rather than in a model.
//

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXTrainerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func setUp() {
        super.setUp()
        NFKMLXRandom.seed(20_260_814)
    }

    private func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
    }

    /// A fixed batch whose target is a known linear map of the input, so the model can reach zero loss.
    private func fixedBatch() -> (input: MLXArray, target: MLXArray) {
        let input = MLXArray([1.0, 0.0, 0.0, 1.0, 0.5, -0.5] as [Float]).reshaped([3, 2])
        return (input, input * 2.0)
    }

    private func meanSquaredError(_ model: Linear, _ input: MLXArray, _ target: MLXArray) -> MLXArray {
        (model(input) - target).square().mean()
    }

    func testTrainingDrivesTheLossDown() throws {
        try requireMLXRuntime()
        let model = Linear(2, 2)
        let batch = fixedBatch()

        let history = try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.1), steps: 100,
                                              batch: { _ in batch }, loss: meanSquaredError)

        XCTAssertEqual(history.count, 100)
        XCTAssertLessThan(history.last!, history.first! * 0.1,
                          "the loss fell by an order of magnitude: \(history.first!) -> \(history.last!)")
    }

    func testTheTrainedModelReproducesTheTarget() throws {
        try requireMLXRuntime()
        let model = Linear(2, 2)
        let batch = fixedBatch()

        try NFKMLXTrainer.train(model, optimizer: Adam(learningRate: 0.05), steps: 400,
                                batch: { _ in batch }, loss: meanSquaredError)

        let error = (model(batch.input) - batch.target).abs().max().item(Float.self)
        XCTAssertLessThan(error, 0.05, "the fitted model maps the input onto the target")
    }

    func testFrozenParametersDoNotChange() throws {
        try requireMLXRuntime()
        let model = Linear(2, 2)
        model.freeze(keys: ["bias"])
        let bias = model.bias!.asArray(Float.self)

        try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.1), steps: 20,
                                batch: { _ in self.fixedBatch() }, loss: meanSquaredError)

        XCTAssertEqual(model.bias!.asArray(Float.self), bias,
                       "freezing is how a fine-tune restricts what trains, and it must hold")
        XCTAssertNotEqual(model.weight.asArray(Float.self), Linear(2, 2).weight.asArray(Float.self),
                          "the unfrozen parameter still trained")
    }

    func testTheObserverCanEndTheRunEarly() throws {
        try requireMLXRuntime()
        let model = Linear(2, 2)

        let history = try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.1), steps: 50,
                                              batch: { _ in self.fixedBatch() },
                                              loss: meanSquaredError) { step in
            step.index < 2
        }

        XCTAssertEqual(history.count, 3, "the run stopped on the step that returned false")
    }

    func testTheObserverSeesEachStepInOrder() throws {
        try requireMLXRuntime()
        var seen: [Int] = []

        try NFKMLXTrainer.train(Linear(2, 2), optimizer: SGD(learningRate: 0.1), steps: 5,
                                batch: { _ in self.fixedBatch() }, loss: meanSquaredError) { step in
            seen.append(step.index)
            XCTAssertEqual(step.count, 5)
            return true
        }

        XCTAssertEqual(seen, [0, 1, 2, 3, 4])
    }

    func testTheBatchSourceReceivesTheStepIndex() throws {
        try requireMLXRuntime()
        var requested: [Int] = []

        try NFKMLXTrainer.train(Linear(2, 2), optimizer: SGD(learningRate: 0.1), steps: 4,
                                batch: { step in
                                    requested.append(step)
                                    return self.fixedBatch()
                                }, loss: meanSquaredError)

        XCTAssertEqual(requested, [0, 1, 2, 3], "a real dataset indexes its batches by step")
    }

    func testAPeriodicCheckpointIsWrittenAndReloads() throws {
        try requireMLXRuntime()
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let model = Linear(2, 2)
        try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.1), steps: 10,
                                batch: { _ in self.fixedBatch() }, loss: meanSquaredError,
                                checkpoint: NFKMLXTrainingCheckpoint(url: url, everySteps: 5))

        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))

        // The point of the checkpoint: a suspended run resumes from the file the same way any model
        // loads its weights.
        let resumed = Linear(2, 2)
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        try NFKMLXWeights.apply(Array(checkpoint.arrays), to: resumed)
        XCTAssertEqual(resumed.weight.asArray(Float.self), model.weight.asArray(Float.self),
                       "the last write holds the state training reached")
    }

    // MARK: - Divergence

    /// A loss that turns non-finite after `finiteSteps`, standing in for the exploding update a real
    /// fine-tune produces on an unrepresentative batch.
    private func divergingLoss(after finiteSteps: Int) -> (Linear, MLXArray, MLXArray) -> MLXArray {
        var step = 0
        return { model, input, target in
            step += 1
            let base = (model(input) - target).square().mean()
            return step > finiteSteps ? base * Float.nan : base
        }
    }

    func testADivergedRunThrowsRatherThanContinuing() throws {
        try requireMLXRuntime()
        XCTAssertThrowsError(
            try NFKMLXTrainer.train(Linear(2, 2), optimizer: SGD(learningRate: 0.1), steps: 20,
                                    batch: { _ in self.fixedBatch() },
                                    loss: divergingLoss(after: 2))
        ) { error in
            guard case NFKMLXError.trainingDiverged(let detail) = error else {
                return XCTFail("expected trainingDiverged, got \(error)")
            }
            XCTAssertTrue(detail.contains("learning rate"), "says how to recover: \(detail)")
        }
    }

    func testADivergedRunLeavesTheCheckpointFinite() throws {
        try requireMLXRuntime()
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        // Writing diverged parameters would replace the run's only good state with a ruined one.
        XCTAssertThrowsError(
            try NFKMLXTrainer.train(Linear(2, 2), optimizer: SGD(learningRate: 0.1), steps: 20,
                                    batch: { _ in self.fixedBatch() },
                                    loss: divergingLoss(after: 2),
                                    checkpoint: NFKMLXTrainingCheckpoint(url: url, everySteps: 1)))

        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let recovered = Linear(2, 2)
        try NFKMLXWeights.apply(Array(checkpoint.arrays), to: recovered)
        XCTAssertTrue(recovered.weight.asArray(Float.self).allSatisfy { $0.isFinite },
                      "the surviving checkpoint holds the last finite state")
    }

    // MARK: - Gradient clipping

    func testClippingBoundsTheUpdateThatWouldOtherwiseDestroyTheWeights() throws {
        try requireMLXRuntime()
        let batch = fixedBatch()

        // A learning rate far too large for the batch: unclipped it diverges, clipped it stays finite.
        let unclipped = Linear(2, 2)
        XCTAssertThrowsError(
            try NFKMLXTrainer.train(unclipped, optimizer: SGD(learningRate: 50), steps: 40,
                                    batch: { _ in batch }, loss: meanSquaredError))

        let clipped = Linear(2, 2)
        let history = try NFKMLXTrainer.train(clipped, optimizer: SGD(learningRate: 50), steps: 40,
                                              batch: { _ in batch }, loss: meanSquaredError,
                                              clipGradientNorm: 0.01)
        XCTAssertTrue(history.allSatisfy { $0.isFinite }, "clipping kept the run alive")
    }

    func testTrainingRestoresTheModuleEvaluationMode() throws {
        try requireMLXRuntime()
        // Factories ship models in evaluation mode so batch normalization uses the checkpoint's running
        // statistics; a training run must hand the model back ready to infer.
        let model = Linear(2, 2)
        model.train(false)

        try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.1), steps: 2,
                                batch: { _ in self.fixedBatch() }, loss: meanSquaredError)

        XCTAssertFalse(model.training)
    }

    // MARK: Bounding the update

    // A gradient can be entirely finite while the SUM of its squares is not. The direct norm then
    // comes back infinite, `maxNorm / infinity` is zero, and the whole update is scaled to nothing —
    // silently, because the loss never stops being finite.
    func testAGradientWhoseSquaresOverflowIsStillScaledToTheNorm() throws {
        try requireMLXRuntime()
        let huge = Float(3e20)      // finite in Float; its square is not
        let gradients = ModuleParameters.unflattened([
            ("a", MLXArray([huge, -huge])),
            ("b", MLXArray([huge]))
        ])
        XCTAssertFalse((huge * huge).isFinite, "the premise: squaring this overflows")

        let bounded = NFKMLXTrainer.bounded(gradients, maxNorm: 1)
        let values = bounded.flattened().map { $0.1 }
        eval(values)

        // The norm is computed in Swift over the bounded values, which are ordinary magnitudes. Doing
        // it in MLX against the ORIGINAL scale would square numbers around 1e-21, and Metal flushes
        // subnormals to zero — the norm would read 0 whatever the gradients held.
        let flat = values.flatMap { $0.asArray(Float.self) }
        let norm = Float(sqrt(flat.reduce(Double(0)) { $0 + Double($1) * Double($1) }))

        XCTAssertTrue(flat.allSatisfy(\.isFinite), "the bounded gradients are finite")
        XCTAssertGreaterThan(norm, 0, "and are NOT scaled to zero, which is the failure being guarded")
        XCTAssertLessThanOrEqual(norm, 1.001, "the global norm is bounded by maxNorm")
    }

    // One non-finite entry would poison the norm and with it every other parameter's update.
    func testNonFiniteGradientEntriesAreZeroedRatherThanPoisoningTheRest() throws {
        try requireMLXRuntime()
        let gradients = ModuleParameters.unflattened([
            ("good", MLXArray([Float(3), 4])),
            ("bad", MLXArray([Float.nan, .infinity, -.infinity]))
        ])
        let bounded = NFKMLXTrainer.bounded(gradients, maxNorm: 100)
        let byName = Dictionary(uniqueKeysWithValues: bounded.flattened())
        eval(Array(byName.values))

        XCTAssertEqual(byName["bad"]!.asArray(Float.self), [0, 0, 0],
                       "the non-finite entries became zero")
        // maxNorm is above the good gradient's own norm of 5, so it passes through unscaled.
        let good = byName["good"]!.asArray(Float.self)
        XCTAssertEqual(good[0], 3, accuracy: 1e-4, "the finite gradient survived intact")
        XCTAssertEqual(good[1], 4, accuracy: 1e-4)
    }

    // Below the bound, the gradients must not be touched at all.
    func testGradientsInsideTheBoundPassThroughUnscaled() throws {
        try requireMLXRuntime()
        let gradients = ModuleParameters.unflattened([("g", MLXArray([Float(0.3), 0.4]))])
        let bounded = NFKMLXTrainer.bounded(gradients, maxNorm: 10)
        let values = bounded.flattened()[0].1
        eval(values)
        let array = values.asArray(Float.self)
        XCTAssertEqual(array[0], 0.3, accuracy: 1e-5)
        XCTAssertEqual(array[1], 0.4, accuracy: 1e-5)
    }

    // An all-zero gradient set divides by the norm floor rather than by zero.
    func testAnAllZeroGradientSetIsHandledWithoutProducingNaN() throws {
        try requireMLXRuntime()
        let gradients = ModuleParameters.unflattened([("g", MLXArray([Float(0), 0, 0]))])
        let bounded = NFKMLXTrainer.bounded(gradients, maxNorm: 1)
        let values = bounded.flattened()[0].1
        eval(values)
        XCTAssertTrue(values.asArray(Float.self).allSatisfy { $0 == 0 })
    }
}
