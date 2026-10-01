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
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        // Seeding initializes MLX's runtime, which needs a Metal library it can find; the methods
        // skip without one.
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
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

    /// An interrupted run resumed from a checkpoint that carries the optimizer's state ends exactly
    /// where the uninterrupted run ends: the same weights, Adam's moments, the schedule, and the batches.
    func testARunResumedWithItsOptimizerStateMatchesTheUninterruptedRun() throws {
        try requireMLXRuntime()
        let weights = temporaryURL(), state = temporaryURL()
        defer {
            try? FileManager.default.removeItem(at: weights)
            try? FileManager.default.removeItem(at: state)
        }
        let schedule = NFKMLXLearningRateSchedule { 1 / Float($0 + 1) }
        func batch(_ step: Int) -> (input: MLXArray, target: MLXArray) {
            let (input, target) = fixedBatch()
            return (input * Float(step + 1), target * Float(step + 1))
        }
        func freshModel() -> Linear { Linear(weight: MLXArray([Float(0.1), 0.2, -0.3, 0.4], [2, 2]), bias: MLXArray.zeros([2])) }
        func adam() -> Optimizer { NFKMLXReferenceOptimizers.adamW(learningRate: 0.05, weightDecay: 0.01) }

        let uninterrupted = freshModel()
        let all = try NFKMLXTrainer.train(uninterrupted, optimizer: adam(), steps: 6, batch: batch,
                                          loss: meanSquaredError, learningRateSchedule: schedule)

        let checkpoint = NFKMLXTrainingCheckpoint(url: weights, everySteps: 1, optimizerStateURL: state, resumes: true)
        _ = try NFKMLXTrainer.train(freshModel(), optimizer: adam(), steps: 6, batch: batch, loss: meanSquaredError,
                                    learningRateSchedule: schedule, checkpoint: checkpoint) { $0.index < 2 }
        var seen = [Int]()
        let resumed = freshModel()
        let rest = try NFKMLXTrainer.train(resumed, optimizer: adam(), steps: 6, batch: batch, loss: meanSquaredError,
                                           learningRateSchedule: schedule, checkpoint: checkpoint) {
            seen.append($0.index)
            return true
        }
        XCTAssertEqual(seen, [3, 4, 5], "the run continues at the update after the last write")
        XCTAssertEqual(rest, Array(all[3...]))
        XCTAssertEqual(resumed.weight.asArray(Float.self), uninterrupted.weight.asArray(Float.self))
    }

    func testAWeightsOnlyCheckpointDoesNotResume() throws {
        try requireMLXRuntime()
        let weights = temporaryURL()
        defer { try? FileManager.default.removeItem(at: weights) }
        let checkpoint = NFKMLXTrainingCheckpoint(url: weights, everySteps: 1, resumes: true)
        _ = try NFKMLXTrainer.train(Linear(2, 2), optimizer: SGD(learningRate: 0.1), steps: 2,
                                    batch: { _ in self.fixedBatch() }, loss: meanSquaredError, checkpoint: checkpoint)
        let history = try NFKMLXTrainer.train(Linear(2, 2), optimizer: SGD(learningRate: 0.1), steps: 2,
                                              batch: { _ in self.fixedBatch() }, loss: meanSquaredError,
                                              checkpoint: checkpoint)
        XCTAssertEqual(history.count, 2, "without the optimizer's state there is nothing to continue")
    }

    func testAnOptimizerStateCheckpointRefusesAnUnreadableOptimizer() throws {
        try requireMLXRuntime()
        let weights = temporaryURL(), state = temporaryURL()
        defer {
            try? FileManager.default.removeItem(at: weights)
            try? FileManager.default.removeItem(at: state)
        }
        XCTAssertThrowsError(try NFKMLXTrainer.train(
            Linear(2, 2), optimizer: Adam(learningRate: 0.1), steps: 2, batch: { _ in self.fixedBatch() },
            loss: meanSquaredError, checkpoint: NFKMLXTrainingCheckpoint(url: weights, everySteps: 1, optimizerStateURL: state)))
    }

    func testValidationScoresInEvaluationModeAndCanStopTheRun() throws {
        try requireMLXRuntime()
        let model = DropoutOverFrozenEncoder()
        model.encoder.freeze()
        var validated = [(step: Int, training: Bool, normTraining: Bool)]()
        var scores = [Float]()
        let observer = NFKMLXTrainer.validating(model, every: 2, evaluate: { model in
            validated.append((-1, model.training, model.normalization.training))
            return Float(scores.count)
        }, report: { step, score in
            validated[validated.count - 1].step = step
            scores.append(score)
            return scores.count < 2
        })
        var modes = [(model: Bool, normalization: Bool)]()
        let history = try NFKMLXTrainer.train(
            model, optimizer: SGD(learningRate: 0.01), steps: 10,
            batch: { _ in (MLXArray.ones([4, 4]), MLXArray.zeros([4, 2])) },
            loss: { model, input, target in
                modes.append((model.training, model.normalization.training))
                return (model.head(model.encoder(input)) - target).square().mean()
            }, observer: observer)
        XCTAssertEqual(validated.map(\.step), [1, 3], "every second update")
        XCTAssertTrue(validated.allSatisfy { !$0.training && !$0.normTraining }, "validation runs in evaluation mode")
        XCTAssertEqual(history.count, 4, "the second report ended the run")
        XCTAssertTrue(modes.allSatisfy { $0.model && !$0.normalization },
                      "the trainer's modes return after each validation, the frozen normalization evaluating")
    }

    // MARK: - Mixed precision

    private final class HeadOverFrozenLinear: Module {
        @ModuleInfo(key: "body") var body = Linear(2, 2)
        @ModuleInfo(key: "head") var head = Linear(2, 2)
        func callAsFunction(_ x: MLXArray) -> MLXArray { head(body(x)) }
    }

    func testABFloat16RunComputesInHalfAndKeepsFloat32Masters() throws {
        try requireMLXRuntime()
        let model = HeadOverFrozenLinear()
        model.body.freeze()
        let frozenBefore = model.body.weight.asArray(Float.self)
        let headBefore = model.head.weight.asArray(Float.self)
        let weights = temporaryURL()
        defer { try? FileManager.default.removeItem(at: weights) }
        var seen = [(head: DType, body: DType, input: DType)]()
        let history = try NFKMLXTrainer.train(
            model, optimizer: SGD(learningRate: 0.1), steps: 6, batch: { _ in self.fixedBatch() },
            loss: { model, input, target in
                seen.append((model.head.weight.dtype, model.body.weight.dtype, input.dtype))
                return (model(input) - target).square().mean()
            }, precision: .bfloat16, checkpoint: NFKMLXTrainingCheckpoint(url: weights, everySteps: 3))
        XCTAssertTrue(seen.allSatisfy { $0 == (.bfloat16, .bfloat16, .bfloat16) }, "the forward runs in bfloat16")
        XCTAssertEqual(model.head.weight.dtype, .float32, "the masters stay float32")
        XCTAssertNotEqual(model.head.weight.asArray(Float.self), headBefore, "the gradient reaches the masters")
        XCTAssertEqual(model.body.weight.dtype, .float32)
        XCTAssertEqual(model.body.weight.asArray(Float.self), frozenBefore, "the frozen values return exactly")
        XCTAssertLessThan(history.last!, history.first!)
        let written = try NFKMLXWeights.loadCheckpoint(url: weights).arrays
        XCTAssertEqual(written["body.weight"]?.dtype, .float32, "a checkpoint holds the frozen parameters at float32")
        XCTAssertEqual(written["body.weight"]?.asArray(Float.self), frozenBefore)
    }

    func testAFloat16RunSkipsAnOverflowingUpdateAndHalvesTheScale() throws {
        try requireMLXRuntime()
        let model = Linear(weight: MLXArray([Float(0.5), 0.25, -0.5, 1], [2, 2]), bias: MLXArray.zeros([2]))
        var weights = [[Float]]()
        // At a scale of 65,536 this gradient is about 2.6e5 a weight, past float16's 65,504; at half
        // the scale it still overflows, and the update applies once the scale has fallen far enough.
        _ = try NFKMLXTrainer.train(
            model, optimizer: SGD(learningRate: 1e-4), steps: 6, batch: { _ in self.fixedBatch() },
            loss: { model, input, _ in model(input).sum() * 2 }, precision: .float16) { _ in
                weights.append(model.weight.asArray(Float.self))
                return true
            }
        XCTAssertEqual(weights[0], [0.5, 0.25, -0.5, 1], "the first update overflowed and was skipped")
        XCTAssertNotEqual(weights.last!, [0.5, 0.25, -0.5, 1], "a later update applies")
        XCTAssertEqual(model.weight.dtype, .float32)
    }

    func testTheLossScaleHalvesOnOverflowAndDoublesAfterTwoThousandCleanUpdates() {
        var scale = NFKMLXLossScale()
        XCTAssertFalse(scale.record(finite: false))
        XCTAssertEqual(scale.scale, 32_768)
        for _ in 0 ..< 1_999 {
            XCTAssertTrue(scale.record(finite: true))
        }
        XCTAssertEqual(scale.scale, 32_768)
        XCTAssertTrue(scale.record(finite: true))
        XCTAssertEqual(scale.scale, 65_536)
    }

    // MARK: - Batch-driven accumulation

    /// The batches that complete an update, from ultralytics 8.4.120's own loop run with numpy
    /// (`engine/trainer.py:297`, `:469`, `:542`). The last case keeps the warm-up's final count of 2
    /// after the warm-up, where the setup's `round(64 / 24)` is 3.
    func testTheUltralyticsAccumulationUpdatesOnTheReferencesBatches() {
        func updates(_ batchSize: Int, _ warmup: Int, _ count: Int) -> [Int] {
            let accumulation = NFKMLXGradientAccumulation.ultralytics(batchSize: batchSize, warmupBatches: warmup)
            var previous = -1
            return (0 ..< count).filter { batch in
                guard accumulation.completesUpdate(batch, previous) else { return false }
                previous = batch
                return true
            }
        }
        XCTAssertEqual(updates(16, 20, 40), [0, 1, 2, 3, 5, 7, 9, 12, 15, 19, 23, 27, 31, 35, 39])
        XCTAssertEqual(updates(8, 0, 30), [7, 15, 23])
        XCTAssertEqual(updates(48, 7, 20), Array(0 ..< 20))
        XCTAssertEqual(updates(24, 10, 30), [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    }

    func testABatchDrivenRunSumsItsBatchesAndSchedulesAtTheCompletingBatch() throws {
        try requireMLXRuntime()
        func freshModel() -> Linear { Linear(weight: MLXArray([Float(0.1), 0.2, -0.3, 0.4], [2, 2]), bias: MLXArray.zeros([2])) }
        func batch(_ index: Int) -> (input: MLXArray, target: MLXArray) {
            let (input, target) = fixedBatch()
            return (input * Float(index + 1), target)
        }
        var scheduled = [Int]()
        let schedule = NFKMLXLearningRateSchedule { step in
            scheduled.append(step)
            return 1
        }
        let everyThird = NFKMLXGradientAccumulation(reduction: .sum) { batch, previous in batch - previous >= 3 }
        let model = freshModel()
        var flags = [Bool]()
        let history = try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.01), steps: 7, batch: batch,
                                              loss: meanSquaredError, accumulation: everyThird,
                                              learningRateSchedule: schedule) { step in
            flags.append(step.updated)
            return true
        }
        XCTAssertEqual(history.count, 7, "one loss a batch")
        XCTAssertEqual(flags, [false, false, true, false, false, true, false])
        XCTAssertEqual(scheduled, [2, 5], "the rate is set at the batch that completes each update")

        // The same two updates by hand: each the sum of three batches' gradients.
        let reference = freshModel()
        let gradient = valueAndGrad(model: reference) { model, arrays in
            [self.meanSquaredError(model, arrays[0], arrays[1])]
        }
        for group in [0 ..< 3, 3 ..< 6] {
            var sum: [String: MLXArray] = [:]
            for index in group {
                let (input, target) = batch(index)
                for (key, value) in gradient(reference, [input, target]).1.flattened() {
                    sum[key] = sum[key].map { $0 + value } ?? value
                }
            }
            SGD(learningRate: 0.01).update(model: reference, gradients: ModuleParameters.unflattened(sum.map { ($0.key, $0.value) }))
        }
        eval(model, reference)
        XCTAssertEqual(model.weight.asArray(Float.self), reference.weight.asArray(Float.self))
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

    // MARK: Frozen normalization

    /// A backbone whose parameters are frozen, under a head that trains: the shape of every
    /// head-only fine-tune in the package.
    private final class HeadOverFrozenBackbone: Module {

        @ModuleInfo(key: "backbone") var backbone: BatchNorm
        @ModuleInfo(key: "head") var head: Linear

        override init() {
            _backbone.wrappedValue = BatchNorm(featureCount: 2)
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

        /// `BatchNorm.runningMean` is internal to MLXNN, so the statistics are read the way a
        /// checkpoint sees them: as parameters of the module.
        var runningMean: [Float] {
            eval(self)
            let statistics = parameters().flattened().first { $0.0 == "backbone.running_mean" }
            return statistics!.1.asArray(Float.self)
        }
    }

    /// `train(true)` sets the flag on every module, and freezing does not touch it, so a frozen
    /// BatchNorm would fold the batch it sees into the statistics it was released with — and
    /// `NFKMLXWeights.save` would then write them.
    func testAFrozenNormalizationKeepsItsReleasedStatistics() throws {
        try requireMLXRuntime()
        let model = HeadOverFrozenBackbone()
        model.freezeBackbone()
        let before = model.runningMean

        let input = MLXArray([3.0, -4.0, 5.0, -6.0] as [Float]).reshaped([2, 2])
        _ = try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.01), steps: 3,
                                    batch: { _ in (input, MLXArray.zeros([2, 2])) },
                                    loss: { model, input, target in
                                        (model(input) - target).square().mean()
                                    })

        XCTAssertEqual(model.runningMean, before,
                       "the frozen backbone's running statistics did not move")
    }

    /// A frozen encoder holding a dropout and a normalization, under a head that trains.
    private final class DropoutOverFrozenEncoder: Module {
        @ModuleInfo(key: "encoder") var encoder: Sequential
        @ModuleInfo(key: "head") var head: Linear

        override init() {
            _encoder.wrappedValue = Sequential(layers: Linear(4, 4), Dropout(p: 0.5), BatchNorm(featureCount: 4))
            _head.wrappedValue = Linear(4, 2)
            super.init()
        }

        var dropout: Module { encoder.layers[1] as Module }
        var normalization: Module { encoder.layers[2] as Module }
    }

    /// The trainer evaluates a frozen normalization and leaves a frozen dropout dropping, as a PyTorch
    /// reference does when it freezes by `requires_grad_(False)` and calls `model.train()`.
    func testAFrozenEncoderStillDropsButKeepsItsStatistics() throws {
        try requireMLXRuntime()
        let model = DropoutOverFrozenEncoder()
        model.train(false)
        model.encoder.freeze()
        var modes = [(dropout: Bool, normalization: Bool)]()
        let input = MLXArray((0 ..< 8).map { Float($0) / 8 }).reshaped([2, 4])
        _ = try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.01), steps: 2,
                                    batch: { _ in (input, MLXArray.zeros([2, 2])) },
                                    loss: { model, input, target in
                                        modes.append((model.dropout.training, model.normalization.training))
                                        return (model.head(model.encoder(input)) - target).square().mean()
                                    })
        XCTAssertFalse(modes.isEmpty)
        XCTAssertTrue(modes.allSatisfy { $0.dropout }, "the frozen encoder's dropout drops")
        XCTAssertTrue(modes.allSatisfy { !$0.normalization }, "the frozen normalization evaluates")
        XCTAssertFalse(model.dropout.training, "the run restores the mode the caller left")
    }

    /// A recipe that unfreezes a whole network thaws its quantized layers with it. A gradient for a
    /// packed weight aborts the process, so the run freezes them again and trains the rest.
    func testARunOverAnUnfrozenQuantizedNetworkTrainsTheRest() throws {
        try requireMLXRuntime()
        let model = Sequential(layers: QuantizedLinear(Linear(64, 64), groupSize: 32, bits: 4), Linear(64, 2))
        model.unfreeze()
        func value(_ key: String) -> MLXArray {
            model.parameters().flattened().first { $0.0 == key }!.1
        }
        let packedBefore = value("layers.0.weight").asArray(UInt32.self)
        let headBefore = value("layers.1.weight").asArray(Float.self)

        let input = MLXArray((0 ..< 128).map { Float($0 % 9) / 9 }).reshaped([2, 64])
        _ = try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.1), steps: 2,
                                    batch: { _ in (input, MLXArray.ones([2, 2])) },
                                    loss: { model, input, target in
                                        (model(input) - target).square().mean()
                                    })

        XCTAssertEqual(value("layers.0.weight").asArray(UInt32.self), packedBefore, "the packed weight is held")
        XCTAssertNotEqual(value("layers.1.weight").asArray(Float.self), headBefore, "the head trains")
    }

    /// The frozen backbone also has to compute what it computes at inference. In training mode a
    /// BatchNorm normalizes with the batch's own mean and variance, which over two examples is a
    /// different function from the released one.
    func testAFrozenNormalizationNormalizesWithItsReleasedStatistics() throws {
        try requireMLXRuntime()
        let model = HeadOverFrozenBackbone()
        model.freezeBackbone()
        let input = MLXArray([3.0, -4.0, 5.0, -6.0] as [Float]).reshaped([2, 2])

        model.train(false)
        let atInference = model.backbone(input)
        eval(atInference)
        let expected = atInference.asArray(Float.self)

        var duringTraining: [Float] = []
        _ = try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0), steps: 1,
                                    batch: { _ in (input, MLXArray.zeros([2, 2])) },
                                    loss: { model, input, target in
                                        let normalized = model.backbone(input)
                                        eval(normalized)
                                        duringTraining = normalized.asArray(Float.self)
                                        return (model(input) - target).square().mean()
                                    })

        XCTAssertEqual(duringTraining.count, expected.count)
        for (during, inference) in zip(duringTraining, expected) {
            XCTAssertEqual(during, inference, accuracy: 1e-6)
        }
    }

    /// A normalization inside the group that trains still updates, or a full fine-tune would never
    /// adapt its statistics to the consumer's data.
    func testAnUnfrozenNormalizationStillUpdatesItsStatistics() throws {
        try requireMLXRuntime()
        let model = HeadOverFrozenBackbone()
        model.unfreeze()
        let before = model.runningMean

        let input = MLXArray([3.0, -4.0, 5.0, -6.0] as [Float]).reshaped([2, 2])
        _ = try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.01), steps: 3,
                                    batch: { _ in (input, MLXArray.zeros([2, 2])) },
                                    loss: { model, input, target in
                                        (model(input) - target).square().mean()
                                    })

        XCTAssertNotEqual(model.runningMean, before)
    }

    /// A parent's recursive `unfreeze()` makes a `BatchNorm`'s running statistics trainable again. Their
    /// gradient is zero, but a decoupled weight decay still shrinks them each step, so the trainer keeps
    /// them out of the trainable set: one step from zero with a batch mean of 3 reaches MLXNN's
    /// `0.1 · 3`, where the decay would take a quarter off it.
    func testARecursiveUnfreezeLeavesTheStatisticsOutOfTheWeightDecay() throws {
        try requireMLXRuntime()
        let model = HeadOverFrozenBackbone()
        model.unfreeze()
        let input = MLXArray([3.0, 3.0, 3.0, 3.0] as [Float]).reshaped([2, 2])
        _ = try NFKMLXTrainer.train(model, optimizer: AdamW(learningRate: 0.5, weightDecay: 0.5), steps: 1,
                                    batch: { _ in (input, MLXArray.zeros([2, 2])) },
                                    loss: { model, input, target in
                                        (model(input) - target).square().mean()
                                    })
        for value in model.runningMean {
            XCTAssertEqual(value, 0.3, accuracy: 1e-6)
        }
    }

    /// The models this rule protects hold their stages in `[Module]` arrays, not in named
    /// properties, so the walk has to descend through array structure to reach a frozen backbone.
    private final class HeadOverArrayHeldBackbone: Module {

        @ModuleInfo(key: "stages") var stages: [BatchNorm]
        @ModuleInfo(key: "head") var head: Linear

        override init() {
            _stages.wrappedValue = [BatchNorm(featureCount: 2), BatchNorm(featureCount: 2)]
            _head.wrappedValue = Linear(2, 2)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            head(stages.reduce(x) { $1($0) })
        }

        var runningMeans: [[Float]] {
            eval(self)
            return parameters().flattened()
                .filter { $0.0.hasPrefix("stages.") && $0.0.hasSuffix(".running_mean") }
                .map { $0.1.asArray(Float.self) }
        }
    }

    func testAFrozenBackboneHeldInAnArrayKeepsItsStatistics() throws {
        try requireMLXRuntime()
        let model = HeadOverArrayHeldBackbone()
        model.unfreeze()
        for stage in model.stages {
            stage.freeze()
        }
        let before = model.runningMeans
        XCTAssertEqual(before.count, 2, "the premise: both array-held stages carry statistics")

        let input = MLXArray([3.0, -4.0, 5.0, -6.0] as [Float]).reshaped([2, 2])
        _ = try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.01), steps: 3,
                                    batch: { _ in (input, MLXArray.zeros([2, 2])) },
                                    loss: { model, input, target in
                                        (model(input) - target).square().mean()
                                    })

        XCTAssertEqual(model.runningMeans, before)
    }

    // MARK: Guards

    /// A predicate that matched nothing leaves every parameter frozen, and a run over it reports a
    /// falling loss curve while changing nothing.
    func testAFullyFrozenModelIsRefusedRatherThanTrainingNothing() throws {
        try requireMLXRuntime()
        let model = Linear(2, 2)
        model.freeze()
        let batch = fixedBatch()

        XCTAssertThrowsError(try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 0.1),
                                                     steps: 5, batch: { _ in batch },
                                                     loss: meanSquaredError)) { error in
            guard case NFKMLXError.nothingToTrain = error else {
                return XCTFail("expected nothingToTrain, got \(error)")
            }
        }
    }

    /// The loop writes on `(step + 1) % everySteps`, which traps on zero.
    func testANonPositiveCheckpointIntervalMeansEveryStep() {
        XCTAssertEqual(NFKMLXTrainingCheckpoint(url: temporaryURL(), everySteps: 0).everySteps, 1)
        XCTAssertEqual(NFKMLXTrainingCheckpoint(url: temporaryURL(), everySteps: -4).everySteps, 1)
    }
}
