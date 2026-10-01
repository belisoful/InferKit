//
//  NFKMLXTrainer.swift
//  InferKitMLX
//
//  The supervised training loop, for customizing a shipped model on a consumer's own data.
//
//  The toolkit owns the loop, the caller owns the task: the loss closure carries what "correct" means
//  for a given model, the same way `NFKMLXModuleBackend` takes the forward closure. Which parameters
//  train is set by freezing the rest before calling in — `valueAndGrad` differentiates only
//  `trainableParameters()`, so a frozen backbone costs neither gradients nor optimizer state, which is
//  what makes fine-tuning on a device viable.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// One completed step of a training run.
public struct NFKMLXTrainingStep: Sendable {

    /// The step's zero-based position in the run.
    public let index: Int

    /// The number of steps the run was asked for.
    public let count: Int

    /// The loss the step reported.
    public let loss: Float

    /// Whether the step ended with an optimizer update. Under an ``NFKMLXGradientAccumulation`` a step
    /// is one batch, and only the batch that completes an update updates; otherwise every step does.
    ///
    /// Introduced in InferKit 0.4.0.
    public let updated: Bool

    init(index: Int, count: Int, loss: Float, updated: Bool = true) {
        self.index = index
        self.count = count
        self.loss = loss
        self.updated = updated
    }
}

/// How a run groups its batches into optimizer updates, for a reference whose grouping is not a fixed
/// count. Under one, a run's `steps` counts batches, the observer sees each batch, and the schedule
/// sets the rate from the index of the batch that completes each update.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXGradientAccumulation: Sendable {

    /// How the gradients of an update's batches combine.
    public enum Reduction: Sendable {
        /// Their mean, as transformers' `gradient_accumulation_steps` takes it.
        case mean
        /// Their sum, as a reference that calls `backward()` per batch and steps later takes it.
        case sum
    }

    public let reduction: Reduction

    /// Whether `batch` completes an update, given the batch that completed the previous one (−1
    /// before the first).
    public let completesUpdate: @Sendable (_ batch: Int, _ previousUpdate: Int) -> Bool

    public init(reduction: Reduction, completesUpdate: @escaping @Sendable (_ batch: Int, _ previousUpdate: Int) -> Bool) {
        self.reduction = reduction
        self.completesUpdate = completesUpdate
    }

    /// ultralytics' `BaseTrainer` (v8.4.120, `engine/trainer.py`): each batch's gradient adds to the
    /// sum, and batch `ni` updates when `ni − last ≥ accumulate`. `accumulate` is
    /// `max(round(nominalBatchSize / batchSize), 1)`, and during the first `warmupBatches` batches it is
    /// `max(1, round(interp(ni, [0, warmupBatches], [1, nominalBatchSize / batchSize])))`, keeping its
    /// last warm-up value after; both round half to even, as Python and numpy do.
    public static func ultralytics(batchSize: Int, warmupBatches: Int,
                                   nominalBatchSize: Int = 64) -> NFKMLXGradientAccumulation {
        let ratio = Double(nominalBatchSize) / Double(max(batchSize, 1))
        let setUp = max(Int(ratio.rounded(.toNearestOrEven)), 1)
        @Sendable func ramp(_ batch: Int) -> Int {
            max(1, Int((1 + (ratio - 1) * Double(batch) / Double(warmupBatches)).rounded(.toNearestOrEven)))
        }
        let afterWarmup = warmupBatches > 0 ? ramp(warmupBatches - 1) : setUp
        return NFKMLXGradientAccumulation(reduction: .sum) { batch, previous in
            batch - previous >= (batch < warmupBatches ? ramp(batch) : afterWarmup)
        }
    }
}

/// Writes the model to `url` every `everySteps` steps, and optionally the optimizer's state beside it so
/// a later run can resume where this one stopped.
///
/// A run lasts minutes, and an app can be suspended part way through one. Periodic checkpoints make
/// the work already done survive that.
///
/// - A weights-only checkpoint (`optimizerStateURL` nil) restarts Adam's moments at zero when a run
///   starts again from it, so its first updates are several times larger than the interrupted run's.
/// - With `optimizerStateURL`, each write also records the optimizer's state and the number of
///   completed updates. A run given the same checkpoint with `resumes` set loads both files when they
///   exist and continues at the next update, with the schedule, the batches, and the observer's step
///   indices where they would have been. The returned losses cover the resumed updates. The optimizer
///   must be a recipe's reference optimizer or an ``NFKMLXResumableOptimizer``.
///
/// Introduced in InferKit 0.4.0 (`optimizerStateURL`, `resumes`).
public struct NFKMLXTrainingCheckpoint: Sendable {

    /// The destination, which must be a `.safetensors` file.
    public let url: URL

    /// How many steps pass between writes. At least one.
    public let everySteps: Int

    /// Where each write also records the optimizer's state, a `.safetensors` file; nil writes weights only.
    public let optimizerStateURL: URL?

    /// Whether a run continues from the files this checkpoint names when both exist.
    public let resumes: Bool

    public init(url: URL, everySteps: Int, optimizerStateURL: URL? = nil, resumes: Bool = false) {
        self.url = url
        // The loop writes on `(step + 1) % everySteps`, which traps on zero. A caller asking for a
        // non-positive interval means every step.
        self.everySteps = max(everySteps, 1)
        self.optimizerStateURL = optimizerStateURL
        self.resumes = resumes
    }

    /// The completed-update count of the run to continue, after loading its weights into `model` and
    /// its state into `optimizer`; 0 when there is nothing to resume.
    func resume(_ model: Module, optimizer: Optimizer) throws -> Int {
        guard resumes, let optimizerStateURL,
              FileManager.default.fileExists(atPath: url.path),
              FileManager.default.fileExists(atPath: optimizerStateURL.path) else {
            return 0
        }
        let weights = try NFKMLXWeights.loadCheckpoint(url: url)
        try NFKMLXWeights.apply(weights.arrays.map { ($0.key, $0.value) }, to: model, verifyShapes: true)
        let state = try NFKMLXOptimizerState.load(from: optimizerStateURL)
        try NFKMLXOptimizerState.restore(state.arrays, into: optimizer)
        return state.completedSteps
    }

    /// Writes the model and, when one is named, the optimizer's state after `completedSteps` updates.
    func write(_ model: Module, optimizer: Optimizer, completedSteps: Int) throws {
        try NFKMLXWeights.save(model, to: url)
        if let optimizerStateURL {
            try NFKMLXOptimizerState.save(optimizer, completedSteps: completedSteps, to: optimizerStateURL)
        }
    }
}

/// How a training run treats MLX's Metal buffer cache.
///
/// A gradient after the first in a process can come back wrong by a factor of about a million, with
/// no infinity and no not-a-number to give it away. MLX's Metal buffer cache is involved, and the
/// mechanism is not established. Measured on an M1 Max over 25 backward passes of a
/// MobileNetV3 stem and four inverted residuals, one of the 25 matched the arbitrated gradient with
/// the cache left alone, and 25 of 25 matched it with the cache held at zero. The full measurements
/// are in `Docs/mlx-runtime-hazards.md`.
///
/// The defect belongs to mlx core, which fixes it in 0.32.0. mlx-swift 0.31.6 vendors core 0.31.1
/// and is the newest tag, so this package still needs the workaround. **Retire this type when
/// mlx-swift ships a release vendoring core 0.32.0 or later:** run
/// `swift test --filter NFKMLXUpstreamWatchTests` alone in a fresh process, and make ``unchanged``
/// the trainer default once the watch reports the fault is not observed.
public enum NFKMLXTrainingCachePolicy: Sendable {

    /// Holds the buffer cache at zero for the run, and only when the run is on the GPU.
    ///
    /// This is the default, because a wrong gradient reports no error. Six-step GPU runs ended above
    /// where they started 8 times in 60 with the cache left alone, and 0 times in 60 under this
    /// policy. The cache is a process-wide setting, so a concurrent inference on another thread
    /// allocates without it until the run returns. Measured cost on a 23.6M-parameter stack is 15%
    /// to 26% of throughput, against 4.58 GB of buffers the run no longer holds.
    ///
    /// A run on the CPU is left alone, because CPU gradients are already accurate and because
    /// returning buffers to the system raises the rate of a separate MLX crash. See
    /// `Docs/mlx-runtime-hazards.md`.
    case disabledOnGPU

    /// Returns the cache to the system before each step, and leaves the limit alone.
    ///
    /// This reduces the fault without removing it, because a step refills the cache before its own
    /// backward pass runs. Six-step GPU runs ended above where they started 4 times in 60 under this
    /// policy. It changes no process-wide setting, so it cannot affect another thread.
    case reclaimedEachStep

    /// Leaves the cache exactly as the process configured it.
    ///
    /// Appropriate where the graph is known not to trigger the fault, which is common: 14 synthetic
    /// graphs were exact on both devices, including smooth stacks to depth 16, gated pooling,
    /// resampling, skip concatenation, and stacks of `relu`, `hardswish`, and `hardsigmoid`. Verify
    /// a given graph against the CPU before relying on this.
    case unchanged
}

/// The precision a training run computes in.
///
/// The half-precision cases keep each trainable parameter in float32 as its master copy. Each forward
/// pass runs on a half-precision copy of those parameters, of every floating parameter the run
/// freezes, and of the batch's floating arrays, and the gradient returns to the float32 masters, which
/// the optimizer updates. The frozen parameters return to their float32 values when the run ends. A
/// PyTorch reference's autocast chooses a precision per operation; this runs every operation of the
/// forward in the half type, so its numbers follow the reference's closely and not exactly.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXTrainingPrecision: Sendable, Equatable {

    /// Every computation in float32. The default.
    case float32

    /// The forward and backward passes in bfloat16, which covers float32's range.
    case bfloat16

    /// The forward and backward passes in float16, with PyTorch `GradScaler`'s dynamic loss scaling: the
    /// loss is multiplied by a scale that starts at 65,536, an update whose gradient overflows is skipped
    /// and halves the scale, and 2,000 updates in a row without one double it.
    case float16

    var dtype: DType? {
        switch self {
        case .float32: return nil
        case .bfloat16: return .bfloat16
        case .float16: return .float16
        }
    }
}

/// The float32 parameters a half-precision run leaves frozen, cast to the half type for the run and
/// restored to their own values for each checkpoint and at the end.
final class NFKMLXHalfFrozenParameters {
    private let model: Module
    private let originals: [(String, MLXArray)]
    private let halves: [(String, MLXArray)]

    init(of model: Module, dtype: DType) {
        self.model = model
        let trainable = Set(model.trainableParameters().flattened().map(\.0))
        // `update(parameters:)` repoints the model's own array objects in place, so the originals are
        // held as copies that no later update can reach.
        originals = model.parameters().flattened()
            .filter { !trainable.contains($0.0) && $0.1.dtype == .float32 }
            .map { ($0.0, $0.1 * 1) }
        halves = originals.map { ($0.0, $0.1.asType(dtype)) }
        eval(originals.map(\.1) + halves.map(\.1))
        apply()
    }

    func apply() {
        model.update(parameters: ModuleParameters.unflattened(halves))
    }

    func restore() {
        model.update(parameters: ModuleParameters.unflattened(originals))
    }
}

/// PyTorch `GradScaler`'s dynamic loss scale, with its defaults.
struct NFKMLXLossScale {
    private(set) var scale: Float = 65_536
    private var cleanUpdates = 0

    /// Records whether an update's gradients were finite; returns whether the update applies.
    mutating func record(finite: Bool) -> Bool {
        guard finite else {
            scale *= 0.5
            cleanUpdates = 0
            return false
        }
        cleanUpdates += 1
        if cleanUpdates == 2_000 {
            scale *= 2
            cleanUpdates = 0
        }
        return true
    }
}

/// Runs a supervised training loop over an MLX module.
public enum NFKMLXTrainer {

    /// Supplies the input and target for one step, given the step's index.
    public typealias BatchSource = (Int) -> (input: MLXArray, target: MLXArray)

    /// Reports a completed step. Return false to end the run early.
    public typealias Observer = (NFKMLXTrainingStep) -> Bool

    /// Trains `model` for `steps` steps and returns the loss from each one.
    ///
    /// - Parameters:
    ///   - model: the module to train. Freeze the parameters that should not change before calling.
    ///   - optimizer: the update rule. `SGD` carries no per-parameter state, `Adam` carries two
    ///     arrays per trainable parameter, which is the dominant memory cost of a large run.
    ///   - steps: how many batches to train on.
    ///   - batch: supplies the input and target for each step.
    ///   - loss: measures the model's output against the target. It receives the model rather than
    ///     capturing it, because `valueAndGrad` calls it with the parameters under differentiation.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. Fine-tuning runs on
    ///     small datasets, where one unrepresentative batch can produce an update large enough to
    ///     destroy the pretrained weights. Non-finite entries are zeroed and the norm is computed
    ///     against the largest magnitude present, so a batch whose gradients are large but finite
    ///     cannot overflow the norm and silently scale the whole update to zero.
    ///   - accumulationSteps: how many batches each update averages, for an update larger than one
    ///     batch fits in memory. Step `s` reads batches `s · accumulationSteps + k` for `k` in
    ///     `0 ..< accumulationSteps` and averages their gradients, as transformers'
    ///     `gradient_accumulation_steps` and mmengine's `accumulative_counts` do. `steps` counts
    ///     updates; the schedule, the clip, the checkpoint, and the observer act once per update, and
    ///     the loss reported for a step is the mean over its batches.
    ///   - learningRateSchedule: multiplies the optimizer's rate at each zero-based step. Every
    ///     group of a `MultiOptimizer` is scaled from its own base rate, and the rates are restored
    ///     when the run ends. ``NFKMLXLearningRateSchedule`` builds the recipes' reference schedules.
    ///   - checkpoint: writes the model periodically, so a suspended run keeps its progress. The
    ///     optimizer's own state is not written, because mlx-swift keeps it private: a resumed `SGD`
    ///     run continues exactly, while a resumed `Adam` run rebuilds its moment estimates and shows
    ///     a brief rise in loss.
    ///   - cachePolicy: how the run treats MLX's Metal buffer cache. The default keeps a GPU run's
    ///     gradients correct. See ``NFKMLXTrainingCachePolicy``.
    ///   - observer: receives each step and can end the run early.
    ///
    /// - Throws: `NFKMLXError.trainingDiverged` when a step's loss stops being finite, before that
    ///   step can reach a checkpoint. `NFKMLXError.nothingToTrain` when every parameter is frozen,
    ///   which a predicate that matched no layer leaves behind. `NFKMLXError.unsupportedConfiguration`
    ///   when a schedule is given for an optimizer without a single rate to scale.
    ///
    /// The module is put in training mode for the duration and restored afterward, so the group that
    /// trains updates its batch-normalization statistics and returns ready to infer. A normalization
    /// whose own parameters are all frozen stays in evaluation mode instead, so a frozen backbone
    /// normalizes with the statistics it was released with and does not fold the training batches
    /// into them. Dropout runs everywhere, frozen parts included, as PyTorch's `model.train()` runs it.
    ///
    /// A run is multi-second; call it off the main thread.
    @discardableResult
    public static func train<Model: Module>(
        _ model: Model,
        optimizer: Optimizer,
        steps: Int,
        batch: BatchSource,
        loss: @escaping (Model, MLXArray, MLXArray) -> MLXArray,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        accumulation: NFKMLXGradientAccumulation? = nil,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        cachePolicy: NFKMLXTrainingCachePolicy = .disabledOnGPU,
        observer: Observer? = nil
    ) throws -> [Float] {
        try run(model, optimizer: optimizer, steps: steps,
                arrays: { let (input, target) = batch($0); return [input, target] },
                loss: { model, arrays in loss(model, arrays[0], arrays[1]) },
                accumulationSteps: accumulationSteps, clipGradientNorm: clipGradientNorm,
                learningRateSchedule: learningRateSchedule,
                precision: precision, accumulation: accumulation, checkpoint: checkpoint, cachePolicy: cachePolicy,
                observer: observer)
    }

    /// Trains `model` for `steps` steps against a loss that needs no ground truth, and returns the
    /// loss from each one.
    ///
    /// A zero-reference objective scores the output on its own properties rather than against a
    /// target, so a consumer customizes the model from unlabeled examples: their own photos, with
    /// nothing to annotate. ``NFKMLXZeroDCEObjective`` is the shipped one.
    ///
    ///
    /// - Parameters:
    ///   - model: the module to train. Freeze the parameters that should not change before calling.
    ///   - optimizer: the update rule.
    ///   - steps: how many batches to train on.
    ///   - sample: supplies the unlabeled batch for each step.
    ///   - loss: scores the model on that batch alone.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - accumulationSteps: how many batches each update averages; see the supervised form.
    ///   - learningRateSchedule: multiplies the optimizer's rate at each zero-based step.
    ///   - checkpoint: writes the model periodically, so a suspended run keeps its progress.
    ///   - cachePolicy: how the run treats MLX's Metal buffer cache. The default keeps a GPU run's
    ///     gradients correct. See ``NFKMLXTrainingCachePolicy``.
    ///   - observer: receives each step and can end the run early.
    @discardableResult
    public static func train<Model: Module>(
        _ model: Model,
        optimizer: Optimizer,
        steps: Int,
        sample: (Int) -> MLXArray,
        loss: @escaping (Model, MLXArray) -> MLXArray,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        accumulation: NFKMLXGradientAccumulation? = nil,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        cachePolicy: NFKMLXTrainingCachePolicy = .disabledOnGPU,
        observer: Observer? = nil
    ) throws -> [Float] {
        try run(model, optimizer: optimizer, steps: steps,
                arrays: { [sample($0)] },
                loss: { model, arrays in loss(model, arrays[0]) },
                accumulationSteps: accumulationSteps, clipGradientNorm: clipGradientNorm,
                learningRateSchedule: learningRateSchedule,
                precision: precision, accumulation: accumulation, checkpoint: checkpoint, cachePolicy: cachePolicy,
                observer: observer)
    }

    /// Trains `model` on any number of arrays per step: an image, a prompt, and a target, for instance.
    ///
    /// `constraint` runs on the model after every optimizer update, before the step is evaluated,
    /// checkpointed, or reported. It is where a reference's weight constraint belongs: Keras applies a
    /// variable's `kernel_constraint` after `apply_gradients`, whatever the optimizer, so the
    /// constraint is a property of the run and applies whichever optimizer is passed. It was
    /// introduced in InferKit 0.4.0. `accumulationSteps` averages that many batches into each update,
    /// as the supervised form describes; the constraint runs once per update.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func train<Model: Module>(
        _ model: Model,
        optimizer: Optimizer,
        steps: Int,
        arrays: (Int) -> [MLXArray],
        loss: @escaping (Model, [MLXArray]) -> MLXArray,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        accumulation: NFKMLXGradientAccumulation? = nil,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        cachePolicy: NFKMLXTrainingCachePolicy = .disabledOnGPU,
        constraint: ((Model) -> Void)? = nil,
        observer: Observer? = nil
    ) throws -> [Float] {
        try run(model, optimizer: optimizer, steps: steps, arrays: arrays, loss: loss,
                accumulationSteps: accumulationSteps, clipGradientNorm: clipGradientNorm,
                learningRateSchedule: learningRateSchedule,
                precision: precision, accumulation: accumulation, checkpoint: checkpoint, cachePolicy: cachePolicy,
                constraint: constraint,
                observer: observer)
    }

    /// An observer that validates `model` every `every` updates, for any run or recipe that takes an
    /// observer.
    ///
    /// At each validation the model switches to evaluation mode, so dropout is off and a `BatchNorm`
    /// normalizes with its running statistics, and `evaluate` scores it on the caller's held-out data.
    /// Training mode then returns as the trainer set it. `report` receives the update's index and the
    /// score, and returning false ends the run there, which is how a run stops early when the score
    /// stops improving. `observer`, when given, still sees every update.
    ///
    /// - Parameters:
    ///   - model: the model the run trains.
    ///   - every: how many updates pass between validations. At least one.
    ///   - evaluate: scores the model; it runs without gradients and should not update the model.
    ///   - report: receives each validation's update index and score; false ends the run.
    ///   - observer: receives every update, as a run's observer does.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func validating<Model: Module>(
        _ model: Model, every: Int, evaluate: @escaping (Model) -> Float,
        report: @escaping (_ step: Int, _ score: Float) -> Bool, observer: Observer? = nil
    ) -> Observer {
        let interval = max(every, 1)
        return { step in
            let continues = observer?(step) ?? true
            guard (step.index + 1) % interval == 0 else {
                return continues
            }
            model.train(false)
            let score = evaluate(model)
            enterTrainingMode(model)
            return report(step.index, score) && continues
        }
    }

    /// The loop every entry point shares, over an arbitrary number of per-step arrays.
    private static func run<Model: Module>(
        _ model: Model,
        optimizer: Optimizer,
        steps: Int,
        arrays: (Int) -> [MLXArray],
        loss: @escaping (Model, [MLXArray]) -> MLXArray,
        accumulationSteps: Int,
        clipGradientNorm: Float?,
        learningRateSchedule: NFKMLXLearningRateSchedule?,
        precision: NFKMLXTrainingPrecision = .float32,
        accumulation: NFKMLXGradientAccumulation? = nil,
        checkpoint: NFKMLXTrainingCheckpoint?,
        cachePolicy: NFKMLXTrainingCachePolicy,
        constraint: ((Model) -> Void)? = nil,
        observer: Observer?
    ) throws -> [Float] {
        freezeHeldState(of: model)
        guard !model.trainableParameters().flattened().isEmpty else {
            throw NFKMLXError.nothingToTrain(
                "every parameter of this model is frozen, so the run would compute gradients for "
                + "nothing and leave the weights exactly as they are. Unfreeze the group to train, "
                + "or check that the LoRA predicate matched the layers it names.")
        }

        var scheduled = [(NFKMLXRateScheduled, Float)]()
        if learningRateSchedule != nil {
            guard let groups = NFKMLXLearningRateSchedule.scheduledGroups(of: optimizer) else {
                throw NFKMLXError.unsupportedConfiguration(
                    "a learning-rate schedule needs an optimizer with a single rate per group; "
                    + "\(type(of: optimizer)) has none to scale")
            }
            scheduled = groups
        }
        defer { for (group, rate) in scheduled { group.learningRate = rate } }

        // Installed before the training mode is set and restored after it is unset, so the swapped
        // convolutions take the run's mode and the originals get back the mode the caller left.
        let convolutions = try NFKMLXGradientSafeConvolution.install(in: model)
        defer { convolutions.restore() }

        let wasTraining = model.training
        enterTrainingMode(model)
        defer { model.train(wasTraining) }

        // The cache limit is process-wide, so it is restored even when a step throws. Reading it and
        // writing it both trim the buffer cache, and a trim frees buffers that work already sent to
        // the GPU is still reading, which corrupts whatever runs next rather than this run. The
        // stream is drained around both changes, and the limit is read only when it is going to be
        // changed, because mlx-swift's getter writes it twice on its first read in a process.
        let disablesCache = cachePolicy == .disabledOnGPU && NFKMLXDevice.currentType == .gpu
        var limitBeforeRun = 0
        if disablesCache {
            NFKMLXGPU.synchronize()
            limitBeforeRun = NFKMLXGPU.cacheLimit
            NFKMLXGPU.setCacheLimit(0)
        }
        defer {
            if disablesCache {
                NFKMLXGPU.synchronize()
                NFKMLXGPU.setCacheLimit(limitBeforeRun)
            }
        }

        guard accumulationSteps >= 1 else {
            throw NFKMLXError.unsupportedConfiguration(
                "an update averages at least one batch; accumulationSteps was \(accumulationSteps)")
        }
        let fullPrecision = valueAndGrad(model: model) { model, arrays in [loss(model, arrays)] }
        var lossScale = NFKMLXLossScale()
        // A resumed run loads its float32 checkpoint before the frozen parameters are cast.
        let firstStep = try checkpoint?.resume(model, optimizer: optimizer) ?? 0
        let frozen = precision.dtype.map { NFKMLXHalfFrozenParameters(of: model, dtype: $0) }
        defer { frozen?.restore() }
        let lossAndGradient: (Model, [MLXArray]) -> ([MLXArray], ModuleParameters) = { model, arrays in
            guard let dtype = precision.dtype else {
                return fullPrecision(model, arrays)
            }
            return halfPrecisionLossAndGradient(model, arrays, dtype: dtype,
                                                lossScale: precision == .float16 ? lossScale.scale : 1, loss: loss)
        }
        var history: [Float] = []
        history.reserveCapacity(steps)

        // One optimizer update from an update's combined gradient; false when float16 skips it.
        func apply(_ gradients: ModuleParameters, at step: Int) -> Bool {
            if let learningRateSchedule {
                let scale = learningRateSchedule.multiplier(step)
                for (group, rate) in scheduled { group.learningRate = rate * scale }
            }
            guard precision != .float16 || lossScale.record(finite: allFinite(gradients)) else {
                return false
            }
            let update = clipGradientNorm.map { bounded(gradients, maxNorm: $0) } ?? gradients
            optimizer.update(model: model, gradients: update)
            constraint?(model)
            return true
        }
        func checked(_ loss: MLXArray, at step: Int) throws -> Float {
            let value = loss.item(Float.self)
            // Diverged parameters are unrecoverable, and writing them would replace a good checkpoint
            // with a ruined one. Stop before the write rather than spending the device's battery on a
            // run that cannot recover.
            guard value.isFinite else {
                throw NFKMLXError.trainingDiverged(
                    "the loss became \(value) at step \(step) of \(steps); "
                    + "the checkpoint was left at its last finite state. Lower the learning rate, "
                    + "or set `clipGradientNorm` to bound the update.")
            }
            return value
        }
        func write(after completedSteps: Int) throws {
            guard let checkpoint else { return }
            // A checkpoint holds the frozen parameters at the precision the model was built in.
            frozen?.restore()
            try checkpoint.write(model, optimizer: optimizer, completedSteps: completedSteps)
            frozen?.apply()
        }

        if let accumulation {
            var previousUpdate = firstStep - 1
            var updates = 0
            var gradientSums = [String: MLXArray]()
            var batches = 0
            for batch in firstStep ..< steps {
                if cachePolicy == .reclaimedEachStep {
                    NFKMLXGPU.clearCache()
                }
                let (values, gradients) = lossAndGradient(model, arrays(batch))
                for (key, gradient) in gradients.flattened() {
                    gradientSums[key] = gradientSums[key].map { $0 + gradient } ?? gradient
                }
                batches += 1
                eval([values[0]] + Array(gradientSums.values))
                let batchLoss = try checked(values[0], at: batch)
                history.append(batchLoss)
                let completes = accumulation.completesUpdate(batch, previousUpdate)
                if completes {
                    let scale: Float = accumulation.reduction == .mean ? 1 / Float(batches) : 1
                    _ = apply(ModuleParameters.unflattened(gradientSums.map { ($0.key, scale == 1 ? $0.value : $0.value * scale) }),
                              at: batch)
                    eval(model, optimizer)
                    previousUpdate = batch
                    gradientSums.removeAll()
                    batches = 0
                    updates += 1
                    if let checkpoint, updates % checkpoint.everySteps == 0 {
                        try write(after: batch + 1)
                    }
                }
                if observer?(NFKMLXTrainingStep(index: batch, count: steps, loss: batchLoss, updated: completes)) == false {
                    break
                }
            }
            return history
        }

        for step in firstStep ..< steps {
            if cachePolicy == .reclaimedEachStep {
                NFKMLXGPU.clearCache()
            }
            let lossValue: MLXArray
            let gradients: ModuleParameters
            if accumulationSteps == 1 {
                let (values, batchGradients) = lossAndGradient(model, arrays(step))
                (lossValue, gradients) = (values[0], batchGradients)
            } else {
                (lossValue, gradients) = averaged(over: accumulationSteps) {
                    lossAndGradient(model, arrays(step * accumulationSteps + $0))
                }
            }
            _ = apply(gradients, at: step)
            // MLX builds the step lazily; this is where it runs.
            eval(model, optimizer)

            let stepLoss = try checked(lossValue, at: step)
            history.append(stepLoss)
            if let checkpoint, (step + 1) % checkpoint.everySteps == 0 {
                try write(after: step + 1)
            }
            if observer?(NFKMLXTrainingStep(index: step, count: steps, loss: stepLoss)) == false {
                break
            }
        }
        return history
    }

    /// Keeps out of the trainable set what no run trains: every normalization's running statistics
    /// and every array of a quantized layer.
    ///
    /// @discussion Both are frozen when their layer is built, and a parent's recursive `unfreeze()`
    /// clears every child's frozen set without calling the child, so a recipe that unfreezes a network
    /// makes them trainable. MLXNN's `QuantizedEmbedding` thaws under its own `unfreeze()` too.
    ///
    /// - A running mean or variance is an average of what the batches looked like, updated in place
    ///   during the forward pass. Its gradient is zero, but the optimizer carries state for it and a
    ///   decoupled weight decay shrinks it every step, which no reference does.
    /// - A quantized layer's packed `uint32` weight, scales, and biases have no gradient. MLX aborts the
    ///   process when asked for one (`[QuantizedMatmul::vjp] no gradient wrt the quantized weights`).
    ///   A quantized base trains through adapters or not at all.
    ///
    /// Freezing them here, after the recipe's freezing and before the loop, holds for every recipe.
    static func freezeHeldState(of model: Module) {
        for (_, module) in model.leafModules().flattened() {
            let held = heldKeys(of: module)
            if !held.isEmpty {
                module.freeze(recursive: false, keys: held)
            }
        }
    }

    /// The keys of `module`'s own parameters that no run trains.
    static func heldKeys(of module: Module) -> [String] {
        let keys = module.parameters().flattened().map(\.0)
        if module is Quantized {
            return keys
        }
        return keys.filter { $0 == "running_mean" || $0 == "running_var" }
    }

    /// Puts the model into training mode, leaving every frozen normalization in evaluation mode.
    ///
    /// @discussion `train(true)` sets the flag on every module in the tree, and freezing does not
    /// touch it. A `BatchNorm` whose flag is set normalizes with the batch's own mean and variance
    /// and folds them into its running statistics, so a head-only run over a pretrained
    /// convolutional backbone changes what the frozen backbone computes and overwrites the
    /// statistics it was released with, from batches of one or two examples.
    /// ``NFKMLXWeights/save(_:to:)`` then writes those statistics into the checkpoint, which is how
    /// the damage outlives the run.
    ///
    /// A module that keeps running statistics and has no trainable parameter is therefore returned
    /// to evaluation mode. Every other module trains, so a dropout or drop path in a frozen encoder
    /// still drops, as it does in a PyTorch reference that calls `model.train()` and freezes by
    /// `requires_grad_(False)`.
    static func enterTrainingMode(_ model: Module) {
        model.train(true)
        for (_, module) in [("", model)] + model.namedModules()
        where keepsRunningStatistics(module) && module.trainableParameters().flattened().isEmpty {
            module.train(false)
        }
    }

    /// Whether `module` holds a running mean or variance of its own.
    private static func keepsRunningStatistics(_ module: Module) -> Bool {
        module is BatchNorm || module.parameters().flattened().contains {
            $0.0 == "running_mean" || $0.0 == "running_var"
        }
    }

    /// Sanitizes the gradients, then scales them so their global norm is at most `maxNorm`.
    ///
    /// @discussion Two things happen here, and the order is the point.
    ///
    /// A gradient array can be entirely finite while the SUM of its squares is not: a value of 1e20
    /// is a perfectly good `Float`, and its square is past the type's maximum. A norm computed the
    /// direct way then comes back infinite, `maxNorm / infinity` is zero, and every gradient is
    /// scaled to nothing — after which the optimizer's moment estimates are built from zeros and the
    /// following steps can produce non-finite PARAMETERS. The loss stays finite the whole time, so
    /// the divergence guard never sees it and the run reports a plausible loss curve while the model
    /// is being destroyed. The norm here is therefore computed against the largest magnitude present,
    /// which keeps the squares in range whatever the scale of the gradients.
    ///
    /// Individually non-finite entries are replaced with zero first, because one such entry poisons
    /// the norm and with it every other parameter's update.
    static func bounded(_ gradients: ModuleParameters, maxNorm: Float) -> ModuleParameters {
        // `where(isFinite)` rather than `nanToNum`: mlx core 0.32 maps an infinity to the float's
        // largest magnitude even when the binding asks for zero, which makes the reference below
        // 3.4e38 and overflows the norm to infinity, scaling every FINITE gradient to zero. Testing
        // finiteness states the intent and does not depend on the core's replacement values.
        let zero = MLXArray(Float(0))
        let sanitized = gradients.flattened().map { ($0.0, MLX.where(isFinite($0.1), $0.1, zero)) }
        guard !sanitized.isEmpty else { return gradients }

        let magnitudes = sanitized.map { abs($0.1).max() }
        let largest = magnitudes.dropFirst().reduce(magnitudes[0]) { maximum($0, $1) }
        // Everything is measured relative to the largest magnitude, so the squares cannot overflow.
        // A floor keeps the division defined when every gradient is zero.
        let reference = maximum(largest, MLXArray(Float.leastNormalMagnitude))
        let squares = sanitized.map { ($0.1 / reference).square().sum() }
        let sumOfSquares = squares.dropFirst().reduce(squares[0]) { $0 + $1 }
        let norm = sqrt(sumOfSquares) * reference
        let factor = minimum(MLXArray(maxNorm) / maximum(norm, MLXArray(Float.leastNormalMagnitude)),
                             MLXArray(Float(1)))
        return ModuleParameters.unflattened(sanitized.map { ($0.0, $0.1 * factor) })
    }

    /// The mean loss and the mean gradients of `count` batches, each batch evaluated before the next
    /// runs, so the graph and the memory it holds stay one batch deep.
    /// The loss and the gradient with respect to the float32 masters, through a forward in `dtype`.
    /// The loss is multiplied by `lossScale` before the backward pass and both are divided by it after.
    private static func halfPrecisionLossAndGradient<Model: Module>(
        _ model: Model, _ arrays: [MLXArray], dtype: DType, lossScale: Float,
        loss: @escaping (Model, [MLXArray]) -> MLXArray
    ) -> ([MLXArray], ModuleParameters) {
        // Copies, because the forward's `update(parameters:)` repoints the model's own array objects.
        let masters = model.trainableParameters().flattened().map { ($0.0, $0.1 * 1) }
        let keys = masters.map(\.0)
        let batch = arrays.map { $0.dtype == .float32 ? $0.asType(dtype) : $0 }
        // Every master is an argument to differentiate; the default differentiates the first alone.
        let scaled = valueAndGrad({ (parameters: [MLXArray]) -> [MLXArray] in
            model.update(parameters: ModuleParameters.unflattened(zip(keys, parameters.map { $0.asType(dtype) }).map { ($0, $1) }))
            return [loss(model, batch).asType(.float32) * lossScale]
        }, argumentNumbers: Array(masters.indices))
        let (values, gradients) = scaled(masters.map(\.1))
        model.update(parameters: ModuleParameters.unflattened(masters))
        let unscale = 1 / lossScale
        return ([values[0] * unscale],
                ModuleParameters.unflattened(zip(keys, gradients.map { $0 * unscale }).map { ($0, $1) }))
    }

    /// Whether every gradient entry is finite.
    private static func allFinite(_ gradients: ModuleParameters) -> Bool {
        gradients.flattened().allSatisfy { MLX.isFinite($0.1).all().item(Bool.self) }
    }

    private static func averaged(over count: Int,
                                 _ batch: (Int) -> ([MLXArray], ModuleParameters)) -> (MLXArray, ModuleParameters) {
        var lossSum = MLXArray(Float(0))
        var gradientSums = [String: MLXArray]()
        for index in 0 ..< count {
            let (values, gradients) = batch(index)
            lossSum = lossSum + values[0]
            for (key, gradient) in gradients.flattened() {
                gradientSums[key] = gradientSums[key].map { $0 + gradient } ?? gradient
            }
            eval([lossSum] + Array(gradientSums.values))
        }
        let scale = 1 / Float(count)
        return (lossSum * scale,
                ModuleParameters.unflattened(gradientSums.map { ($0.key, $0.value * scale) }))
    }
}
