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
}

/// Writes the model to `url` every `everySteps` steps.
///
/// A run lasts minutes, and an app can be suspended part way through one. Periodic checkpoints make
/// the work already done survive that.
public struct NFKMLXTrainingCheckpoint: Sendable {

    /// The destination, which must be a `.safetensors` file.
    public let url: URL

    /// How many steps pass between writes. At least one.
    public let everySteps: Int

    public init(url: URL, everySteps: Int) {
        self.url = url
        // The loop writes on `(step + 1) % everySteps`, which traps on zero. A caller asking for a
        // non-positive interval means every step.
        self.everySteps = max(everySteps, 1)
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
    /// trains updates its batch-normalization statistics and returns ready to infer. A subtree whose
    /// parameters are all frozen stays in evaluation mode instead, so a frozen backbone normalizes
    /// with the statistics it was released with rather than folding the training batches into them.
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
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        cachePolicy: NFKMLXTrainingCachePolicy = .disabledOnGPU,
        observer: Observer? = nil
    ) throws -> [Float] {
        try run(model, optimizer: optimizer, steps: steps,
                arrays: { let (input, target) = batch($0); return [input, target] },
                loss: { model, arrays in loss(model, arrays[0], arrays[1]) },
                clipGradientNorm: clipGradientNorm, learningRateSchedule: learningRateSchedule,
                checkpoint: checkpoint, cachePolicy: cachePolicy, observer: observer)
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
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        cachePolicy: NFKMLXTrainingCachePolicy = .disabledOnGPU,
        observer: Observer? = nil
    ) throws -> [Float] {
        try run(model, optimizer: optimizer, steps: steps,
                arrays: { [sample($0)] },
                loss: { model, arrays in loss(model, arrays[0]) },
                clipGradientNorm: clipGradientNorm, learningRateSchedule: learningRateSchedule,
                checkpoint: checkpoint, cachePolicy: cachePolicy, observer: observer)
    }

    /// Trains `model` on any number of arrays per step: an image, a prompt, and a target, for instance.
    ///
    /// `constraint` runs on the model after every optimizer update, before the step is evaluated,
    /// checkpointed, or reported. It is where a reference's weight constraint belongs: Keras applies a
    /// variable's `kernel_constraint` after `apply_gradients`, whatever the optimizer, so the
    /// constraint is a property of the run and applies whichever optimizer is passed. It was
    /// introduced in InferKit 0.5.0.
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
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        cachePolicy: NFKMLXTrainingCachePolicy = .disabledOnGPU,
        constraint: ((Model) -> Void)? = nil,
        observer: Observer? = nil
    ) throws -> [Float] {
        try run(model, optimizer: optimizer, steps: steps, arrays: arrays, loss: loss,
                clipGradientNorm: clipGradientNorm, learningRateSchedule: learningRateSchedule,
                checkpoint: checkpoint, cachePolicy: cachePolicy, constraint: constraint,
                observer: observer)
    }

    /// The loop every entry point shares, over an arbitrary number of per-step arrays.
    private static func run<Model: Module>(
        _ model: Model,
        optimizer: Optimizer,
        steps: Int,
        arrays: (Int) -> [MLXArray],
        loss: @escaping (Model, [MLXArray]) -> MLXArray,
        clipGradientNorm: Float?,
        learningRateSchedule: NFKMLXLearningRateSchedule?,
        checkpoint: NFKMLXTrainingCheckpoint?,
        cachePolicy: NFKMLXTrainingCachePolicy,
        constraint: ((Model) -> Void)? = nil,
        observer: Observer?
    ) throws -> [Float] {
        freezeRunningStatistics(of: model)
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

        let lossAndGradient = valueAndGrad(model: model) { model, arrays in [loss(model, arrays)] }
        var history: [Float] = []
        history.reserveCapacity(steps)

        for step in 0 ..< steps {
            if cachePolicy == .reclaimedEachStep {
                NFKMLXGPU.clearCache()
            }
            if let learningRateSchedule {
                let scale = learningRateSchedule.multiplier(step)
                for (group, rate) in scheduled { group.learningRate = rate * scale }
            }
            let (values, gradients) = lossAndGradient(model, arrays(step))
            let update = clipGradientNorm.map { bounded(gradients, maxNorm: $0) } ?? gradients
            optimizer.update(model: model, gradients: update)
            constraint?(model)
            // MLX builds the step lazily; this is where it runs.
            eval(model, optimizer)

            let stepLoss = values[0].item(Float.self)
            history.append(stepLoss)

            // Diverged parameters are unrecoverable, and writing them would replace a good checkpoint
            // with a ruined one. Stop before the write rather than spending the device's battery on a
            // run that cannot recover.
            guard stepLoss.isFinite else {
                throw NFKMLXError.trainingDiverged(
                    "the loss became \(stepLoss) at step \(step) of \(steps); "
                    + "the checkpoint was left at its last finite state. Lower the learning rate, "
                    + "or set `clipGradientNorm` to bound the update.")
            }
            if let checkpoint, (step + 1) % checkpoint.everySteps == 0 {
                try NFKMLXWeights.save(model, to: checkpoint.url)
            }
            if observer?(NFKMLXTrainingStep(index: step, count: steps, loss: stepLoss)) == false {
                break
            }
        }
        return history
    }

    /// Keeps every normalization's running statistics out of the trainable set.
    ///
    /// @discussion A running mean or variance is an average of what the batches looked like, updated in
    /// place during the forward pass, never a gradient target. MLXNN's `BatchNorm` freezes the two when
    /// it is built, but a parent's recursive `unfreeze()` clears every child's frozen set without
    /// calling the child, so a recipe that unfreezes a network makes them trainable. Their gradient is
    /// zero, so a plain step leaves them alone, but the optimizer still carries state for them and a
    /// decoupled weight decay shrinks them every step, which no reference does. The loss is unaffected,
    /// because training mode normalizes by each batch's own statistics. Freezing them here, after the
    /// recipe's freezing and before the loop, holds for every recipe.
    static func freezeRunningStatistics(of model: Module) {
        let statistics: Set<String> = ["running_mean", "running_var"]
        for (_, module) in model.leafModules().flattened() {
            let held = statistics.filter { key in
                module.parameters().flattened().contains { $0.0 == key }
            }
            if !held.isEmpty {
                module.freeze(recursive: false, keys: Array(held))
            }
        }
    }

    /// Puts the model into training mode, leaving every fully frozen subtree in evaluation mode.
    ///
    /// @discussion `train(true)` sets the flag on every module in the tree, and freezing does not
    /// touch it. A `BatchNorm` whose flag is set normalizes with the batch's own mean and variance
    /// and folds them into its running statistics, so a head-only run over a pretrained
    /// convolutional backbone changes what the frozen backbone computes and overwrites the
    /// statistics it was released with, from batches of one or two examples.
    /// ``NFKMLXWeights/save(_:to:)`` then writes those statistics into the checkpoint, which is how
    /// the damage outlives the run.
    ///
    /// A subtree that holds parameters and has none trainable is therefore returned to evaluation
    /// mode. A subtree holding no parameters at all follows its parent, so a dropout inside the
    /// trainable group still drops.
    private static func enterTrainingMode(_ model: Module) {
        model.train(true)
        restoreEvaluationMode(inFrozenSubtreesOf: model)
    }

    /// Walks down from `module`, returning each wholly frozen subtree to evaluation mode.
    private static func restoreEvaluationMode(inFrozenSubtreesOf module: Module) {
        guard module.parameters().flattened().isEmpty
                || !module.trainableParameters().flattened().isEmpty else {
            // Nothing below this point trains, so the whole subtree evaluates.
            module.train(false)
            return
        }
        for (_, child) in module.children().flattened() {
            restoreEvaluationMode(inFrozenSubtreesOf: child)
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
}
