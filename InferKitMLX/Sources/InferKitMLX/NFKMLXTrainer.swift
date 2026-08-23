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

    /// How many steps pass between writes.
    public let everySteps: Int

    public init(url: URL, everySteps: Int) {
        self.url = url
        self.everySteps = everySteps
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
    ///     destroy the pretrained weights.
    ///   - checkpoint: writes the model periodically, so a suspended run keeps its progress. The
    ///     optimizer's own state is not written, because mlx-swift keeps it private: a resumed `SGD`
    ///     run continues exactly, while a resumed `Adam` run rebuilds its moment estimates and shows
    ///     a brief rise in loss.
    ///   - observer: receives each step and can end the run early.
    ///
    /// - Throws: `NFKMLXError.trainingDiverged` when a step's loss stops being finite, before that
    ///   step can reach a checkpoint.
    ///
    /// The module is put in training mode for the duration and restored afterward, so a model built
    /// by a factory that set evaluation mode for its batch-normalization statistics updates those
    /// statistics while training and returns ready to infer.
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
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: Observer? = nil
    ) throws -> [Float] {
        try run(model, optimizer: optimizer, steps: steps,
                arrays: { let (input, target) = batch($0); return [input, target] },
                loss: { model, arrays in loss(model, arrays[0], arrays[1]) },
                clipGradientNorm: clipGradientNorm, checkpoint: checkpoint, observer: observer)
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
    ///   - checkpoint: writes the model periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    @discardableResult
    public static func train<Model: Module>(
        _ model: Model,
        optimizer: Optimizer,
        steps: Int,
        sample: (Int) -> MLXArray,
        loss: @escaping (Model, MLXArray) -> MLXArray,
        clipGradientNorm: Float? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: Observer? = nil
    ) throws -> [Float] {
        try run(model, optimizer: optimizer, steps: steps,
                arrays: { [sample($0)] },
                loss: { model, arrays in loss(model, arrays[0]) },
                clipGradientNorm: clipGradientNorm, checkpoint: checkpoint, observer: observer)
    }

    /// The loop both entry points share, over an arbitrary number of per-step arrays.
    private static func run<Model: Module>(
        _ model: Model,
        optimizer: Optimizer,
        steps: Int,
        arrays: (Int) -> [MLXArray],
        loss: @escaping (Model, [MLXArray]) -> MLXArray,
        clipGradientNorm: Float?,
        checkpoint: NFKMLXTrainingCheckpoint?,
        observer: Observer?
    ) throws -> [Float] {
        let wasTraining = model.training
        model.train(true)
        defer { model.train(wasTraining) }

        let lossAndGradient = valueAndGrad(model: model) { model, arrays in [loss(model, arrays)] }
        var history: [Float] = []
        history.reserveCapacity(steps)

        for step in 0 ..< steps {
            let (values, gradients) = lossAndGradient(model, arrays(step))
            let update = clipGradientNorm.map { clipGradNorm(gradients: gradients, maxNorm: $0).0 } ?? gradients
            optimizer.update(model: model, gradients: update)
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
}
