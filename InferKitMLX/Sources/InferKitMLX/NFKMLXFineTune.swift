//
//  NFKMLXFineTune.swift
//  InferKitMLX
//
//  The sequence every fine-tune recipe runs, in one place.
//
//  A recipe is a public `fineTune` over `NFKMLXTrainer.train`, and the sixteen that ship do the same
//  four things in the same order before the loop starts: freeze what the caller's `trainable` excludes,
//  take the caller's optimizer or build the reference's, resolve the schedule against whether the
//  CALLER supplied an optimizer, and hand the rest to the trainer unchanged. Three of those four are
//  places a recipe has already been wrong. Ten recipes shipped against mlx-swift's uncorrected Adam
//  until the reference optimizers landed. A schedule resolved against the optimizer the recipe built
//  rather than the one the caller passed holds every run at a constant rate. Freezing after the
//  optimizer reads the parameters gives the optimizer state for parameters that never move.
//
//  What stays in the recipe is what differs per model: the example tuple, the objective's call shape,
//  the knobs a reference exposes (a learning rate, a warm-up length, a steps-per-epoch), and the
//  preconditions. Those are the parts a caller reads, so they stay in the model's own signature.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Runs the fine-tuning sequence a recipe wraps.
///
/// A recipe calls ``run(_:freezing:optimizer:reference:referenceSchedule:steps:arrays:loss:clipGradientNorm:learningRateSchedule:checkpoint:cachePolicy:observer:)``
/// in place of calling `NFKMLXTrainer.train` directly, so the ordering and the two resolution rules
/// are written once. The recipe keeps its own public signature.
///
/// Introduced in InferKit 0.5.0.
public enum NFKMLXFineTune {

    /// Freezes, resolves the optimizer and the schedule, and runs the training loop.
    ///
    /// - Parameters:
    ///   - net: the network to train.
    ///   - freezing: applies the recipe's freezing policy to `net`, including any LoRA adapters it
    ///     installs. It runs BEFORE the optimizer is built, so a frozen parameter carries no optimizer
    ///     state, and an error it throws ends the run before anything trains.
    ///   - optimizer: the optimizer the CALLER passed, nil where they passed none. It is the caller's
    ///     value, never the recipe's own reference optimizer, because it is what decides the schedule.
    ///   - reference: builds the optimizer the reference trains this model with. It is evaluated only
    ///     when `optimizer` is nil, so a recipe whose reference optimizer walks the parameter tree
    ///     does not pay for it when the caller supplied one.
    ///   - referenceSchedule: builds the schedule the reference runs over these steps. It is evaluated
    ///     only when it is the one that applies.
    ///   - steps: how many batches to train on.
    ///   - arrays: supplies the per-step tensors, which the loss reads positionally.
    ///   - loss: scores the network on one step's tensors.
    ///   - clipGradientNorm: bounds the global gradient norm. Pass the reference's value; nil is the
    ///     reference setting no clip.
    ///   - learningRateSchedule: the caller's schedule, nil to resolve one.
    ///   - checkpoint: writes the network periodically.
    ///   - cachePolicy: the buffer-cache policy for the run.
    ///   - observer: receives each step and can end the run early.
    ///
    /// - Returns: the loss from each completed step.
    ///
    /// - Throws: what `NFKMLXTrainer.train` throws, including `NFKMLXError.nothingToTrain` when
    ///   `freezing` leaves no trainable parameter, which is what a `trainable` case that froze
    ///   everything or a LoRA predicate that matched nothing leaves behind.
    ///
    /// @discussion **The schedule resolves against the caller's optimizer, not the recipe's.** A
    /// caller who passes an optimizer has chosen its learning rate, so the trainer runs it with no
    /// schedule and its rate never changes. A caller who passes none gets the reference's schedule
    /// over the reference's optimizer. An explicit `learningRateSchedule` applies to either. Passing
    /// `reference()` in place of `optimizer` here would make every run constant-rate, which is the
    /// error this parameter separation prevents.
    ///
    /// A run is multi-second; call it off the render thread.
    @discardableResult
    public static func run<Net: Module>(
        _ net: Net,
        freezing: () throws -> Void,
        optimizer: Optimizer?,
        reference: () -> Optimizer,
        referenceSchedule: () -> NFKMLXLearningRateSchedule,
        steps: Int,
        arrays: (Int) -> [MLXArray],
        loss: @escaping (Net, [MLXArray]) -> MLXArray,
        clipGradientNorm: Float?,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        cachePolicy: NFKMLXTrainingCachePolicy = .disabledOnGPU,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try freezing()
        // A caller's optimizer with no schedule runs untouched: the trainer then never asks it for a
        // single rate, which an optimizer such as Adafactor does not have.
        let schedule = learningRateSchedule ?? (optimizer == nil ? referenceSchedule() : nil)
        return try NFKMLXTrainer.train(
            net, optimizer: optimizer ?? reference(), steps: steps,
            arrays: arrays, loss: loss,
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: schedule,
            checkpoint: checkpoint, cachePolicy: cachePolicy, observer: observer)
    }

    /// The unlabeled form, for a recipe whose loss reads one array per step and no target, such as a
    /// reconstruction objective or a loss whose targets travel beside the batch.
    @discardableResult
    public static func run<Net: Module>(
        _ net: Net,
        freezing: () throws -> Void,
        optimizer: Optimizer?,
        reference: () -> Optimizer,
        referenceSchedule: () -> NFKMLXLearningRateSchedule,
        steps: Int,
        sample: (Int) -> MLXArray,
        loss: @escaping (Net, MLXArray) -> MLXArray,
        clipGradientNorm: Float?,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        cachePolicy: NFKMLXTrainingCachePolicy = .disabledOnGPU,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try run(net, freezing: freezing, optimizer: optimizer, reference: reference,
                referenceSchedule: referenceSchedule, steps: steps,
                arrays: { [sample($0)] },
                loss: { net, arrays in loss(net, arrays[0]) },
                clipGradientNorm: clipGradientNorm, learningRateSchedule: learningRateSchedule,
                checkpoint: checkpoint, cachePolicy: cachePolicy, observer: observer)
    }

    /// The supervised form, for a recipe whose loss reads one input and one target.
    @discardableResult
    public static func run<Net: Module>(
        _ net: Net,
        freezing: () throws -> Void,
        optimizer: Optimizer?,
        reference: () -> Optimizer,
        referenceSchedule: () -> NFKMLXLearningRateSchedule,
        steps: Int,
        batch: (Int) -> (input: MLXArray, target: MLXArray),
        loss: @escaping (Net, MLXArray, MLXArray) -> MLXArray,
        clipGradientNorm: Float?,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        cachePolicy: NFKMLXTrainingCachePolicy = .disabledOnGPU,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try run(net, freezing: freezing, optimizer: optimizer, reference: reference,
                referenceSchedule: referenceSchedule, steps: steps,
                arrays: { let example = batch($0); return [example.input, example.target] },
                loss: { net, arrays in loss(net, arrays[0], arrays[1]) },
                clipGradientNorm: clipGradientNorm, learningRateSchedule: learningRateSchedule,
                checkpoint: checkpoint, cachePolicy: cachePolicy, observer: observer)
    }
}
