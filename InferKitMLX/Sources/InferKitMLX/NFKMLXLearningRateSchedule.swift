//
//  NFKMLXLearningRateSchedule.swift
//  InferKitMLX
//
//  The learning-rate schedules the fine-tuning recipes' reference configurations name. Each is a
//  multiplier on the optimizer's base rate for a zero-based step, which `NFKMLXTrainer` applies to
//  every parameter group before that step's update, so the groups keep their ratios.
//

import Foundation
import MLXOptimizers

/// A learning-rate schedule: a multiplier on the optimizer's base rate for each zero-based step.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXLearningRateSchedule {
    /// The multiplier for a zero-based step.
    public let multiplier: (Int) -> Float

    public init(_ multiplier: @escaping (Int) -> Float) {
        self.multiplier = multiplier
    }

    /// The base rate at every step.
    public static var constant: NFKMLXLearningRateSchedule { NFKMLXLearningRateSchedule { _ in 1 } }

    /// fvcore's `CosineParamScheduler` from 1 to `endScale` over a run of `steps`, as SAM 2's trainer
    /// drives it: step `k` sits at `k / steps` of the run.
    public static func cosine(steps: Int, endScale: Float) -> NFKMLXLearningRateSchedule {
        NFKMLXLearningRateSchedule { step in
            let progress = Float(step) / Float(max(steps, 1))
            return endScale + 0.5 * (1 - endScale) * (1 + cos(Float.pi * progress))
        }
    }

    /// mmcv's `poly` policy with its linear warm-up, by iteration: `(1 − k / steps)^power`, scaled
    /// during the first `warmupSteps` by `1 − (1 − k / warmupSteps)(1 − warmupRatio)`.
    public static func poly(steps: Int, power: Float = 1, warmupSteps: Int = 0,
                            warmupRatio: Float = 1e-6) -> NFKMLXLearningRateSchedule {
        NFKMLXLearningRateSchedule { step in
            let regular = pow(max(0, 1 - Float(step) / Float(max(steps, 1))), power)
            guard step < warmupSteps else { return regular }
            return regular * (1 - (1 - Float(step) / Float(warmupSteps)) * (1 - warmupRatio))
        }
    }

    /// SAM 3's `InverseSquareRootParamScheduler`: a linear warm-up over `warmupSteps`, decay as
    /// `1 / √((k + timescale − warmupSteps) / timescale)` after it, and a linear cool-down over the
    /// last `cooldownSteps` of a run of `steps`.
    public static func inverseSquareRoot(steps: Int, timescale: Int, warmupSteps: Int,
                                         cooldownSteps: Int) -> NFKMLXLearningRateSchedule {
        NFKMLXLearningRateSchedule { step in
            // The reference derives the run's length as step / where, which is 1 at step zero.
            let total = step == 0 ? 1 : Float(steps)
            var scale: Float = 1
            if warmupSteps < step {
                scale /= Float(Double(step + timescale - warmupSteps) / Double(timescale)).squareRoot()
            }
            if warmupSteps > 0 {
                scale *= min(1, Float(step) / Float(warmupSteps))
            }
            if cooldownSteps > 0 {
                scale *= min(1, (total - Float(step)) / Float(cooldownSteps))
            }
            return scale
        }
    }

    /// V-JEPA 2's `WarmupCosineLRSchedule`, which steps before each update, so update `k` runs at step
    /// `k + 1`: a linear ramp from `startScale` over `warmupSteps`, then a cosine to `endScale` over the
    /// remaining steps.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func warmupCosine(steps: Int, warmupSteps: Int = 0, startScale: Float = 1,
                                    endScale: Float = 0) -> NFKMLXLearningRateSchedule {
        NFKMLXLearningRateSchedule { update in
            let step = Float(update + 1)
            guard step >= Float(warmupSteps) else {
                return startScale + step / Float(max(1, warmupSteps)) * (1 - startScale)
            }
            let progress = (step - Float(warmupSteps)) / Float(max(1, steps - warmupSteps))
            return max(endScale, endScale + (1 - endScale) * 0.5 * (1 + cos(Float.pi * progress)))
        }
    }

    /// fairseq's `inverse_sqrt`: a linear warm-up from `initialScale` over `warmupSteps` updates, then
    /// `√(warmupSteps / k)`. The trainer sets the rate for update count 0 before the first update, so
    /// update `k` runs at count `k`.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func fairseqInverseSquareRoot(warmupSteps: Int, initialScale: Float = 0) -> NFKMLXLearningRateSchedule {
        NFKMLXLearningRateSchedule { update in
            guard update >= warmupSteps else {
                return initialScale + Float(update) * (1 - initialScale) / Float(warmupSteps)
            }
            return update == 0 ? 1 : (Float(warmupSteps) / Float(update)).squareRoot()
        }
    }

    /// cosmos-predict1's `WarmupLambdaLR`: `(k + 1) / warmupSteps` until it reaches 1.
    public static func linearWarmup(steps warmupSteps: Int) -> NFKMLXLearningRateSchedule {
        NFKMLXLearningRateSchedule { step in min(1, Float(step + 1) / Float(max(warmupSteps, 1))) }
    }

    /// The schedule a recipe runs: the caller's when given, the reference's when the recipe builds the
    /// reference optimizer, and a constant rate when the caller chose the optimizer and so the rate.
    static func resolved(_ given: NFKMLXLearningRateSchedule?, optimizer: Optimizer?,
                         reference: () -> NFKMLXLearningRateSchedule) -> NFKMLXLearningRateSchedule {
        given ?? (optimizer == nil ? reference() : .constant)
    }
}

/// An optimizer whose rate a schedule can set.
protocol NFKMLXRateScheduled: AnyObject {
    var learningRate: Float { get set }
}

extension SGD: NFKMLXRateScheduled {}
extension RMSprop: NFKMLXRateScheduled {}
extension AdaGrad: NFKMLXRateScheduled {}
extension AdaDelta: NFKMLXRateScheduled {}
extension Adam: NFKMLXRateScheduled {}
extension Adamax: NFKMLXRateScheduled {}
extension Lion: NFKMLXRateScheduled {}
extension Muon: NFKMLXRateScheduled {}

extension NFKMLXLearningRateSchedule {

    /// The rate-bearing optimizers under `optimizer`, each with its base rate. Nil when one of them
    /// has no single rate to scale (Adafactor's is relative by default).
    static func scheduledGroups(of optimizer: Optimizer) -> [(NFKMLXRateScheduled, Float)]? {
        if let multi = optimizer as? MultiOptimizer {
            var groups = [(NFKMLXRateScheduled, Float)]()
            for member in multi.optimizers {
                guard let inner = scheduledGroups(of: member) else { return nil }
                groups += inner
            }
            return groups
        }
        guard let scheduled = optimizer as? NFKMLXRateScheduled else { return nil }
        return [(scheduled, scheduled.learningRate)]
    }
}
