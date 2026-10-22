//
//  NFKMLXZeroDCEPlusTraining.swift
//  InferKitMLX
//
//  Fine-tuning Zero-DCE++, the authors' lightweight successor to Zero-DCE.
//
//  The training script (Li-Chongyi/Zero-DCE_extension, `Zero-DCE++/lowlight_train.py`) keeps Zero-DCE's
//  four zero-reference losses and its optimizer, and changes two things: the smoothness term reads the
//  one shared three-channel curve map at a weight of 1600 rather than 200, and the estimator trains at a
//  scale factor of 1 although the released weights run at 12.
//  `NFKMLXReferenceParityTests.testZeroDCEPlusTrainingMatchesTheReference` measures the losses, the
//  gradients, and one optimizer step against that script at both factors.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The weighted combination of the four zero-reference losses that Zero-DCE++ trains against.
///
/// The defaults are `lowlight_train.py`'s. `wellExposedLevel` is the one a consumer-facing app exposes:
/// raising it produces a brighter model, lowering it a more restrained one.
///
/// - Since: InferKit 0.4.0
public struct NFKMLXZeroDCEPlusObjective: Sendable {

    /// The mean intensity every local region is driven toward.
    public var wellExposedLevel: Float

    /// Weight on the exposure loss.
    public var exposureWeight: Float

    /// Weight on the color-constancy loss.
    public var colorWeight: Float

    /// Weight on the illumination-smoothness loss over the shared curve map.
    public var smoothnessWeight: Float

    /// Weight on the spatial-consistency loss.
    public var spatialWeight: Float

    /// The factor the curve estimator downsamples by while training. The reference trains at 1; a
    /// factor above 1 needs a batch whose sides it divides.
    public var scaleFactor: Int

    public init(wellExposedLevel: Float = 0.6, exposureWeight: Float = 10, colorWeight: Float = 5,
                smoothnessWeight: Float = 1600, spatialWeight: Float = 1, scaleFactor: Int = 1) {
        self.wellExposedLevel = wellExposedLevel
        self.exposureWeight = exposureWeight
        self.colorWeight = colorWeight
        self.smoothnessWeight = smoothnessWeight
        self.spatialWeight = spatialWeight
        self.scaleFactor = scaleFactor
    }

    /// Scores an enhanced batch and the curve map that produced it against the unlabeled `photos`, all
    /// `[N, H, W, C]`.
    public func loss(enhanced: MLXArray, curve: MLXArray, photos: MLXArray) -> MLXArray {
        smoothnessWeight * NFKMLXZeroDCELoss.illuminationSmoothness(curve)
            + spatialWeight * NFKMLXZeroDCELoss.spatialConsistency(enhanced, original: photos)
            + colorWeight * NFKMLXZeroDCELoss.colorConstancy(enhanced)
            + exposureWeight * NFKMLXZeroDCELoss.exposure(enhanced, wellExposedLevel: wellExposedLevel)
    }

    /// Scores `net` on a batch of unlabeled photos `[N, H, W, 3]` in `0...1`.
    public func callAsFunction(_ net: NFKMLXZeroDCEPlusNet, _ photos: MLXArray) -> MLXArray {
        let (enhanced, curve) = net.enhancedWithCurve(photos, scaleFactor: scaleFactor)
        return loss(enhanced: enhanced, curve: curve, photos: photos)
    }
}

extension NFKMLXZeroDCEPlus {

    /// Fine-tunes a curve estimator on unlabeled photos, returning the loss from each step.
    ///
    /// The whole customization path is three calls: ``network(weightsURL:scaleFactor:)`` to build, this
    /// to train, and `NFKMLXWeights.save` to write a checkpoint that ``backend(weightsURL:)`` loads like
    /// any other. Every weight trains, as in the reference.
    ///
    /// - Parameters:
    ///   - net: the estimator to train, from ``network(weightsURL:scaleFactor:)``.
    ///   - photos: supplies an unlabeled batch `[N, H, W, 3]` in `0...1` for each step. The reference
    ///     resizes its photos to 512 × 512 and draws eight at a time.
    ///   - objective: what "well exposed" means for this consumer, and the factor the estimator trains at.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.Adam` (bias-corrected) at
    ///     its learning rate 1e-4 and weight decay 1e-4, which torch's Adam adds to the gradient.
    ///   - steps: how many batches to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference clips at 0.1.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates. 1, the
    ///     default, updates after every batch.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    ///
    /// - Since: InferKit 0.4.0
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXZeroDCEPlusNet,
        photos: (Int) -> MLXArray,
        objective: NFKMLXZeroDCEPlusObjective = NFKMLXZeroDCEPlusObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 0.1,
        accumulationSteps: Int = 1,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(net,
                               freezing: {},
                               optimizer: optimizer,
                               reference: { NFKMLXReferenceOptimizers.l2Adam(learningRate: 1e-4, weightDecay: 1e-4) },
                               referenceSchedule: { .constant },
                               steps: steps,
                               sample: photos, loss: objective.callAsFunction,
                               clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, checkpoint: checkpoint,
                               observer: observer)
    }
}
