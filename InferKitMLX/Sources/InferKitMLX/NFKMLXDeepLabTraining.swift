//
//  NFKMLXDeepLabTraining.swift
//  InferKitMLX
//
//  Retargeting DeepLabV3 to a consumer's own classes.
//
//  torchvision's `references/segmentation` (v0.23.0) trains the released model: the logits and the
//  auxiliary head's are each upsampled bilinearly to the label resolution and scored by cross-entropy
//  that ignores label 255, the auxiliary term at half weight, under SGD at 0.02 with momentum 0.9 and
//  weight decay 1e-4, the auxiliary head at ten times the rate, and a polynomial decay of power 0.9.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a DeepLab fine-tune updates.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXDeepLabTrainable: Sendable {

    /// The ASPP head, its 3×3 convolution, and the classifier, with the ResNet-50 backbone frozen.
    ///
    /// The auxiliary head reads the frozen backbone, so its loss reaches no other trained parameter,
    /// and this run leaves it out.
    case head

    /// Every parameter, supervised by both heads, as the reference trains.
    case everything
}

/// The supervised objective a DeepLab fine-tune minimizes: torchvision's `criterion`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXDeepLabObjective: Sendable {

    /// The auxiliary head's share of the loss. The reference's is 0.5.
    public var auxiliaryWeight: Float

    /// The label that marks a pixel left unscored. The reference's is 255.
    public var ignoredLabel: Int32

    public init(auxiliaryWeight: Float = 0.5, ignoredLabel: Int32 = 255) {
        self.auxiliaryWeight = auxiliaryWeight
        self.ignoredLabel = ignoredLabel
    }

    /// Scores class logits against labels.
    ///
    /// Each set of logits is upsampled bilinearly (`align_corners=False`) to the label resolution and
    /// scored by cross-entropy averaged over the pixels whose label is not ``ignoredLabel``; the
    /// auxiliary term joins at ``auxiliaryWeight``.
    ///
    /// - Parameters:
    ///   - logits: class scores `[N, h, w, classCount]` at the network's stride.
    ///   - auxiliaryLogits: the auxiliary head's scores at the same shape, or nil to score the main
    ///     head alone.
    ///   - labels: class indices `[N, H, W]` at full resolution.
    public func loss(logits: MLXArray, auxiliaryLogits: MLXArray?, labels: MLXArray) -> MLXArray {
        let main = crossEntropy(logits, labels)
        guard let auxiliaryLogits else {
            return main
        }
        return main + auxiliaryWeight * crossEntropy(auxiliaryLogits, labels)
    }

    private func crossEntropy(_ logits: MLXArray, _ labels: MLXArray) -> MLXArray {
        let full = NFKMLXResample.resizeBilinear(logits, height: labels.dim(1), width: labels.dim(2))
        let flat = labels.reshaped([-1]).asType(.int32)
        let scored = flat .!= ignoredLabel
        let perPixel = MLXNN.crossEntropy(logits: full.reshaped([-1, full.dim(3)]),
                                          targets: MLX.where(scored, flat, MLXArray(Int32(0))),
                                          reduction: .none)
        let weights = scored.asType(perPixel.dtype)
        return sum(perPixel * weights) / sum(weights)
    }
}

extension NFKMLXDeepLab {

    /// Builds the segmentation network itself, ready to fine-tune.
    ///
    /// - Parameters:
    ///   - weightsURL: the converted release, or a file `NFKMLXWeights.save` wrote. Nil leaves the
    ///     network at its random initialization.
    ///   - configuration: the geometry. Its `classCount` is the consumer's class set; when it differs
    ///     from the checkpoint's, both classifiers are left at their random initialization and
    ///     everything else loads, which is what retargeting a segmentation model means.
    ///
    /// - Since: InferKit 0.4.0
    public static func network(weightsURL: URL?,
                               configuration: NFKMLXDeepLabConfiguration = .base) throws -> NFKMLXDeepLabNet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL, retargeting: true)
        }
        return net
    }

    /// Fine-tunes a segmentation network on a consumer's own annotated images, returning the loss from
    /// each step.
    ///
    /// The whole path is three calls: ``network(weightsURL:configuration:)`` to build, this to train,
    /// and `NFKMLXWeights.save` to write a checkpoint that `backendWithWeightsURL:error:` loads.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:configuration:)``.
    ///   - examples: supplies one batch per step: images `[N, H, W, 3]` in `0...1` and class indices
    ///     `[N, H, W]`, from `NFKMLXTrainingData`, with 255 on pixels left unscored. `N` is at least
    ///     two: the ASPP pooling branch normalizes one pooled value per image, and a batch of one gives
    ///     its batch normalization nothing to normalize, which the reference refuses too.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - objective: torchvision's cross-entropy over both heads.
    ///   - optimizer: the update rule. Nil uses the reference's SGD: rate 0.02, momentum 0.9, weight
    ///     decay 1e-4 added to the gradient, and ten times the rate on the auxiliary head.
    ///   - steps: how many batches to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference does not
    ///     clip.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates. 1, the
    ///     default, updates after every batch. The reference's batches hold 32 images.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the reference's
    ///     `PolynomialLR`, `(1 − k / steps)^0.9`, when the reference optimizer runs. With a caller's
    ///     optimizer, nil holds that optimizer's rate constant.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXDeepLabNet,
        examples: (Int) -> (images: MLXArray, labels: MLXArray),
        trainable: NFKMLXDeepLabTrainable = .head,
        objective: NFKMLXDeepLabObjective = NFKMLXDeepLabObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let first = examples(0)
        guard first.images.ndim == 4, first.images.dim(0) >= 2 else {
            throw NFKMLXError.trainingDataMismatch(
                "DeepLab trains on batches of at least two images [N, H, W, 3]; got \(first.images.shape)")
        }
        return try NFKMLXFineTune.run(
            net,
            freezing: { apply(trainable, to: net) },
            optimizer: optimizer,
            reference: { referenceOptimizer(for: net) },
            referenceSchedule: { .poly(steps: steps, power: 0.9) },
            steps: steps,
            batch: { index in
                let example = index == 0 ? first : examples(index)
                return (NFKMLXDeepLabNet.normalized(example.images), example.labels)
            },
            loss: { net, image, labels in
                guard trainable == .everything else {
                    return objective.loss(logits: net.logits(image), auxiliaryLogits: nil, labels: labels)
                }
                let logits = net.trainingLogits(image)
                return objective.loss(logits: logits.main, auxiliaryLogits: logits.auxiliary, labels: labels)
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// `train.py`'s SGD: the backbone and the classifier at the base rate, the auxiliary head at ten
    /// times it, and weight decay on every parameter.
    static func referenceOptimizer(for net: NFKMLXDeepLabNet) -> Optimizer {
        NFKMLXReferenceOptimizers.sgd(learningRate: 0.02, momentum: 0.9, over: net) { key in
            (key.hasPrefix("auxiliary.") ? 10 : 1, 1e-4)
        }
    }

    private static func apply(_ trainable: NFKMLXDeepLabTrainable, to net: NFKMLXDeepLabNet) {
        net.unfreeze()
        if trainable == .head {
            net.backbone.freeze()
            net.auxiliary.freeze()
        }
    }
}
