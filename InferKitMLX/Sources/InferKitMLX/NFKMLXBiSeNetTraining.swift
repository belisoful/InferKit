//
//  NFKMLXBiSeNetTraining.swift
//  InferKitMLX
//
//  Retargeting BiSeNet V1 to a consumer's own classes.
//
//  CoinCheung/BiSeNet (6b4b67a) trains the released model with `tools/train_amp.py` and
//  `configs/bisenetv1_city.py`: the main head and both auxiliary heads are each scored by an OHEM
//  cross-entropy at threshold 0.7 that ignores label 255, and the three losses add. SGD at 0.01 with
//  momentum 0.9 decays the convolution weights by 5e-4 and nothing else, trains the fusion module and
//  the three heads at ten times the rate, and follows a 1,000-step exponential warm-up into a
//  polynomial decay of power 0.9.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a BiSeNet fine-tune updates.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXBiSeNetTrainable: Sendable {

    /// Every parameter except the context path's ResNet-18, the one part the reference initializes
    /// from ImageNet rather than training from scratch.
    case allButBackbone

    /// Every parameter, as the reference trains.
    case everything
}

/// The supervised objective a BiSeNet fine-tune minimizes: the reference's `OhemCELoss` on each head.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXBiSeNetObjective: Sendable {

    /// The probability below which a pixel counts as hard. The reference's is 0.7.
    public var threshold: Float

    /// The label that marks a pixel left unscored. The reference's is 255.
    public var ignoredLabel: Int32

    public init(threshold: Float = 0.7, ignoredLabel: Int32 = 255) {
        self.threshold = threshold
        self.ignoredLabel = ignoredLabel
    }

    /// The main head's OHEM loss plus each auxiliary head's, unweighted.
    ///
    /// - Parameters:
    ///   - logits: the main head's class scores `[N, H, W, classCount]` at the label resolution.
    ///   - auxiliaryLogits: the auxiliary heads' scores at the same shape.
    ///   - labels: class indices `[N, H, W]`.
    public func loss(logits: MLXArray, auxiliaryLogits: [MLXArray], labels: MLXArray) -> MLXArray {
        auxiliaryLogits.reduce(ohem(logits, labels)) { $0 + ohem($1, labels) }
    }

    /// One head's OHEM cross-entropy.
    ///
    /// Each pixel's cross-entropy (zero where the label is ignored) is hard when it exceeds
    /// `−log(threshold)`. The loss is the mean over the hard pixels, or over the `n` largest when fewer
    /// than `n` are hard, with `n` a sixteenth of the scored pixels, rounded down. The pixels of every
    /// image in the batch rank together.
    public func ohem(_ logits: MLXArray, _ labels: MLXArray) -> MLXArray {
        let classCount = logits.dim(-1)
        let flat = labels.reshaped([-1]).asType(.int32)
        let scored = flat .!= ignoredLabel
        let perPixel = MLXNN.crossEntropy(logits: logits.reshaped([-1, classCount]),
                                          targets: MLX.where(scored, flat, MLXArray(Int32(0))),
                                          reduction: .none) * scored.asType(logits.dtype)
        let minimum = scored.asType(.int32).sum().item(Int.self) / 16
        let hard = perPixel .> -log(threshold)
        guard hard.asType(.int32).sum().item(Int.self) < minimum else {
            let weights = hard.asType(perPixel.dtype)
            return sum(perPixel * weights) / sum(weights)
        }
        let largest = argSort(-stopGradient(perPixel))[0 ..< minimum]
        return mean(perPixel[largest])
    }
}

extension NFKMLXBiSeNet {

    /// Builds the segmentation network itself, ready to fine-tune.
    ///
    /// - Parameters:
    ///   - weightsURL: the converted release, or a file `NFKMLXWeights.save` wrote. Nil leaves the
    ///     network at its random initialization.
    ///   - configuration: the geometry. Its `classCount` is the consumer's class set; when it differs
    ///     from the checkpoint's, the three classifiers are left at their random initialization and
    ///     everything else loads, which is what retargeting a segmentation model means.
    ///
    /// - Since: InferKit 0.4.0
    public static func network(weightsURL: URL?,
                               configuration: NFKMLXBiSeNetConfiguration = .base) throws -> NFKMLXBiSeNetNet {
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
    ///     two: the attention refinement, global-context, and fusion modules each normalize one pooled
    ///     value per image, and a batch of one gives their batch normalizations nothing to normalize,
    ///     which the reference refuses too. The context path needs both sides to be multiples of 32,
    ///     so a larger batch is cropped from its top-left corner to the nearest multiple below.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - objective: the reference's OHEM cross-entropy on all three heads.
    ///   - optimizer: the update rule. Nil uses the reference's SGD: rate 0.01 and momentum 0.9, weight
    ///     decay 5e-4 added to the gradient of every convolution weight and of nothing else, and ten
    ///     times the rate on the fusion module and the heads.
    ///   - steps: how many batches to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference does not
    ///     clip.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates. 1, the
    ///     default, updates after every batch. The reference trains on two GPUs with batches of 8,
    ///     each ranking its own pixels for the OHEM loss.
    ///   - precision: the precision the passes compute in; float32 by default. The reference trains
    ///     under float16 autocast with a gradient scaler, which `.float16` approximates.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the reference's
    ///     `WarmupPolyLrScheduler` when the reference optimizer runs: an exponential warm-up from a
    ///     tenth of the rate over 1,000 steps, then `(1 − progress)^0.9` to the end of the run. With a
    ///     caller's optimizer, nil holds that optimizer's rate constant.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXBiSeNetNet,
        examples: (Int) -> (images: MLXArray, labels: MLXArray),
        trainable: NFKMLXBiSeNetTrainable = .allButBackbone,
        objective: NFKMLXBiSeNetObjective = NFKMLXBiSeNetObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let first = examples(0)
        guard first.images.ndim == 4, first.images.dim(0) >= 2 else {
            throw NFKMLXError.trainingDataMismatch(
                "BiSeNet trains on batches of at least two images [N, H, W, 3]; got \(first.images.shape)")
        }
        return try NFKMLXFineTune.run(
            net,
            freezing: { apply(trainable, to: net) },
            optimizer: optimizer,
            reference: { referenceOptimizer(for: net) },
            referenceSchedule: { .exponentialWarmupPoly(steps: steps, power: 0.9, warmupSteps: 1000, warmupRatio: 0.1) },
            steps: steps,
            batch: { index in
                let example = index == 0 ? first : examples(index)
                let height = example.images.dim(1) / 32 * 32, width = example.images.dim(2) / 32 * 32
                return (NFKMLXBiSeNetNet.normalized(example.images[0..., 0 ..< height, 0 ..< width]),
                        example.labels[0..., 0 ..< height, 0 ..< width])
            },
            loss: { net, image, labels in
                let logits = net.trainingLogits(image)
                return objective.loss(logits: logits.main, auxiliaryLogits: logits.auxiliary, labels: labels)
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// The reference's four groups: `get_params` gives the fusion module and the three heads ten times
    /// the rate, and `set_optimizer` decays the weights of convolutions alone, every one with more than
    /// one dimension.
    static func referenceOptimizer(for net: NFKMLXBiSeNetNet) -> Optimizer {
        let decayed = Set(net.trainableParameters().flattened().filter { $0.1.ndim > 1 }.map(\.0))
        return NFKMLXReferenceOptimizers.sgd(learningRate: 0.01, momentum: 0.9, over: net) { key in
            let boosted = key.hasPrefix("ffm.") || key.hasPrefix("conv_out")
            return (boosted ? 10 : 1, decayed.contains(key) ? 5e-4 : 0)
        }
    }

    private static func apply(_ trainable: NFKMLXBiSeNetTrainable, to net: NFKMLXBiSeNetNet) {
        net.unfreeze()
        if trainable == .allButBackbone {
            net.cp.resnet.freeze()
        }
    }
}
