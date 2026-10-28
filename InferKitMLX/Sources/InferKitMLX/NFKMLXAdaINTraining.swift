//
//  NFKMLXAdaINTraining.swift
//  InferKitMLX
//
//  Training AdaIN's decoder on a consumer's own content and style images, as naoto0804's `train.py`
//  trains it: the normalized VGG-19 encoder frozen, a content term between the decoded image's relu4_1
//  features and the normalized features it was decoded from, a style term between the per-channel means
//  and deviations of four of its activations and the style image's, and Adam at 1e-4 under an inverse
//  time decay. Each image is a random 256-pixel crop.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The loss `train.py` minimizes through `net.Net.forward`.
///
/// @discussion The content and style images pass through the frozen encoder; adaptive instance
/// normalization moves the content's relu4_1 features onto the style's statistics, and the decoder
/// inverts them. The decoded image passes through the encoder again. The content term is the mean
/// squared error between its relu4_1 features and the normalized features. The style term sums, over
/// relu1_1, relu2_1, relu3_1, and relu4_1, the mean squared error between the decoded image's
/// per-channel means and the style image's, plus the same between their deviations. A deviation is the
/// square root of the unbiased variance plus 1e-5, as `calc_mean_std` computes it.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXAdaINObjective: Sendable {
    /// The content term's factor, the reference's `--content_weight`.
    public var contentWeight: Float
    /// The style term's factor, the reference's `--style_weight`.
    public var styleWeight: Float

    public init(contentWeight: Float = 1, styleWeight: Float = 10) {
        self.contentWeight = contentWeight
        self.styleWeight = styleWeight
    }

    /// Scores the decoder on content images and style images, each `[N, H, W, 3]` in `0...1`.
    public func callAsFunction(_ net: NFKMLXAdaINNet, _ content: MLXArray, _ style: MLXArray) -> MLXArray {
        let terms = components(net, content: content, style: style)
        return terms.content * contentWeight + terms.style * styleWeight
    }

    /// The content and style terms before their factors, the two values `net.Net.forward` returns.
    public func components(_ net: NFKMLXAdaINNet, content: MLXArray, style: MLXArray)
        -> (content: MLXArray, style: MLXArray) {
        let styleFeatures = net.encoder.intermediates(stopGradient(style)).map { stopGradient($0) }
        let contentFeatures = net.encoder.features(stopGradient(content))
        let target = stopGradient(NFKMLXAdaINNet.adaptiveInstanceNormalization(content: contentFeatures,
                                                                               style: styleFeatures[3]))
        let decodedFeatures = net.encoder.intermediates(net.decoder(target))
        let contentTerm = square(decodedFeatures[3] - target).mean()
        var styleTerm = MLXArray(Float(0))
        for (decoded, reference) in zip(decodedFeatures, styleFeatures) {
            let decodedStatistics = NFKMLXAdaINNet.statistics(decoded)
            let referenceStatistics = NFKMLXAdaINNet.statistics(reference)
            styleTerm = styleTerm + square(decodedStatistics.mean - referenceStatistics.mean).mean()
                + square(decodedStatistics.deviation - referenceStatistics.deviation).mean()
        }
        return (contentTerm, styleTerm)
    }
}

/// The training transform's random crop, torchvision's `RandomCrop(256)` after `Resize((512, 512))`.
///
/// @discussion Each image takes its own crop, its top drawn before its left, the content batch before the
/// style batch. The resize is the caller's: `NFKMLXTrainingData.batch` takes images that already share a
/// size, and the reference squares every image to 512 × 512 whatever its aspect.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXAdaINAugmentation: Sendable {
    /// The side of the square crop.
    public var patchSize: Int

    public init(patchSize: Int = 256) {
        self.patchSize = patchSize
    }

    /// Crops content and style images `[N, H, W, 3]`, each at least `patchSize` on a side.
    func apply<Draws: NFKDenoiserDraws>(content: MLXArray, style: MLXArray, draws: inout Draws) -> (MLXArray, MLXArray) {
        func cropped(_ images: MLXArray) -> MLXArray {
            stacked((0 ..< images.dim(0)).map { index in
                let top = draws.integer(0, images.dim(1) - patchSize + 1)
                let left = draws.integer(0, images.dim(2) - patchSize + 1)
                return images[index, top ..< (top + patchSize), left ..< (left + patchSize), 0...]
            })
        }
        let contentCrops = cropped(content)
        return (contentCrops, cropped(style))
    }
}

public extension NFKMLXAdaIN {

    /// The optimizer `train.py` builds over the decoder: `torch.optim.Adam` at `--lr` 1e-4,
    /// bias-corrected, with no weight decay. Introduced in InferKit 0.4.0.
    static func referenceOptimizer() -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: 1e-4, weightDecay: 0)
    }

    /// `adjust_learning_rate` at `--lr_decay` 5e-5: the rate divided by `1 + 5e-5 · i` before update `i`.
    /// Introduced in InferKit 0.4.0.
    static var referenceSchedule: NFKMLXLearningRateSchedule { .inverseTime(decay: 5e-5) }

    /// Trains AdaIN's decoder on content and style images.
    ///
    /// - Parameters:
    ///   - net: the networks, from ``network(encoderURL:decoderURL:)`` with the released normalized VGG.
    ///   - examples: supplies one batch per step: content images and style images, each `[N, H, W, 3]` in
    ///     `0...1`. The reference draws eight of each, MS-COCO for content and WikiArt for style, every
    ///     image squared to 512 × 512, for 160,000 updates.
    ///   - objective: the content and style terms; the reference's factors by default.
    ///   - augmentation: the random crop each image takes; nil takes none.
    ///   - augmentationSeed: seeds the crops' draws. The reference seeds nothing.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer()``.
    ///   - steps: how many updates to train for.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is ``referenceSchedule`` with the reference
    ///     optimizer, and none with a caller's.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The decoder trains and the encoder stays frozen. Save `net.decoder` with `NFKMLXWeights.save`;
    /// ``network(encoderURL:decoderURL:)`` and ``backend(encoderURL:decoderURL:)`` read it back as the
    /// decoder. A run is minutes; call it off the render thread.
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXAdaINNet,
        examples: (Int) -> (content: MLXArray, style: MLXArray),
        objective: NFKMLXAdaINObjective = NFKMLXAdaINObjective(),
        augmentation: NFKMLXAdaINAugmentation? = NFKMLXAdaINAugmentation(),
        augmentationSeed: UInt64 = 0,
        optimizer: Optimizer? = nil,
        steps: Int,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        var draws = NFKDenoiserSeededDraws(seed: augmentationSeed)
        return try NFKMLXFineTune.run(
            net,
            freezing: {
                net.unfreeze()
                net.encoder.freeze()
            },
            optimizer: optimizer,
            reference: { referenceOptimizer() },
            referenceSchedule: { referenceSchedule },
            steps: steps,
            batch: { step in
                let example = examples(step)
                guard let augmentation else { return (example.content, example.style) }
                return augmentation.apply(content: example.content, style: example.style, draws: &draws)
            },
            loss: objective.callAsFunction,
            clipGradientNorm: nil, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
