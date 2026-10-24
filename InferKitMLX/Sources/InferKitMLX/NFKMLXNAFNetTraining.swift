//
//  NFKMLXNAFNetTraining.swift
//  InferKitMLX
//
//  Training NAFNet on a consumer's own degraded and clean image pairs, as megvii-research's basicsr
//  `ImageRestorationModel` trains it: every parameter, `PSNRLoss` between the restored and the clean
//  image, AdamW with the gradient clipped at 0.01, a cosine schedule, and the paired dataset's random
//  crop and flips. Each release's option file is one recipe.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The loss basicsr's `ImageRestorationModel` minimizes for NAFNet: `PSNRLoss`, the logarithm of each
/// image's mean squared error, scaled to decibels and averaged over the batch.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXNAFNetObjective: Sendable {
    /// The loss's factor, the reference's `loss_weight`.
    public var weight: Float
    /// Compares BT.601 luminance in place of RGB, the reference's `toY`; off in every release.
    public var luminance: Bool

    public init(weight: Float = 1, luminance: Bool = false) {
        self.weight = weight
        self.luminance = luminance
    }

    /// Scores the network's restoration of degraded images `[N, H, W, 3]` against the clean ones.
    public func callAsFunction(_ net: NFKMLXNAFNetNet, _ degraded: MLXArray, _ clean: MLXArray) -> MLXArray {
        loss(restored: net(degraded), target: clean)
    }

    /// `PSNRLoss` of restored images `[N, H, W, 3]` against their targets: `10 / ln 10` times the batch
    /// mean of `log(mse + 1e-8)`, each image's error averaged over its pixels and channels.
    public func loss(restored: MLXArray, target: MLXArray) -> MLXArray {
        var (prediction, reference) = (restored, target)
        if luminance {
            let coefficients = MLXArray([Float(65.481), 128.553, 24.966])
            prediction = ((prediction * coefficients).sum(axis: -1, keepDims: true) + 16) / 255
            reference = ((reference * coefficients).sum(axis: -1, keepDims: true) + 16) / 255
        }
        let meanSquared = square(prediction - reference).mean(axes: [1, 2, 3])
        return weight * Float(10 / log(10.0)) * log(meanSquared + 1e-8).mean()
    }
}

/// The paired dataset's training augmentations: a random `patchSize` crop taken at the same place in the
/// degraded and the clean image, then the same horizontal flip, vertical flip, and transpose on both.
///
/// @discussion Each example takes its own draws, in basicsr's order: the crop's top and left, then the
/// horizontal flip when `flip` is on, then the vertical flip and the transpose when `rotation` is on. An
/// image smaller than the patch is first padded on the bottom and right by mirroring with its edge
/// repeated, OpenCV's `BORDER_REFLECT`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXNAFNetAugmentation: Sendable {
    /// The side of the square training crop, the reference's `gt_size`.
    public var patchSize: Int
    /// Flips horizontally at random, the reference's `use_flip`.
    public var flip: Bool
    /// Flips vertically and transposes at random, the reference's `use_rot`.
    public var rotation: Bool

    public init(patchSize: Int = 256, flip: Bool = false, rotation: Bool = false) {
        self.patchSize = patchSize
        self.flip = flip
        self.rotation = rotation
    }

    /// The SIDD option files: the crop alone.
    public static let sidd = NFKMLXNAFNetAugmentation()

    /// The GoPro and REDS option files: the crop, the flips, and the transpose.
    public static let goPro = NFKMLXNAFNetAugmentation(flip: true, rotation: true)

    /// Augments degraded and clean images `[N, H, W, 3]`, returning both at `[N, patchSize, patchSize, 3]`.
    func apply<Draws: NFKDenoiserDraws>(degraded: MLXArray, clean: MLXArray, draws: inout Draws) -> (MLXArray, MLXArray) {
        var degradedPatches = [MLXArray](), cleanPatches = [MLXArray]()
        for index in 0 ..< degraded.dim(0) {
            var pair = [mirrored(degraded[index]), mirrored(clean[index])]
            let (height, width) = (pair[0].dim(0), pair[0].dim(1))
            let top = draws.integer(0, height - patchSize + 1)
            let left = draws.integer(0, width - patchSize + 1)
            pair = pair.map { $0[top ..< (top + patchSize), left ..< (left + patchSize), 0...] }
            let horizontal = flip && draws.unit() < 0.5
            let vertical = rotation && draws.unit() < 0.5
            let transposed = rotation && draws.unit() < 0.5
            let reversed = MLXArray(Array((0 ..< Int32(patchSize)).reversed()))
            pair = pair.map { image in
                var x = image
                if horizontal { x = take(x, reversed, axis: 1) }
                if vertical { x = take(x, reversed, axis: 0) }
                if transposed { x = x.transposed(1, 0, 2) }
                return x
            }
            degradedPatches.append(pair[0])
            cleanPatches.append(pair[1])
        }
        return (stacked(degradedPatches), stacked(cleanPatches))
    }

    /// `[H, W, 3]` padded on the bottom and right to at least `patchSize` by `BORDER_REFLECT`.
    private func mirrored(_ image: MLXArray) -> MLXArray {
        func extended(_ x: MLXArray, axis: Int) -> MLXArray {
            let length = x.dim(axis)
            guard length < patchSize else { return x }
            let indices = (0 ..< patchSize).map { position -> Int32 in
                var p = position
                while p >= length { p = 2 * length - 1 - p }
                return Int32(max(p, 0))
            }
            return take(x, MLXArray(indices), axis: axis)
        }
        return extended(extended(image, axis: 0), axis: 1)
    }
}

/// The settings a NAFNet run takes from a release's option file: the augmentations, AdamW's rate and
/// decay, the gradient clip, and the cosine schedule.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXNAFNetTrainingRecipe: Sendable {
    /// The augmentations each batch takes. Nil applies none.
    public var augmentation: NFKMLXNAFNetAugmentation?
    /// AdamW's starting rate.
    public var learningRate: Float
    /// AdamW's decoupled weight decay, applied to every parameter.
    public var weightDecay: Float
    /// The bound on the global gradient norm; basicsr clips at 0.01 unless an option file turns it off.
    public var clipGradientNorm: Float?
    /// The cosine schedule's length in updates, the reference's `T_max`.
    public var cosineSteps: Int
    /// The rate the cosine schedule ends at, the reference's `eta_min`.
    public var finalLearningRate: Float
    /// The reference's batch, 8 GPUs at `batch_size_per_gpu`, which a run reaches through
    /// `accumulationSteps` or a batch of that many.
    public var batchSize: Int

    public init(augmentation: NFKMLXNAFNetAugmentation?, learningRate: Float = 1e-3, weightDecay: Float,
                clipGradientNorm: Float? = 0.01, cosineSteps: Int, finalLearningRate: Float = 1e-7, batchSize: Int) {
        self.augmentation = augmentation
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.clipGradientNorm = clipGradientNorm
        self.cosineSteps = cosineSteps
        self.finalLearningRate = finalLearningRate
        self.batchSize = batchSize
    }

    /// `options/train/SIDD/NAFNet-width32.yml`.
    public static let sidd = NFKMLXNAFNetTrainingRecipe(augmentation: .sidd, weightDecay: 0, cosineSteps: 200_000,
                                                        batchSize: 32)

    /// `options/train/SIDD/NAFNet-width64.yml`.
    public static let siddWidth64 = NFKMLXNAFNetTrainingRecipe(augmentation: .sidd, weightDecay: 0,
                                                               cosineSteps: 400_000, batchSize: 64)

    /// `options/train/GoPro/NAFNet-width32.yml`.
    public static let goPro = NFKMLXNAFNetTrainingRecipe(augmentation: .goPro, weightDecay: 1e-3, cosineSteps: 200_000,
                                                         batchSize: 32)

    /// `options/train/GoPro/NAFNet-width64.yml`.
    public static let goProWidth64 = NFKMLXNAFNetTrainingRecipe(augmentation: .goPro, weightDecay: 1e-3,
                                                                cosineSteps: 400_000, batchSize: 64)

    /// `options/train/REDS/NAFNet-width64.yml`, the GoPro width-64 settings on REDS.
    public static let reds = goProWidth64

    /// The cosine schedule over ``cosineSteps``, as `torch.optim.lr_scheduler.CosineAnnealingLR` steps it.
    public var schedule: NFKMLXLearningRateSchedule {
        .cosine(steps: cosineSteps, endScale: finalLearningRate / learningRate)
    }
}

public extension NFKMLXNAFNet {

    /// The optimizer the option files name: `torch.optim.AdamW` at the recipe's rate and decay, betas 0.9
    /// and 0.9. Introduced in InferKit 0.4.0.
    static func referenceOptimizer(recipe: NFKMLXNAFNetTrainingRecipe = .sidd) -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: recipe.learningRate, betas: (0.9, 0.9),
                                        weightDecay: recipe.weightDecay)
    }

    /// Trains NAFNet on pairs of degraded and clean images.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:configuration:)``.
    ///   - examples: supplies one batch per step: degraded images and the clean images under them, each
    ///     `[N, H, W, 3]` in `0...1`.
    ///   - objective: the loss; `PSNRLoss` by default.
    ///   - recipe: the augmentations, optimizer, clip, and schedule;
    ///     ``NFKMLXNAFNetTrainingRecipe/sidd`` by default. Pick the one for the release being tuned.
    ///   - augmentationSeed: seeds the augmentations' draws, the SIDD option files' `manual_seed`.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer(recipe:)``.
    ///   - steps: how many updates to train for.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is the recipe's cosine schedule with the
    ///     reference optimizer, and none with a caller's.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Every parameter trains. Save with `NFKMLXWeights.save`; ``network(weightsURL:configuration:)`` and
    /// `backend(variant:weightsURL:)` read the file back. A run is minutes; call it off the render thread.
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXNAFNetNet,
        examples: (Int) -> (degraded: MLXArray, clean: MLXArray),
        objective: NFKMLXNAFNetObjective = NFKMLXNAFNetObjective(),
        recipe: NFKMLXNAFNetTrainingRecipe = .sidd,
        augmentationSeed: UInt64 = 10,
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
            freezing: {},
            optimizer: optimizer,
            reference: { referenceOptimizer(recipe: recipe) },
            referenceSchedule: { recipe.schedule },
            steps: steps,
            batch: { step in
                let example = examples(step)
                guard let augmentation = recipe.augmentation else {
                    return (example.degraded, example.clean)
                }
                return augmentation.apply(degraded: example.degraded, clean: example.clean, draws: &draws)
            },
            loss: objective.callAsFunction,
            clipGradientNorm: recipe.clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
