//
//  NFKMLXHTDemucsTraining.swift
//  InferKitMLX
//
//  Training Hybrid Transformer Demucs on a consumer's own stems, as demucs v4's `solver.py` trains it:
//  every parameter, the stems' L1 distance through both branches, and the augmentations the solver
//  applies before the stems are summed into the mix. The two recipes are the ones the released
//  checkpoints record: the base training and the per-stem fine-tune that made `htdemucs_ft`.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

public extension NFKMLXDemucsObjective {

    /// Scores a Hybrid Transformer Demucs network on a batch of stems `[N, sources, channels, L]`: the
    /// stems sum into the mix, and the network's estimate at the mix's own length is compared with them.
    func callAsFunction(_ net: NFKMLXHTDemucsNet, _ stems: MLXArray) -> MLXArray {
        loss(estimate: net.estimate(stems.sum(axis: 1)), sources: stems)
    }
}

/// The settings a Hybrid Transformer Demucs run takes from its reference: the augmentations, the
/// optimizer, and the gradient clip.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXHTDemucsTrainingRecipe: Sendable {
    /// The augmentations each batch takes. Nil applies none.
    public var augmentation: NFKMLXDemucsAugmentation?
    /// The Adam learning rate.
    public var learningRate: Float
    /// The decoupled weight decay on the cross-transformer's parameters, the reference's
    /// `t_weight_decay`. Every other parameter decays by none.
    public var transformerWeightDecay: Float
    /// The bound on the global gradient norm, the reference's `clip_grad`. Nil clips none.
    public var clipGradientNorm: Float?
    /// The reference's batch, which a run reaches through `accumulationSteps` or a batch of that many.
    public var batchSize: Int
    /// The reference's training clip in seconds at 44.1 kHz, before the crop's `shift`.
    public var segmentSeconds: Int

    public init(augmentation: NFKMLXDemucsAugmentation?, learningRate: Float, transformerWeightDecay: Float,
                clipGradientNorm: Float?, batchSize: Int, segmentSeconds: Int) {
        self.augmentation = augmentation
        self.learningRate = learningRate
        self.transformerWeightDecay = transformerWeightDecay
        self.clipGradientNorm = clipGradientNorm
        self.batchSize = batchSize
        self.segmentSeconds = segmentSeconds
    }

    /// The `mmi` grid that trained `htdemucs`: Adam at 3e-4 with no decay and no clip, batches of 32
    /// ten-second clips, and every augmentation.
    public static let htdemucs = NFKMLXHTDemucsTrainingRecipe(
        augmentation: .htdemucs, learningRate: 3e-4, transformerWeightDecay: 0, clipGradientNorm: nil,
        batchSize: 32, segmentSeconds: 10)

    /// The `mmi_ft` grid that fine-tuned `htdemucs` into the four models of `htdemucs_ft`, each with
    /// ``NFKMLXDemucsObjective/stem(_:of:)``: AdamW at 1e-4 with 0.05 decay on the transformer, the
    /// gradient clipped at 5, batches of 8, and only the crop and the flips.
    public static let htdemucsFineTune = NFKMLXHTDemucsTrainingRecipe(
        augmentation: .htdemucsFineTune, learningRate: 1e-4, transformerWeightDecay: 0.05, clipGradientNorm: 5,
        batchSize: 8, segmentSeconds: 10)
}

public extension NFKMLXHTDemucs {

    /// Builds a Hybrid Transformer Demucs network for training or for reloading a trained checkpoint.
    /// With a `weightsURL` the checkpoint loads, a released one or one saved by `NFKMLXWeights`; without
    /// one the net is randomly initialized at `configuration`. Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?,
                        configuration: NFKMLXHTDemucsConfiguration = .htdemucs) throws -> NFKMLXHTDemucsNet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// The optimizer `train.py`'s `get_optimizer` builds for `recipe`: Adam, bias-corrected, at the
    /// recipe's rate, with the cross-transformer in its own group at the recipe's decoupled decay.
    /// Introduced in InferKit 0.4.0.
    static func referenceOptimizer(for net: NFKMLXHTDemucsNet,
                                   recipe: NFKMLXHTDemucsTrainingRecipe = .htdemucs) -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: recipe.learningRate, over: net) { key in
            (rateScale: 1, weightDecay: key.hasPrefix("crosstransformer.") ? recipe.transformerWeightDecay : 0)
        }
    }

    /// Trains a Hybrid Transformer Demucs network on separated stems.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:configuration:)``.
    ///   - stems: supplies one batch per step, `[N, sources, channels, L]` at 44.1 kHz in the network's
    ///     stem order, `L` being the clip plus the augmentation's crop `shift`.
    ///   - objective: the loss; L1 over equally weighted stems by default. The fine-tune recipe trains
    ///     one stem with ``NFKMLXDemucsObjective/stem(_:of:)``.
    ///   - recipe: the augmentations, optimizer, and clip; ``NFKMLXHTDemucsTrainingRecipe/htdemucs`` by
    ///     default.
    ///   - augmentationSeed: seeds the augmentations' draws, the reference's `seed` by default.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer(for:recipe:)``.
    ///   - steps: how many updates to train for.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is the reference's constant rate.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Every parameter trains, and the transformer drops activations at the configuration's
    /// `transformerDropout`. Save with `NFKMLXWeights.save`; ``network(weightsURL:configuration:)`` and
    /// `backend(weightsURL:)` read the file back. A run is minutes to hours; call it off the render thread.
    /// - Throws: `NFKMLXError.unsupportedConfiguration` when the remix's group size does not divide the batch.
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXHTDemucsNet,
        stems: (Int) -> MLXArray,
        objective: NFKMLXDemucsObjective = NFKMLXDemucsObjective(),
        recipe: NFKMLXHTDemucsTrainingRecipe = .htdemucs,
        augmentationSeed: UInt64 = 42,
        optimizer: Optimizer? = nil,
        steps: Int,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let batches = try NFKDemucsAugmentedBatches(stems: stems, augmentation: recipe.augmentation,
                                                    seed: augmentationSeed)
        return try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { referenceOptimizer(for: net, recipe: recipe) },
            referenceSchedule: { .constant },
            steps: steps,
            sample: { batches.next($0, from: stems) },
            loss: objective.callAsFunction,
            clipGradientNorm: recipe.clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
