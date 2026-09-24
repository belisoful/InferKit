//
//  NFKMLXConvTasNetTraining.swift
//  InferKitMLX
//
//  Fine-tuning Conv-TasNet on a consumer's own mixtures: their speakers, their room, their overlap. The
//  network is 5M parameters, so every weight trains.
//
//  The recipe is asteroid's `egs/librimix/ConvTasNet` at v0.5.2, the one the release's model card names:
//  `PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")` (the negative scale-invariant SDR of every
//  estimate against every source, zero-mean, then the speaker assignment with the lowest mean),
//  `torch.optim.Adam` at 1e-3 with no weight decay, and gradient clipping at norm 5. The reference halves
//  the rate after five epochs without validation improvement and stops early after thirty; both need a
//  validation set, so the recipe holds the rate and runs the steps it is given. The network has no
//  dropout and no batch normalization, so training and inference compute the same function.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/*!
 @abstract The objective a Conv-TasNet fine-tune minimizes: asteroid's permutation-invariant negative SI-SDR.
 @discussion For estimates and sources `[speakers, samples]`, each zero-meaned, the pairwise loss is
 `−10 log₁₀(‖proj‖² / (‖ŝ − proj‖² + ε) + ε)` with `proj = ⟨ŝ, s⟩ s / (‖s‖² + ε)` and ε = 1e-8, and the loss
 is the mean over speakers under the assignment of estimates to sources that makes it smallest. Measured
 against asteroid's own `PITLossWrapper` by `run_reference.py convtasnet_loss`. Introduced in InferKit 0.5.0.
 */
public struct NFKMLXConvTasNetObjective: Sendable {

    public init() {}

    /// `[estimates, sources]` of negative SI-SDR, in dB.
    public func pairwise(estimates: MLXArray, sources: MLXArray) -> MLXArray {
        let epsilon: Float = 1e-8
        let target = (sources - sources.mean(axis: -1, keepDims: true)).expandedDimensions(axis: 0)      // [1, S, T]
        let estimate = (estimates - estimates.mean(axis: -1, keepDims: true)).expandedDimensions(axis: 1)  // [E, 1, T]
        let dot = (estimate * target).sum(axis: -1, keepDims: true)
        let energy = target.square().sum(axis: -1, keepDims: true) + epsilon
        let projection = dot * target / energy
        let noise = estimate - projection
        let ratio = projection.square().sum(axis: -1) / (noise.square().sum(axis: -1) + epsilon)
        return -10 * log10(ratio + epsilon)
    }

    /// The loss, a scalar: the smallest mean pairwise loss over the assignments of estimates to sources.
    public func loss(estimates: MLXArray, sources: MLXArray) -> MLXArray {
        let matrix = pairwise(estimates: estimates, sources: sources)
        let speakers = sources.dim(0)
        let candidates = Self.permutations(speakers).map { permutation in
            permutation.enumerated().map { source, estimate in matrix[estimate, source] }
                .reduce(MLXArray(Float(0)), +) / Float(speakers)
        }
        return stacked(candidates).min()
    }

    static func permutations(_ count: Int) -> [[Int]] {
        guard count > 1 else {
            return [[Int](0 ..< count)]
        }
        return permutations(count - 1).flatMap { shorter in
            (0 ... shorter.count).map { position in
                var longer = shorter
                longer.insert(count - 1, at: position)
                return longer
            }
        }
    }
}

extension NFKMLXConvTasNet {

    /// Builds the separation network itself, ready to fine-tune, at the geometry the weights were trained
    /// at: the released checkpoint's own, a file `NFKMLXWeights.save` wrote, or `.libri2Mix16k` without
    /// weights.
    ///
    /// - Since: InferKit 0.5.0
    public static func network(weightsURL: URL?) throws -> NFKMLXConvTasNetNet {
        let configuration = try weightsURL.map(configuration(matching:)) ?? .libri2Mix16k
        let net = NFKMLXConvTasNetNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Fine-tunes every weight of `net` on mixtures and their separate sources, returning the loss from
    /// each step.
    ///
    /// The whole path is three calls: ``network(weightsURL:)`` to build, this to train, and
    /// `NFKMLXWeights.save` to write a checkpoint that `backendWithWeightsURL:error:` loads.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:)``.
    ///   - examples: supplies one mixture per step with each speaker's own signal, all at the model's
    ///     rate and of one length. The reference trains on 3-second segments.
    ///   - objective: asteroid's permutation-invariant negative SI-SDR.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.Adam` (bias-corrected) at 1e-3.
    ///   - steps: how many mixtures to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference clips at 5.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil holds it constant.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is minutes; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXConvTasNetNet,
        examples: (Int) -> (mixture: [Float], sources: [[Float]]),
        objective: NFKMLXConvTasNetObjective = NFKMLXConvTasNetObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 5,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { Adam(learningRate: 1e-3, biasCorrection: true) },
            referenceSchedule: { .constant },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                let length = ([example.mixture.count] + example.sources.map(\.count)).min() ?? 0
                let sources = example.sources.map { Array($0.prefix(length)) }
                return [MLXArray(Array(example.mixture.prefix(length))),
                        MLXArray(sources.flatMap { $0 }).reshaped([sources.count, length])]
            },
            loss: { net, arrays in
                objective.loss(estimates: net.separate(arrays[0]), sources: arrays[1])
            },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }
}
