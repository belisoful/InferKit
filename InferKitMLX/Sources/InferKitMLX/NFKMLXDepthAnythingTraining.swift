//
//  NFKMLXDepthAnythingTraining.swift
//  InferKitMLX
//
//  Depth Anything V2's metric fine-tune, ported from Depth-Anything-V2 at a561b84, `metric_depth/train.py`:
//  `SiLogLoss` over the valid depth range, AdamW with the encoder at a tenth of the head's rate, and the
//  per-iteration poly schedule.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The objective `metric_depth/train.py` minimizes: `SiLogLoss`, the scale-invariant log error over the
/// pixels whose depth is valid and within the metric range.
///
/// @discussion With `d = log(target) − log(prediction)` over the counted pixels, the loss is
/// `√(mean(d²) − λ·mean(d)²)`. A pixel counts where the caller's mask holds and its depth lies in
/// `minimumDepth...maximumDepth`, the reference's `--min-depth` and `--max-depth`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXDepthSiLogObjective: Sendable {
    /// The weight of the squared mean, `lambd`.
    public var lambda: Float = 0.5
    public var minimumDepth: Float = 0.001
    public var maximumDepth: Float

    public init(maximumDepth: Float = 20) {
        self.maximumDepth = maximumDepth
    }

    /// Scores the network on normalized images `[N, H, W, 3]` against depth `[N, H, W]` in meters and the
    /// caller's validity mask `[N, H, W]` (nonzero where the depth is measured).
    public func callAsFunction(_ net: NFKMLXDepthAnythingNet, _ images: MLXArray, _ depth: MLXArray,
                               _ valid: MLXArray) -> MLXArray {
        loss(prediction: net(images), depth: depth, valid: valid)
    }

    /// The loss for a predicted depth map `[N, H, W]`.
    public func loss(prediction: MLXArray, depth: MLXArray, valid: MLXArray) -> MLXArray {
        let counted = logicalAnd(valid .!= 0, logicalAnd(depth .>= minimumDepth, depth .<= maximumDepth))
        let one = MLXArray(Float(1))
        let difference = MLX.where(counted, log(MLX.where(counted, depth, one)) - log(MLX.where(counted, prediction, one)),
                                   MLXArray(Float(0)))
        let count = counted.asType(.float32).sum()
        let mean = difference.sum() / count
        return sqrt(square(difference).sum() / count - lambda * square(mean))
    }
}

public extension NFKMLXDepthAnything {

    /// `train.py`'s batch, 2 images a device. Introduced in InferKit 0.4.0.
    static let referenceBatchSize = 2

    /// `train.py`'s training size, 518 pixels square. Introduced in InferKit 0.4.0.
    static let referenceImageSize = 518

    /// Builds Depth Anything V2 for training, inference, or reloading a trained checkpoint.
    ///
    /// - Parameters:
    ///   - weightsURL: the released checkpoint or a file `NFKMLXWeights` saved. Nil leaves the network at
    ///     its random initialization.
    ///   - configuration: the size, and `maxDepth` for a metric network.
    ///   - encoderOnly: loads only the encoder (`pretrained.*`) and leaves the head at its initialization,
    ///     as `train.py --pretrained-from` starts a metric network from a relative release.
    ///
    /// Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?, configuration: NFKMLXDepthConfiguration = .small,
                        encoderOnly: Bool = false) throws -> NFKMLXDepthAnythingNet {
        let net = makeNet(configuration)
        guard let weightsURL else { return net }
        guard encoderOnly else {
            try loadWeights(into: net, from: weightsURL)
            return net
        }
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: weightsURL)
        let encoder = mapped(checkpoint.arrays, needsConvTranspose: checkpoint.needsConvTranspose)
            .compactMap { key, value in key.hasPrefix("pretrained.") ? (String(key.dropFirst("pretrained.".count)), value) : nil }
        try NFKMLXWeights.apply(encoder, to: net.pretrained)
        return net
    }

    /// The network's input for RGB images `[N, H, W, 3]` in [0, 1]: ImageNet-normalized, as the
    /// reference's `NormalizeImage` leaves it. Introduced in InferKit 0.4.0.
    static func trainingInput(_ images: MLXArray) -> MLXArray {
        (images - MLXArray([Float(0.485), 0.456, 0.406])) / MLXArray([Float(0.229), 0.224, 0.225])
    }

    /// The optimizer `train.py` builds: `torch.optim.AdamW` with betas 0.9 and 0.999 and a weight decay of
    /// 0.01 on every parameter, the encoder at 5e-6 and the head at ten times that. Introduced in
    /// InferKit 0.4.0.
    static func referenceOptimizer(for net: NFKMLXDepthAnythingNet) -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: 5e-6, over: net) { key in
            (key.hasPrefix("pretrained.") ? 1 : 10, 0.01)
        }
    }

    /// `train.py`'s schedule over a run of `steps`: the rate is reset after each update to
    /// `(1 − k / steps)^0.9` for the update `k` just taken, so update `k` runs at `(1 − (k − 1) / steps)^0.9`
    /// and the first at the base rate. Introduced in InferKit 0.4.0.
    static func referenceSchedule(steps: Int) -> NFKMLXLearningRateSchedule {
        NFKMLXLearningRateSchedule { step in
            step == 0 ? 1 : powf(max(0, 1 - Float(step - 1) / Float(max(steps, 1))), 0.9)
        }
    }

    /// Fine-tunes Depth Anything V2 to metric depth on images and their measured depth.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:configuration:encoderOnly:)``. The reference starts
    ///     from a relative release's encoder with a fresh metric head.
    ///   - examples: supplies one batch per step: RGB images `[N, S, S, 3]` in [0, 1] at the configuration's
    ///     input size, their depth `[N, S, S]` in meters, and a mask `[N, S, S]` nonzero where the depth is
    ///     measured. The reference resizes and crops each image to 518 pixels square
    ///     (``referenceImageSize``).
    ///   - objective: the loss. Its maximum depth is the network's `maxDepth`, the reference's
    ///     `--max-depth`.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer(for:)``.
    ///   - steps: how many updates to train for.
    ///   - mirrors: flips each batch left to right with probability one half, as `train.py` does to the
    ///     image, depth, and mask together.
    ///   - mirrorSeed: seeds the flips, so a run repeats.
    ///   - clipGradientNorm: bounds the global gradient norm. The reference clips nothing.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is ``referenceSchedule(steps:)`` with the
    ///     reference optimizer and a constant rate with a caller's.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Every parameter trains. Save with `NFKMLXWeights.save`; ``network(weightsURL:configuration:encoderOnly:)``
    /// reads the file back. A run is minutes; call it off the render thread. Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXDepthAnythingNet,
        examples: (Int) -> (images: MLXArray, depth: MLXArray, valid: MLXArray),
        objective: NFKMLXDepthSiLogObjective? = nil,
        optimizer: Optimizer? = nil,
        steps: Int,
        mirrors: Bool = false,
        mirrorSeed: UInt64 = 0,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let objective = objective ?? NFKMLXDepthSiLogObjective(maximumDepth: net.configuration.maxDepth ?? 20)
        return try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { referenceOptimizer(for: net) },
            referenceSchedule: { referenceSchedule(steps: steps) },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                let flip = mirrors && MLXRandom.uniform(low: 0, high: 1, [1], key: MLXRandom.key(mirrorSeed &+ UInt64(step)))
                    .item(Float.self) < 0.5
                func oriented(_ x: MLXArray) -> MLXArray { flip ? x[0..., 0..., .stride(by: -1)] : x }
                return [trainingInput(oriented(example.images)), oriented(example.depth), oriented(example.valid)]
            },
            loss: { net, arrays in objective(net, arrays[0], arrays[1], arrays[2]) },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
