//
//  NFKMLXSiggraphColorizerTraining.swift
//  InferKitMLX
//
//  The SIGGRAPH-17 colorizer's fine-tune, ported from richzhang/colorization-pytorch at 66a1cb2: the data
//  path of `util.get_colorization_data` (its CIELAB conversion, its grayscale filter, and its random color
//  hints) and the regression phase of `pix2pix_model.py`, which `scripts/train_siggraph.sh` runs last.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The objective the SIGGRAPH-17 colorizer's regression phase minimizes: ten times the per-pixel L1 of the
/// ab regression, summed over a and b and averaged over the pixels.
///
/// @discussion The reference adds a 529-class cross-entropy on `model_class`, which reads the trunk through a
/// `detach` in this phase, so it trains only that head. Inference never reads `model_class` and the port does
/// not carry it; every other parameter's gradient is this objective's.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXSiggraphColorizerObjective: Sendable {
    /// The regression term's weight, the reference's 10.
    public var regressionWeight: Float = 10

    public init() {}

    /// Scores the network on its input `[N, H, W, 4]` against the true ab over 110, `[N, H, W, 2]`, the pair
    /// `NFKMLXSiggraphColorizer.trainingExample(_:hintProbability:seed:)` returns.
    public func callAsFunction(_ net: NFKMLXSiggraphNet, _ input: MLXArray, _ target: MLXArray) -> MLXArray {
        loss(prediction: net(input), target: target)
    }

    /// The loss for a predicted ab `[N, H, W, 2]` over 110.
    public func loss(prediction: MLXArray, target: MLXArray) -> MLXArray {
        regressionWeight * abs(prediction - target).sum(axis: -1).mean()
    }
}

/// The random draws the reference's hint sampler takes from numpy, in its order.
protocol NFKSiggraphHintDraws {
    /// `np.random.rand()`.
    mutating func unit() -> Double
    /// `np.random.choice(sizes)`.
    mutating func choice(_ sizes: [Int]) -> Int
    /// `np.random.normal(mean, deviation)`.
    mutating func normal(mean: Double, deviation: Double) -> Double
}

/// SplitMix64 draws from a seed, so a run repeats.
struct NFKSiggraphSeededDraws: NFKSiggraphHintDraws {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    private mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func unit() -> Double {
        Double(next() >> 11) / Double(UInt64(1) << 53)
    }

    mutating func choice(_ sizes: [Int]) -> Int {
        sizes[Int(next() % UInt64(sizes.count))]
    }

    mutating func normal(mean: Double, deviation: Double) -> Double {
        let radius = (-2 * log(max(unit(), .leastNormalMagnitude))).squareRoot()
        return mean + deviation * radius * cos(2 * .pi * unit())
    }
}

public extension NFKMLXSiggraphColorizer {

    /// The regression phases' hint rate, `--sample_p .125`: an image keeps receiving hints while a uniform
    /// draw lands under 1 − p. Introduced in InferKit 0.4.0.
    static let referenceHintProbability: Float = 0.125

    /// `base_options.py`'s batch, 25 images. Introduced in InferKit 0.4.0.
    static let referenceBatchSize = 25

    /// `base_options.py`'s crop, 176 pixels square. Introduced in InferKit 0.4.0.
    static let referenceImageSize = 176

    /// Builds the SIGGRAPH-17 colorizer for training or for reloading a trained checkpoint. With a
    /// `weightsURL` the released checkpoint or a file `NFKMLXWeights` saved loads; without one the network
    /// is randomly initialized. The network is in evaluation mode. Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?) throws -> NFKMLXSiggraphNet {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        net.train(false)
        return net
    }

    /// The optimizer `train_siggraph.sh`'s regression phase runs: `torch.optim.Adam` at 1e-5, betas 0.9 and
    /// 0.999, no decay. Introduced in InferKit 0.4.0.
    static func referenceOptimizer() -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: 1e-5, weightDecay: 0)
    }

    /// The network's input and target for RGB images `[N, H, W, 3]` in [0, 1], as `get_colorization_data`
    /// builds them: the images whose ab spans less than 5 are dropped, then each kept image receives
    /// random hint patches.
    ///
    /// - Returns: the input `[M, H, W, 4]` (lightness, hint, and the hint mask centered at zero) and the
    ///   true ab over 110, `[M, H, W, 2]`, for the `M` kept images; nil when every image is grayscale.
    ///
    /// Introduced in InferKit 0.4.0.
    static func trainingExample(_ images: MLXArray, hintProbability: Float = referenceHintProbability,
                                seed: UInt64 = 0) -> (input: MLXArray, target: MLXArray)? {
        var draws = NFKSiggraphSeededDraws(seed: seed)
        return trainingExample(images, hintProbability: hintProbability, draws: &draws)
    }

    /// The training example with the hint sampler's draws taken from `draws`.
    internal static func trainingExample<Draws: NFKSiggraphHintDraws>(_ images: MLXArray, hintProbability: Float,
                                                             draws: inout Draws) -> (input: MLXArray, target: MLXArray)? {
        let lab = referenceLab(images)                                     // [N, H, W, 3], normalized
        let lightness = lab[.ellipsis, 0 ..< 1], ab = lab[.ellipsis, 1 ..< 3]
        let span = abs(ab.max(axes: [1, 2]) - ab.min(axes: [1, 2])).sum(axis: -1)    // [N]
        let kept = span.asArray(Float.self).enumerated().compactMap { $0.element >= 5 / 110 ? Int32($0.offset) : nil }
        guard !kept.isEmpty else { return nil }
        let index = MLXArray(kept)
        let keptLightness = lightness[index], keptAB = ab[index]
        let (count, height, width) = (kept.count, keptAB.dim(1), keptAB.dim(2))
        let values = keptAB.asArray(Float.self)
        var hint = [Float](repeating: 0, count: count * height * width * 2)
        var mask = [Float](repeating: 0, count: count * height * width)
        let sizes = Array(1 ... 9)
        for image in 0 ..< count {
            while draws.unit() < Double(1 - hintProbability) {
                let size = draws.choice(sizes)
                func corner(_ extent: Int) -> Int {
                    let span = Double(extent - size + 1)
                    let value = draws.normal(mean: span / 2, deviation: span / 4)
                    return Int(min(max(value, 0), Double(extent - size)))
                }
                let row = corner(height), column = corner(width)
                for channel in 0 ..< 2 {
                    var sum: Float = 0
                    for y in row ..< row + size {
                        for x in column ..< column + size {
                            sum += values[((image * height + y) * width + x) * 2 + channel]
                        }
                    }
                    let mean = sum / Float(size * size)
                    for y in row ..< row + size {
                        for x in column ..< column + size {
                            hint[((image * height + y) * width + x) * 2 + channel] = mean
                        }
                    }
                }
                for y in row ..< row + size {
                    for x in column ..< column + size {
                        mask[(image * height + y) * width + x] = 1
                    }
                }
            }
        }
        let input = concatenated([keptLightness, MLXArray(hint, [count, height, width, 2]),
                                  MLXArray(mask, [count, height, width, 1]) - 0.5], axis: -1)
        return (input, keptAB)
    }

    /// `util.rgb2lab` on `[N, H, W, 3]`, normalized as `--l_cent 50`, `--l_norm 100`, and `--ab_norm 110`
    /// set.
    internal static func referenceLab(_ rgb: MLXArray) -> MLXArray {
        let lab = NFKColorizerTrainingLab.reference(rgb)
        return concatenated([(lab[.ellipsis, 0 ..< 1] - 50) / 100, lab[.ellipsis, 1 ..< 3] / 110], axis: -1)
    }

    /// Fine-tunes the SIGGRAPH-17 colorizer on color images.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:)``.
    ///   - examples: supplies one batch per step: RGB images `[N, H, W, 3]` in [0, 1]. The reference
    ///     resizes to 256 and crops 176 pixels square (``referenceImageSize``), with a random mirror; those
    ///     are the caller's.
    ///   - objective: the loss.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer()``.
    ///   - steps: how many updates to train for.
    ///   - hintProbability: the hint rate; one disables hints, as the reference's classification phases run.
    ///   - hintSeed: seeds the hints, so a run repeats.
    ///   - clipGradientNorm: bounds the global gradient norm. The reference clips nothing.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is a constant rate, the reference's with
    ///     `--niter_decay 0`.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Every parameter trains, the batch normalizations on each batch's statistics. A batch whose images are
    /// all grayscale trains on all of them, where the reference skips it. Save with `NFKMLXWeights.save`;
    /// ``network(weightsURL:)`` and `backend(weightsURL:)` read the file back. A run is minutes; call it off
    /// the render thread. Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXSiggraphNet,
        examples: (Int) -> MLXArray,
        objective: NFKMLXSiggraphColorizerObjective = NFKMLXSiggraphColorizerObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        hintProbability: Float = referenceHintProbability,
        hintSeed: UInt64 = 0,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { referenceOptimizer() },
            referenceSchedule: { .constant },
            steps: steps,
            arrays: { step in
                let images = examples(step)
                if let example = trainingExample(images, hintProbability: hintProbability, seed: hintSeed &+ UInt64(step)) {
                    return [example.input, example.target]
                }
                let lab = referenceLab(images)
                return [concatenated([lab[.ellipsis, 0 ..< 1], MLXArray.zeros(lab[.ellipsis, 1 ..< 3].shape),
                                      MLXArray.zeros(lab[.ellipsis, 0 ..< 1].shape) - 0.5], axis: -1),
                        lab[.ellipsis, 1 ..< 3]]
            },
            loss: { net, arrays in objective(net, arrays[0], arrays[1]) },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
