//
//  NFKMLXDemucsTraining.swift
//  InferKitMLX
//
//  Training the music separators on a consumer's own stems, as facebookresearch/demucs trains them: the
//  stems are augmented, summed into the mix, and the network's estimate of each stem is scored against it
//  by a waveform distance. The released Demucs v2 trained under the v2 branch's `train.py`, and Hybrid
//  Transformer Demucs under v4's `solver.py`; the two share the loss and the augmentations, which live
//  here.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The loss Demucs trains with: a waveform distance between each estimated stem and its source, averaged
/// over channels and samples, then over the batch, and combined across the stems by their weights.
///
/// @discussion With equal weights this is the mean distance over every stem, channel, and sample, the v2
/// branch's `nn.L1Loss`. v4's `solver.py` weights the stems, and the fine-tuned `htdemucs_ft` release trains
/// each of its four models with one stem's weight at 1 and the others at 0.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXDemucsObjective: Sendable {

    /// The waveform distance, the reference's `optim.loss`.
    public enum Distance: Sendable {
        /// `F.l1_loss`, every release's setting.
        case l1
        /// `F.mse_loss`.
        case l2
    }

    public var distance: Distance
    /// One weight per stem, the reference's `weights`. Nil weighs every stem equally.
    public var weights: [Float]?

    public init(distance: Distance = .l1, weights: [Float]? = nil) {
        self.distance = distance
        self.weights = weights
    }

    /// The objective that trains one stem alone, as each model of `htdemucs_ft` was fine-tuned.
    public static func stem(_ index: Int, of count: Int) -> NFKMLXDemucsObjective {
        NFKMLXDemucsObjective(weights: (0 ..< count).map { $0 == index ? 1 : 0 })
    }

    /// Scores a Demucs v2 network on a batch of stems `[N, stems, channels, L]`: the stems sum into the
    /// mix, and the estimate, shorter than the mix by what the decoder's context consumes, is compared
    /// with the stems trimmed to its middle, as the v2 branch's `train.py` does.
    public func callAsFunction(_ net: NFKMLXDemucsNet, _ stems: MLXArray) -> MLXArray {
        let (batch, count, channels) = (stems.dim(0), stems.dim(1), stems.dim(2))
        let mix = stems.sum(axis: 1).transposed(0, 2, 1)
        let estimate = net.estimate(mix)
        let length = estimate.dim(1)
        let stemsFirst = estimate.reshaped([batch, length, count, channels]).transposed(0, 2, 3, 1)
        return loss(estimate: stemsFirst, sources: Self.centerTrimmed(stems, to: length))
    }

    /// Scores estimated stems against their sources, both `[N, stems, channels, L]`.
    public func loss(estimate: MLXArray, sources: MLXArray) -> MLXArray {
        let difference = estimate - sources
        let distances = distance == .l1 ? abs(difference) : square(difference)
        let perStem = distances.mean(axes: [2, 3]).mean(axis: 0)
        guard let weights else {
            return perStem.mean()
        }
        let weight = MLXArray(weights)
        return (perStem * weight).sum() / weight.sum()
    }

    /// The reference's `center_trim` over the sample axis of `[N, stems, channels, L]`.
    static func centerTrimmed(_ stems: MLXArray, to length: Int) -> MLXArray {
        let delta = stems.dim(3) - length
        guard delta > 0 else { return stems }
        return stems[0..., 0..., 0..., (delta / 2) ..< (stems.dim(3) - (delta - delta / 2))]
    }
}

/// The random draws the augmentations take, in the order the reference takes them.
protocol NFKDemucsDraws {
    /// `random.random()`.
    mutating func unit() -> Double
    /// `torch.randint(bound, [count])`.
    mutating func integers(below bound: Int, count: Int) -> [Int]
    /// `torch.empty(count).uniform_(low, high)`; `torch.rand(count)` over 0 to 1.
    mutating func uniforms(_ count: Int, low: Float, high: Float) -> [Float]
}

extension NFKDemucsDraws {
    mutating func integers(below bound: Int, count: Int) -> [Int] {
        (0 ..< count).map { _ in min(Int(unit() * Double(bound)), bound - 1) }
    }

    mutating func uniforms(_ count: Int, low: Float, high: Float) -> [Float] {
        (0 ..< count).map { _ in low + (high - low) * Float(unit()) }
    }
}

extension NFKDenoiserSeededDraws: NFKDemucsDraws {}

/// The augmentations Demucs applies to a batch of stems before they are summed into the mix.
///
/// @discussion They apply in the reference's order: a random crop `shift` samples shorter than the
/// stems, a swap of the left and right channels, a sign flip, a gain, and a remix that shuffles each stem
/// among the examples of a group, so a mix combines stems from different songs. Each stem of each example
/// draws its own crop, swap, sign, and gain. The v2 branch flips the sign before the channels; the order
/// matters only for which draw feeds which flip.
///
/// The reference's data loader also changes the tempo and pitch of a fifth of the examples through the
/// `soundstretch` program. That runs before batching, on the consumer's data, and is not ported here.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXDemucsAugmentation: Sendable {
    /// How many samples shorter than the stems the random crop is; 0 crops nothing (`Shift`).
    public var shift: Int
    /// Crops every stem of an example at the same offset rather than independently (`shift_same`).
    public var shiftSame: Bool
    /// Swaps the stereo channels and flips the sign of each stem at random (`flip`).
    public var flip: Bool
    /// Flips the sign before swapping the channels, the v2 branch's order.
    public var signBeforeChannels: Bool
    /// The probability a batch takes a random gain per stem (`Scale`); 0 turns it off.
    public var scaleProbability: Float
    /// The gain's range.
    public var scaleRange: ClosedRange<Float>
    /// The probability a batch is remixed (`Remix`); 0 turns it off.
    public var remixProbability: Float
    /// How many examples a remix shuffles stems among. The batch must be a multiple of it.
    public var remixGroupSize: Int

    public init(shift: Int = 44_100, shiftSame: Bool = false, flip: Bool = true, signBeforeChannels: Bool = false,
                scaleProbability: Float = 1, scaleRange: ClosedRange<Float> = 0.25 ... 1.25,
                remixProbability: Float = 1, remixGroupSize: Int = 4) {
        self.shift = shift
        self.shiftSame = shiftSame
        self.flip = flip
        self.signBeforeChannels = signBeforeChannels
        self.scaleProbability = scaleProbability
        self.scaleRange = scaleRange
        self.remixProbability = remixProbability
        self.remixGroupSize = remixGroupSize
    }

    /// The v2 branch's `train.py`, which trained the released Demucs v2: a one-second crop, both flips,
    /// a gain on every batch, and a remix in groups of four.
    public static let demucsV2 = NFKMLXDemucsAugmentation(signBeforeChannels: true)

    /// v4's `config.yaml`, which trained `htdemucs` and `htdemucs_6s`: the same augmentations in v4's
    /// order.
    public static let htdemucs = NFKMLXDemucsAugmentation()

    /// The `mmi_ft` grid that fine-tuned `htdemucs_ft`: one crop shared by an example's stems and both
    /// flips, with no gain and no remix.
    public static let htdemucsFineTune = NFKMLXDemucsAugmentation(shiftSame: true, scaleProbability: 0,
                                                                   remixProbability: 0)

    /// Whether `batch` examples can be remixed: the reference refuses a batch its group size does not divide.
    func accepts(batch: Int) -> Bool {
        remixProbability <= 0 || batch % remixGroupSize == 0
    }

    /// Augments stems `[N, stems, channels, L]`, returning `[N, stems, channels, L - shift]`.
    func apply<Draws: NFKDemucsDraws>(_ stems: MLXArray, draws: inout Draws) -> MLXArray {
        var x = stems
        if shift > 0 {
            x = shifted(x, draws: &draws)
        }
        if flip {
            if signBeforeChannels {
                x = signFlipped(x, draws: &draws)
                x = channelsSwapped(x, draws: &draws)
            } else {
                x = channelsSwapped(x, draws: &draws)
                x = signFlipped(x, draws: &draws)
            }
        }
        // `Scale` and `Remix` exist only when their probability is set, and then take a draw on every batch.
        if scaleProbability > 0, draws.unit() < Double(scaleProbability) {
            x = scaled(x, draws: &draws)
        }
        if remixProbability > 0, draws.unit() < Double(remixProbability) {
            x = remixed(x, draws: &draws)
        }
        return x
    }

    private func shifted<Draws: NFKDemucsDraws>(_ x: MLXArray, draws: inout Draws) -> MLXArray {
        let (batch, count, channels, length) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3) - shift)
        let drawn = draws.integers(below: shift, count: batch * (shiftSame ? 1 : count))
        let offsets = MLXArray(drawn.map(Int32.init), [batch, shiftSame ? 1 : count, 1, 1])
        let positions = MLXArray(Array(Int32(0) ..< Int32(length)), [1, 1, 1, length])
        let indices = broadcast(offsets + positions, to: [batch, count, channels, length])
        return takeAlong(x, indices, axis: 3)
    }

    private func channelsSwapped<Draws: NFKDemucsDraws>(_ x: MLXArray, draws: inout Draws) -> MLXArray {
        guard x.dim(2) == 2 else { return x }
        let (batch, count) = (x.dim(0), x.dim(1))
        let left = draws.integers(below: 2, count: batch * count)
        let swaps = MLXArray(left.map { $0 == 1 }, [batch, count, 1, 1])
        let swapped = concatenated([x[0..., 0..., 1 ..< 2, 0...], x[0..., 0..., 0 ..< 1, 0...]], axis: 2)
        return MLX.where(swaps, swapped, x)
    }

    private func signFlipped<Draws: NFKDemucsDraws>(_ x: MLXArray, draws: inout Draws) -> MLXArray {
        let (batch, count) = (x.dim(0), x.dim(1))
        let signs = draws.integers(below: 2, count: batch * count).map { Float(2 * $0 - 1) }
        return x * MLXArray(signs, [batch, count, 1, 1])
    }

    private func scaled<Draws: NFKDemucsDraws>(_ x: MLXArray, draws: inout Draws) -> MLXArray {
        let (batch, count) = (x.dim(0), x.dim(1))
        let scales = draws.uniforms(batch * count, low: scaleRange.lowerBound, high: scaleRange.upperBound)
        return x * MLXArray(scales, [batch, count, 1, 1])
    }

    /// Each group's examples trade stems: stem `s` of example `i` comes from the example whose random key
    /// ranks `i`-th among the group's keys for that stem (`argsort` over the group).
    private func remixed<Draws: NFKDemucsDraws>(_ x: MLXArray, draws: inout Draws) -> MLXArray {
        let (batch, count, channels, length) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let groups = batch / remixGroupSize
        let keys = draws.uniforms(groups * remixGroupSize * count, low: 0, high: 1)
        var sources = [Int32](repeating: 0, count: batch * count)
        for group in 0 ..< groups {
            for stem in 0 ..< count {
                let members = (0 ..< remixGroupSize).map { keys[(group * remixGroupSize + $0) * count + stem] }
                let order = members.indices.sorted { members[$0] < members[$1] }
                for (rank, member) in order.enumerated() {
                    sources[(group * remixGroupSize + rank) * count + stem] = Int32(group * remixGroupSize + member)
                }
            }
        }
        let rows = MLXArray(sources, [batch, count, 1, 1])
        let flat = x.reshaped([batch, count, channels * length])
        let gathered = takeAlong(flat, broadcast(rows.reshaped([batch, count, 1]), to: [batch, count, channels * length]),
                                 axis: 0)
        return gathered.reshaped([batch, count, channels, length])
    }
}

public extension NFKMLXDemucs {

    /// The v2 branch's batch, 64 examples, which the recipe reaches through `accumulationSteps` or a
    /// batch of that many. Introduced in InferKit 0.4.0.
    static let referenceBatchSize = 64

    /// The v2 branch's training clip, ten seconds at 44.1 kHz before ``NFKMLXDemucsNet/validLength(_:)``
    /// rounds it. Introduced in InferKit 0.4.0.
    static let referenceSegmentSamples = 441_000

    /// Builds a Demucs v2 network for training or for reloading a trained checkpoint. With a `weightsURL`
    /// the checkpoint loads, the released `demucs` or one saved by `NFKMLXWeights`, and its width, depth,
    /// bottleneck layers, stem count, and audio channels are read from the file, the rest of the geometry
    /// coming from `configuration`; without one the net is randomly initialized at `configuration`.
    /// Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?,
                        configuration: NFKMLXDemucsConfiguration = NFKMLXDemucsConfiguration()) throws -> NFKMLXDemucsNet {
        guard let weightsURL else {
            return makeNet(configuration)
        }
        let arrays = try NFKMLXWeights.loadCheckpoint(url: weightsURL).arrays
        let net = makeNet(try Self.configuration(matching: arrays, base: configuration))
        try loadWeights(into: net, from: weightsURL)
        return net
    }

    /// The geometry a checkpoint holds, in the reference's layout or the module's: the first encoder's
    /// output width and input channels, the number of encoders, the bottleneck's layers, and the last
    /// decoder's output channels over the audio channels.
    internal static func configuration(matching arrays: [String: MLXArray],
                                       base: NFKMLXDemucsConfiguration) throws -> NFKMLXDemucsConfiguration {
        let reference = arrays["encoder.0.0.weight"] != nil
        guard let first = arrays["encoder.0.0.weight"] ?? arrays["encoder.0.conv1.weight"] else {
            throw NFKMLXError.unsupportedConfiguration("the checkpoint has no first encoder weight")
        }
        var configuration = base
        configuration.baseChannels = first.dim(0)
        configuration.audioChannels = reference ? first.dim(1) : first.dim(2)
        configuration.depth = (0...).first { arrays["encoder.\($0).0.weight"] == nil && arrays["encoder.\($0).conv1.weight"] == nil }!
        configuration.lstmLayers = (0...).first {
            arrays["lstm.lstm.weight_ih_l\($0)"] == nil && arrays["lstm.lstm.\($0).Wx"] == nil
        }!
        let last = configuration.depth - 1
        guard let output = reference ? arrays["decoder.\(last).2.weight"].map({ $0.dim(1) })
                                     : arrays["decoder.\(last).convt.conv.weight"].map({ $0.dim(0) }) else {
            throw NFKMLXError.unsupportedConfiguration("the checkpoint has no last decoder weight")
        }
        configuration.stems = output / configuration.audioChannels
        return configuration
    }

    /// `torch.optim.Adam` at `learningRate` with betas 0.9 and 0.999 and no decay, the optimizer both the
    /// music separator and the speech denoiser train with.
    ///
    /// @discussion The bottleneck's LSTMs fold the reference's two biases (`bias_ih`, `bias_hh`) into one.
    /// Adam moves each of the two by the same normalized step, so the folded bias steps at twice the rate.
    /// Introduced in InferKit 0.4.0.
    static func referenceOptimizer(for net: NFKMLXDemucsNet, learningRate: Float = 3e-4) -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: learningRate, over: net) { key in
            let recurrentBias = (key.hasPrefix("lstm.lstm.") || key.hasPrefix("lstm.reverse.")) && key.hasSuffix(".bias")
            return (rateScale: recurrentBias ? 2 : 1, weightDecay: 0)
        }
    }

    /// Trains a Demucs v2 network on separated stems.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:configuration:)``.
    ///   - stems: supplies one batch per step, `[N, stems, channels, L]` at 44.1 kHz in the network's stem
    ///     order. With the augmentation's crop, `L` is ``NFKMLXDemucsNet/validLength(_:)`` of the clip
    ///     length plus the crop's `shift`, as the reference's loader reads it.
    ///   - objective: the loss; L1 over equally weighted stems by default.
    ///   - augmentation: augments each batch; ``NFKMLXDemucsAugmentation/demucsV2`` by default. Nil applies
    ///     none.
    ///   - augmentationSeed: seeds the augmentations' draws, the reference's `seed` by default.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer(for:learningRate:)``.
    ///   - steps: how many updates to train for.
    ///   - clipGradientNorm: bounds the global gradient norm; the reference clips none.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is the reference's constant rate.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Every parameter trains. Save with `NFKMLXWeights.save`; ``network(weightsURL:configuration:)`` and
    /// `backend(weightsURL:)` read the file back. A run is minutes; call it off the render thread.
    /// - Throws: `NFKMLXError.unsupportedConfiguration` when the remix's group size does not divide the batch.
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXDemucsNet,
        stems: (Int) -> MLXArray,
        objective: NFKMLXDemucsObjective = NFKMLXDemucsObjective(),
        augmentation: NFKMLXDemucsAugmentation? = .demucsV2,
        augmentationSeed: UInt64 = 42,
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let batches = try NFKDemucsAugmentedBatches(stems: stems, augmentation: augmentation, seed: augmentationSeed)
        return try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { referenceOptimizer(for: net) },
            referenceSchedule: { .constant },
            steps: steps,
            sample: { batches.next($0, from: stems) },
            loss: objective.callAsFunction,
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}

/// A recipe's batches with the augmentation applied, checked against the remix's group size before the
/// run starts.
final class NFKDemucsAugmentedBatches {
    private let augmentation: NFKMLXDemucsAugmentation?
    private var draws: NFKDenoiserSeededDraws
    private var first: MLXArray?

    init(stems: (Int) -> MLXArray, augmentation: NFKMLXDemucsAugmentation?, seed: UInt64) throws {
        let first = stems(0)
        if let augmentation, !augmentation.accepts(batch: first.dim(0)) {
            throw NFKMLXError.unsupportedConfiguration(
                "a remix in groups of \(augmentation.remixGroupSize) needs a batch that is a multiple of it, not \(first.dim(0))")
        }
        self.augmentation = augmentation
        self.draws = NFKDenoiserSeededDraws(seed: seed)
        self.first = first
    }

    func next(_ step: Int, from stems: (Int) -> MLXArray) -> MLXArray {
        let batch: MLXArray
        if step == 0, let first {
            batch = first
            self.first = nil
        } else {
            batch = stems(step)
        }
        guard let augmentation else {
            return batch
        }
        return augmentation.apply(batch, draws: &draws)
    }
}
