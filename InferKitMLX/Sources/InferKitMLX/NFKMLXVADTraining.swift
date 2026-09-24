//
//  NFKMLXVADTraining.swift
//  InferKitMLX
//
//  Fine-tuning the MarbleNet voice activity detector on a consumer's own audio: their microphone, their
//  room, their definition of what counts as speech. The network is under 100k parameters, so every
//  weight trains.
//
//  The recipe is the one the release's own `model_config.yaml` names for NeMo's
//  `EncDecFrameClassificationModel`: per-frame cross-entropy over the speech and non-speech classes,
//  masked to the labeled frames; SGD with momentum 0.9 at 0.01 and weight decay 0.001; NeMo's
//  `PolynomialHoldDecayAnnealing` (5% warm-up, 15% hold, power 2, a floor of 1e-8). While it trains the
//  front end adds its 1e-5 dither, `SpectrogramAugmentation` masks five frequency bands up to 10 bins
//  wide and five time spans up to 5% of the clip, and the encoder drops 10% after every activation.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/*!
 @abstract The objective a MarbleNet fine-tune minimizes: NeMo's masked per-frame cross-entropy.
 @discussion Scores logits `[1, frames, classes]` against one class index per frame, speech 1 and
 non-speech 0, averaged over the frames the mask keeps, with each class's weight applied as
 `torch.nn.CrossEntropyLoss` applies it. The release trains with equal weights. Measured against the
 release by `run_reference.py vad_training`. Introduced in InferKit 0.5.0.
 */
public struct NFKMLXVADObjective: Sendable {

    /// The weight of each class, non-speech first. Nil weighs them equally.
    public var classWeights: [Float]?

    public init(classWeights: [Float]? = nil) {
        self.classWeights = classWeights
    }

    /// The loss of logits `[1, frames, classes]` against labels `[1, frames]`, over the frames `mask`
    /// keeps (all of them when nil).
    public func loss(logits: MLXArray, labels: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let logProbabilities = logits - logSumExp(logits, axis: -1, keepDims: true)
        let indices = labels.asType(.int32).expandedDimensions(axis: -1)
        let picked = takeAlong(logProbabilities, indices, axis: -1).squeezed(axis: -1)
        let kept = (mask ?? MLXArray.ones(labels.shape)).asType(.float32)
        let weights = classWeights.map { take(MLXArray($0), labels.asType(.int32)) } ?? MLXArray.ones(labels.shape)
        let weighted = weights * kept
        return -(weighted * picked).sum() / weighted.sum()
    }
}

/*!
 @abstract NeMo's `SpectrogramAugmentation` as the release configures it, on a log-mel `[1, frames, mels]`.
 @discussion Its vectorized masking: each of `frequencyMasks` bands is `floor(u · frequencyWidth)` bins
 wide starting at `floor(u′ · (mels − width))`, each of `timeMasks` spans is `floor(u · timeFraction ·
 valid)` frames wide starting at `floor(u′ · (valid − width))` over the clip's valid frames, and every
 masked value becomes 0.
 Introduced in InferKit 0.5.0.
 */
public struct NFKMLXVADSpecAugment: Sendable {
    public var frequencyMasks = 5
    public var frequencyWidth = 10
    public var timeMasks = 5
    public var timeFraction: Float = 0.05

    public init() {}

    /// The release's settings.
    public static let reference = NFKMLXVADSpecAugment()

    /// `mel` with the masks drawn from MLX's random state applied; `validFrames` defaults to every frame.
    public func callAsFunction(_ mel: MLXArray, validFrames: Int? = nil) -> MLXArray {
        let frames = mel.dim(1), bins = mel.dim(2)
        let valid = Float(validFrames ?? frames)
        func mask(count: Int, width: Float, length: Int, span: Float) -> MLXArray {
            guard count > 0 else {
                return MLXArray.zeros([length], type: Bool.self)
            }
            let widths = floor(MLXRandom.uniform(low: 0, high: 1, [count]) * width)
            let starts = floor(MLXRandom.uniform(low: 0, high: 1, [count]) * (span - widths))
            let positions = MLXArray((0 ..< length).map { Float($0) }).reshaped([1, length])
            let inside = logicalAnd(positions .>= starts.reshaped([count, 1]), positions .< (starts + widths).reshaped([count, 1]))
            return inside.any(axis: 0)
        }
        let time = mask(count: timeMasks, width: min(timeFraction * valid, Float(frames)), length: frames, span: valid)
            .reshaped([1, frames, 1])
        let frequency = mask(count: frequencyMasks, width: Float(frequencyWidth), length: bins, span: Float(bins))
            .reshaped([1, 1, bins])
        return MLX.where(logicalOr(time, frequency), MLXArray(Float(0)), mel)
    }
}

extension NFKMLXVAD {

    /// Builds the detection network itself, ready to fine-tune, from the converted release or a file
    /// `NFKMLXWeights.save` wrote.
    ///
    /// - Since: InferKit 0.5.0
    public static func network(weightsURL: URL?) throws -> NFKMLXVADNet {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// The frames the network scores for a clip of `samples` at 16 kHz, each 20 ms.
    ///
    /// - Since: InferKit 0.5.0
    public static func frameCount(samples: Int, configuration c: NFKMLXVADConfiguration = .marbleNet) -> Int {
        let melFrames = samples / c.hopSamples + 1
        let padded = melFrames + melFrames % 2
        return (padded + c.totalStride - 1) / c.totalStride
    }

    /// One label per network frame from the spans that hold speech, in seconds: a frame is speech (1)
    /// when its midpoint falls inside a span.
    ///
    /// - Since: InferKit 0.5.0
    public static func frameLabels(speech spans: [(start: Double, end: Double)], frameCount: Int,
                                   configuration c: NFKMLXVADConfiguration = .marbleNet) -> [Int32] {
        let frameSeconds = Double(c.hopSamples * c.totalStride) / Double(c.sampleRate)
        return (0 ..< frameCount).map { frame in
            let middle = (Double(frame) + 0.5) * frameSeconds
            return spans.contains { middle >= $0.start && middle < $0.end } ? 1 : 0
        }
    }

    /// Fine-tunes every weight of `net` on labeled 16 kHz audio, returning the loss from each step.
    ///
    /// The whole path is three calls: ``network(weightsURL:)`` to build, this to train, and
    /// `NFKMLXWeights.save` to write a checkpoint that `backendWithWeightsURL:error:` loads.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:)``.
    ///   - examples: supplies one clip per step: 16 kHz samples and one label per network frame
    ///     (``frameCount(samples:configuration:)``, ``frameLabels(speech:frameCount:configuration:)``).
    ///     Labels past the network's frames are dropped, and frames past the labels are not scored.
    ///   - objective: NeMo's masked per-frame cross-entropy.
    ///   - augmentation: the masks applied to each training spectrogram. Nil trains on it unmasked.
    ///   - optimizer: the update rule. Nil uses the release's SGD: momentum 0.9, rate 0.01, weight
    ///     decay 0.001 added to the gradient.
    ///   - steps: how many clips to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The release does not clip.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the release's
    ///     `PolynomialHoldDecayAnnealing` over `steps` when the reference optimizer runs.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The batch normalizations train on each clip's statistics, as the release's do. A run is seconds to
    /// minutes; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXVADNet,
        examples: (Int) -> (samples: [Float], labels: [Int32]),
        objective: NFKMLXVADObjective = NFKMLXVADObjective(),
        augmentation: NFKMLXVADSpecAugment? = .reference,
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { SGD(learningRate: 0.01, momentum: 0.9, weightDecay: 0.001) },
            referenceSchedule: {
                .nemoPolynomialHoldDecay(steps: steps, warmupRatio: 0.05, holdRatio: 0.15, power: 2,
                                         minimumScale: 1e-8 / 0.01)
            },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                let dither = MLXRandom.normal([example.samples.count]).asArray(Float.self)
                let dithered = zip(example.samples, dither).map { $0 + 1e-5 * $1 }
                var (mel, validFrames) = net.frontEnd.logMelAndLength(dithered)
                if let augmentation {
                    mel = augmentation(mel, validFrames: validFrames)
                }
                return [mel, MLXArray(example.labels).reshaped([1, example.labels.count]), MLXArray(Int32(validFrames))]
            },
            loss: { net, arrays in
                let logits = net.logits(arrays[0], validFrames: arrays[2].item(Int.self))
                let frames = min(logits.dim(1), arrays[1].dim(1))
                return objective.loss(logits: logits[0..., 0 ..< frames, 0...], labels: arrays[1][0..., 0 ..< frames])
            },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }
}
