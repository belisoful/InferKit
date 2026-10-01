//
//  NFKMLXAudioTaggerTraining.swift
//  InferKitMLX
//
//  Retargeting the PANNs Cnn14 tagger to a consumer's own sound classes.
//
//  PANNs (qiuqiangkong/audioset_tagging_cnn at d2f4b8c) publishes the pieces: `finetune_template.py`
//  freezes the AudioSet-pretrained Cnn14 and trains a new linear layer over its 2048-wide embedding,
//  and `main.py` trains with `clip_bce`, a binary cross-entropy over the per-class sigmoid scores, under
//  `torch.optim.Adam` at 1e-3 with AMSGrad. The model stays in training mode throughout, so its dropouts
//  and its SpecAugment stripes apply to the frozen base as well.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Which parameters an audio-tagger fine-tune updates.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXAudioTaggerTrainable: Sendable {

    /// The classifier only, over the frozen Cnn14, as `finetune_template.py` with `--freeze_base`.
    case classifier

    /// Every parameter, as `main.py` trains Cnn14 on AudioSet.
    case everything
}

/// The SpecAugment masking Cnn14 applies to its normalized spectrogram while training: torchlibrosa's
/// `SpecAugmentation`, which zeroes stripes along time, then along the mel bands.
///
/// Each stripe's width is drawn from `0 ..< dropWidth` and its start from `0 ..< total − width`, per clip.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXAudioTaggerSpecAugment: Sendable {
    public var timeDropWidth: Int
    public var timeStripes: Int
    public var frequencyDropWidth: Int
    public var frequencyStripes: Int

    public init(timeDropWidth: Int = 64, timeStripes: Int = 2, frequencyDropWidth: Int = 8,
                frequencyStripes: Int = 2) {
        self.timeDropWidth = timeDropWidth
        self.timeStripes = timeStripes
        self.frequencyDropWidth = frequencyDropWidth
        self.frequencyStripes = frequencyStripes
    }

    /// Cnn14's own setting: two time stripes up to 64 frames wide, two band stripes up to 8 bands wide.
    public static let reference = NFKMLXAudioTaggerSpecAugment()

    /// Masks a spectrogram image `[1, frames, mels, 1]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var masked = x
        for (axis, width, stripes) in [(1, timeDropWidth, timeStripes), (2, frequencyDropWidth, frequencyStripes)] {
            let total = x.dim(axis)
            var shape = [1, 1, 1, 1]
            shape[axis] = total
            let position = MLXArray(Array(0 ..< total).map { Int32($0) }).reshaped(shape)
            for _ in 0 ..< stripes {
                let distance = MLXRandom.randInt(0 ..< width, [Int]()).item(Int.self)
                // torch.randint raises on an empty range; a spectrogram no wider than the stripe is
                // left whole instead.
                guard total - distance > 0 else { continue }
                let start = MLXRandom.randInt(0 ..< total - distance, [Int]()).item(Int.self)
                let kept = (position .< Int32(start)) .|| (position .>= Int32(start + distance))
                masked = masked * kept.asType(masked.dtype)
            }
        }
        return masked
    }
}

/// The supervised objective an audio-tagger fine-tune minimizes: PANNs' `clip_bce`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXAudioTaggerObjective: Sendable {

    public init() {}

    /// `F.binary_cross_entropy(sigmoid(logits), targets)`: the mean over every clip and class, with
    /// each log term floored at −100 as torch floors it.
    ///
    /// - Parameters:
    ///   - logits: the classifier's scores, `[clips, classCount]`.
    ///   - targets: 1 where a class is present and 0 where it is not, `[clips, classCount]`. A soft
    ///     target in between is scored as torch scores it.
    public func loss(logits: MLXArray, targets: MLXArray) -> MLXArray {
        let probabilities = sigmoid(logits)
        let logPresent = maximum(log(probabilities), MLXArray(Float(-100)))
        let logAbsent = maximum(log(1 - probabilities), MLXArray(Float(-100)))
        return mean(-(targets * logPresent + (1 - targets) * logAbsent))
    }
}

extension NFKMLXAudioTagger {

    /// Builds the tagging network itself, ready to fine-tune.
    ///
    /// - Parameters:
    ///   - weightsURL: the converted release, or a file ``save(_:to:)`` wrote. Nil leaves the network
    ///     at its random initialization.
    ///   - configuration: the geometry. Its `classCount` is the consumer's class set; when it differs
    ///     from the checkpoint's, the classifier is left at its random initialization and everything
    ///     else loads, which is what retargeting the tagger means.
    ///
    /// - Since: InferKit 0.4.0
    public static func network(weightsURL: URL?,
                               configuration: NFKMLXAudioTaggerConfiguration = .panns) throws -> NFKMLXAudioTaggerNet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL, retargeting: true)
        }
        return net
    }

    /// Writes `net` with the mel filterbank it runs on, so the file reloads through
    /// `backendWithWeightsURL:labels:error:` onto the release's own filterbank rather than a recomputed
    /// one.
    ///
    /// - Since: InferKit 0.4.0
    public static func save(_ net: NFKMLXAudioTaggerNet, to url: URL) throws {
        try NFKMLXWeights.save(net, extraArrays: ["logmel_extractor.melW": net.frontEnd.filterbank], to: url)
    }

    /// Fine-tunes `net` on a consumer's own labeled clips, returning the loss from each step.
    ///
    /// The whole path is three calls: ``network(weightsURL:configuration:)`` to build, this to train,
    /// and ``save(_:to:)`` to write a checkpoint that `backendWithWeightsURL:labels:error:` loads.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:configuration:)``.
    ///   - examples: supplies one clip per step: mono samples, their rate, and one target per class.
    ///     A clip at another rate is resampled to the network's.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - objective: PANNs' `clip_bce`.
    ///   - augmentation: the stripes masked on each training spectrogram. Nil trains on it unmasked.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.Adam` at 1e-3 with AMSGrad.
    ///   - steps: how many clips to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference does not
    ///     clip.
    ///   - accumulationSteps: how many clips each update averages; `steps` counts updates. 1, the
    ///     default, updates after every clip. The reference averages over 32.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil holds the rate constant, as the
    ///     reference does.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The reference mixes pairs of clips by default (`--augmentation mixup`), which needs a batch; a
    /// caller mixes its own examples for the same effect. A run is seconds to minutes; call it off the
    /// render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXAudioTaggerNet,
        examples: (Int) -> (samples: [Float], sampleRate: Int, targets: [Float]),
        trainable: NFKMLXAudioTaggerTrainable = .classifier,
        objective: NFKMLXAudioTaggerObjective = NFKMLXAudioTaggerObjective(),
        augmentation: NFKMLXAudioTaggerSpecAugment? = .reference,
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let rate = net.configuration.sampleRate
        return try NFKMLXFineTune.run(
            net,
            freezing: { apply(trainable, to: net) },
            optimizer: optimizer,
            reference: { NFKMLXAMSGrad(learningRate: 1e-3) },
            referenceSchedule: { .constant },
            steps: steps,
            batch: { step in
                let example = examples(step)
                let samples = NFKMLXAudioRate.matched(example.samples, from: example.sampleRate, to: rate)
                return (net.frontEnd.logMel(samples), MLXArray(example.targets).reshaped([1, example.targets.count]))
            },
            loss: { net, mel, targets in
                objective.loss(logits: net.logits(mel, augmentation: augmentation), targets: targets)
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    private static func apply(_ trainable: NFKMLXAudioTaggerTrainable, to net: NFKMLXAudioTaggerNet) {
        net.unfreeze()
        if trainable == .classifier {
            net.freeze()
            net.classifier.unfreeze()
        }
    }
}
