//
//  NFKMLXSileroVADTraining.swift
//  InferKitMLX
//
//  Fine-tuning Silero VAD's decoder on a consumer's own labeled audio.
//
//  snakers4 publishes the recipe in `tuning/` (v6.2.1): the learned STFT and the convolutional encoder
//  stay as released, and the LSTM decoder alone retrains under Adam at 5e-4. Each 512-sample chunk is
//  labeled speech when more than half of its samples fall inside an annotated span, and the per-chunk
//  binary cross-entropy weighs a non-speech chunk by 0.5.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a Silero VAD fine-tune updates.
///
/// Introduced in InferKit 0.5.0.
public enum NFKMLXSileroVADTrainable: Sendable {

    /// The LSTM decoder and its output convolution, with the STFT and the encoder frozen, as the
    /// reference's `tune.py` trains.
    case decoder

    /// Every parameter, including the learned STFT basis and the encoder.
    case everything
}

/// The supervised objective a Silero VAD fine-tune minimizes: the reference's per-chunk binary
/// cross-entropy, weighted by a mask, averaged over every chunk.
///
/// Introduced in InferKit 0.5.0.
public struct NFKMLXSileroVADObjective: Sendable {

    public init() {}

    /// `mean(BCE(probabilities, labels) · mask)` over the chunks, with each log term floored at −100
    /// as `torch.nn.BCELoss` floors it.
    ///
    /// - Parameters:
    ///   - probabilities: the decoder's speech probability per chunk, `[chunks]`.
    ///   - labels: 1 for a speech chunk and 0 otherwise, `[chunks]`.
    ///   - mask: each chunk's weight, `[chunks]`; ``NFKMLXSileroVAD/chunkTargets(speech:sampleCount:noiseWeight:configuration:)``
    ///     builds it.
    public func loss(probabilities: MLXArray, labels: MLXArray, mask: MLXArray) -> MLXArray {
        let logSpeech = maximum(log(probabilities), MLXArray(Float(-100)))
        let logSilence = maximum(log(1 - probabilities), MLXArray(Float(-100)))
        let crossEntropy = -(labels * logSpeech + (1 - labels) * logSilence)
        return mean(crossEntropy * mask)
    }
}

extension NFKMLXSileroVAD {

    /// Builds the voice-activity network itself, ready to fine-tune, from the converted release or a
    /// file `NFKMLXWeights.save` wrote. Nil weights leave it at its random initialization.
    ///
    /// - Since: InferKit 0.5.0
    public static func network(weightsURL: URL?) throws -> NFKMLXSileroVADNet {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// The reference's per-chunk targets for a 16 kHz clip of `sampleCount` samples whose speech lies in
    /// `spans` (seconds).
    ///
    /// The clip is zero-padded to whole 512-sample chunks. A sample is speech from
    /// `Int(start · 16000)` up to, not including, `Int(end · 16000)`, and a chunk is speech when more than
    /// half of its samples are. The mask weighs a speech chunk 1 and any other `noiseWeight`, the
    /// reference's `noise_loss`.
    ///
    /// - Since: InferKit 0.5.0
    public static func chunkTargets(speech spans: [(start: Double, end: Double)], sampleCount: Int,
                                    noiseWeight: Float = 0.5,
                                    configuration c: NFKMLXSileroVADConfiguration = .v6)
        -> (labels: [Float], mask: [Float]) {
        let chunks = (sampleCount + c.numSamples - 1) / c.numSamples
        let padded = chunks * c.numSamples
        var speech = [Bool](repeating: false, count: padded)
        for span in spans {
            let first = max(0, min(padded, Int(span.start * Double(c.sampleRate))))
            let end = max(0, min(padded, Int(span.end * Double(c.sampleRate))))
            for sample in first ..< max(first, end) {
                speech[sample] = true
            }
        }
        let labels = (0 ..< chunks).map { chunk -> Float in
            let voiced = speech[chunk * c.numSamples ..< (chunk + 1) * c.numSamples].filter { $0 }.count
            return 2 * voiced > c.numSamples ? 1 : 0
        }
        return (labels, labels.map { $0 == 1 ? 1 : noiseWeight })
    }

    /// Fine-tunes `net` on labeled 16 kHz audio, returning the loss from each step.
    ///
    /// The whole path is three calls: ``network(weightsURL:)`` to build, this to train, and
    /// `NFKMLXWeights.save` to write a checkpoint that `backendWithWeightsURL:error:` loads.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:)``.
    ///   - examples: supplies one clip per step: 16 kHz samples and the chunk targets
    ///     ``chunkTargets(speech:sampleCount:noiseWeight:configuration:)`` builds for them.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - objective: the reference's masked binary cross-entropy.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.Adam` at 5e-4. The
    ///     reference's LSTM keeps two bias vectors that both receive the gradient this port's one
    ///     folded bias receives, so the folded bias steps at twice the rate, which moves it exactly as
    ///     the pair moves.
    ///   - steps: how many clips to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference does not
    ///     clip.
    ///   - accumulationSteps: how many clips each update averages; `steps` counts updates. 1, the
    ///     default, updates after every clip. The reference averages over batches of 128 clips padded
    ///     to the longest, with padded chunks weighted zero.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil holds the rate constant, as the
    ///     reference does.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The reference also augments each training clip and crops it to eight seconds; both are the
    /// caller's data choices here. A run is seconds to minutes; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXSileroVADNet,
        examples: (Int) -> (samples: [Float], labels: [Float], mask: [Float]),
        trainable: NFKMLXSileroVADTrainable = .decoder,
        objective: NFKMLXSileroVADObjective = NFKMLXSileroVADObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let configuration = net.configuration
        return try NFKMLXFineTune.run(
            net,
            freezing: { apply(trainable, to: net) },
            optimizer: optimizer,
            reference: { referenceOptimizer(for: net) },
            referenceSchedule: { .constant },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                return [NFKMLXSileroVADNet.chunkedInput(example.samples, configuration),
                        MLXArray(example.labels), MLXArray(example.mask)]
            },
            loss: { net, arrays in
                objective.loss(probabilities: net.decoder(net.encoder(arrays[0])), labels: arrays[1],
                               mask: arrays[2])
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// `torch.optim.Adam` at 5e-4 over the decoder, with the folded LSTM bias at twice the rate.
    static func referenceOptimizer(for net: NFKMLXSileroVADNet) -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: 5e-4, over: net) { key in
            (key.hasSuffix("rnn.bias") ? 2 : 1, 0)
        }
    }

    private static func apply(_ trainable: NFKMLXSileroVADTrainable, to net: NFKMLXSileroVADNet) {
        net.unfreeze()
        if trainable == .decoder {
            net.encoder.freeze()
        }
    }
}
