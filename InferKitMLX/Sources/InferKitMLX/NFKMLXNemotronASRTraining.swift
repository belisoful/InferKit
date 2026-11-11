//
//  NFKMLXNemotronASRTraining.swift
//  InferKitMLX
//
//  Fine-tuning Nemotron ASR Streaming's transducer on a consumer's own transcribed audio.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a Nemotron ASR Streaming fine-tune updates.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXNemotronASRTrainable: Sendable {

    /// The transducer: the encoder projection, the prediction network, and the joint head, with the
    /// 600-million-parameter encoder frozen. A head retarget to a consumer's vocabulary of words and names.
    case transducer

    /// Every parameter, the encoder included.
    case everything
}

/// The transducer loss Nemotron ASR Streaming trains with: NeMo's `RNNTLoss` as the release's
/// `model_config.yaml` configures it, the negative log-likelihood of the label sequence over every
/// alignment of the joint's `[frames, labels + 1]` lattice with FastEmit regularization, summed over the
/// batch and divided by its label count (`mean_volume`).
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXNemotronASRObjective: Sendable {

    /// How a batch's utterance losses combine.
    public enum Reduction: Sendable {
        /// The sum over the batch divided by its total label count, the release's `mean_volume`.
        case meanVolume
        /// The mean over the batch, NeMo's default `mean_batch`.
        case meanBatch
    }

    /// The blank's index, the joint's last logit.
    public let blank: Int
    /// FastEmit's weight: each label emission's gradient is scaled by `1 + fastEmitLambda`, and each
    /// utterance's reported loss by the same factor, as NeMo's numba kernel does. The release trains at
    /// 0.005.
    public let fastEmitLambda: Float
    public let reduction: Reduction

    public init(blank: Int, fastEmitLambda: Float = 0.005, reduction: Reduction = .meanVolume) {
        self.blank = blank
        self.fastEmitLambda = fastEmitLambda
        self.reduction = reduction
    }

    /// One utterance's negative log-likelihood `−log P(labels | frames)`, scaled by `1 + fastEmitLambda`,
    /// with the label emissions' gradient scaled by the same factor and the blanks' left as they are.
    ///
    /// The forward variable runs over frames: `α(0, u)` sums the first `u` labels' log-probabilities at
    /// frame 0, and each later frame's `α(t, ·) = C + logcumsumexp(α(t − 1, ·) + blank(t − 1, ·) − C)`,
    /// where `C` is the running sum of the frame's label log-probabilities. That is the recursion
    /// `α(t, u) = logaddexp(α(t − 1, u) + blank(t − 1, u), α(t, u − 1) + label(t, u − 1))` evaluated a
    /// frame at a time. The likelihood is `α(T − 1, U) + blank(T − 1, U)`.
    ///
    /// - Parameters:
    ///   - logits: the joint's raw logits, `[frames, labels + 1, vocabulary]`.
    ///   - labels: the label ids, `[labels]`.
    public func negativeLogLikelihood(logits: MLXArray, labels: MLXArray) -> MLXArray {
        let logProbabilities = logits - logSumExp(logits, axis: -1, keepDims: true)
        let frames = logits.dim(0), count = labels.dim(0)
        let blanks = logProbabilities[0..., 0..., blank]                                         // [T, U + 1]
        var emitted = takeAlong(logProbabilities[0..., 0 ..< count, 0...],
                                labels.reshaped([1, count, 1]).asType(.int32), axis: 2)[0..., 0..., 0]   // [T, U]
        if fastEmitLambda != 0 {
            // The same values, with their gradient scaled by 1 + λ.
            emitted = emitted + fastEmitLambda * (emitted - stopGradient(emitted))
        }
        func runningSums(_ frame: Int) -> MLXArray {
            concatenated([MLXArray.zeros([1]), emitted[frame].cumsum()], axis: 0)                // [U + 1]
        }
        var alpha = runningSums(0)
        for frame in 1 ..< frames {
            let sums = runningSums(frame)
            alpha = sums + (alpha + blanks[frame - 1] - sums).logCumsumExp(axis: 0)
        }
        let nll = -(alpha[count] + blanks[frames - 1, count])
        return nll + stopGradient(fastEmitLambda * nll)
    }

    /// The batch's loss under the objective's reduction.
    public func loss(_ utterances: [(logits: MLXArray, labels: MLXArray)]) -> MLXArray {
        let total = stacked(utterances.map { negativeLogLikelihood(logits: $0.logits, labels: $0.labels) }).sum()
        switch reduction {
        case .meanVolume:
            return total / Float(max(utterances.reduce(0) { $0 + $1.labels.dim(0) }, 1))
        case .meanBatch:
            return total / Float(max(utterances.count, 1))
        }
    }
}

extension NFKMLXNemotronASRNet {

    /// The joint's logits over the whole alignment lattice, `[frames, labels + 1, vocabulary]`: every encoder
    /// frame's projection against the prediction network's output after the blank and each prefix of
    /// `labels`, through the joint.
    public func latticeLogits(encoded: MLXArray, labels: [Int]) -> MLXArray {
        let projected = encoderProjector(encoded[0])                                              // [T, H]
        let tokens = MLXArray(([configuration.blank] + labels).map { Int32($0) }).reshaped([1, labels.count + 1])
        let (output, _) = decoder.lstm(decoder.embedding(tokens), state: nil)
        let predicted = decoder.projector(output[0])                                              // [U + 1, H]
        return joint(encoder: projected.expandedDimensions(axis: 1), prediction: predicted.expandedDimensions(axis: 0))
    }
}

extension NFKMLXNemotronASR {

    /// Builds the recognizer's network, ready to fine-tune, from a release directory, or from a release's
    /// configuration with the weights at `weightsURL` (a release's `model.safetensors` or a file
    /// `NFKMLXWeights.save` wrote). Nil weights leave it at its random initialization.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, weightsURL: URL? = nil) throws -> NFKMLXNemotronASRNet {
        let net = NFKMLXNemotronASRNet(try configuration(directoryURL: directoryURL))
        try loadWeights(into: net, from: weightsURL ?? directoryURL.appendingPathComponent("model.safetensors"))
        return net
    }

    /// Fine-tunes `net` on transcribed 16 kHz audio, returning the loss from each step.
    ///
    /// The whole path is three calls: ``network(directoryURL:weightsURL:)`` to build, this to train, and
    /// `NFKMLXWeights.save` to write a checkpoint that ``network(directoryURL:weightsURL:)`` and the backend
    /// load.
    ///
    /// - Parameters:
    ///   - net: the network.
    ///   - examples: one batch per step: each utterance's 16 kHz samples and its transcript's token ids,
    ///     without the blank.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - lookahead: the encoder's right context while training; nil is the release's default.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer(for:)``.
    ///   - steps: how many batches to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - accumulationSteps: how many batches each update averages.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil holds the rate constant.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXNemotronASRNet,
        examples: (Int) -> [(samples: [Float], labels: [Int])],
        trainable: NFKMLXNemotronASRTrainable = .transducer,
        lookahead: Int? = nil,
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let configuration = net.configuration
        let right = lookahead ?? configuration.defaultLookahead
        let objective = NFKMLXNemotronASRObjective(blank: configuration.blank)
        return try NFKMLXFineTune.run(
            net,
            freezing: { apply(trainable, to: net) },
            optimizer: optimizer,
            reference: { referenceOptimizer(for: net) },
            referenceSchedule: { referenceSchedule },
            steps: steps,
            arrays: { step in
                let batch = examples(step)
                return batch.flatMap { example in
                    [net.frontEnd.features(example.samples), MLXArray(example.labels.map { Int32($0) })]
                }
            },
            loss: { net, arrays in
                let utterances = stride(from: 0, to: arrays.count, by: 2).map { index -> (logits: MLXArray, labels: MLXArray) in
                    var encoded = net.encode(features: arrays[index], lookahead: right)
                    if trainable == .transducer {
                        encoded = stopGradient(encoded)
                    }
                    let labels = arrays[index + 1]
                    return (net.latticeLogits(encoded: encoded, labels: labels.asArray(Int32.self).map(Int.init)), labels)
                }
                return objective.loss(utterances)
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// The release's own optimizer, `model_config.yaml`'s `optim`: AdamW at a peak rate of 3.0 with betas
    /// (0.9, 0.98) and a weight decay of 0.001, over the trainable parameters. The prediction network's
    /// LSTM keeps one bias where PyTorch keeps two that each receive the gradient, so the folded bias steps
    /// at twice the rate, which moves it as the pair moves. ``referenceSchedule`` scales the rate.
    public static func referenceOptimizer(for net: NFKMLXNemotronASRNet) -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: 3.0, betas: (0.9, 0.98), over: net) { key in
            (key.hasPrefix("decoder.lstm.") && key.hasSuffix(".bias") ? 2 : 1, 0.001)
        }
    }

    /// The release's `NoamAnnealing` schedule: the rate times `d_model^−½ · min(step^−½, step · warmup^−1.5)`
    /// with `d_model` 1024 and 8,000 warmup steps, held at or above 1e-6 once the warmup has passed.
    public static var referenceSchedule: NFKMLXLearningRateSchedule {
        .noam(modelDimension: 1024, warmupSteps: 8000, minimumRate: 1e-6, peakRate: 3.0)
    }

    static func apply(_ trainable: NFKMLXNemotronASRTrainable, to net: NFKMLXNemotronASRNet) {
        net.unfreeze()
        if trainable == .transducer {
            net.encoder.freeze()
        }
    }
}
