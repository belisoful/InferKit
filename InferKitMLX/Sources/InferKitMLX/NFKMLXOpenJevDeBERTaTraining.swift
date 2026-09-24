//
//  NFKMLXOpenJevDeBERTaTraining.swift
//  InferKitMLX
//
//  Customizing open-jev-deberta: the release's objective (cross-entropy plus a weighted Brier score
//  over each question's options), a freezing policy, and a fine-tune on labeled decisions with the
//  reference's two learning rates, one for the backbone and a larger one for the head.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters an open-jev-deberta fine-tune moves.
public enum NFKMLXOpenJevDeBERTaTrainable: Sendable {
    /// The scoring head alone; the encoder stays as released. The cheapest run.
    case head
    /// The encoder and the head, which is how the release was trained.
    case all
}

/// One labeled state: its questions and the index of each question's right option (a choice's option,
/// a score's level, a noul's 1 for yes and 0 for no).
public struct NFKMLXOpenJevDeBERTaExample {
    public var state: Any
    public var questions: [NFKDecisionQuestion]
    public var labels: [Int]

    public init(state: Any, questions: [NFKDecisionQuestion], labels: [Int]) {
        self.state = state
        self.questions = questions
        self.labels = labels
    }

    /// One question with the index of its right option.
    public init(state: Any, question: NFKDecisionQuestion, label: Int) {
        self.init(state: state, questions: [question], labels: [label])
    }

    /// One noul and whether its statement holds.
    public init(state: Any, question: NFKDecisionQuestion, holds: Bool) {
        self.init(state: state, questions: [question], labels: [holds ? 1 : 0])
    }
}

/// The release's `decision_loss`: the mean cross-entropy of each question's gold option plus
/// `brierWeight` times the mean Brier score, the squared error of the softmax against the one-hot
/// gold summed over the question's options. Both read the raw logits, before the temperature.
public struct NFKMLXOpenJevDeBERTaObjective: Sendable {
    /// The weight on the Brier term; the release trains at 1.
    public var brierWeight: Float

    public init(brierWeight: Float = 1) {
        self.brierWeight = brierWeight
    }

    /// The loss over `[batch, questions, options]` logits.
    ///
    /// - Parameters:
    ///   - logits: `-inf` where `optionMask` is false.
    ///   - gold: `[batch, questions]`, the right option's index, or -1 where a row has no question.
    ///   - optionMask: `[batch, questions, options]`, true where an option exists.
    public func loss(logits: MLXArray, gold: MLXArray, optionMask: MLXArray) -> MLXArray {
        terms(logits: logits, gold: gold, optionMask: optionMask).total
    }

    /// The loss and its two terms.
    public func terms(logits: MLXArray, gold: MLXArray, optionMask: MLXArray)
        -> (total: MLXArray, crossEntropy: MLXArray, brier: MLXArray) {
        let options = logits.shape[2]
        let rows = logits.reshaped([-1, options])
        let mask = optionMask.reshaped([-1, options])
        let labels = gold.reshaped([-1])
        let valid = (labels .>= 0).asType(.float32)
        // A padded question has no finite logit; zeroing its row keeps the softmax finite, and its
        // weight of zero keeps it out of both means.
        let safe = which(mask, rows, MLXArray(-Float.infinity))
        let rowExists = expandedDimensions(valid, axis: -1) .> 0
        let finiteRows = which(rowExists, safe, MLXArray(Float(0)))
        let logProbabilities = finiteRows - logSumExp(finiteRows, axis: -1, keepDims: true)
        let index = maximum(labels, MLXArray(Int32(0))).asType(.int32).expandedDimensions(axis: -1)
        let picked = takeAlong(logProbabilities, index, axis: -1).squeezed(axis: -1)
        let count = maximum(valid.sum(), MLXArray(Float(1)))
        let crossEntropy = -(picked * valid).sum() / count
        let probabilities = exp(logProbabilities)
        let oneHot = (MLXArray((0 ..< options).map(Int32.init)).reshaped([1, options]) .== index).asType(.float32)
        let squared = which(mask, (probabilities - oneHot) * (probabilities - oneHot), MLXArray(Float(0)))
        let brier = (squared.sum(axis: -1) * valid).sum() / count
        return (crossEntropy + brierWeight * brier, crossEntropy, brier)
    }
}

extension NFKMLXOpenJevDeBERTa {

    /// Freezes what a policy leaves alone.
    public static func freeze(_ net: NFKMLXOpenJevDeBERTaNet, trainable: NFKMLXOpenJevDeBERTaTrainable) {
        net.freeze()
        for layer in net.head { layer.unfreeze() }
        if trainable == .all {
            net.backbone.unfreeze()
        }
    }

    /// The batched inputs and golds a set of examples makes.
    func trainingBatch(_ examples: [NFKMLXOpenJevDeBERTaExample]) throws
        -> (tokens: MLXArray, attentionMask: MLXArray?, pooling: MLXArray, optionMask: MLXArray, gold: MLXArray) {
        let prompts = try examples.map { try prompt(state: $0.state, questions: $0.questions) }
        let inputs = NFKMLXOpenJevDeBERTaNet.inputs(prompts, padToken: configuration.padToken)
        let questions = inputs.optionMask.shape[1]
        var gold = [Int32](repeating: -1, count: examples.count * questions)
        for (b, example) in examples.enumerated() {
            for (q, label) in example.labels.enumerated() { gold[b * questions + q] = Int32(label) }
        }
        return (inputs.tokens, inputs.attentionMask, inputs.pooling, inputs.optionMask,
                MLXArray(gold, [examples.count, questions]))
    }

    /// Fine-tunes the model on labeled states, `batchSize` examples a step in order, cycling.
    ///
    /// - Parameters:
    ///   - examples: labeled states; each label must index an option of its question.
    ///   - steps: how many optimizer steps to take.
    ///   - learningRate: the encoder's AdamW rate; the release trains at 3e-5.
    ///   - headLearningRate: the head's AdamW rate; the release trains at 1e-3, because the head starts
    ///     from scratch and moves too little at the encoder's rate.
    ///   - trainable: which parameters move.
    ///   - objective: the release's objective.
    ///   - batchSize: examples per step, padded together.
    ///   - learningRateSchedule: multiplies both rates at each step. Nil is the release's: a linear
    ///     warm-up over the first 6% of the run, then a linear decay to zero
    ///     (``NFKMLXLearningRateSchedule/openJevDeBERTa(steps:)``); `.constant` opts out.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Both parameter groups are PyTorch's AdamW (bias-corrected) with weight decay 0.01, as the
    /// release's `train_encoder.py` builds them, and the global gradient norm is clipped to 1. The
    /// encoder's dropout is not applied, so a step is deterministic. The tuned network saves through
    /// `NFKMLXWeights.save` and reloads through `openJev(directoryURL:weightsURL:)`. A run is
    /// multi-second; call it off the main thread.
    @discardableResult
    public func fineTune(examples: [NFKMLXOpenJevDeBERTaExample], steps: Int, learningRate: Float = 3e-5,
                         headLearningRate: Float = 1e-3, trainable: NFKMLXOpenJevDeBERTaTrainable = .head,
                         objective: NFKMLXOpenJevDeBERTaObjective = NFKMLXOpenJevDeBERTaObjective(),
                         batchSize: Int = 1, learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
                         observer: NFKMLXTrainer.Observer? = nil) throws -> [Float] {
        guard tokenizer != nil else { throw NFKMLXError.trainingDataMismatch("the model has no tokenizer to encode the examples with") }
        guard !examples.isEmpty, batchSize > 0 else { throw NFKMLXError.trainingDataMismatch("no examples") }
        for example in examples {
            guard example.labels.count == example.questions.count,
                  zip(example.labels, example.questions).allSatisfy({ (0 ..< $1.renderedOptions.count).contains($0) }) else {
                throw NFKMLXError.trainingDataMismatch("every question needs one label that indexes one of its options")
            }
        }
        let batches = try stride(from: 0, to: examples.count, by: batchSize).map {
            try trainingBatch(Array(examples[$0 ..< Swift.min($0 + batchSize, examples.count)]))
        }
        Self.freeze(net, trainable: trainable)
        let optimizer = NFKMLXReferenceOptimizers.adamW(learningRate: learningRate, over: net) { key in
            (rateScale: key.hasPrefix("head.") ? headLearningRate / learningRate : 1, weightDecay: 0.01)
        }
        var current = 0
        return try NFKMLXTrainer.train(
            net, optimizer: optimizer, steps: steps,
            sample: { step in current = step % batches.count; return MLXArray(Int32(current)) },
            loss: { model, _ in
                let batch = batches[current]
                let logits = model.logits(tokens: batch.tokens, attentionMask: batch.attentionMask,
                                          pooling: batch.pooling, optionMask: batch.optionMask)
                return objective.loss(logits: logits, gold: batch.gold, optionMask: batch.optionMask)
            },
            clipGradientNorm: 1,
            learningRateSchedule: learningRateSchedule ?? .openJevDeBERTa(steps: steps),
            observer: observer)
    }
}

extension NFKMLXLearningRateSchedule {
    /// The release's `train_encoder.py` schedule, a `LambdaLR` by optimizer step: a linear warm-up to
    /// the base rate over `max(1, ⌊0.06 · steps⌋)` steps, then a linear decay that reaches zero at the
    /// end of the run.
    public static func openJevDeBERTa(steps: Int) -> NFKMLXLearningRateSchedule {
        let warmup = Swift.max(1, Int(Double(steps) * 0.06))
        return NFKMLXLearningRateSchedule { step in
            guard step >= warmup else { return Float(step + 1) / Float(warmup) }
            return Swift.max(0, Float(steps - step) / Float(Swift.max(1, steps - warmup)))
        }
    }
}
