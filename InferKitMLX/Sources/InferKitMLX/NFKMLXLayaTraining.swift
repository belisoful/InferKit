//
//  NFKMLXLayaTraining.swift
//  InferKitMLX
//
//  Customizing Laya on a consumer's own decisions. The release's own README says what fine-tuning
//  is for: the base checkpoints sit near chance on a new decision task, and the capability comes from
//  training on that task's examples. The objective is the reference's strictly proper scoring rule
//  (`rl_common.proper_reward`): the log score plus half the spherical score for every type, minus the
//  ranked probability score for an ordinal question, so the reward is maximized only by reporting
//  honest probabilities. The reference optimizes it by REINFORCE over noised logits with a group-mean
//  baseline; that update rule is described and not published, so the recipe here takes the gradient of
//  the expected reward under the reported distribution directly, which has the same optimum.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a fine-tune moves.
public enum NFKMLXLayaTrainable: Sendable {
    /// The decision head, the type embedding, and the scorer; the encoder stays as released. The
    /// cheapest run, and the one a device holds comfortably.
    case head
    /// The encoder too, which is how the reference trains. The act head and the temperature buffer
    /// stay frozen under either policy: the act head's cost-weighted objective is not published.
    case all
}

/// One labeled decision: a state, a question about it, and the target distribution over the
/// question's rendered options.
public struct NFKMLXLayaExample {
    public var state: Any
    public var question: NFKDecisionQuestion
    /// The target over the rendered options, in option order: one-hot for a labeled answer, or soft.
    /// A noul's target is `[1 - y, y]`.
    public var target: [Float]

    public init(state: Any, question: NFKDecisionQuestion, target: [Float]) {
        self.state = state
        self.question = question
        self.target = target
    }

    /// A one-hot example from the index of the right option (a choice's option, a score's level).
    public init(state: Any, question: NFKDecisionQuestion, label: Int) {
        let count = NFKMLXLayaPrompt.renderedOptions(for: question).count
        var target = [Float](repeating: 0, count: count)
        if label >= 0, label < count { target[label] = 1 }
        self.init(state: state, question: question, target: target)
    }

    /// A noul example from whether the statement holds.
    public init(state: Any, question: NFKDecisionQuestion, holds: Bool) {
        self.init(state: state, question: question, target: holds ? [0, 1] : [1, 0])
    }
}

/// One labeled conversation: a context every prefix carries, the turns oldest first, a noul asked of
/// each prefix, and whether the statement held once the conversation ended.
///
/// @discussion The reference trains a conversation as its prefixes, so the model learns to judge the
/// outcome early. The prefixes are `NFKMLXLayaPrompt.prefixLengths(turnCount:maxPrefixes:)`, each
/// serialized as the context's fields followed by the turns so far under `conversation` and cut from
/// the left so the newest turns survive. Their targets are TD(λ) blends of the outcome and the model's
/// own prediction on the next prefix (``NFKMLXLayaObjective/temporalDifferenceTargets(outcome:nextTrueProbabilities:lambda:)``).
public struct NFKMLXLayaEpisode {
    /// The fields every prefix carries beside the conversation: the account, the channel, the plan.
    public var context: [String: Any]
    /// The turns, oldest first: strings, or JSON-serializable records such as `{role, text}`.
    public var turns: [Any]
    /// The noul asked of every prefix.
    public var question: NFKDecisionQuestion
    /// Whether the statement held at the end of the conversation.
    public var holds: Bool

    public init(context: [String: Any], turns: [Any], question: NFKDecisionQuestion, holds: Bool) {
        self.context = context
        self.turns = turns
        self.question = question
        self.holds = holds
    }
}

/// The reference's proper-scoring objective over a reported distribution.
public struct NFKMLXLayaObjective: Sendable {
    /// The weight of the spherical score beside the log score.
    public var sphericalWeight: Float
    /// The weight of the ranked probability score an ordinal (score) question subtracts.
    public var rankedProbabilityWeight: Float
    /// The floor a log probability is clamped to, `log(1e-4)`.
    public var logFloor: Float

    public init(sphericalWeight: Float = 0.5, rankedProbabilityWeight: Float = 1.0, logFloor: Float = -9.21) {
        self.sphericalWeight = sphericalWeight
        self.rankedProbabilityWeight = rankedProbabilityWeight
        self.logFloor = logFloor
    }

    /// The reward of each row: `[rows]` from logits `[rows, options]`, targets of the same shape, a
    /// mask of the same shape (true at a real option), and a type per row (0 choice, 1 score, 2 noul).
    /// Masked logits are pushed to `-1e4` before the softmax, as the reference pushes them.
    public func reward(logits: MLXArray, targets: MLXArray, mask: MLXArray, types: MLXArray) -> MLXArray {
        let masked = MLX.where(mask, logits, MLXArray(Float(-1e4)))
        let q = softmax(masked, axis: -1) * mask.asType(.float32)
        let logQ = maximum(log(maximum(q, 1e-12)), MLXArray(logFloor))
        let logScore = (targets * logQ).sum(axis: -1)
        let norm = maximum(sqrt((q * q).sum(axis: -1)), MLXArray(Float(1e-9)))
        let spherical = (targets * q).sum(axis: -1) / norm
        var reward = logScore + sphericalWeight * spherical

        let isScore = (types .== MLXArray(Int32(NFKDecisionType.score.rawValue))).asType(.float32)
        let count = maximum(mask.asType(.float32).sum(axis: -1), MLXArray(Float(2)))
        let cumulativeQ = cumsum(q, axis: -1)
        let cumulativeTarget = cumsum(targets, axis: -1)
        let ranked = (((cumulativeQ - cumulativeTarget) * (cumulativeQ - cumulativeTarget)) * mask.asType(.float32)).sum(axis: -1) / (count - 1)
        reward = reward - rankedProbabilityWeight * ranked * isScore
        return reward
    }

    /// The loss to minimize: the negative mean reward over the rows.
    public func loss(logits: MLXArray, targets: MLXArray, mask: MLXArray, types: MLXArray) -> MLXArray {
        -reward(logits: logits, targets: targets, mask: mask, types: types).mean()
    }

    /// The TD(λ) targets of an episode's prefixes, oldest first, as `[1 - G, G]` rows.
    ///
    /// @discussion The last prefix's target is the outcome. Walking back, each earlier prefix's return
    /// is `(1 - λ)` times the model's predicted true-probability on the next prefix plus `λ` times the
    /// next prefix's return, which is the reference's `td_lambda_targets`. At λ = 1, the release's own
    /// setting, every prefix is trained toward the outcome; at λ = 0, toward the next prefix's
    /// prediction alone. `nextTrueProbabilities` holds the model's prediction for each prefix in the
    /// same order; only the entries after the first are read.
    public static func temporalDifferenceTargets(outcome: Bool, nextTrueProbabilities: [Float],
                                                 lambda: Float) -> [[Float]] {
        let count = nextTrueProbabilities.count
        guard count > 0 else { return [] }
        var targets = [[Float]](repeating: [0, 0], count: count)
        var g: Float = outcome ? 1 : 0
        for index in stride(from: count - 1, through: 0, by: -1) {
            if index < count - 1 {
                g = (1 - lambda) * nextTrueProbabilities[index + 1] + lambda * g
            }
            targets[index] = [1 - g, g]
        }
        return targets
    }

    /// The loss of one question's raw logits `[options]` against its target `[options]`.
    public func loss(logits: MLXArray, target: [Float], type: NFKDecisionType) -> MLXArray {
        let count = logits.shape[0]
        return loss(logits: logits.reshaped([1, count]),
                    targets: MLXArray(target).reshaped([1, count]),
                    mask: MLXArray.ones([1, count]).asType(.bool),
                    types: MLXArray([Int32(type.rawValue)]))
    }
}

extension NFKMLXLaya {

    /// Freezes the parameters a policy leaves alone. The temperature buffer and the act head are
    /// always frozen.
    public static func freeze(_ net: NFKMLXLayaNet, trainable: NFKMLXLayaTrainable) {
        net.freeze()
        net.head.unfreeze()
        net.typeEmbedding.unfreeze()
        for layer in net.scorer { layer.unfreeze() }
        if trainable == .all {
            net.encoder.unfreeze()
        }
    }

    /// Fine-tunes the model on the examples, one example per step in order, cycling.
    ///
    /// - Parameters:
    ///   - examples: the labeled decisions. Each is encoded once, through this model's tokenizer.
    ///   - steps: how many optimizer steps to take.
    ///   - learningRate: the learning rate of a bias-corrected Adam, PyTorch's. The reference publishes
    ///     neither its optimizer nor a value; 1e-4 suits a head run, and a full run wants an order of
    ///     magnitude less.
    ///   - trainable: which parameters move.
    ///   - objective: the proper-scoring objective.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the main thread. The fine-tuned network saves through
    /// `NFKMLXWeights.save` and reloads through `laya(directoryURL:weightsURL:)`.
    @discardableResult
    public func fineTune(examples: [NFKMLXLayaExample], steps: Int, learningRate: Float = 1e-4,
                         trainable: NFKMLXLayaTrainable = .head,
                         objective: NFKMLXLayaObjective = NFKMLXLayaObjective(),
                         observer: NFKMLXTrainer.Observer? = nil) throws -> [Float] {
        guard let tokenizer else { throw NFKMLXError.trainingDataMismatch("the model has no tokenizer to encode the examples with") }
        guard !examples.isEmpty else { throw NFKMLXError.trainingDataMismatch("no examples") }
        let encoded = try examples.map { example -> (prompt: NFKMLXLayaPrompt, target: [Float], type: NFKDecisionType) in
            let prompt = NFKMLXLayaPrompt(state: example.state, question: example.question,
                                          tokenizer: tokenizer, configuration: configuration)
            guard prompt.markers.count == example.target.count else {
                throw NFKMLXError.trainingDataMismatch(
                    "a target of \(example.target.count) entries does not match the question's \(prompt.markers.count) options that fit")
            }
            return (prompt, example.target, example.question.type)
        }
        var current = 0
        return try NFKMLXFineTune.run(
            net,
            freezing: { Self.freeze(net, trainable: trainable) },
            optimizer: nil,
            reference: { Adam(learningRate: learningRate, biasCorrection: true) },
            referenceSchedule: { .constant },
            steps: steps,
            sample: { step in current = step % encoded.count; return MLXArray(Int32(current)) },
            loss: { model, _ in
                let example = encoded[current]
                let (logits, _) = model.forward(tokens: example.prompt.tokens, markers: example.prompt.markers, type: example.type)
                return objective.loss(logits: logits, target: example.target, type: example.type)
            },
            clipGradientNorm: 1, observer: observer)
    }

    /// The prompts an episode's prefixes build over this model's tokenizer, oldest first.
    public func prefixes(of episode: NFKMLXLayaEpisode) -> [(length: Int, prompt: NFKMLXLayaPrompt)] {
        guard let tokenizer else { return [] }
        return NFKMLXLayaPrompt.prefixes(context: episode.context, turns: episode.turns, question: episode.question,
                                         tokenizer: tokenizer, configuration: configuration)
    }

    /// Fine-tunes the model on conversations, one episode per step in order, cycling. Every prefix of
    /// the episode is scored in the step, and its TD(λ) target is built from the model's own
    /// predictions on the later prefixes, which are treated as constants.
    ///
    /// - Parameters:
    ///   - episodes: the labeled conversations. Each needs a noul question and at least one prefix
    ///     whose two markers fit the budget.
    ///   - steps: how many optimizer steps to take.
    ///   - learningRate: the learning rate of a bias-corrected Adam, PyTorch's.
    ///   - trainable: which parameters move.
    ///   - lambda: the TD(λ) weight; 1 is the release's own setting.
    ///   - objective: the proper-scoring objective.
    ///   - observer: receives each step and can end the run early.
    @discardableResult
    public func fineTune(episodes: [NFKMLXLayaEpisode], steps: Int, learningRate: Float = 1e-4,
                         trainable: NFKMLXLayaTrainable = .head, lambda: Float = 1,
                         objective: NFKMLXLayaObjective = NFKMLXLayaObjective(),
                         observer: NFKMLXTrainer.Observer? = nil) throws -> [Float] {
        guard tokenizer != nil else { throw NFKMLXError.trainingDataMismatch("the model has no tokenizer to encode the episodes with") }
        guard !episodes.isEmpty else { throw NFKMLXError.trainingDataMismatch("no episodes") }
        let encoded = try episodes.map { episode -> (prompts: [NFKMLXLayaPrompt], holds: Bool) in
            guard episode.question.type == .noul else {
                throw NFKMLXError.trainingDataMismatch("an episode's question is a noul; a \(NFKDecisionQuestion.name(for: episode.question.type)) cannot take a TD target")
            }
            let prompts = prefixes(of: episode).map(\.prompt)
            guard !prompts.isEmpty else {
                throw NFKMLXError.trainingDataMismatch("no prefix of an episode fits the sequence budget")
            }
            return (prompts, episode.holds)
        }
        var current = 0
        return try NFKMLXFineTune.run(
            net,
            freezing: { Self.freeze(net, trainable: trainable) },
            optimizer: nil,
            reference: { Adam(learningRate: learningRate, biasCorrection: true) },
            referenceSchedule: { .constant },
            steps: steps,
            sample: { step in current = step % encoded.count; return MLXArray(Int32(current)) },
            loss: { model, _ in
                let episode = encoded[current]
                let logits = episode.prompts.map { model.forward(tokens: $0.tokens, markers: $0.markers, type: .noul).logits }
                let stacked = stacked(logits)                                              // [prefixes, 2]
                let predicted = stopGradient(softmax(stacked, axis: -1))[0..., 1]
                eval(predicted)
                let targets = NFKMLXLayaObjective.temporalDifferenceTargets(outcome: episode.holds,
                                                             nextTrueProbabilities: predicted.asArray(Float.self),
                                                             lambda: lambda)
                let count = episode.prompts.count
                return objective.loss(logits: stacked,
                                      targets: MLXArray(targets.flatMap { $0 }).reshaped([count, 2]),
                                      mask: MLXArray.ones([count, 2]).asType(.bool),
                                      types: MLXArray([Int32](repeating: Int32(NFKDecisionType.noul.rawValue), count: count)))
            },
            clipGradientNorm: 1, observer: observer)
    }
}
