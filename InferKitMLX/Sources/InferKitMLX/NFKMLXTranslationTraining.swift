//
//  NFKMLXTranslationTraining.swift
//  InferKitMLX
//
//  Adapting a translator to a consumer's own domain: their terminology, their register, their
//  language pair. The objective is teacher forcing, as the reference trains: the decoder reads the
//  target shifted right behind the start token and every position is scored on the target token at
//  that position, so one step is one forward pass.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The token-level cross-entropy a translation fine-tune minimizes.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTranslationObjective: Sendable {
    /// Smooths the target distribution.
    public var labelSmoothing: Float

    public init(labelSmoothing: Float = 0) {
        self.labelSmoothing = labelSmoothing
    }

    /// The decoder input for `target`: the start token followed by the target without its last token.
    public static func decoderInput(for target: MLXArray, startToken: Int) -> MLXArray {
        let length = target.shape[0]
        guard length > 1 else { return MLXArray([Int32(startToken)]) }
        return concatenated([MLXArray([Int32(startToken)]), target[0 ..< (length - 1)]])
    }

    /// Scores a BART-family network on one sentence pair: source ids `[S]` and target ids `[T]`
    /// (the target ends with the end token, as the tokenizer emits it).
    public func callAsFunction(_ net: NFKMLXSeq2SeqNet, _ source: MLXArray, _ target: MLXArray) -> MLXArray {
        let input = Self.decoderInput(for: target, startToken: net.configuration.decoderStartTokenId)
        let logits = net(source: source.reshaped([1, source.shape[0]]), target: input.reshaped([1, input.shape[0]]))
        return loss(logits: logits, target: target)
    }

    /// Scores a T5 network on one sentence pair.
    public func callAsFunction(_ net: NFKMLXT5Seq2SeqNet, _ source: MLXArray, _ target: MLXArray) -> MLXArray {
        let input = Self.decoderInput(for: target, startToken: net.configuration.decoderStartTokenId)
        let logits = net(source: source.reshaped([1, source.shape[0]]), target: input.reshaped([1, input.shape[0]]))
        return loss(logits: logits, target: target)
    }

    /// Scores decoder logits `[1, T, vocabulary]` against the target `[T]` directly, without a
    /// forward pass, so the objective can be compared against the reference on identical logits.
    public func loss(logits: MLXArray, target: MLXArray) -> MLXArray {
        let length = target.shape[0]
        guard length > 0 else { return MLXArray(Float(0)) }
        let vocabulary = logits.shape[2]
        let predictions = logits[0, 0 ..< length, 0...].reshaped([length, vocabulary]).asType(.float32)
        return crossEntropy(logits: predictions, targets: target, labelSmoothing: labelSmoothing, reduction: .mean)
    }
}

/// The LoRA recipe the three translators share.
enum NFKMLXTranslationTraining {
    /// The projections LoRA targets in a BART-family decoder: query and value of both attentions.
    static func isSeq2SeqDecoderProjection(_ path: String) -> Bool {
        path.hasPrefix("decoder.layers.") && (path.hasSuffix(".q_proj") || path.hasSuffix(".v_proj"))
    }

    /// The projections LoRA targets in a T5 decoder: `q` and `v` of both attentions.
    static func isT5DecoderProjection(_ path: String) -> Bool {
        path.hasPrefix("decoder.block.") && (path.hasSuffix(".q") || path.hasSuffix(".v"))
    }

    static func fineTune<Net: Module>(
        _ net: Net, adapting predicate: @escaping (String) -> Bool,
        examples: (Int) -> (source: MLXArray, target: MLXArray), rank: Int?, alpha: Float,
        loss: @escaping (Net, MLXArray, MLXArray) -> MLXArray, optimizer: Optimizer?, steps: Int,
        clipGradientNorm: Float?, checkpoint: NFKMLXTrainingCheckpoint?, observer: NFKMLXTrainer.Observer?
    ) throws -> [Float] {
        if let rank {
            let adapted = try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha) { path, _ in predicate(path) }
            guard adapted > 0 else {
                throw NFKMLXError.trainingDataMismatch("no decoder attention projections were found to adapt, so nothing would train")
            }
        }
        return try NFKMLXTrainer.train(
            net, optimizer: optimizer ?? NFKMLXReferenceOptimizers.adamW(learningRate: 1e-4, weightDecay: 0), steps: steps,
            batch: { let example = examples($0); return (example.source, example.target) },
            loss: loss, clipGradientNorm: clipGradientNorm, checkpoint: checkpoint, observer: observer)
    }
}

extension NFKMLXMarian {
    /// Adapts a Marian network's decoder attention with LoRA and trains it on sentence pairs.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(directoryURL:configuration:)``.
    ///   - examples: supplies one pair per step: source ids from
    ///     ``NFKMLXMarianTranslator/sourceIds(for:target:)`` and target ids from the target tokenizer
    ///     with the end token appended.
    ///   - rank: the LoRA width. Nil trains every parameter.
    ///   - alpha: the adapter's strength, applied as `alpha / rank`.
    ///   - objective: the token loss.
    ///   - optimizer: the update rule. Nil uses the optimizer transformers' `Trainer` defaults to,
    ///     `torch.optim.AdamW` (bias-corrected) with no weight decay; the reference publishes no training
    ///     script beyond its model's `labels=` loss, and the learning rate is this package's choice.
    ///   - steps: how many pairs to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Only the decoder's query and value projections adapt; the encoder stays frozen. Call
    /// `NFKMLXLoRA.merge(into:)` before saving, so the result is one checkpoint
    /// ``NFKMLXSeq2SeqNet/loadWeights(from:)`` reads back. A run is minutes; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXSeq2SeqNet,
        examples: (Int) -> (source: MLXArray, target: MLXArray),
        rank: Int? = 8,
        alpha: Float = 16,
        objective: NFKMLXTranslationObjective = NFKMLXTranslationObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXTranslationTraining.fineTune(
            net, adapting: NFKMLXTranslationTraining.isSeq2SeqDecoderProjection, examples: examples,
            rank: rank, alpha: alpha, loss: objective.callAsFunction, optimizer: optimizer, steps: steps,
            clipGradientNorm: clipGradientNorm, checkpoint: checkpoint, observer: observer)
    }
}

extension NFKMLXM2M100 {
    /// Adapts an M2M-100 network the way ``NFKMLXMarian/fineTune(_:examples:rank:alpha:objective:optimizer:steps:clipGradientNorm:checkpoint:observer:)``
    /// adapts Marian: source ids from ``NFKMLXM2M100Translator/sourceIds(for:source:target:)``, target
    /// ids as the target marker, the pieces, and the end token.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXSeq2SeqNet,
        examples: (Int) -> (source: MLXArray, target: MLXArray),
        rank: Int? = 8,
        alpha: Float = 16,
        objective: NFKMLXTranslationObjective = NFKMLXTranslationObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXTranslationTraining.fineTune(
            net, adapting: NFKMLXTranslationTraining.isSeq2SeqDecoderProjection, examples: examples,
            rank: rank, alpha: alpha, loss: objective.callAsFunction, optimizer: optimizer, steps: steps,
            clipGradientNorm: clipGradientNorm, checkpoint: checkpoint, observer: observer)
    }
}

extension NFKMLXMADLAD {
    /// Adapts a MADLAD network's decoder `q` and `v` projections with LoRA: source ids from
    /// ``NFKMLXMADLADTranslator/sourceIds(for:target:)``, target ids as the pieces and the end token.
    /// Adapt a float32 load; a `half` load holds bfloat16 weights that LoRA does not train.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXT5Seq2SeqNet,
        examples: (Int) -> (source: MLXArray, target: MLXArray),
        rank: Int? = 8,
        alpha: Float = 16,
        objective: NFKMLXTranslationObjective = NFKMLXTranslationObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXTranslationTraining.fineTune(
            net, adapting: NFKMLXTranslationTraining.isT5DecoderProjection, examples: examples,
            rank: rank, alpha: alpha, loss: objective.callAsFunction, optimizer: optimizer, steps: steps,
            clipGradientNorm: clipGradientNorm, checkpoint: checkpoint, observer: observer)
    }
}
