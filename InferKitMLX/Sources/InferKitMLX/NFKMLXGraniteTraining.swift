//
//  NFKMLXGraniteTraining.swift
//  InferKitMLX
//
//  Adapting a Granite 4.0-H decoder to a consumer's own text with LoRA. The objective is causal
//  language-model teacher forcing, as the reference trains: every position is scored on the next
//  token, so one forward pass scores a whole sequence. LoRA adapts the attention query and value
//  projections (the layers Granite gives attention rather than a Mamba scan); everything else freezes.
//
//  This is the first language-decoder fine-tune in the package. The net builder, the single-file
//  loader, and the `fineTune` recipe are the customization path the parity rule requires.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The token-level cross-entropy a Granite 4.0-H fine-tune minimizes: each position predicts the next
/// token, matching `GraniteMoeHybridForCausalLM`'s shifted loss.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXGraniteObjective: Sendable {
    /// Smooths the target distribution.
    public var labelSmoothing: Float

    public init(labelSmoothing: Float = 0) {
        self.labelSmoothing = labelSmoothing
    }

    /// Scores a Granite network on one token sequence `[T]`: the decoder reads the whole sequence and
    /// every position is scored on the following token.
    public func callAsFunction(_ net: NFKMLXGraniteHybridNet, _ tokens: MLXArray) -> MLXArray {
        let logits = net(tokens.reshaped([1, tokens.shape[0]]))
        return loss(logits: logits, tokens: tokens)
    }

    /// Scores decoder logits `[1, T, vocabulary]` against the token sequence `[T]` directly, without a
    /// forward pass, so the objective can be compared against the reference on identical logits. The
    /// shift matches the reference: position `t` predicts token `t + 1`, averaged over the `T - 1`
    /// scored positions.
    public func loss(logits: MLXArray, tokens: MLXArray) -> MLXArray {
        let length = tokens.shape[0]
        guard length > 1 else { return MLXArray(Float(0)) }
        let vocabulary = logits.shape[2]
        let predictions = logits[0, 0 ..< (length - 1), 0...].reshaped([length - 1, vocabulary]).asType(.float32)
        let targets = tokens[1 ..< length]
        return crossEntropy(logits: predictions, targets: targets, labelSmoothing: labelSmoothing, reduction: .mean)
    }
}

public extension NFKMLXGraniteHybrid {

    /// Builds a Granite 4.0-H network for fine-tuning or reloading a fine-tuned checkpoint. With a
    /// `weightsURL` the single-file checkpoint loads; without one the net is randomly initialized.
    static func network(weightsURL: URL?, configuration: NFKMLXGraniteHybridConfiguration) throws
        -> NFKMLXGraniteHybridNet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Loads one checkpoint file, released or fine-tuned, at float32. A fine-tuned file `NFKMLXWeights`
    /// wrote already carries this module's own layout (the depthwise convolution squeezed), so it loads
    /// straight, unlike the release directory whose convolution is `[C, 1, K]`.
    static func loadWeights(into net: NFKMLXGraniteHybridNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { ($0.key, $0.value.asType(.float32)) }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    /// The projections LoRA adapts: the query and value of every attention layer. Granite gives only a
    /// minority of layers attention (the rest are Mamba scans), so this is the reference's attention
    /// LoRA target set restricted to the layers that carry attention. `NFKMLXLoRA` reaches this sparse
    /// subset of the layer array through its per-owner fallback.
    static func isAttentionProjection(_ path: String) -> Bool {
        path.contains("self_attn") && (path.hasSuffix(".q_proj") || path.hasSuffix(".v_proj"))
    }

    /// Adapts a Granite 4.0-H network's attention with LoRA and trains it on token sequences.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:configuration:)``.
    ///   - examples: supplies one token sequence per step (the consumer's own text as ids).
    ///   - rank: the LoRA width. Nil trains every parameter.
    ///   - alpha: the adapter's strength, applied as `alpha / rank`.
    ///   - objective: the causal language-model loss.
    ///   - optimizer: the update rule. Nil uses the optimizer transformers' `Trainer` defaults to,
    ///     `torch.optim.AdamW` (bias-corrected) with no weight decay; the reference publishes no training
    ///     script beyond its model's `labels=` loss, and the learning rate is this package's choice.
    ///   - steps: how many sequences to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Only the attention query and value projections adapt; the Mamba layers, the embeddings, and the
    /// feed-forward stay frozen. Call `NFKMLXLoRA.merge(into:)` before saving, so the result is one
    /// checkpoint ``loadWeights(into:from:)`` reads back. A run is minutes; call it off the render thread.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXGraniteHybridNet,
        examples: (Int) -> MLXArray,
        rank: Int? = 8,
        alpha: Float = 16,
        objective: NFKMLXGraniteObjective = NFKMLXGraniteObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: {
                guard let rank else {
                    return
                }
                let adapted = try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha) { path, _ in
                    isAttentionProjection(path)
                }
                guard adapted > 0 else {
                    throw NFKMLXError.trainingDataMismatch("no attention projections were found to adapt, so nothing would train")
                }
            },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: 1e-4, weightDecay: 0) },
            referenceSchedule: { .constant },
            steps: steps,
            sample: examples, loss: objective.callAsFunction,
            clipGradientNorm: clipGradientNorm, checkpoint: checkpoint, observer: observer)
    }
}
