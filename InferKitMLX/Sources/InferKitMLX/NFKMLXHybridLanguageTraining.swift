//
//  NFKMLXHybridLanguageTraining.swift
//  InferKitMLX
//
//  Adapting a Qwen3.5 hybrid decoder to a consumer's own text with LoRA, under the causal language-model
//  loss transformers computes from `labels=` (`NFKMLXCausalLanguageObjective`). Three layers in four are
//  a gated delta-rule recurrence, so LoRA adapts the recurrence's input and output projections beside
//  the attention projections: the target set of the LoRA adapter released for this decoder (Open-Jev on
//  Qwen3.5-2B), since PEFT names no default for `qwen3_5`.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

public extension NFKMLXHybridLanguage {

    /// Builds a hybrid decoder for fine-tuning or for reloading a fine-tuned checkpoint. With a
    /// `weightsURL` the single-file checkpoint loads at float32; without one the net is randomly
    /// initialized. Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?, configuration: NFKMLXHybridConfiguration) throws -> NFKMLXHybridLanguageNet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Loads one checkpoint file `NFKMLXWeights` wrote from this module, at float32. The file carries the
    /// module's own keys and its depthwise convolution already transposed, unlike a release directory,
    /// which ``loadWeights(into:fromDirectory:precision:)`` reads. Introduced in InferKit 0.4.0.
    static func loadWeights(into net: NFKMLXHybridLanguageNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { ($0.key, $0.value.dtype.isFloatingPoint ? $0.value.asType(.float32) : $0.value) }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    /// A text-generation backend over `network`, a decoder this package built and the caller adapted,
    /// with the tokenizer, chat template, and stop ids of the release directory it came from.
    /// Introduced in InferKit 0.4.0.
    static func backend(network: NFKMLXHybridLanguageNet, directoryURL: URL) throws -> any NFKInferenceBackend {
        try NFKMLXDecoderBackend.release(directoryURL: directoryURL, identifier: modelName,
                                         modules: [network]) { network($0) }
    }

    /// The projections LoRA adapts: every attention projection and the recurrence's fused input and
    /// output projections, the target set of the adapter released for this decoder.
    internal static let loraTargets: Set<String> = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]

    /// Whether `path` names a projection in ``loraTargets``.
    internal static func isAdaptedProjection(_ path: String) -> Bool {
        guard let name = path.split(separator: ".").last else {
            return false
        }
        return loraTargets.contains(String(name))
    }

    /// Adapts a hybrid decoder with LoRA and trains it on token sequences.
    ///
    /// - Parameters:
    ///   - net: the decoder, from ``network(directoryURL:precision:)`` at `.float32`.
    ///   - examples: supplies one example per step: the token ids (from
    ///     `NFKMLXLanguage.tokenizer(directoryURL:)`, or a chat rendered through the release's template)
    ///     and how many leading tokens are prompt. The prompt is context and is not scored; 0 scores
    ///     the whole sequence.
    ///   - rank: the LoRA width. Nil trains every parameter.
    ///   - alpha: the adapter's strength, applied as `alpha / rank`.
    ///   - loraDropout: the dropout on each adapter's input while training, PEFT's `lora_dropout`. 0 by
    ///     default, PEFT's default.
    ///   - objective: the causal language-model loss.
    ///   - optimizer: the update rule. Nil uses the optimizer transformers' `Trainer` defaults to,
    ///     `torch.optim.AdamW` (bias-corrected) with no weight decay; the releases publish no training
    ///     script beyond the model's `labels=` loss, and the learning rate is this package's choice.
    ///   - steps: how many updates to train for.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - accumulationSteps: how many examples each update averages; `steps` counts updates. 1, the
    ///     default, updates after every example. transformers' `Trainer` updates on 8 examples by
    ///     default (``NFKMLXFineTune/transformersTrainerBatchSize``).
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is the constant rate.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The embeddings, the feed-forwards, the norms, and the recurrence's decay and gate projections
    /// stay frozen. Call `NFKMLXLoRA.merge(into:)` before saving, so the result is one checkpoint
    /// ``network(weightsURL:configuration:)`` reads back; or wrap the adapted decoder with
    /// ``backend(network:directoryURL:)``. A run is minutes; call it off the render thread.
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXHybridLanguageNet,
        examples: (Int) -> (tokens: MLXArray, promptLength: Int),
        rank: Int? = 8,
        alpha: Float = 16,
        loraDropout: Float = 0,
        objective: NFKMLXCausalLanguageObjective = NFKMLXCausalLanguageObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: {
                guard let rank else {
                    return
                }
                let adapted = try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha, dropout: loraDropout) { path, _ in
                    isAdaptedProjection(path)
                }
                guard adapted > 0 else {
                    throw NFKMLXError.trainingDataMismatch("no projections were found to adapt, so nothing would train")
                }
            },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: 1e-4, weightDecay: 0) },
            referenceSchedule: { .constant },
            steps: steps,
            batch: { let example = examples($0); return (example.tokens, MLXArray(Int32(example.promptLength))) },
            loss: { net, tokens, promptLength in
                objective.loss(logits: net(tokens.reshaped([1, tokens.shape[0]])), tokens: tokens, scoredFrom: promptLength)
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
