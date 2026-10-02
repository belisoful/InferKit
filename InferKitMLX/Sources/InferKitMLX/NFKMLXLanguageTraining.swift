//
//  NFKMLXLanguageTraining.swift
//  InferKitMLX
//
//  Adapting a dense decoder (Qwen3, Qwen2.5, Llama, Mistral) to a consumer's own text with LoRA. The
//  objective is the causal language-model loss transformers computes from `labels=`: every position is
//  scored on the next token, and a prompt's positions can be left unscored, as `labels=-100` leaves
//  them. LoRA adapts the attention query and value projections, PEFT's default target set for these
//  architectures; everything else freezes.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/// The token-level cross-entropy a decoder fine-tune minimizes: each position predicts the next token,
/// matching the shifted loss transformers' `*ForCausalLM` computes from `labels=`.
///
/// @discussion A prompt's tokens are context and are not scored: an example of `promptLength` P scores
/// the tokens from position P on, which is the reference with the first P labels set to -100. The loss
/// is the mean over the scored tokens. A P of 0 or 1 scores every token after the first.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXCausalLanguageObjective: Sendable {
    /// Smooths the target distribution.
    public var labelSmoothing: Float

    public init(labelSmoothing: Float = 0) {
        self.labelSmoothing = labelSmoothing
    }

    /// Scores a dense decoder on one token sequence `[T]`, the first `promptLength` tokens unscored.
    public func callAsFunction(_ net: NFKMLXLanguageNet, _ tokens: MLXArray, promptLength: Int = 0) -> MLXArray {
        loss(logits: net(tokens.reshaped([1, tokens.shape[0]])), tokens: tokens, promptLength: promptLength)
    }

    /// Scores a hybrid decoder on one token sequence `[T]`, the first `promptLength` tokens unscored.
    public func callAsFunction(_ net: NFKMLXHybridLanguageNet, _ tokens: MLXArray, promptLength: Int = 0) -> MLXArray {
        loss(logits: net(tokens.reshaped([1, tokens.shape[0]])), tokens: tokens, promptLength: promptLength)
    }

    /// Scores a Gemma 3 decoder on one token sequence `[T]`, the first `promptLength` tokens unscored.
    public func callAsFunction(_ net: NFKMLXGemma3Net, _ tokens: MLXArray, promptLength: Int = 0) -> MLXArray {
        loss(logits: net(tokens.reshaped([1, tokens.shape[0]])), tokens: tokens, promptLength: promptLength)
    }

    /// Scores decoder logits `[1, T, vocabulary]` against the token sequence `[T]` directly, without a
    /// forward pass, so the objective can be compared against the reference on identical logits.
    /// Position `t` predicts token `t + 1`; a token before `promptLength` is not scored.
    public func loss(logits: MLXArray, tokens: MLXArray, promptLength: Int = 0) -> MLXArray {
        loss(logits: logits, tokens: tokens, scoredFrom: MLXArray(Int32(promptLength)))
    }

    /// The loss with the first scored position as an array, which is how the trainer carries it beside
    /// the tokens. The scored tokens are selected by a weight rather than a slice, so the selection is
    /// the same computation whatever the prompt's length.
    func loss(logits: MLXArray, tokens: MLXArray, scoredFrom: MLXArray) -> MLXArray {
        let length = tokens.shape[0]
        guard length > 1 else { return MLXArray(Float(0)) }
        let vocabulary = logits.shape[2]
        // The reference upcasts the logits before its loss, whatever the weights' precision.
        let predictions = logits[0, 0 ..< (length - 1), 0...].reshaped([length - 1, vocabulary]).asType(.float32)
        let targets = tokens[1 ..< length]
        let scored = (MLXArray(Int32(1) ..< Int32(length)) .>= scoredFrom.asType(.int32)).asType(.float32)
        let perToken = crossEntropy(logits: predictions, targets: targets, labelSmoothing: labelSmoothing, reduction: .none)
        return (perToken * scored).sum() / maximum(scored.sum(), MLXArray(Float(1)))
    }
}

public extension NFKMLXLanguage {

    /// Builds a dense decoder for fine-tuning or for reloading a fine-tuned checkpoint. With a
    /// `weightsURL` the single-file checkpoint loads at float32; without one the net is randomly
    /// initialized. Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?, configuration: NFKMLXLanguageConfiguration) throws -> NFKMLXLanguageNet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Builds a dense decoder from a release directory at the geometry its `config.json` declares.
    ///
    /// @discussion `.float32`, the default, is the precision a fine-tune adapts: a half-precision
    /// adapter cannot hold the small steps a fine-tune takes. A release at 4B is about 16 GB at float32.
    /// A mixture-of-experts release loads resident. Introduced in InferKit 0.4.0.
    static func network(directoryURL: URL, precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXLanguageNet {
        let net = makeNet(try configuration(fromHuggingFace: directoryURL.appendingPathComponent("config.json")))
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
        return net
    }

    /// The tokenizer a release directory describes, read without loading the weights, or nil where the
    /// directory has none it can read. A Qwen3.5 hybrid release reads through here too. Introduced in
    /// InferKit 0.4.0.
    @objc(tokenizerWithDirectoryURL:)
    static func tokenizer(directoryURL: URL) -> NFKTokenizer? {
        releaseTokenizer(inDirectory: directoryURL)
    }

    /// A text-generation backend over `network`, a decoder this package built and the caller adapted,
    /// with the tokenizer of the release directory it came from. Introduced in InferKit 0.4.0.
    static func backend(network: NFKMLXLanguageNet, directoryURL: URL,
                        options: NFKMLXGenerationOptions = NFKMLXGenerationOptions()) throws -> any NFKInferenceBackend {
        guard let tokenizer = releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("\(directoryURL.lastPathComponent) has no readable tokenizer")
        }
        return NFKMLXLanguageBackend(net: network, tokenizer: tokenizer, identifier: modelName, options: options)
    }

    /// The projections LoRA adapts: the query and value of every attention layer, PEFT's default
    /// target set for `qwen2`, `qwen3`, `llama`, and `mistral`.
    internal static func isAttentionProjection(_ path: String) -> Bool {
        path.contains(".self_attn.") && (path.hasSuffix(".q_proj") || path.hasSuffix(".v_proj"))
    }

    /// Adapts a dense decoder's attention with LoRA and trains it on token sequences.
    ///
    /// - Parameters:
    ///   - net: the decoder, from ``network(directoryURL:precision:)`` at float32.
    ///   - examples: supplies one example per step: the token ids (from ``tokenizer(directoryURL:)``, or
    ///     a chat rendered through the release's template) and how many leading tokens are prompt. The
    ///     prompt is context and is not scored; 0 scores the whole sequence.
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
    /// Only the attention query and value projections adapt; the embeddings, the feed-forwards, and the
    /// norms stay frozen. A mixture-of-experts decoder trains its attention the same way, and a paged
    /// one is refused because its experts are read from disk. Call `NFKMLXLoRA.merge(into:)` before
    /// saving, so the result is one checkpoint ``network(weightsURL:configuration:)`` reads back; or
    /// wrap the adapted decoder with ``backend(network:directoryURL:options:)``. A run is minutes; call
    /// it off the render thread. Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXLanguageNet,
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
        guard net.expertStore == nil else {
            throw NFKMLXError.trainingDataMismatch("a paged mixture of experts reads its experts from disk and cannot train; load it resident")
        }
        return try NFKMLXFineTune.run(
            net,
            freezing: {
                guard let rank else {
                    return
                }
                let adapted = try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha, dropout: loraDropout) { path, _ in
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
            batch: { let example = examples($0); return (example.tokens, MLXArray(Int32(example.promptLength))) },
            loss: { net, tokens, promptLength in
                objective.loss(logits: net(tokens.reshaped([1, tokens.shape[0]])), tokens: tokens, scoredFrom: promptLength)
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
