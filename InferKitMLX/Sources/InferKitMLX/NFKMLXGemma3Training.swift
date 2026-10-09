//
//  NFKMLXGemma3Training.swift
//  InferKitMLX
//
//  Adapting a Gemma 3 text decoder to a consumer's own text with LoRA, under the causal language-model
//  loss transformers computes from `labels=` (`NFKMLXCausalLanguageObjective`). LoRA adapts the attention
//  query and value projections, PEFT's default target set for `gemma3_text`; everything else freezes.
//  A multimodal release's vision tower and projector are not part of the decoder and do not train.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

public extension NFKMLXGemma3Language {

    /// Builds a Gemma 3 decoder for fine-tuning or for reloading a fine-tuned checkpoint. With a
    /// `weightsURL` the single-file checkpoint loads at float32; without one the net is randomly
    /// initialized. Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?, configuration: NFKMLXGemma3Configuration) throws -> NFKMLXGemma3Net {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Builds a Gemma 3 decoder from a release directory at the geometry its `config.json` declares,
    /// the decoder's tensors alone, held as `residency` says (see
    /// `NFKMLXGemma3.load(directoryURL:precision:residency:)`).
    ///
    /// @discussion `.float32`, the default, is the precision a fine-tune adapts. The 4B decoder is about
    /// 16 GB at float32. A streamed decoder runs and does not train. Introduced in InferKit 0.4.0.
    static func network(directoryURL: URL, precision: NFKMLXWeightPrecision = .float32,
                        residency: NFKMLXResidency = .automatic) throws -> NFKMLXGemma3Net {
        try network(directoryURL: directoryURL, precision: precision, residency: residency,
                    budget: NFKMLXResidencyBudget.current())
    }

    /// ``network(directoryURL:precision:residency:)`` planned against `budget`.
    internal static func network(directoryURL: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                                 budget: Int) throws -> NFKMLXGemma3Net {
        let net = makeNet(try configuration(fromHuggingFace: directoryURL.appendingPathComponent("config.json")))
        let streamed = try NFKMLXGemma3.streamedLayers(directory: directoryURL, precision: precision, includesVision: false,
                                                       residency: residency, budget: budget)
        let placeholders = try stream(streamed, of: net, directory: directoryURL, precision: precision)
        let held = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directoryURL, precision: precision,
                                                               remap: { heldDecoderName(of: $0, streamed: streamed) })
        try NFKMLXWeights.apply(held + placeholders, to: net)
        net.layerStream?.prime()
        return net
    }

    /// Loads one checkpoint file `NFKMLXWeights` wrote from this module, at float32. The file carries the
    /// module's own keys, unlike a release directory, which ``loadWeights(into:fromDirectory:precision:)``
    /// reads. Introduced in InferKit 0.4.0.
    static func loadWeights(into net: NFKMLXGemma3Net, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { ($0.key, $0.value.dtype.isFloatingPoint ? $0.value.asType(.float32) : $0.value) }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    /// The projections LoRA adapts: the query and value of every attention layer, PEFT's default
    /// target set for `gemma3_text`.
    internal static func isAttentionProjection(_ path: String) -> Bool {
        path.hasPrefix("layers.") && (path.hasSuffix(".q_proj") || path.hasSuffix(".v_proj"))
    }

    /// Adapts a Gemma 3 decoder's attention with LoRA and trains it on token sequences.
    ///
    /// - Parameters:
    ///   - net: the decoder, from ``network(directoryURL:precision:)`` at `.float32`, or the `decoder`
    ///     of an `NFKMLXGemma3Model`.
    ///   - examples: supplies one example per step: the token ids (from
    ///     `NFKMLXGemma3Model.promptTokens(_:withImage:)` or `chatTokens(messages:withImage:)`) and how
    ///     many leading tokens are prompt. The prompt is context and is not scored; 0 scores the whole
    ///     sequence.
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
    /// Only the attention query and value projections adapt; the embedding, the feed-forwards, and the
    /// norms stay frozen. Call `NFKMLXLoRA.merge(into:)` before saving, so the result is one checkpoint
    /// ``network(weightsURL:configuration:)`` reads back; `NFKMLXGemma3.model(decoder:directoryURL:precision:)`
    /// pairs the decoder with its release's tokenizer and vision tower. A run is minutes; call it off
    /// the render thread. Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXGemma3Net,
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
        try net.requireHeldLayers()
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

public extension NFKMLXGemma3 {

    /// The model object over `decoder`, a Gemma 3 decoder the caller adapted or reloaded, with the
    /// tokenizer, chat template, and (for a multimodal release) vision tower and projector of the
    /// release directory it came from. Introduced in InferKit 0.4.0.
    static func model(decoder: NFKMLXGemma3Net, directoryURL directory: URL,
                      precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXGemma3Model {
        let parts = try load(directory: directory, precision: precision, decoder: false)
        return NFKMLXGemma3Model(decoder: decoder, vision: parts.vision, projector: parts.projector,
                                 tokenizer: parts.tokenizer, tokens: parts.tokens,
                                 chatTemplate: chatTemplate(inDirectory: directory))
    }

    /// A backend over a loaded model, such as one ``model(decoder:directoryURL:precision:)`` built
    /// around an adapted decoder. Introduced in InferKit 0.4.0.
    static func backend(model: NFKMLXGemma3Model) -> any NFKInferenceBackend {
        NFKMLXGemma3Backend(model: model, identifier: modelName)
    }
}
