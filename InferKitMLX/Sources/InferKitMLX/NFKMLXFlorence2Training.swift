//
//  NFKMLXFlorence2Training.swift
//  InferKitMLX
//
//  Adapting Florence-2 to a consumer's own task or domain: an image and a task prompt in, the answer
//  the consumer wants out. Microsoft publishes no training script; the objective is the release's own
//  `labels=` loss (its `Florence2LanguageForConditionalGeneration`), teacher forcing over the answer
//  behind the decoder start token, and the adaptation is LoRA on the decoder's attention, as the
//  translators' recipe is.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/// The objective a Florence-2 fine-tune minimizes: the release's `labels=` cross-entropy, the mean over
/// the answer's tokens.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXFlorence2Objective: Sendable {
    public init() {}

    /// Scores `net` on one example: pixels `[1, side, side, 3]` from
    /// ``NFKMLXFlorence2Processor/pixelValues(_:side:)``, the prompt ids `[P]` and the answer ids `[T]`,
    /// both from ``NFKMLXFlorence2Processor/encodePrompt(_:tokenizer:eosTokenId:)`` (the answer wrapped
    /// `<s> … </s>`, the sequence a release generates after its decoder start token).
    public func callAsFunction(_ net: NFKMLXFlorence2Net, pixels: MLXArray, prompt: MLXArray, answer: MLXArray) -> MLXArray {
        let memory = net.encode(pixels: pixels, inputIds: prompt.reshaped([1, -1]))
        let input = NFKMLXTranslationObjective.decoderInput(for: answer, startToken: net.textConfig.decoderStartTokenId)
        let logits = net.language.decode(input.reshaped([1, -1]), memory: memory, cache: net.language.makeCache())
        return loss(logits: logits, target: answer)
    }

    /// The mean negative log-likelihood of `target` `[T]` under teacher-forced logits `[1, T, vocabulary]`.
    public func loss(logits: MLXArray, target: MLXArray) -> MLXArray {
        let length = target.shape[0]
        return crossEntropy(logits: logits[0, 0 ..< length, 0...].asType(.float32), targets: target, reduction: .mean)
    }
}

extension NFKMLXFlorence2 {

    /// Builds the network itself, ready to fine-tune, from a release directory or a directory
    /// ``save(_:toDirectoryURL:release:)`` wrote.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL) throws -> NFKMLXFlorence2Net {
        let geometry = try NFKMLXFlorence2Net.configuration(fromConfigURL: directoryURL.appendingPathComponent("config.json"))
        let net = NFKMLXFlorence2Net(vision: geometry.vision, text: geometry.text)
        try net.loadWeights(from: directoryURL.appendingPathComponent("model.safetensors"))
        return net
    }

    /// Adapts a Florence-2 network to a consumer's own examples, returning the loss from each step.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(directoryURL:)``.
    ///   - examples: supplies one example per step: its pixels, prompt ids, and answer ids.
    ///   - rank: the LoRA width on the decoder's query and value projections. Nil trains the whole
    ///     language model and the projector, with the vision tower frozen.
    ///   - alpha: the adapter's strength, applied as `alpha / rank`.
    ///   - objective: the token loss.
    ///   - optimizer: the update rule. Nil uses `torch.optim.AdamW` (bias-corrected) at 1e-4 with no
    ///     weight decay, the translators' default; Microsoft publishes no fine-tuning script, so the rate
    ///     is this package's choice.
    ///   - steps: how many examples to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Call `NFKMLXLoRA.merge(into:)` before ``save(_:toDirectoryURL:release:)``, so the result is one
    /// checkpoint the ordinary factory reads.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXFlorence2Net,
        examples: (Int) -> (pixels: MLXArray, prompt: MLXArray, answer: MLXArray),
        rank: Int? = 8,
        alpha: Float = 16,
        objective: NFKMLXFlorence2Objective = NFKMLXFlorence2Objective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: {
                if let rank {
                    let adapted = try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha) { path, _ in
                        path.hasPrefix("language_model.decoder.layers.")
                            && (path.hasSuffix(".q_proj") || path.hasSuffix(".v_proj"))
                    }
                    guard adapted > 0 else {
                        throw NFKMLXError.trainingDataMismatch(
                            "no decoder attention projections were found to adapt, so nothing would train")
                    }
                } else {
                    net.unfreeze()
                    net.vision.freeze()
                }
            },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: 1e-4, weightDecay: 0) },
            referenceSchedule: { .constant },
            steps: steps,
            arrays: { let example = examples($0); return [example.pixels, example.prompt, example.answer] },
            loss: { model, arrays in objective(model, pixels: arrays[0], prompt: arrays[1], answer: arrays[2]) },
            clipGradientNorm: clipGradientNorm,
            checkpoint: checkpoint, observer: observer)
    }

    /// Writes `net` as a release directory: `model.safetensors` in the module's own layout beside the
    /// release's `config.json` and `tokenizer.json`, which fine-tuning does not change.
    /// ``backend(directoryURL:)`` and ``network(directoryURL:)`` load it.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func save(_ net: NFKMLXFlorence2Net, toDirectoryURL directory: URL, release: URL) throws {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        try NFKMLXWeights.save(net, to: directory.appendingPathComponent("model.safetensors"))
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                     "generation_config.json"] {
            let source = release.appendingPathComponent(name), destination = directory.appendingPathComponent(name)
            guard manager.fileExists(atPath: source.path), source.standardizedFileURL != destination.standardizedFileURL else {
                continue
            }
            if manager.fileExists(atPath: destination.path) {
                try manager.removeItem(at: destination)
            }
            try manager.copyItem(at: source, to: destination)
        }
    }
}
