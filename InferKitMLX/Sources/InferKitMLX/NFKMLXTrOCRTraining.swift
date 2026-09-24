//
//  NFKMLXTrOCRTraining.swift
//  InferKitMLX
//
//  Fine-tuning TrOCR on a consumer's own handwriting or print, the way its authors fine-tuned the
//  releases: `fairseq-train --task text_recognition` (microsoft/unilm/trocr), teacher-forced
//  cross-entropy over every weight. The target is the text's pieces followed by the end token, and the
//  decoder reads it shifted right behind the start token, so one step is one forward pass.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a TrOCR fine-tune updates.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXTrOCRTrainable: Sendable {
    /// Every weight, as the reference fine-tunes.
    case everything
    /// The text decoder, with the image encoder frozen.
    case decoder
}

/// The objective a TrOCR fine-tune minimizes: fairseq's `cross_entropy` criterion, the token sum the
/// trainer divides by the token count, which is the mean over the target's tokens.
///
/// @discussion transformers 4.57's `VisionEncoderDecoderModel` scores `labels=` with `ForCausalLMLoss`,
/// which shifts the logits against the labels a second time; its loss is misaligned by one token and is
/// not the objective the releases were trained on.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTrOCRObjective: Sendable {
    public init() {}

    /// Scores `net` on one line: pixels `[1, side, side, 3]` from ``NFKMLXTrOCRProcessor/pixelValues(_:side:bicubic:)``
    /// and the target ids `[T]` from ``NFKMLXTrOCRProcessor/targetIds(for:tokenizer:endToken:)``.
    public func callAsFunction(_ net: NFKMLXTrOCRNet, _ pixels: MLXArray, _ target: MLXArray) -> MLXArray {
        let input = NFKMLXTranslationObjective.decoderInput(for: target,
                                                           startToken: net.languageConfiguration.decoderStartTokenId)
        let logits = net.decode(input.reshaped([1, input.shape[0]]), memory: net.imageFeatures(pixels),
                                cache: net.makeCache())
        return loss(logits: logits, target: target)
    }

    /// The mean negative log-likelihood of `target` `[T]` under teacher-forced logits `[1, T, vocabulary]`.
    public func loss(logits: MLXArray, target: MLXArray) -> MLXArray {
        let length = target.shape[0]
        let predictions = logits[0, 0 ..< length, 0...].asType(.float32)
        return crossEntropy(logits: predictions, targets: target, reduction: .mean)
    }
}

extension NFKMLXTrOCRProcessor {
    /// A transcription's target ids: its pieces under the release's tokenizer, with no start token, then
    /// the end token. This is fairseq's `encode_line` form and the sequence a release generates after
    /// its decoder start token.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func targetIds(for text: String, tokenizer: NFKTokenizer, endToken: Int) -> MLXArray {
        MLXArray((tokenizer.encode(text).map(\.intValue) + [endToken]).map(Int32.init))
    }
}

extension NFKMLXTrOCR {

    /// Builds the network itself, ready to fine-tune, from a release directory or a directory
    /// ``save(_:toDirectoryURL:release:)`` wrote.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL) throws -> NFKMLXTrOCRNet {
        let net = try NFKMLXTrOCRNet(configurationURL: directoryURL.appendingPathComponent("config.json"))
        try net.loadWeights(fromDirectory: directoryURL)
        return net
    }

    /// Fine-tunes a TrOCR network on a consumer's own lines, returning the loss from each step.
    ///
    /// The whole customization path is three calls: ``network(directoryURL:)`` to build, this to train,
    /// and ``save(_:toDirectoryURL:release:)`` to write a directory ``backend(directoryURL:)`` loads.
    ///
    /// - Parameters:
    ///   - net: the network to train.
    ///   - examples: supplies one line per step: its pixels and its target ids.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - objective: the token loss.
    ///   - optimizer: the update rule. Nil uses the reference's Adam with decoupled weight decay 1e-4
    ///     (fairseq's `adam`, bias-corrected, betas 0.9 and 0.999) at `learningRate`.
    ///   - learningRate: the reference optimizer's peak rate: 2e-5 in the IAM and receipt recipes, 5e-5
    ///     in the SROIE one.
    ///   - warmupSteps: the reference schedule's warm-up: 500 updates for IAM, 800 for SROIE.
    ///   - steps: how many lines to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference does not clip.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the reference's fairseq
    ///     `inverse_sqrt`, a linear warm-up from 1e-8 then `√(warmup / k)`, when the reference optimizer
    ///     runs. With a caller's optimizer, nil holds that optimizer's rate constant.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second per step for the large releases; call it off the render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXTrOCRNet,
        examples: (Int) -> (pixels: MLXArray, target: MLXArray),
        trainable: NFKMLXTrOCRTrainable = .everything,
        objective: NFKMLXTrOCRObjective = NFKMLXTrOCRObjective(),
        optimizer: Optimizer? = nil,
        learningRate: Float = 2e-5,
        warmupSteps: Int = 500,
        steps: Int,
        clipGradientNorm: Float? = nil,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        return try NFKMLXFineTune.run(
            net,
            freezing: {
                net.unfreeze()
                if trainable == .decoder {
                    net.vision.freeze()
                }
            },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: learningRate, weightDecay: 1e-4) },
            referenceSchedule: { .fairseqInverseSquareRoot(warmupSteps: warmupSteps,
                                                           initialScale: 1e-8 / learningRate) },
            steps: steps,
            batch: { let example = examples($0); return (example.pixels, example.target) },
            loss: objective.callAsFunction,
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// Writes `net` as a release directory: `model.safetensors` in the module's own layout, and the
    /// release's configuration, processor, and tokenizer files copied from `release`, which fine-tuning
    /// does not change. ``backend(directoryURL:)`` and ``network(directoryURL:)`` load it.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func save(_ net: NFKMLXTrOCRNet, toDirectoryURL directory: URL, release: URL) throws {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        try NFKMLXWeights.save(net, to: directory.appendingPathComponent("model.safetensors"))
        let copied = ["config.json", "generation_config.json", "preprocessor_config.json", "vocab.json", "merges.txt",
                      "sentencepiece.bpe.model", "tokenizer_config.json", "special_tokens_map.json"]
        for name in copied {
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
