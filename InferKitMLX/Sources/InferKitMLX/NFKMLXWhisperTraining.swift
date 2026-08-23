//
//  NFKMLXWhisperTraining.swift
//  InferKitMLX
//
//  Adapting speech recognition to a consumer's own domain: their jargon, their accents, their recording
//  conditions. This is the heaviest recipe here and the one LoRA exists for — Whisper has no small head
//  to retrain, and a full fine-tune needs optimizer state proportional to the whole model.
//
//  The objective is teacher forcing, the same as the reference: the decoder sees the whole target
//  sequence at once and each position is scored on predicting the next token. That makes a training step
//  one forward pass rather than one per token, which is the difference between a run finishing and not.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The next-token objective a Whisper fine-tune minimizes.
public struct NFKMLXWhisperObjective: Sendable {

    /// Smooths the target distribution, which steadies a run over few transcribed clips.
    public var labelSmoothing: Float

    public init(labelSmoothing: Float = 0) {
        self.labelSmoothing = labelSmoothing
    }

    /// Scores `net` on one transcribed clip: a log-mel spectrogram and the full target token sequence
    /// including its decode prompt.
    ///
    /// Each position predicts the next token, so the last position has no target and the first has no
    /// prediction: the logits are taken up to the final position and the tokens from the second.
    public func callAsFunction(_ net: NFKMLXWhisperNet, _ mel: MLXArray, _ tokens: MLXArray) -> MLXArray {
        let length = tokens.shape[0]
        let audio = net.encoder(mel)
        let logits = net.decoder(tokens.reshaped([1, length]), audio: audio)
        return loss(logits: logits, tokens: tokens)
    }

    /// Scores decoder logits directly, without a forward pass.
    ///
    /// Separable so the objective can be compared against the reference on identical logits, the way
    /// the Zero-DCE and SegFormer objectives are.
    ///
    /// - Parameters:
    ///   - logits: `[1, length, vocabulary]`, as the decoder produces them.
    ///   - tokens: the target sequence `[length]`.
    ///
    public func loss(logits: MLXArray, tokens: MLXArray) -> MLXArray {
        let length = tokens.shape[0]
        guard length > 1 else {
            return MLXArray(Float(0))
        }
        let vocabulary = logits.shape[2]
        let predictions = logits[0, 0 ..< (length - 1), 0...].reshaped([length - 1, vocabulary])
        let targets = tokens[1 ..< length]
        return crossEntropy(logits: predictions, targets: targets,
                            labelSmoothing: labelSmoothing, reduction: .mean)
    }
}

extension NFKMLXWhisper {

    /// Builds the transcription network itself, ready to adapt, rather than a backend wrapping it.
    public static func network(weightsURL: URL?,
                               configuration: NFKMLXWhisperConfiguration = NFKMLXWhisperConfiguration()) throws -> NFKMLXWhisperNet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Turns audio samples into the log-mel spectrogram the objective takes, padded or trimmed to the
    /// 30-second window Whisper is trained on.
    ///
    /// The padding is not a detail: Whisper only ever sees 30-second inputs, and training on shorter
    /// ones was the single biggest accuracy factor when this model was brought to reference parity.
    public static func spectrogram(for samples: [Float], sampleRate: Int,
                                   configuration: NFKMLXWhisperConfiguration = NFKMLXWhisperConfiguration()) -> MLXArray {
        let target = 30 * sampleRate
        var window = samples
        if window.count > target {
            window = Array(window[0 ..< target])
        } else if window.count < target {
            window.append(contentsOf: [Float](repeating: 0, count: target - window.count))
        }
        return NFKMLXMel.logMel(window, sampleRate: sampleRate, nMels: configuration.nMels)
    }

    /// Adapts the decoder's attention projections with LoRA and trains them on transcribed clips.
    ///
    /// - Parameters:
    ///   - net: the network to adapt, from ``network(weightsURL:configuration:)``.
    ///   - examples: supplies one transcribed clip per step: a spectrogram from
    ///     ``spectrogram(for:sampleRate:configuration:)`` and its full target token sequence,
    ///     decode prompt included.
    ///   - rank: the LoRA detour's width. Nil skips adaptation and trains every parameter, which needs
    ///     a workstation rather than a device.
    ///   - alpha: the adapter's strength, applied as `alpha / rank`.
    ///   - objective: the next-token loss.
    ///   - optimizer: the update rule. Nil uses `AdamW`.
    ///   - steps: how many clips to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Only the decoder is adapted. The encoder's audio features transfer across domains, and leaving
    /// it frozen halves what a run has to hold. Call `NFKMLXLoRA.merge(into:)` before saving, so the
    /// result is one ordinary checkpoint that ``backend(weightsURL:)`` loads.
    ///
    /// A run is minutes rather than seconds; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXWhisperNet,
        examples: (Int) -> (mel: MLXArray, tokens: MLXArray),
        rank: Int? = 8,
        alpha: Float = 16,
        objective: NFKMLXWhisperObjective = NFKMLXWhisperObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        if let rank {
            let adapted = try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha) { path, _ in
                isDecoderAttentionProjection(path)
            }
            guard adapted > 0 else {
                throw NFKMLXError.trainingDataMismatch(
                    "no decoder attention projections were found to adapt, so nothing would train")
            }
        }
        return try NFKMLXTrainer.train(
            net, optimizer: optimizer ?? AdamW(learningRate: 1e-4), steps: steps,
            batch: { let example = examples($0); return (example.mel, example.tokens) },
            loss: objective.callAsFunction,
            clipGradientNorm: clipGradientNorm, checkpoint: checkpoint, observer: observer)
    }

    /// The projections LoRA targets: query and value inside the decoder's attention blocks.
    ///
    /// Adapting query and value rather than all four projections is the reference LoRA finding, and it
    /// halves the adapters for the same effect.
    static func isDecoderAttentionProjection(_ path: String) -> Bool {
        guard path.hasPrefix("decoder.blocks.") else {
            return false
        }
        return path.hasSuffix(".query") || path.hasSuffix(".value")
    }
}
