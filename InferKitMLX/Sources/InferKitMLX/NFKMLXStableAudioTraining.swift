//
//  NFKMLXStableAudioTraining.swift
//  InferKitMLX
//
//  Fine-tuning Stable Audio Open on a consumer's own clips the way stable-audio-tools trains it
//  (Stability-AI/stable-audio-tools at 3241adb, `training/diffusion.py`, `DiffusionCondTrainingWrapper`
//  with a `lora_config`): LoRA adapters on every linear layer of the transformer and of the seconds
//  conditioners, the frozen T5 outside them; the v-objective on a cosine schedule with timesteps drawn
//  uniformly; the cross-attention conditioning dropped for one example in ten; the mean squared error over
//  each clip's unpadded frames; and the release's AdamW and `InverseLR` from its `model_config.json`.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// The networks a Stable Audio fine-tune updates: the diffusion transformer and the projection model
/// whose seconds conditioners stable-audio-tools adapts beside it. t5-base stays outside, frozen, as the
/// reference holds it.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXStableAudioTrainingNet: Module {
    @ModuleInfo(key: "transformer") public var transformer: NFKMLXStableAudioDiTNet
    @ModuleInfo(key: "projection") public var projection: NFKMLXStableAudioProjectionNet

    public init(transformer: NFKMLXStableAudioDiTNet, projection: NFKMLXStableAudioProjectionNet) {
        _transformer.wrappedValue = transformer
        _projection.wrappedValue = projection
    }

    /// The conditioning for masked text states `[B, L, d]` and start and total seconds `[B]`: the sequence
    /// `[B, L + 2, d]` the cross-attention reads and the global token `[B, 1, 2d]`, as stable-audio-tools'
    /// conditioner assembles `prompt`, `seconds_start`, and `seconds_total`.
    public func conditioning(text: MLXArray, start: MLXArray, total: MLXArray) -> (context: MLXArray, global: MLXArray) {
        let seconds = projection.seconds(start: start, end: total)
        return (concatenated([projection.projectedText(text), seconds.start, seconds.end], axis: 1),
                concatenated([seconds.start, seconds.end], axis: -1))
    }
}

/// Which parameters a Stable Audio fine-tune updates.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXStableAudioTrainable: Sendable {
    /// LoRA adapters on every linear layer of the transformer and of the seconds conditioners, the rest
    /// frozen: stable-audio-tools' `lora_config`, whose default alpha equals the rank.
    case lora(rank: Int, alpha: Float)

    /// stable-audio-tools' default adapter: rank 8, alpha 8.
    public static let reference = NFKMLXStableAudioTrainable.lora(rank: 8, alpha: 8)
}

/// One training example: a clip's latents, how many of them hold the clip, and what it was described as.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXStableAudioExample {
    /// The clip's latents `[frames, 64]`, a draw from the autoencoder's posterior, as stable-audio-tools'
    /// pretransform encodes.
    public var latents: MLXArray
    /// How many leading frames hold the clip; the rest are padding, which the loss leaves out.
    public var validFrames: Int
    /// The masked text states `[L, d]` t5-base encodes for the description.
    public var text: MLXArray
    /// Where the clip starts in its source and how long the source runs, in seconds:
    /// `seconds_start` and `seconds_total`.
    public var startSeconds: Float
    public var totalSeconds: Float

    public init(latents: MLXArray, validFrames: Int, text: MLXArray, startSeconds: Float, totalSeconds: Float) {
        self.latents = latents
        self.validFrames = validFrames
        self.text = text
        self.startSeconds = startSeconds
        self.totalSeconds = totalSeconds
    }
}

/// The objective stable-audio-tools trains Stable Audio Open with: v-diffusion on a cosine schedule.
///
/// @discussion For a timestep `t` in `[0, 1]`, `α = cos(t·π/2)` and `σ = sin(t·π/2)`. The noised latents are
/// `x·α + n·σ`, the transformer reads them at timestep `t` (the `atan(σ_edm)·2/π` the sampler passes at
/// inference is the same `t`), and the target is `v = n·α − x·σ`. Each example's loss is the squared error
/// summed over its unpadded frames and every channel, divided by that count plus 1e-8, and the batch's
/// loss is the mean over the examples (`compute_masked_loss` with `mask_loss_weight` 0). An example whose
/// `dropped` flag is 1 reads zeros in place of its cross-attention conditioning; its global token stays.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXStableAudioObjective: Sendable {
    /// The probability an example trains without its conditioning, `cfg_dropout_prob` (0.1).
    public var conditioningDropout: Float

    public init(conditioningDropout: Float = 0.1) {
        self.conditioningDropout = conditioningDropout
    }

    /// The signal and noise scales `(α, σ)` at timesteps `t`.
    public static func scales(_ t: MLXArray) -> (alpha: MLXArray, sigma: MLXArray) {
        (cos(t * Float.pi / 2), sin(t * Float.pi / 2))
    }

    /// The masked loss of `prediction` against the v-targets for latents `x` `[B, T, C]`, noise `n`,
    /// timesteps `t` `[B]`, and the frame mask `[B, T]` (1 where a frame holds the clip).
    public func loss(prediction: MLXArray, latents: MLXArray, noise: MLXArray, timesteps: MLXArray,
                     mask: MLXArray) -> MLXArray {
        let (alpha, sigma) = Self.scales(timesteps.reshaped([-1, 1, 1]))
        let target = noise * alpha - latents * sigma
        let frames = mask.asType(prediction.dtype).expandedDimensions(axis: -1)
        let signal = ((prediction - target).square() * frames).sum(axes: [1, 2])
        let count = mask.asType(prediction.dtype).sum(axis: 1) * Float(prediction.dim(-1))
        return (signal / (count + 1e-8)).mean()
    }

    /// Scores `net` on a batch: noises each example's latents at `timesteps` with `noise`, conditions it,
    /// drops the cross-attention conditioning where `dropped` is 1, and returns the masked loss.
    public func callAsFunction(_ net: NFKMLXStableAudioTrainingNet, latents: MLXArray, mask: MLXArray,
                               text: MLXArray, start: MLXArray, total: MLXArray, noise: MLXArray,
                               timesteps: MLXArray, dropped: MLXArray) -> MLXArray {
        let (alpha, sigma) = Self.scales(timesteps.reshaped([-1, 1, 1]))
        let noised = latents * alpha + noise * sigma
        let condition = net.conditioning(text: text, start: start, total: total)
        let keep = (1 - dropped.reshaped([-1, 1, 1])).asType(condition.context.dtype)
        let prediction = net.transformer(noised, timestep: timesteps, context: condition.context * keep,
                                         global: condition.global)
        return loss(prediction: prediction, latents: latents, noise: noise, timesteps: timesteps, mask: mask)
    }
}

extension NFKMLXStableAudio {

    /// Builds the transformer and the projection model, ready to fine-tune, from a release directory or one
    /// ``save(_:toDirectoryURL:)`` wrote: its `transformer/` and `projection_model/`.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL) throws -> NFKMLXStableAudioTrainingNet {
        let transformer = NFKMLXStableAudioDiTNet(try .configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("transformer/config.json")))
        try loadTransformerWeights(into: transformer, fromDirectory: directoryURL.appendingPathComponent("transformer"))
        let projection = NFKMLXStableAudioProjectionNet(try .configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("projection_model/config.json")))
        try loadProjectionWeights(into: projection,
                                  from: directoryURL.appendingPathComponent("projection_model/diffusion_pytorch_model.safetensors"))
        return NFKMLXStableAudioTrainingNet(transformer: transformer, projection: projection)
    }

    /// The release's optimizer settings, from its `model_config.json`: AdamW at 5e-5, betas 0.9 and
    /// 0.999, weight decay 1e-3, under `InverseLR` with `inv_gamma` 1e6, `power` 0.5, and `warmup` 0.99.
    ///
    /// Introduced in InferKit 0.4.0.
    public static let referenceLearningRate: Float = 5e-5
    public static let referenceWeightDecay: Float = 1e-3
    public static let referenceSchedule = NFKMLXLearningRateSchedule.inverse(inverseGamma: 1_000_000, power: 0.5,
                                                                             warmup: 0.99)

    /// Adapts `net` for `trainable`.
    static func apply(_ trainable: NFKMLXStableAudioTrainable, to net: NFKMLXStableAudioTrainingNet,
                      loraDropout: Float = 0) throws {
        switch trainable {
        case .lora(let rank, let alpha):
            try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha, dropout: loraDropout)
        }
    }

    /// Fine-tunes Stable Audio on a consumer's own clips, returning the loss from each step.
    ///
    /// The whole customization path is four calls: ``NFKMLXStableAudioGenerator/trainingExample(waveform:prompt:startSeconds:totalSeconds:seed:)``
    /// for each clip, ``network(directoryURL:)`` to build, this to train, and ``save(_:toDirectoryURL:)`` to
    /// fold the adapters in and write the two directories a copy of the release takes in place of its own.
    ///
    /// - Parameters:
    ///   - net: the networks to train.
    ///   - examples: one batch per step. Every example in a batch holds the same number of latent frames
    ///     and of text positions, which ``NFKMLXStableAudioGenerator/trainingExample(waveform:prompt:startSeconds:totalSeconds:seed:)``
    ///     gives by padding each clip to the release's window and each description to 128 tokens.
    ///   - trainable: the reference's LoRA (rank 8, alpha 8). Adapters persist on `net`; applying again adds none.
    ///   - objective: the reference's v-objective with its conditioning dropout.
    ///   - optimizer: the update rule. Nil uses `torch.optim.AdamW` (bias-corrected, betas 0.9 and 0.999)
    ///     at `learningRate` with `weightDecay`, the release's training configuration.
    ///   - steps: how many batches to train on.
    ///   - seed: seeds the timesteps, the noise, and the conditioning dropout.
    ///   - clipGradientNorm: bounds the global gradient norm before the update; the reference sets none.
    ///   - learningRateSchedule: nil uses the reference's `InverseLR` when the reference optimizer runs.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The reference draws its timesteps from a scrambled Sobol sequence; this draws them uniformly, the
    /// same distribution without the low-discrepancy spacing. A run is multi-second; call it off the
    /// render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXStableAudioTrainingNet,
        examples: (Int) -> [NFKMLXStableAudioExample],
        trainable: NFKMLXStableAudioTrainable = .reference,
        loraDropout: Float = 0,
        objective: NFKMLXStableAudioObjective = NFKMLXStableAudioObjective(),
        optimizer: Optimizer? = nil,
        learningRate: Float = referenceLearningRate,
        weightDecay: Float = referenceWeightDecay,
        steps: Int,
        seed: UInt64 = 0,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        var draws = MLXRandom.key(seed)
        return try NFKMLXFineTune.run(
            net,
            freezing: { try apply(trainable, to: net, loraDropout: loraDropout) },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: learningRate, weightDecay: weightDecay) },
            referenceSchedule: { referenceSchedule },
            steps: steps,
            arrays: { step in
                let batch = examples(step)
                let latents = MLX.stacked(batch.map(\.latents))
                let frames = latents.dim(1)
                let mask = MLX.stacked(batch.map { example in
                    MLXArray((0 ..< frames).map { Float($0 < example.validFrames ? 1 : 0) })
                })
                let keys = MLXRandom.split(key: draws, into: 4)
                draws = keys[0]
                let count = batch.count
                return [latents, mask, MLX.stacked(batch.map(\.text)), MLXArray(batch.map(\.startSeconds)),
                        MLXArray(batch.map(\.totalSeconds)), MLXRandom.normal(latents.shape, key: keys[1]),
                        MLXRandom.uniform(low: 0, high: 1, [count], key: keys[2]),
                        (MLXRandom.uniform(low: 0, high: 1, [count], key: keys[3]) .< objective.conditioningDropout)
                            .asType(.float32)]
            },
            loss: { net, arrays in
                objective(net, latents: arrays[0], mask: arrays[1], text: arrays[2], start: arrays[3],
                          total: arrays[4], noise: arrays[5], timesteps: arrays[6], dropped: arrays[7])
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// Folds any adapters into their layers and writes `net` as the `transformer/` and `projection_model/`
    /// directories of a release under `directory`, each a `diffusion_pytorch_model.safetensors` in the
    /// modules' own layout and the `config.json` its configuration reads back. A copy of the release with
    /// these in place of its own loads through ``NFKMLXStableAudioGenerator/generator(directoryURL:)`` and
    /// ``network(directoryURL:)``.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func save(_ net: NFKMLXStableAudioTrainingNet, toDirectoryURL directory: URL) throws {
        try NFKMLXLoRA.merge(into: net)
        let transformerDirectory = directory.appendingPathComponent("transformer")
        let projectionDirectory = directory.appendingPathComponent("projection_model")
        for folder in [transformerDirectory, projectionDirectory] {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        }
        try NFKMLXWeights.save(net.transformer,
                               to: transformerDirectory.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        try NFKMLXWeights.save(net.projection,
                               to: projectionDirectory.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        let t = net.transformer.configuration, p = net.projection.configuration
        try writeConfig(["_class_name": "StableAudioDiTModel", "sample_size": t.sampleSize, "in_channels": t.inChannels,
                         "num_layers": t.layers, "attention_head_dim": t.headDim, "num_attention_heads": t.heads,
                         "num_key_value_attention_heads": t.keyValueHeads, "out_channels": t.outChannels,
                         "cross_attention_dim": t.crossAttentionDim, "time_proj_dim": t.timeProjectionDim,
                         "global_states_input_dim": t.globalStatesInputDim,
                         "cross_attention_input_dim": t.crossAttentionInputDim], to: transformerDirectory)
        try writeConfig(["_class_name": "StableAudioProjectionModel", "text_encoder_dim": p.textEncoderDim,
                         "conditioning_dim": p.conditioningDim, "min_value": Double(p.minValue),
                         "max_value": Double(p.maxValue)], to: projectionDirectory)
    }

    private static func writeConfig(_ config: [String: Any], to directory: URL) throws {
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appendingPathComponent("config.json"))
    }
}
