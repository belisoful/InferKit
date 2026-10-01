//
//  NFKMLXTimesFMTraining.swift
//  InferKitMLX
//
//  Fine-tuning TimesFM 2.5 the way google-research/timesfm's own example does
//  (`timesfm-forecasting/examples/finetuning/finetune_lora.py`): LoRA adapters of rank 4 on every linear
//  layer, AdamW at 1e-4 with weight decay 0.01, a cosine decay over the run, clipping at 1.0, and the loss
//  transformers' `TimesFm2_5ModelForPrediction` computes when given `future_values`.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a TimesFM fine-tune updates.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXTimesFMTrainable: Sendable {
    /// LoRA adapters on every linear layer, the base frozen: the reference's recipe.
    case lora(rank: Int, alpha: Float)
    /// Every parameter.
    case all

    /// The reference's adapter: rank 4, alpha 8.
    public static let reference = NFKMLXTimesFMTrainable.lora(rank: 4, alpha: 8)
}

/// The objective TimesFM fine-tunes against: transformers' `TimesFm2_5ModelForPrediction` loss.
///
/// Both terms are computed in the space the forecast is normalized to (the context's mean and sample
/// deviation), on the forecast after flip averaging and the continuous quantile head, as the reference's
/// training forward produces it: the mean squared error of the median channel, plus the pinball loss of
/// each non-median channel averaged over the nine levels. The reference pairs the nine remaining channels
/// (the mean, then the 0.1 … 0.4 and 0.6 … 0.9 quantiles) with the nine levels in order, so the mean channel
/// takes level 0.1 and each quantile channel the level after its own; the loss reproduces that pairing.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTimesFMObjective: Sendable {
    public init() {}

    /// The loss of forecasts `[B, horizon, 10]` against targets `[B, horizon]`, both normalized.
    public func loss(forecast: MLXArray, targets: MLXArray, quantiles: [Float], decodeIndex: Int = 5) -> MLXArray {
        let median = forecast[.ellipsis, decodeIndex]
        let squared = ((median - targets) * (median - targets)).mean()
        let channels = (0 ..< forecast.dim(-1)).filter { $0 != decodeIndex }
        var pinball = [MLXArray]()
        for (level, channel) in zip(quantiles, channels) {
            let errors = targets - forecast[.ellipsis, channel]
            pinball.append(maximum((level - 1) * errors, level * errors).mean())
        }
        return squared + MLX.stacked(pinball).mean()
    }

    /// Scores `net` on windows of context and the values that follow each. Every target holds the same
    /// number of steps, at most 128.
    public func callAsFunction(_ net: NFKMLXTimesFMNet, windows: [(context: [Float], target: [Float])]) -> MLXArray {
        let c = net.configuration
        var forecasts = [MLXArray](), targets = [MLXArray]()
        for window in windows {
            let (forecast, normalizedTarget) = NFKMLXTimesFM.trainingForecast(net, context: window.context, target: window.target)
            forecasts.append(forecast)
            targets.append(normalizedTarget)
        }
        return loss(forecast: MLX.stacked(forecasts), targets: MLX.stacked(targets), quantiles: c.quantiles,
                    decodeIndex: c.decodeIndex)
    }
}

extension NFKMLXTimesFM {

    /// The reference's training forward on one window, in normalized space: the context padded in front to
    /// a whole number of patches, normalized by its mean and sample deviation over that padded length,
    /// decoded with flip averaging and the continuous quantile head, and the target normalized alike.
    /// Returns the forecast `[horizon, 10]` and the target `[horizon]`.
    static func trainingForecast(_ net: NFKMLXTimesFMNet, context: [Float], target: [Float]) -> (MLXArray, MLXArray) {
        let c = net.configuration
        let p = c.patchLength
        let padded = (context.count + p - 1) / p * p
        let (mean, deviation) = globalStatistics(context, paddedLength: padded)
        let divisor = deviation < 1e-6 ? 1 : deviation
        let normalized = context.map { ($0 - mean) / divisor }
        let horizon = min(target.count, c.outputPatchLength)
        let order = MLXArray([Int32(0)] + (1 ..< c.outputChannels).reversed().map(Int32.init))
        let direct = decode(net, normalized, steps: 0)
        let flipped = decode(net, normalized.map { -$0 }, steps: 0)
        let point = (direct.point - take(flipped.point, order, axis: -1)) / 2
        let spreads = (direct.spreads - take(flipped.spreads, order, axis: -1)) / 2
        let forecast = point[..<horizon]
        let medianColumn = forecast[0..., c.decodeIndex ..< (c.decodeIndex + 1)]
        let spread = spreads[..<horizon]
        let continuous = spread - spread[0..., c.decodeIndex ..< (c.decodeIndex + 1)] + medianColumn
        let keep = MLXArray((0 ..< c.outputChannels).map { $0 == 0 || $0 == c.decodeIndex })
        let targets = MLXArray(target.prefix(horizon).map { ($0 - mean) / divisor })
        return (MLX.where(keep, forecast, continuous), targets)
    }

    /// Builds the network itself, ready to fine-tune, from a release directory or one
    /// ``save(_:toDirectoryURL:)`` wrote.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL) throws -> NFKMLXTimesFMNet {
        let net = NFKMLXTimesFMNet(try NFKMLXTimesFMConfiguration(configurationURL: directoryURL.appendingPathComponent("config.json")))
        try net.loadWeights(fromDirectory: directoryURL)
        return net
    }

    /// The windows the reference trains on in each step, `finetune_lora.py`'s batch of 32.
    ///
    /// Introduced in InferKit 0.4.0.
    public static let referenceWindowsPerStep = 32

    /// Fine-tunes TimesFM on a consumer's own series, returning the loss from each step.
    ///
    /// The whole customization path is three calls: ``network(directoryURL:)`` to build, this to train, and
    /// ``save(_:toDirectoryURL:)`` to fold the adapters in and write a directory that
    /// ``timesFM(directoryURL:)`` loads like any release.
    ///
    /// - Parameters:
    ///   - net: the network to train.
    ///   - windows: supplies one batch per step: context windows and the values that follow each. The
    ///     reference draws 32 windows a step, each a full context of a whole number of 32-step patches with
    ///     no padding, since padding would enter the normalization's statistics. Every target in a batch
    ///     holds the same number of steps, at most 128.
    ///   - trainable: LoRA adapters (the reference's rank 4 and alpha 8 by default) or every parameter.
    ///     Adapters are applied here, to every linear layer as PEFT's `all-linear` selects them, and persist
    ///     on `net`; applying again adds none.
    ///   - loraDropout: the dropout on each adapter's input while the run trains, PEFT's `lora_dropout`;
    ///     the reference sets 0.05. 0, the default, drops nothing.
    ///   - objective: the reference's loss.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.AdamW` (bias-corrected, betas
    ///     0.9 and 0.999, epsilon 1e-8) at `learningRate` with `weightDecay`.
    ///   - learningRate: the reference's rate.
    ///   - weightDecay: the reference's decoupled weight decay.
    ///   - steps: how many batches to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update, as the reference's
    ///     `clip_grad_norm_(…, max_norm=1.0)`.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates. 1, the
    ///     default, updates after every batch. The reference updates on a batch of 32 windows
    ///     (``referenceWindowsPerStep``), which `windows` supplies.
    ///   - precision: the precision the passes compute in; float32 by default. The reference loads the
    ///     model in bfloat16, which `.bfloat16` approximates.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the reference's
    ///     `CosineAnnealingLR` to zero over the run when the reference optimizer runs. With a caller's
    ///     optimizer, nil holds that optimizer's rate constant.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXTimesFMNet,
        windows: (Int) -> [(context: [Float], target: [Float])],
        trainable: NFKMLXTimesFMTrainable = .reference,
        loraDropout: Float = 0,
        objective: NFKMLXTimesFMObjective = NFKMLXTimesFMObjective(),
        optimizer: Optimizer? = nil,
        learningRate: Float = 1e-4,
        weightDecay: Float = 0.01,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        var batch = [(context: [Float], target: [Float])]()
        return try NFKMLXFineTune.run(
            net,
            freezing: { try apply(trainable, to: net, loraDropout: loraDropout) },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: learningRate, weightDecay: weightDecay) },
            referenceSchedule: { .cosine(steps: steps, endScale: 0) },
            steps: steps,
            arrays: { step in
                batch = windows(step)
                return [MLXArray(Int32(batch.count))]
            },
            loss: { net, _ in objective(net, windows: batch) },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// Adapts or unfreezes `net` for `trainable`.
    static func apply(_ trainable: NFKMLXTimesFMTrainable, to net: NFKMLXTimesFMNet, loraDropout: Float = 0) throws {
        switch trainable {
        case .lora(let rank, let alpha):
            try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha, dropout: loraDropout)
        case .all:
            net.unfreeze()
        }
    }

    /// Folds any adapters into their layers and writes `net` as a release directory: `model.safetensors`
    /// in the module's own layout and the `config.json` its configuration reads back.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func save(_ net: NFKMLXTimesFMNet, toDirectoryURL directory: URL) throws {
        try NFKMLXLoRA.merge(into: net)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try NFKMLXWeights.save(net, to: directory.appendingPathComponent("model.safetensors"))
        let c = net.configuration
        let config: [String: Any] = [
            "model_type": "timesfm", "patch_length": c.patchLength, "horizon_length": c.outputPatchLength,
            "quantile_horizon_length": c.outputQuantileLength, "quantiles": c.quantiles.map(Double.init),
            "decode_index": c.decodeIndex, "hidden_size": c.hiddenSize, "intermediate_size": c.intermediateSize,
            "num_hidden_layers": c.numLayers, "num_attention_heads": c.numHeads, "head_dim": c.headDimension,
            "rms_norm_eps": Double(c.rmsNormEps), "context_length": c.contextLimit,
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appendingPathComponent("config.json"))
    }
}
