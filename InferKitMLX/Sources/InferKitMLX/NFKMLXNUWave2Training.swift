//
//  NFKMLXNUWave2Training.swift
//  InferKitMLX
//
//  Fine-tuning NU-Wave 2 on a consumer's own wide-band audio, so bandwidth extension learns the voice,
//  the instrument, or the room it will be asked to restore. Every weight trains: the network is 1.8M
//  parameters.
//
//  The objective is the reference's `NuWave2.common_step` (`lightning_model.py`): a diffusion time is
//  drawn, the clean clip is noised along the continuous logSNR schedule `Diffusion.diffusion` defines,
//  the network predicts the noise from the noised clip, the narrow-band input, and the band, and the loss
//  is the mean absolute error between the prediction and the noise. It has no adversary.
//
//  The training input follows `dataloader.py`: the clip peak-normalized, a random gain between 0.5 and 1,
//  and a narrow-band copy made by band-limiting it to a lower rate and bringing it back to 48 kHz, with
//  the band marking the bins below the lower rate's Nyquist. The reference draws a random Chebyshev
//  type I pre-filter for that copy as augmentation; this recipe band-limits with the package's
//  windowed-sinc resampler, so a consumer who needs the reference's exact degradation supplies the
//  narrow-band copy through `NFKMLXNUWave2Objective` directly.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/*!
 @abstract The objective a NU-Wave 2 fine-tune minimizes: the reference's noise-prediction L1.
 @discussion At diffusion time `t` in `(0, 1]`, the logSNR is `−2 log tan(a·t + b)` with `a` and `b`
 set by the configuration's logSNR range, `α² = sigmoid(logSNR)`, and `σ² = sigmoid(−logSNR)`. The clean
 clip noises to `α·x + σ·z`, the network reads it at the normalized level `(max − logSNR) / (max − min)`,
 and the loss is `mean |ε̂ − z|`. Measured against the reference by `run_reference.py nuwave2_loss`.
 Introduced in InferKit 0.5.0.
 */
public struct NFKMLXNUWave2Objective: Sendable {

    public init() {}

    /// The logSNR at diffusion times `[B]` in `(0, 1]`, the reference's `Diffusion.snr`.
    public func logSNR(time: MLXArray, configuration c: NFKMLXNUWave2Configuration) -> MLXArray {
        let b = atan(exp(-Double(c.logSNRMaximum) / 2))
        let a = atan(exp(-Double(c.logSNRMinimum) / 2)) - b
        return -2 * log(tan(Float(a) * time + Float(b)))
    }

    /// The loss of `net` on clean clips `[B, L]`, their narrow-band inputs `[B, L]`, bands `[B, bins]`
    /// (0/1), diffusion times `[B]`, and the noise draw `[B, L]`.
    public func loss(_ net: NFKMLXNUWave2Net, audio: MLXArray, narrowband: MLXArray, band: MLXArray,
                     time: MLXArray, noise: MLXArray) -> MLXArray {
        let c = net.configuration
        let logSNR = logSNR(time: time, configuration: c)
        let alpha = sigmoid(logSNR).sqrt().expandedDimensions(axis: -1)
        let sigma = sigmoid(-logSNR).sqrt().expandedDimensions(axis: -1)
        let noised = alpha * audio + sigma * noise
        let level = (c.logSNRMaximum - logSNR) / (c.logSNRMaximum - c.logSNRMinimum)
        let estimate = net(noised, narrowband: narrowband, band: band, level: level)
        return abs(estimate - noise).mean()
    }

    /// The reference's time draw for a batch of `count`: one uniform offset, then evenly spaced
    /// strata, `((1 − u) + i / count) mod 1`.
    public static func time(count: Int) -> MLXArray {
        let offset = 1 - MLXRandom.uniform(low: 0, high: 1, [1])
        let strata = MLXArray((0 ..< count).map { Float($0) / Float(count) })
        return remainder(offset + strata, 1)
    }
}

extension NFKMLXNUWave2 {

    /// Builds the bandwidth-extension network itself, ready to fine-tune, from the official checkpoint
    /// or a file `NFKMLXWeights.save` wrote.
    ///
    /// - Since: InferKit 0.5.0
    public static func network(weightsURL: URL?) throws -> NFKMLXNUWave2Net {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// A training pair built the way `dataloader.py` builds one: the 48 kHz clip peak-normalized and
    /// trimmed to a multiple of the hop, a narrow-band copy band-limited to `narrowbandRate` and brought
    /// back to 48 kHz, and the band marking the bins below `narrowbandRate / 2`. The random gain is
    /// applied by the recipe per step.
    ///
    /// - Since: InferKit 0.5.0
    public static func trainingPair(wideband samples: [Float], narrowbandRate: Int,
                                    configuration c: NFKMLXNUWave2Configuration = NFKMLXNUWave2Configuration())
        -> (audio: MLXArray, narrowband: MLXArray, band: MLXArray) {
        let peak = samples.map { abs($0) }.max() ?? 0
        let normalized = samples.map { $0 / (peak > 0 ? peak : 1) }
        let wide = Array(normalized.prefix(normalized.count - normalized.count % c.hopSize))
        var narrow = NFKMLXAudioRate.matched(NFKMLXAudioRate.matched(wide, from: c.sampleRate, to: narrowbandRate),
                                             from: narrowbandRate, to: c.sampleRate)
        if narrow.count < wide.count {
            narrow += [Float](repeating: 0, count: wide.count - narrow.count)
        }
        narrow = Array(narrow.prefix(wide.count))
        let cutoff = Int(Double(narrowbandRate / 2) / (0.5 * Double(c.sampleRate)) * Double(c.bins))
        let band = MLXArray((0 ..< c.bins).map { Int32($0 < cutoff ? 1 : 0) }).reshaped([1, c.bins])
        return (MLXArray(wide).reshaped([1, wide.count]), MLXArray(narrow).reshaped([1, narrow.count]), band)
    }

    /// Fine-tunes every weight of `net` on wide-band audio, returning the loss from each step.
    ///
    /// The whole path is three calls: ``network(weightsURL:)`` to build, this to train, and
    /// `NFKMLXWeights.save` to write a checkpoint that `backendWithWeightsURL:error:` loads.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:)``.
    ///   - examples: supplies one pair per step, from ``trainingPair(wideband:narrowbandRate:configuration:)``.
    ///     The reference trains on 32,768-sample segments (about 0.7 s at 48 kHz) and draws the
    ///     narrow-band rate between 6 and 48 kHz.
    ///   - objective: the reference's noise-prediction L1.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.Adam` (bias-corrected) at
    ///     `hparameter.yaml`'s learning rate 2e-4, betas 0.9 and 0.99, and epsilon 1e-9.
    ///   - steps: how many pairs to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference does not clip.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil holds it constant, as the reference does.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is minutes; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXNUWave2Net,
        examples: (Int) -> (audio: MLXArray, narrowband: MLXArray, band: MLXArray),
        objective: NFKMLXNUWave2Objective = NFKMLXNUWave2Objective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { Adam(learningRate: 2e-4, betas: (0.9, 0.99), eps: 1e-9, biasCorrection: true) },
            referenceSchedule: { .constant },
            steps: steps,
            arrays: { step in
                let pair = examples(step)
                let gain = MLXRandom.uniform(low: 0, high: 1, [pair.audio.dim(0), 1]) / 2 + 0.5
                let audio = pair.audio * gain
                let narrowband = pair.narrowband * gain
                return [audio, narrowband, pair.band, NFKMLXNUWave2Objective.time(count: audio.dim(0)),
                        MLXRandom.normal(audio.shape)]
            },
            loss: { net, arrays in
                objective.loss(net, audio: arrays[0], narrowband: arrays[1], band: arrays[2],
                               time: arrays[3], noise: arrays[4])
            },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }
}
