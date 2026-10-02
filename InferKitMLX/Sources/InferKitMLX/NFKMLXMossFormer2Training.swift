//
//  NFKMLXMossFormer2Training.swift
//  InferKitMLX
//
//  MossFormer2 SE 48K's training path, ported from ClearerVoice-Studio's `train/speech_enhancement`:
//  `psm_loss`, the loader's Kaldi features with their optional dither, and `train.py`'s Adam.
//

import Foundation
import MLX
import MLXFFT
import MLXNN
import MLXOptimizers

/// The objective `loss_mossformer2_se_48k` computes: `psm_loss`, the squared error between the predicted
/// mask and the phase-sensitive mask, weighted by the noisy magnitude.
///
/// @discussion The target is `|S|² / |Y|² · cos(∠S − ∠Y)` clamped to [0, 1], from `torch.stft` with a
/// symmetric Hamming window and no centering. Each bin's squared error is weighted by the noisy magnitude
/// over the loudest bin of its frame; the sum is halved and divided by the batch's frame count. The clips
/// are scaled by 32768 first, as the reference's loader scales them.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXMossFormer2Objective: Sendable {

    /// The STFT geometry the target mask is computed with.
    public var configuration: NFKMLXMossFormer2Configuration

    public init(configuration: NFKMLXMossFormer2Configuration = NFKMLXMossFormer2Configuration()) {
        self.configuration = configuration
    }

    /// Scores the network on its features `[N, frames, 180]` against the noisy clips they were computed
    /// from and the clean speech under them, each `[N, L]` in [-1, 1].
    public func callAsFunction(_ net: NFKMLXMossFormer2SENet, _ feature: MLXArray, _ noisy: MLXArray,
                               _ clean: MLXArray) -> MLXArray {
        loss(noisy: noisy, clean: clean, mask: net(feature))
    }

    /// The loss for a predicted mask `[N, frames, bins]` on noisy and clean clips `[N, L]` in [-1, 1].
    public func loss(noisy: MLXArray, clean: MLXArray, mask: MLXArray) -> MLXArray {
        let eps: Float = 1e-6
        let scale = NFKMLXMossFormer2Backend.waveScale
        let y = Self.spectrum(noisy * scale, configuration)
        let s = Self.spectrum(clean * scale, configuration)
        let noisyMagnitude = sqrt(square(y.real) + square(y.imaginary))
        let cleanMagnitude = sqrt(square(s.real) + square(s.imaginary))
        let weight = noisyMagnitude / (noisyMagnitude.max(axis: -1, keepDims: true) + 1e-6)
        let cosine = (y.real / (noisyMagnitude + eps)) * (s.real / (cleanMagnitude + eps))
            + (y.imaginary / (noisyMagnitude + eps)) * (s.imaginary / (cleanMagnitude + eps))
        let target = clip(square(cleanMagnitude) / (square(noisyMagnitude) + eps) * cosine, min: 0, max: 1)
        let frames = mask[0..., 0 ..< y.real.dim(1), 0...]
        return 0.5 * (square(frames - target) * weight).sum() / Float(y.real.dim(0) * y.real.dim(1))
    }

    /// `utils.misc.stft` on clips `[N, L]`: `torch.stft` uncentered with
    /// `torch.hamming_window(win_len, periodic=False)`, real and imaginary each `[N, frames, bins]`.
    static func spectrum(_ clips: MLXArray, _ configuration: NFKMLXMossFormer2Configuration)
        -> (real: MLXArray, imaginary: MLXArray) {
        let size = configuration.fftLen, hop = configuration.winInc
        let frames = 1 + (clips.dim(1) - size) / hop
        let indices = MLXArray((0 ..< frames).flatMap { f in (0 ..< size).map { Int32(f * hop + $0) } }, [frames, size])
        let window = NFKMossSTFT(nFFT: size, hop: hop).window
        let framed = clips[0..., indices] * window
        let transformed = MLXFFT.rfft(framed, axis: -1)
        return (transformed.realPart(), transformed.imaginaryPart())
    }
}

public extension NFKMLXMossFormer2Factory {

    /// The training loader's batch, 4 clips. Introduced in InferKit 0.4.0.
    static let referenceBatchSize = 4

    /// The batches each update accumulates, `effec_batch_size` 8 over `batch_size` 4. Introduced in
    /// InferKit 0.4.0.
    static let referenceAccumulationSteps = 2

    /// The length the loader cuts each recording to, `max_length` 4 seconds at 48 kHz. Introduced in
    /// InferKit 0.4.0.
    static let referenceSegmentSeconds = 4

    /// The loader's fbank dither, `Fbank_Processor`'s 1.0 in 16-bit sample units. Introduced in
    /// InferKit 0.4.0.
    static let referenceDither: Float = 1

    /// Builds the MossFormer2 SE network for training or for reloading a trained checkpoint. With a
    /// `weightsURL` the released `last_best_checkpoint.pt` or a file `NFKMLXWeights` saved loads; without
    /// one the network is randomly initialized. Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?,
                        configuration: NFKMLXMossFormer2Configuration = NFKMLXMossFormer2Configuration())
        throws -> NFKMLXMossFormer2SENet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// The optimizer `train.py` builds for MossFormer2 SE: `torch.optim.Adam` at 5e-4 with no weight
    /// decay. Introduced in InferKit 0.4.0.
    static func referenceOptimizer() -> Optimizer {
        NFKMLXReferenceOptimizers.l2Adam(learningRate: 5e-4, weightDecay: 0)
    }

    /// The network's input for noisy clips `[N, L]` in [-1, 1]: each clip's Kaldi fbank with its deltas,
    /// `[N, frames, 180]`, computed on the clip scaled by 32768 as the reference's loader does. A
    /// positive `dither` adds Gaussian noise of that deviation, in 16-bit sample units, to every framed
    /// sample, drawn from `key`. Introduced in InferKit 0.4.0.
    static func features(_ noisy: MLXArray, configuration: NFKMLXMossFormer2Configuration = NFKMLXMossFormer2Configuration(),
                         dither: Float = 0, key: MLXArray? = nil) -> MLXArray {
        let frameLength = NFKMLXKaldiFbank.framing(configuration).length
        let frames = NFKMLXKaldiFbank.frameCount(noisy.dim(1), config: configuration)
        let keys = dither > 0 ? MLXRandom.split(key: key ?? MLXRandom.key(0), into: noisy.dim(0)) : []
        return concatenated((0 ..< noisy.dim(0)).map { clip in
            let samples = (noisy[clip] * NFKMLXMossFormer2Backend.waveScale).asArray(Float.self)
            let noise = dither > 0
                ? (MLXRandom.normal([frames * frameLength], key: keys[clip]) * dither).asArray(Float.self) : nil
            return NFKMLXKaldiFbank.features(samples: samples, config: configuration, dither: noise)
        }, axis: 0)
    }

    /// Trains MossFormer2 SE on pairs of noisy and clean speech.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:configuration:)``.
    ///   - examples: supplies one batch per step: noisy clips and the clean speech under them, each
    ///     `[N, L]` mono at 48 kHz in [-1, 1]. The reference loader scales each recording with
    ///     `NFKMLXTrainingData.speechLevelNormalized(_:)` before cutting clips of
    ///     ``referenceSegmentSeconds``.
    ///   - objective: the loss.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer()``.
    ///   - steps: how many updates to train for.
    ///   - dither: the fbank dither, in 16-bit sample units. The reference loader's is
    ///     ``referenceDither``; zero computes the features the released model infers from.
    ///   - ditherSeed: seeds the dither, so a run repeats.
    ///   - clipGradientNorm: bounds the global gradient norm, the reference's 10 by default. The reference
    ///     clips after every batch's backward pass within an accumulation; this clips the averaged
    ///     gradient once per update.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates. The
    ///     reference accumulates ``referenceAccumulationSteps``.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is a constant rate; the reference halves
    ///     the rate after five epochs without a better validation loss, which needs a validation set.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Every parameter trains, with the configuration's dropout. Save with `NFKMLXWeights.save`;
    /// ``network(weightsURL:configuration:)`` and `backend(weightsURL:)` read the file back. A run is
    /// minutes; call it off the render thread. Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXMossFormer2SENet,
        examples: (Int) -> (noisy: MLXArray, clean: MLXArray),
        objective: NFKMLXMossFormer2Objective = NFKMLXMossFormer2Objective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        dither: Float = 0,
        ditherSeed: UInt64 = 777,
        clipGradientNorm: Float? = 10,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let configuration = objective.configuration
        return try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { referenceOptimizer() },
            referenceSchedule: { .constant },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                let key = MLXRandom.key(ditherSeed &+ UInt64(step))
                return [features(example.noisy, configuration: configuration, dither: dither, key: key),
                        example.noisy, example.clean]
            },
            loss: { net, arrays in objective(net, arrays[0], arrays[1], arrays[2]) },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
