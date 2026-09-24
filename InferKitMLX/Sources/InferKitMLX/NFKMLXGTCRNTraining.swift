//
//  NFKMLXGTCRNTraining.swift
//  InferKitMLX
//
//  Fine-tuning GTCRN on a consumer's own noisy and clean recordings: their room, their microphone,
//  their kind of noise. At 48.2K parameters every weight trains on a device, so the level is a full
//  fine-tune.
//
//  The objective is the repo's own `HybridLoss` (`loss.py`): mean squared errors on the power-law
//  compressed real, imaginary, and magnitude spectra, weighted 30, 30, and 70, plus the negative log of
//  the scale-invariant signal-to-noise ratio of the resynthesized waveforms. It has no adversary. The
//  SI-SNR term runs through an inverse STFT, so the synthesis here is built from differentiable array
//  operations; the inference path's overlap-add runs on the host and carries no gradient.
//
//  The repo publishes the loss and no training script (training lives in the author's separate SEtrain
//  template), so the optimizer is this package's choice.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/*!
 @abstract The objective a GTCRN fine-tune minimizes: the reference's `HybridLoss`.
 @discussion Scores a predicted spectrogram against a clean one, each real and imaginary `[1, bins,
 frames]` in the network's layout. `30·(real + imaginary) + 70·magnitude − log₁₀ SI-SNR`, where each
 spectral term is the mean squared error after compressing the magnitude by the power 0.3 (and each
 real or imaginary part by dividing by the magnitude to the 0.7), and the SI-SNR is measured on the
 waveforms the sqrt-Hann inverse STFT resynthesizes. Measured against the reference module by
 `run_reference.py gtcrn_loss`. Introduced in InferKit 0.5.0.
 */
public struct NFKMLXGTCRNObjective: Sendable {

    /// The weight on each compressed real and imaginary term. The reference's is 30.
    public var complexWeight: Float

    /// The weight on the compressed magnitude term. The reference's is 70.
    public var magnitudeWeight: Float

    /// The weight on the negative log SI-SNR term. The reference's is 1.
    public var siSNRWeight: Float

    public init(complexWeight: Float = 30, magnitudeWeight: Float = 70, siSNRWeight: Float = 1) {
        self.complexWeight = complexWeight
        self.magnitudeWeight = magnitudeWeight
        self.siSNRWeight = siSNRWeight
    }

    /// The four terms before they are weighted, each a scalar, so a parity run can say which one
    /// disagrees.
    public func components(predicted: (real: MLXArray, imaginary: MLXArray),
                           target: (real: MLXArray, imaginary: MLXArray))
        -> (real: MLXArray, imaginary: MLXArray, magnitude: MLXArray, siSNR: MLXArray) {
        let predictedMagnitude = sqrt(predicted.real.square() + predicted.imaginary.square() + 1e-12)
        let targetMagnitude = sqrt(target.real.square() + target.imaginary.square() + 1e-12)
        let predictedScale = pow(predictedMagnitude, Float(0.7))
        let targetScale = pow(targetMagnitude, Float(0.7))
        let real = (predicted.real / predictedScale - target.real / targetScale).square().mean()
        let imaginary = (predicted.imaginary / predictedScale - target.imaginary / targetScale).square().mean()
        let magnitude = (pow(predictedMagnitude, Float(0.3)) - pow(targetMagnitude, Float(0.3))).square().mean()

        let estimate = NFKGTCRNSynthesis.waveform(real: predicted.real, imaginary: predicted.imaginary)
        let reference = NFKGTCRNSynthesis.waveform(real: target.real, imaginary: target.imaginary)
        let projection = (reference * estimate).sum(axis: -1, keepDims: true) * reference
            / (reference.square().sum(axis: -1, keepDims: true) + 1e-8)
        let ratio = projection.square().sum(axis: -1, keepDims: true)
            / ((estimate - projection).square().sum(axis: -1, keepDims: true) + 1e-8)
        let siSNR = -log10(ratio + 1e-8).mean()
        return (real, imaginary, magnitude, siSNR)
    }

    /// The weighted loss, a scalar.
    public func loss(predicted: (real: MLXArray, imaginary: MLXArray),
                     target: (real: MLXArray, imaginary: MLXArray)) -> MLXArray {
        let terms = components(predicted: predicted, target: target)
        return complexWeight * (terms.real + terms.imaginary) + magnitudeWeight * terms.magnitude
            + siSNRWeight * terms.siSNR
    }

    /// Scores `net` on one pair: the noisy spectrogram it enhances and the clean one it should reach.
    public func callAsFunction(_ net: NFKMLXGTCRN, noisy: (real: MLXArray, imaginary: MLXArray),
                               clean: (real: MLXArray, imaginary: MLXArray)) -> MLXArray {
        let enhanced = net(real: noisy.real, imaginary: noisy.imaginary)
        return loss(predicted: enhanced, target: clean)
    }
}

/// GTCRN's inverse STFT from array operations, so a gradient reaches the spectrogram: 512-point frames,
/// hop 256, the sqrt-Hann window, and `torch.istft`'s window-squared normalization with the center
/// padding removed.
enum NFKGTCRNSynthesis {
    static let fftSize = 512
    static let hopSize = 256

    /// The sqrt-Hann window `infer.py` and `loss.py` use, `[fftSize]`.
    static let window = MLXArray(nfkPeriodicHann(fftSize).map { sqrtf($0) })

    /// The real inverse DFT as two matrices `[bins, fftSize]`, so `irfft` is a pair of matrix
    /// multiplications. The DC and Nyquist rows count once and their imaginary parts drop, as `irfft`'s
    /// do; every other bin counts twice.
    static let inverseBasis: (cosine: MLXArray, sine: MLXArray) = {
        let bins = fftSize / 2 + 1
        var cosine = [Float](repeating: 0, count: bins * fftSize)
        var sine = [Float](repeating: 0, count: bins * fftSize)
        for k in 0 ..< bins {
            let edge = k == 0 || k == bins - 1
            let scale = (edge ? 1 : 2) / Float(fftSize)
            for n in 0 ..< fftSize {
                let angle = 2 * Double.pi * Double(k * n % fftSize) / Double(fftSize)
                cosine[k * fftSize + n] = scale * Float(cos(angle))
                sine[k * fftSize + n] = edge ? 0 : scale * Float(sin(angle))
            }
        }
        return (MLXArray(cosine, [bins, fftSize]), MLXArray(sine, [bins, fftSize]))
    }()

    /// Real and imaginary `[1, bins, frames]` → samples `[1, (frames − 1) · hop]`.
    static func waveform(real: MLXArray, imaginary: MLXArray) -> MLXArray {
        let frames = real.dim(2)
        let basis = inverseBasis
        let time = matmul(real.transposed(0, 2, 1), basis.cosine)
            - matmul(imaginary.transposed(0, 2, 1), basis.sine)                // [1, frames, fftSize]
        let summed = overlapAdd(time * window.reshaped([1, 1, fftSize]), frames: frames)
        let envelope = overlapAdd((window * window).reshaped([1, 1, fftSize]), frames: frames, broadcastFrames: true)
        let pad = fftSize / 2
        let length = (frames - 1) * hopSize
        let normalized = summed / maximum(envelope, 1e-11)
        return normalized[0..., pad ..< (pad + length)]
    }

    /// Sums frames `[1, frames, fftSize]` at hop spacing into `[1, (frames − 1) · hop + fftSize]`. The
    /// frame splits into `fftSize / hop` chunks, and chunk `k` of frame `f` lands at block `f + k`.
    private static func overlapAdd(_ framed: MLXArray, frames: Int, broadcastFrames: Bool = false) -> MLXArray {
        let overlap = fftSize / hopSize
        let source = broadcastFrames ? broadcast(framed, to: [1, frames, fftSize]) : framed
        let chunks = source.reshaped([1, frames, overlap, hopSize])
        var blocks = MLXArray.zeros([1, frames + overlap - 1, hopSize])
        for k in 0 ..< overlap {
            let chunk = chunks[0..., 0..., k, 0...]                             // [1, frames, hop]
            blocks = blocks + MLX.padded(chunk, widths: [IntOrPair((0, 0)), IntOrPair((k, overlap - 1 - k)),
                                                         IntOrPair((0, 0))])
        }
        return blocks.reshaped([1, (frames + overlap - 1) * hopSize])
    }
}

extension NFKMLXGTCRNFactory {

    /// Builds the enhancement network itself, ready to fine-tune, from an optional released
    /// checkpoint or a file `NFKMLXWeights.save` wrote.
    ///
    /// - Since: InferKit 0.5.0
    public static func network(weightsURL: URL?) throws -> NFKMLXGTCRN {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// The spectrogram the network reads, real and imaginary `[1, bins, frames]`, of 16 kHz samples:
    /// the 512-point sqrt-Hann STFT at hop 256 the reference uses.
    ///
    /// - Since: InferKit 0.5.0
    public static func spectrogram(for samples: [Float]) -> (real: MLXArray, imaginary: MLXArray) {
        let stft = NFKMLXComplexSTFT(nFFT: NFKGTCRNSynthesis.fftSize, hop: NFKGTCRNSynthesis.hopSize,
                                     window: NFKGTCRNSynthesis.window)
        let signal = samples.withUnsafeBufferPointer { MLXArray($0, [1, samples.count]) }
        return stft.transformComplex(signal)
    }

    /// Fine-tunes every weight of `net` on noisy and clean recordings, returning the loss from each step.
    ///
    /// The whole path is three calls: ``network(weightsURL:)`` to build, this to train, and
    /// `NFKMLXWeights.save` to write a checkpoint that `backendWithWeightsURL:error:` loads.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:)``.
    ///   - examples: supplies one pair per step: the noisy recording and the same speech clean, both
    ///     16 kHz samples. A pair of unequal lengths trains on the shorter one's span.
    ///   - objective: the reference's `HybridLoss`.
    ///   - optimizer: the update rule. Nil uses bias-corrected Adam at 1e-3; the repo publishes no
    ///     training script, so the rate is this package's choice.
    ///   - steps: how many pairs to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil holds it constant.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The batch normalizations train as the reference's do, on each clip's own statistics. A run is
    /// seconds to minutes; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXGTCRN,
        examples: (Int) -> (noisy: [Float], clean: [Float]),
        objective: NFKMLXGTCRNObjective = NFKMLXGTCRNObjective(),
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
            reference: { Adam(learningRate: 1e-3, biasCorrection: true) },
            referenceSchedule: { .constant },
            steps: steps,
            arrays: { step in
                let pair = examples(step)
                let length = min(pair.noisy.count, pair.clean.count)
                let noisy = spectrogram(for: Array(pair.noisy[0 ..< length]))
                let clean = spectrogram(for: Array(pair.clean[0 ..< length]))
                return [noisy.real, noisy.imaginary, clean.real, clean.imaginary]
            },
            loss: { net, arrays in
                objective(net, noisy: (arrays[0], arrays[1]), clean: (arrays[2], arrays[3]))
            },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }
}
