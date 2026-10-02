//
//  NFKMLXFRCRNTraining.swift
//  InferKitMLX
//
//  Training FRCRN on a consumer's own noisy and clean recordings, as ClearerVoice-Studio's
//  `train/speech_enhancement` trains `FRCRN_SE_16K`: every parameter, the complex ratio mask held to the
//  one the clean and noisy spectra imply, the scale-invariant SNR of the resynthesized waveform, and
//  `torch.optim.Adam` at 1e-3 with L2 decay on every weight but the biases.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

public extension NFKMLXTrainingData {

    /// ClearerVoice-Studio's `audio_norm`, which its training loader applies to every recording before
    /// cropping: the recording is scaled to an RMS of −25 dBFS, then again so that the RMS of its
    /// samples above the mean power is −25 dBFS. Noisy and clean recordings are each scaled on their
    /// own. Introduced in InferKit 0.4.0.
    static func speechLevelNormalized(_ samples: [Float]) -> [Float] {
        let target = pow(10, -25.0 / 20)
        var x = samples.map(Double.init)
        let rms = (x.reduce(0) { $0 + $1 * $1 } / Double(max(x.count, 1))).squareRoot()
        let first = target / (rms + 1e-6)
        x = x.map { $0 * first }
        let power = x.map { $0 * $0 }
        let mean = power.reduce(0, +) / Double(max(power.count, 1))
        let loud = power.filter { $0 > mean }
        let active = (loud.reduce(0, +) / Double(max(loud.count, 1))).squareRoot()
        let second = target / (active + 1e-6)
        return x.map { Float($0 * second) }
    }
}

/// FRCRN's inverse STFT from array operations, so a gradient reaches the masked spectrum: the
/// reference `ConviSTFT`, a transposed convolution with the pseudo-inverse of the analysis basis and the
/// square-root periodic Hann window, divided by the overlap-added squared window plus 1e-8. The
/// pseudo-inverse is the real inverse DFT: the basis's imaginary DC and Nyquist rows are zero, so both
/// ignore those parts.
struct NFKFRCRNSynthesis {
    let fftSize: Int
    let hopSize: Int
    let window: MLXArray
    let cosine: MLXArray
    let sine: MLXArray

    init(_ configuration: NFKMLXFRCRNConfiguration) {
        fftSize = configuration.fftSize
        hopSize = configuration.hopSize
        window = MLXArray(nfkPeriodicHann(fftSize).map { $0.squareRoot() })
        let bins = fftSize / 2 + 1
        var cosine = [Float](repeating: 0, count: bins * fftSize)
        var sine = [Float](repeating: 0, count: bins * fftSize)
        for k in 0 ..< bins {
            let edge = k == 0 || k == bins - 1
            let scale = (edge ? 1 : 2) / Double(fftSize)
            for n in 0 ..< fftSize {
                let angle = 2 * Double.pi * Double(k * n % fftSize) / Double(fftSize)
                cosine[k * fftSize + n] = Float(scale * cos(angle))
                sine[k * fftSize + n] = edge ? 0 : Float(scale * sin(angle))
            }
        }
        self.cosine = MLXArray(cosine, [bins, fftSize])
        self.sine = MLXArray(sine, [bins, fftSize])
    }

    /// Real and imaginary `[B, bins, frames]` → samples `[B, (frames − 1) · hop + fftSize]`.
    func waveform(real: MLXArray, imaginary: MLXArray) -> MLXArray {
        let (batch, frames) = (real.dim(0), real.dim(2))
        let time = matmul(real.transposed(0, 2, 1), cosine) - matmul(imaginary.transposed(0, 2, 1), sine)
        let summed = overlapAdd(time * window, batch: batch, frames: frames)
        let envelope = overlapAdd(broadcast(window * window, to: [1, frames, fftSize]), batch: 1, frames: frames)
        return summed / (envelope + 1e-8)
    }

    /// Sums frames `[B, frames, fftSize]` at hop spacing; chunk `k` of frame `f` lands at block `f + k`.
    private func overlapAdd(_ framed: MLXArray, batch: Int, frames: Int) -> MLXArray {
        let overlap = fftSize / hopSize
        let chunks = framed.reshaped([batch, frames, overlap, hopSize])
        var total = MLXArray.zeros([batch, (frames + overlap - 1) * hopSize])
        for k in 0 ..< overlap {
            let chunk = chunks[0..., 0..., k, 0...].reshaped([batch, frames * hopSize])
            total = total + MLX.padded(chunk, widths: [IntOrPair((0, 0)), IntOrPair((k * hopSize, (overlap - 1 - k) * hopSize))])
        }
        return total
    }
}

/// The objective `loss_frcrn_se_16k` computes: the complex-mask MSE plus the negative SI-SNR.
///
/// @discussion The target mask is the clean spectrum divided by the noisy one, each from `torch.stft`
/// with a symmetric Hann window (the training loss's STFT, not the model's). Its real and imaginary
/// halves are each scored by mean squared error times the FFT size. A target entry above 2 becomes 1,
/// and one below −2 becomes −1, as the reference's clamp writes them.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXFRCRNObjective: Sendable {

    public init() {}

    /// Scores the network on noisy clips `[N, L]` against the clean speech under them, `[N, L]`, where
    /// `L − fftSize` is a multiple of the hop.
    public func callAsFunction(_ net: NFKMLXFRCRNNet, _ noisy: MLXArray, _ clean: MLXArray) -> MLXArray {
        let outputs = net.trainingOutputs(noisy)
        return loss(noisy: noisy, clean: clean, estimate: outputs.waveform, maskReal: outputs.maskReal,
                    maskImaginary: outputs.maskImaginary, configuration: net.configuration).total
    }

    /// The loss and its two terms for an estimate `[N, L]` and the network's mask, real and imaginary
    /// each `[N, bins, frames]`.
    public func loss(noisy: MLXArray, clean: MLXArray, estimate: MLXArray, maskReal: MLXArray,
                     maskImaginary: MLXArray, configuration: NFKMLXFRCRNConfiguration = .init())
        -> (total: MLXArray, mask: MLXArray, scaleInvariantSNR: MLXArray) {
        let stft = Self.lossSTFT(configuration)
        let s = Self.spectrum(clean, stft), y = Self.spectrum(noisy, stft)
        let power = square(y.real) + square(y.imaginary)
        func clamped(_ m: MLXArray) -> MLXArray {
            MLX.where(m .> 2, MLXArray(Float(1)), MLX.where(m .< -2, MLXArray(Float(-1)), m))
        }
        let targetReal = clamped((s.real * y.real + s.imaginary * y.imaginary) / (power + 1e-8))
        let targetImaginary = clamped((s.imaginary * y.real - s.real * y.imaginary) / (power + 1e-8))
        let size = Float(configuration.fftSize)
        let maskLoss = square(targetReal - maskReal).mean() * size + square(targetImaginary - maskImaginary).mean() * size
        let snr = -Self.scaleInvariantSNR(clean: clean, estimate: estimate).mean()
        return (maskLoss + snr, maskLoss, snr)
    }

    /// The combined mask `tanh(unet2) + tanh(unet1)` as real and imaginary arrays.
    static func combine(_ unet1: NFKFRCRNComplex, _ unet2: NFKFRCRNComplex) -> [MLXArray] {
        [tanh(unet2.real) + tanh(unet1.real), tanh(unet2.imaginary) + tanh(unet1.imaginary)]
    }

    /// `cal_SISNR` per clip `[N]`, at its epsilon of 1e-6.
    static func scaleInvariantSNR(clean: MLXArray, estimate: MLXArray) -> MLXArray {
        let eps: Float = 1e-6
        let source = clean - clean.mean(axis: -1, keepDims: true)
        let estimated = estimate - estimate.mean(axis: -1, keepDims: true)
        let energy = square(source).sum(axis: -1, keepDims: true) + eps
        let projection = (source * estimated).sum(axis: -1, keepDims: true) * source / energy
        let noise = estimated - projection
        let ratio = square(projection).sum(axis: -1) / (square(noise).sum(axis: -1) + eps)
        return 10 * log10(ratio + eps)
    }

    /// `utils.misc.stft`: `torch.stft` uncentered, with `torch.hann_window(win_len, periodic=False)`.
    static func lossSTFT(_ configuration: NFKMLXFRCRNConfiguration) -> NFKMLXComplexSTFT {
        let n = configuration.fftSize
        let window = (0 ..< n).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n - 1))) }
        return NFKMLXComplexSTFT(nFFT: n, hop: configuration.hopSize, window: MLXArray(window), centered: false)
    }

    /// The spectrum of each clip of `[N, L]`, real and imaginary each `[N, bins, frames]`.
    static func spectrum(_ clips: MLXArray, _ stft: NFKMLXComplexSTFT) -> (real: MLXArray, imaginary: MLXArray) {
        let rows = (0 ..< clips.dim(0)).map { stft.transformComplex(clips[$0 ..< ($0 + 1)]) }
        return (concatenated(rows.map(\.real), axis: 0), concatenated(rows.map(\.imaginary), axis: 0))
    }
}

extension NFKMLXFRCRNNet {

    /// `DCCRN.forward` on noisy clips `[N, L]`: the resynthesized estimate `[N, L]` and the combined
    /// mask, real and imaginary each `[N, bins, frames]`.
    func trainingOutputs(_ noisy: MLXArray) -> (waveform: MLXArray, maskReal: MLXArray, maskImaginary: MLXArray) {
        let analysis = NFKMLXFRCRNObjective.spectrum(noisy, NFKMLXFRCRNBackend.stft(configuration))
        let spectrum = NFKFRCRNComplex(real: analysis.real.expandedDimensions(axis: 3),
                                       imaginary: analysis.imaginary.expandedDimensions(axis: 3))
        let m = mask(spectrum).mask
        let real = (spectrum.real * m.real - spectrum.imaginary * m.imaginary).squeezed(axis: 3)
        let imaginary = (spectrum.real * m.imaginary + spectrum.imaginary * m.real).squeezed(axis: 3)
        let waveform = NFKFRCRNSynthesis(configuration).waveform(real: real, imaginary: imaginary)
        return (waveform, m.real.squeezed(axis: 3), m.imaginary.squeezed(axis: 3))
    }
}

public extension NFKMLXFRCRN {

    /// The training loader's batch, 4 one-second clips. Introduced in InferKit 0.4.0.
    static let referenceBatchSize = 4

    /// The batches each update accumulates, `effec_batch_size` 12 over `batch_size` 4. Introduced in
    /// InferKit 0.4.0.
    static let referenceAccumulationSteps = 3

    /// Builds FRCRN for training or for reloading a trained checkpoint. With a `weightsURL` the released
    /// `last_best_checkpoint.pt` or a file `NFKMLXWeights` saved loads; without one the network is
    /// randomly initialized. Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?) throws -> NFKMLXFRCRNNet {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// The optimizer `train.py` builds from `get_params`: `torch.optim.Adam` at 1e-3 with an L2 weight
    /// decay of 1e-5 on every parameter but the biases. Introduced in InferKit 0.4.0.
    static func referenceOptimizer() -> Optimizer {
        NFKMLXReferenceOptimizers.l2Adam(learningRate: 1e-3, weightDecay: 1e-5) { $0.contains("bias") }
    }

    /// Trains FRCRN on pairs of noisy and clean speech.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:)``.
    ///   - examples: supplies one batch per step: noisy clips and the clean speech under them, each
    ///     `[N, L]` mono at 16 kHz. The reference loader scales each recording with
    ///     `NFKMLXTrainingData.speechLevelNormalized(_:)` before cutting one-second clips. A clip is cut
    ///     to the longest length whose frames the STFT covers exactly, `640 + 320k` samples.
    ///   - objective: the loss.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer()``.
    ///   - steps: how many updates to train for.
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
    /// Every parameter trains, the batch normalizations on each batch's statistics. Save with
    /// `NFKMLXWeights.save`; ``network(weightsURL:)`` and `backend(weightsURL:)` read the file back. A run is
    /// minutes; call it off the render thread. Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXFRCRNNet,
        examples: (Int) -> (noisy: MLXArray, clean: MLXArray),
        objective: NFKMLXFRCRNObjective = NFKMLXFRCRNObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 10,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let configuration = net.configuration
        return try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { referenceOptimizer() },
            referenceSchedule: { .constant },
            steps: steps,
            batch: { step in
                let example = examples(step)
                let usable = (example.noisy.dim(1) - configuration.fftSize) / configuration.hopSize * configuration.hopSize
                    + configuration.fftSize
                return (example.noisy[0..., 0 ..< usable], example.clean[0..., 0 ..< usable])
            },
            loss: objective.callAsFunction,
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
