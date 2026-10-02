//
//  NFKMLXDenoiserTraining.swift
//  InferKitMLX
//
//  Training the speech denoiser on a consumer's own noisy and clean recordings, as facebookresearch/
//  denoiser's `solver.py` trains it: every parameter, a waveform distance between the estimate and the
//  clean speech (L1 for the released DNS models), an optional multi-resolution STFT term, Adam at 3e-4,
//  and the solver's four augmentations over the noise and the speech before they are summed.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The objective `solver.py` minimizes: a distance between the estimate and the clean waveform, plus the
/// multi-resolution STFT loss when it is on.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXDenoiserObjective: Sendable {

    /// The waveform distance, the reference's `loss` setting.
    public enum Distance: Sendable {
        /// `F.l1_loss`, the released DNS models' setting.
        case l1
        /// `F.mse_loss`.
        case l2
        /// `F.smooth_l1_loss` at its default threshold of 1.
        case huber
    }

    public var distance: Distance
    /// Whether the multi-resolution STFT loss is added, the reference's `stft_loss`.
    public var stftLoss: Bool
    /// The spectral-convergence term's factor, the reference's `stft_sc_factor`.
    public var stftConvergenceFactor: Float
    /// The log-magnitude term's factor, the reference's `stft_mag_factor`.
    public var stftMagnitudeFactor: Float

    public init(distance: Distance = .l1, stftLoss: Bool = false, stftConvergenceFactor: Float = 0.5,
                stftMagnitudeFactor: Float = 0.5) {
        self.distance = distance
        self.stftLoss = stftLoss
        self.stftConvergenceFactor = stftConvergenceFactor
        self.stftMagnitudeFactor = stftMagnitudeFactor
    }

    /// `launch_dns.sh`, the recipe the released DNS models train under: L1 alone.
    public static let dns = NFKMLXDenoiserObjective()

    /// `launch_valentini.sh`: L1 plus the STFT loss at 0.1 for each term.
    public static let valentini = NFKMLXDenoiserObjective(stftLoss: true, stftConvergenceFactor: 0.1,
                                                          stftMagnitudeFactor: 0.1)

    /// The STFT resolutions `MultiResolutionSTFTLoss` defaults to: FFT size, hop, and Hann window length.
    static let resolutions = [(fft: 1024, hop: 120, window: 600), (fft: 2048, hop: 240, window: 1200),
                              (fft: 512, hop: 50, window: 240)]

    /// Scores the network's estimate for noisy speech `[N, L]` against the clean speech `[N, L]`.
    public func callAsFunction(_ net: NFKMLXDemucsNet, _ noisy: MLXArray, _ clean: MLXArray) -> MLXArray {
        let estimate = net(noisy.expandedDimensions(axis: -1))
        return loss(estimate: estimate.reshaped(clean.shape), clean: clean)
    }

    /// Scores an estimate `[N, L]` against the clean speech `[N, L]` directly, without a forward pass.
    public func loss(estimate: MLXArray, clean: MLXArray) -> MLXArray {
        var total: MLXArray
        switch distance {
        case .l1:
            total = abs(estimate - clean).mean()
        case .l2:
            total = square(estimate - clean).mean()
        case .huber:
            total = smoothL1Loss(predictions: estimate, targets: clean, beta: 1, reduction: .mean)
        }
        if stftLoss {
            let terms = Self.resolutions.map { resolution in
                Self.stftTerms(estimate: estimate, clean: clean, fft: resolution.fft, hop: resolution.hop,
                               window: resolution.window)
            }
            let count = Float(terms.count)
            total = total + stftConvergenceFactor * terms.map(\.convergence).reduce(MLXArray(Float(0)), +) / count
                + stftMagnitudeFactor * terms.map(\.magnitude).reduce(MLXArray(Float(0)), +) / count
        }
        return total
    }

    /// One resolution's spectral convergence (Frobenius norm of the magnitude error over that of the
    /// clean magnitude, across the whole batch) and log-magnitude L1, as `STFTLoss` computes them.
    static func stftTerms(estimate: MLXArray, clean: MLXArray, fft: Int, hop: Int, window: Int)
        -> (convergence: MLXArray, magnitude: MLXArray) {
        let x = magnitude(estimate, fft: fft, hop: hop, window: window)
        let y = magnitude(clean, fft: fft, hop: hop, window: window)
        let convergence = sqrt(square(y - x).sum()) / sqrt(square(y).sum())
        return (convergence, abs(log(y) - log(x)).mean())
    }

    /// `torch.stft` magnitude `[N, frames, bins]` of `[N, L]`: centered with reflect padding, a periodic
    /// Hann window of `window` centered into `fft`, and the reference's `clamp(min: 1e-7)` under the root.
    /// The transform is a pair of DFT matrices, so a gradient reaches the waveform.
    static func magnitude(_ x: MLXArray, fft: Int, hop: Int, window: Int) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let pad = fft / 2
        var indices = [Int32]()
        for i in Swift.stride(from: pad, through: 1, by: -1) { indices.append(Int32(i)) }
        indices.append(contentsOf: (0 ..< length).map { Int32($0) })
        for i in Swift.stride(from: length - 2, through: length - 1 - pad, by: -1) { indices.append(Int32(i)) }
        let padded = take(x, MLXArray(indices), axis: 1)
        let frames = 1 + (length + 2 * pad - fft) / hop
        var gather = [Int32]()
        gather.reserveCapacity(frames * fft)
        for f in 0 ..< frames {
            for n in 0 ..< fft { gather.append(Int32(f * hop + n)) }
        }
        let framed = take(padded, MLXArray(gather), axis: 1).reshaped([batch, frames, fft])
        let basis = dftBasis(fft: fft, window: window)
        let real = matmul(framed, basis.cosine)
        let imaginary = matmul(framed, basis.sine)
        return sqrt(maximum(square(real) + square(imaginary), MLXArray(Float(1e-7))))
    }

    /// The windowed real DFT `[fft, bins]` as cosine and sine matrices.
    static func dftBasis(fft: Int, window: Int) -> (cosine: MLXArray, sine: MLXArray) {
        let bins = fft / 2 + 1
        let hann = nfkPeriodicHann(window)
        let offset = (fft - window) / 2
        var cosine = [Float](repeating: 0, count: fft * bins)
        var sine = [Float](repeating: 0, count: fft * bins)
        for n in offset ..< (offset + window) {
            let w = Double(hann[n - offset])
            for k in 0 ..< bins {
                let angle = 2 * Double.pi * Double(k * n % fft) / Double(fft)
                cosine[n * bins + k] = Float(w * cos(angle))
                sine[n * bins + k] = Float(-w * sin(angle))
            }
        }
        return (MLXArray(cosine, [fft, bins]), MLXArray(sine, [fft, bins]))
    }
}

/// The random draws the augmentations take, in the order the reference's `augment.py` takes them.
protocol NFKDenoiserDraws {
    /// `random.random()`.
    mutating func unit() -> Double
    /// `random.uniform(low, high)`.
    mutating func uniform(_ low: Double, _ high: Double) -> Double
    /// `random.randrange(low, high)`.
    mutating func integer(_ low: Int, _ high: Int) -> Int
    /// `torch.argsort(torch.rand(count))`, a random order of the batch.
    mutating func permutation(_ count: Int) -> [Int]
    /// `torch.randint(bound, [count])`.
    mutating func offsets(below bound: Int, count: Int) -> [Int]
}

/// SplitMix64 draws from a seed, so an augmented run repeats.
struct NFKDenoiserSeededDraws: NFKDenoiserDraws {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    private mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func unit() -> Double {
        Double(next() >> 11) / Double(UInt64(1) << 53)
    }

    mutating func uniform(_ low: Double, _ high: Double) -> Double {
        low + (high - low) * unit()
    }

    mutating func integer(_ low: Int, _ high: Int) -> Int {
        low + Int(next() % UInt64(high - low))
    }

    mutating func permutation(_ count: Int) -> [Int] {
        var order = Array(0 ..< count)
        for i in Swift.stride(from: count - 1, to: 0, by: -1) {
            order.swapAt(i, Int(next() % UInt64(i + 1)))
        }
        return order
    }

    mutating func offsets(below bound: Int, count: Int) -> [Int] {
        (0 ..< count).map { _ in Int(next() % UInt64(bound)) }
    }
}

/// The augmentations `solver.py` applies to a training batch: the noise (noisy minus clean) and the
/// clean speech are augmented as a pair and summed back into the noisy input.
///
/// @discussion They apply in the reference's order: `remix`, `bandMask`, `shift`, `revEcho`. Each is off
/// at its zero value, the reference's default.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXDenoiserAugmentation: Sendable {
    /// Mixes each example's speech with another example's noise from the same batch (`Remix`).
    public var remix: Bool
    /// The largest fraction of 120 mel bands one band-stop removes from both signals (`BandMask`).
    public var bandMask: Float
    /// Crops a random window `shift` samples shorter than the input (`Shift`).
    public var shift: Int
    /// Crops the noise and the speech at the same offset rather than independently.
    public var shiftSame: Bool
    /// The probability of the synthetic reverb (`RevEcho`), which adds attenuated echo trains to the
    /// noise and the speech and moves 90% of the speech's reverb into the noise.
    public var revEcho: Float
    /// The rate the clips are at, which places the mel bands and the echo delays.
    public var sampleRate: Int

    public init(remix: Bool = false, bandMask: Float = 0, shift: Int = 0, shiftSame: Bool = false,
                revEcho: Float = 0, sampleRate: Int = 16_000) {
        self.remix = remix
        self.bandMask = bandMask
        self.shift = shift
        self.shiftSame = shiftSame
        self.revEcho = revEcho
        self.sampleRate = sampleRate
    }

    /// `launch_dns.sh`: the reverb on every batch and a one-second shared shift.
    public static let dns = NFKMLXDenoiserAugmentation(shift: 16_000, shiftSame: true, revEcho: 1)

    /// `launch_valentini.sh`: remix, band masks up to 20%, and a half-second shared shift.
    public static let valentini = NFKMLXDenoiserAugmentation(remix: true, bandMask: 0.2, shift: 8_000, shiftSame: true)

    /// Augments a batch of noisy and clean clips `[N, L]`, returning the new noisy and clean clips.
    func apply<Draws: NFKDenoiserDraws>(noisy: MLXArray, clean: MLXArray, draws: inout Draws) -> (noisy: MLXArray, clean: MLXArray) {
        var noise = noisy - clean
        var speech = clean
        if remix {
            noise = take(noise, MLXArray(draws.permutation(noise.dim(0)).map(Int32.init)), axis: 0)
        }
        if Int(abs(bandMask) * 120) > 0 {
            (noise, speech) = bandMasked(noise, speech, draws: &draws)
        }
        if shift > 0 {
            (noise, speech) = shifted(noise, speech, draws: &draws)
        }
        // `RevEcho` exists only when its probability is set, and then takes a draw on every batch.
        if revEcho > 0, draws.unit() < Double(revEcho) {
            (noise, speech) = reverberated(noise, speech, draws: &draws)
        }
        return (noise + speech, speech)
    }

    private func bandMasked<Draws: NFKDenoiserDraws>(_ noise: MLXArray, _ speech: MLXArray, draws: inout Draws)
        -> (MLXArray, MLXArray) {
        let bands = 120
        let bandwidth = Int(abs(bandMask) * Float(bands))
        let mels = Self.melFrequencies(count: bands, low: 40, high: Double(sampleRate) / 2).map { $0 / Double(sampleRate) }
        let low = draws.integer(0, bands)
        let high = draws.integer(low, min(bands, low + bandwidth))
        let width = Int(2 / min(mels[low], mels[high]))
        func masked(_ x: MLXArray) -> MLXArray {
            x - Self.lowPass(x, cutoff: mels[high], width: width) + Self.lowPass(x, cutoff: mels[low], width: width)
        }
        return (masked(noise), masked(speech))
    }

    /// `dsp.mel_frequencies`: `count` frequencies evenly spaced in mel between `low` and `high` Hz.
    static func melFrequencies(count: Int, low: Double, high: Double) -> [Double] {
        func mel(_ f: Double) -> Double { 2595 * log10(1 + f / 700) }
        let (a, b) = (mel(low), mel(high))
        return (0 ..< count).map { index in
            let m = count == 1 ? a : a + (b - a) * Double(index) / Double(count - 1)
            return 700 * (pow(10, m / 2595) - 1)
        }
    }

    /// `dsp.LowPassFilters` for one cutoff (a fraction of the sample rate): a Hamming-windowed sinc of
    /// `2 · width + 1` taps, zero-padded by `width` so the clip keeps its length.
    static func lowPass(_ x: MLXArray, cutoff: Double, width: Int) -> MLXArray {
        let taps = 2 * width + 1
        let filter = (0 ..< taps).map { index -> Float in
            let t = Double(index - width)
            let argument = 2 * cutoff * t
            let sinc = argument == 0 ? 1 : sin(Double.pi * argument) / (Double.pi * argument)
            let window = 0.54 - 0.46 * cos(2 * Double.pi * Double(index) / Double(taps - 1))
            return Float(2 * cutoff * sinc * window)
        }
        let weight = MLXArray(filter, [1, taps, 1])
        return conv1d(x.expandedDimensions(axis: -1), weight, padding: width).squeezed(axis: -1)
    }

    private func shifted<Draws: NFKDenoiserDraws>(_ noise: MLXArray, _ speech: MLXArray, draws: inout Draws)
        -> (MLXArray, MLXArray) {
        let (batch, length) = (noise.dim(0), noise.dim(1) - shift)
        let offsets = draws.offsets(below: shift, count: shiftSame ? batch : 2 * batch)
        func cropped(_ x: MLXArray, _ starts: ArraySlice<Int>) -> MLXArray {
            stacked(zip(0 ..< batch, starts).map { row, start in x[row, start ..< (start + length)] })
        }
        let speechStarts = shiftSame ? offsets[0 ..< batch] : offsets[batch ..< (2 * batch)]
        return (cropped(noise, offsets[0 ..< batch]), cropped(speech, speechStarts))
    }

    private func reverberated<Draws: NFKDenoiserDraws>(_ noise: MLXArray, _ speech: MLXArray, draws: inout Draws)
        -> (MLXArray, MLXArray) {
        let initial = draws.unit() * 0.3
        let firstDelay = draws.uniform(0.01, 0.03)
        let rt60 = draws.uniform(0.3, 1.3)
        let noiseReverb = reverb(noise, initial: initial, firstDelay: firstDelay, rt60: rt60, draws: &draws)
        let speechReverb = reverb(speech, initial: initial, firstDelay: firstDelay, rt60: rt60, draws: &draws)
        let keepClean: Float = 0.1
        return (noise + noiseReverb + (1 - keepClean) * speechReverb, speech + keepClean * speechReverb)
    }

    /// `RevEcho._reverb`: three trains of echoes, each delayed by a jittered first delay and attenuated
    /// toward 1e-3 of the first echo over `rt60`.
    private func reverb<Draws: NFKDenoiserDraws>(_ source: MLXArray, initial: Double, firstDelay: Double, rt60: Double,
                                                 draws: inout Draws) -> MLXArray {
        let length = source.dim(1)
        var total = MLXArray.zeros(like: source)
        for _ in 0 ..< 3 {
            var fraction = 1.0
            var echo = Float(initial) * source
            while fraction > 1e-3 {
                let delayJitter = 1 + 0.1 * draws.uniform(-1, 1)
                let delay = min(1 + Int(delayJitter * firstDelay * Double(sampleRate)), length)
                echo = MLX.padded(echo[0..., 0 ..< (length - delay)],
                                  widths: [IntOrPair((0, 0)), IntOrPair((delay, 0))])
                total = total + echo
                let attenuationJitter = 1 + 0.1 * draws.uniform(-1, 1)
                let attenuation = pow(10, -3 * attenuationJitter * firstDelay / rt60)
                echo = echo * Float(attenuation)
                fraction *= attenuation
            }
        }
        return total
    }
}

public extension NFKMLXDenoiser {

    /// `launch_dns.sh`'s batch, 128 clips, which the recipe reaches through `accumulationSteps` or a
    /// batch of that many. Introduced in InferKit 0.4.0.
    static let referenceBatchSize = 128

    /// `launch_dns.sh`'s clip length in seconds at 16 kHz. Introduced in InferKit 0.4.0.
    static let referenceSegmentSeconds = 10

    /// Builds a denoiser for training or for reloading a trained checkpoint. With a `weightsURL` the
    /// checkpoint loads, released (`dns48`, `dns64`) or saved by `NFKMLXWeights`, and its base width is
    /// read from its first encoder weight; without one the net is randomly initialized at `baseChannels`.
    /// Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?, baseChannels: Int = 48) throws -> NFKMLXDemucsNet {
        guard let weightsURL else {
            return makeNet(baseChannels: baseChannels)
        }
        let arrays = try NFKMLXWeights.loadCheckpoint(url: weightsURL).arrays
        guard let first = arrays["encoder.0.0.weight"] ?? arrays["encoder.0.conv1.weight"] else {
            throw NFKMLXError.unsupportedConfiguration("\(weightsURL.lastPathComponent) has no first encoder weight")
        }
        let net = makeNet(baseChannels: first.dim(0))
        try NFKMLXDemucs.loadWeights(into: net, from: weightsURL)
        return net
    }

    /// The optimizer `train.py` builds, `torch.optim.Adam` at 3e-4 with betas 0.9 and 0.999 and no decay.
    ///
    /// @discussion The bottleneck's LSTMs fold the reference's two biases (`bias_ih`, `bias_hh`) into one.
    /// Adam moves each of the two by the same normalized step, so the folded bias steps at twice the rate.
    /// Introduced in InferKit 0.4.0.
    static func referenceOptimizer(for net: NFKMLXDemucsNet) -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: 3e-4, over: net) { key in
            let recurrentBias = (key.hasPrefix("lstm.lstm.") || key.hasPrefix("lstm.reverse.")) && key.hasSuffix(".bias")
            return (rateScale: recurrentBias ? 2 : 1, weightDecay: 0)
        }
    }

    /// Trains a denoiser on pairs of noisy and clean speech.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:baseChannels:)``.
    ///   - examples: supplies one batch per step: noisy clips and the clean speech under them, each
    ///     `[N, L]` mono at 16 kHz.
    ///   - objective: the loss; ``NFKMLXDenoiserObjective/dns`` by default.
    ///   - augmentation: augments each batch before the forward pass. Nil, the default, applies none;
    ///     ``NFKMLXDenoiserAugmentation/dns`` is the released models' recipe.
    ///   - augmentationSeed: seeds the augmentations' draws, the reference's `seed` by default.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer(for:)``.
    ///   - steps: how many updates to train for.
    ///   - clipGradientNorm: bounds the global gradient norm; the reference clips none.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is the reference's constant rate.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Every parameter trains. Save with `NFKMLXWeights.save`; ``network(weightsURL:baseChannels:)`` and
    /// `backend(weightsURL:)` read the file back. A run is minutes; call it off the render thread.
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXDemucsNet,
        examples: (Int) -> (noisy: MLXArray, clean: MLXArray),
        objective: NFKMLXDenoiserObjective = .dns,
        augmentation: NFKMLXDenoiserAugmentation? = nil,
        augmentationSeed: UInt64 = 2036,
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        var draws = NFKDenoiserSeededDraws(seed: augmentationSeed)
        return try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { referenceOptimizer(for: net) },
            referenceSchedule: { .constant },
            steps: steps,
            batch: { step in
                let example = examples(step)
                guard let augmentation else {
                    return (example.noisy, example.clean)
                }
                let augmented = augmentation.apply(noisy: example.noisy, clean: example.clean, draws: &draws)
                return (augmented.noisy, augmented.clean)
            },
            loss: objective.callAsFunction,
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
