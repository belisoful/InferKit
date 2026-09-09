// Shared audio front-end primitives for the speech-restoration ports (dereverberation, denoising,
// bandwidth extension). Two things most of that family reads: a complex STFT that carries PHASE (the
// magnitude-and-phase pair, and its inverse by window-normalized overlap-add), and an ERB filterbank.
// Building these once de-risks the family — SGMSE+, MP-SENet, CMGAN, FRCRN, and MossFormer2 all read
// the complex STFT; GTCRN and DeepFilterNet read the ERB bank.
//
// The STFT reproduces `torch.stft` / `torch.istft` at `center=true`, `pad_mode="reflect"`,
// `normalized=false` — the settings the restoration references use — and is generalized from the
// proven Kokoro iSTFT (window-squared overlap-add normalization, center padding removed). The FFT is
// evaluated in MLX; the framing and overlap-add run over Swift buffers, because MLX has no scatter-add.

import Foundation
import MLX
import MLXFFT

/// A periodic Hann window of `length` (`torch.hann_window(length)`, `periodic=true`), matching
/// `get_window("hann", length, fftbins=true)`.
func nfkPeriodicHann(_ length: Int) -> [Float] {
    (0 ..< length).map { 0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(length)) }
}

/// A complex STFT that returns magnitude and phase, and inverts them back to samples. It reproduces
/// `torch.stft(center: true, pad_mode: "reflect", normalized: false, return_complex: true)` and the
/// matching `torch.istft`. The window defaults to a periodic Hann of `winLength`, centered into `nFFT`
/// when `winLength < nFFT`. Held as a value type outside any module graph, so its window stays out of
/// `parameters()`.
struct NFKMLXComplexSTFT {
    let nFFT: Int
    let hop: Int
    let window: MLXArray                        // [nFFT]
    /// `pad_mode="constant"` (zeros at both ends) instead of the reflect padding `torch.stft` defaults
    /// to; speechbrain's STFT is built that way, which MetricGAN+ inherits.
    let zeroPadded: Bool
    /// `center=false`: no padding at all, so the first frame starts at sample 0 and the inverse keeps
    /// every synthesized sample. The convolutional STFT of FRCRN / MossFormer2 is built that way.
    let centered: Bool

    /// `winLength` defaults to `nFFT`. A shorter window is zero-centered into `nFFT`, as `torch.stft`
    /// does. Pass `window` to override (already `nFFT`-wide).
    init(nFFT: Int, hop: Int, winLength: Int? = nil, window: MLXArray? = nil, zeroPadded: Bool = false,
         centered: Bool = true) {
        self.nFFT = nFFT
        self.hop = hop
        self.zeroPadded = zeroPadded
        self.centered = centered
        if let window {
            self.window = window
        } else {
            let length = winLength ?? nFFT
            let raw = MLXArray(nfkPeriodicHann(length))
            self.window = Self.centered(raw, in: nFFT)
        }
    }

    private static func centered(_ raw: MLXArray, in size: Int) -> MLXArray {
        let total = size - raw.shape[0]
        guard total > 0 else { return raw }
        return MLX.padded(raw, widths: [IntOrPair((total / 2, total - total / 2))], mode: .constant)
    }

    /// Reflect-pads `[1, L]` by `pad` on each side (`pad_mode="reflect"`: the edge sample is NOT
    /// repeated, so `[a, b, c]` padded by 2 is `[c, b, a, b, c, b, a]`).
    private func reflectPad(_ signal: MLXArray, pad: Int) -> MLXArray {
        if !centered { return signal }
        if zeroPadded {
            return MLX.padded(signal, widths: [IntOrPair((0, 0)), IntOrPair((pad, pad))], mode: .constant)
        }
        let l = signal.dim(1)
        var indices = [Int32]()
        for i in stride(from: pad, through: 1, by: -1) { indices.append(Int32(i)) }
        indices.append(contentsOf: (0 ..< l).map { Int32($0) })
        for i in stride(from: l - 2, through: l - 1 - pad, by: -1) { indices.append(Int32(i)) }
        return take(signal, MLXArray(indices), axis: 1)
    }

    /// `[1, L]` → magnitude and phase, each `[1, nFFT/2 + 1, frames]`. Center-padded by `nFFT/2`.
    func transform(_ signal: MLXArray) -> (magnitude: MLXArray, phase: MLXArray) {
        let pad = nFFT / 2
        let padded = reflectPad(signal, pad: pad)
        let length = padded.dim(1)
        let frames = 1 + (length - nFFT) / hop
        var gather = [Int32]()
        for f in 0 ..< frames {
            for k in 0 ..< nFFT { gather.append(Int32(f * hop + k)) }
        }
        let framed = take(padded, MLXArray(gather), axis: 1).reshaped([1, frames, nFFT]) * window.reshaped([1, 1, nFFT])
        let spectrum = MLXFFT.rfft(framed, axis: 2)                       // [1, frames, bins] complex
        let magnitude = spectrum.abs().transposed(0, 2, 1)               // [1, bins, frames]
        let phase = atan2(spectrum.imaginaryPart(), spectrum.realPart()).transposed(0, 2, 1)
        return (magnitude, phase)
    }

    /// `[1, L]` → real and imaginary parts, each `[1, nFFT/2 + 1, frames]` — the same center-padded
    /// framing as `transform`, without the magnitude/phase conversion. This is what the score-based
    /// restoration front ends read (`torch.stft(return_complex=true)` gives them a complex spectrogram).
    func transformComplex(_ signal: MLXArray) -> (real: MLXArray, imaginary: MLXArray) {
        let pad = nFFT / 2
        let padded = reflectPad(signal, pad: pad)
        let length = padded.dim(1)
        let frames = 1 + (length - nFFT) / hop
        var gather = [Int32]()
        for f in 0 ..< frames {
            for k in 0 ..< nFFT { gather.append(Int32(f * hop + k)) }
        }
        let framed = take(padded, MLXArray(gather), axis: 1).reshaped([1, frames, nFFT]) * window.reshaped([1, 1, nFFT])
        let spectrum = MLXFFT.rfft(framed, axis: 2)                       // [1, frames, bins] complex
        return (spectrum.realPart().transposed(0, 2, 1), spectrum.imaginaryPart().transposed(0, 2, 1))
    }

    /// real and imaginary parts `[1, bins, frames]` → `[1, samples]`, the complex counterpart of
    /// `inverse`, with the same window-squared overlap-add normalization.
    func inverseComplex(real: MLXArray, imaginary: MLXArray, length: Int? = nil) -> MLXArray {
        let frames = real.dim(2)
        let re = real.transposed(0, 2, 1)                                 // [1, frames, bins]
        let im = imaginary.transposed(0, 2, 1)
        let complex = re.asType(.complex64) + im.asType(.complex64) * MLXArray(real: 0, imaginary: 1)
        let time = MLXFFT.irfft(complex, n: nFFT, axis: 2)               // [1, frames, nFFT] real
        return overlapAdd(time, frames: frames, length: length)
    }

    /// The shared window-squared overlap-add of framed time samples `[1, frames, nFFT]` → `[1, samples]`,
    /// with the center padding removed. `torch.istft`'s normalization. A `length` is `torch.istft`'s
    /// `length`: the output starts at the center pad and runs that many samples, so the last frame's
    /// tail is kept (or the result zero-padded) rather than trimmed at the symmetric pad.
    private func overlapAdd(_ time: MLXArray, frames: Int, length: Int? = nil) -> MLXArray {
        let windowed = time * window.reshaped([1, 1, nFFT])
        let outLength = (frames - 1) * hop + nFFT
        let framesValues = windowed[0].asArray(Float.self)
        let windowValues = (window * window).asArray(Float.self)
        var output = [Float](repeating: 0, count: outLength)
        var normalization = [Float](repeating: 0, count: outLength)
        for f in 0 ..< frames {
            let start = f * hop
            for k in 0 ..< nFFT {
                output[start + k] += framesValues[f * nFFT + k]
                normalization[start + k] += windowValues[k]
            }
        }
        let pad = centered ? nFFT / 2 : 0
        var result = [Float](repeating: 0, count: length ?? (outLength - 2 * pad))
        for i in 0 ..< min(result.count, outLength - pad) {
            let norm = normalization[i + pad]
            result[i] = norm > 1e-11 ? output[i + pad] / norm : 0
        }
        return MLXArray(result).reshaped([1, result.count])
    }

    /// magnitude and phase `[1, bins, frames]` → `[1, samples]`, overlap-add with the window-squared
    /// normalization `torch.istft` applies, center padding removed.
    func inverse(magnitude: MLXArray, phase: MLXArray, length: Int? = nil) -> MLXArray {
        let frames = magnitude.dim(2)
        let real = (magnitude * cos(phase)).transposed(0, 2, 1)          // [1, frames, bins]
        let imaginary = (magnitude * sin(phase)).transposed(0, 2, 1)
        let complex = real.asType(.complex64) + imaginary.asType(.complex64) * MLXArray(real: 0, imaginary: 1)
        let time = MLXFFT.irfft(complex, n: nFFT, axis: 2)               // [1, frames, nFFT] real
        return overlapAdd(time, frames: frames, length: length)
    }
}

/// An ERB (equivalent rectangular bandwidth) triangular filterbank, the frequency banding GTCRN and
/// DeepFilterNet compress the spectrum into before their recurrent cores. This builds the Glasberg-Moore
/// ERB-scale triangular bank, the analog of `NFKMLXMel.melFilters` on the ERB scale.
///
/// NOTE: the shipped restoration models do not all agree on the exact band edges — DeepFilterNet uses
/// integer per-band widths derived from a target band count, and GTCRN uses its own table. This
/// primitive gives the ERB-scale machinery; the exact per-model banding is pinned against that model's
/// reference when it is ported (the way each mel front end loads its own filterbank).
enum NFKMLXERB {
    /// Hz → ERB-rate scale (Glasberg & Moore 1990).
    static func hzToERB(_ hz: Float) -> Float { 21.4 * log10f(1 + 0.00437 * hz) }
    /// ERB-rate scale → Hz.
    static func erbToHz(_ erb: Float) -> Float { (powf(10, erb / 21.4) - 1) / 0.00437 }

    /// `[bands, bins]` triangular filters spaced evenly on the ERB scale between `0` and Nyquist.
    /// `bins` is `nFFT/2 + 1`. Row-normalized is left to the caller, as banks differ on that.
    static func filters(sampleRate: Int, bins: Int, bands: Int) -> MLXArray {
        let nyquist = Float(sampleRate) / 2
        let binHz = (0 ..< bins).map { Float($0) * nyquist / Float(bins - 1) }
        let lowERB = hzToERB(0), highERB = hzToERB(nyquist)
        // `bands + 2` edge points on the ERB scale → `bands` overlapping triangles.
        let edges = (0 ..< bands + 2).map { erbToHz(lowERB + (highERB - lowERB) * Float($0) / Float(bands + 1)) }
        var bank = [Float](repeating: 0, count: bands * bins)
        for band in 0 ..< bands {
            let lower = edges[band], center = edges[band + 1], upper = edges[band + 2]
            for bin in 0 ..< bins {
                let hz = binHz[bin]
                let weight: Float
                if hz >= lower, hz <= center {
                    weight = center > lower ? (hz - lower) / (center - lower) : 0
                } else if hz > center, hz <= upper {
                    weight = upper > center ? (upper - hz) / (upper - center) : 0
                } else {
                    weight = 0
                }
                bank[band * bins + bin] = weight
            }
        }
        return MLXArray(bank).reshaped([bands, bins])
    }
}
