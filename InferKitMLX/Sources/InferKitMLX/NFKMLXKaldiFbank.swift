// Kaldi-compatible mel filter-bank features + deltas, the MossFormer2 SE 48K network input front end.
// Reproduces `torchaudio.compliance.kaldi.fbank(window_type='hamming', dither=0, num_mel_bins=60,
// frame_length=40ms, frame_shift=8ms, sample_frequency=48000)` followed by `compute_deltas` once and
// twice, concatenated to 180 dims. No MLX primitive existed for this; it is the SE parity crux.
//
// Kaldi's per-frame pipeline (snip_edges framing): remove DC offset (subtract the frame mean),
// pre-emphasis 0.97 (with x[-1] replicated as x[0]), a Povey/hamming window, an FFT padded up to the
// next power of two, the power spectrum, a kaldi-mel triangular bank (mel = 1127·ln(1 + f/700)), and a
// natural log floored at the float epsilon. `dither` is FORCED to 0 (it is random noise; parity is
// impossible with it on).
//
// SCAFFOLD STATUS: faithful to kaldi's documented steps and marked `PARITY:` at the load-bearing
// choices. Validated against the recorded `feature` from `run_reference.py mossformer2_se` on the M1.

import Foundation
import MLX
import MLXFFT

/// Kaldi-compatible fbank + deltas for MossFormer2 SE.
enum NFKMLXKaldiFbank {
    /// `samples` mono at `sampleRate` → `[1, frames, 180]` (fbank60 ‖ Δ ‖ ΔΔ).
    static func features(samples: [Float], config: NFKMLXMossFormer2Configuration) -> MLXArray {
        // PARITY: kaldi derives the window/shift in samples from the ms values; for 48 kHz these are
        // 1920 / 384. Validated at reference parity against the recorded feature.
        let n = Int((Double(config.winLen) / Double(config.sampleRate) * 1000.0) * Double(config.sampleRate) / 1000.0 + 0.5)
        let hop = Int((Double(config.winInc) / Double(config.sampleRate) * 1000.0) * Double(config.sampleRate) / 1000.0 + 0.5)
        guard samples.count >= n else { return MLXArray.zeros([1, 0, config.numMels * 3]) }
        let frames = 1 + (samples.count - n) / hop                     // snip_edges
        let padded = nextPow2(n)                                       // kaldi round_to_power_of_two
        let window = poveyHamming(n)

        // Build the framed, DC-removed, pre-emphasized, windowed, zero-padded matrix [frames, padded].
        var buffer = [Float](repeating: 0, count: frames * padded)
        for f in 0 ..< frames {
            let start = f * hop
            var frame = [Float](repeating: 0, count: n)
            var mean: Float = 0
            for i in 0 ..< n { mean += samples[start + i] }
            mean /= Float(n)
            for i in 0 ..< n { frame[i] = samples[start + i] - mean }   // remove DC
            // Pre-emphasis 0.97 (x[-1] = x[0]); walk high→low so each read uses the un-emphasized value.
            let coeff: Float = 0.97
            let first = frame[0]                                       // unchanged by the loop below
            for i in stride(from: n - 1, through: 1, by: -1) { frame[i] -= coeff * frame[i - 1] }
            frame[0] -= coeff * first                                  // kaldi replicates x[-1] = x[0]
            for i in 0 ..< n { buffer[f * padded + i] = frame[i] * window[i] }
        }

        let framed = MLXArray(buffer, [frames, padded])
        let spectrum = MLXFFT.rfft(framed, axis: 1)                    // [frames, padded/2+1] complex
        let power = spectrum.abs() * spectrum.abs()                    // PARITY: use_power=true
        let bank = melBank(fftBins: padded / 2 + 1, config: config)   // [numMels, bins]
        let mel = matmul(power, bank.transposed(1, 0))                // [frames, numMels]
        let logMel = log(maximum(mel, Float.leastNormalMagnitude))    // PARITY: floor at float epsilon

        let delta = deltas(logMel)
        let deltaDelta = deltas(delta)
        return concatenated([logMel, delta, deltaDelta], axis: 1).expandedDimensions(axis: 0)
    }

    /// The Povey/hamming window kaldi applies for `window_type='hamming'`:
    /// `0.54 - 0.46·cos(2π i / (N-1))`.
    private static func poveyHamming(_ n: Int) -> [Float] {
        let a = 2 * Float.pi / Float(n - 1)
        return (0 ..< n).map { 0.54 - 0.46 * cosf(a * Float($0)) }
    }

    private static func nextPow2(_ n: Int) -> Int {
        var p = 1
        while p < n { p <<= 1 }
        return p
    }

    /// The kaldi-mel triangular filterbank: `mel = 1127·ln(1 + f/700)`, `numMels` triangles between
    /// `low_freq` 20 Hz and the Nyquist, over the FFT power bins.
    private static func melBank(fftBins: Int, config: NFKMLXMossFormer2Configuration) -> MLXArray {
        let sr = Float(config.sampleRate)
        let nyquist = sr / 2
        let lowFreq: Float = 20
        let highFreq = nyquist
        func hzToMel(_ hz: Float) -> Float { 1127 * logf(1 + hz / 700) }
        func melToHz(_ mel: Float) -> Float { 700 * (expf(mel / 1127) - 1) }
        let lowMel = hzToMel(lowFreq), highMel = hzToMel(highFreq)
        let numMels = config.numMels
        let fftBinWidth = sr / Float((fftBins - 1) * 2)                // Hz per FFT bin
        var bank = [Float](repeating: 0, count: numMels * fftBins)
        let melStep = (highMel - lowMel) / Float(numMels + 1)
        for m in 0 ..< numMels {
            let leftMel = lowMel + Float(m) * melStep
            let centerMel = lowMel + Float(m + 1) * melStep
            let rightMel = lowMel + Float(m + 2) * melStep
            for bin in 0 ..< fftBins {
                let mel = hzToMel(Float(bin) * fftBinWidth)
                let weight: Float
                if mel > leftMel, mel <= centerMel {
                    weight = (mel - leftMel) / (centerMel - leftMel)
                } else if mel > centerMel, mel < rightMel {
                    weight = (rightMel - mel) / (rightMel - centerMel)
                } else {
                    weight = 0
                }
                bank[m * fftBins + bin] = weight
            }
        }
        return MLXArray(bank, [numMels, fftBins])
    }

    /// `torchaudio.functional.compute_deltas` (default `win_length=5`): a symmetric regression filter
    /// `Σ_{n=1}^{2} n·(x[t+n] − x[t−n]) / (2·Σ n²)` with edge replication.
    private static func deltas(_ x: MLXArray) -> MLXArray {
        let n = 2
        let denom: Float = 2 * (1 + 4)                                 // 2·Σ n² for n=1,2
        let padded = MLX.padded(x, widths: [IntOrPair((n, n)), IntOrPair(0)], mode: .edge)
        let frames = x.dim(0)
        var acc = MLXArray.zeros(x.shape)
        for k in 1 ... n {
            let ahead = padded[.stride(from: n + k, to: n + k + frames), 0...]
            let behind = padded[.stride(from: n - k, to: n - k + frames), 0...]
            acc = acc + Float(k) * (ahead - behind)
        }
        return acc / denom
    }
}
