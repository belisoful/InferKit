//
//  NFKMLXWanDancerMusic.swift
//  InferKitMLX
//
//  Wan-Dancer's 35 music features per video frame, as `get_music_base_feature` in Wan-Video/Wan-Dancer's
//  `gen_video/gen_video_global.py` computes them with librosa 0.10.2: the onset envelope, 20 MFCCs, 12
//  CENS chroma, the onset peaks, and the beats, at 30 frames a second (512-sample hops of 15360 Hz audio).
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX

/// Wan-Dancer's music features.
///
/// @discussion The reference loads the music at 15360 Hz (30 frames a second at a 512-sample hop) and then
/// computes every feature as though the rate were 22050 Hz, which sets the mel and constant-Q frequencies
/// and the onset-picking windows. Its starting tempo comes from a second load of the music at 22050 Hz.
/// librosa resamples with soxr; this resamples with SciPy's `resample_poly` (a Kaiser β 5 windowed sinc),
/// which is also how the constant-Q transform halves each octave here. Music already at 15360 Hz loads
/// without a resample, so its envelope, MFCCs, onsets, and beats follow librosa's arithmetic; the chroma
/// differs from librosa's by the octave decimation alone. Introduced in InferKit 0.4.0.
public enum NFKMLXWanDancerMusic {
    /// The rate the reference loads music at: 30 frames a second at a 512-sample hop.
    public static let sampleRate = 15360
    /// The feature width: envelope, 20 MFCCs, 12 chroma, the onset and the beat indicators.
    public static let featureCount = 35
    static let hop = 512
    static let fftSize = 2048
    /// The rate every feature is computed as, whatever the samples' rate.
    static let assumedRate = 22050.0
    static let tempoRate = 22050

    /// The features `[frames, 35]` of mono samples at `sampleRate` Hz, one row per 512-sample hop of the
    /// music at 15360 Hz.
    public static func features(samples: [Float], sampleRate rate: Int) -> MLXArray {
        let rows = featureRows(samples: samples, sampleRate: rate)
        return MLXArray(rows.flatMap { (row: [Float]) -> [Float] in row }, [rows.count, featureCount])
    }

    static func featureRows(samples: [Float], sampleRate rate: Int) -> [[Float]] {
        let data = resampled(samples.map(Double.init), from: rate, to: sampleRate)
        let power = powerSpectrogram(data, fftSize: fftSize, hop: hop)
        let mel = melSpectrogram(power, rate: assumedRate, fftSize: fftSize, bands: 128)
        let decibels = powerToDecibels(mel)
        let envelope = onsetStrength(decibels)
        let mfcc = cepstrum(decibels, count: 20)
        let chroma = chromaCENS(data, rate: assumedRate, hop: hop)
        let peaks = Set(onsetPeaks(envelope, rate: assumedRate, hop: hop))
        let tempoData = resampled(samples.map(Double.init), from: rate, to: tempoRate)
        let startTempo = tempo(onsetStrength(powerToDecibels(melSpectrogram(
            powerSpectrogram(tempoData, fftSize: fftSize, hop: hop), rate: Double(tempoRate), fftSize: fftSize,
            bands: 128))), rate: Double(tempoRate), hop: hop, startTempo: 120)
        let beats = Set(beatTrack(envelope, rate: assumedRate, hop: hop, startTempo: startTempo, tightness: 100))
        let frames: Int = envelope.count
        let framesAgree: Bool = mfcc.count == frames && chroma.count == frames
        precondition(framesAgree, "the features disagree on the frame count")
        return (0 ..< frames).map { (t: Int) -> [Float] in
            var row: [Float] = [Float(envelope[t])]
            row.append(contentsOf: mfcc[t].map { (value: Double) -> Float in Float(value) })
            row.append(contentsOf: chroma[t].map { (value: Double) -> Float in Float(value) })
            row.append(peaks.contains(t) ? Float(1) : Float(0))
            row.append(beats.contains(t) ? Float(1) : Float(0))
            return row
        }
    }

    // MARK: - Resampling

    /// librosa's `resample(res_type="polyphase")`: SciPy's `resample_poly` by the reduced ratio, cut or
    /// zero-filled to `ceil(count · target / source)` samples, stored as float32.
    static func resampled(_ samples: [Double], from source: Int, to target: Int, scale: Bool = false) -> [Double] {
        guard source != target else { return samples.map { Double(Float($0)) } }
        let divisor = greatestCommonDivisor(source, target)
        let up = target / divisor, down = source / divisor
        let scaledCount: Double = Double(samples.count) * Double(target) / Double(source)
        let count = Int(scaledCount.rounded(.up))
        var output = resamplePoly(samples, up: up, down: down)
        if output.count > count {
            output.removeLast(output.count - count)
        } else if output.count < count {
            output += [Double](repeating: 0, count: count - output.count)
        }
        let gain = scale ? (Double(target) / Double(source)).squareRoot() : 1
        return output.map { Double(Float($0 / gain)) }
    }

    /// SciPy's `resample_poly(x, up, down)` with its default `("kaiser", 5.0)` window and zero padding:
    /// a `firwin` low-pass of `20·max(up, down) + 1` taps at cutoff `1/max(up, down)`, scaled by `up`,
    /// applied through `upfirdn` and trimmed to `ceil(count · up / down)` samples.
    static func resamplePoly(_ x: [Double], up: Int, down: Int) -> [Double] {
        let maximumRate = Swift.max(up, down)
        let half = 10 * maximumRate
        let taps = firwinKaiser(count: 2 * half + 1, cutoff: 1 / Double(maximumRate), beta: 5).map { $0 * Double(up) }
        let outputCount = x.count * up / down + (x.count * up % down != 0 ? 1 : 0)
        let prePad = down - half % down
        let preRemove = (half + prePad) / down
        var filter = [Double](repeating: 0, count: prePad) + taps
        func upfirdnLength(_ filterLength: Int) -> Int { ((x.count - 1) * up + filterLength - 1) / down + 1 }
        while upfirdnLength(filter.count) < outputCount + preRemove {
            filter.append(0)
        }
        var output = [Double](repeating: 0, count: outputCount)
        for index in 0 ..< outputCount {
            let position = (index + preRemove) * down
            var sum = 0.0
            var tap = position % up
            while tap < filter.count {
                let source = (position - tap) / up
                if source < 0 { break }
                if source < x.count { sum += filter[tap] * x[source] }
                tap += up
            }
            output[index] = sum
        }
        return output
    }

    /// SciPy's `firwin(count, cutoff, window=("kaiser", beta))`: a windowed-sinc low-pass normalized to
    /// unit gain at DC.
    static func firwinKaiser(count: Int, cutoff: Double, beta: Double) -> [Double] {
        let alpha = 0.5 * Double(count - 1)
        let normalizer = besselI0(beta)
        let taps = (0 ..< count).map { (n: Int) -> Double in
            let m: Double = Double(n) - alpha
            let argument: Double = cutoff * m
            let sinc: Double = argument == 0 ? 1 : sin(Double.pi * argument) / (Double.pi * argument)
            let ratio: Double = m / alpha
            return cutoff * sinc * besselI0(beta * Swift.max(0, 1 - ratio * ratio).squareRoot()) / normalizer
        }
        let total = taps.reduce(0, +)
        return taps.map { (tap: Double) -> Double in tap / total }
    }

    static func besselI0(_ x: Double) -> Double {
        var sum = 1.0, term = 1.0
        for k in 1 ..< 60 {
            term *= (x / 2) / Double(k)
            sum += term * term
        }
        return sum
    }

    static func greatestCommonDivisor(_ a: Int, _ b: Int) -> Int { b == 0 ? a : greatestCommonDivisor(b, a % b) }

    // MARK: - Spectra

    /// A periodic Hann window (`scipy.signal.get_window("hann", n, fftbins=True)`).
    static func periodicHann(_ count: Int) -> [Double] {
        (0 ..< count).map { (n: Int) -> Double in 0.5 - 0.5 * cos(2 * Double.pi * Double(n) / Double(count)) }
    }

    /// The frames `[1 + count / hop][fftSize]` of a centered STFT: the signal zero-padded by half a
    /// window on each side, as librosa 0.10 pads with `pad_mode="constant"`.
    static func centeredFrames(_ samples: [Double], fftSize: Int, hop: Int) -> [[Double]] {
        let pad = fftSize / 2
        let frames = 1 + samples.count / hop
        return (0 ..< frames).map { (t: Int) -> [Double] in
            (0 ..< fftSize).map { (i: Int) -> Double in
                let index = t * hop + i - pad
                return index >= 0 && index < samples.count ? samples[index] : 0
            }
        }
    }

    /// `|stft(y)|²` `[frames][fftSize / 2 + 1]` with a periodic Hann window, stored as float32.
    static func powerSpectrogram(_ samples: [Double], fftSize: Int, hop: Int, power: Double = 2) -> [[Double]] {
        let window = periodicHann(fftSize)
        return centeredFrames(samples, fftSize: fftSize, hop: hop).map { (frame: [Double]) -> [Double] in
            let spectrum = NFKMLXRealFFT.forward(zip(frame, window).map { (sample: Double, weight: Double) -> Double in
                sample * weight
            })
            return spectrum.map { (bin: (re: Double, im: Double)) -> Double in
                let re: Double = Double(Float(bin.re))
                let im: Double = Double(Float(bin.im))
                let magnitude: Double = re * re + im * im
                return Double(Float(power == 2 ? magnitude : magnitude.squareRoot()))
            }
        }
    }

    static func hertzToMel(_ hertz: Double) -> Double {
        let linearStep = 200.0 / 3, logStart = 1000.0
        guard hertz >= logStart else { return hertz / linearStep }
        return logStart / linearStep + log(hertz / logStart) / (log(6.4) / 27)
    }

    static func melToHertz(_ mel: Double) -> Double {
        let linearStep = 200.0 / 3, logStart = 1000.0, logStartMel = logStart / linearStep
        guard mel >= logStartMel else { return mel * linearStep }
        return logStart * exp((log(6.4) / 27) * (mel - logStartMel))
    }

    /// librosa's Slaney mel filterbank `[bands][fftSize / 2 + 1]` from 0 Hz to Nyquist, area-normalized,
    /// stored as float32.
    static func melFilterbank(rate: Double, fftSize: Int, bands: Int) -> [[Double]] {
        let bins = fftSize / 2 + 1
        let low = hertzToMel(0), high = hertzToMel(rate / 2)
        let edges = (0 ..< bands + 2).map { (index: Int) -> Double in
            melToHertz(low + (high - low) * Double(index) / Double(bands + 1))
        }
        let frequencies = (0 ..< bins).map { (bin: Int) -> Double in rate / 2 * Double(bin) / Double(bins - 1) }
        return (0 ..< bands).map { (band: Int) -> [Double] in
            let lowerWidth: Double = edges[band + 1] - edges[band]
            let upperWidth: Double = edges[band + 2] - edges[band + 1]
            let area: Double = 2 / (edges[band + 2] - edges[band])
            return frequencies.map { (f: Double) -> Double in
                let lower: Double = -(edges[band] - f) / lowerWidth
                let upper: Double = (edges[band + 2] - f) / upperWidth
                return Double(Float(Swift.max(0, Swift.min(lower, upper)) * area))
            }
        }
    }

    /// The mel power spectrogram `[frames][bands]`, stored as float32.
    static func melSpectrogram(_ power: [[Double]], rate: Double, fftSize: Int, bands: Int) -> [[Double]] {
        let bank = melFilterbank(rate: rate, fftSize: fftSize, bands: bands)
        return power.map { (frame: [Double]) -> [Double] in
            bank.map { (filter: [Double]) -> Double in dotProduct(filter, frame) }
        }
    }

    /// `Σ a[i]·b[i]`, summed left to right in double precision and stored as float32.
    static func dotProduct(_ a: [Double], _ b: [Double]) -> Double {
        var sum = 0.0
        for (x, y) in zip(a, b) {
            sum += x * y
        }
        return Double(Float(sum))
    }

    /// librosa's `power_to_db(S)`: `10 log10(max(S, 1e-10))`, floored 80 dB below the loudest value,
    /// stored as float32.
    static func powerToDecibels(_ power: [[Double]]) -> [[Double]] {
        let decibels = power.map { (frame: [Double]) -> [Double] in
            frame.map { (value: Double) -> Double in Double(Float(10 * log10(Swift.max(value, 1e-10)))) }
        }
        let loudest: Double = decibels.flatMap { (frame: [Double]) -> [Double] in frame }.max() ?? 0
        let floor: Double = loudest - 80
        return decibels.map { (frame: [Double]) -> [Double] in
            frame.map { (value: Double) -> Double in Swift.max(value, floor) }
        }
    }

    /// The first `count` coefficients of SciPy's orthonormal DCT-II over the mel bands, per frame.
    static func cepstrum(_ decibels: [[Double]], count: Int) -> [[Double]] {
        guard let bands = decibels.first?.count else { return [] }
        let basis = (0 ..< count).map { (k: Int) -> [Double] in
            let scale: Double = k == 0 ? (1 / Double(4 * bands)).squareRoot() : (1 / Double(2 * bands)).squareRoot()
            return (0 ..< bands).map { (n: Int) -> Double in
                let angle: Double = Double.pi * Double(k) * Double(2 * n + 1) / Double(2 * bands)
                return 2 * cos(angle) * scale
            }
        }
        return decibels.map { (frame: [Double]) -> [Double] in
            basis.map { (row: [Double]) -> Double in dotProduct(row, frame) }
        }
    }

    /// librosa's `onset_strength`: the mean over mel bands of each frame's rise from the previous frame,
    /// shifted three frames later (the lag and half a window over the hop), stored as float32.
    static func onsetStrength(_ decibels: [[Double]]) -> [Double] {
        let frames = decibels.count
        let shift = 1 + fftSize / (2 * hop)
        var envelope = [Double](repeating: 0, count: frames)
        for t in shift ..< frames {
            let current = decibels[t - shift + 1], previous = decibels[t - shift]
            let rise = zip(current, previous).reduce(0.0) { (sum: Double, pair: (Double, Double)) -> Double in
                sum + Swift.max(0, pair.0 - pair.1)
            }
            envelope[t] = Double(Float(rise / Double(current.count)))
        }
        return envelope
    }

    // MARK: - Onsets and beats

    static let float32Tiny: Double = Double(Float.leastNormalMagnitude)

    /// librosa's `onset_detect` over an envelope: normalized to `[0, 1]`, then `peak_pick` with the
    /// windows `0.03`, `0`, `0.1` and `0.1` seconds and a wait of `0.03` seconds at `rate`, delta 0.07.
    static func onsetPeaks(_ envelope: [Double], rate: Double, hop: Int) -> [Int] {
        guard let low = envelope.min() else { return [] }
        let normalized = envelope.map { (value: Double) -> Double in Double(Float(value - low)) }
        guard normalized.contains(where: { $0 != 0 }) else { return [] }
        let scale: Double = (normalized.max() ?? 0) + float32Tiny
        let x = normalized.map { (value: Double) -> Double in Double(Float(value / scale)) }
        func frames(_ seconds: Double) -> Int { Int((seconds * rate / Double(hop)).rounded(.down)) }
        return peakPick(x, preMaximum: frames(0.03), postMaximum: 1, preAverage: frames(0.1),
                        postAverage: frames(0.1) + 1, delta: 0.07, wait: frames(0.03))
    }

    /// librosa's `__peak_pick`.
    static func peakPick(_ x: [Double], preMaximum: Int, postMaximum: Int, preAverage: Int, postAverage: Int,
                         delta: Double, wait: Int) -> [Int] {
        func maximum(_ range: Range<Int>) -> Double { x[range].max() ?? 0 }
        func mean(_ range: Range<Int>) -> Double { Double(Float(x[range].reduce(0, +) / Double(range.count))) }
        var peaks = [Int]()
        let count = x.count
        var first = x[0] >= maximum(0 ..< Swift.min(postMaximum, count))
        first = first && x[0] >= mean(0 ..< Swift.min(postAverage, count)) + delta
        var n = 1
        if first {
            peaks.append(0)
            n = wait + 1
        }
        while n < count {
            guard x[n] == maximum(Swift.max(0, n - preMaximum) ..< Swift.min(n + postMaximum, count)) else {
                n += 1
                continue
            }
            guard x[n] >= mean(Swift.max(0, n - preAverage) ..< Swift.min(n + postAverage, count)) + delta else {
                n += 1
                continue
            }
            peaks.append(n)
            n += wait + 1
        }
        return peaks
    }

    /// librosa's `tempo`: the autocorrelation tempogram's mean over frames under a log-normal prior
    /// centered on `startTempo` (one octave wide), capped at 320 beats a minute.
    static func tempo(_ envelope: [Double], rate: Double, hop: Int, startTempo: Double) -> Double {
        let window = Int((8.0 * rate / Double(hop)).rounded(.down))
        let profile = tempogramMean(envelope, window: window)
        var best = -Double.infinity, bestTempo = 0.0
        var capped = true
        for lag in 0 ..< window {
            let bpm = lag == 0 ? Double.infinity : 60 * rate / (Double(hop) * Double(lag))
            if capped && bpm < 320 { capped = false }
            let prior = capped ? -Double.infinity : -0.5 * pow(log2(bpm) - log2(startTempo), 2)
            let score = log1p(1e6 * profile[lag]) + prior
            if score > best {
                best = score
                bestTempo = bpm
            }
        }
        return bestTempo
    }

    /// librosa's `tempogram(win_length: window)` averaged over frames: each frame's Hann-windowed slice of
    /// the envelope (linearly ramped to zero over half a window at each end), autocorrelated and
    /// normalized to its peak.
    static func tempogramMean(_ envelope: [Double], window: Int) -> [Double] {
        let count = envelope.count
        let pad = window / 2
        var padded = [Double](repeating: 0, count: count + 2 * pad)
        for i in 0 ..< count { padded[pad + i] = envelope[i] }
        if count > 0 {
            for i in 0 ..< pad {
                padded[i] = envelope[0] * Double(i) / Double(pad)
                padded[pad + count + i] = envelope[count - 1] * Double(pad - 1 - i) / Double(pad)
            }
        }
        let hann = periodicHann(window)
        var fftLength = 1
        while fftLength < 2 * window - 1 { fftLength *= 2 }
        var sum = [Double](repeating: 0, count: window)
        for t in 0 ..< count {
            var frame = [Double](repeating: 0, count: fftLength)
            for i in 0 ..< window { frame[i] = Double(Float(padded[t + i])) * hann[i] }
            let spectrum = NFKMLXRealFFT.forward(frame)
            let power = spectrum.map { (bin: (re: Double, im: Double)) -> Double in bin.re * bin.re + bin.im * bin.im }
            let bins = power.map { (value: Double) -> (re: Double, im: Double) in (re: value, im: 0.0) }
            let autocorrelation = NFKMLXRealFFT.inverse(bins, count: fftLength)
            let peak = autocorrelation[0 ..< window].map { (value: Double) -> Double in Swift.abs(value) }.max() ?? 0
            let length = peak < Double.leastNormalMagnitude ? 1 : peak
            for i in 0 ..< window { sum[i] += autocorrelation[i] / length }
        }
        let frames = Double(Swift.max(count, 1))
        return sum.map { (value: Double) -> Double in value / frames }
    }

    /// librosa's `beat_track(onset_envelope:, start_bpm:, tightness:)`: the tempo re-estimated from the
    /// envelope, the dynamic-programming beat path over the Gaussian-smoothed, standardized envelope,
    /// trimmed of weak leading and trailing beats.
    static func beatTrack(_ envelope: [Double], rate: Double, hop: Int, startTempo: Double, tightness: Double) -> [Int] {
        guard envelope.contains(where: { $0 != 0 }) else { return [] }
        let bpm = tempo(envelope, rate: rate, hop: hop, startTempo: startTempo)
        let framesPerBeat = Double(Float((rate / Double(hop) * 60 / bpm).rounded(.toNearestOrEven)))
        let count = envelope.count
        let mean: Double = envelope.reduce(0, +) / Double(count)
        let squares: Double = envelope.reduce(0) { (sum: Double, value: Double) -> Double in
            sum + (value - mean) * (value - mean)
        }
        let deviation: Double = (squares / Double(count - 1)).squareRoot()
        let divisor: Double = Double(Float(deviation)) + float32Tiny
        let normalized = envelope.map { (value: Double) -> Double in Double(Float(value / divisor)) }
        let span = Int(framesPerBeat)
        let window = (-span ... span).map { (offset: Int) -> Double in
            exp(-0.5 * pow(Double(offset) * 32 / framesPerBeat, 2))
        }
        let width = window.count
        var local = [Double](repeating: 0, count: count)
        for i in 0 ..< count {
            var sum = 0.0
            var k = Swift.max(0, i + width / 2 - count + 1)
            while k < Swift.min(i + width / 2, width) {
                sum += window[k] * normalized[i + width / 2 - k]
                k += 1
            }
            local[i] = Double(Float(sum))
        }
        let threshold = 0.01 * (local.max() ?? 0)
        var backlink = [Int](repeating: -1, count: count)
        var cumulative = [Double](repeating: 0, count: count)
        var firstBeat = true
        let logPeriod = Double(Float(log(Float(framesPerBeat))))
        let nearest = Int((framesPerBeat / 2).rounded(.toNearestOrEven))
        for i in 0 ..< count {
            var bestScore = -Double.infinity, location = -1
            var candidate = i - nearest
            while candidate > i - 2 * Int(framesPerBeat) - 1 {
                if candidate < 0 { break }
                let score = cumulative[candidate] - tightness * pow(log(Double(i - candidate)) - logPeriod, 2)
                if score > bestScore {
                    bestScore = score
                    location = candidate
                }
                candidate -= 1
            }
            cumulative[i] = Double(Float(location >= 0 ? local[i] + bestScore : local[i]))
            if firstBeat && local[i] < threshold {
                backlink[i] = -1
            } else {
                backlink[i] = location
                firstBeat = false
            }
        }
        var isPeak = [Bool](repeating: false, count: count)
        for i in 1 ..< Swift.max(1, count - 1) {
            isPeak[i] = cumulative[i] > cumulative[i - 1] && cumulative[i] >= cumulative[i + 1]
        }
        if count > 1 { isPeak[count - 1] = cumulative[count - 1] > cumulative[count - 2] }
        let peakScores: [Double] = (0 ..< count).filter { (i: Int) -> Bool in isPeak[i] }
            .map { (i: Int) -> Double in cumulative[i] }.sorted()
        let middle = peakScores.count / 2
        let median: Double
        if peakScores.isEmpty {
            median = 0
        } else if peakScores.count % 2 == 1 {
            median = peakScores[middle]
        } else {
            median = (peakScores[middle - 1] + peakScores[middle]) / 2
        }
        let cut = 0.5 * median
        var tail = count - 1
        var n = count - 1
        while n >= 0 {
            if isPeak[n] && cumulative[n] >= cut {
                tail = n
                break
            }
            n -= 1
        }
        var beats = [Bool](repeating: false, count: count)
        n = tail
        while n >= 0 {
            beats[n] = true
            n = backlink[n]
        }
        let beatScores = (0 ..< count).filter { (i: Int) -> Bool in beats[i] }.map { (i: Int) -> Double in local[i] }
        let hann = (0 ..< 5).map { (n: Int) -> Double in 0.5 - 0.5 * cos(2 * Double.pi * Double(n) / 4) }
        var smoothed = [Double]()
        for i in 0 ..< Swift.min(beatScores.count + 2, count) {
            var sum = 0.0
            for (k, weight) in hann.enumerated() {
                let source = i + 2 - k
                if source >= 0 && source < beatScores.count { sum += weight * beatScores[source] }
            }
            smoothed.append(sum)
        }
        let energy: Double = smoothed.reduce(0) { (sum: Double, value: Double) -> Double in sum + value * value }
        let rms: Double = smoothed.isEmpty ? 0 : (energy / Double(smoothed.count)).squareRoot()
        let floor = 0.5 * rms
        n = 0
        while n < count && local[n] <= floor {
            beats[n] = false
            n += 1
        }
        n = count - 1
        while n >= 0 && local[n] <= floor {
            beats[n] = false
            n -= 1
        }
        return (0 ..< count).filter { (i: Int) -> Bool in beats[i] }
    }

    // MARK: - Chroma

    /// librosa's `chroma_cens(n_chroma: 12, bins_per_octave: 36, n_octaves: 7)` `[frames][12]`: the
    /// constant-Q magnitudes at the estimated tuning folded to 12 pitch classes, L1-normalized,
    /// quantized at 0.4, 0.2, 0.1 and 0.05, smoothed over 41 frames, and L2-normalized.
    static func chromaCENS(_ samples: [Double], rate: Double, hop: Int) -> [[Double]] {
        let binsPerOctave = 36
        let tuning = estimatedTuning(samples, rate: rate, binsPerOctave: binsPerOctave)
        let magnitudes = constantQ(samples, rate: rate, hop: hop, bins: 7 * binsPerOctave, binsPerOctave: binsPerOctave,
                                   tuning: tuning)
        let frames = magnitudes.first?.count ?? 0
        var chroma = [[Double]](repeating: [Double](repeating: 0, count: 12), count: frames)
        for (bin, row) in magnitudes.enumerated() {
            let pitchClass = (((bin % binsPerOctave) + 1) / 3) % 12
            for t in 0 ..< frames { chroma[t][pitchClass] += row[t] }
        }
        let steps = [0.4, 0.2, 0.1, 0.05]
        let quantized = chroma.map { (column: [Double]) -> [Double] in
            let values = column.map { (value: Double) -> Double in Double(Float(value)) }
            let total: Double = values.reduce(0) { (sum: Double, value: Double) -> Double in sum + Swift.abs(value) }
            let length: Double = total < float32Tiny ? 1 : total
            return values.map { (value: Double) -> Double in
                let normalized: Double = Double(Float(value / length))
                return steps.reduce(0.0) { (sum: Double, step: Double) -> Double in
                    sum + (normalized > step ? 0.25 : 0)
                }
            }
        }
        let smoothing = 41 + 2
        var window = (0 ..< smoothing).map { (n: Int) -> Double in
            0.5 - 0.5 * cos(2 * Double.pi * Double(n) / Double(smoothing - 1))
        }
        let windowSum = window.reduce(0, +)
        window = window.map { (weight: Double) -> Double in weight / windowSum }
        let center = smoothing / 2
        return (0 ..< frames).map { (t: Int) -> [Double] in
            let smoothed = (0 ..< 12).map { (c: Int) -> Double in
                var sum = 0.0
                for k in 0 ..< smoothing {
                    let source = t + center - k
                    if source >= 0 && source < frames { sum += window[k] * quantized[source][c] }
                }
                return Double(Float(sum))
            }
            let length: Double = smoothed.reduce(0) { (sum: Double, value: Double) -> Double in sum + value * value }
                .squareRoot()
            let divisor: Double = length < float32Tiny ? 1 : length
            return smoothed.map { (value: Double) -> Double in Double(Float(value / divisor)) }
        }
    }

    /// librosa's `estimate_tuning(n_fft: 2048, resolution: 0.01)`: the most common deviation, in fractions of
    /// a bin, of the strong `piptrack` peaks between 150 and 4000 Hz from the equal-tempered grid.
    static func estimatedTuning(_ samples: [Double], rate: Double, binsPerOctave: Int) -> Double {
        let magnitudes = powerSpectrogram(samples, fftSize: fftSize, hop: fftSize / 4, power: 1)
        let bins = fftSize / 2 + 1
        var pitches = [Float](), strengths = [Float]()
        for frame in magnitudes {
            let reference: Double = 0.1 * (frame.max() ?? 0)
            let gated = frame.map { (value: Double) -> Double in value > reference ? value : 0 }
            for bin in 1 ..< bins - 1 {
                let frequency = rate * Double(bin) / Double(fftSize)
                guard frequency >= 150, frequency < Swift.min(4000, rate / 2) else { continue }
                guard gated[bin] > gated[bin - 1], gated[bin] >= gated[bin + 1] else { continue }
                let a = frame[bin + 1] + frame[bin - 1] - 2 * frame[bin]
                let b = (frame[bin + 1] - frame[bin - 1]) / 2
                let shift = Double(Float(Swift.abs(b) >= Swift.abs(a) ? 0 : -b / a))
                let gradient = (frame[bin + 1] - frame[bin - 1]) / 2
                pitches.append(Float((Double(bin) + shift) * rate / Double(fftSize)))
                strengths.append(Float(frame[bin] + Double(Float(0.5 * gradient * shift))))
            }
        }
        let positive = zip(pitches, strengths).filter { (pair: (Float, Float)) -> Bool in pair.0 > 0 }
        guard !positive.isEmpty else { return 0 }
        let sorted: [Float] = positive.map { (pair: (Float, Float)) -> Float in pair.1 }.sorted()
        let middle = sorted.count / 2
        let median: Float
        if sorted.count % 2 == 1 {
            median = sorted[middle]
        } else {
            median = (sorted[middle - 1] + sorted[middle]) / 2
        }
        let selected = positive.filter { (pair: (Float, Float)) -> Bool in pair.1 >= median }
            .map { (pair: (Float, Float)) -> Float in pair.0 }
        return pitchTuning(selected, binsPerOctave: binsPerOctave)
    }

    /// librosa's `pitch_tuning(resolution: 0.01)`: the left edge of the fullest of 100 histogram bins of
    /// the frequencies' deviations from the grid, computed in float32 as librosa computes them.
    static func pitchTuning(_ frequencies: [Float], binsPerOctave: Int) -> Double {
        let edges = (0 ... 100).map { (index: Int) -> Double in Double(index) * 0.01 + -0.5 }
        var counts = [Int](repeating: 0, count: 100)
        for frequency in frequencies where frequency > 0 {
            var residual = Float(binsPerOctave) * log2(frequency / Float(440.0 / 16))
            residual = residual - (residual / 1).rounded(.down)
            if residual >= 0.5 { residual -= 1 }
            let value = Double(residual)
            var index = Int((value + 0.5) * 100)
            if index == 100 { index -= 1 }
            if value < edges[index] { index -= 1 }
            if index != 99 && value >= edges[index + 1] { index += 1 }
            counts[index] += 1
        }
        let best = counts.firstIndex(of: counts.max() ?? 0) ?? 0
        return edges[best]
    }

    /// librosa's `cqt(fmin: C1, sparsity: 0.01, window: "hann")` magnitudes `[bins][frames]`: each octave's
    /// Hann-windowed, L1-normalized complex wavelets applied in the frequency domain to the signal at that
    /// octave's rate, the signal halved (with `resample_poly`) between octaves, the bins scaled by
    /// `1/√length`.
    static func constantQ(_ samples: [Double], rate: Double, hop: Int, bins: Int, binsPerOctave: Int,
                          tuning: Double) -> [[Double]] {
        let minimum = 440 * pow(2, (24.0 - 69) / 12) * pow(2, tuning / Double(binsPerOctave))
        let octaves = (bins + binsPerOctave - 1) / binsPerOctave
        let ratios = (0 ..< binsPerOctave).map { (n: Int) -> Double in pow(2, Double(n) / Double(binsPerOctave)) }
        let frequencies = (0 ..< bins).map { (index: Int) -> Double in
            minimum * pow(2, Double(index / binsPerOctave)) * ratios[index % binsPerOctave]
        }
        let logs = frequencies.map { (frequency: Double) -> Double in log2(frequency) }
        let alpha = (0 ..< bins).map { (index: Int) -> Double in
            let bandsPerOctave: Double
            if index == 0 {
                bandsPerOctave = 1 / (logs[1] - logs[0])
            } else if index == bins - 1 {
                bandsPerOctave = 1 / (logs[bins - 1] - logs[bins - 2])
            } else {
                bandsPerOctave = 2 / (logs[index + 1] - logs[index - 1])
            }
            let ratio: Double = pow(2, 2 / bandsPerOctave)
            return (ratio - 1) / (ratio + 1)
        }
        func lengths(rate: Double) -> [Double] {
            (0 ..< bins).map { (index: Int) -> Double in 1 / alpha[index] * rate / frequencies[index] }
        }
        var signal = samples, octaveRate = rate, octaveHop = hop
        var responses = [[[Double]]]()
        for octave in 0 ..< octaves {
            let upper = bins - binsPerOctave * octave
            let range = Swift.max(0, upper - binsPerOctave) ..< upper
            let octaveLengths = lengths(rate: octaveRate)
            let basis = waveletBasis(frequencies: Array(frequencies[range]), lengths: Array(octaveLengths[range]),
                                     rate: octaveRate)
            let gain = (rate / octaveRate).squareRoot()
            responses.append(constantQResponse(signal, basis: basis.rows, fftSize: basis.fftSize, hop: octaveHop,
                                               gain: gain))
            if octaveHop % 2 == 0 {
                octaveHop /= 2
                octaveRate /= 2
                signal = resampled(signal, from: 2, to: 1, scale: true)
            }
        }
        let frames = responses.map { (response: [[Double]]) -> Int in response.first?.count ?? 0 }.min() ?? 0
        let fullLengths = lengths(rate: rate)
        var output = [[Double]](repeating: [], count: bins)
        for (octave, response) in responses.enumerated() {
            let upper = bins - binsPerOctave * octave
            for (row, index) in (Swift.max(0, upper - binsPerOctave) ..< upper).enumerated() {
                let scale = fullLengths[index].squareRoot()
                output[index] = response[row][0 ..< frames].map { (value: Double) -> Double in
                    Double(Float(value / scale))
                }
            }
        }
        return output
    }

    struct SparseRow {
        var indices: [Int]
        var re: [Double]
        var im: [Double]
    }

    /// One octave's wavelets in the frequency domain, as `__vqt_filter_fft` builds them: each wavelet
    /// centered in a power-of-two frame, scaled by its length over the frame, transformed, its positive
    /// frequencies kept, and the smallest coefficients carrying 1% of its magnitude dropped.
    static func waveletBasis(frequencies: [Double], lengths: [Double], rate: Double) -> (rows: [SparseRow], fftSize: Int) {
        var size = 1
        while Double(size) < (lengths.max() ?? 1) { size *= 2 }
        let rows = zip(frequencies, lengths).map { (frequency: Double, length: Double) -> SparseRow in
            let start = Int((-length / 2).rounded(.down)), end = Int((length / 2).rounded(.down))
            let count = end - start
            let hann = periodicHann(count)
            var re = [Double](), im = [Double]()
            for (i, n) in (start ..< end).enumerated() {
                let phase = Double(n) * 2 * Double.pi * frequency / rate
                re.append(cos(phase) * hann[i])
                im.append(sin(phase) * hann[i])
            }
            let norm: Double = zip(re, im).reduce(0) { (sum: Double, pair: (Double, Double)) -> Double in
                sum + (pair.0 * pair.0 + pair.1 * pair.1).squareRoot()
            }
            let offset = (size - count) / 2
            var paddedRe = [Double](repeating: 0, count: size), paddedIm = [Double](repeating: 0, count: size)
            for i in 0 ..< count {
                paddedRe[offset + i] = Double(Float(re[i] / norm)) * (length / Double(size))
                paddedIm[offset + i] = Double(Float(im[i] / norm)) * (length / Double(size))
            }
            let storedRe = paddedRe.map { (value: Double) -> Double in Double(Float(value)) }
            let storedIm = paddedIm.map { (value: Double) -> Double in Double(Float(value)) }
            let spectrum = NFKMLXRealFFT.complexForward(re: storedRe, im: storedIm)
            let kept = size / 2 + 1
            let magnitudes = (0 ..< kept).map { (i: Int) -> Double in
                let re: Double = spectrum.re[i], im: Double = spectrum.im[i]
                return (re * re + im * im).squareRoot()
            }
            let total = magnitudes.reduce(0, +)
            let sortedMagnitudes = magnitudes.sorted()
            var cumulative = 0.0, cutoffIndex = 0
            for (i, value) in sortedMagnitudes.enumerated() {
                cumulative += value / total
                if cumulative >= 0.01 {
                    cutoffIndex = i
                    break
                }
            }
            let cutoff = sortedMagnitudes[cutoffIndex]
            var row = SparseRow(indices: [], re: [], im: [])
            for i in 0 ..< kept where magnitudes[i] >= cutoff {
                row.indices.append(i)
                row.re.append(Double(Float(spectrum.re[i])))
                row.im.append(Double(Float(spectrum.im[i])))
            }
            return row
        }
        return (rows, size)
    }

    /// The magnitudes `[rows][frames]` of each basis row applied to a rectangular-window centered STFT.
    static func constantQResponse(_ signal: [Double], basis: [SparseRow], fftSize: Int, hop: Int,
                                  gain: Double) -> [[Double]] {
        let frames = centeredFrames(signal, fftSize: fftSize, hop: hop)
        var output = [[Double]](repeating: [Double](repeating: 0, count: frames.count), count: basis.count)
        for (t, frame) in frames.enumerated() {
            typealias Bin = (re: Double, im: Double)
            let spectrum = NFKMLXRealFFT.forward(frame).map { (bin: Bin) -> Bin in
                (re: Double(Float(bin.re)), im: Double(Float(bin.im)))
            }
            for (r, row) in basis.enumerated() {
                var re = 0.0, im = 0.0
                for (k, index) in row.indices.enumerated() {
                    let a = row.re[k] * gain, b = row.im[k] * gain
                    re += a * spectrum[index].re - b * spectrum[index].im
                    im += a * spectrum[index].im + b * spectrum[index].re
                }
                output[r][t] = Double(Float((re * re + im * im).squareRoot()))
            }
        }
        return output
    }
}

/// A radix-2 fast Fourier transform in double precision for power-of-two lengths.
enum NFKMLXRealFFT {
    /// The complex transform of `re + i·im`.
    static func complexForward(re: [Double], im: [Double], inverse: Bool = false) -> (re: [Double], im: [Double]) {
        let n = re.count
        precondition(n > 0 && n & (n - 1) == 0, "the transform length is a power of two")
        var real = re, imaginary = im
        var j = 0
        for i in 1 ..< n {
            var bit = n >> 1
            while j & bit != 0 {
                j ^= bit
                bit >>= 1
            }
            j |= bit
            if i < j {
                real.swapAt(i, j)
                imaginary.swapAt(i, j)
            }
        }
        var length = 2
        let sign = inverse ? 1.0 : -1.0
        while length <= n {
            let angle = sign * 2 * Double.pi / Double(length)
            for start in stride(from: 0, to: n, by: length) {
                for k in 0 ..< length / 2 {
                    let wr = cos(angle * Double(k)), wi = sin(angle * Double(k))
                    let a = start + k, b = a + length / 2
                    let tr = real[b] * wr - imaginary[b] * wi
                    let ti = real[b] * wi + imaginary[b] * wr
                    real[b] = real[a] - tr
                    imaginary[b] = imaginary[a] - ti
                    real[a] += tr
                    imaginary[a] += ti
                }
            }
            length *= 2
        }
        return (real, imaginary)
    }

    /// The non-negative frequencies `[n / 2 + 1]` of a real signal's transform.
    static func forward(_ x: [Double]) -> [(re: Double, im: Double)] {
        let result = complexForward(re: x, im: [Double](repeating: 0, count: x.count))
        return (0 ... x.count / 2).map { (k: Int) -> (re: Double, im: Double) in (re: result.re[k], im: result.im[k]) }
    }

    /// The real signal of length `count` whose non-negative frequencies are `spectrum`.
    static func inverse(_ spectrum: [(re: Double, im: Double)], count: Int) -> [Double] {
        var re = [Double](repeating: 0, count: count), im = [Double](repeating: 0, count: count)
        for (k, bin) in spectrum.enumerated() where k < count {
            re[k] = bin.re
            im[k] = bin.im
            if k > 0 && k < count - k {
                re[count - k] = bin.re
                im[count - k] = -bin.im
            }
        }
        return complexForward(re: re, im: im, inverse: true).re.map { $0 / Double(count) }
    }
}
