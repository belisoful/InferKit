//
//  NFKMLXGemma4AudioFeatureExtractor.swift
//  InferKitMLX
//
//  The Gemma 4 audio front end: raw 16 kHz audio to the log-mel features `[frames, 128]` the audio
//  Conformer's subsampler reads. It frames the waveform with a semicausal padding, applies a periodic
//  Hann window, takes the magnitude of a 512-point real FFT, projects it through a 128-band HTK
//  triangular mel filterbank, and takes `log(mel + melFloor)`.
//

import Foundation
import MLX
import MLXFFT

/// The Gemma 4 audio feature extractor. The defaults are the released extractor's: 16 kHz, 20 ms
/// frames, 10 ms hop, 128 HTK mel bands from 0 to 8 kHz, a 512-point FFT, and a `1e-3` mel floor.
public struct NFKMLXGemma4AudioFeatureExtractor {
    public var sampleRate: Int
    public var frameLength: Int
    public var hopLength: Int
    public var featureSize: Int
    public var fftLength: Int
    public var minFrequency: Float
    public var maxFrequency: Float
    public var melFloor: Float

    public init(sampleRate: Int = 16_000, frameLength: Int = 320, hopLength: Int = 160,
                featureSize: Int = 128, fftLength: Int = 512, minFrequency: Float = 0,
                maxFrequency: Float = 8_000, melFloor: Float = 1e-3) {
        self.sampleRate = sampleRate
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.featureSize = featureSize
        self.fftLength = fftLength
        self.minFrequency = minFrequency
        self.maxFrequency = maxFrequency
        self.melFloor = melFloor
    }

    /// The log-mel features `[frames, featureSize]` for a mono waveform in `-1 … 1`.
    public func features(_ waveform: [Float]) -> MLXArray {
        // Semicausal padding: prepend `frameLength / 2` zeros so the first frame is centered at t = 0.
        let padded = [Float](repeating: 0, count: frameLength / 2) + waveform
        // The reference unfolds `frameLength + 1` samples per frame and drops the last, so the frame
        // count is measured against that window.
        let unfoldSize = frameLength + 1
        let frames = padded.count >= unfoldSize ? (padded.count - unfoldSize) / hopLength + 1 : 0
        guard frames > 0 else { return MLXArray.zeros([0, featureSize]) }

        let window = (0 ..< frameLength).map { 0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(frameLength)) }
        var windowed = [Float](repeating: 0, count: frames * fftLength)   // right-padded to fftLength
        for frame in 0 ..< frames {
            for sample in 0 ..< frameLength {
                windowed[frame * fftLength + sample] = padded[frame * hopLength + sample] * window[sample]
            }
        }
        let frameArray = windowed.withUnsafeBufferPointer { MLXArray($0, [frames, fftLength]) }
        let spectrum = rfft(frameArray, axis: 1)                          // [frames, fftLength/2 + 1]
        let magnitude = sqrt(spectrum.realPart().square() + spectrum.imaginaryPart().square())
        let mel = magnitude.matmul(melFilterBank())                      // [frames, featureSize]
        return log(mel + melFloor)
    }

    /// The HTK triangular mel filterbank `[fftLength/2 + 1, featureSize]`, unnormalized — the same
    /// `mel_filter_bank(mel_scale="htk", norm=None)` the reference builds.
    func melFilterBank() -> MLXArray {
        let bins = fftLength / 2 + 1
        func hertzToMel(_ frequency: Float) -> Float { 2595 * log10f(1 + frequency / 700) }
        func melToHertz(_ mel: Float) -> Float { 700 * (powf(10, mel / 2595) - 1) }
        let melMinimum = hertzToMel(minFrequency), melMaximum = hertzToMel(maxFrequency)
        let filterFrequencies = (0 ... featureSize + 1).map {
            melToHertz(melMinimum + (melMaximum - melMinimum) * Float($0) / Float(featureSize + 1))
        }
        let nyquist = Float(sampleRate) / 2
        let fftFrequencies = (0 ..< bins).map { nyquist * Float($0) / Float(bins - 1) }

        var filters = [Float](repeating: 0, count: bins * featureSize)
        for filter in 0 ..< featureSize {
            let lower = filterFrequencies[filter]
            let center = filterFrequencies[filter + 1]
            let upper = filterFrequencies[filter + 2]
            for bin in 0 ..< bins {
                let frequency = fftFrequencies[bin]
                let leftSlope = (frequency - lower) / (center - lower)
                let rightSlope = (upper - frequency) / (upper - center)
                filters[bin * featureSize + filter] = Swift.max(0, Swift.min(leftSlope, rightSlope))
            }
        }
        return filters.withUnsafeBufferPointer { MLXArray($0, [bins, featureSize]) }
    }
}
