//
//  NFKMLXGemma3nProcessors.swift
//  InferKitMLX
//
//  Turning a caller's picture and audio into what Gemma 3n's towers read.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXFFT
import MLXNN

/// Gemma 3n's image processor: a picture becomes one `[1, 768, 768, 3]` frame in `0...1`, the aspect
/// ratio squashed to the square.
///
/// @discussion The release's preprocessor file states an `image_mean` and an `image_std` and then sets
/// `do_normalize` false, so neither is applied: the pixels are scaled to `0...1` and nothing more.
/// Feeding a `-1...1` or ImageNet-normalized frame produces a plausible picture and wrong features.
///
/// The resize is CoreGraphics bilinear rather than the reference's PIL bilinear, so the pixel values
/// differ slightly and an answer may not be token-identical to the reference's; the tower is at
/// reference parity on the reference's own pixel values.
public enum NFKMLXGemma3nImageProcessor {
    /// `image` is a `CGImage`, `CVPixelBuffer`, or `MTLTexture` — what `NFKInputImage` carries.
    public static func pixelValues(from image: Any, imageSize: Int = 768) throws -> MLXArray {
        let rgb = try NFKMLXImageBridge.tensor(from: image, channels: 3,
                                               colorSpace: CGColorSpaceCreateDeviceRGB())
        let batched = rgb.reshaped([1, rgb.shape[0], rgb.shape[1], rgb.shape[2]])
        return NFKMLXResample.resizeBilinear(batched, height: imageSize, width: imageSize)
    }
}

/// Gemma 3n's audio front end: a 16 kHz waveform becomes the `[1, frames, 128]` log-mel the encoder
/// reads.
///
/// @discussion Three details separate it from the other mel front ends here. The frames are cut ONE
/// SAMPLE LONGER than the window so the pre-emphasis has a predecessor for every sample it keeps, and
/// that pre-emphasis is HTK's flavor, which scales the first sample rather than dropping it. The
/// transform is run at twice the window's length (`fft_overdrive`), so the spectrum is finer than the
/// window alone would give. And the mel scale is HTK's while the normalization is Slaney's, a pairing
/// neither of this package's other filterbanks uses.
public struct NFKMLXGemma3nAudioFeatures: Sendable {
    public var sampleRate: Int
    public var frameLength: Int
    public var hopLength: Int
    public var fftLength: Int
    public var bands: Int
    public var minimumFrequency: Float
    public var maximumFrequency: Float
    public var preemphasis: Float
    public var melFloor: Float

    public init(sampleRate: Int = 16_000, frameLength: Int = 512, hopLength: Int = 160,
                fftLength: Int = 1024, bands: Int = 128, minimumFrequency: Float = 125,
                maximumFrequency: Float = 7600, preemphasis: Float = 0.97, melFloor: Float = 1e-5) {
        self.sampleRate = sampleRate
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.fftLength = fftLength
        self.bands = bands
        self.minimumFrequency = minimumFrequency
        self.maximumFrequency = maximumFrequency
        self.preemphasis = preemphasis
        self.melFloor = melFloor
    }

    /// The log-mel spectrogram `[1, frames, bands]` for a mono waveform.
    public func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        let samples = waveform.reshaped([-1])
        // A frame is cut one sample longer than the window, because the pre-emphasis reads a
        // predecessor for each sample it keeps.
        let width = frameLength + 1
        let count = (samples.shape[0] - width) / hopLength + 1
        guard count > 0 else { return MLXArray.zeros([1, 0, bands]) }

        var indices = [Int32]()
        indices.reserveCapacity(count * width)
        for frame in 0 ..< count {
            for position in 0 ..< width {
                indices.append(Int32(frame * hopLength + position))
            }
        }
        let frames = samples.take(MLXArray(indices)).reshaped([count, width])

        // HTK's pre-emphasis keeps the first sample, scaled, where the plain form drops it.
        let first = frames[0..., 0 ..< 1] * (1 - preemphasis)
        let rest = frames[0..., 1 ..< (width - 1)] - frames[0..., 0 ..< (width - 2)] * preemphasis
        let emphasized = concatenated([first, rest], axis: 1) * MLXArray(window)

        let spectrum = MLXFFT.rfft(emphasized, n: fftLength, axis: -1)
        let magnitude = sqrt(spectrum.realPart() * spectrum.realPart()
                             + spectrum.imaginaryPart() * spectrum.imaginaryPart())
        let mel = matmul(magnitude, MLXArray(filterbank).reshaped([fftLength / 2 + 1, bands]))
        return log(maximum(mel, melFloor)).expandedDimensions(axis: 0)
    }

    /// The periodic Hann window, which divides by the frame length rather than one less.
    var window: [Float] {
        (0 ..< frameLength).map { 0.5 * (1 - Foundation.cos(2 * .pi * Float($0) / Float(frameLength))) }
    }

    /// A triangular filterbank on HTK's mel scale, NOT area-normalized.
    ///
    /// @discussion The reference builds it with `norm=None`, so the triangles keep their unit peaks.
    /// Slaney's area normalization is what every other filterbank in this package applies, and adding
    /// it here shifts each band's log by a constant — which no shape reveals.
    var filterbank: [Float] {
        func mel(_ hertz: Double) -> Double { 2595 * Foundation.log10(1 + hertz / 700) }
        func hertz(_ mel: Double) -> Double { 700 * (Foundation.pow(10, mel / 2595) - 1) }

        let bins = fftLength / 2 + 1
        let frequencies = (0 ..< bins).map { Double($0) * Double(sampleRate) / Double(fftLength) }
        let low = mel(Double(minimumFrequency)), high = mel(Double(maximumFrequency))
        let points = (0 ..< (bands + 2)).map {
            hertz(low + (high - low) * Double($0) / Double(bands + 1))
        }

        var matrix = [Float](repeating: 0, count: bins * bands)
        for band in 0 ..< bands {
            let (left, center, right) = (points[band], points[band + 1], points[band + 2])
            for bin in 0 ..< bins {
                let frequency = frequencies[bin]
                let rising = (frequency - left) / (center - left)
                let falling = (right - frequency) / (right - center)
                matrix[bin * bands + band] = Float(Swift.max(0, Swift.min(rising, falling)))
            }
        }
        return matrix
    }
}
