//
//  NFKMLXPhi4MMProcessor.swift
//  InferKitMLX
//
//  Phi-4-multimodal's input preprocessing, ported from the release's `processing_phi4mm.py`: the
//  SpeechLib log-mel filterbank the speech tower reads, and the dynamic-HD crop layout the image tower
//  reads. Both are measured against the reference processor's own outputs on the validation clip and
//  photo.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXFFT

// MARK: - Audio

/// The SpeechLib log-mel filterbank: 80 bands, 25 ms Hamming frames every 10 ms with no centering,
/// per-frame pre-emphasis, and a natural log of the band power floored at one.
///
/// @discussion The sample-rate handling is the reference processor's, reproduced as written:
/// - above 16 kHz → decimated by the integer `rate / 16000` with SciPy's `resample_poly` (a Kaiser β 5
///   FIR), then read as 16 kHz. 48 kHz becomes true 16 kHz; 44.1 kHz becomes 22.05 kHz read as 16 kHz,
///   and 22.05 or 24 kHz pass through unchanged, read as 16 kHz.
/// - between 8 and 16 kHz → decimated by the integer `rate / 8000`, then read as 8 kHz.
/// - 8 kHz → a 256-point transform over 25 ms frames every 10 ms, its bins below Nyquist placed in the
///   lower half of the 16 kHz spectrum and the upper half zero-filled.
/// - below 8 kHz → refused.
public enum NFKMLXPhi4MMAudioFeatures {
    public static let sampleRate = 16000
    static let window = 400
    static let hop = 160
    static let fftSize = 512
    static let bands = 80
    static let preemphasis: Float = 0.97

    /// SpeechLib's triangular mel bank, `[fftSize/2 + 1, bands]`: triangles evenly spaced on the
    /// `1127·ln(1 + f/700)` scale up to 7690 Hz, over FFT bins `1 ..< f2bin(7690)`.
    static let melBank: [Float] = {
        let width = fftSize / 2 + 1
        let rate = Double(sampleRate), n = Double(fftSize)
        func mel(_ f: Double) -> Double { 1127.0 * log(1.0 + f / 700.0) }
        func binToMel(_ bin: Int) -> Double { 1127.0 * log(1.0 + Double(bin) * rate / (n * 700.0)) }
        func frequencyToBin(_ f: Double) -> Int { Int(f * n / rate + 0.5) }
        let low = frequencyToBin(0) + 1
        let high = Swift.max(frequencyToBin(7690), low)
        let top = mel(7690)
        let centers = (0 ..< (bands + 2)).map { top * Double($0) / Double(bands + 1) }
        let spacing = top / Double(bands + 1)
        var matrix = [Float](repeating: 0, count: width * bands)
        for band in 0 ..< bands {
            let (left, center, right) = (centers[band], centers[band + 1], centers[band + 2])
            for bin in low ..< high {
                let m = binToMel(bin)
                if left < m && m < right {
                    matrix[bin * bands + band] = Float(1.0 - abs(center - m) / spacing)
                }
            }
        }
        return matrix
    }()

    /// NumPy's symmetric `hamming(length)`.
    static func hamming(_ length: Int) -> [Float] {
        (0 ..< length).map { Float(0.54 - 0.46 * cos(2 * Double.pi * Double($0) / Double(length - 1))) }
    }

    /// The log-mel features `[frames, 80]` of 16 kHz mono samples. A trailing partial frame is dropped.
    public static func logMel(_ samples: [Float]) -> MLXArray {
        magnitudes(samples, window: window, hop: hop, fftSize: fftSize, bins: fftSize / 2 + 1).logMel
    }

    /// The log-mel features `[frames, 80]` of mono samples at `sampleRate`, handling the rate the way the
    /// reference processor does.
    public static func logMel(_ samples: [Float], sampleRate rate: Int) throws -> MLXArray {
        if rate > 16000 {
            return logMel(resampledByDecimation(samples, factor: rate / 16000))
        }
        if rate == 16000 { return logMel(samples) }
        guard rate >= 8000 else {
            throw NFKMLXError.unsupportedConfiguration("Phi-4-multimodal reads audio at 8 kHz or above, not \(rate) Hz")
        }
        let eightKilohertz = rate == 8000 ? samples : resampledByDecimation(samples, factor: rate / 8000)
        // 25 ms every 10 ms at 8 kHz over a 256-point transform; the bins below Nyquist fill the lower
        // half of the 16 kHz spectrum, the rest stay zero.
        return magnitudes(eightKilohertz, window: 200, hop: 80, fftSize: 256, bins: 128).logMel
    }

    /// Frames, pre-emphasizes, windows, and transforms `samples`, keeping the first `bins` magnitudes and
    /// zero-filling up to the 16 kHz spectrum's 257 bins.
    private static func magnitudes(_ samples: [Float], window: Int, hop: Int, fftSize: Int, bins: Int)
        -> (spectrum: MLXArray, logMel: MLXArray) {
        let frames = Swift.max((samples.count - window) / hop + 1, 0)
        var emphasized = [Float](repeating: 0, count: frames * window)
        for frame in 0 ..< frames {
            let start = frame * hop
            // Pre-emphasis runs within each frame; the first sample is its own predecessor.
            for i in 0 ..< window {
                let previous = samples[start + Swift.max(i - 1, 0)]
                emphasized[frame * window + i] = (samples[start + i] - preemphasis * previous) * 32768
            }
        }
        let windowed = MLXArray(emphasized).reshaped([frames, window]) * MLXArray(hamming(window))
        let zeroPadded = concatenated([windowed, MLXArray.zeros([frames, fftSize - window])], axis: 1)
        var spectrum = abs(MLXFFT.rfft(zeroPadded, axis: 1))[0..., 0 ..< bins]
        let full = Self.fftSize / 2 + 1
        if bins < full {
            spectrum = concatenated([spectrum, MLXArray.zeros([frames, full - bins])], axis: 1)
        }
        let power = spectrum * spectrum
        let bank = MLXArray(melBank).reshaped([full, bands])
        return (spectrum, log(maximum(matmul(power, bank), MLXArray(Float(1)))))
    }

    /// SciPy's `resample_poly(x, 1, factor)` with its default Kaiser β 5 window: a `firwin` low-pass of
    /// `20·factor + 1` taps at cutoff `1/factor`, applied centered at every `factor`-th sample, with
    /// samples outside the signal read as zero. Output length is `ceil(count / factor)`.
    static func resampledByDecimation(_ samples: [Float], factor: Int) -> [Float] {
        guard factor > 1 else { return samples }
        let half = 10 * factor
        let taps = firwinKaiser(count: 2 * half + 1, cutoff: 1 / Double(factor), beta: 5)
        let outputCount = (samples.count + factor - 1) / factor
        var output = [Float](repeating: 0, count: outputCount)
        for index in 0 ..< outputCount {
            let center = index * factor + half
            var sum = 0.0
            for (tap, weight) in taps.enumerated() {
                let source = center - tap
                if source >= 0 && source < samples.count { sum += weight * Double(samples[source]) }
            }
            output[index] = Float(sum)
        }
        return output
    }

    /// SciPy's `firwin(count, cutoff, window=("kaiser", beta))`: a windowed-sinc low-pass normalized to
    /// unit gain at DC.
    static func firwinKaiser(count: Int, cutoff: Double, beta: Double) -> [Double] {
        func besselI0(_ x: Double) -> Double {
            var sum = 1.0, term = 1.0
            for k in 1 ..< 50 {
                term *= (x / 2) / Double(k)
                sum += term * term
            }
            return sum
        }
        let alpha = 0.5 * Double(count - 1)
        let normalizer = besselI0(beta)
        var taps = (0 ..< count).map { n -> Double in
            let m = Double(n) - alpha
            let argument = cutoff * m
            let sinc = argument == 0 ? 1 : sin(Double.pi * argument) / (Double.pi * argument)
            let ratio = m / alpha
            return cutoff * sinc * besselI0(beta * Swift.max(0, 1 - ratio * ratio).squareRoot()) / normalizer
        }
        let total = taps.reduce(0, +)
        taps = taps.map { $0 / total }
        return taps
    }

    /// The audio-token count the prompt reserves for `frames` feature frames: the frames the encoder's
    /// eight-fold subsampler leaves, rounded up.
    public static func tokenCount(frames: Int) -> Int { (frames + 7) / 8 }
}

// MARK: - Image

/// One image prepared for the tower: the global view followed by the sub-image crops, the padded HD
/// size, the unpadded patch region, and the number of image tokens the prompt reserves.
public struct NFKMLXPhi4MMImageInput {
    /// `[crops, 448, 448, 3]`, channels last, normalized to `[-1, 1]`.
    public let pixels: MLXArray
    public let imageSize: (height: Int, width: Int)
    public let validPatches: (rows: Int, columns: Int)
    public let tokenCount: Int
}

/// One audio clip for Phi-4-multimodal: mono samples in `[-1, 1]` at their own rate, which the feature
/// extractor reads the way the reference processor reads that rate.
public struct NFKMLXPhi4MMAudioInput {
    public let samples: [Float]
    public let sampleRate: Int

    public init(samples: [Float], sampleRate: Int = 16000) {
        self.samples = samples
        self.sampleRate = sampleRate
    }
}

/// The dynamic-HD image preprocessor: the picture is fit into a grid of 448-pixel crops (up to 36),
/// resized with PIL's bilinear filter, padded white on the right and bottom, and normalized; a bicubic
/// 448×448 global view precedes the crops.
public enum NFKMLXPhi4MMImageProcessor {
    static let crop = 448
    static let patch = 14
    static let maximumCrops = 36

    /// Prepares 8-bit RGB pixels `[height, width, 3]`.
    public static func process(rgb: [UInt8], width: Int, height: Int) -> NFKMLXPhi4MMImageInput {
        let grid = cropGrid(width: width, height: height)
        let targetWidth = crop * grid.columns, targetHeight = crop * grid.rows
        let ratioWidth = Double(targetWidth) / Double(width)
        let ratioHeight = Double(targetHeight) / Double(height)
        let newWidth: Int, newHeight: Int, padWidth: Int, padHeight: Int
        if ratioWidth < ratioHeight {
            (newWidth, newHeight) = (targetWidth, Int(Double(height) * ratioWidth))
            (padWidth, padHeight) = (0, targetHeight - newHeight)
        } else {
            (newWidth, newHeight) = (Int(Double(width) * ratioHeight), targetHeight)
            (padWidth, padHeight) = (targetWidth - newWidth, 0)
        }
        let maskRows = (crop / patch) * grid.rows, maskColumns = (crop / patch) * grid.columns
        let validPatches = (rows: maskRows - (padHeight >= patch ? padHeight / patch : 0),
                            columns: maskColumns - (padWidth >= patch ? padWidth / patch : 0))

        let resized = NFKMLXPILResample.resampled(rgb, width: width, height: height,
                                                  toWidth: newWidth, toHeight: newHeight)
        var padded = [UInt8](repeating: 255, count: targetWidth * targetHeight * 3)
        for row in 0 ..< newHeight {
            let source = row * newWidth * 3, destination = row * targetWidth * 3
            padded.replaceSubrange(destination ..< (destination + newWidth * 3),
                                   with: resized[source ..< (source + newWidth * 3)])
        }
        let normalized = MLXArray(padded.map { Float($0) / 255 }).reshaped([targetHeight, targetWidth, 3]) * 2 - 1

        let global = bicubicResized(normalized, height: crop, width: crop)            // [448, 448, 3]
        let tiles = normalized
            .reshaped([grid.rows, crop, grid.columns, crop, 3])
            .transposed(0, 2, 1, 3, 4)
            .reshaped([grid.rows * grid.columns, crop, crop, 3])
        let pixels = concatenated([global.expandedDimensions(axis: 0), tiles], axis: 0)
        return NFKMLXPhi4MMImageInput(
            pixels: pixels, imageSize: (height: targetHeight, width: targetWidth), validPatches: validPatches,
            tokenCount: NFKMLXPhi4MMImageNet.tokenCount(validPatches: validPatches))
    }

    /// The crop grid `(rows, columns)`: one crop per 448 pixels on each side, or, past 36 crops, the
    /// allowed grid whose aspect ratio is closest to the picture's (ties go to the larger grid when the
    /// picture covers more than half of it).
    static func cropGrid(width: Int, height: Int) -> (rows: Int, columns: Int) {
        let columns = (width + crop - 1) / crop, rows = (height + crop - 1) / crop
        guard columns * rows > maximumCrops else { return (rows, columns) }
        var ratios = [(Int, Int)]()
        for i in 1 ... maximumCrops {
            for j in 1 ... maximumCrops where i * j <= maximumCrops { ratios.append((i, j)) }
        }
        ratios.sort { ($0.0 * $0.1, $0.0, $0.1) < ($1.0 * $1.1, $1.0, $1.1) }
        let aspect = Double(width) / Double(height)
        let area = Double(width * height)
        var best = (1, 1), bestDifference = Double.infinity
        for ratio in ratios {
            let difference = abs(aspect - Double(ratio.0) / Double(ratio.1))
            if difference < bestDifference {
                (bestDifference, best) = (difference, ratio)
            } else if difference == bestDifference,
                      area > 0.5 * Double(crop * crop * ratio.0 * ratio.1) {
                best = ratio
            }
        }
        return (rows: best.1, columns: best.0)
    }

    /// `torch.nn.functional.interpolate(mode: "bicubic", align_corners: false)` without antialiasing
    /// (cubic coefficient −0.75, edge taps clamped), applied to `[height, width, channels]`.
    static func bicubicResized(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let rows = resampleMatrix(from: x.dim(0), to: height)                          // [height, inH]
        let columns = resampleMatrix(from: x.dim(1), to: width)                        // [width, inW]
        let vertical = matmul(rows, x.transposed(2, 0, 1))                             // [c, height, inW]
        return matmul(vertical, columns.transposed(1, 0)).transposed(1, 2, 0)           // [height, width, c]
    }

    static func resampleMatrix(from inSize: Int, to outSize: Int) -> MLXArray {
        let a = -0.75
        func near(_ x: Double) -> Double { ((a + 2) * x - (a + 3)) * x * x + 1 }
        func far(_ x: Double) -> Double { ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
        let scale = Double(inSize) / Double(outSize)
        var matrix = [Float](repeating: 0, count: outSize * inSize)
        for index in 0 ..< outSize {
            let source = (Double(index) + 0.5) * scale - 0.5
            let floor = source.rounded(.down)
            let t = source - floor
            let weights = [far(t + 1), near(t), near(1 - t), far(2 - t)]
            for (tap, weight) in weights.enumerated() {
                let column = Swift.min(Swift.max(Int(floor) - 1 + tap, 0), inSize - 1)
                matrix[index * inSize + column] += Float(weight)
            }
        }
        return MLXArray(matrix).reshaped([outSize, inSize])
    }
}

/// PIL's 8-bit resample, exactly: a horizontal pass into an 8-bit intermediate, then a vertical pass,
/// each accumulating fixed-point weights and rounding once. The filter's support widens with the scale
/// when shrinking, which is PIL's antialiasing.
enum NFKMLXPILResample {
    /// PIL's `BILINEAR` (a triangle, support 1) or `BICUBIC` (Keys' cubic with a = −0.5, support 2).
    enum Filter: Sendable {
        case bilinear
        case bicubic

        var support: Double { self == .bilinear ? 1 : 2 }

        func weight(_ value: Double) -> Double {
            let x = abs(value)
            switch self {
            case .bilinear:
                return Swift.max(0, 1 - x)
            case .bicubic:
                let a = -0.5
                if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
                if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
                return 0
            }
        }

        /// The filter a `resample` code in a preprocessor configuration names (PIL's numbering).
        init(pilCode: Int) {
            self = pilCode == 3 ? .bicubic : .bilinear
        }
    }

    /// How the filter weights are quantized before the fixed-point sum.
    enum Precision: Sendable {
        /// PIL's: 22 fractional bits.
        case pil
        /// torchvision's 8-bit antialiased resize (`F.resize` on a `uint8` tensor): every weight of an
        /// axis held in an `int16`, at the most fractional bits that axis's largest weight allows.
        case torchvision
    }

    private static let precisionBits = 32 - 8 - 2

    /// Resamples the width first, then the height, skipping an axis whose size does not change, as PIL
    /// and torchvision's 8-bit resize both do.
    static func resampled(_ pixels: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int,
                          filter: Filter = .bilinear, precision: Precision = .pil) -> [UInt8] {
        var image = pixels
        if toWidth != width {
            image = pass(image, rows: height, columns: width, toColumns: toWidth, filter: filter, precision: precision)
        }
        guard toHeight != height else { return image }
        let columns = toWidth
        var transposed = [UInt8](repeating: 0, count: columns * height * 3)
        for row in 0 ..< height {
            for column in 0 ..< columns {
                for channel in 0 ..< 3 {
                    transposed[(column * height + row) * 3 + channel] = image[(row * columns + column) * 3 + channel]
                }
            }
        }
        let scaled = pass(transposed, rows: columns, columns: height, toColumns: toHeight, filter: filter,
                          precision: precision)
        var result = [UInt8](repeating: 0, count: columns * toHeight * 3)
        for column in 0 ..< columns {
            for row in 0 ..< toHeight {
                for channel in 0 ..< 3 {
                    result[(row * columns + column) * 3 + channel] = scaled[(column * toHeight + row) * 3 + channel]
                }
            }
        }
        return result
    }

    private static func pass(_ pixels: [UInt8], rows: Int, columns: Int, toColumns: Int, filter: Filter,
                             precision: Precision) -> [UInt8] {
        let (weights, bits) = quantizedCoefficients(from: columns, to: toColumns, filter: filter, precision: precision)
        let half = 1 << (bits - 1)
        var result = [UInt8](repeating: 0, count: rows * toColumns * 3)
        for row in 0 ..< rows {
            for column in 0 ..< toColumns {
                let span = weights[column]
                var sums = (half, half, half)
                for (offset, weight) in span.weights.enumerated() {
                    let source = (row * columns + span.start + offset) * 3
                    sums.0 += Int(pixels[source]) * weight
                    sums.1 += Int(pixels[source + 1]) * weight
                    sums.2 += Int(pixels[source + 2]) * weight
                }
                let destination = (row * toColumns + column) * 3
                result[destination] = clipped(sums.0, bits: bits)
                result[destination + 1] = clipped(sums.1, bits: bits)
                result[destination + 2] = clipped(sums.2, bits: bits)
            }
        }
        return result
    }

    private static func clipped(_ accumulated: Int, bits: Int) -> UInt8 {
        UInt8(Swift.max(0, Swift.min(255, accumulated >> bits)))
    }

    static func coefficients(from inSize: Int, to outSize: Int,
                             filter: Filter = .bilinear) -> [(start: Int, weights: [Int])] {
        quantizedCoefficients(from: inSize, to: outSize, filter: filter, precision: .pil).spans
    }

    /// The normalized filter weights of each output sample, quantized as `precision` does, with the
    /// number of fractional bits they carry.
    static func quantizedCoefficients(from inSize: Int, to outSize: Int, filter: Filter,
                                      precision: Precision) -> (spans: [(start: Int, weights: [Int])], bits: Int) {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = Swift.max(scale, 1)
        let support = filter.support * filterScale
        let normalized: [(Int, [Double])] = (0 ..< outSize).map { index in
            let center = scale * (Double(index) + 0.5)
            let start = Swift.max(Int(center - support + 0.5), 0)
            let end = Swift.min(Int(center + support + 0.5), inSize)
            var raw = [Double]()
            var total = 0.0
            for source in start ..< end {
                let weight = filter.weight((Double(source) - center + 0.5) / filterScale)
                raw.append(weight)
                total += weight
            }
            return (start, raw.map { total != 0 ? $0 / total : $0 })
        }
        var bits = precisionBits
        if precision == .torchvision {
            let largest = normalized.flatMap(\.1).max() ?? 0
            bits = 0
            while bits < 22, Int(0.5 + largest * Double(1 << (bits + 1))) < 1 << 15 {
                bits += 1
            }
        }
        let spans = normalized.map { start, weights in
            (start, weights.map { weight -> Int in
                let scaled = weight * Double(1 << bits)
                return Int(scaled < 0 ? scaled - 0.5 : scaled + 0.5)
            })
        }
        return (spans, bits)
    }
}
