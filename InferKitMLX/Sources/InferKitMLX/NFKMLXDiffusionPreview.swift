//
//  NFKMLXDiffusionPreview.swift
//  InferKitMLX
//
//  A cheap picture of a latent, for the progress callback of a diffusion run.
//
//  A sampler takes tens of seconds. The loop reports a step count, which says nothing about what is
//  being made, and decoding the latent through the autoencoder every step costs more than the
//  sampling does. A latent is close enough to a linear function of the image it decodes to that one
//  1×1 convolution over the channel axis reproduces it recognizably. That convolution is twelve
//  weights and three biases for a four-channel latent, so a preview costs nothing measurable.
//
//  Introduced in InferKit 0.1.0.
//

import Foundation
import MLX

/// Maps a latent to an RGB image by a per-channel linear combination, for a live preview.
///
/// @discussion The map is a 1×1 convolution: each output channel is a weighted sum of the latent's
/// channels plus a bias. It approximates the autoencoder's decode at the latent's own resolution —
/// an eighth of the image for Stable Diffusion — so a preview is a small, blurry, correctly colored
/// version of what the run is producing, at a cost that does not scale with the step count.
///
/// The shipped coefficients are an approximation and make no parity claim.
/// ``fitted(latentChannels:decode:sample:count:)`` derives the map from a model's own decoder
/// instead, which is the measured alternative and the way to get a preview for a model with no
/// published factors.
public struct NFKDiffusionLatentPreview: Sendable, Equatable {

    /// The per-channel weights, `latentChannels` rows of three, row-major: `weights[c * 3 + k]` is
    /// channel `c`'s contribution to output channel `k`.
    public let weights: [Float]

    /// The three output biases, added after the weighted sum.
    public let biases: [Float]

    /// The number of latent channels the map expects.
    public var latentChannels: Int { weights.count / 3 }

    /// - Parameters:
    ///   - weights: `latentChannels × 3` values, row-major by latent channel.
    ///   - biases: three values, added after the weighted sum.
    public init(weights: [Float], biases: [Float]) {
        precondition(weights.count % 3 == 0, "weights are three per latent channel")
        precondition(biases.count == 3, "one bias per output channel")
        self.weights = weights
        self.biases = biases
    }

    /// Maps a latent `[H, W, latentChannels]` to an image `[H, W, 3]` clamped to `0...1`.
    ///
    /// - Returns: the preview, or `nil` when the latent's channel count does not match the map's.
    public func image(from latent: MLXArray) -> MLXArray? {
        guard latent.ndim == 3, latent.shape[2] == latentChannels else { return nil }
        let (height, width) = (latent.shape[0], latent.shape[1])
        let matrix = MLXArray(weights, [latentChannels, 3])
        let flat = latent.reshaped([height * width, latentChannels])
        let rgb = matmul(flat, matrix) + MLXArray(biases, [1, 3])
        return clip(rgb.reshaped([height, width, 3]), min: 0, max: 1)
    }

    /// A latent already in image space: three channels in `-1...1`, which is what the reference
    /// stand-in pipelines here produce.
    public static let passthrough = NFKDiffusionLatentPreview(
        weights: [0.5, 0, 0,
                  0, 0.5, 0,
                  0, 0, 0.5],
        biases: [0.5, 0.5, 0.5])

    /// The four-channel Stable Diffusion 1.x / 2.x latent.
    ///
    /// @discussion These are the latent-RGB factors in common use for this autoencoder. Measured
    /// against the released SD 1.5 decode on a real photograph's latent, they reproduce its structure
    /// at a mean-removed correlation of **0.93**
    /// (`NFKMLXDiffusionPreviewTests.testTheShippedStableDiffusionMapTracksARealDecode`).
    ///
    /// The map is applied to the SAMPLER's latent, which is the scaled one — the pipeline divides by
    /// `scaleFactor` before the autoencoder sees it. Handing these coefficients a latent at the
    /// autoencoder's own scale washes the preview toward flat grey.
    public static let stableDiffusion = NFKDiffusionLatentPreview(
        weights: [0.298, 0.207, 0.208,
                  0.187, 0.286, 0.173,
                  -0.158, 0.189, 0.264,
                  -0.184, -0.271, -0.473],
        biases: [0.5, 0.5, 0.5])

    /// The four-channel SDXL latent, whose autoencoder was retrained and does not share 1.x's factors.
    public static let stableDiffusionXL = NFKDiffusionLatentPreview(
        weights: [0.3651, 0.4232, 0.4341,
                  -0.2533, -0.0042, 0.1068,
                  0.1076, 0.1111, 0.0652,
                  -0.3165, -0.2492, -0.2188],
        biases: [0.5, 0.5, 0.5])

    // MARK: Deriving a map from a decoder

    /// Fits the map to a model's own decoder by least squares, so a preview is derived rather than
    /// assumed.
    ///
    /// @discussion The decode is run on sampled latents, its output is averaged down to the latent's
    /// resolution, and the per-channel linear map that best reproduces it is solved for. A handful of
    /// samples is enough: the map has only `3 × (latentChannels + 1)` unknowns.
    ///
    /// - Parameters:
    ///   - latentChannels: the latent's channel count.
    ///   - decode: the model's decoder, latent `[H, W, C]` → image `[H·s, W·s, 3]` in `0...1`.
    ///   - sample: supplies latent `i`, in the SAMPLER's scale rather than the autoencoder's. The
    ///     latents should look like the ones the run will preview. Unit-variance noise is a workable
    ///     stand-in, but it is the least structured case: measured against SD 1.5, a map fitted on
    ///     noise scores 0.917 on a photograph's latent where the shipped coefficients score 0.931.
    ///     Fitting on latents encoded from real images gives a map that generalizes better.
    ///   - count: how many latents to fit over.
    ///
    /// - Returns: the fitted map, or `nil` when the decode produced nothing usable.
    public static func fitted(latentChannels: Int,
                              decode: (MLXArray) -> MLXArray,
                              sample: (Int) -> MLXArray,
                              count: Int = 4) -> NFKDiffusionLatentPreview? {
        // Normal equations over the latent's channels plus a constant column, which is the bias.
        let columns = latentChannels + 1
        var gram = [Double](repeating: 0, count: columns * columns)
        var moment = [Double](repeating: 0, count: columns * 3)
        var rows = 0

        for index in 0 ..< max(count, 1) {
            let latent = sample(index)
            guard latent.ndim == 3, latent.shape[2] == latentChannels else { continue }
            let decoded = decode(latent)
            guard decoded.ndim == 3, decoded.shape[2] == 3 else { continue }
            let scale = decoded.shape[0] / latent.shape[0]
            guard scale >= 1, decoded.shape[1] / latent.shape[1] == scale else { continue }

            // Compare at the latent's resolution: the map cannot represent detail finer than that.
            let pooled = scale == 1 ? decoded
                : NFKMLXResample.averagePooled(decoded.expandedDimensions(axis: 0),
                                               kernel: .init(scale), stride: .init(scale))
                    .squeezed(axis: 0)
            eval(latent, pooled)
            let latentValues = latent.asType(.float32).asArray(Float.self)
            let targetValues = pooled.asType(.float32).asArray(Float.self)
            let pixels = latent.shape[0] * latent.shape[1]
            rows += pixels

            for pixel in 0 ..< pixels {
                var row = [Double](repeating: 1, count: columns)
                for channel in 0 ..< latentChannels {
                    row[channel] = Double(latentValues[pixel * latentChannels + channel])
                }
                for i in 0 ..< columns {
                    for j in 0 ..< columns {
                        gram[i * columns + j] += row[i] * row[j]
                    }
                    for k in 0 ..< 3 {
                        moment[i * 3 + k] += row[i] * Double(targetValues[pixel * 3 + k])
                    }
                }
            }
        }
        guard rows > columns, let solution = solve(gram, moment, size: columns, columns: 3) else {
            return nil
        }

        var weights = [Float](repeating: 0, count: latentChannels * 3)
        for channel in 0 ..< latentChannels {
            for k in 0 ..< 3 {
                weights[channel * 3 + k] = Float(solution[channel * 3 + k])
            }
        }
        let biases = (0 ..< 3).map { Float(solution[latentChannels * 3 + $0]) }
        return NFKDiffusionLatentPreview(weights: weights, biases: biases)
    }

    /// Gauss-Jordan with partial pivoting over a small symmetric system — the fit has at most five
    /// unknowns, so this needs no linear-algebra dependency.
    private static func solve(_ matrix: [Double], _ rightHand: [Double],
                              size: Int, columns: Int) -> [Double]? {
        var a = matrix
        var b = rightHand
        for pivot in 0 ..< size {
            var best = pivot
            for row in (pivot + 1) ..< size where abs(a[row * size + pivot]) > abs(a[best * size + pivot]) {
                best = row
            }
            guard abs(a[best * size + pivot]) > 1e-12 else { return nil }
            if best != pivot {
                for column in 0 ..< size { a.swapAt(pivot * size + column, best * size + column) }
                for column in 0 ..< columns { b.swapAt(pivot * columns + column, best * columns + column) }
            }
            let diagonal = a[pivot * size + pivot]
            for column in 0 ..< size { a[pivot * size + column] /= diagonal }
            for column in 0 ..< columns { b[pivot * columns + column] /= diagonal }
            for row in 0 ..< size where row != pivot {
                let factor = a[row * size + pivot]
                guard factor != 0 else { continue }
                for column in 0 ..< size { a[row * size + column] -= factor * a[pivot * size + column] }
                for column in 0 ..< columns {
                    b[row * columns + column] -= factor * b[pivot * columns + column]
                }
            }
        }
        return b
    }
}
