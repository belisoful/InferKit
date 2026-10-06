//
//  NFKMLXMatteOperations.swift
//  InferKitMLX
//

import Foundation
import MLX

/// Operations on a matte and the images around it, for any matting model's output. Introduced in
/// InferKit 0.4.0.
///
/// @discussion A matte is `[H, W, 1]` in 0...1 and an image `[H, W, C]`. The compositing operations
/// expect linear light. Every operation runs in MLX on the whole image; none tiles.
public enum NFKMLXMatteOperations {

    // MARK: Levels

    /// The matte remapped so `black` and below become 0 and `white` and above become 1, linearly between.
    /// Pixels where `protected` is true keep their value.
    public static func clipped(_ alpha: MLXArray, black: Float, white: Float, protected: MLXArray? = nil) -> MLXArray {
        let span = max(white - black, 1e-6)
        let remapped = clip((alpha - black) / span, min: 0, max: 1)
        guard let protected else { return remapped }
        return which(protected, alpha, remapped)
    }

    /// The matte raised to `1 / gamma`: a gamma above 1 lifts the partial values, below 1 lowers them.
    public static func gammaAdjusted(_ alpha: MLXArray, gamma: Float) -> MLXArray {
        pow(clip(alpha, min: 0, max: 1), 1 / max(gamma, 1e-6))
    }

    /// True `[H, W, 1]` where the image `[H, W, C]` changes by more than `tolerance` in some channel within
    /// the `(2 · radius + 1)²` window around a pixel: the edges a clip leaves alone.
    public static func edges(of image: MLXArray, radius: Int, tolerance: Float) -> MLXArray {
        let high = windowExtreme(windowExtreme(image, radius: radius, axis: 0, largest: true),
                                 radius: radius, axis: 1, largest: true)
        let low = windowExtreme(windowExtreme(image, radius: radius, axis: 0, largest: false),
                                radius: radius, axis: 1, largest: false)
        return ((high - low) .> tolerance).any(axis: -1, keepDims: true)
    }

    /// The matte with the garbage matte's region forced toward 0 and the core matte's toward 1:
    /// `max(min(alpha, 1 − garbage), core)`.
    public static func masked(_ alpha: MLXArray, garbage: MLXArray?, core: MLXArray?) -> MLXArray {
        var result = alpha
        if let garbage {
            result = minimum(result, 1 - garbage)
        }
        if let core {
            result = maximum(result, core)
        }
        return result
    }

    // MARK: Shape

    /// The matte grown (positive `radius`) or shrunk (negative) by a disc of that radius in pixels, its
    /// partial values kept. A fractional radius blends the two neighboring whole radii.
    public static func morphed(_ alpha: MLXArray, radius: Float) -> MLXArray {
        guard radius != 0 else { return alpha }
        let magnitude = abs(radius)
        let (lower, upper) = (Int(magnitude.rounded(.down)), Int(magnitude.rounded(.up)))
        let grows = radius > 0
        let low = discExtreme(alpha, radius: lower, largest: grows)
        guard upper != lower else { return low }
        let fraction = magnitude - Float(lower)
        return low * (1 - fraction) + discExtreme(alpha, radius: upper, largest: grows) * fraction
    }

    /// The matte blurred by a Gaussian reaching `radius` pixels (sigma `radius / 3`), edges replicated.
    public static func feathered(_ alpha: MLXArray, radius: Float) -> MLXArray {
        gaussianBlurred(alpha, radius: radius)
    }

    /// The matte with the 8-connected islands above `threshold` that hold fewer than `minimumArea` pixels
    /// set to 0.
    public static func specksRemoved(_ alpha: MLXArray, threshold: Float = 0.5, minimumArea: Int) -> MLXArray {
        let small = smallComponents(alpha .> threshold, maximumArea: minimumArea - 1, keepingBorder: true)
        return which(small, MLXArray(Float(0)), alpha)
    }

    /// The matte with the 8-connected holes at or below `threshold` that hold at most `maximumArea` pixels
    /// and do not touch the image edge set to 1.
    public static func holesFilled(_ alpha: MLXArray, threshold: Float = 0.5, maximumArea: Int) -> MLXArray {
        let small = smallComponents(alpha .<= threshold, maximumArea: maximumArea, keepingBorder: false)
        return which(small, MLXArray(Float(1)), alpha)
    }

    // MARK: Refinement

    /// The matte refined by the guided filter (He, Sun, and Tang) with the image `[H, W, 3]` as its guide,
    /// which carries the image's edges into the matte. `radius` is in full-resolution pixels; the filter's
    /// coefficients are computed at `1 / subsampling` scale and resized back, the fast guided filter.
    public static func guidedFiltered(_ alpha: MLXArray, guide: MLXArray, radius: Int, epsilon: Float,
                                      subsampling: Int = 4) -> MLXArray {
        let (height, width) = (alpha.dim(0), alpha.dim(1))
        let scale = max(subsampling, 1)
        let (smallHeight, smallWidth) = (max(1, height / scale), max(1, width / scale))
        let image = linearResized(guide, height: smallHeight, width: smallWidth)
        let input = linearResized(alpha, height: smallHeight, width: smallWidth)
        let r = max(1, radius / scale)
        let meanImage = boxMean(image, radius: r)
        let meanInput = boxMean(input, radius: r)
        let covariance = boxMean(image * input, radius: r) - meanImage * meanInput
        func channel(_ x: MLXArray, _ index: Int) -> MLXArray { x[.ellipsis, index ..< index + 1] }
        func variance(_ i: Int, _ j: Int) -> MLXArray {
            boxMean(channel(image, i) * channel(image, j), radius: r) - channel(meanImage, i) * channel(meanImage, j)
        }
        let (a, b, c) = (variance(0, 0) + epsilon, variance(0, 1), variance(0, 2))
        let (d, e, f) = (variance(1, 1) + epsilon, variance(1, 2), variance(2, 2) + epsilon)
        let (c11, c12, c13) = (d * f - e * e, c * e - b * f, b * e - c * d)
        let (c22, c23, c33) = (a * f - c * c, b * c - a * e, a * d - b * b)
        let determinant = a * c11 + b * c12 + c * c13
        let (p, q, s) = (channel(covariance, 0), channel(covariance, 1), channel(covariance, 2))
        let coefficients = concatenated([(c11 * p + c12 * q + c13 * s) / determinant,
                                         (c12 * p + c22 * q + c23 * s) / determinant,
                                         (c13 * p + c23 * q + c33 * s) / determinant], axis: -1)
        let offset = meanInput - (coefficients * meanImage).sum(axis: -1, keepDims: true)
        let meanCoefficients = linearResized(boxMean(coefficients, radius: r), height: height, width: width)
        let meanOffset = linearResized(boxMean(offset, radius: r), height: height, width: width)
        return clip((meanCoefficients * guide).sum(axis: -1, keepDims: true) + meanOffset, min: 0, max: 1)
    }

    /// The foreground and background colors `[H, W, 3]` that composite to the image `[H, W, 3]` under the
    /// matte, by multi-level local estimation (Germer, Uelwer, Conrad, and Harmeling 2020): each level
    /// solves a per-pixel 2 × 2 system against its four neighbors, from 1 × 1 up to the full size, each
    /// level at most twice the last. `omega` weighs the matte's gradient and `epsilon` regularizes. Levels
    /// up to `smallSize` on their long side run `smallIterations` Jacobi updates, larger ones
    /// `largeIterations`.
    public static func estimatedColors(image: MLXArray, alpha: MLXArray, omega: Float = 0.1, epsilon: Float = 5e-3,
                                       smallIterations: Int = 10, largeIterations: Int = 2,
                                       smallSize: Int = 32) -> (foreground: MLXArray, background: MLXArray) {
        let (height, width) = (image.dim(0), image.dim(1))
        let levels = max(1, Int(ceil(log2(Double(max(height, width))))))
        var foreground = image.mean(axes: [0, 1], keepDims: true)
        var background = foreground
        for level in 0 ... levels {
            let exponent = Double(level) / Double(levels)
            let levelHeight = max(1, Int(pow(Double(height), exponent).rounded()))
            let levelWidth = max(1, Int(pow(Double(width), exponent).rounded()))
            let levelImage = linearResized(image, height: levelHeight, width: levelWidth)
            let levelAlpha = linearResized(alpha, height: levelHeight, width: levelWidth)
            foreground = linearResized(foreground, height: levelHeight, width: levelWidth)
            background = linearResized(background, height: levelHeight, width: levelWidth)
            let iterations = max(levelHeight, levelWidth) <= smallSize ? smallIterations : largeIterations
            for _ in 0 ..< iterations {
                (foreground, background) = colorUpdate(image: levelImage, alpha: levelAlpha, foreground: foreground,
                                                       background: background, omega: omega, epsilon: epsilon)
            }
            eval(foreground, background)
        }
        return (foreground, background)
    }

    /// One Jacobi update of the multi-level estimate at one level.
    static func colorUpdate(image: MLXArray, alpha: MLXArray, foreground: MLXArray, background: MLXArray,
                            omega: Float, epsilon: Float) -> (MLXArray, MLXArray) {
        var weightSum = MLXArray.zeros(like: alpha)
        var foregroundSum = MLXArray.zeros(like: foreground)
        var backgroundSum = MLXArray.zeros(like: background)
        for (axis, step) in [(0, -1), (0, 1), (1, -1), (1, 1)] {
            let neighborAlpha = shiftedReplicating(alpha, axis: axis, step: step)
            let weight = epsilon + omega * abs(alpha - neighborAlpha)
            weightSum = weightSum + weight
            foregroundSum = foregroundSum + weight * shiftedReplicating(foreground, axis: axis, step: step)
            backgroundSum = backgroundSum + weight * shiftedReplicating(background, axis: axis, step: step)
        }
        let a00 = alpha * alpha + weightSum
        let a01 = alpha * (1 - alpha)
        let a11 = (1 - alpha) * (1 - alpha) + weightSum
        let rightForeground = alpha * image + foregroundSum
        let rightBackground = (1 - alpha) * image + backgroundSum
        let determinant = a00 * a11 - a01 * a01
        return (clip((a11 * rightForeground - a01 * rightBackground) / determinant, min: 0, max: 1),
                clip((a00 * rightBackground - a01 * rightForeground) / determinant, min: 0, max: 1))
    }

    // MARK: Compositing

    /// The straight foreground over the background, both linear `[H, W, 3]`, with the background's blur
    /// wrapped onto the subject's edge: `C · (1 − w) + blur(B) · w`, where
    /// `w = strength · alpha · blur(1 − alpha)` and both blurs reach `radius` pixels.
    public static func lightWrapped(foreground: MLXArray, alpha: MLXArray, background: MLXArray, radius: Float,
                                    strength: Float) -> MLXArray {
        let composite = foreground * alpha + background * (1 - alpha)
        let wrap = clip(gaussianBlurred(1 - alpha, radius: radius) * alpha * strength, min: 0, max: 1)
        return composite * (1 - wrap) + gaussianBlurred(background, radius: radius) * wrap
    }

    /// The straight foreground over the background with the plate's departure from the screen added to
    /// the background where the matte lets it through: `F · alpha + max(B + gain · R, 0) · (1 − alpha)`.
    /// `R` is the plate less the screen color (`[3]` or a per-pixel `[H, W, 3]`), its saturation scaled by
    /// `saturation` around its Rec. 709 luma. Fine detail the matte leaves partial keeps its light.
    public static func additiveKeyed(foreground: MLXArray, alpha: MLXArray, background: MLXArray, plate: MLXArray,
                                     screenColor: MLXArray, saturation: Float = 0, gain: Float = 1) -> MLXArray {
        let residual = plate - screenColor
        let weights = MLXArray([Float(0.2126), 0.7152, 0.0722])
        let luma = (residual * weights).sum(axis: -1, keepDims: true)
        let graded = (luma + (residual - luma) * saturation) * gain
        return foreground * alpha + maximum(background + graded, 0) * (1 - alpha)
    }

    /// The foreground with the plate's own pixels where the matte is solid: the plate's share rises from 0
    /// at `threshold` to 1 at an alpha of 1.
    public static func sourcePassedThrough(foreground: MLXArray, plate: MLXArray, alpha: MLXArray,
                                           threshold: Float = 0.95) -> MLXArray {
        let share = clip((alpha - threshold) / max(1 - threshold, 1e-6), min: 0, max: 1)
        return foreground * (1 - share) + plate * share
    }

    // MARK: Time

    /// The matte blended toward the previous frame's by `weight` where the image barely moved: where the
    /// largest channel difference from the previous image is below `motionThreshold`. Fed its own output,
    /// it is a moving average that stops at motion.
    public static func temporallyBlended(_ alpha: MLXArray, previous: MLXArray, image: MLXArray, previousImage: MLXArray,
                                         weight: Float, motionThreshold: Float = 0.05) -> MLXArray {
        let still = abs(image - previousImage).max(axis: -1, keepDims: true) .< motionThreshold
        return which(still, alpha * (1 - weight) + previous * weight, alpha)
    }

    // MARK: Building blocks

    /// The largest (or smallest) value over each `[i − radius, i + radius]` window along `axis`, pixels
    /// outside the image excluded. Doubling windows give `O(log radius)` passes.
    static func windowExtreme(_ x: MLXArray, radius: Int, axis: Int, largest: Bool) -> MLXArray {
        guard radius > 0 else { return x }
        let moved = x.swappedAxes(0, axis)
        let count = moved.dim(0)
        let fill = MLXArray(largest ? -Float.infinity : Float.infinity)
        var widths = [IntOrPair](repeating: 0, count: moved.ndim)
        widths[0] = .init((radius, radius))
        var table = MLX.padded(moved, widths: widths, value: fill)
        let window = 2 * radius + 1
        var span = 1
        while span * 2 <= window {
            let length = table.dim(0)
            let shifted = table[span ..< length]
            table = largest ? maximum(table[0 ..< length - span], shifted) : minimum(table[0 ..< length - span], shifted)
            span *= 2
        }
        let head = table[0 ..< count]
        let tail = table[(window - span) ..< (window - span + count)]
        return (largest ? maximum(head, tail) : minimum(head, tail)).swappedAxes(0, axis)
    }

    /// The largest (or smallest) value over a disc of `radius` around each pixel of `[H, W, C]`, pixels
    /// outside the image excluded: one horizontal window per disc row, shifted into place.
    static func discExtreme(_ x: MLXArray, radius: Int, largest: Bool) -> MLXArray {
        guard radius > 0 else { return x }
        let height = x.dim(0)
        let fill = MLXArray(largest ? -Float.infinity : Float.infinity)
        let halfWidths = (-radius ... radius).map { dy in
            Int((Double(radius * radius - dy * dy)).squareRoot() + 1e-9)
        }
        var rows: [Int: MLXArray] = [:]
        for halfWidth in Set(halfWidths) {
            var widths = [IntOrPair](repeating: 0, count: x.ndim)
            widths[0] = .init((radius, radius))
            rows[halfWidth] = MLX.padded(windowExtreme(x, radius: halfWidth, axis: 1, largest: largest),
                                         widths: widths, value: fill)
        }
        var result = x
        for (row, halfWidth) in halfWidths.enumerated() {
            let shifted = rows[halfWidth]![row ..< row + height]
            result = largest ? maximum(result, shifted) : minimum(result, shifted)
        }
        return result
    }

    /// A Gaussian blur of `[H, W, C]` with sigma `radius / 3` and taps out to `radius`, edges replicated.
    static func gaussianBlurred(_ x: MLXArray, radius: Float) -> MLXArray {
        let reach = Int(radius.rounded(.up))
        guard reach > 0 else { return x }
        let sigma = Double(radius) / 3
        let taps = (-reach ... reach).map { exp(-Double($0 * $0) / (2 * sigma * sigma)) }
        let total = taps.reduce(0, +)
        let kernel = taps.map { Float($0 / total) }
        func pass(_ x: MLXArray, axis: Int) -> MLXArray {
            let count = x.dim(axis)
            var sum = MLXArray.zeros(like: x)
            for (index, weight) in kernel.enumerated() {
                let sources = (0 ..< count).map { Int32(min(max($0 + index - reach, 0), count - 1)) }
                sum = sum + x.take(MLXArray(sources), axis: axis) * weight
            }
            return sum
        }
        return pass(pass(x, axis: 0), axis: 1)
    }

    /// The mean over each `(2 · radius + 1)²` window of `[H, W, C]`, the window clipped at the edges.
    static func boxMean(_ x: MLXArray, radius: Int) -> MLXArray {
        func pass(_ x: MLXArray, axis: Int) -> MLXArray {
            let count = x.dim(axis)
            let moved = x.swappedAxes(0, axis)
            var widths = [IntOrPair](repeating: 0, count: moved.ndim)
            widths[0] = .init((radius, radius))
            let padded = MLX.padded(moved, widths: widths)
            var sum = padded[0 ..< count]
            if radius > 0 {
                for offset in 1 ... 2 * radius {
                    sum = sum + padded[offset ..< offset + count]
                }
            }
            let counts = (0 ..< count).map { Float(min($0 + radius, count - 1) - max($0 - radius, 0) + 1) }
            var shape = [Int](repeating: 1, count: moved.ndim)
            shape[0] = count
            return (sum / MLXArray(counts).reshaped(shape)).swappedAxes(0, axis)
        }
        return pass(pass(x, axis: 0), axis: 1)
    }

    /// `[H, W, C]` resized by linear interpolation with half-pixel centers, edge samples replicated.
    static func linearResized(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        func pass(_ x: MLXArray, axis: Int, count: Int) -> MLXArray {
            let source = x.dim(axis)
            guard source != count else { return x }
            var lower = [Int32](), upper = [Int32](), fractions = [Float]()
            for index in 0 ..< count {
                let position = min(max((Double(index) + 0.5) * Double(source) / Double(count) - 0.5, 0), Double(source - 1))
                let base = Int(position.rounded(.down))
                lower.append(Int32(base))
                upper.append(Int32(min(base + 1, source - 1)))
                fractions.append(Float(position - Double(base)))
            }
            var shape = [Int](repeating: 1, count: x.ndim)
            shape[axis] = count
            let t = MLXArray(fractions).reshaped(shape)
            return x.take(MLXArray(lower), axis: axis) * (1 - t) + x.take(MLXArray(upper), axis: axis) * t
        }
        return pass(pass(x, axis: 0, count: height), axis: 1, count: width)
    }

    /// `[H, W, C]` shifted one pixel along `axis` (`step` −1 reads the previous pixel, +1 the next), the
    /// edge replicated.
    static func shiftedReplicating(_ x: MLXArray, axis: Int, step: Int) -> MLXArray {
        let count = x.dim(axis)
        let sources = (0 ..< count).map { Int32(min(max($0 + step, 0), count - 1)) }
        return x.take(MLXArray(sources), axis: axis)
    }

    /// True `[H, W, 1]` on the 8-connected components of `mask` `[H, W, 1]` that hold at most `maximumArea`
    /// pixels; with `keepingBorder` false, components that touch the image edge are never small.
    static func smallComponents(_ mask: MLXArray, maximumArea: Int, keepingBorder: Bool) -> MLXArray {
        let (height, width) = (mask.dim(0), mask.dim(1))
        let count = height * width
        let inside = mask.reshaped([height, width])
        let labels = componentLabels(inside).reshaped([-1])
        let flat = inside.reshaped([-1])
        let owners = which(flat, labels, MLXArray(Int32(0)))
        let areas = MLXArray.zeros([count], dtype: .int32).at[owners].add(flat.asType(.int32))
        var small = flat .&& (areas.take(owners) .<= Int32(maximumArea))
        if !keepingBorder {
            let rows = MLXArray(Int32(0) ..< Int32(height)).reshaped([height, 1])
            let columns = MLXArray(Int32(0) ..< Int32(width)).reshaped([1, width])
            let border = broadcast((rows .== Int32(0)) .|| (rows .== Int32(height - 1)) .|| (columns .== Int32(0))
                                   .|| (columns .== Int32(width - 1)), to: [height, width]).reshaped([-1])
            let touching = MLXArray.zeros([count], dtype: .int32).at[owners].add((flat .&& border).asType(.int32))
            small = small .&& (touching.take(owners) .== Int32(0))
        }
        return small.reshaped([height, width, 1])
    }

    /// Each pixel of `mask` `[H, W]` labeled by the smallest row-order index in its 8-connected component
    /// (`Int32.max` outside): neighbor minima and label jumps repeated until no label changes.
    static func componentLabels(_ mask: MLXArray) -> MLXArray {
        let (height, width) = (mask.dim(0), mask.dim(1))
        let count = height * width
        let outside = MLXArray(Int32.max)
        var labels = which(mask, MLXArray(Int32(0) ..< Int32(count)).reshaped([height, width]), outside)
        while true {
            let previous = labels
            for _ in 0 ..< 8 {
                let padded = MLX.padded(labels, widths: [.init((1, 1)), .init((1, 1))], value: outside)
                var smallest = labels
                for dy in 0 ... 2 {
                    for dx in 0 ... 2 where dy != 1 || dx != 1 {
                        smallest = minimum(smallest, padded[dy ..< dy + height, dx ..< dx + width])
                    }
                }
                labels = which(mask, smallest, outside)
                // A label names a pixel of the same component whose own label is no larger.
                let followed = labels.reshaped([-1]).take(minimum(labels, Int32(count - 1)).reshaped([-1]))
                labels = which(mask, followed.reshaped([height, width]), outside)
            }
            eval(labels)
            if !(labels .!= previous).any().item(Bool.self) {
                return labels
            }
        }
    }
}
