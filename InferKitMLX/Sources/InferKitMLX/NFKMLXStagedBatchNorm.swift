//
//  NFKMLXStagedBatchNorm.swift
//  InferKitMLX
//

import Foundation
import MLX
import MLXNN

/// Reductions over leading axes taken one axis at a time.
///
/// MLX's CPU backend adds the terms of a reduction over leading axes in order into one float32
/// accumulator, so a sum over `B·H·W` values loses digits in proportion to that count. Summing one axis
/// per stage bounds each run at that axis's length. A broadcast taken one axis at a time stages the
/// backward's reductions, which are the broadcasts' gradients, the same way. MLX's GPU reductions over
/// these layouts lose digits too: on FRCRN's released weights the staged form moves the first UNet's GPU
/// gradients from 4.3% to 0.27% of float64.
enum NFKMLXStagedReduction {

    /// Sums `x` over `axes`, last axis first, keeping every dimension.
    static func sum(_ x: MLXArray, axes: [Int]) -> MLXArray {
        axes.sorted().reversed().reduce(x) { $0.sum(axis: $1, keepDims: true) }
    }

    /// The mean over `axes`, without them.
    static func mean(_ x: MLXArray, axes: [Int]) -> MLXArray {
        let count = axes.reduce(1) { $0 * x.dim($1) }
        return sum(x, axes: axes).squeezed(axes: axes) / Float(count)
    }

    /// The batch statistics of `x` over every axis but the last, and `x` normalized by them without an
    /// affine, in the corrected two-pass form: the mean is refined by the mean of the deviations from it,
    /// and the variance subtracts that refinement's square. A channel whose values vary far less than
    /// their mean keeps its variance's digits that way. `mean` and `variance` are `[C]`.
    static func normalized(_ x: MLXArray, eps: Float) -> (normalized: MLXArray, mean: MLXArray, variance: MLXArray) {
        let axes = Array(0 ..< x.ndim - 1)
        let count = Float(x.size / x.dim(-1))
        let estimate = sum(x, axes: axes) / count
        let deviation = x - broadcast(estimate, like: x)
        let shift = sum(deviation, axes: axes) / count
        let variance = sum(square(deviation), axes: axes) / count - square(shift)
        let normalized = (deviation - broadcast(shift, like: x)) * broadcast(rsqrt(variance + eps), like: x)
        return (normalized, (estimate + shift).flattened(), variance.flattened())
    }

    /// `weight · x + bias` for per-channel `[C]` parameters, broadcast one axis at a time.
    static func affine(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let shape = Array(repeating: 1, count: x.ndim - 1) + [x.dim(-1)]
        return broadcast(weight.reshaped(shape), like: x) * x + broadcast(bias.reshaped(shape), like: x)
    }

    /// Broadcasts `statistic` (`[1, …, 1, C]`) across every axis of `x` but the last two, first axis first.
    /// The arithmetic that consumes the result broadcasts the remaining axis.
    static func broadcast(_ statistic: MLXArray, like x: MLXArray) -> MLXArray {
        var shape = statistic.shape
        var result = statistic
        for axis in 0 ..< max(x.ndim - 2, 0) {
            shape[axis] = x.dim(axis)
            result = MLX.broadcast(result, to: shape)
        }
        return result
    }
}
