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

/// `BatchNorm` whose training statistics are staged reductions (`NFKMLXStagedReduction`).
///
/// The mean is refined by the mean of the deviations from it, and the variance subtracts that
/// refinement's square, the corrected two-pass form, so a channel whose values vary far less than their
/// mean keeps its variance's digits. In evaluation mode it computes exactly what `BatchNorm` does. The running statistics fold the population variance as `BatchNorm`'s do, and the
/// type and keys are `BatchNorm`'s, so the trainer and the loaders treat it as one.
final class NFKStagedBatchNorm: BatchNorm {

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard training else {
            return super.callAsFunction(x)
        }
        let axes = Array(0 ..< x.ndim - 1)
        let count = Float(x.size / x.dim(-1))
        let estimate = NFKMLXStagedReduction.sum(x, axes: axes) / count
        let deviation = x - NFKMLXStagedReduction.broadcast(estimate, like: x)
        let shift = NFKMLXStagedReduction.sum(deviation, axes: axes) / count
        let variance = NFKMLXStagedReduction.sum(square(deviation), axes: axes) / count - square(shift)
        let statistics = Dictionary(uniqueKeysWithValues: parameters().flattened())
        if let runningMean = statistics["running_mean"], let runningVar = statistics["running_var"] {
            runningMean._updateInternal((1 - momentum) * runningMean + momentum * (estimate + shift).flattened())
            runningVar._updateInternal((1 - momentum) * runningVar + momentum * variance.flattened())
        }
        let scale = rsqrt(variance + eps)
        let normalized = (deviation - NFKMLXStagedReduction.broadcast(shift, like: x)) * NFKMLXStagedReduction.broadcast(scale, like: x)
        guard let weight, let bias else {
            return normalized
        }
        let shape = Array(repeating: 1, count: x.ndim - 1) + [x.dim(-1)]
        return NFKMLXStagedReduction.broadcast(weight.reshaped(shape), like: x) * normalized
            + NFKMLXStagedReduction.broadcast(bias.reshaped(shape), like: x)
    }
}
