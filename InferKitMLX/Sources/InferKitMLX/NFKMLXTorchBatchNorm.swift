//
//  NFKMLXTorchBatchNorm.swift
//  InferKitMLX
//

import Foundation
import MLX
import MLXNN

/// PyTorch's `BatchNorm` in training mode: it normalizes with the batch's population variance, as
/// MLXNN's does, and folds the UNBIASED variance into the running variance, where MLXNN's folds the
/// population one.
///
/// The two differ by `n / (n − 1)` for `n` values per channel, which matters for a normalization after
/// a global pool: there `n` is the batch size, and a batch of two halves the statistics a fine-tuned
/// checkpoint carries. In evaluation mode it computes exactly what `BatchNorm` does, and it keeps the
/// type and the keys, so the trainer and the loaders treat it as one.
final class NFKTorchBatchNorm: BatchNorm {

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let statistics = Dictionary(uniqueKeysWithValues: parameters().flattened())
        guard training, let runningMean = statistics["running_mean"], let runningVar = statistics["running_var"] else {
            return super.callAsFunction(x)
        }
        let axes = Array(0 ..< x.ndim - 1)
        let mean = x.mean(axes: axes)
        let variance = x.variance(axes: axes)
        let count = Float(x.size / x.dim(-1))
        let unbiased = variance * (count / max(count - 1, 1))
        runningMean._updateInternal((1 - momentum) * runningMean + momentum * mean)
        runningVar._updateInternal((1 - momentum) * runningVar + momentum * unbiased)
        let normalized = (x - mean) * rsqrt(variance + eps)
        guard let weight, let bias else {
            return normalized
        }
        return weight * normalized + bias
    }
}
