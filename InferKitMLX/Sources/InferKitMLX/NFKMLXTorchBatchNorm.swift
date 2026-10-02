//
//  NFKMLXTorchBatchNorm.swift
//  InferKitMLX
//

import Foundation
import MLX
import MLXNN

/// PyTorch's `BatchNorm` in training mode, with its batch statistics staged (`NFKMLXStagedReduction`).
///
/// It normalizes with the batch's population variance, as MLXNN's does, and folds the UNBIASED variance
/// into the running variance, where MLXNN's folds the population one. The two differ by `n / (n − 1)` for
/// `n` values per channel, which matters where `n` is small: after a global pool `n` is the batch size,
/// and a batch of two halves the statistics a fine-tuned checkpoint carries.
///
/// The statistics are summed one axis at a time, the mean is refined by the mean of the deviations from
/// it, and the variance subtracts that refinement's square, the corrected two-pass form. A channel whose
/// values vary far less than their mean keeps its variance's digits that way, where MLX's own reductions
/// lose them. In evaluation mode it computes exactly what `BatchNorm` does, and it keeps the type and the
/// keys, so the trainer and the loaders treat it as one.
final class NFKTorchBatchNorm: BatchNorm {

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard training else {
            return super.callAsFunction(x)
        }
        let statistics = NFKMLXStagedReduction.normalized(x, eps: eps)
        let stored = Dictionary(uniqueKeysWithValues: parameters().flattened())
        if let runningMean = stored["running_mean"], let runningVar = stored["running_var"] {
            let count = Float(x.size / x.dim(-1))
            let unbiased = statistics.variance * (count / max(count - 1, 1))
            runningMean._updateInternal((1 - momentum) * runningMean + momentum * statistics.mean)
            runningVar._updateInternal((1 - momentum) * runningVar + momentum * unbiased)
        }
        guard let weight, let bias else {
            return statistics.normalized
        }
        return NFKMLXStagedReduction.affine(statistics.normalized, weight: weight, bias: bias)
    }
}
