//
//  NFKMLXZeroDCETraining.swift
//  InferKitMLX
//
//  Fine-tuning Zero-DCE, the first customization recipe.
//
//  Zero-DCE is zero-reference: the reference trains it with no ground truth, scoring the enhanced image
//  on four properties of its own rather than against a brightened target. That is what makes it the
//  natural first model to customize on a device — a consumer supplies their own dark photos and nothing
//  to annotate, which is the only kind of data an end user actually has.
//
//  The losses and their weights follow the reference training script (Guo et al., CVPR 2020) and match
//  it numerically: `NFKMLXReferenceParityTests.testZeroDCETrainingLossesMatchTheReference` scores the
//  same tensors the reference scored and agrees on all four to float precision. A wrong loss is
//  invisible in a fine-tune's output — a model trained against a subtly different objective still
//  produces plausible images — so the oracle is the only check that catches it.
//
//  Three of the four reproduce expressions in the reference that read like slips. They are marked
//  where they occur, and they are deliberate: the released weights were trained against them.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The four zero-reference losses Zero-DCE trains against.
///
/// Each takes a batch `[N, H, W, 3]` in `0...1` and returns a scalar. They are separate so that a
/// consumer can reweight them, drop one, or score a model without training it.
public enum NFKMLXZeroDCELoss {

    /// Drives the mean intensity of every local region toward `wellExposedLevel`.
    ///
    /// This is the loss a consumer personalizes: the level is the preferred brightness, and the
    /// reference's 0.6 is a choice rather than a constant of nature.
    public static func exposure(_ enhanced: MLXArray, wellExposedLevel: Float = 0.6,
                                region: Int = 16) -> MLXArray {
        (regionMeans(grayscale(enhanced), region: region) - wellExposedLevel).square().mean()
    }

    /// Penalizes a color cast, following the gray-world assumption that the three channel averages
    /// of a natural image are close.
    ///
    /// The reference squares each channel difference and then squares it again under the square root,
    /// so the sum is of 4th powers. That reads like a slip, and it is reproduced here deliberately:
    /// the released weights were trained against this expression.
    public static func colorConstancy(_ enhanced: MLXArray) -> MLXArray {
        let means = enhanced.mean(axes: [1, 2])                          // [N, 3]
        let red = means[0..., 0], green = means[0..., 1], blue = means[0..., 2]
        let redGreen = (red - green).square()
        let redBlue = (red - blue).square()
        let greenBlue = (blue - green).square()
        return sqrt(redGreen.square() + redBlue.square() + greenBlue.square()).mean()
    }

    /// Keeps neighboring curve parameters close, so the estimated illumination stays monotonic and
    /// free of banding. Takes the curve maps `[N, H, W, 24]`, not the image.
    ///
    /// The reference sums the squared differences over the channels as well, while dividing by a
    /// count that covers only the spatial extent. The result therefore carries a factor of the
    /// channel count on top of the leading 2, which is what its weight of 200 is calibrated against.
    public static func illuminationSmoothness(_ curveMaps: MLXArray) -> MLXArray {
        let (batch, height, width) = (curveMaps.shape[0], curveMaps.shape[1], curveMaps.shape[2])
        let vertical = neighborDifference(curveMaps, axis: 1).square().sum()
        let horizontal = neighborDifference(curveMaps, axis: 2).square().sum()
        let verticalCount = Float((height - 1) * width)
        let horizontalCount = Float(height * (width - 1))
        return 2 * (vertical / verticalCount + horizontal / horizontalCount) / Float(batch)
    }

    /// Preserves the contrast between neighboring regions, so brightening does not flatten the image.
    ///
    /// The reference convolves the pooled map with four one-sided difference kernels using zero
    /// padding. Squaring makes the interior pairs symmetric, so the four directions double-count
    /// there, but the border regions difference against the padding and are not redundant. Both
    /// effects are reproduced.
    public static func spatialConsistency(_ enhanced: MLXArray, original: MLXArray,
                                          region: Int = 4) -> MLXArray {
        let after = regionMeans(grayscale(enhanced), region: region)
        let before = regionMeans(grayscale(original), region: region)

        var total = MLXArray(Float(0))
        for direction in NeighborDirection.allCases {
            let differenceBefore = before - zeroPaddedNeighbor(before, direction)
            let differenceAfter = after - zeroPaddedNeighbor(after, direction)
            total = total + (differenceBefore - differenceAfter).square()
        }
        return total.mean()
    }

    // MARK: - Shared

    private static func grayscale(_ batch: MLXArray) -> MLXArray {
        batch.mean(axis: 3)                                              // [N, H, W]
    }

    /// Averages each `region × region` block of a plane `[N, H, W]`.
    ///
    /// The region shrinks to fit an image smaller than it, so a small input scores rather than
    /// producing an empty array.
    private static func regionMeans(_ plane: MLXArray, region: Int) -> MLXArray {
        let size = max(1, min(region, min(plane.shape[1], plane.shape[2])))
        let height = (plane.shape[1] / size) * size
        let width = (plane.shape[2] / size) * size
        let cropped = plane[0..., 0 ..< height, 0 ..< width]
        return cropped
            .reshaped([plane.shape[0], height / size, size, width / size, size])
            .mean(axes: [2, 4])
    }

    /// The difference between each element and its predecessor along `axis`, over a 4-D array.
    /// A length of one along that axis has no neighbors, so the difference is empty and contributes
    /// nothing.
    private static func neighborDifference(_ array: MLXArray, axis: Int) -> MLXArray {
        let length = array.shape[axis]
        guard length > 1 else {
            return MLXArray.zeros([1])
        }
        if axis == 1 {
            return array[0..., 1 ..< length, 0..., 0...] - array[0..., 0 ..< (length - 1), 0..., 0...]
        }
        return array[0..., 0..., 1 ..< length, 0...] - array[0..., 0..., 0 ..< (length - 1), 0...]
    }

    /// The four neighbors the reference's difference kernels reach.
    private enum NeighborDirection: CaseIterable {
        case left, right, up, down
    }

    /// The neighbor of every position in a plane `[N, H, W]`, with positions outside the plane
    /// reading as zero. This is what the reference's `padding: 1` convolution does at the border.
    private static func zeroPaddedNeighbor(_ plane: MLXArray, _ direction: NeighborDirection) -> MLXArray {
        let (height, width) = (plane.shape[1], plane.shape[2])
        let padded = MLX.padded(plane, widths: [0, 1, 1])
        switch direction {
        case .left:  return padded[0..., 1 ... height, 0 ..< width]
        case .right: return padded[0..., 1 ... height, 2 ... (width + 1)]
        case .up:    return padded[0..., 0 ..< height, 1 ... width]
        case .down:  return padded[0..., 2 ... (height + 1), 1 ... width]
        }
    }
}

/// The weighted combination of the four zero-reference losses that Zero-DCE trains against.
///
/// The defaults are the reference training script's. `wellExposedLevel` is the one a consumer-facing
/// app exposes: raising it produces a brighter model, lowering it a more restrained one.
public struct NFKMLXZeroDCEObjective: Sendable {

    /// The mean intensity every local region is driven toward.
    public var wellExposedLevel: Float

    /// Weight on the exposure loss.
    public var exposureWeight: Float

    /// Weight on the color-constancy loss.
    public var colorWeight: Float

    /// Weight on the illumination-smoothness loss, which dominates the others by design.
    public var smoothnessWeight: Float

    /// Weight on the spatial-consistency loss.
    public var spatialWeight: Float

    public init(wellExposedLevel: Float = 0.6, exposureWeight: Float = 10, colorWeight: Float = 5,
                smoothnessWeight: Float = 200, spatialWeight: Float = 1) {
        self.wellExposedLevel = wellExposedLevel
        self.exposureWeight = exposureWeight
        self.colorWeight = colorWeight
        self.smoothnessWeight = smoothnessWeight
        self.spatialWeight = spatialWeight
    }

    /// Scores `net` on a batch of unlabeled photos `[N, H, W, 3]` in `0...1`.
    public func callAsFunction(_ net: NFKMLXZeroDCENet, _ photos: MLXArray) -> MLXArray {
        let maps = net.curveMaps(photos)
        let enhanced = net.applyCurves(maps, to: photos)
        return exposureWeight * NFKMLXZeroDCELoss.exposure(enhanced, wellExposedLevel: wellExposedLevel)
            + colorWeight * NFKMLXZeroDCELoss.colorConstancy(enhanced)
            + smoothnessWeight * NFKMLXZeroDCELoss.illuminationSmoothness(maps)
            + spatialWeight * NFKMLXZeroDCELoss.spatialConsistency(enhanced, original: photos)
    }
}

extension NFKMLXZeroDCE {

    /// Fine-tunes a curve estimator on unlabeled photos, returning the loss from each step.
    ///
    /// The whole customization path is three calls: ``network(weightsURL:)`` to build, this to train,
    /// and `NFKMLXWeights.save` to write a checkpoint that ``backend(weightsURL:)`` loads like any
    /// other.
    ///
    /// - Parameters:
    ///   - net: the estimator to train, from ``network(weightsURL:)``.
    ///   - photos: supplies an unlabeled batch `[N, H, W, 3]` in `0...1` for each step. No brightened
    ///     target is needed, because the objective is zero-reference.
    ///   - objective: what "well exposed" means for this consumer.
    ///   - optimizer: the update rule. Nil uses `Adam` at the reference learning rate.
    ///   - steps: how many batches to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXZeroDCENet,
        photos: (Int) -> MLXArray,
        objective: NFKMLXZeroDCEObjective = NFKMLXZeroDCEObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 0.1,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXTrainer.train(net, optimizer: optimizer ?? Adam(learningRate: 1e-4), steps: steps,
                                sample: photos, loss: objective.callAsFunction,
                                clipGradientNorm: clipGradientNorm, checkpoint: checkpoint,
                                observer: observer)
    }
}
