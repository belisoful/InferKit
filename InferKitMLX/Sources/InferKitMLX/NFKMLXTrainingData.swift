//
//  NFKMLXTrainingData.swift
//  InferKitMLX
//
//  Turning an app's own data into training batches.
//
//  A consumer holds `CGImage`s — photos from a library, masks drawn in the app — and the trainer wants
//  `MLXArray`s. These adapters are the bridge, reusing `NFKMLXImageBridge` so a training batch and an
//  inference input are built the same way and cannot drift apart.
//

import CoreGraphics
import Foundation
import MLX

/// Builds training tensors from an app's `CGImage` data.
public enum NFKMLXTrainingData {

    /// A single image as `[H, W, 3]` in `0...1`, the layout every image model here takes.
    public static func tensor(_ image: CGImage,
                              colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()) throws -> MLXArray {
        try NFKMLXImageBridge.tensor(from: image, channels: 3, colorSpace: colorSpace)
    }

    /// Several images stacked as `[N, H, W, 3]`.
    ///
    /// Every image must already share one size. Resizing is the caller's decision, because the right
    /// choice differs by model: a crop preserves scale, a scale preserves framing, and picking one
    /// here would silently impose it on both.
    public static func batch(_ images: [CGImage],
                             colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()) throws -> MLXArray {
        guard let first = images.first else {
            throw NFKMLXError.unsupportedInput
        }
        let tensors = try images.map { image -> MLXArray in
            guard image.width == first.width, image.height == first.height else {
                throw NFKMLXError.trainingDataMismatch(
                    "a batch needs one size: \(first.width)×\(first.height) and "
                    + "\(image.width)×\(image.height) were both supplied. Crop or scale first.")
            }
            return try tensor(image, colorSpace: colorSpace)
        }
        return stacked(tensors, axis: 0)
    }

    /// A grayscale image as a matte `[H, W]` in `0...1`, for a matting or alpha target.
    public static func matte(_ image: CGImage,
                             colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()) throws -> MLXArray {
        let rgb = try tensor(image, colorSpace: colorSpace)
        return rgb.mean(axis: 2)
    }

    /// A grayscale image as class indices `[H, W]`, inverting the label-map convention the
    /// segmentation backends emit: a class index is stored as `index / (classCount − 1)`.
    ///
    /// A mask painted in an app is written back through the same convention, so what a model outputs
    /// and what it trains against are the same encoding.
    public static func labels(_ image: CGImage, classCount: Int,
                              colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()) throws -> MLXArray {
        let gray = try matte(image, colorSpace: colorSpace)
        let scale = Float(max(classCount - 1, 1))
        return MLX.round(gray * scale).asType(.int32)
    }
}

/// Draws shuffled batches of indices, so a short run still sees the whole dataset in a varied order.
///
/// A fine-tune on a device runs over a handful of examples. Cycling them in a fixed order lets the
/// optimizer chase the sequence rather than the data, and reshuffling each pass is the standard
/// remedy. The order is derived from `seed` and the pass number, so a run is repeatable.
public struct NFKMLXBatchSampler: Sendable {

    private let count: Int
    private let batchSize: Int
    private let seed: UInt64

    /// - Parameters:
    ///   - count: how many examples the dataset holds.
    ///   - batchSize: how many to draw per step.
    ///   - seed: fixes the shuffle, so two runs see the same order.
    public init(count: Int, batchSize: Int = 1, seed: UInt64 = 0) {
        self.count = max(count, 1)
        self.batchSize = max(batchSize, 1)
        self.seed = seed
    }

    /// The example indices for `step`, drawn from a fresh shuffle each time the dataset is exhausted.
    public func indices(forStep step: Int) -> [Int] {
        let batchesPerPass = max((count + batchSize - 1) / batchSize, 1)
        let pass = step / batchesPerPass
        let offset = (step % batchesPerPass) * batchSize
        let order = shuffledOrder(pass: pass)
        return (0 ..< batchSize).map { order[(offset + $0) % count] }
    }

    /// A Fisher-Yates shuffle over a SplitMix64 stream keyed by the pass, matching how the diffusion
    /// schedulers derive repeatable randomness without touching MLX's global state.
    private func shuffledOrder(pass: Int) -> [Int] {
        var state = seed &+ (UInt64(bitPattern: Int64(pass)) &* 0x9E37_79B9_7F4A_7C15)
        func next() -> UInt64 {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }

        var order = Array(0 ..< count)
        guard count > 1 else {
            return order
        }
        for index in stride(from: count - 1, to: 0, by: -1) {
            let swap = Int(next() % UInt64(index + 1))
            order.swapAt(index, swap)
        }
        return order
    }
}
