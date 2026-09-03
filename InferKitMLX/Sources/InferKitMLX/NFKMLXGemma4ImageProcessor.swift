//
//  NFKMLXGemma4ImageProcessor.swift
//  InferKitMLX
//
//  Turns a `CGImage` into the flattened patches and their positions the Gemma 4 vision tower reads.
//  The image is resized preserving aspect ratio to fit within a patch budget (with both sides a
//  multiple of `poolingKernelSize · patchSize`), rescaled to `0 … 1`, split into non-overlapping
//  patches flattened `(row, column, channel)`, and padded to the budget; each patch carries its
//  `(x, y)` grid position, and a padding patch is `(-1, -1)`.
//
//  The resize is CoreGraphics rather than the reference's torchvision BICUBIC, so the patch pixel
//  values are a close approximation rather than byte-identical — the same documented difference the
//  SmolVLM processor carries. The patch layout, the position ids, and the resized dimensions are the
//  reference's exactly.
//

import Foundation
import CoreGraphics
import MLX

/// The Gemma 4 image processor. The defaults are the released processor's: 16-pixel patches, a
/// 3×3 pooling kernel, and a 280-soft-token budget (2520 patches).
public struct NFKMLXGemma4ImageProcessor {
    public var patchSize: Int
    public var poolingKernelSize: Int
    public var maxSoftTokens: Int

    public init(patchSize: Int = 16, poolingKernelSize: Int = 3, maxSoftTokens: Int = 280) {
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.maxSoftTokens = maxSoftTokens
    }

    /// The most patches an image is split into: the soft-token budget times the pooling area.
    public var maxPatches: Int { maxSoftTokens * poolingKernelSize * poolingKernelSize }

    /// The resized `(height, width)`: the largest that stays within the patch budget and keeps both
    /// sides a multiple of `poolingKernelSize · patchSize`, preserving aspect ratio.
    public func resizedSize(width: Int, height: Int) -> (height: Int, width: Int) {
        let sideMultiple = poolingKernelSize * patchSize
        let targetPixels = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPixels / Double(height * width)).squareRoot()
        var targetHeight = Int((factor * Double(height) / Double(sideMultiple)).rounded(.down)) * sideMultiple
        var targetWidth = Int((factor * Double(width) / Double(sideMultiple)).rounded(.down)) * sideMultiple
        let maxSide = (maxPatches / (poolingKernelSize * poolingKernelSize)) * sideMultiple
        if targetHeight == 0 && targetWidth == 0 {
            targetHeight = sideMultiple; targetWidth = sideMultiple
        } else if targetHeight == 0 {
            targetHeight = sideMultiple
            targetWidth = Swift.min((width / height) * sideMultiple, maxSide)
        } else if targetWidth == 0 {
            targetWidth = sideMultiple
            targetHeight = Swift.min((height / width) * sideMultiple, maxSide)
        }
        return (targetHeight, targetWidth)
    }

    /// The flattened patches `[maxPatches, 3·patchSize²]` in `0 … 1` and their positions
    /// `[maxPatches, 2]`, ready for ``NFKMLXGemma4VisionNet``.
    public func process(_ image: CGImage) -> (pixelValues: MLXArray, positionIds: MLXArray) {
        let (height, width) = resizedSize(width: image.width, height: image.height)
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        context?.interpolationQuality = .high
        context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let patchRows = height / patchSize, patchColumns = width / patchSize
        let patchCount = patchRows * patchColumns
        let patchPixels = patchSize * patchSize * 3
        var pixels = [Float](); pixels.reserveCapacity(maxPatches * patchPixels)
        var positions = [Int32](); positions.reserveCapacity(maxPatches * 2)

        for patchRow in 0 ..< patchRows {
            for patchColumn in 0 ..< patchColumns {
                // Each patch is flattened (row, column, channel) with the channel innermost.
                for row in 0 ..< patchSize {
                    for column in 0 ..< patchSize {
                        let y = patchRow * patchSize + row, x = patchColumn * patchSize + column
                        let base = (y * width + x) * 4
                        for channel in 0 ..< 3 { pixels.append(Float(bytes[base + channel]) / 255) }
                    }
                }
                positions.append(Int32(patchColumn))   // x
                positions.append(Int32(patchRow))       // y
            }
        }
        // Pad to the budget with zero patches and (-1, -1) positions.
        if patchCount < maxPatches {
            pixels.append(contentsOf: repeatElement(0, count: (maxPatches - patchCount) * patchPixels))
            positions.append(contentsOf: repeatElement(-1, count: (maxPatches - patchCount) * 2))
        }
        return (MLXArray(pixels).reshaped([maxPatches, patchPixels]),
                MLXArray(positions).reshaped([maxPatches, 2]))
    }
}
