//
//  NFKMLXDeepSeekQuantization.swift
//  InferKitMLX
//
//  Block-scaled fp8 and fp4 dequantization, which is how the released DeepSeek V4 checkpoint stores
//  its weights. Every other model here publishes float weights, so this is the first loader that has
//  to decode a storage format before a module can hold the values.
//
//  Introduced in InferKit 0.1.0.
//

import Foundation
import MLX

/// Decodes the block-scaled 8-bit and 4-bit formats the DeepSeek V4 release stores its weights in.
///
/// @discussion A quantized tensor is a byte array plus a second array of per-block scales. A value's
/// float weight is its own decoded magnitude times the scale of the block it belongs to, and a block
/// is a fixed run along each axis: 128×128 for the fp8 attention and shared-expert weights, and 32
/// values along the last axis for a routed expert's 4-bit weights.
///
/// The scales are themselves a narrow format — `e8m0`, an exponent with no sign and no mantissa — so
/// a scale is exactly a power of two and the decode is a shift rather than a multiply. That is what
/// `fast_round_scale` in the reference produces.
///
/// Both formats decode through a lookup table, because a byte has only 256 possible values and a
/// nibble 16. The table is built once and indexed with `take`, which is one gather instead of the
/// bit arithmetic a per-element decode would need.
public enum NFKMLXDeepSeekQuantization {

    /// The block a scale covers in an fp8 weight, along each of the two axes.
    public static let fp8BlockSize = 128

    /// The run of values a scale covers in a 4-bit weight, along the last axis.
    public static let fp4BlockSize = 32

    // MARK: - Value tables

    /// Every `e4m3` byte as a float: sign, a 4-bit exponent biased by 7, and a 3-bit mantissa.
    ///
    /// @discussion The `fn` in `e4m3fn` means finite: the format spends no encoding on infinity, so
    /// the largest magnitude is 448 and the all-ones significand is the only NaN.
    static let fp8Values: [Float] = (0 ..< 256).map { bits -> Float in
        let sign: Float = (bits & 0x80) != 0 ? -1 : 1
        let exponent = (bits >> 3) & 0x0F
        let mantissa = bits & 0x07
        if exponent == 0x0F && mantissa == 0x07 { return Float.nan }
        if exponent == 0 { return sign * Float(mantissa) / 8 * powf(2, -6) }
        return sign * (1 + Float(mantissa) / 8) * powf(2, Float(exponent) - 7)
    }

    /// Every `e2m1` nibble as a float: sign, a 2-bit exponent biased by 1, and a 1-bit mantissa.
    ///
    /// @discussion The eight magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, and 6. There is no NaN and no
    /// infinity, so 6 is the largest value a 4-bit weight can hold, which is the bound the reference
    /// quantizer clamps to.
    static let fp4Values: [Float] = (0 ..< 16).map { bits -> Float in
        let sign: Float = (bits & 0x08) != 0 ? -1 : 1
        let exponent = (bits >> 1) & 0x03
        let mantissa = bits & 0x01
        if exponent == 0 { return sign * Float(mantissa) / 2 }
        return sign * (1 + Float(mantissa) / 2) * powf(2, Float(exponent) - 1)
    }

    /// Every `e8m0` byte as a float: an 8-bit exponent biased by 127, with no sign and no mantissa.
    static let scaleValues: [Float] = (0 ..< 256).map { bits -> Float in
        bits == 0xFF ? Float.nan : exp2(Float(bits) - 127)
    }

    // MARK: - Dequantization

    /// Decodes an fp8 weight and its 128×128 block scales into float.
    ///
    /// - Parameters:
    ///   - bytes: the stored weight, one byte a value, of the weight's own shape.
    ///   - scaleBytes: one `e8m0` byte per block, so `ceil(rows / 128) × ceil(columns / 128)`.
    public static func dequantizeFP8(bytes: MLXArray, scaleBytes: MLXArray) -> MLXArray {
        let values = decoded(bytes, table: fp8Values)
        let scales = decoded(scaleBytes, table: scaleValues)
        return values * expanded(scales, toShape: values.shape, blockSize: fp8BlockSize)
    }

    /// Decodes a 4-bit weight packed two values to a byte, with its 32-value block scales.
    ///
    /// @discussion The pair in a byte is ordered low nibble first, so a byte holds elements `2i` and
    /// `2i + 1` of the last axis in that order, and the decoded weight is twice as wide as the stored
    /// one. Packing runs along the last axis, which is the reduction axis of the matrix multiply the
    /// reference performs.
    ///
    /// The nibble order is NOT measurable from the checkpoint: both orders decode to the same values
    /// within a block, so no statistic separates them. It follows the format's own convention, and
    /// `testTheFourBitNibbleOrderIsTheFormatsOwn` pins it against a hand-encoded byte.
    ///
    /// - Parameters:
    ///   - packedBytes: the stored weight, two values a byte, so half the weight's last axis.
    ///   - scaleBytes: one `e8m0` byte per 32 values along the last axis.
    public static func dequantizeFP4(packedBytes: MLXArray, scaleBytes: MLXArray) -> MLXArray {
        let low = decoded(packedBytes & MLXArray(UInt8(0x0F)), table: fp4Values)
        let high = decoded(packedBytes >> MLXArray(UInt8(4)), table: fp4Values)
        var shape = packedBytes.shape
        shape[shape.count - 1] *= 2
        let values = stacked([low, high], axis: -1).reshaped(shape)
        let scales = decoded(scaleBytes, table: scaleValues)
        return values * expandedLastAxis(scales, toWidth: shape[shape.count - 1],
                                         blockSize: fp4BlockSize)
    }

    // MARK: - Helpers

    private static func decoded(_ bytes: MLXArray, table: [Float]) -> MLXArray {
        MLXArray(table).take(bytes.asType(.int32).flattened()).reshaped(bytes.shape)
    }

    /// Repeats each scale over the block it covers, on every axis, then trims to the weight's shape.
    ///
    /// @discussion A weight whose side is not a multiple of the block size has a partial last block,
    /// which the trim is for.
    private static func expanded(_ scales: MLXArray, toShape shape: [Int],
                                 blockSize: Int) -> MLXArray {
        var expanded = scales
        for axis in 0 ..< scales.ndim {
            expanded = repeated(expanded, count: blockSize, axis: axis)
        }
        return trimmed(expanded, to: shape)
    }

    private static func expandedLastAxis(_ scales: MLXArray, toWidth width: Int,
                                         blockSize: Int) -> MLXArray {
        var shape = scales.shape
        shape[shape.count - 1] = min(scales.shape[scales.ndim - 1] * blockSize, width)
        return trimmed(repeated(scales, count: blockSize, axis: -1), to: shape)
    }

    private static func trimmed(_ array: MLXArray, to shape: [Int]) -> MLXArray {
        var result = array
        for (axis, extent) in shape.enumerated() where result.shape[axis] > extent {
            result = result.split(indices: [extent], axis: axis)[0]
        }
        return result
    }
}
