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

    /// The block a scale covers in an fp8 weight, along each of the two axes, where a release does
    /// not state its own.
    ///
    /// @discussion This is NOT a property of the format. V4 blocks at 128 and V4.1 at 32, each
    /// stating it as `quantization_config.weight_block_size`, so a caller passes the release's own
    /// number and this default serves only a caller that has none. Assuming it reads a quarter of
    /// the scale rows and columns for a V4.1 weight and decodes wrong values with no error.
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

    /// Decodes an fp8 weight and its square block scales into float.
    ///
    /// - Parameters:
    ///   - bytes: the stored weight, one byte a value, of the weight's own shape.
    ///   - scaleBytes: one `e8m0` byte per block, so `ceil(rows / block) × ceil(columns / block)`.
    ///   - blockSize: the release's `weight_block_size`, 128 for V4 and 32 for V4.1.
    public static func dequantizeFP8(bytes: MLXArray, scaleBytes: MLXArray,
                                     blockSize: Int = fp8BlockSize) -> MLXArray {
        let values = decoded(bytes, table: fp8Values)
        let scales = decoded(scaleBytes, table: scaleValues)
        // Two blockings, told apart by the scale's own shape. An attention weight carries one scale
        // per SQUARE block, so its scale is smaller on both axes. The n-gram table carries one per
        // run of columns within EVERY row, so its scale has as many rows as the weight has; sharing
        // a scale down 32 rows of that table would decode 31 of them against another row's range.
        if scales.ndim == 2, values.ndim == 2, scales.dim(0) == values.dim(0) {
            return values * expandedLastAxis(scales, toWidth: values.dim(1), blockSize: blockSize)
        }
        return values * expanded(scales, toShape: values.shape, blockSize: blockSize)
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

    // MARK: - Activation round trips

    /// The release's in-place `act_quant`: block-wise fp8 with power-of-two scales.
    ///
    /// @discussion Transcribed from `kernel.py`. A block's absolute maximum, floored at 1e-4, times
    /// 1/448 and raised to the next power of two is the scale; the value is divided by it, clamped
    /// to ±448, rounded to e4m3 and multiplied back. The reciprocal is a MULTIPLY, as the kernel
    /// has it, because at an exact power of two a divide and a multiply can land either side of it.
    static func roundTripFP8(_ x: MLXArray, blockSize: Int) -> MLXArray {
        let (blocks, restore) = inBlocks(x, blockSize: blockSize)
        let amax = maximum(abs(blocks).max(axis: -1, keepDims: true), MLXArray(Float(1e-4)))
        let scale = nextPowerOfTwo(amax * MLXArray(Float(1.0 / 448.0)))
        let narrow = roundedE4M3(clip(blocks / scale, min: -448, max: 448))
        return restore(narrow * scale)
    }

    /// The release's in-place `fp4_act_quant` with power-of-two scales, which the indexer uses.
    static func roundTripFP4(_ x: MLXArray, blockSize: Int) -> MLXArray {
        let (blocks, restore) = inBlocks(x, blockSize: blockSize)
        let floor = MLXArray(Float(6) * powf(2, -126))
        let amax = maximum(abs(blocks).max(axis: -1, keepDims: true), floor)
        let scale = nextPowerOfTwo(amax * MLXArray(Float(1.0 / 6.0)))
        return restore(roundedE2M1(clip(blocks / scale, min: -6, max: 6)) * scale)
    }

    /// The same fp4 with an E4M3 scale, which the compressed latent uses.
    ///
    /// @discussion The scale here is not a power of two: the kernel DIVIDES the maximum by 6 and
    /// rounds the quotient to e4m3, so the division of each value by it is inexact and has to be the
    /// same division. The maximum is floored at 6·2⁻⁹ so an all-zero group keeps a nonzero scale,
    /// and the quotient is held to e4m3's range, which no activation approaches and where an
    /// overflowing cast is the one place the kernel and torch could disagree.
    static func roundTripFP4WithE4M3Scale(_ x: MLXArray, blockSize: Int) -> MLXArray {
        let (blocks, restore) = inBlocks(x, blockSize: blockSize)
        let floor = MLXArray(Float(6) * powf(2, -9))
        let amax = maximum(abs(blocks).max(axis: -1, keepDims: true), floor)
        let scale = roundedE4M3(minimum(amax / 6, MLXArray(Float(448))))
        return restore(roundedE2M1(clip(blocks / scale, min: -6, max: 6)) * scale)
    }

    /// Splits the last axis into blocks, and the inverse, back to the input's own dtype.
    private static func inBlocks(_ x: MLXArray, blockSize: Int)
        -> (MLXArray, (MLXArray) -> MLXArray) {
        let shape = x.shape
        let width = shape[shape.count - 1]
        // The kernel asserts this. A configuration that quantizes is checked for it at
        // `makeNet`, so reaching here without it is a caller that built the decoder by hand.
        precondition(width % blockSize == 0,
                     "a row of \(width) does not divide into blocks of \(blockSize)")
        let blocks = x.asType(.float32)
            .reshaped(Array(shape.dropLast()) + [width / blockSize, blockSize])
        return (blocks, { $0.reshaped(shape).asType(x.dtype) })
    }

    /// `⌊log₂|v|⌋` from the exponent field, which is exact where a float `log2` is not.
    private static func binade(_ v: MLXArray) -> MLXArray {
        let bits = abs(v).asType(.float32).view(dtype: .int32)
        return ((bits >> MLXArray(Int32(23))) & MLXArray(Int32(0xFF))) - MLXArray(Int32(127))
    }

    /// `2^e` for an integer exponent, built from the bits so it is exact.
    private static func powerOfTwo(_ exponent: MLXArray) -> MLXArray {
        ((exponent + MLXArray(Int32(127))) << MLXArray(Int32(23))).view(dtype: .float32)
    }

    /// `2^⌈log₂ v⌉` for a positive normal `v`: the kernel's `fast_log2_ceil` then `fast_pow2`, which
    /// is the exponent plus one when any mantissa bit is set, so a power of two is its own ceiling.
    static func nextPowerOfTwo(_ v: MLXArray) -> MLXArray {
        let bits = v.asType(.float32).view(dtype: .int32)
        let hasMantissa = ((bits & MLXArray(Int32(0x7F_FFFF))) .!= MLXArray(Int32(0))).asType(.int32)
        return powerOfTwo(binade(v) + hasMantissa)
    }

    /// Rounds to the nearest e4m3 value, ties to even, for a value already within ±448.
    ///
    /// @discussion MLX has no fp8 type, so this is the format's spacing made explicit: `2^(e−3)` in
    /// binade `e`, never finer than the subnormal step `2⁻⁹`. Dividing by a power of two is exact,
    /// and MLX's `round` is `rint`, which is ties-to-even; an integer multiple of the spacing with
    /// even parity is exactly an even mantissa, because a normal's integer is its mantissa plus 8.
    static func roundedE4M3(_ v: MLXArray) -> MLXArray {
        let quantum = powerOfTwo(maximum(binade(v), MLXArray(Int32(-6))) - MLXArray(Int32(3)))
        return round(v / quantum) * quantum
    }

    /// Rounds to the nearest e2m1 value, ties to even, for a value already within ±6: the
    /// spacing is `2^(e−1)` in binade `e`, never finer than the subnormal step `0.5`.
    static func roundedE2M1(_ v: MLXArray) -> MLXArray {
        let quantum = powerOfTwo(maximum(binade(v), MLXArray(Int32(0))) - MLXArray(Int32(1)))
        return round(v / quantum) * quantum
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
