//
//  NFKMLXPillowLanczos.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation

// Pillow's Lanczos resize of an 8-bit RGB image (`Image.resize(…, Image.Resampling.LANCZOS)`), to the
// integer: a reference that stretches decoded frames through PIL hands the next stage those exact bytes.
// Each axis is a separable pass of a three-lobe Lanczos window whose support widens with the downscale
// factor; the taps are normalized in double precision, then quantized to 22-bit fixed point, and every
// pass rounds its sum back to 8 bits. Width resamples first, then height.

enum NFKMLXPillowLanczos {

    /// Bits of fixed-point precision Pillow keeps for an 8-bit image's taps.
    static let precisionBits = 32 - 8 - 2

    /// The truncated sinc of three lobes.
    static func lanczos(_ x: Double) -> Double {
        guard -3 <= x, x < 3 else { return 0 }
        func sinc(_ value: Double) -> Double {
            guard value != 0 else { return 1 }
            let v = value * Double.pi
            return sin(v) / v
        }
        return sinc(x) * sinc(x / 3)
    }

    /// Each output position's first input index and tap count, and its fixed-point taps.
    static func coefficients(inSize: Int, outSize: Int) -> (bounds: [(Int, Int)], taps: [[Int32]]) {
        let scale = Double(Float(inSize) - Float(0)) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = 3 * filterScale
        var bounds = [(Int, Int)](), taps = [[Int32]]()
        for xx in 0 ..< outSize {
            let center = (Double(xx) + 0.5) * scale
            let ss = 1 / filterScale
            let xmin = max(Int(center - support + 0.5), 0)
            let xmax = min(Int(center + support + 0.5), inSize) - xmin
            var weights = (0 ..< xmax).map { lanczos((Double($0 + xmin) - center + 0.5) * ss) }
            let total = weights.reduce(0, +)
            if total != 0 { weights = weights.map { $0 / total } }
            let unit = Double(1 << precisionBits)
            taps.append(weights.map { $0 < 0 ? Int32(-0.5 + $0 * unit) : Int32(0.5 + $0 * unit) })
            bounds.append((xmin, xmax))
        }
        return (bounds, taps)
    }

    static func clip8(_ value: Int32) -> UInt8 { UInt8(min(max(value >> Int32(precisionBits), 0), 255)) }

    /// `rgb` (`height × width × 3`, row-major) resized to `outWidth × outHeight`.
    static func resize(_ rgb: [UInt8], width: Int, height: Int, outWidth: Int, outHeight: Int) -> [UInt8] {
        let half = Int32(1 << (precisionBits - 1))
        var current = rgb, w = width
        if outWidth != width {
            let (bounds, taps) = coefficients(inSize: width, outSize: outWidth)
            var next = [UInt8](repeating: 0, count: height * outWidth * 3)
            for y in 0 ..< height {
                for xx in 0 ..< outWidth {
                    let (xmin, count) = bounds[xx]
                    for channel in 0 ..< 3 {
                        var sum = half
                        for x in 0 ..< count {
                            sum &+= Int32(current[(y * w + x + xmin) * 3 + channel]) &* taps[xx][x]
                        }
                        next[(y * outWidth + xx) * 3 + channel] = clip8(sum)
                    }
                }
            }
            current = next
            w = outWidth
        }
        if outHeight != height {
            let (bounds, taps) = coefficients(inSize: height, outSize: outHeight)
            var next = [UInt8](repeating: 0, count: outHeight * w * 3)
            for yy in 0 ..< outHeight {
                let (ymin, count) = bounds[yy]
                for x in 0 ..< w {
                    for channel in 0 ..< 3 {
                        var sum = half
                        for y in 0 ..< count {
                            sum &+= Int32(current[((y + ymin) * w + x) * 3 + channel]) &* taps[yy][y]
                        }
                        next[(yy * w + x) * 3 + channel] = clip8(sum)
                    }
                }
            }
            current = next
        }
        return current
    }
}
