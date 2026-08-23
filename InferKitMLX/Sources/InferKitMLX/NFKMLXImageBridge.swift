//
//  NFKMLXImageBridge.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import Metal
import MLX

/// How the image bridge reads and writes pixels.
public struct NFKMLXImageOptions: @unchecked Sendable {
    /// The color space CGImages are created in and read against. Defaults to device RGB.
    public var colorSpace: CGColorSpace
    /// When writing an output, premultiply RGB by alpha (compositing) instead of leaving it straight.
    public var premultiply: Bool

    public init(colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB(), premultiply: Bool = false) {
        self.colorSpace = colorSpace
        self.premultiply = premultiply
    }
}

/// Converts image values (a `CGImage` or an `MTLTexture`) to and from `MLXArray`, preserving the
/// alpha channel. The byte-level halves — CoreGraphics and Metal — hold no MLX and are exercised
/// directly; only the thin `bytes ↔ MLXArray` step needs the MLX runtime.
enum NFKMLXImageBridge {

    enum BridgeError: Error {
        case unsupportedInput
        case unsupportedTextureFormat
        case badShape
        case imageCreationFailed
        case noMetalDevice
    }

    // MARK: CGImage <-> RGBA8 bytes (no MLX)

    static func rgbaBytes(from image: CGImage, colorSpace: CGColorSpace) -> (bytes: [UInt8], width: Int, height: Int) {
        let width = image.width
        let height = image.height
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        bytes.withUnsafeMutableBytes { raw in
            let context = CGContext(data: raw.baseAddress, width: width, height: height,
                                    bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace,
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return (bytes, width, height)
    }

    static func cgImage(rgba: [UInt8], width: Int, height: Int, options: NFKMLXImageOptions) throws -> CGImage {
        let pixels = options.premultiply ? premultiplied(rgba) : rgba
        let alpha = options.premultiply ? CGImageAlphaInfo.premultipliedLast : CGImageAlphaInfo.last
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else {
            throw BridgeError.imageCreationFailed
        }
        guard let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: width * 4, space: options.colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: alpha.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) else {
            throw BridgeError.imageCreationFailed
        }
        return image
    }

    // MARK: MTLTexture <-> RGBA8 bytes (no MLX)

    static func rgbaBytes(from texture: MTLTexture) throws -> (bytes: [UInt8], width: Int, height: Int) {
        let width = texture.width
        let height = texture.height
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        bytes.withUnsafeMutableBytes { raw in
            texture.getBytes(raw.baseAddress!, bytesPerRow: width * 4,
                             from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        switch texture.pixelFormat {
        case .rgba8Unorm, .rgba8Unorm_srgb:
            return (bytes, width, height)
        case .bgra8Unorm, .bgra8Unorm_srgb:
            for pixel in 0 ..< (width * height) {                    // swap B and R
                bytes.swapAt(pixel * 4, pixel * 4 + 2)
            }
            return (bytes, width, height)
        default:
            throw BridgeError.unsupportedTextureFormat
        }
    }

    static func texture(rgba: [UInt8], width: Int, height: Int, device: MTLDevice,
                        options: NFKMLXImageOptions) throws -> MTLTexture {
        let pixels = options.premultiply ? premultiplied(rgba) : rgba
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                                  width: width, height: height,
                                                                  mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw BridgeError.imageCreationFailed
        }
        pixels.withUnsafeBytes { raw in
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                            withBytes: raw.baseAddress!, bytesPerRow: width * 4)
        }
        return texture
    }

    // MARK: RGBA8 bytes <-> MLXArray (needs the MLX runtime)

    static func tensor(rgba: [UInt8], width: Int, height: Int, channels: Int) -> MLXArray {
        var floats = [Float](repeating: 0, count: width * height * channels)
        for pixel in 0 ..< (width * height) {
            for channel in 0 ..< channels {
                floats[pixel * channels + channel] = Float(rgba[pixel * 4 + channel]) / 255.0
            }
        }
        return floats.withUnsafeBufferPointer { MLXArray($0, [height, width, channels]) }
    }

    /// Straight RGBA8 bytes from an `[H, W, C]` tensor (C is 1, 3, or 4). C=1 replicates to gray.
    static func rgbaBytes(from array: MLXArray) throws -> (bytes: [UInt8], width: Int, height: Int) {
        let values = array.asType(Float.self)
        eval(values)
        let shape = values.shape
        guard shape.count >= 3 else {
            throw BridgeError.badShape
        }
        let channels = shape[shape.count - 1]
        let width = shape[shape.count - 2]
        let height = shape[shape.count - 3]
        guard channels == 1 || channels == 3 || channels == 4 else {
            throw BridgeError.badShape
        }

        let flat = values.asArray(Float.self)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        func byte(_ v: Float) -> UInt8 { UInt8((max(0, min(1, v)) * 255).rounded()) }
        for pixel in 0 ..< (width * height) {
            let base = pixel * channels
            let out = pixel * 4
            switch channels {
            case 4:
                rgba[out] = byte(flat[base]); rgba[out + 1] = byte(flat[base + 1])
                rgba[out + 2] = byte(flat[base + 2]); rgba[out + 3] = byte(flat[base + 3])
            case 3:
                rgba[out] = byte(flat[base]); rgba[out + 1] = byte(flat[base + 1])
                rgba[out + 2] = byte(flat[base + 2]); rgba[out + 3] = 255
            default:
                let v = byte(flat[base])
                rgba[out] = v; rgba[out + 1] = v; rgba[out + 2] = v; rgba[out + 3] = 255
            }
        }
        return (rgba, width, height)
    }

    // MARK: Composed — a value (CGImage or MTLTexture) in, a value out

    /// Straight RGBA8 bytes + dimensions from a `CGImage` or an `MTLTexture` input value.
    static func rgbaBytes(from value: Any, colorSpace: CGColorSpace) throws -> (bytes: [UInt8], width: Int, height: Int) {
        if let texture = value as? MTLTexture {
            return try rgbaBytes(from: texture)
        }
        let cf = value as CFTypeRef
        if CFGetTypeID(cf) == CGImage.typeID {
            return rgbaBytes(from: cf as! CGImage, colorSpace: colorSpace)
        }
        throw BridgeError.unsupportedInput
    }

    static func tensor(from value: Any, channels: Int, colorSpace: CGColorSpace) throws -> MLXArray {
        if let texture = value as? MTLTexture {
            let (bytes, width, height) = try rgbaBytes(from: texture)
            return tensor(rgba: bytes, width: width, height: height, channels: channels)
        }
        let cf = value as CFTypeRef
        if CFGetTypeID(cf) == CGImage.typeID {
            let (bytes, width, height) = rgbaBytes(from: cf as! CGImage, colorSpace: colorSpace)
            return tensor(rgba: bytes, width: width, height: height, channels: channels)
        }
        throw BridgeError.unsupportedInput
    }

    static func cgImage(from array: MLXArray, options: NFKMLXImageOptions) throws -> CGImage {
        let (bytes, width, height) = try rgbaBytes(from: array)
        return try cgImage(rgba: bytes, width: width, height: height, options: options)
    }

    static func texture(from array: MLXArray, device: MTLDevice, options: NFKMLXImageOptions) throws -> MTLTexture {
        let (bytes, width, height) = try rgbaBytes(from: array)
        return try texture(rgba: bytes, width: width, height: height, device: device, options: options)
    }

    // MARK: Helpers

    static func premultiplied(_ rgba: [UInt8]) -> [UInt8] {
        var out = rgba
        var index = 0
        while index < out.count {
            let alpha = Float(out[index + 3]) / 255.0
            out[index] = UInt8((Float(out[index]) * alpha).rounded())
            out[index + 1] = UInt8((Float(out[index + 1]) * alpha).rounded())
            out[index + 2] = UInt8((Float(out[index + 2]) * alpha).rounded())
            index += 4
        }
        return out
    }
}
