//
//  NFKMLXImageBridgeTests.swift
//  InferKitMLXTests
//
//  The byte-level halves of the bridge (CoreGraphics, CoreVideo, and Metal) hold no MLX and run here. The
//  bytes <-> MLXArray step needs the MLX runtime; its tests here skip without it.
//

import XCTest
import CoreGraphics
import CoreVideo
import ImageIO
import Metal
import MLX
import InferKit
@testable import InferKitMLX

final class NFKMLXImageBridgeTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    func testStraightRGBASurvivesTheCGImageWriteAndReadBack() throws {
        let rgba: [UInt8] = [200, 100, 50, 128, 10, 20, 30, 255, 0, 0, 0, 0, 255, 255, 255, 200]
        let image = try NFKMLXImageBridge.cgImage(rgba: rgba, width: 2, height: 2, options: NFKMLXImageOptions())
        let data = [UInt8](image.dataProvider!.data! as Data)
        XCTAssertEqual(data, rgba, "straight RGBA should be stored verbatim")
    }

    func testPremultiplyOptionScalesRGBByAlphaOnWrite() throws {
        let rgba: [UInt8] = [200, 100, 50, 128]
        let options = NFKMLXImageOptions(premultiply: true)
        let image = try NFKMLXImageBridge.cgImage(rgba: rgba, width: 1, height: 1, options: options)
        let data = [UInt8](image.dataProvider!.data! as Data)
        XCTAssertEqual(data, NFKMLXImageBridge.premultiplied(rgba))
        XCTAssertEqual(data, [100, 50, 25, 128])            // 200*0.5, 100*0.5, 50*0.5
    }

    func testAnOpaqueCGImageReadsBackItsColors() {
        let source: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255]      // red, green (opaque)
        let provider = CGDataProvider(data: Data(source) as CFData)!
        let image = CGImage(width: 2, height: 1, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let (bytes, width, height) = NFKMLXImageBridge.rgbaBytes(from: image, colorSpace: CGColorSpaceCreateDeviceRGB())
        XCTAssertEqual([width, height], [2, 1])
        XCTAssertEqual(Array(bytes[0..<3]), [255, 0, 0])
        XCTAssertEqual(Array(bytes[4..<7]), [0, 255, 0])
    }

    func testAPixelBufferReadsAsTheImageItWasDrawnFrom() throws {
        let source: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 40, 80, 120, 255]
        let image = try NFKMLXImageBridge.cgImage(rgba: source, width: 2, height: 2, options: NFKMLXImageOptions())
        let buffer = try XCTUnwrap(NFKImageCoding.pixelBuffer(with: image))
        let read = try NFKMLXImageBridge.rgbaBytes(from: buffer, colorSpace: CGColorSpaceCreateDeviceRGB())
        XCTAssertEqual([read.width, read.height], [2, 2])
        XCTAssertEqual(read.bytes, source, "a 32BGRA pixel buffer reads as the RGBA it holds")
        XCTAssertEqual(try NFKMLXImageBridge.cgImage(fromValue: buffer).width, 2)
        XCTAssertThrowsError(try NFKMLXImageBridge.cgImage(fromValue: "not an image"))
    }

    func testTheTextureBridgeRoundTripsRGBA() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device")
        }
        let rgba: [UInt8] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        let texture = try NFKMLXImageBridge.texture(rgba: rgba, width: 2, height: 2, device: device, options: NFKMLXImageOptions())
        let (readback, width, height) = try NFKMLXImageBridge.rgbaBytes(from: texture)
        XCTAssertEqual([width, height], [2, 2])
        XCTAssertEqual(readback, rgba)
    }

    func testBGRATextureIsSwizzledToRGBAOnRead() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device")
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 1, height: 1, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        let texture = device.makeTexture(descriptor: descriptor)!
        var bgra: [UInt8] = [50, 100, 200, 255]                     // B=50, G=100, R=200
        texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &bgra, bytesPerRow: 4)
        let (rgba, _, _) = try NFKMLXImageBridge.rgbaBytes(from: texture)
        XCTAssertEqual(rgba, [200, 100, 50, 255], "R and B should be swapped")
    }

    // MARK: Straight alpha on every read

    private static let translucent: [UInt8] = [200, 200, 200, 128, 50, 100, 150, 0]

    private func image(_ pixels: [UInt8], width: Int, alpha: CGImageAlphaInfo) throws -> CGImage {
        let provider = try XCTUnwrap(CGDataProvider(data: Data(pixels) as CFData))
        return try XCTUnwrap(CGImage(width: width, height: pixels.count / 4 / width, bitsPerComponent: 8, bitsPerPixel: 32,
                                     bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGBitmapInfo(rawValue: alpha.rawValue), provider: provider,
                                     decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    }

    func testATranslucentPixelReadsStraightThroughEveryPath() throws {
        let space = CGColorSpaceCreateDeviceRGB()
        let straight = try image(Self.translucent, width: 2, alpha: .last)
        XCTAssertEqual(NFKMLXImageBridge.rgbaBytes(from: straight, colorSpace: space).bytes,
                       [200, 200, 200, 128, 0, 0, 0, 0], "a straight image reads as stored; zero alpha reads as zero color")
        XCTAssertEqual(try NFKMLXImageBridge.rgbaBytes(from: straight as Any, colorSpace: space).bytes,
                       [200, 200, 200, 128, 0, 0, 0, 0])
        XCTAssertEqual(try NFKMLXImageBridge.rgbBytes(from: straight, colorSpace: space).bytes, [200, 200, 200, 0, 0, 0])

        let premultiplied = try image([100, 100, 100, 128], width: 1, alpha: .premultipliedLast)
        XCTAssertEqual(NFKMLXImageBridge.rgbaBytes(from: premultiplied, colorSpace: space).bytes, [199, 199, 199, 128],
                       "premultiplied color is divided back by alpha: 100 × 255 / 128 = 199.2")

        let encoded = NSMutableData()
        let destination = try XCTUnwrap(CGImageDestinationCreateWithData(encoded, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, straight, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let source = try XCTUnwrap(CGImageSourceCreateWithData(encoded, nil))
        let decoded = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual(NFKMLXImageBridge.rgbaBytes(from: decoded, colorSpace: space).bytes, [200, 200, 200, 128, 0, 0, 0, 0],
                       "a decoded PNG reads as Pillow reads it")

        // NFKImageCoding draws a buffer premultiplied in 8 bits, so it holds 100 for 200 under alpha 128.
        let buffer = try XCTUnwrap(NFKImageCoding.pixelBuffer(with: straight))
        XCTAssertEqual(try NFKMLXImageBridge.rgbaBytes(from: buffer, colorSpace: space).bytes, [199, 199, 199, 128, 0, 0, 0, 0])

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device")
        }
        let texture = try NFKMLXImageBridge.texture(rgba: Self.translucent, width: 2, height: 1, device: device,
                                                    options: NFKMLXImageOptions())
        XCTAssertEqual(try NFKMLXImageBridge.rgbaBytes(from: texture as Any, colorSpace: space).bytes, Self.translucent,
                       "a texture reads its bytes as straight color")
    }

    func testEveryStraightColorUnderNonzeroAlphaSurvivesTheDraw() throws {
        var pixels = [UInt8](repeating: 0, count: 256 * 256 * 4)
        for alpha in 0 ..< 256 {
            for color in 0 ..< 256 {
                let index = (alpha * 256 + color) * 4
                pixels[index] = UInt8(color)
                pixels[index + 1] = UInt8(255 - color)
                pixels[index + 2] = UInt8(color / 2)
                pixels[index + 3] = UInt8(alpha)
            }
        }
        let read = NFKMLXImageBridge.rgbaBytes(from: try image(pixels, width: 256, alpha: .last),
                                               colorSpace: CGColorSpaceCreateDeviceRGB()).bytes
        XCTAssertEqual(Array(read[(256 * 4)...]), Array(pixels[(256 * 4)...]))
        XCTAssertEqual(Array(read[..<(256 * 4)]), [UInt8](repeating: 0, count: 256 * 4), "zero alpha reads as zero color")
    }

    func testTheTensorOfATranslucentImageHoldsStraightColor() throws {
        try requireMLXRuntime()
        let straight = try image(Self.translucent, width: 2, alpha: .last)
        let tensor = try NFKMLXImageBridge.tensor(from: straight, channels: 4, colorSpace: CGColorSpaceCreateDeviceRGB())
        XCTAssertEqual(tensor.shape, [1, 2, 4])
        let values = tensor.asArray(Float.self)
        XCTAssertEqual(values[0], 200 / 255, accuracy: 1e-6)
        XCTAssertEqual(values[3], 128 / 255, accuracy: 1e-6)
    }

    // MARK: Pillow's RGB conversions

    /// Four pixels and what Pillow 11.3 makes of them, recorded from the `llm` oracle interpreter.
    private static let pillowSource: [UInt8] = [200, 100, 150, 128, 10, 20, 30, 0, 37, 255, 1, 1, 90, 180, 250, 254]
    private static let pillowOnWhite: [UInt8] = [227, 177, 202, 255, 255, 255, 254, 255, 254, 91, 180, 250]
    private static let pillowConvertedRGB: [UInt8] = [200, 100, 150, 10, 20, 30, 37, 255, 1, 90, 180, 250]

    private func opaqueImage(rgb: [UInt8], width: Int) throws -> CGImage {
        var rgba = [UInt8]()
        for pixel in stride(from: 0, to: rgb.count, by: 3) {
            rgba += [rgb[pixel], rgb[pixel + 1], rgb[pixel + 2], 255]
        }
        return try image(rgba, width: width, alpha: .noneSkipLast)
    }

    private func rgb(of image: CGImage) -> [UInt8] {
        let rgba = NFKMLXImageBridge.rgbaBytes(from: image, colorSpace: CGColorSpaceCreateDeviceRGB()).bytes
        return stride(from: 0, to: rgba.count, by: 4).flatMap { rgba[$0 ..< $0 + 3] }
    }

    func testDroppingAlphaKeepsTheStoredColorAsPillowConvertsToRGB() throws {
        let straight = try image(Self.pillowSource, width: 4, alpha: .last)
        let dropped = NFKMLXImageBridge.droppingAlpha(straight)
        XCTAssertFalse(NFKMLXImageBridge.hasAlpha(dropped))
        XCTAssertEqual(rgb(of: dropped), Self.pillowConvertedRGB, "the color under zero alpha survives")

        let premultiplied = try image([100, 100, 100, 128], width: 1, alpha: .premultipliedLast)
        XCTAssertEqual(rgb(of: NFKMLXImageBridge.droppingAlpha(premultiplied)), [199, 199, 199])

        let opaque = try opaqueImage(rgb: [1, 2, 3], width: 1)
        XCTAssertTrue(NFKMLXImageBridge.droppingAlpha(opaque) === opaque)
    }

    func testCompositingOnWhiteMatchesPillow() throws {
        let straight = try image(Self.pillowSource, width: 4, alpha: .last)
        let composited = NFKMLXImageBridge.compositedOnWhite(straight)
        let rgba = NFKMLXImageBridge.rgbaBytes(from: composited, colorSpace: CGColorSpaceCreateDeviceRGB()).bytes
        XCTAssertTrue(stride(from: 3, to: rgba.count, by: 4).allSatisfy { rgba[$0] == 255 }, "the composite is opaque")
        XCTAssertEqual(rgb(of: composited), Self.pillowOnWhite)

        let opaqueWithAlphaChannel = try image([1, 2, 3, 255], width: 1, alpha: .last)
        XCTAssertTrue(NFKMLXImageBridge.compositedOnWhite(opaqueWithAlphaChannel) === opaqueWithAlphaChannel,
                      "an image whose every pixel is opaque is used as it is")
    }

    func testTheVisionProcessorsAndTheControlImageReadAlphaAsTheirReferences() throws {
        try requireMLXRuntime()
        let translucent = try image(Self.pillowSource, width: 4, alpha: .last)
        let onWhite = try opaqueImage(rgb: Self.pillowOnWhite, width: 4)
        let converted = try opaqueImage(rgb: Self.pillowConvertedRGB, width: 4)
        func same(_ a: MLXArray, _ b: MLXArray, _ message: String) {
            XCTAssertEqual(a.shape, b.shape, message)
            XCTAssertEqual(abs(a - b).max().item(Float.self), 0, message)
        }

        let pixtral = NFKMLXPixtralImageProcessor(patchSize: 2, longestEdge: 8)
        same(pixtral.process(translucent).pixelValues, pixtral.process(onWhite).pixelValues, "Pixtral pastes onto white")
        same(NFKMLXSmolVLMImageProcessor.process(translucent, tileSize: 4, longestEdge: 4).pixelValues,
             NFKMLXSmolVLMImageProcessor.process(onWhite, tileSize: 4, longestEdge: 4).pixelValues,
             "SmolVLM composites onto white")
        let gemma4 = NFKMLXGemma4ImageProcessor(patchSize: 2, poolingKernelSize: 1, maxSoftTokens: 8)
        same(gemma4.process(translucent).pixelValues, gemma4.process(converted).pixelValues, "Gemma 4 drops alpha")
        same(try NFKMLXControlImage.pixels(translucent, width: 4, height: 1),
             try NFKMLXControlImage.pixels(converted, width: 4, height: 1), "a control image drops alpha")
    }

    func testPremultipliedHelper() {
        XCTAssertEqual(NFKMLXImageBridge.premultiplied([255, 255, 255, 0]), [0, 0, 0, 0])
        XCTAssertEqual(NFKMLXImageBridge.premultiplied([100, 100, 100, 255]), [100, 100, 100, 255])
    }
}
