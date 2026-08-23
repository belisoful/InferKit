//
//  NFKMLXImageBridgeTests.swift
//  InferKitMLXTests
//
//  The byte-level halves of the bridge (CoreGraphics and Metal) hold no MLX and run here. The
//  bytes <-> MLXArray step needs the MLX runtime and is covered by the gated backend tests.
//

import XCTest
import CoreGraphics
import Metal
@testable import InferKitMLX

final class NFKMLXImageBridgeTests: XCTestCase {

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

    func testPremultipliedHelper() {
        XCTAssertEqual(NFKMLXImageBridge.premultiplied([255, 255, 255, 0]), [0, 0, 0, 0])
        XCTAssertEqual(NFKMLXImageBridge.premultiplied([100, 100, 100, 255]), [100, 100, 100, 255])
    }
}
