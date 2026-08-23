//
//  NFKMLXMattingBackendTests.swift
//  InferKitMLXTests
//
//  Contract and byte-level tiling/matte tests need no GPU. The full round-trip evaluates MLX arrays,
//  which needs the MLX Metal library — the Xcode build system bundles it, a plain `swift test` does
//  not — so those tests skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXMattingBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        // MLX needs its Metal library, which the Xcode build system bundles (in mlx-swift_Cmlx.bundle)
        // but a plain `swift test` does not, so evaluating there aborts. Detect the SwiftPM CLI build
        // (the test bundle sits under .build) and skip; under xcodebuild these run.
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    // MARK: Contract (no MLX)

    func testTheBackendReportsTheSuppliedIdentityAndReadiness() {
        let backend = NFKMLXMattingBackend(identifier: "corridor-key", isReady: false) { plate, _ in plate }
        XCTAssertEqual(backend.backendIdentifier, "corridor-key")
        XCTAssertFalse(backend.isReady)
    }

    func testAnInferenceWithoutAPlateFails() {
        let backend = NFKMLXMattingBackend { plate, _ in plate }
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "no image here"])
        XCTAssertThrowsError(try backend.runInference(for: request))
    }

    // MARK: Byte-level tiling and matte (no MLX)

    func testSubImageAndPlaceRoundTripARectangle() {
        let source = [UInt8](0 ..< UInt8(3 * 2 * 4))                 // 3x2 RGBA
        let tile = NFKMLXMattingBackend.subImage(source, width: 3, x: 1, y: 0, tileWidth: 2, tileHeight: 2)
        var destination = [UInt8](repeating: 0, count: source.count)
        NFKMLXMattingBackend.place(tile, tileWidth: 2, tileHeight: 2, into: &destination, width: 3, x: 1, y: 0)
        for row in 0 ..< 2 {
            for column in 1 ..< 3 {
                let index = (row * 3 + column) * 4
                XCTAssertEqual(Array(destination[index ..< index + 4]), Array(source[index ..< index + 4]))
            }
        }
    }

    func testSubImageOfTheWholeImageIsIdentity() {
        let source = [UInt8](0 ..< UInt8(2 * 2 * 4))
        XCTAssertEqual(NFKMLXMattingBackend.subImage(source, width: 2, x: 0, y: 0, tileWidth: 2, tileHeight: 2), source)
    }

    func testMatteExtractsAlphaToGray() {
        let rgba: [UInt8] = [200, 100, 50, 128, 10, 20, 30, 255]
        XCTAssertEqual(NFKMLXMattingBackend.matte(from: rgba), [128, 128, 128, 255, 255, 255, 255, 255])
    }

    // MARK: Round trip (needs MLX)

    func testAStraightAlphaMatteSurvivesTheRoundTrip() throws {
        try requireMLXRuntime()
        let plate = Self.solidImage(width: 2, height: 2, red: 255, green: 0, blue: 0)
        let backend = NFKMLXMattingBackend(configuration: NFKMattingConfiguration(emitsMatte: true)) { plateTensor, _ in
            Self.appendConstantAlpha(0.5, to: plateTensor)
        }
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: plate]))

        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        let pixels = Self.straightPixels(of: output)
        XCTAssertEqual(Array(pixels[0..<4]), [255, 0, 0, 128], "straight red with a 0.5 matte")

        let matte = try Self.cgImage(result.output(forKey: NFKOutputMask))
        XCTAssertEqual(Array(Self.straightPixels(of: matte)[0..<4]), [128, 128, 128, 255], "matte on its own")
    }

    func testPremultiplyScalesTheForegroundByTheMatte() throws {
        try requireMLXRuntime()
        let plate = Self.solidImage(width: 1, height: 1, red: 255, green: 0, blue: 0)
        let config = NFKMattingConfiguration(imageOptions: NFKMLXImageOptions(premultiply: true))
        let backend = NFKMLXMattingBackend(configuration: config) { plateTensor, _ in
            Self.appendConstantAlpha(0.5, to: plateTensor)
        }
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: plate]))
        let pixels = Self.straightPixels(of: try Self.cgImage(result.output(forKey: NFKOutputImage)))
        XCTAssertEqual(Array(pixels[0..<4]), [128, 0, 0, 128], "red premultiplied by 0.5")
    }

    func testTilingProducesTheSameResultAsWholeImage() throws {
        try requireMLXRuntime()
        let plate = Self.solidImage(width: 4, height: 4, red: 120, green: 60, blue: 30)
        let forward: NFKMLXMattingBackend.Forward = { plateTensor, _ in Self.appendConstantAlpha(1, to: plateTensor) }
        let whole = try NFKMLXMattingBackend(configuration: NFKMattingConfiguration(tileSize: 0), forward: forward)
            .runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: plate]))
        let tiled = try NFKMLXMattingBackend(configuration: NFKMattingConfiguration(tileSize: 2), forward: forward)
            .runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: plate]))
        XCTAssertEqual(Self.straightPixels(of: try Self.cgImage(whole.output(forKey: NFKOutputImage))),
                       Self.straightPixels(of: try Self.cgImage(tiled.output(forKey: NFKOutputImage))))
    }

    // MARK: Helpers

    static func appendConstantAlpha(_ value: Float, to plate: MLXArray) -> MLXArray {
        let height = plate.shape[0]
        let width = plate.shape[1]
        let alpha = [Float](repeating: value, count: height * width)
            .withUnsafeBufferPointer { MLXArray($0, [height, width, 1]) }
        return concatenated([plate, alpha], axis: 2)
    }

    static func solidImage(width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8) -> CGImage {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for pixel in 0 ..< (width * height) {
            pixels[pixel * 4 + 0] = red; pixels[pixel * 4 + 1] = green
            pixels[pixel * 4 + 2] = blue; pixels[pixel * 4 + 3] = 255
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    static func cgImage(_ value: Any?) throws -> CGImage {
        guard let value, CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
            throw NFKMLXError.noOutput
        }
        return (value as! CGImage)
    }

    static func straightPixels(of image: CGImage) -> [UInt8] {
        [UInt8](image.dataProvider!.data! as Data)
    }
}
