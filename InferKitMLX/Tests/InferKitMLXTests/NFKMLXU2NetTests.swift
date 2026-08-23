//
//  NFKMLXU2NetTests.swift
//  InferKitMLXTests
//
//  A nested-U saliency network. The light configuration keeps the forward and round-trip fast. These
//  evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXU2NetTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    func testSaliencyIsAProbabilityMapAtInputSize() throws {
        try requireMLXRuntime()
        let net = NFKMLXU2Net.makeNet(light: true)
        let map = net.saliency(Self.tensor(height: 24, width: 32))
        eval(map)
        XCTAssertEqual(map.shape, [1, 24, 32, 1], "one saliency channel at input size")
        let values = map.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0, "sigmoid output")
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
    }

    func testTheMattingBackendReturnsForegroundPlusAlpha() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXU2Net.backend(variant: .light, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(16, 16, 200, 40, 40)]))

        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 16)
        let pixels = Self.pixels(of: output)
        XCTAssertEqual(Array(pixels[0 ..< 3]), [200, 40, 40], "foreground RGB is the straight plate")
        XCTAssertNotNil(result.output(forKey: NFKOutputMask), "the matte is emitted on its own")
    }

    func testACheckpointRoundTripReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = NFKMLXU2Net.makeNet(light: true)
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("u2net-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXU2Net.makeNet(light: true)
        try NFKMLXU2Net.loadWeights(into: loaded, from: url)

        let input = Self.tensor(height: 16, width: 16)
        let expected = trained.saliency(input)
        let actual = loaded.saliency(input)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }

    static func tensor(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 29) % 256) / 255.0
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [1, height, width, 3]) }
    }

    static func solid(_ width: Int, _ height: Int, _ red: UInt8, _ green: UInt8, _ blue: UInt8) -> CGImage {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for pixel in 0 ..< (width * height) {
            pixels[pixel * 4] = red; pixels[pixel * 4 + 1] = green
            pixels[pixel * 4 + 2] = blue; pixels[pixel * 4 + 3] = 255
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    static func cgImage(_ value: Any?) throws -> CGImage {
        guard let value, CFGetTypeID(value as CFTypeRef) == CGImage.typeID else { throw NFKMLXError.noOutput }
        return (value as! CGImage)
    }

    static func pixels(of image: CGImage) -> [UInt8] { [UInt8](image.dataProvider!.data! as Data) }
}
