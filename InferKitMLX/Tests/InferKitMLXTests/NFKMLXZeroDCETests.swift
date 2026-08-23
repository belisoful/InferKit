//
//  NFKMLXZeroDCETests.swift
//  InferKitMLXTests
//
//  The DCE-Net curve estimator. The forward and the weight round-trip evaluate MLX arrays, so they
//  skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXZeroDCETests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func smallNet() -> NFKMLXZeroDCENet {
        NFKMLXZeroDCENet(filters: 8)
    }

    func testParameterNamesMatchTheReferenceCheckpoint() throws {
        try requireMLXRuntime()
        let names = Set(smallNet().parameters().flattened().map(\.0))
        for expected in ["e_conv1.weight", "e_conv1.bias", "e_conv4.weight", "e_conv5.weight",
                         "e_conv6.weight", "e_conv7.weight", "e_conv7.bias"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testEnhancementKeepsSizeAndRangeAndIsDeterministic() throws {
        try requireMLXRuntime()
        let net = smallNet()
        let input = Self.image(height: 12, width: 16)
        let a = net.enhance(input)
        let b = net.enhance(input)
        eval(a, b)
        XCTAssertEqual(a.shape, [12, 16, 3], "output matches input size")
        let values = a.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0, "clipped to 0...1")
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
        XCTAssertEqual(values, b.asArray(Float.self), "deterministic")
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = smallNet()
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("zerodce-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = smallNet()
        try NFKMLXZeroDCE.loadWeights(into: loaded, from: url)

        let input = Self.image(height: 8, width: 8)
        let expected = trained.enhance(input)
        let actual = loaded.enhance(input)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendEnhancesACGImage() throws {
        try requireMLXRuntime()
        NFKMLXZeroDCE.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXZeroDCE.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXZeroDCE.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(8, 8)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 8, "enhancement keeps the input size")
    }

    // MARK: Helpers

    static func image(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 17) % 128) / 255.0                   // deliberately dark
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [height, width, 3]) }
    }

    static func solid(_ width: Int, _ height: Int) -> CGImage {
        let pixels = [UInt8](repeating: 40, count: width * height * 4)
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
}
