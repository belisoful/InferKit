//
//  NFKMLXStyleTransferTests.swift
//  InferKitMLXTests
//
//  Parameter names match the reference TransformerNet without a GPU. The forward and the weight
//  round-trip evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXStyleTransferTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func smallNet() -> NFKStyleTransferNet {
        NFKStyleTransferNet(base: 4)
    }

    // MARK: Parameter names

    func testParameterNamesMatchTheReferenceCheckpoint() throws {
        try requireMLXRuntime()                                 // building a Conv2d initializes MLX RNG
        let names = Set(smallNet().parameters().flattened().map(\.0))
        for expected in ["conv1.conv2d.weight", "conv1.conv2d.bias", "in1.weight", "in1.bias",
                         "conv2.conv2d.weight", "conv3.conv2d.weight",
                         "res1.conv1.conv2d.weight", "res1.in1.weight", "res5.conv2.conv2d.weight",
                         "deconv1.conv2d.weight", "in4.weight", "deconv2.conv2d.weight", "deconv3.conv2d.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // MARK: Forward and weight loading (needs MLX)

    func testStylizingPreservesSizeAndStaysInRangeAndIsDeterministic() throws {
        try requireMLXRuntime()
        let net = smallNet()
        let input = Self.image(height: 8, width: 12)              // even dims: the two ÷2 stages divide cleanly
        let a = net.stylize(input)
        let b = net.stylize(input)
        eval(a, b)
        XCTAssertEqual(a.shape, [8, 12, 3], "output matches input size")
        let values = a.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0, "clipped to 0...1")
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
        XCTAssertEqual(values, b.asArray(Float.self), "deterministic")
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = smallNet()

        // Save the trained weights in PyTorch convolution layout [out, in, kH, kW], the on-disk shape a
        // real checkpoint uses, so the loader's transpose back to MLX layout is exercised.
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("style-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = smallNet()                                 // different random weights
        try NFKMLXStyleTransfer.loadWeights(into: loaded, from: url)

        let input = Self.image(height: 8, width: 8)
        let expected = trained.stylize(input)
        let actual = loaded.stylize(input)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendStylizesACGImage() throws {
        try requireMLXRuntime()
        NFKMLXStyleTransfer.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXStyleTransfer.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXStyleTransfer.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(8, 8)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 8, "stylization keeps the input size")
        XCTAssertEqual(output.height, 8)
    }

    // MARK: Helpers

    static func image(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 37) % 256) / 255.0             // deterministic, spatially varied
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [height, width, 3]) }
    }

    static func solid(_ width: Int, _ height: Int) -> CGImage {
        let pixels = [UInt8](repeating: 128, count: width * height * 4)
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
