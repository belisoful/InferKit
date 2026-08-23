//
//  NFKMLXRealESRGANTests.swift
//  InferKitMLXTests
//
//  Parameter names match the reference RRDBNet without a GPU. The forward and the weight round-trip
//  evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXRealESRGANTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func smallNet() -> NFKRealESRGANNet {
        NFKRealESRGANNet(features: 8, blocks: 1, growth: 4)
    }

    // MARK: Parameter names

    func testParameterNamesMatchTheReferenceCheckpoint() throws {
        try requireMLXRuntime()                                 // building a Conv2d initializes MLX RNG
        let names = Set(smallNet().parameters().flattened().map(\.0))
        for expected in ["conv_first.weight", "conv_first.bias",
                         "body.0.rdb1.conv1.weight", "body.0.rdb3.conv5.weight",
                         "conv_body.weight", "conv_up1.weight", "conv_up2.weight",
                         "conv_hr.weight", "conv_last.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // MARK: Forward and weight loading (needs MLX)

    func testTheGeneratorUpscalesByFourAndIsDeterministic() throws {
        try requireMLXRuntime()
        let net = smallNet()
        let input = Self.image(height: 4, width: 5)
        let a = net.upscale(input)
        let b = net.upscale(input)
        eval(a, b)
        XCTAssertEqual(a.shape, [16, 20, 3], "×4 upscale")
        XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self), "deterministic")
    }

    func testTheX2VariantUpscalesByTwoViaPixelUnshuffle() throws {
        try requireMLXRuntime()
        let net = NFKRealESRGANNet(features: 8, blocks: 1, growth: 4, scale: 2)
        let output = net.upscale(Self.image(height: 8, width: 8))
        eval(output)
        XCTAssertEqual(output.shape, [16, 16, 3], "input ÷2 by unshuffle, then ×4 body → net ×2")
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = smallNet()

        // Save the trained weights in PyTorch convolution layout [out, in, kH, kW], the on-disk shape
        // a real checkpoint uses, so the loader's transpose back to MLX layout is exercised.
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("rrdb-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = smallNet()                                 // different random weights
        try NFKMLXRealESRGAN.loadWeights(into: loaded, from: url)

        let input = Self.image(height: 3, width: 3)
        let expected = trained.upscale(input)
        let actual = loaded.upscale(input)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendUpscalesACGImage() throws {
        try requireMLXRuntime()
        NFKMLXRealESRGAN.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXRealESRGAN.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXRealESRGAN.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(2, 2)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 8, "2 → 8 (×4)")
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
