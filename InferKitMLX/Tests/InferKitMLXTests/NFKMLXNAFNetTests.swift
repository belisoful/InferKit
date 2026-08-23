//
//  NFKMLXNAFNetTests.swift
//  InferKitMLXTests
//
//  A U-shaped restoration network. A tiny configuration keeps the forward and round-trip fast. These
//  evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXNAFNetTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyConfiguration() -> NFKMLXNAFNetConfiguration {
        var configuration = NFKMLXNAFNetConfiguration()
        configuration.width = 4
        configuration.encoderBlocks = [1, 1]
        configuration.middleBlocks = 1
        configuration.decoderBlocks = [1, 1]
        return configuration
    }

    func testRestoreReturnsTheInputSizeEvenWhenNotDivisible() throws {
        try requireMLXRuntime()
        let net = NFKMLXNAFNet.makeNet(tinyConfiguration())
        let output = net.restore(Self.image(height: 10, width: 14))   // 10, 14 are not multiples of 4
        eval(output)
        XCTAssertEqual(output.shape, [10, 14, 3], "padded to a multiple of 2^levels, then cropped back")
    }

    func testACheckpointRoundTripReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = NFKMLXNAFNet.makeNet(tinyConfiguration())
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("nafnet-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXNAFNet.makeNet(tinyConfiguration())
        try NFKMLXNAFNet.loadWeights(into: loaded, from: url)

        let input = Self.image(height: 8, width: 8)
        let expected = trained.restore(input)
        let actual = loaded.restore(input)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }

    func testTheRegisteredBackendRestoresACGImage() throws {
        try requireMLXRuntime()
        NFKMLXNAFNet.register()
        let backend = try NFKMLXModelRegistry.backend(named: "nafnet", weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(16, 16)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 16, "restoration keeps the input size")
        XCTAssertEqual(output.height, 16)
    }

    static func image(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 31) % 256) / 255.0
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [height, width, 3]) }
    }

    static func solid(_ width: Int, _ height: Int) -> CGImage {
        let pixels = [UInt8](repeating: 120, count: width * height * 4)
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
}
