//
//  NFKMLXRAFTTests.swift
//  InferKitMLXTests
//
//  Feature/context encoders, an all-pairs correlation pyramid + lookup, and a ConvGRU update loop.
//  These evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXRAFTTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    func testFlowFieldHasTwoChannelsAtInputSize() throws {
        try requireMLXRuntime()
        let net = NFKMLXRAFT.makeNet(iterations: 2)
        let flow = net.flow(Self.gradient(64, 64, shift: 0), Self.gradient(64, 64, shift: 2))
        eval(flow)
        XCTAssertEqual(flow.shape, [64, 64, 2], "a dense flow field at the input size")
    }

    func testACheckpointRoundTripReproducesTheFlow() throws {
        try requireMLXRuntime()
        let trained = NFKMLXRAFT.makeNet(iterations: 2)
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("raft-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXRAFT.makeNet(iterations: 2)
        try NFKMLXRAFT.loadWeights(into: loaded, from: url)

        let (a, b) = (Self.gradient(64, 64, shift: 0), Self.gradient(64, 64, shift: 3))
        let expected = trained.flow(a, b)
        let actual = loaded.flow(a, b)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }

    func testTheRegisteredBackendReturnsAPackedFlowMap() throws {
        try requireMLXRuntime()
        NFKMLXRAFT.register()
        let backend = try NFKMLXModelRegistry.backend(named: "raft", weightsURL: nil)
        let request = NFKInferenceRequest(inputs: ["frame0": Self.solid(64, 64, 60, 60, 60),
                                                   "frame1": Self.solid(64, 64, 200, 200, 200)])
        let output = try Self.cgImage(try backend.runInference(for: request).output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 64)
        XCTAssertEqual(output.height, 64)
    }

    static func gradient(_ height: Int, _ width: Int, shift: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let base = (y * width + x) * 3
                values[base] = Float((x + shift) % width) / Float(width)
                values[base + 1] = Float(y) / Float(height)
                values[base + 2] = 0.5
            }
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
}
