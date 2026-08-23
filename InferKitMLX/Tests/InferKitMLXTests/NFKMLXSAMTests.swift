//
//  NFKMLXSAMTests.swift
//  InferKitMLXTests
//
//  ViT encoder + prompt encoder + two-way-transformer mask decoder. A tiny configuration keeps it
//  fast. These evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXSAMTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyConfiguration() -> NFKMLXSAMConfiguration {
        var configuration = NFKMLXSAMConfiguration()
        configuration.imageSize = 64                            // grid 4
        configuration.encoderEmbed = 32
        configuration.encoderDepth = 1
        configuration.encoderHeads = 2
        configuration.embedDim = 32
        configuration.decoderDepth = 2
        configuration.decoderHeads = 2
        configuration.encoderDepth = 2
        configuration.windowSize = 2                            // grid 4 → 2×2 windows
        configuration.globalAttnIndexes = [1]                  // block 0 windowed, block 1 global
        return configuration
    }

    func testSegmentProducesAProbabilityMaskAtQuarterResolution() throws {
        try requireMLXRuntime()
        let net = NFKMLXSAM.makeNet(tinyConfiguration())
        let mask = net.segment(Self.image(64), pointX: 0.5, pointY: 0.5)
        eval(mask)
        XCTAssertEqual(mask.shape, [1, 16, 16, 1], "4× the 4×4 token grid")
        let values = mask.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0)
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
    }

    func testACheckpointRoundTripReproducesTheMask() throws {
        try requireMLXRuntime()
        let trained = NFKMLXSAM.makeNet(tinyConfiguration())
        // Mirror how a released checkpoint actually stores each tensor: convolutions in PyTorch's
        // `[out, in, kH, kW]`, transposed convolutions in `[in, out, kH, kW]`, and `pos_embed` already
        // channels-last (SAM's ViT is NHWC), so the loader's per-kind handling round-trips.
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            guard value.ndim == 4, !key.hasSuffix("pos_embed") else { return (key, value) }
            if key.hasSuffix("up1.weight") || key.hasSuffix("up2.weight") {
                return (key, value.transposed(3, 0, 1, 2))
            }
            return (key, value.transposed(0, 3, 1, 2))
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sam-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXSAM.makeNet(tinyConfiguration())
        try NFKMLXSAM.loadWeights(into: loaded, from: url)

        let image = Self.image(64)
        let expected = trained.segment(image, pointX: 0.3, pointY: 0.7)
        let actual = loaded.segment(image, pointX: 0.3, pointY: 0.7)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }

    static func image(_ size: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: size * size * 3)
        for y in 0 ..< size {
            for x in 0 ..< size {
                let base = (y * size + x) * 3
                values[base] = Float(x) / Float(size)
                values[base + 1] = Float(y) / Float(size)
                values[base + 2] = 0.5
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [1, size, size, 3]) }
    }
}
