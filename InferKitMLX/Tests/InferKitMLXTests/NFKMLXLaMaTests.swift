//
//  NFKMLXLaMaTests.swift
//  InferKitMLXTests
//
//  FFC-ResNet with an FFT spectral branch. A tiny configuration keeps the forward and weight
//  round-trip fast. These evaluate MLX arrays, so they skip without a Metal library for MLX (see
//  Tools/mlx-metallib.sh).
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXLaMaTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tinyConfiguration() -> NFKMLXLaMaConfiguration {
        var configuration = NFKMLXLaMaConfiguration()
        configuration.baseChannels = 8
        configuration.downsampling = 2
        configuration.blocks = 1
        configuration.ratio = 0.5
        return configuration
    }

    func testInpaintKeepsTheUnmaskedRegionAndReturnsInputSize() throws {
        try requireMLXRuntime()
        let net = NFKMLXLaMa.makeNet(tinyConfiguration())
        let image = Self.image(height: 16, width: 16)
        let mask = Self.leftHalfMask(height: 16, width: 16)     // regenerate the left, keep the right
        let output = net.inpaint(image, mask: mask)
        eval(output)
        XCTAssertEqual(output.shape, [16, 16, 3], "returns at input size")

        let outValues = output.asArray(Float.self)
        let inValues = image.asArray(Float.self)
        for row in 0 ..< 16 {                                    // right half (mask 0) equals the source
            for col in 8 ..< 16 {
                let index = (row * 16 + col) * 3
                XCTAssertEqual(outValues[index], inValues[index], accuracy: 1e-5)
            }
        }
    }

    /// The backend inpaints through the network as built, so a BatchNorm left in training mode would
    /// normalize each plate by its own statistics and overwrite the released ones.
    func testANetworkIsBuiltInEvaluationModeAndKeepsItsStatistics() throws {
        try requireMLXRuntime()
        let net = NFKMLXLaMaNet(tinyConfiguration())
        let normalizations = net.leafModules().flattened().filter { $0.1 is BatchNorm }
        XCTAssertFalse(normalizations.isEmpty)
        XCTAssertTrue(normalizations.allSatisfy { !$0.1.training })
        func statistics() -> [Float] {
            net.parameters().flattened().filter { $0.0.hasSuffix("running_mean") }.flatMap { $0.1.asArray(Float.self) }
        }
        let before = statistics()
        eval(net.inpaint(Self.image(height: 16, width: 16), mask: Self.leftHalfMask(height: 16, width: 16)))
        XCTAssertEqual(statistics(), before, "inference reads the running statistics without folding into them")
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = NFKMLXLaMa.makeNet(tinyConfiguration())
        // A transposed convolution is stored `[in, out, kH, kW]` upstream and a forward one
        // `[out, in, kH, kW]`, so the round trip has to write each in its own layout — the loader
        // undoes them separately.
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value -> (String, MLXArray) in
            guard value.ndim == 4 else { return (key, value) }
            return (key, key.hasPrefix("up.") ? value.transposed(3, 0, 1, 2) : value.transposed(0, 3, 1, 2))
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("lama-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXLaMa.makeNet(tinyConfiguration())
        try NFKMLXLaMa.loadWeights(into: loaded, from: url)

        let image = Self.image(height: 16, width: 16)
        let mask = Self.leftHalfMask(height: 16, width: 16)
        let expected = trained.inpaint(image, mask: mask)
        let actual = loaded.inpaint(image, mask: mask)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }

    static func image(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 41) % 256) / 255.0
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [height, width, 3]) }
    }

    static func leftHalfMask(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width)
        for row in 0 ..< height {
            for col in 0 ..< width where col < width / 2 {
                values[row * width + col] = 1
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [height, width, 1]) }
    }
}
