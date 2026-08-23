//
//  NFKMLXResampleTests.swift
//  InferKitMLXTests
//
//  The shared resampling and pooling helpers. They evaluate MLX arrays, so these tests skip under
//  `swift test` and run under `xcodebuild test`.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXResampleTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func ramp(_ height: Int, _ width: Int, channels: Int = 1) -> MLXArray {
        repeated(MLXArray((0 ..< height * width).map { Float($0) }).reshaped([1, height, width, 1]),
                 count: channels, axis: 3)
    }

    private func values(_ x: MLXArray) -> [Float] {
        eval(x)
        return x.asArray(Float.self)
    }

    // The values a `MaxPool2d(3, stride: 2, padding: 1)` produces, spelled out, so the helper stays
    // proven the day mlx-swift repairs its own padding and the layer could be called directly.
    func testMaxPoolingWithABorderMatchesTheReferenceWindows() throws {
        try requireMLXRuntime()
        let pooled = NFKMLXResample.maxPooled(ramp(4, 4), kernel: 3, stride: 2, padding: 1)
        XCTAssertEqual(pooled.shape, [1, 2, 2, 1])
        XCTAssertEqual(values(pooled), [5, 7, 13, 15])
    }

    // The signature of the framework's pad-width bug is a pooled channel axis, so a tensor whose
    // channel count differs from its width shows it.
    func testMaxPoolingWithABorderLeavesBatchAndChannelsAlone() throws {
        try requireMLXRuntime()
        let pooled = NFKMLXResample.maxPooled(ramp(8, 8, channels: 3), kernel: 3, stride: 2, padding: 1)
        XCTAssertEqual(pooled.shape, [1, 4, 4, 3])
    }

    func testMaxPoolingWithoutABorderWindowsTheInputUnchanged() throws {
        try requireMLXRuntime()
        let pooled = NFKMLXResample.maxPooled(ramp(4, 4), kernel: 2, stride: 2)
        XCTAssertEqual(pooled.shape, [1, 2, 2, 1])
        XCTAssertEqual(values(pooled), [5, 7, 13, 15])
    }

    // The zero border counts toward the mean, as `count_include_pad` does, so the corners read low.
    func testAveragePoolingCountsTheBorderTowardEachWindow() throws {
        try requireMLXRuntime()
        let pooled = NFKMLXResample.averagePooled(MLXArray.ones([1, 4, 4, 1]), kernel: 2, stride: 2, padding: 1)
        XCTAssertEqual(pooled.shape, [1, 3, 3, 1])
        XCTAssertEqual(values(pooled), [0.25, 0.5, 0.25,
                                        0.5, 1.0, 0.5,
                                        0.25, 0.5, 0.25])
    }

    func testSpatialPaddingBordersHeightAndWidthOnly() throws {
        try requireMLXRuntime()
        let padded = NFKMLXResample.spatiallyPadded(MLXArray.ones([1, 4, 6, 3]), 2, value: 0)
        XCTAssertEqual(padded.shape, [1, 8, 10, 3])
    }

    func testReflectionPaddingMirrorsWithoutRepeatingTheEdge() throws {
        try requireMLXRuntime()
        let padded = NFKMLXResample.reflectPadded(ramp(3, 3), 1)
        XCTAssertEqual(padded.shape, [1, 5, 5, 1])
        XCTAssertEqual(values(padded), [4, 3, 4, 5, 4,
                                        1, 0, 1, 2, 1,
                                        4, 3, 4, 5, 4,
                                        7, 6, 7, 8, 7,
                                        4, 3, 4, 5, 4])
    }
}
