//
//  NFKMLXTrainingDataTests.swift
//  InferKitMLXTests
//
//  Turning app data into training tensors, and drawing batches from it. The label-map conversion is
//  the one that has to be exact: it inverts the encoding the segmentation backends emit, so a mask an
//  app writes back must round-trip to the class index it came from.
//

import XCTest
import CoreGraphics
import MLX
@testable import InferKitMLX

final class NFKMLXTrainingDataTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    // MARK: - Tensors

    func testAnImageBecomesAZeroToOneTensor() throws {
        try requireMLXRuntime()
        let tensor = try NFKMLXTrainingData.tensor(Self.solid(8, red: 255, green: 0, blue: 0))
        XCTAssertEqual(tensor.shape, [8, 8, 3])
        let values = tensor.asArray(Float.self)
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-3, "red channel is full")
        XCTAssertEqual(values[1], 0.0, accuracy: 1e-3, "green channel is empty")
    }

    func testABatchStacksImages() throws {
        try requireMLXRuntime()
        let batch = try NFKMLXTrainingData.batch([Self.solid(8), Self.solid(8), Self.solid(8)])
        XCTAssertEqual(batch.shape, [3, 8, 8, 3])
    }

    func testABatchOfMixedSizesFailsRatherThanResizingSilently() throws {
        try requireMLXRuntime()
        // Whether to crop or scale changes what the model learns, so the choice stays with the caller.
        XCTAssertThrowsError(try NFKMLXTrainingData.batch([Self.solid(8), Self.solid(16)])) { error in
            guard case NFKMLXError.trainingDataMismatch(let detail) = error else {
                return XCTFail("expected trainingDataMismatch, got \(error)")
            }
            XCTAssertTrue(detail.contains("8×8"), "names both sizes: \(detail)")
        }
    }

    func testAnEmptyBatchFails() throws {
        try requireMLXRuntime()
        XCTAssertThrowsError(try NFKMLXTrainingData.batch([]))
    }

    func testAGrayscaleImageBecomesAMatte() throws {
        try requireMLXRuntime()
        let matte = try NFKMLXTrainingData.matte(Self.solid(4, red: 128, green: 128, blue: 128))
        XCTAssertEqual(matte.shape, [4, 4])
        XCTAssertEqual(matte.mean().item(Float.self), 128.0 / 255.0, accuracy: 1e-3)
    }

    // MARK: - Labels

    func testALabelMapInvertsTheSegmentationEncoding() throws {
        try requireMLXRuntime()
        // The backends emit index / (classCount - 1); a mask written back through that encoding must
        // recover the same index.
        let classCount = 5
        for index in 0 ..< classCount {
            let level = UInt8(round(Double(index) / Double(classCount - 1) * 255))
            let image = Self.solid(4, red: level, green: level, blue: level)
            let labels = try NFKMLXTrainingData.labels(image, classCount: classCount)
            XCTAssertEqual(labels.shape, [4, 4])
            XCTAssertEqual(labels.max().item(Int32.self), Int32(index), "class \(index) round-tripped")
        }
    }

    func testALabelMapIsAnIntegerType() throws {
        try requireMLXRuntime()
        let labels = try NFKMLXTrainingData.labels(Self.solid(4), classCount: 3)
        XCTAssertEqual(labels.dtype, .int32, "cross-entropy takes indices, not floats")
    }

    // MARK: - Sampling

    func testTheSamplerCoversEveryExampleInAPass() throws {
        let sampler = NFKMLXBatchSampler(count: 6, batchSize: 2, seed: 1)
        let drawn = (0 ..< 3).flatMap { sampler.indices(forStep: $0) }
        XCTAssertEqual(Set(drawn), Set(0 ..< 6), "one pass sees the whole dataset")
        XCTAssertEqual(drawn.count, 6, "and sees each example once")
    }

    func testTheSamplerReshufflesBetweenPasses() throws {
        let sampler = NFKMLXBatchSampler(count: 8, batchSize: 8, seed: 1)
        // Cycling in a fixed order lets the optimizer chase the sequence rather than the data.
        XCTAssertNotEqual(sampler.indices(forStep: 0), sampler.indices(forStep: 1))
    }

    func testTheSamplerIsRepeatableForASeed() throws {
        let first = NFKMLXBatchSampler(count: 10, batchSize: 3, seed: 42)
        let second = NFKMLXBatchSampler(count: 10, batchSize: 3, seed: 42)
        for step in 0 ..< 8 {
            XCTAssertEqual(first.indices(forStep: step), second.indices(forStep: step))
        }
        XCTAssertNotEqual(first.indices(forStep: 0),
                          NFKMLXBatchSampler(count: 10, batchSize: 3, seed: 7).indices(forStep: 0))
    }

    func testTheSamplerHandlesASingleExample() throws {
        let sampler = NFKMLXBatchSampler(count: 1, batchSize: 1, seed: 3)
        XCTAssertEqual(sampler.indices(forStep: 0), [0])
        XCTAssertEqual(sampler.indices(forStep: 5), [0], "a one-example dataset still draws")
    }

    func testTheSamplerHandlesABatchLargerThanTheDataset() throws {
        let sampler = NFKMLXBatchSampler(count: 3, batchSize: 5, seed: 3)
        let drawn = sampler.indices(forStep: 0)
        XCTAssertEqual(drawn.count, 5)
        XCTAssertTrue(drawn.allSatisfy { (0 ..< 3).contains($0) }, "indices stay in range: \(drawn)")
    }

    // MARK: - Helpers

    static func solid(_ side: Int, red: UInt8 = 40, green: UInt8 = 40, blue: UInt8 = 40) -> CGImage {
        var pixels = [UInt8](repeating: 255, count: side * side * 4)
        for pixel in 0 ..< (side * side) {
            pixels[pixel * 4] = red
            pixels[pixel * 4 + 1] = green
            pixels[pixel * 4 + 2] = blue
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)!
    }
}
