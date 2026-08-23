//
//  NFKMLXZeroDCETrainingTests.swift
//  InferKitMLXTests
//
//  Fine-tuning Zero-DCE. The four losses are checked on inputs whose correct score is known by
//  construction (a perfectly exposed batch, a gray image, a constant curve map), then the whole
//  customization path runs: train on unlabeled photos, save, and load the result through the same
//  public factory a converted checkpoint uses.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXZeroDCETrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func setUp() {
        super.setUp()
        NFKMLXRandom.seed(20_260_814)
    }

    // MARK: - The four losses

    func testExposureIsZeroForAPerfectlyExposedBatch() throws {
        try requireMLXRuntime()
        let exposed = MLXArray.full([1, 16, 16, 3], values: MLXArray(Float(0.6)))
        XCTAssertEqual(NFKMLXZeroDCELoss.exposure(exposed).item(Float.self), 0, accuracy: 1e-6)
    }

    func testExposureFallsAsABatchApproachesTheTargetLevel() throws {
        try requireMLXRuntime()
        let dark = MLXArray.full([1, 16, 16, 3], values: MLXArray(Float(0.1)))
        let closer = MLXArray.full([1, 16, 16, 3], values: MLXArray(Float(0.45)))
        XCTAssertLessThan(NFKMLXZeroDCELoss.exposure(closer).item(Float.self),
                          NFKMLXZeroDCELoss.exposure(dark).item(Float.self))
    }

    func testExposureFollowsTheChosenLevel() throws {
        try requireMLXRuntime()
        // The level is the consumer-facing knob, so a batch scores zero against its own brightness.
        let batch = MLXArray.full([1, 16, 16, 3], values: MLXArray(Float(0.25)))
        XCTAssertEqual(NFKMLXZeroDCELoss.exposure(batch, wellExposedLevel: 0.25).item(Float.self),
                       0, accuracy: 1e-6)
    }

    func testColorConstancyIsZeroForAGrayImageAndPositiveForATint() throws {
        try requireMLXRuntime()
        let gray = MLXArray.full([1, 8, 8, 3], values: MLXArray(Float(0.4)))
        XCTAssertEqual(NFKMLXZeroDCELoss.colorConstancy(gray).item(Float.self), 0, accuracy: 1e-6)

        var tinted = [Float](repeating: 0.4, count: 8 * 8 * 3)
        for i in stride(from: 0, to: tinted.count, by: 3) {
            tinted[i] = 0.9                                              // red cast
        }
        let cast = tinted.withUnsafeBufferPointer { MLXArray($0, [1, 8, 8, 3]) }
        XCTAssertGreaterThan(NFKMLXZeroDCELoss.colorConstancy(cast).item(Float.self), 0.1)
    }

    func testIlluminationSmoothnessIsZeroForAConstantCurveMap() throws {
        try requireMLXRuntime()
        let constant = MLXArray.full([1, 8, 8, 24], values: MLXArray(Float(0.3)))
        XCTAssertEqual(NFKMLXZeroDCELoss.illuminationSmoothness(constant).item(Float.self),
                       0, accuracy: 1e-6)
    }

    func testIlluminationSmoothnessPenalizesAVaryingCurveMap() throws {
        try requireMLXRuntime()
        // The banding has to be spatial: the loss measures neighboring positions, not neighboring
        // channels, so a pattern that alternates only along the last axis is genuinely smooth.
        var values = [Float](repeating: 0, count: 8 * 8 * 24)
        for row in 0 ..< 8 {
            for column in 0 ..< 8 {
                let value: Float = (row + column).isMultiple(of: 2) ? 1 : -1
                for channel in 0 ..< 24 {
                    values[(row * 8 + column) * 24 + channel] = value
                }
            }
        }
        let banded = values.withUnsafeBufferPointer { MLXArray($0, [1, 8, 8, 24]) }
        XCTAssertGreaterThan(NFKMLXZeroDCELoss.illuminationSmoothness(banded).item(Float.self), 0.5)
    }

    func testSpatialConsistencyIsZeroWhenTheImageIsUnchanged() throws {
        try requireMLXRuntime()
        let photo = Self.darkPhotos(count: 1, size: 16)
        XCTAssertEqual(NFKMLXZeroDCELoss.spatialConsistency(photo, original: photo).item(Float.self),
                       0, accuracy: 1e-6)
    }

    func testSpatialConsistencyPenalizesLostContrast() throws {
        try requireMLXRuntime()
        let photo = Self.darkPhotos(count: 1, size: 16)
        // A flat image has none of the original's neighbor structure left.
        let flattened = MLXArray.full([1, 16, 16, 3], values: photo.mean())
        XCTAssertGreaterThan(
            NFKMLXZeroDCELoss.spatialConsistency(flattened, original: photo).item(Float.self), 0)
    }

    // MARK: - The objective

    func testTheObjectiveScoresAndFallsWithTraining() throws {
        try requireMLXRuntime()
        let net = NFKMLXZeroDCENet(filters: 8)
        let photos = Self.darkPhotos(count: 2, size: 32)

        let history = try NFKMLXZeroDCE.fineTune(net, photos: { _ in photos },
                                                 optimizer: Adam(learningRate: 1e-3), steps: 60)

        XCTAssertEqual(history.count, 60)
        XCTAssertLessThan(history.last!, history.first!,
                          "the zero-reference objective fell: \(history.first!) -> \(history.last!)")
    }

    func testFineTuningBrightensTowardTheChosenLevel() throws {
        try requireMLXRuntime()
        let net = NFKMLXZeroDCENet(filters: 8)
        let photos = Self.darkPhotos(count: 2, size: 32)
        let before = net.enhance(photos[0]).mean().item(Float.self)

        try NFKMLXZeroDCE.fineTune(net, photos: { _ in photos },
                                   optimizer: Adam(learningRate: 1e-3), steps: 120)

        let after = net.enhance(photos[0]).mean().item(Float.self)
        XCTAssertGreaterThan(after, before, "training on dark photos alone brightened the model")
    }

    func testADivergingRunIsCaughtRatherThanSaved() throws {
        try requireMLXRuntime()
        // The default clip is what keeps a large learning rate from destroying the weights, so
        // removing it must be what fails.
        let photos = Self.darkPhotos(count: 1, size: 16)
        XCTAssertThrowsError(
            try NFKMLXZeroDCE.fineTune(NFKMLXZeroDCENet(filters: 8), photos: { _ in photos },
                                       optimizer: SGD(learningRate: 1e9), steps: 50,
                                       clipGradientNorm: nil))
    }

    // MARK: - The whole customization path

    func testAFineTunedCheckpointLoadsThroughThePublicFactory() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("zerodce-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXZeroDCE.network(weightsURL: nil)
        let photos = Self.darkPhotos(count: 1, size: 32)
        try NFKMLXZeroDCE.fineTune(net, photos: { _ in photos },
                                   optimizer: Adam(learningRate: 1e-3), steps: 20)
        try NFKMLXWeights.save(net, to: url)

        // The saved file is in the module's layout, so loadWeights must skip the transpose it applies
        // to a converted PyTorch checkpoint. Reproducing the forward exactly is what proves it did.
        let reloaded = try NFKMLXZeroDCE.network(weightsURL: url)
        let expected = net.enhance(photos[0])
        let actual = reloaded.enhance(photos[0])
        eval(expected, actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self),
                       "the fine-tuned checkpoint round-trips exactly")
    }

    func testAFineTunedBackendEnhancesMoreThanAnUntrainedOne() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("zerodce-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = NFKMLXZeroDCENet()
        let photos = Self.darkPhotos(count: 2, size: 32)
        try NFKMLXZeroDCE.fineTune(net, photos: { _ in photos },
                                   optimizer: Adam(learningRate: 1e-3), steps: 120)
        try NFKMLXWeights.save(net, to: url)

        // The end of the path: an ordinary backend, built by the shipped factory from the customized
        // weights, with no separate loading route.
        let plate = Self.darkImage(24)
        let tuned = try NFKMLXZeroDCE.backend(weightsURL: url)
        let output = try Self.cgImage(
            try tuned.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: plate]))
                .output(forKey: NFKOutputImage))

        XCTAssertEqual(output.width, 24, "enhancement keeps the input size")
        XCTAssertGreaterThan(try Self.meanLuma(output), try Self.meanLuma(plate),
                             "the customized backend brightened the plate")
    }

    // MARK: - Helpers

    /// Dark, textured photos `[N, size, size, 3]`, standing in for a consumer's own low-light shots.
    static func darkPhotos(count: Int, size: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: count * size * size * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 37) % 60) / 255.0
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [count, size, size, 3]) }
    }

    static func darkImage(_ side: Int) -> CGImage {
        var pixels = [UInt8](repeating: 255, count: side * side * 4)
        for i in 0 ..< (side * side) {
            pixels[i * 4] = UInt8((i * 37) % 60)
            pixels[i * 4 + 1] = UInt8((i * 23) % 60)
            pixels[i * 4 + 2] = UInt8((i * 11) % 60)
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)!
    }

    static func meanLuma(_ image: CGImage) throws -> Float {
        var pixels = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let context = try XCTUnwrap(CGContext(
            data: &pixels, width: image.width, height: image.height, bitsPerComponent: 8,
            bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))

        var total = 0
        for i in stride(from: 0, to: pixels.count, by: 4) {
            total += Int(pixels[i])
            total += Int(pixels[i + 1])
            total += Int(pixels[i + 2])
        }
        let samples = image.width * image.height * 3
        return Float(total) / Float(samples)
    }

    static func cgImage(_ value: Any?) throws -> CGImage {
        guard let value, CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
            throw NFKMLXError.noOutput
        }
        return (value as! CGImage)
    }
}
