//
//  NFKMLXYOLOTests.swift
//  InferKitMLXTests
//
//  The YOLOv8 detector. The forward, decode, and weight round-trip evaluate MLX arrays, so those skip
//  under `swift test`; non-max suppression and the key remap are pure Swift and run anywhere.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXYOLOTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXYOLONet {
        NFKMLXYOLONet(.tiny)
    }

    // MARK: Pure-Swift NMS (no GPU)

    func testNonMaxSuppressionKeepsTheBestAndDropsOverlapsOfTheSameClass() {
        let strong = (classIndex: 0, confidence: Float(0.9), rect: CGRect(x: 0.1, y: 0.1, width: 0.4, height: 0.4))
        let overlap = (classIndex: 0, confidence: Float(0.5), rect: CGRect(x: 0.12, y: 0.12, width: 0.4, height: 0.4))
        let otherClass = (classIndex: 1, confidence: Float(0.6), rect: CGRect(x: 0.1, y: 0.1, width: 0.4, height: 0.4))
        let far = (classIndex: 0, confidence: Float(0.7), rect: CGRect(x: 0.6, y: 0.6, width: 0.3, height: 0.3))

        let kept = NFKMLXYOLONet.nonMaxSuppress([strong, overlap, otherClass, far], iouThreshold: 0.45)
        XCTAssertEqual(kept.count, 3, "the weaker same-class overlap is suppressed; the other-class and far boxes stay")
        XCTAssertFalse(kept.contains { $0.confidence == 0.5 }, "the suppressed box is the 0.5 overlap")
    }

    // MARK: Parameter names

    func testParameterNamesFollowTheReferenceLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["conv0.conv.weight", "conv0.bn.running_mean", "c2f2.cv1.conv.weight",
                         "c2f2.m.0.cv1.conv.weight", "sppf.cv1.conv.weight", "c2f12.cv2.bn.weight",
                         "conv16.conv.weight", "detect.cv2.0.conv1.conv.weight",
                         "detect.cv2.0.out.bias", "detect.cv3.2.out.weight", "detect.dfl.conv.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheReferenceKeyRemapCoversTheStagesAndTheHead() {
        XCTAssertEqual(NFKMLXYOLO.remapReferenceKey("model.0.conv.weight"), "conv0.conv.weight")
        XCTAssertEqual(NFKMLXYOLO.remapReferenceKey("model.9.cv2.bn.running_var"), "sppf.cv2.bn.running_var")
        XCTAssertEqual(NFKMLXYOLO.remapReferenceKey("model.12.m.0.cv1.conv.weight"), "c2f12.m.0.cv1.conv.weight")
        XCTAssertEqual(NFKMLXYOLO.remapReferenceKey("model.22.cv2.0.0.conv.weight"),
                       "detect.cv2.0.conv1.conv.weight")
        XCTAssertEqual(NFKMLXYOLO.remapReferenceKey("model.22.cv3.1.2.bias"), "detect.cv3.1.out.bias")
        XCTAssertEqual(NFKMLXYOLO.remapReferenceKey("model.22.dfl.conv.weight"), "detect.dfl.conv.weight")
    }

    // MARK: Forward and decode (needs MLX)

    func testPredictionsCoverThreeScalesWithProbabilityClasses() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        net.train(false)
        let predicted = net.predictions(Self.image(height: 24, width: 24))
        eval(predicted)
        let resolution = NFKMLXYOLOConfiguration.tiny.inputResolution
        let anchors = (resolution / 8) * (resolution / 8) + (resolution / 16) * (resolution / 16)
            + (resolution / 32) * (resolution / 32)
        XCTAssertEqual(predicted.shape, [anchors, 4 + NFKMLXYOLOConfiguration.tiny.classCount],
                       "one row per anchor across the strides 8, 16, and 32")
        let classes = predicted[0..., 4...].asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(classes.min()), 0, "sigmoid outputs")
        XCTAssertLessThanOrEqual(try XCTUnwrap(classes.max()), 1)
    }

    func testDetectionsHaveValidClassesAndNormalizedBoxes() throws {
        try requireMLXRuntime()
        let net = tinyNet()                                     // tiny threshold is 0, so every anchor is a candidate
        net.train(false)
        let detections = net.detect(Self.image(height: 24, width: 24), labels: ["a", "b", "c"])
        XCTAssertFalse(detections.isEmpty, "a zero threshold keeps candidates through NMS")
        for detection in detections {
            XCTAssertLessThan(detection.classIndex, NFKMLXYOLOConfiguration.tiny.classCount)
            XCTAssertGreaterThanOrEqual(detection.boundingBox.minX, 0)
            XCTAssertLessThanOrEqual(detection.boundingBox.width, 1)
            XCTAssertNotNil(detection.label, "labels were supplied")
        }
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("yolo-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = tinyNet()
        try NFKMLXYOLO.loadWeights(into: loaded, from: url)
        trained.train(false)
        loaded.train(false)

        let image = Self.image(height: 24, width: 24)
        let expected = trained.predictions(image)
        let actual = loaded.predictions(image)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheBackendReturnsDetectionsForACGImage() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXYOLO.backend(weightsURL: nil, labels: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(32, 32)]))
        XCTAssertNotNil(result.detections, "detections output present (possibly empty)")
    }

    // MARK: Helpers

    static func image(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 37) % 256) / 255.0
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
}
