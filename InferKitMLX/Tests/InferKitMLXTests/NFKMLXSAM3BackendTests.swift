//
//  NFKMLXSAM3BackendTests.swift
//  InferKitMLXTests
//
//  SAM 3's image path behind the InferKit contract, at a tiny random geometry: a picture and a worded
//  concept become NFKDetection objects and a mask at the input's size, the score threshold is a request
//  key, and a request missing either input is refused.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSAM3BackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private static let queryCount = 8

    /// A CLIP vocabulary small enough to write, with its two markers, inside the tiny text tower's 64.
    private func tokenizer() throws -> NFKMLXSDPromptTokenizer {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("sam3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let vocabulary: [String: Int] = ["a</w>": 1, "cat</w>": 2, "<|startoftext|>": 10, "<|endoftext|>": 11]
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        return try NFKMLXSDPromptTokenizer(directoryURL: directory)
    }

    private func backend() throws -> any NFKInferenceBackend {
        MLXRandom.seed(17)
        var vision = NFKMLXSAM3Configuration.tiny
        vision.fpnHiddenSize = 32
        var text = NFKMLXSAM3TextConfiguration.tiny
        text.promptSize = 32
        let model = NFKMLXSAM3ImageModel(
            vision: NFKMLXSAM3VisionNet(vision), text: NFKMLXSAM3TextNet(text),
            detector: NFKMLXSAM3DetectorNet(NFKMLXSAM3DetectorConfiguration(
                hiddenSize: 32, intermediateSize: 48, heads: 2, encoderLayers: 1, decoderLayers: 2,
                queryCount: Self.queryCount)))
        return NFKMLXSAM3.inferenceBackend(model: model, tokenizer: try tokenizer())
    }

    private func plate(width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for index in stride(from: 0, to: bytes.count, by: 4) {
            bytes[index] = UInt8((index / 4) % 251)
            bytes[index + 1] = UInt8((index / 8) % 241)
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    func testAConceptBecomesDetectionsAndAMaskAtTheInputSize() throws {
        try requireMLXRuntime()
        let backend = try backend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXSAM3.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputImage, NFKInputPrompt])

        let request = NFKInferenceRequest(inputs: [NFKInputImage: plate(width: 40, height: 30), NFKInputPrompt: "a cat"],
                                          parameters: [NFKMLXSAM3ParameterKey.scoreThreshold: 0])
        let result = try backend.runInference(for: request)
        let detections = try XCTUnwrap(result.output(forKey: NFKOutputDetections) as? [NFKDetection])
        XCTAssertEqual(detections.count, Self.queryCount, "every query passes a threshold of zero")
        XCTAssertEqual(detections.map(\.confidence), detections.map(\.confidence).sorted(by: >), "most confident first")
        for detection in detections {
            XCTAssertEqual(detection.label, "a cat")
            XCTAssertGreaterThanOrEqual(detection.boundingBox.minX, 0)
            XCTAssertLessThanOrEqual(detection.boundingBox.maxY, 1)
        }
        let mask = try XCTUnwrap(result.output(forKey: NFKOutputMask))
        XCTAssertEqual(CFGetTypeID(mask as CFTypeRef), CGImage.typeID)
        XCTAssertEqual((mask as! CGImage).width, 40)
        XCTAssertEqual((mask as! CGImage).height, 30)
    }

    func testAThresholdNothingPassesReturnsNoInstanceAndNoMask() throws {
        try requireMLXRuntime()
        let request = NFKInferenceRequest(inputs: [NFKInputImage: plate(width: 20, height: 20), NFKInputPrompt: "cat"],
                                          parameters: [NFKMLXSAM3ParameterKey.scoreThreshold: 1])
        let result = try backend().runInference(for: request)
        XCTAssertEqual((result.output(forKey: NFKOutputDetections) as? [NFKDetection])?.count, 0)
        XCTAssertNil(result.output(forKey: NFKOutputMask))
    }

    func testARequestMissingItsImageOrItsConceptIsRefused() throws {
        try requireMLXRuntime()
        let backend = try backend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "cat"], parameters: [:])))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: plate(width: 20, height: 20)], parameters: [:])))
    }

    func testTheRegistryRefusesAMissingReleaseDirectory() throws {
        NFKMLXSAM3.register()
        XCTAssertThrowsError(try NFKMLXModelRegistry.backend(named: NFKMLXSAM3.modelName, weightsURL: nil))
    }
}
