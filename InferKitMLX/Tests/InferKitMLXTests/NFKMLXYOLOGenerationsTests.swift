//
//  NFKMLXYOLOGenerationsTests.swift
//  InferKitMLXTests
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXYOLOGenerationsTests: XCTestCase {

    private var config: [String: String] { NFKMLXValidationConfig.environment }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// Every release the port covers, with the keys its checkpoint and record live under.
    private static let releases: [(NFKMLXYOLORelease, String, String)] = {
        let names: [(NFKMLXYOLORelease, String)] = [
            (.v9Tiny, "YOLOV9T"), (.v9Small, "YOLOV9S"), (.v9Medium, "YOLOV9M"), (.v9Compact, "YOLOV9C"),
            (.v9Extended, "YOLOV9E"),
            (.v10Nano, "YOLOV10N"), (.v10Small, "YOLOV10S"), (.v10Medium, "YOLOV10M"),
            (.v10Balanced, "YOLOV10B"), (.v10Large, "YOLOV10L"), (.v10ExtraLarge, "YOLOV10X"),
            (.v11Nano, "YOLO11N"), (.v11Small, "YOLO11S"), (.v11Medium, "YOLO11M"),
            (.v11Large, "YOLO11L"), (.v11ExtraLarge, "YOLO11X"),
            (.v12Nano, "YOLO12N"), (.v12Small, "YOLO12S"), (.v12Medium, "YOLO12M"),
            (.v12Large, "YOLO12L"), (.v12ExtraLarge, "YOLO12X"),
            (.v26Nano, "YOLO26N"), (.v26Small, "YOLO26S"), (.v26Medium, "YOLO26M"),
            (.v26Large, "YOLO26L"), (.v26ExtraLarge, "YOLO26X"),
        ]
        return names.map { ($0.0, "IK_VAL_\($0.1)", "IK_PARITY_\($0.1)") }
    }()

    // Every released checkpoint loads into the graph the port builds for it. The loader refuses a
    // partial cover, so this pins each generation's channel counts, repeat counts and per-size flags
    // before any numeric comparison runs.
    func testEveryReleaseLoadsItsCheckpoint() throws {
        try requireMLXRuntime()
        var checked = 0
        for (release, weightsKey, _) in Self.releases {
            guard let path = config[weightsKey], FileManager.default.fileExists(atPath: path) else {
                print("SKIP \(NFKMLXYOLOGenerations.modelName(for: release)): no \(weightsKey)")
                continue
            }
            let net = NFKMLXYOLOGenerations.makeNet(release)
            try NFKMLXYOLOGenerations.loadWeights(into: net, from: URL(fileURLWithPath: path))
            print("VALIDATION structure \(NFKMLXYOLOGenerations.modelName(for: release)): loaded")
            checked += 1
        }
        XCTAssertGreaterThan(checked, 0, "at least one release was available to check")
    }

    // Each release against ultralytics' own model on the released weights, over the full
    // pre-suppression tensor: the boxes in pixels and the class scores through a sigmoid, for every
    // anchor across the three strides. The record also carries three backbone stages, so a mismatch
    // says which stage rather than only that the output moved.
    func testEveryGenerationMatchesTheReference() throws {
        try requireMLXRuntime()
        var checked = 0
        for (release, weightsKey, parityKey) in Self.releases {
            let name = NFKMLXYOLOGenerations.modelName(for: release)
            guard let parityPath = config[parityKey], FileManager.default.fileExists(atPath: parityPath),
                  let weightsPath = config[weightsKey], FileManager.default.fileExists(atPath: weightsPath) else {
                print("SKIP \(name): no \(parityKey)")
                continue
            }
            let arrays = try loadArrays(url: URL(fileURLWithPath: parityPath))
            let net = NFKMLXYOLOGenerations.makeNet(release)
            try NFKMLXYOLOGenerations.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))

            let plate = try XCTUnwrap(arrays["input_image"]).expandedDimensions(axis: 0).asType(.float32)
            for stage in ["stage2", "stage4", "stage9"] {
                guard let recorded = arrays[stage], let index = Int(stage.dropFirst("stage".count)) else { continue }
                let produced = net.stageOutput(plate, at: index)
                eval(produced)
                let value = cosine(produced, recorded)
                XCTAssertGreaterThan(value, 0.9999, "\(name): the \(stage) seam matches the reference")
            }

            let predictions = net.predictions(plate)
            eval(predictions)
            let reference = try XCTUnwrap(arrays["output"])
            XCTAssertEqual(predictions.shape, reference.shape, "\(name): same pre-suppression shape")
            let boxes = cosine(predictions[0..., 0 ..< 4], reference[0..., 0 ..< 4])
            let classes = cosine(predictions[0..., 4...], reference[0..., 4...])
            print("VALIDATION PARITY \(name): box cosine \(boxes), class cosine \(classes)")
            XCTAssertGreaterThan(boxes, 0.9999, "\(name): the decoded boxes match the reference")
            XCTAssertGreaterThan(classes, 0.9999, "\(name): the class scores match the reference")
            checked += 1
        }
        XCTAssertGreaterThan(checked, 0, "at least one release was available to check")
    }

    // Detections on a NON-SQUARE frame, through the public backend, against ultralytics' own
    // `predict`. The square-plate comparison cannot see this path: letterboxing is an identity there.
    // The end-to-end generations matter most here, because their consumer path differs from every
    // earlier one — the head decodes corners rather than a center and an extent, and no suppression
    // runs.
    func testGenerationDetectionsMatchTheReferenceOnANonSquareFrame() throws {
        try requireMLXRuntime()
        var checked = 0
        for (release, weightsKey, parityKey) in [
            (NFKMLXYOLORelease.v11Nano, "IK_VAL_YOLO11N", "IK_PARITY_YOLO11N_DETECTIONS"),
            (.v10Nano, "IK_VAL_YOLOV10N", "IK_PARITY_YOLOV10N_DETECTIONS"),
            (.v26Nano, "IK_VAL_YOLO26N", "IK_PARITY_YOLO26N_DETECTIONS"),
        ] {
            let name = NFKMLXYOLOGenerations.modelName(for: release)
            guard let parityPath = config[parityKey], FileManager.default.fileExists(atPath: parityPath),
                  let weightsPath = config[weightsKey], FileManager.default.fileExists(atPath: weightsPath) else {
                print("SKIP \(name) detections: no \(parityKey)")
                continue
            }
            let arrays = try loadArrays(url: URL(fileURLWithPath: parityPath))
            let plate = try XCTUnwrap(arrays["plate"])
            let referenceBoxes = try XCTUnwrap(arrays["output"]).asArray(Float.self)
            let referenceClasses = try XCTUnwrap(arrays["classes"]).asArray(Int32.self)

            let backend = try NFKMLXYOLOGenerations.backend(release: release,
                                                            weightsURL: URL(fileURLWithPath: weightsPath),
                                                            labels: nil)
            let result = try backend.runInference(
                for: NFKInferenceRequest(inputs: [NFKInputImage: try Self.image(from: plate)]))
            let ours = try XCTUnwrap(result.detections)

            XCTAssertEqual(ours.count, referenceClasses.count, "\(name): same number of detections")
            var worstOverlap = 1.0
            for (index, detection) in ours.enumerated() where index < referenceClasses.count {
                XCTAssertEqual(Int32(detection.classIndex), referenceClasses[index],
                               "\(name): detection \(index) is the reference's class")
                let reference = CGRect(x: CGFloat(referenceBoxes[index * 4]),
                                       y: CGFloat(referenceBoxes[index * 4 + 1]),
                                       width: CGFloat(referenceBoxes[index * 4 + 2] - referenceBoxes[index * 4]),
                                       height: CGFloat(referenceBoxes[index * 4 + 3] - referenceBoxes[index * 4 + 1]))
                worstOverlap = min(worstOverlap,
                                   Double(NFKMLXYOLONet.intersectionOverUnion(detection.boundingBox, reference)))
            }
            print("VALIDATION PARITY \(name) (non-square): \(ours.count) detections, worst box IoU \(worstOverlap)")
            XCTAssertGreaterThan(worstOverlap, 0.98, "\(name): every box lands where the reference put it")
            checked += 1
        }
        XCTAssertGreaterThan(checked, 0, "at least one release was available to check")
    }

    /// The record's plate as a `CGImage`, which is what the public backend reads.
    private static func image(from tensor: MLXArray) throws -> CGImage {
        try NFKMLXImageBridge.cgImage(from: tensor, options: NFKMLXImageOptions())
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.reshaped([-1]).asType(.float32), y = b.reshaped([-1]).asType(.float32)
        return Double((x * y).sum().item(Float.self))
            / Double((sqrt((x * x).sum()) * sqrt((y * y).sum())).item(Float.self))
    }

    private func loadArrays(url: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: url)
    }
}
