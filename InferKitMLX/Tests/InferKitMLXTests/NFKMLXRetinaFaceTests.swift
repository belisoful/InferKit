//
//  NFKMLXRetinaFaceTests.swift
//  InferKitMLXTests
//
//  The detector's preprocessing, anchors, and decoding. Numeric agreement with facexlib is measured in
//  `NFKMLXReferenceParityTests`; what these cover is the surrounding contract, including the parts a
//  cosine cannot see.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXRetinaFaceTests: XCTestCase {

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func trainedNet() throws -> NFKMLXRetinaFaceNet {
        guard let weights = config["IK_VAL_RETINAFACE"] else { throw XCTSkip("set IK_VAL_RETINAFACE") }
        let net = NFKMLXRetinaFace.makeNet()
        try NFKMLXRetinaFace.loadWeights(into: net, from: URL(fileURLWithPath: weights))
        return net
    }

    private func portrait() throws -> CGImage {
        guard let path = config["IK_VAL_FACE"], FileManager.default.fileExists(atPath: path),
              let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw XCTSkip("set IK_VAL_FACE to a photograph containing a human face") }
        return image
    }

    // MARK: Preprocessing

    // The reference reads its frames through OpenCV, so its channel order is BGR and its mean is
    // subtracted in that order. Nothing in the values reveals the order, and feeding RGB produces
    // plausible-looking output with worse boxes — the failure this pins is a silent one.
    func testTheChannelOrderIsLoadBearing() throws {
        try requireMLXRuntime()
        let net = try trainedNet()
        let image = try portrait()

        let correct = NFKMLXRetinaFace.detect(image, using: net)
        XCTAssertFalse(correct.isEmpty, "the detector finds the face when fed BGR, as the reference is")

        // The same frame with the channels already swapped, so `prepared` swaps them back to RGB —
        // exactly what dropping the swap from `prepared` would do.
        let (bytes, width, height) = NFKMLXImageBridge.rgbaBytes(from: image,
                                                                 colorSpace: CGColorSpaceCreateDeviceRGB())
        let rgb = NFKMLXImageBridge.tensor(rgba: bytes, width: width, height: height, channels: 3)
        let swapped = concatenated([rgb[0..., 0..., 2 ..< 3], rgb[0..., 0..., 1 ..< 2],
                                    rgb[0..., 0..., 0 ..< 1]], axis: -1)
        let wrong = NFKMLXRetinaFace.detect(tensor: swapped, using: net)
        XCTAssertFalse(wrong.isEmpty, "the wrong order still detects a face — which is why it is a trap")

        // Confidence barely moves, so it cannot pin this. What moves is WHERE the face is: the boxes
        // and the landmarks, which is exactly what alignment consumes.
        let a = correct[0].boundingBox, b = try XCTUnwrap(wrong.first).boundingBox
        let intersection = a.intersection(b), union = a.union(b)
        let iou = (intersection.width * intersection.height) / (union.width * union.height)
        let shift = zip(correct[0].landmarks, try XCTUnwrap(wrong.first).landmarks)
            .map { hypot($0.x - $1.x, $0.y - $1.y) }.max() ?? 0

        print("VALIDATION retinaface channel order: confidence \(correct[0].confidence) vs "
              + "\(try XCTUnwrap(wrong.first).confidence), box IoU \(iou), worst landmark shift \(shift) px")
        XCTAssertLessThan(iou, 0.995, "the swapped order moves the box")
        XCTAssertGreaterThan(shift, 1, "the swapped order moves the landmarks")
    }

    func testPreparationRemovesTheReferenceMean() throws {
        try requireMLXRuntime()
        // A mid-grey frame: every channel is 128, so the prepared value is 128 minus that channel's
        // mean, in BGR order.
        let side = 8
        let grey = MLXArray.zeros([side, side, 3]) + (128.0 / 255.0)
        let prepared = NFKMLXRetinaFace.prepared(grey)
        eval(prepared)
        XCTAssertEqual(prepared.shape, [1, side, side, 3])

        let sample = prepared[0, 0, 0].asArray(Float.self)
        for (channel, mean) in NFKMLXRetinaFace.channelMean.enumerated() {
            XCTAssertEqual(sample[channel], 128 - mean, accuracy: 0.5,
                           "channel \(channel) carries the reference's mean")
        }
    }

    // MARK: Anchors

    func testAnchorsCoverEveryCellOfEveryLevel() {
        let configuration = NFKMLXRetinaFaceConfiguration()
        let (height, width) = (640, 640)
        let anchors = NFKMLXRetinaFace.anchors(height: height, width: width, configuration: configuration)

        let expected = configuration.steps.enumerated().reduce(0) { total, entry in
            let (level, step) = entry
            let rows = Int(ceil(Double(height) / Double(step)))
            let columns = Int(ceil(Double(width) / Double(step)))
            return total + rows * columns * configuration.minSizes[level].count
        }
        XCTAssertEqual(anchors.shape, [expected, 4])

        eval(anchors)
        let values = anchors.asArray(Float.self)
        // Centres are normalized, so they stay inside the frame; sizes are a fraction of it.
        for index in stride(from: 0, to: values.count, by: 4) {
            XCTAssertGreaterThan(values[index], 0)
            XCTAssertLessThan(values[index], 1)
            XCTAssertGreaterThan(values[index + 2], 0)
        }
    }

    // MARK: Suppression

    func testSuppressionKeepsTheBestOfAnOverlappingPair() {
        let boxes: [[Float]] = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
        let kept = NFKMLXRetinaFace.suppress(boxes: boxes, scores: [0.9, 0.95, 0.8], threshold: 0.4)
        XCTAssertEqual(kept.count, 2, "the overlapping pair collapses")
        XCTAssertEqual(kept.first, 1, "the higher score survives")
    }

    func testSuppressionKeepsDisjointBoxes() {
        let boxes: [[Float]] = [[0, 0, 10, 10], [50, 50, 60, 60]]
        let kept = NFKMLXRetinaFace.suppress(boxes: boxes, scores: [0.9, 0.8], threshold: 0.4)
        XCTAssertEqual(kept.count, 2)
    }

    // MARK: Loading

    func testTheCheckpointCoversEveryParameter() throws {
        try requireMLXRuntime()
        _ = try trainedNet()          // a strict load throws when a parameter is uncovered
    }

    func testTheClassifierAndCountersAreNotLoaded() {
        XCTAssertNil(NFKMLXRetinaFace.remapReferenceKey("body.fc.weight"))
        XCTAssertNil(NFKMLXRetinaFace.remapReferenceKey("body.stage1.0.1.num_batches_tracked"))
        XCTAssertEqual(NFKMLXRetinaFace.remapReferenceKey("body.stage1.0.0.weight"), "stem.conv.weight")
        XCTAssertEqual(NFKMLXRetinaFace.remapReferenceKey("body.stage1.1.0.weight"), "stage1.0.dwconv.weight")
        XCTAssertEqual(NFKMLXRetinaFace.remapReferenceKey("ClassHead.0.conv1x1.weight"),
                       "ClassHead.0.conv1x1.weight")
    }

    // MARK: The backend

    func testTheBackendReturnsDetections() throws {
        try requireMLXRuntime()
        guard let weights = config["IK_VAL_RETINAFACE"] else { throw XCTSkip("set IK_VAL_RETINAFACE") }
        let backend = try NFKMLXRetinaFace.backend(weightsURL: URL(fileURLWithPath: weights))
        let result = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: try portrait()]))
        let detections = try XCTUnwrap(result.output(forKey: NFKOutputDetections) as? [NFKDetection])
        XCTAssertFalse(detections.isEmpty)

        let face = try XCTUnwrap(detections.first)
        XCTAssertEqual(face.label, "face")
        // Boxes are normalized 0...1 with a top-left origin, as NFKMLXYOLO emits them.
        XCTAssertGreaterThan(face.boundingBox.minX, 0)
        XCTAssertLessThan(face.boundingBox.maxX, 1)
        XCTAssertLessThan(face.boundingBox.maxY, 1)
    }
}
