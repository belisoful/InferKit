//
//  NFKMLXFaceAlignmentTests.swift
//  InferKitMLXTests
//
//  Detection, alignment, and paste-back — the step between a photograph and the aligned 512×512 crop
//  `NFKMLXCodeFormer` restores. The transform is solved in closed form, so most of this needs no model
//  and no GPU: a similarity transform is verified by constructing one and recovering it.
//

import XCTest
import CoreGraphics
import ImageIO
import InferKit
@testable import InferKitMLX

final class NFKMLXFaceAlignmentTests: XCTestCase {

    // MARK: The similarity solve

    // A similarity applied to known points must be recovered exactly from those points.
    func testTheTransformRecoversAKnownSimilarity() throws {
        let known = CGAffineTransform(a: 1.4, b: 0.6, c: -0.6, d: 1.4, tx: 17, ty: -9)
        let source = [CGPoint(x: 10, y: 20), CGPoint(x: 90, y: 25), CGPoint(x: 50, y: 60),
                      CGPoint(x: 20, y: 95), CGPoint(x: 80, y: 92)]
        let destination = source.map { $0.applying(known) }

        let solved = try XCTUnwrap(NFKMLXFaceAlignment.similarityTransform(from: source, to: destination))
        for (component, expected) in [(solved.a, known.a), (solved.b, known.b), (solved.c, known.c),
                                      (solved.d, known.d), (solved.tx, known.tx), (solved.ty, known.ty)] {
            XCTAssertEqual(component, expected, accuracy: 1e-6)
        }
    }

    // The solve is least-squares, so points that fit no similarity exactly still produce the best one
    // rather than failing — and it stays a similarity: uniform scale and rotation, never shear.
    func testTheTransformStaysASimilarityUnderNoise() throws {
        let source = [CGPoint(x: 0, y: 0), CGPoint(x: 100, y: 0), CGPoint(x: 50, y: 50),
                      CGPoint(x: 10, y: 100), CGPoint(x: 90, y: 103)]
        let destination = [CGPoint(x: 5, y: 2), CGPoint(x: 203, y: -1), CGPoint(x: 99, y: 96),
                           CGPoint(x: 17, y: 210), CGPoint(x: 181, y: 199)]
        let solved = try XCTUnwrap(NFKMLXFaceAlignment.similarityTransform(from: source, to: destination))

        XCTAssertEqual(solved.a, solved.d, accuracy: 1e-9, "uniform scale on both axes")
        XCTAssertEqual(solved.b, -solved.c, accuracy: 1e-9, "a rotation, not a shear")
    }

    func testTooFewPointsHaveNoTransform() {
        XCTAssertNil(NFKMLXFaceAlignment.similarityTransform(from: [.zero], to: [.zero]))
        XCTAssertNil(NFKMLXFaceAlignment.similarityTransform(from: [], to: []))
    }

    // Coincident points span nothing, so no scale or rotation is recoverable.
    func testDegenerateLandmarksHaveNoTransform() {
        let same = [CGPoint](repeating: CGPoint(x: 4, y: 4), count: 5)
        XCTAssertNil(NFKMLXFaceAlignment.similarityTransform(from: same, to: NFKMLXFaceAlignment.template512))
    }

    func testTheTemplateIsTheReferenceGeometry() {
        let template = NFKMLXFaceAlignment.template512
        XCTAssertEqual(template.count, 5)
        // The eyes sit above the mouth and the nose between them; a template stored upside down would
        // align every face inverted while still "working".
        XCTAssertLessThan(template[0].y, template[2].y, "eyes above the nose")
        XCTAssertLessThan(template[2].y, template[3].y, "nose above the mouth")
        XCTAssertLessThan(template[0].x, template[1].x, "left eye left of the right eye")
    }

    // MARK: Detection on a real photograph

    // xcodebuild sanitizes the runner's environment, so configuration comes from the same JSON file
    // the validation suites read, with the process environment as a fallback.
    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    private func validationImage() throws -> CGImage {
        // A face image is user-supplied: IK_VAL_IMAGE is the general validation plate and is not a
        // portrait, so these tests take IK_VAL_FACE and skip rather than silently testing nothing.
        guard let path = config["IK_VAL_FACE"], FileManager.default.fileExists(atPath: path),
              let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw XCTSkip("set IK_VAL_FACE to a photograph containing a human face") }
        return image
    }

    func testAFaceIsDetectedAndAlignedToTheTemplate() throws {
        let image = try validationImage()
        let faces = try NFKMLXFaceAlignment.detectFaces(in: image)
        try XCTSkipIf(faces.isEmpty, "the validation photograph has no detectable face")

        let face = faces[0]
        XCTAssertEqual(face.landmarks.count, 5)
        XCTAssertTrue(face.boundingBox.width > 0 && face.boundingBox.height > 0)
        XCTAssertLessThan(face.landmarks[0].x, face.landmarks[1].x, "left eye left of the right eye")

        let (crop, transform) = try XCTUnwrap(NFKMLXFaceAlignment.alignedCrop(from: image, face: face))
        XCTAssertEqual(crop.width, 512)
        XCTAssertEqual(crop.height, 512)

        // Set IK_VAL_OUTDIR to write the crop out and look at it: the numeric tolerance below says
        // the landmarks are near the template, and only the picture says the face is upright.
        if let directory = config["IK_VAL_OUTDIR"] {
            let url = URL(fileURLWithPath: directory).appendingPathComponent("face-aligned.png")
            if let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) {
                CGImageDestinationAddImage(destination, crop, nil)
                CGImageDestinationFinalize(destination)
            }
        }

        // The transform is what defines the crop: it must land the detected landmarks on the template.
        for (landmark, target) in zip(face.landmarks, NFKMLXFaceAlignment.template512) {
            let mapped = landmark.applying(transform)
            XCTAssertEqual(mapped.x, target.x, accuracy: 40, "landmarks land near the template")
            XCTAssertEqual(mapped.y, target.y, accuracy: 40)
        }
    }

    // The landmark assertion above cannot catch a bad crop: the transform maps landmarks onto the
    // template BY CONSTRUCTION, so it stays true even when the drawing lands somewhere else entirely.
    // Only the crop's CONTENT can say the right region was taken — an aligned face is a face, and
    // detecting one in the crop is what proves it. This failed on a vertically mirrored draw, which
    // produced a crop of the subject's chest while every number stayed in tolerance.
    func testTheAlignedCropContainsTheFace() throws {
        let image = try validationImage()
        let faces = try NFKMLXFaceAlignment.detectFaces(in: image)
        try XCTSkipIf(faces.isEmpty, "the validation photograph has no detectable face")
        let (crop, _) = try XCTUnwrap(NFKMLXFaceAlignment.alignedCrop(from: image, face: faces[0]))

        let inCrop = try NFKMLXFaceAlignment.detectFaces(in: crop)
        XCTAssertFalse(inCrop.isEmpty, "the aligned crop contains a detectable face")

        // And it is the framing the template describes: the face fills the crop and is roughly centred.
        let found = try XCTUnwrap(inCrop.first)
        let side = CGFloat(crop.width)
        XCTAssertGreaterThan(found.boundingBox.width / side, 0.3, "the face fills the crop")
        XCTAssertEqual(found.boundingBox.midX / side, 0.5, accuracy: 0.2, "horizontally centred")
        XCTAssertEqual(found.boundingBox.midY / side, 0.5, accuracy: 0.25, "vertically centred")
    }

    func testPasteBackReturnsTheOriginalGeometry() throws {
        let image = try validationImage()
        let faces = try NFKMLXFaceAlignment.detectFaces(in: image)
        try XCTSkipIf(faces.isEmpty, "the validation photograph has no detectable face")

        let (crop, transform) = try XCTUnwrap(NFKMLXFaceAlignment.alignedCrop(from: image, face: faces[0]))
        let composited = try XCTUnwrap(NFKMLXFaceAlignment.pasteBack(crop, into: image, transform: transform))
        XCTAssertEqual(composited.width, image.width, "the frame keeps its size")
        XCTAssertEqual(composited.height, image.height)

        if let directory = config["IK_VAL_OUTDIR"] {
            let url = URL(fileURLWithPath: directory).appendingPathComponent("face-pasted.png")
            if let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) {
                CGImageDestinationAddImage(destination, composited, nil)
                CGImageDestinationFinalize(destination)
            }
        }

        // Round-tripping the crop back through the inverse must reconstruct the frame, so the face
        // still detects in the same place. A mirrored or displaced paste destroys that.
        let faceAfter = try NFKMLXFaceAlignment.detectFaces(in: composited)
        XCTAssertFalse(faceAfter.isEmpty, "the composited frame still shows the face")
        let before = faces[0].boundingBox, after = try XCTUnwrap(faceAfter.first).boundingBox
        XCTAssertEqual(after.midX, before.midX, accuracy: before.width * 0.25, "the face stays put")
        XCTAssertEqual(after.midY, before.midY, accuracy: before.height * 0.25)
    }

    // MARK: The two detectors

    // Both detectors have to find the same face and produce a usable crop. They will NOT agree
    // exactly — Vision derives five points from contours where RetinaFace regresses them directly —
    // and measuring that disagreement is the point: it is the cost of the zero-download path, stated
    // as a number rather than a caveat.
    func testRetinaFaceAndVisionAgreeOnTheFace() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
        guard let weights = config["IK_VAL_RETINAFACE"] else { throw XCTSkip("set IK_VAL_RETINAFACE") }
        let image = try validationImage()

        let vision = try NFKMLXVisionFaceDetector().faces(in: image)
        let retina = try NFKMLXRetinaFace.detector(weightsURL: URL(fileURLWithPath: weights))
            .faces(in: image)
        try XCTSkipIf(vision.isEmpty || retina.isEmpty, "both detectors need to find the face")

        let visionBox = vision[0].boundingBox, retinaBox = retina[0].boundingBox
        let intersection = visionBox.intersection(retinaBox)
        let overlap = (intersection.width * intersection.height)
            / (visionBox.union(retinaBox).width * visionBox.union(retinaBox).height)
        XCTAssertGreaterThan(overlap, 0.3, "the two detectors find the same face")

        // Landmark disagreement, reported so a change in either path is visible.
        let spread = zip(vision[0].landmarks, retina[0].landmarks)
            .map { hypot($0.x - $1.x, $0.y - $1.y) }
        let worst = spread.max() ?? 0
        print("VALIDATION detectors: box IoU \(overlap), worst landmark disagreement \(worst) px "
              + "over a \(image.width)×\(image.height) frame")

        // And each produces a crop containing a face, which is what the restorer needs.
        for face in [vision[0], retina[0]] {
            let (crop, _) = try XCTUnwrap(NFKMLXFaceAlignment.alignedCrop(from: image, face: face))
            XCTAssertFalse(try NFKMLXFaceAlignment.detectFaces(in: crop).isEmpty,
                           "the aligned crop contains a face")
        }
    }
}
