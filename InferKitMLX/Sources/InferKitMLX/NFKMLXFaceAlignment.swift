//
//  NFKMLXFaceAlignment.swift
//  InferKitMLX
//
//  Finding a face in a photograph, cropping it to the geometry a restoration model is trained on, and
//  putting the result back where it came from.
//
//  `NFKMLXCodeFormer` restores an ALIGNED 512×512 crop; producing that crop was the caller's problem.
//  The reference pipeline uses facexlib's RetinaFace for it. This uses **Vision**, which needs no
//  weights, no download, and no third-party code — the same rule the core applies to its own backends.
//  The consequence is stated rather than hidden: the crop is not byte-identical to the one RetinaFace
//  produces, so a restored photograph differs slightly from the reference pipeline's. What CodeFormer
//  does to a crop is unaffected, and that is what the parity record measures.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import Vision

/// A face found in an image, with the five points an alignment is computed from.
///
/// Coordinates are in image pixels with a top-left origin, which is the convention the InferKit value
/// types use. Vision reports a bottom-left origin, and the conversion happens at detection. An `@objc`
/// class (immutable, like the core's `NFKKeypoint`/`NFKDetection`), so Objective-C reads the landmarks
/// a face detector returns, not only its box.
@objc(NFKFaceObservation)
public final class NFKFaceObservation: NSObject, @unchecked Sendable {
    /// The face's bounds in image pixels.
    @objc public let boundingBox: CGRect
    /// Left eye, right eye, nose, left mouth corner, right mouth corner — the reference's order.
    public let landmarks: [CGPoint]
    /// Vision's confidence in the detection.
    @objc public let confidence: Float

    public init(boundingBox: CGRect, landmarks: [CGPoint], confidence: Float) {
        self.boundingBox = boundingBox
        self.landmarks = landmarks
        self.confidence = confidence
        super.init()
    }

    private func landmark(_ index: Int) -> CGPoint { landmarks.indices.contains(index) ? landmarks[index] : .zero }
    /// The five landmarks by name, for Objective-C (the array is Swift-only). Image pixels, top-left origin.
    @objc public var leftEye: CGPoint { landmark(0) }
    @objc public var rightEye: CGPoint { landmark(1) }
    @objc public var nose: CGPoint { landmark(2) }
    @objc public var leftMouthCorner: CGPoint { landmark(3) }
    @objc public var rightMouthCorner: CGPoint { landmark(4) }
}

/// Something that finds faces in an image.
///
/// Two ship: `NFKMLXVisionFaceDetector` needs no weights, and `NFKMLXRetinaFaceDetector` is the
/// detector the CodeFormer reference pipeline runs, so a crop taken through it is the crop the
/// reference produces.
public protocol NFKMLXFaceDetecting: Sendable {
    func faces(in image: CGImage) throws -> [NFKFaceObservation]
}

/// Detection through Vision. No weights, no download; not the reference's detector.
public struct NFKMLXVisionFaceDetector: NFKMLXFaceDetecting {
    public init() {}
    public func faces(in image: CGImage) throws -> [NFKFaceObservation] {
        try NFKMLXFaceAlignment.detectFaces(in: image)
    }
}

/// Face detection, alignment, and paste-back.
public enum NFKMLXFaceAlignment {

    /// The five-point template a 512×512 restoration crop is aligned to.
    ///
    /// These are facexlib's `face_template` coordinates, which CodeFormer and the other FFHQ-trained
    /// restorers share. Aligning to anything else puts the eyes and mouth where the model does not
    /// expect them, which degrades the restoration without looking like an error.
    public static let template512: [CGPoint] = [
        CGPoint(x: 192.98138, y: 239.94708),        // left eye
        CGPoint(x: 318.90277, y: 240.19360),        // right eye
        CGPoint(x: 256.63416, y: 314.01935),        // nose
        CGPoint(x: 201.26117, y: 371.41043),        // left mouth corner
        CGPoint(x: 313.08905, y: 371.15118),        // right mouth corner
    ]

    /// Finds every face in `image`, most confident first.
    public static func detectFaces(in image: CGImage) throws -> [NFKFaceObservation] {
        let request = VNDetectFaceLandmarksRequest()
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        let size = CGSize(width: image.width, height: image.height)

        return (request.results ?? [])
            .compactMap { observation -> NFKFaceObservation? in
                guard let landmarks = observation.landmarks,
                      let points = fivePoints(from: landmarks, face: observation.boundingBox, in: size)
                else { return nil }
                let box = VNImageRectForNormalizedRect(observation.boundingBox,
                                                       Int(size.width), Int(size.height))
                // Vision's origin is bottom-left; flip the box to the top-left convention.
                let flipped = CGRect(x: box.minX, y: size.height - box.maxY,
                                     width: box.width, height: box.height)
                return NFKFaceObservation(boundingBox: flipped, landmarks: points,
                                          confidence: observation.confidence)
            }
            .sorted { $0.confidence > $1.confidence }
    }

    /// Reduces Vision's landmark regions to the reference's five points.
    ///
    /// Vision reports each feature as a contour rather than a single point, so an eye is its centroid
    /// and a mouth corner is the extreme of the outer lip contour along x.
    private static func fivePoints(from landmarks: VNFaceLandmarks2D, face: CGRect,
                                   in size: CGSize) -> [CGPoint]? {
        func imagePoints(_ region: VNFaceLandmarkRegion2D?) -> [CGPoint]? {
            guard let region, region.pointCount > 0 else { return nil }
            return region.normalizedPoints.map { point in
                // Landmarks are normalized inside the face box; lift them to the image, then flip.
                let x = (face.minX + point.x * face.width) * size.width
                let y = (face.minY + point.y * face.height) * size.height
                return CGPoint(x: x, y: size.height - y)
            }
        }
        func centroid(_ points: [CGPoint]) -> CGPoint {
            let sum = points.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
        }

        guard let leftEye = imagePoints(landmarks.leftEye),
              let rightEye = imagePoints(landmarks.rightEye),
              let lips = imagePoints(landmarks.outerLips)
        else { return nil }

        // The nose contour's lowest point is the tip; Vision has no single nose-tip landmark.
        let nose = imagePoints(landmarks.nose).map { points in
            points.max { $0.y < $1.y } ?? centroid(points)
        } ?? centroid(lips)

        guard let leftCorner = lips.min(by: { $0.x < $1.x }),
              let rightCorner = lips.max(by: { $0.x < $1.x })
        else { return nil }

        return [centroid(leftEye), centroid(rightEye), nose, leftCorner, rightCorner]
    }

    /// The least-squares similarity transform taking `source` onto `destination`.
    ///
    /// @discussion A similarity keeps the face's shape — uniform scale, rotation, translation, no
    /// shear — which is what an alignment must do: a full affine would stretch the face to hit the
    /// template exactly and hand the model a distorted subject. Solved in closed form: with both point
    /// sets centered, the rotation and scale are one complex multiply, so `a` and `b` below are its
    /// real and imaginary parts.
    public static func similarityTransform(from source: [CGPoint],
                                           to destination: [CGPoint]) -> CGAffineTransform? {
        guard source.count == destination.count, source.count >= 2 else { return nil }
        let n = CGFloat(source.count)
        let sourceMean = source.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        let destinationMean = destination.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        let sc = CGPoint(x: sourceMean.x / n, y: sourceMean.y / n)
        let dc = CGPoint(x: destinationMean.x / n, y: destinationMean.y / n)

        var norm: CGFloat = 0, dot: CGFloat = 0, cross: CGFloat = 0
        for (s, d) in zip(source, destination) {
            let (sx, sy) = (s.x - sc.x, s.y - sc.y)
            let (dx, dy) = (d.x - dc.x, d.y - dc.y)
            norm += sx * sx + sy * sy
            dot += sx * dx + sy * dy
            cross += sx * dy - sy * dx
        }
        guard norm > 0 else { return nil }
        let a = dot / norm, b = cross / norm

        return CGAffineTransform(a: a, b: b, c: -b, d: a,
                                 tx: dc.x - (a * sc.x - b * sc.y),
                                 ty: dc.y - (b * sc.x + a * sc.y))
    }

    /// Crops and aligns a face to `side`×`side`, returning the crop and the transform that made it.
    ///
    /// Keep the transform: pasting the restored face back needs its inverse.
    public static func alignedCrop(from image: CGImage, face: NFKFaceObservation,
                                   side: Int = 512) -> (image: CGImage, transform: CGAffineTransform)? {
        let scale = CGFloat(side) / 512
        let template = template512.map { CGPoint(x: $0.x * scale, y: $0.y * scale) }
        guard let transform = similarityTransform(from: face.landmarks, to: template),
              let context = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8,
                                      bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        context.interpolationQuality = .high

        // Both the landmarks and the template are top-left; CoreGraphics is bottom-left. Flipping the
        // context lets the transform be applied exactly as solved — and then the DRAW needs its own
        // flip, because `draw(_:in:)` orients an image for a y-up space and would otherwise land it
        // upside down in this one. Without that second flip the crop is mirrored about the image's
        // centre, which lands on whatever sits opposite the face rather than on the face.
        context.translateBy(x: 0, y: CGFloat(side))
        context.scaleBy(x: 1, y: -1)
        context.concatenate(transform)
        context.translateBy(x: 0, y: CGFloat(image.height))
        context.scaleBy(x: 1, y: -1)
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))

        guard let cropped = context.makeImage() else { return nil }
        return (cropped, transform)
    }

    /// Draws `restored` back into `original` through the inverse of the alignment transform.
    ///
    /// @discussion The restored crop is composited with a soft-edged mask, because a hard edge at the
    /// crop boundary is visible on a photograph even when the restoration itself is good.
    public static func pasteBack(_ restored: CGImage, into original: CGImage,
                                 transform: CGAffineTransform, featherFraction: CGFloat = 0.06) -> CGImage? {
        let (width, height) = (original.width, original.height)
        // A degenerate alignment (collinear landmarks) has no inverse to paste through.
        guard transform.a != 0 || transform.b != 0,
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let masked = feathered(restored, fraction: featherFraction)
        else { return nil }
        context.interpolationQuality = .high

        // Same two flips as the crop: one to work in top-left coordinates, one per draw so each image
        // lands upright rather than mirrored.
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        context.saveGState()
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        context.draw(original, in: CGRect(x: 0, y: 0, width: width, height: height))
        context.restoreGState()

        context.concatenate(transform.inverted())
        context.translateBy(x: 0, y: CGFloat(masked.height))
        context.scaleBy(x: 1, y: -1)
        context.draw(masked, in: CGRect(x: 0, y: 0, width: masked.width, height: masked.height))
        return context.makeImage()
    }

    /// Applies a linear alpha ramp around the border of `image`.
    private static func feathered(_ image: CGImage, fraction: CGFloat) -> CGImage? {
        let (width, height) = (image.width, image.height)
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = context.data else { return nil }

        let border = max(1, Int(CGFloat(min(width, height)) * fraction))
        let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let edge = min(min(x, width - 1 - x), min(y, height - 1 - y))
                guard edge < border else { continue }
                let alpha = CGFloat(edge) / CGFloat(border)
                let offset = (y * width + x) * 4
                for channel in 0 ..< 4 {
                    pixels[offset + channel] = UInt8(CGFloat(pixels[offset + channel]) * alpha)
                }
            }
        }
        return context.makeImage()
    }
}

/// Holds a photograph transform for capture in the backend's `@Sendable` body.
private final class NFKMLXPhotoTransformBox: @unchecked Sendable {
    let transform: (CGImage) throws -> CGImage
    init(_ transform: @escaping (CGImage) throws -> CGImage) { self.transform = transform }
}

/// An image backend whose transform works on `CGImage` rather than tensors.
///
/// Face restoration on a photograph is detection, alignment, and compositing around a model, and all
/// three are CoreGraphics work. Bridging the whole frame to a tensor and back would only add a
/// conversion at each end.
public final class NFKMLXPhotoFaceBackend: NSObject, NFKInferenceBackend {

    private let identifier: String
    private let box: NFKMLXPhotoTransformBox

    init(identifier: String, transform: @escaping (CGImage) throws -> CGImage) {
        self.identifier = identifier
        self.box = NFKMLXPhotoTransformBox(transform)
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let value = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedInput
        }
        guard CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
            throw NFKMLXError.unsupportedInput
        }
        let image = value as! CGImage
        return NFKInferenceResult(outputs: [NFKOutputImage: try box.transform(image)])
    }
}
