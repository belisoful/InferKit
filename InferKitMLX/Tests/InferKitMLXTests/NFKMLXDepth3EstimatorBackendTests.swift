//
//  NFKMLXDepth3EstimatorBackendTests.swift
//  InferKitMLXTests
//
//  The Depth Anything 3 estimator behind the InferKit contract: one request carries the depth image, the
//  camera, and the ray map, and the camera is the one the estimator's own method reports.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDepth3EstimatorBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private static func gradient(width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for row in 0 ..< height {
            for column in 0 ..< width {
                let base = (row * width + column) * 4
                bytes[base] = UInt8(column * 255 / max(width - 1, 1))
                bytes[base + 1] = UInt8(row * 255 / max(height - 1, 1))
                bytes[base + 2] = 96
            }
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    func testOneRequestCarriesTheDepthTheCameraAndTheRays() throws {
        try requireMLXRuntime()
        let estimator = try NFKMLXDepth3Estimator.estimator(variant: .small, weightsURL: nil)
        let backend = estimator.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXDepth3Estimator.estimatorModelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputImage])

        let image = Self.gradient(width: 40, height: 30)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
        let depth = try XCTUnwrap(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(CFGetTypeID(depth as CFTypeRef), CGImage.typeID)
        XCTAssertEqual((depth as! CGImage).width, 40)
        XCTAssertEqual((depth as! CGImage).height, 30)

        let structured = try XCTUnwrap(result.output(forKey: NFKOutputStructured) as? [String: Any])
        let camera = try XCTUnwrap(structured["camera"] as? [String: Any])
        XCTAssertEqual((camera["rotation"] as? [NSNumber])?.count, 9)
        XCTAssertEqual((camera["translation"] as? [NSNumber])?.count, 3)
        for key in ["focalLengthX", "focalLengthY", "fieldOfViewX", "fieldOfViewY"] {
            XCTAssertNotNil(camera[key] as? Double, key)
        }

        let rays = try XCTUnwrap(structured["rays"] as? [String: Any])
        let width = try XCTUnwrap(rays["width"] as? Int)
        let height = try XCTUnwrap(rays["height"] as? Int)
        XCTAssertEqual(rays["channels"] as? Int, 6)
        XCTAssertEqual((rays["data"] as? Data)?.count, width * height * 6 * MemoryLayout<Float>.size)

        let confidence = try XCTUnwrap(structured["rayConfidence"] as? [String: Any])
        XCTAssertEqual(confidence["width"] as? Int, width)
        XCTAssertEqual(confidence["height"] as? Int, height)
        XCTAssertEqual(confidence["channels"] as? Int, 1)
        XCTAssertEqual((confidence["data"] as? Data)?.count, width * height * MemoryLayout<Float>.size)
    }

    func testTheBackendsCameraIsTheEstimatorsOwn() throws {
        try requireMLXRuntime()
        let estimator = try NFKMLXDepth3Estimator.estimator(variant: .small, weightsURL: nil)
        let image = Self.gradient(width: 48, height: 32)
        let result = try estimator.inferenceBackend()
            .runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
        let structured = try XCTUnwrap(result.output(forKey: NFKOutputStructured) as? [String: Any])
        let camera = try XCTUnwrap(structured["camera"] as? [String: Any])
        let expected = try estimator.camera(for: image)
        XCTAssertEqual(try XCTUnwrap(camera["focalLengthX"] as? Double), expected.focalLengthX, accuracy: 1e-3)
        XCTAssertEqual(try XCTUnwrap(camera["focalLengthY"] as? Double), expected.focalLengthY, accuracy: 1e-3)
        let rotation = try XCTUnwrap(camera["rotation"] as? [NSNumber]).map(\.doubleValue)
        for (actual, reference) in zip(rotation, expected.rotation.map(\.doubleValue)) {
            XCTAssertEqual(actual, reference, accuracy: 1e-4)
        }
    }

    func testARequestWithoutAnImageIsRefused() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXDepth3Estimator.estimator(variant: .small, weightsURL: nil).inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }
}
