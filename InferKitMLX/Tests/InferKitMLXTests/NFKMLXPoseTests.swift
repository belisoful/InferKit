//
//  NFKMLXPoseTests.swift
//  InferKitMLXTests
//
//  The SimpleBaseline pose network. The forward, heatmap decode, and weight round-trip evaluate MLX
//  arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXPoseTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXPoseNet {
        let net = NFKMLXPoseNet(.tiny)
        net.train(false)
        return net
    }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["backbone.conv1.weight", "backbone.bn1.weight", "backbone.layer1.0.conv1.weight",
                         "backbone.layer1.0.downsample_conv.weight", "backbone.layer4.0.conv3.weight",
                         "deconv.0.weight", "deconv_bn.0.weight", "final.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testHeatmapsAreAtQuarterResolutionWithOneChannelPerJoint() throws {
        try requireMLXRuntime()
        let heatmaps = tinyNet().heatmaps(MLXArray.zeros([1, 64, 32, 3]))
        eval(heatmaps)
        XCTAssertEqual(heatmaps.shape, [1, 16, 8, NFKMLXPoseConfiguration.tiny.keypointCount],
                       "stride-4 heatmaps, one channel per joint")
    }

    // The released checkpoint prefixes the backbone and the head, and addresses the head's
    // (convolution, normalization, activation) triples positionally.
    func testTheReferenceHeadRemapsFromItsPositionalSequential() {
        XCTAssertEqual(NFKMLXPose.remapReferenceKey("head.deconv_layers.0.weight"), "deconv.0.weight")
        XCTAssertEqual(NFKMLXPose.remapReferenceKey("head.deconv_layers.4.running_mean"), "deconv_bn.1.running_mean")
        XCTAssertEqual(NFKMLXPose.remapReferenceKey("head.deconv_layers.6.weight"), "deconv.2.weight")
        XCTAssertEqual(NFKMLXPose.remapReferenceKey("head.final_layer.bias"), "final.bias")
        XCTAssertEqual(NFKMLXPose.remapReferenceKey("backbone.layer2.0.downsample.1.weight"),
                       "backbone.layer2.0.downsample_bn.weight")
        XCTAssertEqual(NFKMLXPose.remapReferenceKey("deconv.0.weight"), "deconv.0.weight",
                       "a module key passes through, so a round-trip reloads")
    }

    func testEstimateReturnsOneKeypointPerJointWithNormalizedPositions() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let keypoints = net.estimate(Self.image(height: 40, width: 30), jointNames: ["a", "b", "c", "d"])
        XCTAssertEqual(keypoints.count, NFKMLXPoseConfiguration.tiny.keypointCount)
        for (index, keypoint) in keypoints.enumerated() {
            XCTAssertEqual(keypoint.index, index)
            XCTAssertEqual(keypoint.name, ["a", "b", "c", "d"][index])
            XCTAssertGreaterThanOrEqual(keypoint.position.x, 0)
            XCTAssertLessThanOrEqual(keypoint.position.x, 1)
            XCTAssertGreaterThanOrEqual(keypoint.position.y, 0)
            XCTAssertLessThanOrEqual(keypoint.position.y, 1)
            XCTAssertGreaterThanOrEqual(keypoint.confidence, 0)
            XCTAssertLessThanOrEqual(keypoint.confidence, 1)
        }
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("pose-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXPoseNet(.tiny)
        loaded.train(false)
        try NFKMLXPose.loadWeights(into: loaded, from: url)

        let image = MLXArray.zeros([1, 64, 32, 3]).asType(.float32) + 0.3
        let expected = trained.heatmaps(image)
        let actual = loaded.heatmaps(image)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheBackendReturnsAPoseForACGImage() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXPose.backend(weightsURL: nil, jointNames: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(32, 32)]))
        let pose = try XCTUnwrap(result.pose, "pose output present")
        XCTAssertEqual(pose.count, NFKMLXPoseConfiguration.simpleBaseline.keypointCount, "17 COCO joints")
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
