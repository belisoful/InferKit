//
//  NFKMLXVitPoseTests.swift
//  InferKitMLXTests
//
//  Weight-free structure, decoding, and backend tests for ViTPose. Reference parity lives in
//  NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXVitPoseTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    func testTheParameterNamesMatchTheCheckpointLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXVitPose.makeNet(.tiny)
        let names = Set(net.parameters().flattened().map { $0.0 })
        for expected in ["backbone.embeddings.patch_embeddings.projection.weight",
                         "backbone.embeddings.position_embeddings",
                         "backbone.encoder.layer.0.attention.attention.query.weight",
                         "backbone.encoder.layer.0.attention.output.dense.weight",
                         "backbone.encoder.layer.0.layernorm_before.weight",
                         "backbone.encoder.layer.0.mlp.fc1.weight",
                         "backbone.layernorm.weight",
                         "head.conv.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheClassicDecoderCarriesItsTransposedConvolutions() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXVitPoseConfiguration.tiny
        configuration.decoder = .classic
        let net = NFKMLXVitPose.makeNet(configuration)
        let names = Set(net.parameters().flattened().map { $0.0 })
        for expected in ["head.deconv1.weight", "head.batchnorm1.weight", "head.deconv2.weight",
                         "head.conv.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
        // The transposed convolutions carry no bias in the reference.
        XCTAssertFalse(names.contains("head.deconv1.bias"))
    }

    // The patch convolution pads by two, so the grid is what that padded convolution produces. At the
    // trained geometry it happens to equal the plain quotient, which is exactly why a missing padding
    // loads cleanly and samples every window two pixels early.
    func testThePaddedPatchGridMatchesThePlainQuotientAtTheTrainedGeometry() {
        let configuration = NFKMLXVitPoseConfiguration.baseSimple
        XCTAssertEqual(configuration.gridHeight, 16)
        XCTAssertEqual(configuration.gridWidth, 12)
        XCTAssertEqual(configuration.patchCount, 192)
    }

    func testTheSimpleDecoderUpsamplesByTheScaleFactor() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXVitPoseConfiguration.tiny
        let net = NFKMLXVitPose.makeNet(configuration)
        let maps = net.heatmaps(MLXArray.zeros([1, configuration.inputHeight, configuration.inputWidth, 3]))
        eval(maps)
        XCTAssertEqual(maps.shape, [1, configuration.gridHeight * configuration.scaleFactor,
                                    configuration.gridWidth * configuration.scaleFactor,
                                    configuration.keypointCount])
    }

    // The Gaussian the DARK decode blurs with is scipy's, normalized over exactly its own taps.
    func testTheBlurKernelIsTheReferenceGaussian() {
        let weights = NFKVitPoseDecoding.blurWeights
        XCTAssertEqual(weights.count, 2 * NFKVitPoseDecoding.blurRadius + 1)
        XCTAssertEqual(Double(weights.reduce(0, +)), 1.0, accuracy: 1e-6)
        // Symmetric about the centre, and the centre is the largest tap.
        for offset in 0 ..< NFKVitPoseDecoding.blurRadius {
            XCTAssertEqual(weights[offset], weights[weights.count - 1 - offset], accuracy: 1e-9)
            XCTAssertLessThan(weights[offset], weights[NFKVitPoseDecoding.blurRadius])
        }
    }

    // A single bright cell refines toward itself, and the reported peak is that cell.
    func testASingleBrightCellDecodesToItsOwnPosition() throws {
        try requireMLXRuntime()
        var values = [Float](repeating: 0.01, count: 8 * 8)
        values[3 * 8 + 5] = 1.0
        let maps = MLXArray(values, [1, 8, 8, 1])
        let peaks = NFKVitPoseDecoding.peaks(from: maps)
        XCTAssertEqual(peaks[0].row, 3)
        XCTAssertEqual(peaks[0].column, 5)

        let keypoints = NFKVitPoseDecoding.keypoints(from: maps, jointNames: ["nose"])
        XCTAssertEqual(keypoints.count, 1)
        XCTAssertEqual(keypoints[0].name, "nose")
        XCTAssertEqual(Double(keypoints[0].position.x), 5.0 / 8.0, accuracy: 0.15)
        XCTAssertEqual(Double(keypoints[0].position.y), 3.0 / 8.0, accuracy: 0.15)
    }

    func testTheConfigurationReaderRefusesAMixtureRelease() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("vitpose-plus-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("config.json")
        let json: [String: Any] = ["model_type": "vitpose", "use_simple_decoder": false,
                                   "backbone_config": ["num_experts": 6, "part_features": 192]]
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        XCTAssertThrowsError(try NFKMLXVitPoseConfiguration.configuration(fromHuggingFace: url))
    }

    func testTheBackendReturnsAPoseForACGImage() throws {
        try requireMLXRuntime()
        let width = 48, height = 64
        var bytes = [UInt8](repeating: 128, count: width * height * 4)
        for index in stride(from: 0, to: bytes.count, by: 4) { bytes[index + 3] = 255 }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false,
                            intent: .defaultIntent)!

        var configuration = NFKMLXVitPoseConfiguration.tiny
        configuration.keypointCount = 4
        let net = NFKMLXVitPose.makeNet(configuration)
        let backend = NFKMLXVitPoseBackend(net: net, identifier: "vitpose-test",
                                           jointNames: ["a", "b", "c", "d"])
        let request = NFKInferenceRequest(inputs: [NFKInputImage: image], parameters: [:],
                                          outputModality: NFKModality.image)
        let result = try backend.runInference(for: request)
        let pose = try XCTUnwrap(result.pose)
        XCTAssertEqual(pose.count, 4)
        for keypoint in pose {
            XCTAssertTrue((0 ... 1).contains(Double(keypoint.position.x)))
            XCTAssertTrue((0 ... 1).contains(Double(keypoint.position.y)))
        }
    }
}
