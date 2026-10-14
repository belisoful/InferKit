//
//  NFKMLXSAM2VideoBackendTests.swift
//  InferKitMLXTests
//
//  SAM 2 video tracking behind the InferKit contract: a clip and a first-frame prompt become a mask
//  clip at the source's size and rate with one object score per frame, a box prompt reaches the tracker
//  as its two corner labels, and a request without a clip is refused. The tracker is the training
//  tests' small one, at the released decoder and memory widths over a 64-pixel frame.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXSAM2VideoBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func smallNet() -> NFKMLXSAM2TrackerNet {
        let encoder = NFKMLXSAM2Configuration(
            embedDimensions: 12, heads: 1, stages: [1, 1, 1, 1], windowSpec: [2, 2, 2, 2],
            globalAttentionBlocks: [3], backgroundWindow: 2, neckChannels: 256)
        return NFKMLXSAM2TrackerNet(
            NFKMLXSAM2TrackerConfiguration(encoder: encoder, imageSize: 64,
                                           occlusionSpatialEmbedding: true,
                                           temporalPositionEncodingForObjectPointers: true))
    }

    private func frame(_ index: Int, side: Int = 32) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: side * side * 4)
        for row in 0 ..< side {
            for column in 0 ..< side {
                let base = (row * side + column) * 4
                let inside = abs(row - side / 2) < side / 4 && abs(column - side / 2 - index) < side / 4
                bytes[base] = inside ? 230 : 30
                bytes[base + 1] = inside ? 220 : 40
                bytes[base + 2] = UInt8(truncatingIfNeeded: 50 + index * 20)
            }
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    private func writeClip(_ count: Int, fps: Double = 24) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).mov")
        try NFKMLXVideoFile.write((0 ..< count).map { frame($0) }, framesPerSecond: fps, to: url)
        return url
    }

    private func run(_ backend: any NFKInferenceBackend, parameters: [String: Any]) throws
        -> (frames: Int, asset: NFKVideoAsset, scores: [NSNumber]) {
        let source = try writeClip(3)
        defer { try? FileManager.default.removeItem(at: source) }
        let result = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputVideo: NFKVideoAsset(fileURL: source)], parameters: parameters))
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let url = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: url) }
        let structured = try XCTUnwrap(result.output(forKey: NFKOutputStructured) as? [String: Any])
        let scores = try XCTUnwrap(structured["objectScores"] as? [NSNumber])
        let frames = try NFKMLXVideoFile.read(url).frames.count
        return (frames, asset, scores)
    }

    func testAClickedClipBecomesAMaskClipWithOneScorePerFrame() throws {
        try requireMLXRuntime()
        let backend = NFKMLXSAM2.videoBackend(net: smallNet())
        XCTAssertEqual(backend.backendIdentifier, NFKMLXSAM2.videoModelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputVideo])
        XCTAssertEqual(backend.supportedParameterKeys, [NFKSAMPointKey, NFKMLXSAM2ParameterKey.box])

        let (frames, asset, scores) = try run(backend, parameters: [NFKSAMPointKey: [16, 16]])
        XCTAssertEqual(frames, 3)
        XCTAssertEqual(asset.dimensions, CGSize(width: 32, height: 32))
        XCTAssertEqual(asset.framesPerSecond, 24, accuracy: 0.5)
        XCTAssertEqual(scores.count, 3)
    }

    func testABoxPromptRunsThroughTheTracker() throws {
        try requireMLXRuntime()
        let backend = NFKMLXSAM2.videoBackend(net: smallNet())
        let (frames, _, scores) = try run(backend, parameters: [NFKMLXSAM2ParameterKey.box: [8, 8, 24, 24]])
        XCTAssertEqual(frames, 3)
        XCTAssertEqual(scores.count, 3)
    }

    func testThePromptReadsTheBoxAsItsCornersAndTheClickAfterIt() {
        func prompt(_ parameters: [String: Any]) -> [[Float]] {
            NFKMLXSAM2.videoPrompt(from: NFKInferenceRequest(inputs: [:], parameters: parameters),
                                   width: 32, height: 32, size: 64).map { [$0.x, $0.y, Float($0.label)] }
        }
        XCTAssertEqual(prompt([:]), [[32, 32, 1]], "no prompt clicks the center")
        XCTAssertEqual(prompt([NFKMLXSAM2ParameterKey.box: [8, 4, 24, 28]]), [[16, 8, 2], [48, 56, 3]])
        XCTAssertEqual(prompt([NFKMLXSAM2ParameterKey.box: [8, 4, 24, 28], NFKSAMPointKey: [16, 8]]),
                       [[16, 8, 2], [48, 56, 3], [32, 16, 1]])
    }

    func testARequestWithoutAClipIsRefused() throws {
        try requireMLXRuntime()
        let backend = NFKMLXSAM2.videoBackend(net: smallNet())
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }

    func testTheVideoBackendIsRegistered() throws {
        NFKMLXSAM2.registerVideo()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered("sam2-video"))
    }
}
