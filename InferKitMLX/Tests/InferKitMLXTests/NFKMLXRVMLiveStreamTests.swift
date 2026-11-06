//
//  NFKMLXRVMLiveStreamTests.swift
//  InferKitMLXTests
//
//  Robust Video Matting on a live source: a stream carries the recurrent state from frame to frame as a
//  clip does, and reset starts it again.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXRVMLiveStreamTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func image(_ index: Int, side: Int = 32) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: side * side * 4)
        for pixel in 0 ..< side * side {
            bytes[pixel * 4] = UInt8(truncatingIfNeeded: 30 + index * 40 + pixel % side * 4)
            bytes[pixel * 4 + 1] = UInt8(truncatingIfNeeded: 90 + pixel / side * 3)
            bytes[pixel * 4 + 2] = 140
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    private func bytes(_ value: Any?) throws -> [UInt8] {
        try NFKMLXImageBridge.rgbaBytes(from: try XCTUnwrap(value), colorSpace: CGColorSpaceCreateDeviceRGB()).bytes
    }

    func testAStreamMattesEachFrameAsAClipDoes() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(15)
        let net = NFKMLXRVMNet(.tiny)
        let frames = (0 ..< 3).map { NFKMLXRVMTests.frame(height: 16, width: 16, seed: 3 + $0) }
        let alpha = NFKMLXRVM.matteClip(frames, net: net, output: .alpha)
        let foreground = NFKMLXRVM.matteClip(frames, net: net, output: .foreground)
        let stream = NFKMLXRVM.streamForward(net: net)
        let request = NFKInferenceRequest(inputs: [:])
        for (index, frame) in frames.enumerated() {
            let matted = stream(frame, nil, request)
            XCTAssertEqual(matted[0..., 0..., 0 ..< 3].asArray(Float.self), foreground[index].asArray(Float.self), "frame \(index)")
            XCTAssertEqual(matted[0..., 0..., 3].asArray(Float.self), alpha[index][0..., 0..., 0].asArray(Float.self), "frame \(index)")
        }
    }

    func testTheBackendsStreamCarriesStateUntilReset() throws {
        try requireMLXRuntime()
        // As in the clip tests, the tiny network at seed 15 keeps alpha off the clamp random weights
        // can hold it at, so the matte discriminates.
        NFKMLXRandom.seed(15)
        let net = NFKMLXRVMNet(.tiny)
        var configuration = NFKMattingConfiguration()
        configuration.emitsMatte = true
        let backend = NFKMLXMattingBackend(identifier: "rvm-live-test", configuration: configuration,
                                           requestForward: { plate, _, _ in net.matte(plate) },
                                           streamForward: { NFKMLXRVM.streamForward(net: net) })
        func matte(_ result: NFKInferenceResult) throws -> [UInt8] { try bytes(result.output(forKey: NFKOutputMask)) }
        let frame = NFKInferenceRequest(inputs: [NFKInputImage: image(0)])
        let stream = try XCTUnwrap(NFKInferenceOpenLiveStream(backend, nil, nil))
        let first = try matte(stream.result(for: frame))
        let alone = try matte(backend.runInference(for: frame))
        XCTAssertTrue(first == alone, "the first frame starts from the empty state, as a single image does")
        XCTAssertTrue(first.contains { $0 != 0 }, "the matte is off the clamp")
        let second = try matte(stream.result(for: frame))
        XCTAssertFalse(second == first, "the second frame reads the state the first left")
        stream.reset()
        XCTAssertTrue(try matte(stream.result(for: frame)) == first, "reset starts from the empty state")
        stream.close()
        XCTAssertThrowsError(try stream.result(for: frame))
    }
}
