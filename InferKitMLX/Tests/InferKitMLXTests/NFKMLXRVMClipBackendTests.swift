//
//  NFKMLXRVMClipBackendTests.swift
//  InferKitMLXTests
//
//  Robust Video Matting over whole clips: the recurrent state carries from frame to frame, each output
//  writes a clip at the source's size and rate, and both encoders' clip backends are registered.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXRVMClipBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func frame(_ index: Int, side: Int = 32) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: side * side * 4)
        for pixel in 0 ..< side * side {
            bytes[pixel * 4] = UInt8(truncatingIfNeeded: 30 + index * 30 + pixel % side * 4)
            bytes[pixel * 4 + 1] = UInt8(truncatingIfNeeded: 90 + pixel / side * 3)
            bytes[pixel * 4 + 2] = 140
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    func testTheStateCarriesFromFrameToFrame() throws {
        try requireMLXRuntime()
        let net = NFKMLXRVMNet(.tiny)
        let still = NFKMLXRVMTests.frame(height: 16, width: 16, seed: 3)
        let clip = NFKMLXRVM.matteClip([still, still], net: net, output: .alpha)
        XCTAssertEqual(clip.count, 2)
        XCTAssertEqual(clip[0].shape, [16, 16, 3])
        XCTAssertNotEqual(clip[0].asArray(Float.self), clip[1].asArray(Float.self),
                          "the second frame reads the state the first left")
        let first = net.matte(still)[0..., 0..., 3]
        XCTAssertEqual(clip[0][0..., 0..., 0].asArray(Float.self), first.asArray(Float.self),
                       "the first frame starts from the empty state, as a single image does")
    }

    func testEveryOutputWritesAClipAtTheSourcesSizeAndRate() throws {
        try requireMLXRuntime()
        let source = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).mov")
        try NFKMLXVideoFile.write((0 ..< 3).map { frame($0) }, framesPerSecond: 24, to: source)
        defer { try? FileManager.default.removeItem(at: source) }

        for output in [NFKMLXRVMClipOutput.alpha, .foreground, .composition] {
            let backend = NFKMLXRVM.clipBackend(net: NFKMLXRVMNet(.tiny), identifier: "rvm-clip-test", output: output)
            XCTAssertEqual(backend.supportedInputKeys, [NFKInputVideo])
            let result = try backend.runInference(for: NFKInferenceRequest(
                inputs: [NFKInputVideo: NFKVideoAsset(fileURL: source)]))
            let asset = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
            let url = try XCTUnwrap(asset.fileURL)
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertEqual(try NFKMLXVideoFile.read(url).frames.count, 3)
            XCTAssertEqual(asset.dimensions, CGSize(width: 32, height: 32))
            XCTAssertEqual(asset.framesPerSecond, 24, accuracy: 0.5)
        }
    }

    func testARequestWithoutAClipIsRefused() throws {
        try requireMLXRuntime()
        let backend = NFKMLXRVM.clipBackend(net: NFKMLXRVMNet(.tiny), identifier: "rvm-clip-test", output: .alpha)
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }

    func testBothEncodersClipBackendsAreRegistered() throws {
        NFKMLXRVM.registerClip()
        XCTAssertEqual(NFKMLXRVM.clipModelName(variant: .mobileNetV3), "robust-video-matting-clip")
        XCTAssertEqual(NFKMLXRVM.clipModelName(variant: .resNet50), "robust-video-matting-resnet50-clip")
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered("robust-video-matting-clip"))
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered("robust-video-matting-resnet50-clip"))
    }
}
