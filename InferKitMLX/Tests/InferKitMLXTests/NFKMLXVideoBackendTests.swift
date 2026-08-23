//
//  NFKMLXVideoBackendTests.swift
//  InferKitMLXTests
//
//  The video modality end to end: a clip is written, decoded, transformed, encoded, and read back.
//  `NFKModalityVideo` was in the core's vocabulary with nothing emitting a clip, so these tests are
//  what establish that the output key carries a real, re-readable asset.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXVideoBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    /// A frame whose brightness encodes `index`, so frame order survives a round trip observably.
    private func frame(_ index: Int, side: Int = 32) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: side * side * 4)
        let level = UInt8(truncatingIfNeeded: 20 + index * 25)
        for pixel in 0 ..< side * side {
            bytes[pixel * 4 + 0] = level
            bytes[pixel * 4 + 1] = level
            bytes[pixel * 4 + 2] = level
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    private func writeClip(_ count: Int, fps: Double = 30) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(UUID().uuidString).mov")
        try NFKMLXVideoFile.write((0 ..< count).map { frame($0) }, framesPerSecond: fps, to: url)
        return url
    }

    // MARK: The file layer

    func testAClipRoundTripsThroughTheFileLayer() throws {
        let url = try writeClip(6)
        defer { try? FileManager.default.removeItem(at: url) }

        let clip = try NFKMLXVideoFile.read(url)
        XCTAssertEqual(clip.frames.count, 6, "every frame comes back")
        XCTAssertEqual(clip.framesPerSecond, 30, accuracy: 0.5)
        XCTAssertEqual(clip.size, CGSize(width: 32, height: 32))
    }

    // H.264 dimensions must be even, and a caller who hands over an odd frame should be told rather
    // than have a row silently cropped.
    func testAnOddFrameSizeIsRejected() {
        let odd = frame(0, side: 31)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("odd.mov")
        XCTAssertThrowsError(try NFKMLXVideoFile.write([odd], framesPerSecond: 30, to: url))
    }

    func testFramesOfDifferentSizesAreRejected() {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mixed.mov")
        XCTAssertThrowsError(try NFKMLXVideoFile.write([frame(0, side: 32), frame(1, side: 16)],
                                                       framesPerSecond: 30, to: url))
    }

    // MARK: The backend

    func testTheBackendEmitsAVideoAssetUnderTheOutputKey() throws {
        try requireMLXRuntime()
        let sourceURL = try writeClip(5)
        defer { try? FileManager.default.removeItem(at: sourceURL) }

        let backend = NFKMLXVideoBackend(identifier: "identity") { $0 }
        let request = NFKInferenceRequest(inputs: [NFKInputVideo: NFKVideoAsset(fileURL: sourceURL)])
        let result = try backend.runInference(for: request)

        let asset = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let outputURL = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: outputURL) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path), "the clip is on disk")

        let written = try NFKMLXVideoFile.read(outputURL)
        XCTAssertEqual(written.frames.count, 5, "the identity transform keeps the frame count")
        XCTAssertEqual(asset.dimensions, CGSize(width: 32, height: 32))
        XCTAssertEqual(asset.framesPerSecond, 30, accuracy: 0.5)
    }

    // A transform is free to change the frame COUNT, which is what frame interpolation does. The rate
    // has to follow, or a doubled clip plays as slow motion instead of as smoother footage.
    func testATransformThatAddsFramesRaisesTheFrameRate() throws {
        try requireMLXRuntime()
        let sourceURL = try writeClip(4)
        defer { try? FileManager.default.removeItem(at: sourceURL) }

        var configuration = NFKMLXVideoConfiguration()
        configuration.frameRateMultiplier = 2
        let backend = NFKMLXVideoBackend(identifier: "doubler", configuration: configuration) { frames in
            guard frames.count > 1 else { return frames }
            var output = [MLXArray]()
            for index in 0 ..< frames.count - 1 {
                output.append(frames[index])
                output.append((frames[index] + frames[index + 1]) / 2)      // a stand-in for a model
            }
            output.append(frames[frames.count - 1])
            return output
        }
        let request = NFKInferenceRequest(inputs: [NFKInputVideo: NFKVideoAsset(fileURL: sourceURL)])
        let asset = try XCTUnwrap(try backend.runInference(for: request)
            .output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let outputURL = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: outputURL) }

        XCTAssertEqual(try NFKMLXVideoFile.read(outputURL).frames.count, 7, "2n - 1 frames")
        XCTAssertEqual(asset.framesPerSecond, 60, accuracy: 0.5, "the rate doubles with the frames")
        // Duration is preserved, which is the point: more frames at a higher rate, same running time.
        XCTAssertEqual(asset.durationSeconds, 7.0 / 60.0, accuracy: 0.01)
    }

    func testARequestWithNoVideoIsRejected() throws {
        let backend = NFKMLXVideoBackend(identifier: "identity") { $0 }
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:])))
    }

    // MARK: The shipped clip backends

    func testRIFEInterpolatesAClipToTwiceItsFrameRate() throws {
        try requireMLXRuntime()
        let sourceURL = try writeClip(3)
        defer { try? FileManager.default.removeItem(at: sourceURL) }

        let backend = try NFKMLXRIFE.clipBackend(weightsURL: nil)      // random weights still run
        let request = NFKInferenceRequest(inputs: [NFKInputVideo: NFKVideoAsset(fileURL: sourceURL)])
        let asset = try XCTUnwrap(try backend.runInference(for: request)
            .output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let outputURL = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: outputURL) }

        XCTAssertEqual(try NFKMLXVideoFile.read(outputURL).frames.count, 5, "2n - 1 frames")
        XCTAssertEqual(asset.framesPerSecond, 60, accuracy: 0.5)
    }

    // MARK: A real model through the framework

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    /// A frame of a real photograph translated `offset` pixels to the left, so a clip built from
    /// several of them carries genuine motion for an interpolator to reason about. A synthetic
    /// gradient would not: the models are trained on photographs.
    private func translatedFrame(_ offset: Int, side: Int = 128) throws -> CGImage {
        guard let path = config["IK_VAL_FACE"] ?? config["IK_VAL_IMAGE"],
              let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let photograph = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw XCTSkip("set IK_VAL_IMAGE to a photograph") }

        guard let context = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8,
                                      bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw XCTSkip("could not build a frame") }
        context.interpolationQuality = .high
        // Draw the photograph oversized and shifted, so each frame is a real crop of a moving scene.
        let scale = CGFloat(side) * 2 / CGFloat(photograph.width)
        context.draw(photograph, in: CGRect(x: -CGFloat(offset), y: -CGFloat(side) / 2,
                                            width: CGFloat(photograph.width) * scale,
                                            height: CGFloat(photograph.height) * scale))
        guard let frame = context.makeImage() else { throw XCTSkip("could not build a frame") }
        return frame
    }

    /// Redraws `image` at `side`×`side`, so two frames of different sizes can be compared pixelwise.
    private func resized(_ image: CGImage, side: Int) -> CGImage? {
        guard let context = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8,
                                      bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
        return context.makeImage()
    }

    /// Pixelwise correlation. Both images must be the same size — comparing a large image with a
    /// small one silently compares the large one's first rows, which measures nothing.
    private func correlation(_ a: CGImage, _ b: CGImage) -> Double {
        guard a.width == b.width, a.height == b.height else { return .nan }
        let space = CGColorSpaceCreateDeviceRGB()
        let (left, _, _) = NFKMLXImageBridge.rgbaBytes(from: a, colorSpace: space)
        let (right, _, _) = NFKMLXImageBridge.rgbaBytes(from: b, colorSpace: space)
        let count = min(left.count, right.count)
        var dot = 0.0, na = 0.0, nb = 0.0
        for index in 0 ..< count where index % 4 != 3 {          // skip alpha
            let x = Double(left[index]), y = Double(right[index])
            dot += x * y; na += x * x; nb += y * y
        }
        return (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 0
    }

    // The framework carries a real trained model end to end: decode, bridge, run, encode, and read
    // back. The assertion is about the RESULT, not the plumbing — the synthesized frame has to look
    // more like the moment between its neighbours than like either neighbour, which is the only thing
    // that distinguishes interpolation from copying a frame.
    func testRIFEInterpolatesARealClipToTheMidpoint() throws {
        try requireMLXRuntime()
        guard let weights = config["IK_VAL_RIFE"] else { throw XCTSkip("set IK_VAL_RIFE") }

        // Frames at 0 and 8 pixels; the true midpoint is 4.
        let sourceURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: sourceURL) }
        try NFKMLXVideoFile.write([try translatedFrame(0), try translatedFrame(8)],
                                  framesPerSecond: 30, to: sourceURL)
        let trueMidpoint = try translatedFrame(4)

        let backend = try NFKMLXRIFE.clipBackend(weightsURL: URL(fileURLWithPath: weights))
        let request = NFKInferenceRequest(inputs: [NFKInputVideo: NFKVideoAsset(fileURL: sourceURL)])
        let asset = try XCTUnwrap(try backend.runInference(for: request)
            .output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let outputURL = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: outputURL) }

        let written = try NFKMLXVideoFile.read(outputURL)
        XCTAssertEqual(written.frames.count, 3, "2n - 1 frames")
        XCTAssertEqual(asset.framesPerSecond, 60, accuracy: 0.5)

        let synthesized = written.frames[1]
        let toMidpoint = correlation(synthesized, trueMidpoint)
        let toFirst = correlation(synthesized, written.frames[0])
        let toLast = correlation(synthesized, written.frames[2])
        print("VALIDATION video: RIFE midpoint correlation \(toMidpoint), "
              + "to neighbours \(toFirst) / \(toLast)")
        XCTAssertGreaterThan(toMidpoint, toFirst, "the synthesized frame is nearer the true midpoint")
        XCTAssertGreaterThan(toMidpoint, toLast, "than either neighbour")
    }

    // BasicVSR through the same path. Its propagation runs both directions over the clip, so this
    // also covers the framework handing the transform a whole sequence rather than frames one at a
    // time.
    func testVideoSuperResolutionUpscalesARealClip() throws {
        try requireMLXRuntime()
        guard let weights = config["IK_VAL_VIDEOSR"] else { throw XCTSkip("set IK_VAL_VIDEOSR") }

        let sourceURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: sourceURL) }
        let frames = try (0 ..< 3).map { try translatedFrame($0 * 4, side: 32) }
        try NFKMLXVideoFile.write(frames, framesPerSecond: 24, to: sourceURL)

        let backend = try NFKMLXVideoSR.clipBackend(weightsURL: URL(fileURLWithPath: weights))
        let request = NFKInferenceRequest(inputs: [NFKInputVideo: NFKVideoAsset(fileURL: sourceURL)])
        let asset = try XCTUnwrap(try backend.runInference(for: request)
            .output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let outputURL = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: outputURL) }

        let written = try NFKMLXVideoFile.read(outputURL)
        XCTAssertEqual(written.frames.count, 3, "one frame out per frame in")
        XCTAssertEqual(written.size, CGSize(width: 128, height: 128), "×4 on both sides")
        XCTAssertEqual(asset.framesPerSecond, 24, accuracy: 0.5, "the rate is unchanged")

        // The upscale is of the SOURCE, not of noise: brought back to a common size it still matches
        // the frame it came from. Comparing the ×4 output against the small source directly would
        // compare mismatched buffers and measure nothing.
        let reference = try XCTUnwrap(resized(frames[0], side: 128))
        let similarity = correlation(written.frames[0], reference)
        print("VALIDATION video: BasicVSR upscale correlates \(similarity) with its source frame")
        XCTAssertGreaterThan(similarity, 0.9, "the upscaled frame is the source frame, enlarged")
    }
}
