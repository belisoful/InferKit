//
//  NFKMLXDownloadPeerTests.swift
//  InferKitMLXTests
//
//  The download peers of the RetinaFace detector and the RIFE and BasicVSR clip backends, with no
//  network: a checkpoint saved by `NFKMLXWeights.save` is placed where the hub resolves the repo, so the
//  hub cache-hits and the factory builds from it. Also the clip backends' registered names.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXDownloadPeerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func temporaryCache() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("peer-hub-\(UUID().uuidString)")
    }

    /// Saves `module` at the path the hub resolves for (`repo`, revision `main`, `weightsPath`).
    private func place(_ module: Module, in cache: URL, repo: String, weightsPath: String) throws {
        var url = cache
        let components = repo.split(separator: "/").map(String.init) + ["main"]
            + weightsPath.split(separator: "/").map(String.init)
        for component in components {
            url.appendPathComponent(component)
        }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try NFKMLXWeights.save(module, to: url)
    }

    private func frame(_ index: Int, side: Int = 32) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: side * side * 4)
        for pixel in 0 ..< side * side {
            bytes[pixel * 4] = UInt8(truncatingIfNeeded: 20 + index * 40 + pixel % side)
            bytes[pixel * 4 + 1] = UInt8(truncatingIfNeeded: 60 + pixel / side)
            bytes[pixel * 4 + 2] = 128
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

    // MARK: RetinaFace

    func testTheDetectorBuildsFromACachedCheckpoint() throws {
        try requireMLXRuntime()
        let cache = temporaryCache()
        defer { try? FileManager.default.removeItem(at: cache) }
        try place(try NFKMLXRetinaFace.network(weightsURL: nil), in: cache, repo: "test-org/retinaface",
                  weightsPath: "retinaface.safetensors")

        let detector = try NFKMLXRetinaFace.detector(repo: "test-org/retinaface", weightsPath: "retinaface.safetensors",
                                                     revision: nil, cacheDirectoryURL: cache)
        XCTAssertNoThrow(try detector.faces(inImage: frame(0, side: 64)))

        let done = expectation(description: "async detector")
        var delivered: NFKMLXRetinaFaceDetector?
        NFKMLXRetinaFace.detector(repo: "test-org/retinaface", weightsPath: "retinaface.safetensors", revision: nil,
                                  cacheDirectoryURL: cache, confidenceThreshold: 0.8,
                                  suppressionThreshold: 0.4) { detector, error in
            XCTAssertNil(error)
            delivered = detector
            done.fulfill()
        }
        wait(for: [done], timeout: 60)
        XCTAssertNotNil(delivered)
    }

    // MARK: RIFE

    func testTheRIFEClipBackendBuildsFromACachedCheckpointAndDoublesTheRate() throws {
        try requireMLXRuntime()
        let cache = temporaryCache()
        defer { try? FileManager.default.removeItem(at: cache) }
        try place(NFKMLXRIFE.makeNet(), in: cache, repo: "test-org/rife", weightsPath: "flownet.safetensors")

        let backend = try NFKMLXRIFE.clipBackend(repo: "test-org/rife", weightsPath: "flownet.safetensors",
                                                 revision: nil, cacheDirectoryURL: cache)
        XCTAssertEqual(backend.backendIdentifier, NFKMLXRIFE.clipModelName)
        let source = try writeClip(3)
        defer { try? FileManager.default.removeItem(at: source) }
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputVideo: NFKVideoAsset(fileURL: source)]))
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let url = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(try NFKMLXVideoFile.read(url).frames.count, 5, "three frames gain two between them")
        XCTAssertEqual(asset.framesPerSecond, 48, accuracy: 0.5)

        let done = expectation(description: "async clip backend")
        NFKMLXRIFE.clipBackend(repo: "test-org/rife", weightsPath: "flownet.safetensors", revision: nil,
                               cacheDirectoryURL: cache) { backend, error in
            XCTAssertNil(error)
            XCTAssertEqual(backend?.backendIdentifier, NFKMLXRIFE.clipModelName)
            done.fulfill()
        }
        wait(for: [done], timeout: 60)
    }

    func testTheRIFEClipBackendIsRegistered() throws {
        try requireMLXRuntime()
        NFKMLXRIFE.registerClip()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered("rife-clip"))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXRIFE.clipModelName, weightsURL: nil)
        XCTAssertEqual(backend.backendIdentifier, "rife-clip")
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputVideo])
    }

    // MARK: BasicVSR

    func testTheVideoSRClipBackendBuildsFromACachedCheckpoint() throws {
        try requireMLXRuntime()
        let cache = temporaryCache()
        defer { try? FileManager.default.removeItem(at: cache) }
        try place(NFKMLXVideoSRNet(.base), in: cache, repo: "test-org/basicvsr", weightsPath: "basicvsr.safetensors")

        let backend = try NFKMLXVideoSR.clipBackend(repo: "test-org/basicvsr", weightsPath: "basicvsr.safetensors",
                                                    revision: nil, cacheDirectoryURL: cache)
        XCTAssertEqual(backend.backendIdentifier, NFKMLXVideoSR.clipModelName)

        let done = expectation(description: "async clip backend")
        NFKMLXVideoSR.clipBackend(repo: "test-org/basicvsr", weightsPath: "basicvsr.safetensors", revision: nil,
                                  cacheDirectoryURL: cache) { backend, error in
            XCTAssertNil(error)
            XCTAssertEqual(backend?.backendIdentifier, NFKMLXVideoSR.clipModelName)
            done.fulfill()
        }
        wait(for: [done], timeout: 60)
    }

    func testTheVideoSRClipBackendIsRegistered() throws {
        try requireMLXRuntime()
        NFKMLXVideoSR.registerClip()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered("video-super-resolution-clip"))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXVideoSR.clipModelName, weightsURL: nil)
        XCTAssertEqual(backend.backendIdentifier, "video-super-resolution-clip")
    }
}
