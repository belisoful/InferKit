//
//  NFKMLXDownloadTests.swift
//  InferKitMLXTests
//
//  The download-and-build path (NFKMLXHub + the per-model factories), covered two ways:
//
//  1. Hermetically, with no network: a real safetensors is placed at the exact cache location the hub
//     resolves, so NFKHFHub cache-hits and returns it without a fetch. This proves the hub → registry
//     → factory → loadWeights chain end to end (sync and async) and that the downloaded file reaches
//     the loader, deterministically. It builds and evaluates MLX, so it runs under `xcodebuild test`.
//
//  2. Live, gated behind environment variables (INFERKIT_LIVE_*): a real Hugging Face download through
//     the hub. Skipped unless the caller supplies a repo + weights path for a registered model, so CI
//     stays green while the real network path stays runnable on demand.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

// Captures the net for the module backend's @Sendable forward closure (the shipped models use an
// equivalent private holder).
private final class TestNetHolder: @unchecked Sendable {
    let net: NFKRealESRGANNet
    init(_ net: NFKRealESRGANNet) { self.net = net }
}

final class NFKMLXDownloadTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    // A model registered around a small RRDBNet whose factory loads the weights URL — the same shape as
    // a shipped model's `register()`, so it exercises the real download-and-build chain.
    private func registerTestModel(named name: String) {
        NFKMLXModelRegistry.register(name: name) { url in
            let net = NFKRealESRGANNet(features: 8, blocks: 1, growth: 4)
            if let url {
                try NFKMLXRealESRGAN.loadWeights(into: net, from: url)
            }
            let holder = TestNetHolder(net)
            return NFKMLXModuleBackend(identifier: name, isReady: true) { holder.net.upscale($0) }
        }
    }

    // Writes a real safetensors for the test architecture (random weights, PyTorch conv layout) at the
    // exact path the hub resolves for (repo, revision "main", weightsPath), and returns that file URL.
    private func placeCheckpoint(in cache: URL, repo: String, weightsPath: String) throws -> URL {
        let net = NFKRealESRGANNet(features: 8, blocks: 1, growth: 4)
        let pytorchLayout = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        var fileURL = cache
        for component in repo.split(separator: "/") {
            fileURL.appendPathComponent(String(component))
        }
        fileURL.appendPathComponent("main")
        for component in weightsPath.split(separator: "/") {
            fileURL.appendPathComponent(String(component))
        }
        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try save(arrays: pytorchLayout, url: fileURL)
        return fileURL
    }

    private func upscaledBytes(_ backend: any NFKInferenceBackend) throws -> [UInt8] {
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(2, 2)]))
        let value = try XCTUnwrap(result.output(forKey: NFKOutputImage))
        guard CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
            throw NFKMLXError.noOutput
        }
        return Self.rgbaBytes(value as! CGImage)
    }

    // MARK: Hermetic cache-hit happy path (no network)

    func testHubBuildsFromACacheHitAndDeliversTheSameForwardAsADirectLoad() throws {
        try requireMLXRuntime()
        let modelName = "test-rrdb-\(UUID().uuidString)"
        let repo = "test-org/rrdb"
        let weightsPath = "model.safetensors"
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("hub-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }

        registerTestModel(named: modelName)
        let checkpoint = try placeCheckpoint(in: cache, repo: repo, weightsPath: weightsPath)

        // Cache hit: the hub returns the on-disk file without contacting the endpoint, then builds.
        let viaHub = try NFKMLXHub.backend(named: modelName, repo: repo, weightsPath: weightsPath,
                                           revision: nil, cacheDirectoryURL: cache)
        XCTAssertTrue(viaHub.isReady)
        XCTAssertEqual(viaHub.backendIdentifier, modelName)

        // The factory received the cached file: loading it directly reproduces the hub backend's forward.
        let direct = try NFKMLXModelRegistry.backend(named: modelName, weightsURL: checkpoint)
        XCTAssertEqual(try upscaledBytes(viaHub), try upscaledBytes(direct),
                       "the hub delivered the cached checkpoint to the factory")
    }

    func testAsyncHubBuildsFromACacheHitAndDeliversAReadyBackend() throws {
        try requireMLXRuntime()
        let modelName = "test-rrdb-async-\(UUID().uuidString)"
        let repo = "test-org/rrdb"
        let weightsPath = "model.safetensors"
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("hub-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }

        registerTestModel(named: modelName)
        let checkpoint = try placeCheckpoint(in: cache, repo: repo, weightsPath: weightsPath)
        let expected = try upscaledBytes(NFKMLXModelRegistry.backend(named: modelName, weightsURL: checkpoint))

        let done = expectation(description: "async cache-hit build")
        var delivered: (any NFKInferenceBackend)?
        NFKMLXHub.backend(named: modelName, repo: repo, weightsPath: weightsPath,
                          revision: nil, cacheDirectoryURL: cache) { backend, error in
            XCTAssertNil(error)
            delivered = backend
            done.fulfill()
        }
        wait(for: [done], timeout: 30)

        let backend = try XCTUnwrap(delivered)
        XCTAssertTrue(backend.isReady)
        XCTAssertEqual(try upscaledBytes(backend), expected, "the async peer builds from the same cached file")
    }

    // MARK: Live download (network-gated)

    // Set INFERKIT_LIVE_MODEL (a registered name), _REPO, and _WEIGHTS_PATH (optionally _REVISION) to a
    // real Hugging Face checkpoint whose weights match the model, to exercise the real download → build
    // path through the hub, synchronously and asynchronously. Skipped otherwise, so CI stays green.
    func testLiveDownloadThroughTheHubBuildsAReadyBackend() throws {
        try requireMLXRuntime()
        let env = ProcessInfo.processInfo.environment
        guard let model = env["INFERKIT_LIVE_MODEL"],
              let repo = env["INFERKIT_LIVE_REPO"],
              let weightsPath = env["INFERKIT_LIVE_WEIGHTS_PATH"] else {
            throw XCTSkip("set INFERKIT_LIVE_MODEL, _REPO, and _WEIGHTS_PATH to run the live download test")
        }
        let revision = env["INFERKIT_LIVE_REVISION"]
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("live-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }

        NFKMLXReferenceModels.registerAll()

        let sync = try NFKMLXHub.backend(named: model, repo: repo, weightsPath: weightsPath,
                                         revision: revision, cacheDirectoryURL: cache)
        XCTAssertTrue(sync.isReady, "the downloaded checkpoint built a ready backend")

        let done = expectation(description: "async live download")
        var delivered: (any NFKInferenceBackend)?
        NFKMLXHub.backend(named: model, repo: repo, weightsPath: weightsPath,
                          revision: revision, cacheDirectoryURL: cache) { backend, error in
            XCTAssertNil(error)
            delivered = backend
            done.fulfill()
        }
        wait(for: [done], timeout: 300)
        XCTAssertTrue(try XCTUnwrap(delivered).isReady)
    }

    // MARK: Helpers

    static func solid(_ width: Int, _ height: Int) -> CGImage {
        let pixels = [UInt8](repeating: 128, count: width * height * 4)
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    static func rgbaBytes(_ image: CGImage) -> [UInt8] {
        let width = image.width, height = image.height
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &buffer, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}
