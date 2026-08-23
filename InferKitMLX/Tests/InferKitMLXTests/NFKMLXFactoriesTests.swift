//
//  NFKMLXFactoriesTests.swift
//  InferKitMLXTests
//
//  The direct @objc factories (build from local weights, or download-and-build) for the shipped real
//  models. Building a net initializes MLX, so those checks run under `xcodebuild test`; the
//  download-failure check fails before any net is built, so it needs no GPU.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXFactoriesTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    func testLocalFactoriesBuildBackendsWithTheExpectedIdentifiers() throws {
        try requireMLXRuntime()
        XCTAssertEqual(try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: nil).backendIdentifier, "real-esrgan-x4")
        XCTAssertEqual(try NFKMLXRealESRGAN.backend(variant: .anime, weightsURL: nil).backendIdentifier, "real-esrgan-x4-anime")
        XCTAssertEqual(try NFKMLXRealESRGAN.backend(variant: .x2, weightsURL: nil).backendIdentifier, "real-esrgan-x2")
        XCTAssertEqual(try NFKMLXDepthAnything.backend(variant: .small, weightsURL: nil).backendIdentifier, "depth-anything-v2-small")
        XCTAssertEqual(try NFKMLXDepthAnything.backend(variant: .base, weightsURL: nil).backendIdentifier, "depth-anything-v2-base")
        XCTAssertEqual(try NFKMLXDepthAnything.backend(variant: .large, weightsURL: nil).backendIdentifier, "depth-anything-v2-large")
        XCTAssertEqual(try NFKMLXU2Net.backend(variant: .full, weightsURL: nil).backendIdentifier, "u2net")
        XCTAssertEqual(try NFKMLXU2Net.backend(variant: .light, weightsURL: nil).backendIdentifier, "u2netp")
        XCTAssertEqual(try NFKMLXNAFNet.backend(weightsURL: nil).backendIdentifier, "nafnet")
        XCTAssertEqual(try NFKMLXSAM.backend(weightsURL: nil).backendIdentifier, "sam")
        XCTAssertEqual(try NFKMLXLaMa.backend(weightsURL: nil).backendIdentifier, "lama-inpaint")
        XCTAssertEqual(try NFKMLXStableDiffusionInpaint.backend(weightsURL: nil).backendIdentifier, "sd-inpaint")
        XCTAssertEqual(try NFKMLXMarigold.backend(weightsURL: nil).backendIdentifier, "marigold-depth")
        XCTAssertEqual(try NFKMLXSDUpscaler.backend(weightsURL: nil).backendIdentifier, "sd-x4-upscaler")
        XCTAssertEqual(try NFKMLXRIFE.backend(weightsURL: nil).backendIdentifier, "rife")
        XCTAssertEqual(try NFKMLXRAFT.backend(weightsURL: nil).backendIdentifier, "raft")
        XCTAssertEqual(try NFKMLXWhisper.backend(weightsURL: nil).backendIdentifier, "whisper-tiny")
        XCTAssertEqual(try NFKMLXDemucs.backend(weightsURL: nil).backendIdentifier, "demucs")
    }

    func testANilWeightsURLBuildsAReadyBackend() throws {
        try requireMLXRuntime()
        XCTAssertTrue(try NFKMLXNAFNet.backend(weightsURL: nil).isReady, "no weights → random weights, isReady true")
    }

    func testTheVariantEnumsMapToTheExpectedNamesAndPresets() throws {
        try requireMLXRuntime()
        // Depth variants select the small/base/large presets (verified through the built identifiers).
        XCTAssertEqual(try NFKMLXDepthAnything.backend(variant: .base, weightsURL: nil).backendIdentifier,
                       NFKMLXDepthAnything.baseModelName)
        XCTAssertEqual(try NFKMLXRealESRGAN.backend(variant: .anime, weightsURL: nil).backendIdentifier,
                       NFKMLXRealESRGAN.animeModelName)
        XCTAssertEqual(try NFKMLXU2Net.backend(variant: .light, weightsURL: nil).backendIdentifier,
                       NFKMLXU2Net.lightModelName)
    }

    // Fails at the download step (empty repo → NFKHFHub rejects it) before any net is built — no GPU,
    // no network. Confirms the download factory delegates cleanly rather than duplicating the build.
    func testDownloadFactoryFailsCleanlyForAnInvalidRepoWithoutBuilding() {
        let cache = FileManager.default.temporaryDirectory
        XCTAssertThrowsError(try NFKMLXNAFNet.backend(repo: "", weightsPath: "model.safetensors", revision: nil, cacheDirectoryURL: cache))
        XCTAssertThrowsError(try NFKMLXDepthAnything.backend(variant: .small, repo: "", weightsPath: "model.safetensors", revision: nil, cacheDirectoryURL: cache))
    }

    // The async factory delivers a download failure to the completion handler (nil backend + error)
    // rather than throwing, and never builds a net — no GPU, no successful network. Covers a plain and a
    // variant factory.
    func testAsyncDownloadFactoryDeliversFailureToTheCompletionHandler() {
        let cache = FileManager.default.temporaryDirectory
        let plain = expectation(description: "plain factory handler")
        NFKMLXNAFNet.backend(repo: "", weightsPath: "model.safetensors", revision: nil, cacheDirectoryURL: cache) { backend, error in
            XCTAssertNil(backend)
            XCTAssertNotNil(error)
            plain.fulfill()
        }
        let variant = expectation(description: "variant factory handler")
        NFKMLXDepthAnything.backend(variant: .small, repo: "", weightsPath: "model.safetensors", revision: nil, cacheDirectoryURL: cache) { backend, error in
            XCTAssertNil(backend)
            XCTAssertNotNil(error)
            variant.fulfill()
        }
        wait(for: [plain, variant], timeout: 10)
    }

    // NFKMLXHub's async factory fails fast for an unregistered name: the handler receives an error before
    // any download starts.
    func testAsyncHubFailsFastForAnUnregisteredModel() {
        let done = expectation(description: "hub handler")
        NFKMLXHub.backend(named: "no-such-model-xyz", repo: "org/x", weightsPath: "m.safetensors",
                          revision: nil, cacheDirectoryURL: FileManager.default.temporaryDirectory) { backend, error in
            XCTAssertNil(backend)
            XCTAssertNotNil(error)
            done.fulfill()
        }
        wait(for: [done], timeout: 5)
    }
}
