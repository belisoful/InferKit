//
//  NFKMLXWhisperProviderTests.swift
//  InferKitMLXTests
//
//  With InferKitMLX linked, the core's transcription capability resolves to the bundled Whisper
//  backend through the dynamically discovered provider. Discovery is pure runtime lookup (no MLX);
//  building the backend constructs MLXNN layers, so that check runs where MLX has a Metal library.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXWhisperProviderTests: XCTestCase {

    func testTheCoreDiscoversTheTranscriptionProviderByName() {
        // Present because this bundle links InferKitMLX; no MLX evaluation in the lookup.
        XCTAssertTrue(NFKDynamicBackend.isProviderAvailable("NFKMLXWhisperProvider"))
        XCTAssertTrue(NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityTranscription),
                      "the built-in transcription capability is available once InferKitMLX is linked")
    }

    func testTheProviderBuildsAWhisperBackend() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "builds MLXNN layers; run Tools/mlx-metallib.sh or xcodebuild")
        let backend = try XCTUnwrap(NFKMLXWhisperProvider.makeInferenceBackend())
        XCTAssertEqual(backend.backendIdentifier, "whisper-tiny")

        let resolved = try NFKDynamicBackend.backend(forCapability: NFKCapabilityTranscription)
        XCTAssertEqual(resolved.backendIdentifier, "whisper-tiny")
    }
}
