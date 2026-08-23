//
//  NFKStableDiffusionProviderTests.swift
//  InferKitMLXTests
//
//  With InferKitMLX linked, the core's Stable Diffusion capability resolves to the bundled backend
//  through the dynamically discovered provider. No MLX evaluation happens here (the backend is built
//  lazily), so these run without the metallib.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKStableDiffusionProviderTests: XCTestCase {

    func testTheProviderBuildsTheBundledStableDiffusionBackend() {
        let backend = NFKStableDiffusionProvider.makeInferenceBackend()
        XCTAssertNotNil(backend)
        XCTAssertEqual(backend?.backendIdentifier, "mlx-stable-diffusion")
        XCTAssertEqual((backend as? NFKMLXBackend)?.model, .stableDiffusion15,
                       "the ungated release, so the capability activates with no credential")
    }

    func testTheCoreDiscoversTheProviderByNameWhenInferKitMLXIsLinked() throws {
        // The default provider name the core tries; present because this bundle links InferKitMLX.
        XCTAssertTrue(NFKDynamicBackend.isProviderAvailable("NFKStableDiffusionProvider"))
        XCTAssertTrue(NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityStableDiffusion),
                      "the built-in stable-diffusion capability is available once InferKitMLX is linked")

        let backend = try NFKDynamicBackend.stableDiffusionBackend()
        XCTAssertEqual(backend.backendIdentifier, "mlx-stable-diffusion")
    }
}
