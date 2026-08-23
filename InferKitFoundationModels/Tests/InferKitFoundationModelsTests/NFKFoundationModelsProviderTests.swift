//
//  NFKFoundationModelsProviderTests.swift
//  InferKitFoundationModelsTests
//
//  With InferKitFoundationModels linked, the core's text-generation capability resolves to the on-device
//  LLM backend through the dynamically discovered provider. Discovery and construction do not load the
//  model, so these run without Apple Intelligence enabled.
//

import XCTest
import InferKit
@testable import InferKitFoundationModels

final class NFKFoundationModelsProviderTests: XCTestCase {

    func testTheProviderBuildsTheFoundationModelsBackend() {
        let backend = NFKFoundationModelsProvider.makeInferenceBackend()
        XCTAssertNotNil(backend)
        XCTAssertEqual(backend?.backendIdentifier, "foundation-models")
    }

    func testTheCoreDiscoversTheProviderByNameWhenLinked() throws {
        XCTAssertTrue(NFKDynamicBackend.isProviderAvailable("NFKFoundationModelsProvider"))
        XCTAssertTrue(NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityTextGeneration),
                      "the built-in text-generation capability is available once the package is linked")

        let backend = try NFKDynamicBackend.backend(forCapability: NFKCapabilityTextGeneration)
        XCTAssertEqual(backend.backendIdentifier, "foundation-models")
    }
}
