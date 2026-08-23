//
//  NFKMLXTensorBackendTests.swift
//  InferKitMLXTests
//
//  Contract tests need no GPU. A multi-input/multi-output forward is host-verified where MLX runs.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXTensorBackendTests: XCTestCase {

    private func configuration() -> NFKMLXTensorConfiguration {
        NFKMLXTensorConfiguration(
            inputs: [
                NFKMLXTensorPort(key: NFKInputImage, tensorName: "plate", channels: 3),
                NFKMLXTensorPort(key: NFKInputMask, tensorName: "hint", channels: 1),
            ],
            outputs: [
                NFKMLXTensorPort(key: NFKOutputImage, tensorName: "foreground"),
                NFKMLXTensorPort(key: NFKOutputMask, tensorName: "matte"),
            ])
    }

    func testTheBackendReportsTheSuppliedIdentity() {
        let backend = NFKMLXTensorBackend(identifier: "composite", configuration: configuration()) { $0 }
        XCTAssertEqual(backend.backendIdentifier, "composite")
        XCTAssertTrue(backend.isReady)
    }

    func testAnInferenceWithNoConfiguredInputPresentFails() {
        let backend = NFKMLXTensorBackend(configuration: configuration()) { $0 }
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "no images"])
        XCTAssertThrowsError(try backend.runInference(for: request))
    }
}
