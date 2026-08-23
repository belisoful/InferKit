//
//  NFKMLXModuleBackendTests.swift
//  InferKitMLXTests
//
//  These cover the bring-your-own-module backend's contract, which needs no GPU or weights. Running
//  a forward closure is host-verified (it evaluates MLX arrays on the GPU).
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXModuleBackendTests: XCTestCase {

    func testTheBackendReportsTheSuppliedIdentityAndReadiness() {
        let backend = NFKMLXModuleBackend(identifier: "green-former", isReady: false) { $0 }
        XCTAssertEqual(backend.backendIdentifier, "green-former")
        XCTAssertFalse(backend.isReady)
    }

    func testTheBackendDefaultsToReadyWithAGenericIdentifier() {
        let backend = NFKMLXModuleBackend { $0 }
        XCTAssertEqual(backend.backendIdentifier, "mlx-module")
        XCTAssertTrue(backend.isReady)
    }

    func testAnInferenceWithoutAnImageInputFails() {
        let backend = NFKMLXModuleBackend { $0 }
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "no image here"])
        XCTAssertThrowsError(try backend.runInference(for: request))
    }
}
