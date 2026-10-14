//
//  NFKMLXTextToImageInputKeyTests.swift
//  InferKitMLXTests
//
//  The IP-Adapter embedding key reaches Objective-C as the same string the Swift constant holds.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXTextToImageInputKeyTests: XCTestCase {

    func testTheObjectiveCKeyIsTheSwiftConstant() {
        XCTAssertEqual(NFKMLXTextToImageInputKey.imageEmbedding, NFKMLXInputImageEmbedding)
        XCTAssertEqual(NFKMLXTextToImageInputKey.imageEmbedding, "NFKMLXInputImageEmbedding")
    }
}
