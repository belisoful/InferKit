//
//  NFKMLXTextToImageInputKeyTests.swift
//  InferKitMLXTests
//
//  The IP-Adapter embedding key reaches Objective-C as the same string the Swift constant holds, and a
//  backend with an adapter attached declares it.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXTextToImageInputKeyTests: XCTestCase {

    func testTheObjectiveCKeyIsTheSwiftConstant() {
        XCTAssertEqual(NFKMLXTextToImageInputKey.imageEmbedding, NFKMLXInputImageEmbedding)
        XCTAssertEqual(NFKMLXTextToImageInputKey.imageEmbedding, "NFKMLXInputImageEmbedding")
    }

    func testAnAdapterBackendDeclaresTheEmbeddingItReads() {
        XCTAssertTrue(NFKMLXTextToImage.encodedInputKeys(readsImageEmbedding: true).contains(NFKMLXInputImageEmbedding))
        XCTAssertFalse(NFKMLXTextToImage.encodedInputKeys(readsImageEmbedding: false).contains(NFKMLXInputImageEmbedding))
        XCTAssertTrue(NFKMLXTextToImage.encodedInputKeys.isSubset(of: NFKMLXTextToImage.encodedInputKeys(readsImageEmbedding: true)))
    }
}
