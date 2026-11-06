//
//  NFKMLXCatalogLiveTests.swift
//  InferKitMLXTests
//
//  The live profiles in the package's catalog. Building descriptors touches no MLX.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXCatalogLiveTests: XCTestCase {

    private lazy var descriptors = NFKMLXModelDescriptors.modelDescriptors()

    func testEveryLiveModelIsAShippedEntryWhosePortsCarryTheFrame() throws {
        for (identifier, outputs) in NFKMLXCatalogLive.videoFrameOutputs {
            let descriptor = try XCTUnwrap(descriptors.first { $0.identifier == identifier }, "\(identifier) has no catalog entry")
            let live = try XCTUnwrap(descriptor.liveProfile, identifier)
            XCTAssertEqual(live.unit, NFKModelLiveUnitVideoFrame, identifier)
            XCTAssertEqual(live.inputKeys, [NFKInputImage], identifier)
            XCTAssertEqual(live.outputKeys, outputs, identifier)
            XCTAssertFalse(live.isStateful, identifier)
            XCTAssertTrue(descriptor.inputs.contains { $0.key == NFKInputImage && $0.isRequired }, identifier)
        }
    }

    func testOnlyTheDeclaredModelsAreLive() {
        let live = Set(descriptors.filter { $0.liveProfile != nil }.map(\.identifier))
        XCTAssertEqual(live, Set(NFKMLXCatalogLive.videoFrameOutputs.keys))
        XCTAssertNil(descriptors.first { $0.identifier == "birefnet" }?.liveProfile, "a seconds-per-frame matter is not live")
    }
}
