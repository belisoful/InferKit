//
//  NFKMLXCatalogTests.swift
//  InferKitMLXTests
//
//  The package's model descriptors. Building and reading descriptors touches no MLX, so every test but
//  the stand-in backend comparison runs under `swift test` without a Metal library.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXCatalogTests: XCTestCase {

    private static let standInNames = ["diffusion-controlnet", "diffusion-depth", "diffusion-inpaint",
                                       "diffusion-upscaler", "green-screen-keyer", "sam", "swinir-x4", "tone-speech"]

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    func testEveryEntryBuildsADescriptor() throws {
        for entry in NFKMLXModelDescriptors.entries {
            XCTAssertNoThrow(try entry.descriptor(), entry.identifier)
        }
        XCTAssertEqual(NFKMLXModelDescriptors.modelDescriptors().count, NFKMLXModelDescriptors.entries.count)
    }

    func testTheSharedCatalogFindsTheProviderAndHidesTheStandIns() {
        XCTAssertTrue(NFKModelCatalog.providerClassNames.contains("NFKMLXModelDescriptors"))
        let catalog = NFKModelCatalog.shared
        let visible = Set(catalog.visibleDescriptors.map(\.identifier))
        for name in Self.standInNames {
            let descriptor = catalog.descriptors.first { $0.registryName == name }
            XCTAssertNotNil(descriptor, "the shared catalog has no descriptor registered as \(name)")
            guard let descriptor else { continue }
            XCTAssertTrue(descriptor.isStandIn, name)
            XCTAssertEqual(descriptor.engine, NFKModelEngineMLX, name)
            XCTAssertFalse(visible.contains(descriptor.identifier), "\(name) is listed as visible")
        }
    }

    func testEveryDescriptorRoundTripsThroughItsDictionary() throws {
        for descriptor in NFKMLXModelDescriptors.modelDescriptors() {
            let dictionary = descriptor.dictionaryRepresentation()
            let read = try NFKModelDescriptor(dictionary: dictionary)
            XCTAssertEqual(read.dictionaryRepresentation() as NSDictionary, dictionary as NSDictionary,
                           descriptor.identifier)
        }
    }

    func testTheProviderIdentifiersAreUnique() {
        let identifiers = NFKMLXModelDescriptors.modelDescriptors().map(\.identifier)
        XCTAssertEqual(Set(identifiers).count, identifiers.count, "duplicate identifiers in \(identifiers)")
        for descriptor in NFKMLXModelDescriptors.modelDescriptors() {
            let variants = descriptor.variants.map(\.identifier)
            XCTAssertEqual(Set(variants).count, variants.count, "duplicate variants in \(descriptor.identifier)")
        }
    }

    func testEveryRegistryNameIsRegisteredAfterRegisterAll() {
        NFKMLXReferenceModels.registerAll()
        let registered = Set(NFKMLXModelRegistry.registeredModelNames)
        for descriptor in NFKMLXModelDescriptors.modelDescriptors() {
            let names = [descriptor.registryName].compactMap { $0 } + descriptor.variants.compactMap(\.registryName)
            for name in names {
                XCTAssertTrue(registered.contains(name), "\(descriptor.identifier) names unregistered \(name)")
            }
        }
        let standIns = NFKMLXModelDescriptors.modelDescriptors().filter(\.isStandIn).compactMap(\.registryName)
        XCTAssertEqual(Set(standIns), Set(Self.standInNames))
    }

    func testEachStandInBackendDeclaresItsDescriptorsPortsAndOptions() throws {
        try requireMLXRuntime()
        NFKMLXReferenceModels.registerAll()
        for descriptor in NFKMLXModelDescriptors.modelDescriptors() where descriptor.isStandIn {
            let name = try XCTUnwrap(descriptor.registryName, descriptor.identifier)
            let backend = try NFKMLXModelRegistry.backend(named: name, weightsURL: nil)
            XCTAssertEqual(backend.supportedInputKeys ?? [], Set(descriptor.inputs.map(\.key)), name)
            XCTAssertEqual(backend.supportedParameterKeys ?? [], Set(descriptor.options.map(\.key)), name)
            XCTAssertEqual(backend.supportedOutputKeys ?? [], Set(descriptor.outputs.map(\.key)), name)
        }
    }
}
