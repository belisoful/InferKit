//
//  NFKFoundationModelsDescriptorsTests.swift
//  InferKitFoundationModelsTests
//
//  Building the descriptor builds the backend without loading the model, so these run without Apple
//  Intelligence enabled.
//

import XCTest
import InferKit
@testable import InferKitFoundationModels

final class NFKFoundationModelsDescriptorsTests: XCTestCase {

    private var descriptor: NFKModelDescriptor {
        NFKFoundationModelsDescriptors.modelDescriptors()[0]
    }

    func testTheDescriptorStatesTheBackendsKeys() {
        let backend = NFKFoundationModelsBackend()
        XCTAssertEqual(NFKFoundationModelsDescriptors.modelDescriptors().count, 1)
        XCTAssertEqual(Set(descriptor.inputs.map(\.key)), backend.supportedInputKeys)
        XCTAssertEqual(Set(descriptor.outputs.map(\.key)), backend.supportedOutputKeys)
        XCTAssertEqual(Set(descriptor.parameters.map(\.key)), backend.supportedParameterKeys)
        for port in descriptor.outputs {
            XCTAssertEqual(port.valueType, NFKModelValueTypeForOutputKey(port.key), port.key)
        }
    }

    func testTheBackendDeclaresTheOutputsItWrites() {
        let keys = NFKFoundationModelsBackend().supportedOutputKeys
        for key in [NFKOutputText, NFKOutputStructured, NFKOutputToolCalls, NFKOutputToolInvocations] {
            XCTAssertTrue(keys.contains(key), key)
        }
        if #unavailable(macOS 27, iOS 27) {
            XCTAssertFalse(keys.contains(NFKOutputUsage), "the framework reports usage from macOS 27 / iOS 27")
        }
    }

    func testTheDescriptorIsTheSystemLanguageModel() {
        XCTAssertEqual(descriptor.identifier, NFKFoundationModelsDescriptors.systemLanguageModelIdentifier)
        XCTAssertEqual(descriptor.engine, NFKModelEngineFoundationModels)
        XCTAssertEqual(descriptor.providerClassName, "NFKFoundationModelsProvider")
        XCTAssertTrue(NFKDynamicBackend.isProviderAvailable(descriptor.providerClassName ?? ""))
        XCTAssertEqual(descriptor.tasks.first, NFKModelTaskTextGeneration)
        XCTAssertEqual(descriptor.defaultVariant?.source?.kind, NFKModelSourceKindSystem)
        XCTAssertEqual(descriptor.defaultVariant?.source?.layout, NFKModelSourceLayoutNone)
        XCTAssertEqual(descriptor.license?.name, "Apple system framework")
        XCTAssertEqual(descriptor.requirements.minimumOSVersions, ["macOS": "26.0", "iOS": "26.0"])
        XCTAssertEqual(descriptor.introducedVersion, "0.4.0")
    }

    func testTheSharedCatalogListsTheDescriptorWhenThePackageIsLinked() {
        XCTAssertTrue(NFKModelCatalog.providerClassNames.contains("NFKFoundationModelsDescriptors"))
        let listed = NFKModelCatalog.shared.descriptor(withIdentifier: NFKFoundationModelsDescriptors.systemLanguageModelIdentifier)
        XCTAssertEqual(listed?.engine, NFKModelEngineFoundationModels)
    }
}
