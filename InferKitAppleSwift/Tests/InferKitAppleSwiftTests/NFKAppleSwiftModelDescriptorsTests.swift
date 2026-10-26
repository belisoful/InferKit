//
//  NFKAppleSwiftModelDescriptorsTests.swift
//  InferKitAppleSwiftTests
//
//  Building a descriptor builds its backend without preparing it, so no locale asset, language
//  model, or answer from the system is needed.
//

import XCTest
import InferKit
@testable import InferKitAppleSwift

final class NFKAppleSwiftModelDescriptorsTests: XCTestCase {

    func testEveryDescriptorStatesTheKeysOfTheBackendBuiltForIt() throws {
        let descriptors = NFKAppleSwiftModelDescriptors.modelDescriptors()
        XCTAssertEqual(descriptors.map(\.identifier), NFKAppleSwiftModelDescriptors.registryNames)
        XCTAssertEqual(Set(descriptors.map(\.identifier)).count, descriptors.count, "identifiers are unique")
        for descriptor in descriptors {
            let backend = try XCTUnwrap(NFKAppleSwiftModelDescriptors.backend(named: descriptor.identifier))
            XCTAssertEqual(backend.backendIdentifier, descriptor.identifier)
            XCTAssertEqual(Set(descriptor.inputs.map(\.key)), backend.supportedInputKeys ?? [], descriptor.identifier)
            XCTAssertEqual(Set(descriptor.outputs.map(\.key)), backend.supportedOutputKeys ?? [], descriptor.identifier)
            XCTAssertEqual(Set(descriptor.options.map(\.key)), backend.supportedParameterKeys ?? [], descriptor.identifier)
            XCTAssertFalse(descriptor.outputs.isEmpty, descriptor.identifier)
            for port in descriptor.outputs {
                XCTAssertEqual(port.valueType, NFKModelValueTypeForOutputKey(port.key), port.key)
            }
        }
        XCTAssertNil(NFKAppleSwiftModelDescriptors.backend(named: "no-such-engine"))
    }

    func testEveryDescriptorIsASystemModelOfThisPackage() {
        for descriptor in NFKAppleSwiftModelDescriptors.modelDescriptors() {
            XCTAssertEqual(descriptor.engine, NFKModelEngineAppleSwift)
            XCTAssertFalse(descriptor.tasks.isEmpty, descriptor.identifier)
            XCTAssertEqual(descriptor.variants.count, 1)
            XCTAssertEqual(descriptor.defaultVariant?.source?.kind, NFKModelSourceKindSystem)
            XCTAssertEqual(descriptor.defaultVariant?.registryName, descriptor.identifier)
            XCTAssertEqual(descriptor.license?.name, "Apple system framework")
            XCTAssertEqual(descriptor.requirements.minimumOSVersions, ["macOS": "26.0", "iOS": "26.0"])
            if let provider = descriptor.providerClassName {
                XCTAssertTrue(NFKDynamicBackend.isProviderAvailable(provider), provider)
            }
        }
    }

    func testTheBackendsDeclareTheOutputsTheyWrite() {
        XCTAssertEqual(NFKSpeechAnalyzerBackend().supportedOutputKeys, [NFKOutputText, NFKOutputSegments])
        XCTAssertEqual(NFKTranslationBackend().supportedOutputKeys, [NFKOutputText])
        XCTAssertEqual(NFKVisionDocumentBackend().supportedOutputKeys, [NFKOutputText, NFKOutputStructured])
        XCTAssertEqual(NFKVisionSmudgeBackend().supportedOutputKeys, [NFKOutputClassifications])
    }

    func testTheSharedCatalogListsTheDescriptorsWhenThePackageIsLinked() {
        XCTAssertTrue(NFKModelCatalog.providerClassNames.contains("NFKAppleSwiftModelDescriptors"))
        for identifier in NFKAppleSwiftModelDescriptors.registryNames {
            XCTAssertEqual(NFKModelCatalog.shared.descriptor(withIdentifier: identifier)?.engine,
                           NFKModelEngineAppleSwift, identifier)
        }
        let translators = NFKModelCatalog.shared.descriptors(forTask: NFKModelTaskTranslation).map(\.identifier)
        XCTAssertTrue(translators.contains("apple-translation"))
    }

    func testTheCatalogBuildsEveryEngineByItsIdentifier() throws {
        for identifier in NFKAppleSwiftModelDescriptors.registryNames {
            let backend = try NFKModelCatalog.shared.backend(forModelIdentifier: identifier, options: nil)
            XCTAssertEqual(backend.backendIdentifier, identifier)
        }
    }
}
