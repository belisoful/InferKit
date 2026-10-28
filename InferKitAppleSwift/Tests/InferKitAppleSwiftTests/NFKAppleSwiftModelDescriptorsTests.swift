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
            for option in descriptor.options {
                XCTAssertNotNil(option.title, "\(descriptor.identifier) \(option.key)")
                XCTAssertNotNil(option.help, "\(descriptor.identifier) \(option.key)")
            }
            XCTAssertFalse(descriptor.outputs.isEmpty, descriptor.identifier)
            for port in descriptor.outputs {
                XCTAssertEqual(port.valueType, NFKModelValueTypeForOutputKey(port.key), port.key)
            }
        }
        XCTAssertNil(NFKAppleSwiftModelDescriptors.backend(named: "no-such-engine"))
    }

    func testTheAnalyzerDescribesItsLanguageAsALocale() throws {
        let analyzer = try XCTUnwrap(NFKAppleSwiftModelDescriptors.modelDescriptors().first { $0.identifier == "apple-speech-analyzer" })
        let language = try XCTUnwrap(analyzer.option(withKey: NFKParameterSourceLanguage))
        XCTAssertEqual(language.title, "Language")
        XCTAssertTrue(language.help?.contains("locale identifier") == true, language.help ?? "")
        let translator = try XCTUnwrap(NFKAppleSwiftModelDescriptors.modelDescriptors().first { $0.identifier == "apple-translation" })
        XCTAssertEqual(translator.option(withKey: NFKParameterSourceLanguage)?.title, "Source language")
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

    func testTheTranslatorRequiresItsTargetLanguage() throws {
        let translator = try XCTUnwrap(NFKAppleSwiftModelDescriptors.modelDescriptors().first { $0.identifier == "apple-translation" })
        XCTAssertEqual(translator.option(withKey: NFKParameterTargetLanguage)?.isRequired, true)
        XCTAssertEqual(translator.option(withKey: NFKParameterSourceLanguage)?.isRequired, false)
    }

    func testTheCatalogBuildsEveryEngineByItsIdentifier() throws {
        for identifier in NFKAppleSwiftModelDescriptors.registryNames {
            let backend = try NFKModelCatalog.shared.backend(forModelIdentifier: identifier, options: nil)
            XCTAssertEqual(backend.backendIdentifier, identifier)
        }
    }
}
