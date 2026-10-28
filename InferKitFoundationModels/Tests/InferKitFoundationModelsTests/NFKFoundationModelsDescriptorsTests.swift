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
        let requestOptions = descriptor.options.filter { $0.stage == NFKModelOptionStageRequest }
        XCTAssertEqual(Set(requestOptions.map(\.key)), backend.supportedParameterKeys)
        for option in requestOptions {
            XCTAssertNotNil(option.title, option.key)
            XCTAssertNotNil(option.help, option.key)
        }
        for port in descriptor.outputs {
            XCTAssertEqual(port.valueType, NFKModelValueTypeForOutputKey(port.key), port.key)
        }
    }

    func testTheRequestOptionsStateTheFrameworksRanges() {
        XCTAssertEqual(descriptor.option(withKey: NFKParameterTopP)?.maximum, 1)
        XCTAssertEqual(descriptor.option(withKey: NFKParameterTopK)?.minimum, 1)
        XCTAssertEqual(descriptor.option(withKey: NFKParameterTemperature)?.kind, NFKModelOptionKindNumber)
        XCTAssertNil(descriptor.option(withKey: NFKParameterTemperature)?.defaultValue, "the framework applies its own")
        let reasoning = NFKFoundationModelsDescriptors.requestOption(NFKParameterReasoningEffort)
        XCTAssertEqual(reasoning.choices.map { $0.value as? String },
                       [NFKReasoningEffortLight, NFKReasoningEffortModerate, NFKReasoningEffortDeep])
        XCTAssertEqual(reasoning.requirements?.minimumOSVersions["macOS"], "27.0")
    }

    func testTheLoadOptionsSetTheModelTheUseCaseAndTheGuardrails() throws {
        let catalog = NFKModelCatalog()
        catalog.addDescriptors(NFKFoundationModelsDescriptors.modelDescriptors())
        let identifier = NFKFoundationModelsDescriptors.systemLanguageModelIdentifier
        let options: [String: Any] = ["useCase": NFKFoundationModelUseCase.contentTagging.rawValue,
                                      "guardrails": NFKFoundationModelGuardrails.permissiveContentTransformations.rawValue]
        let backend = try XCTUnwrap(try catalog.backend(forModelIdentifier: identifier, options: options)
                                        as? NFKFoundationModelsBackend)
        XCTAssertEqual(backend.useCase, .contentTagging)
        XCTAssertEqual(backend.guardrails, .permissiveContentTransformations)
        XCTAssertEqual(backend.model, .onDevice, "the default model is on device")

        let model = try XCTUnwrap(descriptor.option(withKey: "model"))
        XCTAssertEqual(model.stage, NFKModelOptionStageLoad)
        let cloud = try XCTUnwrap(model.choices.first { ($0.value as? Int) == NFKFoundationModel.privateCloudCompute.rawValue })
        XCTAssertEqual(cloud.requirements?.minimumOSVersions["macOS"], "27.0")
        if #unavailable(macOS 27, iOS 27) {
            XCTAssertThrowsError(try catalog.backend(forModelIdentifier: identifier,
                                                     options: ["model": NFKFoundationModel.privateCloudCompute.rawValue]),
                                 "the catalog refuses Private Cloud Compute below macOS 27")
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
