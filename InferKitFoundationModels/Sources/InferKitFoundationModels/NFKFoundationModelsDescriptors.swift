//
//  NFKFoundationModelsDescriptors.swift
//  InferKitFoundationModels
//

import Foundation
import InferKit

/// The model descriptor of Apple's system language model, for `NFKModelCatalog`.
///
/// The shared catalog looks this class up by name, so linking the package is what lists the model.
/// The descriptor's engine is `NFKModelEngineFoundationModels`, its provider class is
/// `NFKFoundationModelsProvider`, and its inputs, options, and outputs are the key sets of
/// `NFKFoundationModelsBackend` on the running OS: the image inputs, the reasoning level, and the
/// reasoning and usage outputs appear on macOS 27 / iOS 27, where the framework takes and reports
/// them. Each request option has a title, help, and the range the framework takes. The load options
/// `model`, `useCase`, and `guardrails` set the backend properties of the same names; Private Cloud
/// Compute is a choice that needs macOS 27 / iOS 27. Its one variant's source is
/// `NFKModelSourceKindSystem`, because the model is part of the operating system. Introduced in
/// InferKit 0.4.0.
@objc(NFKFoundationModelsDescriptors)
public final class NFKFoundationModelsDescriptors: NSObject, NFKModelDescriptorProvider {

    /// The identifier of the system language model's descriptor: `foundation-models`.
    @objc public static let systemLanguageModelIdentifier = "foundation-models"

    /// The descriptor of the system language model.
    @objc public static func modelDescriptors() -> [NFKModelDescriptor] {
        [systemLanguageModelDescriptor()]
    }

    static func systemLanguageModelDescriptor() -> NFKModelDescriptor {
        let backend = NFKFoundationModelsBackend()
        let inputs = backend.supportedInputKeys.sorted().map { NFKModelPort(key: $0, required: false) }
        let outputs = backend.supportedOutputKeys.sorted().map { NFKModelPort.outputPort(withKey: $0, required: false) }
        let options = backend.supportedParameterKeys.sorted().map(requestOption(_:)) + loadOptions
        var tasks = [NFKModelTaskTextGeneration]
        if backend.supportedInputKeys.contains(NFKInputImage) {
            tasks.append(NFKModelTaskVisionQuestionAnswering)
        }
        let source = NFKModelSource(kind: NFKModelSourceKindSystem, repository: nil, revision: nil, url: nil,
                                    layout: NFKModelSourceLayoutNone, files: nil, requiresConversion: false,
                                    extensions: nil)
        let variant = NFKModelVariant(identifier: "system", displayName: "System model", isDefault: true,
                                      registryName: nil, source: source, downloadByteCount: nil,
                                      residentByteCount: nil, peakByteCount: nil, parameterCount: nil,
                                      precision: nil, minimumMemoryByteCount: nil, extensions: nil)
        let license = NFKModelLicense(identifier: nil, name: "Apple system framework", url: nil, gated: false,
                                     restrictsCommercialUse: false,
                                     note: "The model is part of the operating system and runs under its license.",
                                     extensions: nil)
        // Apple Intelligence, which the framework needs, runs only on Apple silicon.
        let requirements = NFKModelRequirements(minimumOSVersions: ["macOS": "26.0", "iOS": "26.0"],
                                                requiresAppleSilicon: true, minimumMemoryByteCount: nil,
                                                extensions: nil)
        return NFKModelDescriptor(identifier: systemLanguageModelIdentifier,
                                  displayName: "Apple Foundation Model",
                                  family: "apple-foundation-models",
                                  summary: "Apple's on-device system language model, with tools, structured output, and streaming.",
                                  engine: NFKModelEngineFoundationModels,
                                  registryName: nil,
                                  providerClassName: "NFKFoundationModelsProvider",
                                  tasks: tasks,
                                  inputModalities: nil,
                                  outputModalities: nil,
                                  inputs: inputs,
                                  outputs: outputs,
                                  options: options,
                                  variants: [variant],
                                  license: license,
                                  requirements: requirements,
                                  residencies: nil,
                                  standIn: false,
                                  introducedVersion: "0.4.0",
                                  extensions: nil)
    }

    /// A request option the backend reads, described for a user interface. The framework applies its own
    /// value where a request sets none, so the options state no defaults.
    static func requestOption(_ key: String) -> NFKModelOption {
        func option(_ kind: String, _ title: String, _ help: String, minimum: NSNumber? = nil, maximum: NSNumber? = nil,
                    step: NSNumber? = nil, choices: [NFKModelOptionChoice]? = nil,
                    requirements: NFKModelRequirements? = nil) -> NFKModelOption {
            NFKModelOption(key: key, kind: kind, title: title, help: help, defaultValue: nil, minimum: minimum,
                           maximum: maximum, step: step, choices: choices, stage: nil, group: nil,
                           requirements: requirements, propertyKeyPath: nil, memoryBytesPerUnit: nil, extensions: nil)
        }
        switch key {
        case NFKParameterTemperature:
            return option(NFKModelOptionKindNumber, "Temperature", "The sampling temperature. 0 is greedy decoding.",
                          minimum: 0, step: 0.05)
        case NFKParameterMaxTokens:
            return option(NFKModelOptionKindInteger, "Maximum tokens", "The most tokens the reply runs to.",
                          minimum: 1, step: 1)
        case NFKParameterTopK:
            return option(NFKModelOptionKindInteger, "Top-k", "Samples among the k most likely tokens.",
                          minimum: 1, step: 1)
        case NFKParameterTopP:
            return option(NFKModelOptionKindNumber, "Top-p", "Samples from the smallest set of tokens holding this probability mass.",
                          minimum: 0, maximum: 1, step: 0.05)
        case NFKParameterSeed:
            return option(NFKModelOptionKindInteger, "Seed", "The seed of the sampling, with top-k or top-p.",
                          minimum: 0, step: 1)
        case NFKParameterJSONSchema:
            return option(NFKModelOptionKindJSON, "JSON schema", "A JSON Schema the reply is generated to match.")
        case NFKParameterChoices:
            return option(NFKModelOptionKindTextList, "Choices", "The strings the reply must be one of.")
        case NFKParameterTools:
            return option(NFKModelOptionKindJSON, "Tools", "The tools the model may call, as function declarations.")
        case NFKParameterReasoningEffort:
            let levels = [(NFKReasoningEffortLight, "Light"), (NFKReasoningEffortModerate, "Moderate"),
                          (NFKReasoningEffortDeep, "Deep")]
            return option(NFKModelOptionKindChoice, "Reasoning", "How much the model reasons before it answers.",
                          choices: levels.map { NFKModelOptionChoice(value: $0.0, title: $0.1) },
                          requirements: macOS27)
        default:
            return NFKModelOption(key: key)
        }
    }

    /// The load options that set the backend's `model`, `useCase`, and `guardrails`, each a choice of the
    /// property's integer values with the property's default first.
    static var loadOptions: [NFKModelOption] {
        func choices(_ values: [(Int, String, NFKModelRequirements?)]) -> [NFKModelOptionChoice] {
            values.map { NFKModelOptionChoice(value: NSNumber(value: $0.0), title: $0.1, requirements: $0.2, memory: nil,
                                              memoryByVariant: nil, extensions: nil) }
        }
        func option(_ property: String, _ title: String, _ help: String,
                    _ values: [(Int, String, NFKModelRequirements?)]) -> NFKModelOption {
            NFKModelOption(key: property, kind: NFKModelOptionKindChoice, title: title, help: help,
                           defaultValue: NSNumber(value: values[0].0), minimum: nil, maximum: nil, step: nil,
                           choices: choices(values), stage: NFKModelOptionStageLoad, group: nil, requirements: nil,
                           propertyKeyPath: property, memoryBytesPerUnit: nil, extensions: nil)
        }
        return [
            option("model", "Model", "The on-device model, or Apple's larger model on Private Cloud Compute, where the request leaves the device and counts against a quota.",
                   [(NFKFoundationModel.onDevice.rawValue, "On device", nil),
                    (NFKFoundationModel.privateCloudCompute.rawValue, "Private Cloud Compute", macOS27)]),
            option("useCase", "Use case", "The on-device model's specialization. Private Cloud Compute ignores it.",
                   [(NFKFoundationModelUseCase.general.rawValue, "General", nil),
                    (NFKFoundationModelUseCase.contentTagging.rawValue, "Content tagging", nil)]),
            option("guardrails", "Guardrails", "The on-device model's content guardrails. Private Cloud Compute ignores them.",
                   [(NFKFoundationModelGuardrails.default.rawValue, "Default", nil),
                    (NFKFoundationModelGuardrails.permissiveContentTransformations.rawValue,
                     "Permissive content transformations", nil)]),
        ]
    }

    /// What Private Cloud Compute and the reasoning level need.
    private static let macOS27 = NFKModelRequirements(minimumOSVersions: ["macOS": "27.0", "iOS": "27.0"],
                                                      requiresAppleSilicon: true, minimumMemoryByteCount: nil,
                                                      extensions: nil)
}
