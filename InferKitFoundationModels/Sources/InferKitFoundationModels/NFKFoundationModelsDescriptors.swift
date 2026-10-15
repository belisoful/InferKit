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
/// `NFKFoundationModelsProvider`, and its inputs, parameters, and outputs are the key sets of
/// `NFKFoundationModelsBackend` on the running OS: the image inputs, the reasoning level, and the
/// reasoning and usage outputs appear on macOS 27 / iOS 27, where the framework takes and reports
/// them. Its one variant's source is `NFKModelSourceKindSystem`, because the model is part of the
/// operating system. Introduced in InferKit 0.4.0.
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
        let parameters = backend.supportedParameterKeys.sorted().map { NFKModelOption(key: $0) }
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
                                  parameters: parameters,
                                  variants: [variant],
                                  license: license,
                                  requirements: requirements,
                                  residencies: nil,
                                  standIn: false,
                                  introducedVersion: "0.4.0",
                                  extensions: nil)
    }
}
