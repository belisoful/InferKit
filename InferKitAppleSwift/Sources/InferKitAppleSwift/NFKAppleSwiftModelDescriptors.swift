//
//  NFKAppleSwiftModelDescriptors.swift
//  InferKitAppleSwift
//

import Foundation
import InferKit

/// The model descriptors of this package's engines, for `NFKModelCatalog`, and the registry that
/// builds their backends.
///
/// The shared catalog looks this class up by name, so linking the package is what lists the engines:
/// `apple-speech-analyzer`, `apple-translation`, `vision-document`, and `vision-smudge`. Each
/// descriptor's identifier is its backend's `backendIdentifier` and its registry name, its engine is
/// `NFKModelEngineAppleSwift`, and its inputs, options, and outputs are the key sets of the backend
/// `backend(named:)` builds. Each has one variant whose source is `NFKModelSourceKindSystem`, because
/// the model is part of the operating system; the speech analyzer and the translator install their
/// language assets through the system on first use. Introduced in InferKit 0.4.0.
@objc(NFKAppleSwiftModelDescriptors)
public final class NFKAppleSwiftModelDescriptors: NSObject, NFKModelDescriptorProvider {

    private struct Engine {
        let identifier: String
        let displayName: String
        let family: String
        let summary: String
        let tasks: [String]
        let providerClassName: String?
        var requiredOptions: Set<String> = []
        /// The engine's own title and help for a parameter, where the shared description does not fit.
        var optionDescriptions: [String: (title: String, help: String)] = [:]
        let build: () -> any NFKInferenceBackend
    }

    private static let engines: [Engine] = [
        Engine(identifier: "apple-speech-analyzer", displayName: "Apple Speech Analyzer", family: "apple-speech",
               summary: "Transcribes speech on device with Apple's speech analyzer, with a time range for each span.",
               tasks: [NFKModelTaskTranscription], providerClassName: "NFKSpeechAnalyzerProvider",
               optionDescriptions: [NFKParameterSourceLanguage: (
                   "Language", "The language to transcribe, a locale identifier such as en-US. Without one, the user's current locale. A locale used the first time installs its speech assets, which is a download.")],
               build: { NFKSpeechAnalyzerBackend() }),
        Engine(identifier: "apple-translation", displayName: "Apple Translation", family: "apple-translation",
               summary: "Translates text on device with Apple's Translation framework.",
               tasks: [NFKModelTaskTranslation], providerClassName: "NFKTranslationProvider",
               requiredOptions: [NFKParameterTargetLanguage],
               build: { NFKTranslationBackend() }),
        Engine(identifier: "vision-document", displayName: "Vision Documents", family: "apple-vision",
               summary: "Reads a document's text with its paragraphs, lists, and tables.",
               tasks: [NFKModelTaskTextRecognition], providerClassName: nil,
               build: { NFKVisionDocumentBackend() }),
        Engine(identifier: "vision-smudge", displayName: "Vision Lens Smudge", family: "apple-vision",
               summary: "Judges whether a photograph was taken through a smudged lens.",
               tasks: [NFKModelTaskClassification], providerClassName: nil,
               build: { NFKVisionSmudgeBackend() }),
    ]

    /// The title and help each parameter an engine reads shows a user interface. The languages take no
    /// choices: the Translation framework lists them only asynchronously.
    private static let optionDescriptions: [String: (title: String, help: String)] = [
        NFKParameterSourceLanguage: ("Source language",
                                     "The language the text is in, a BCP-47 tag such as en. Without one, the language NaturalLanguage detects in the text."),
        NFKParameterTargetLanguage: ("Target language", "The language to translate into, a BCP-47 tag such as de."),
    ]

    private static func option(for key: String, required: Bool,
                               description own: (title: String, help: String)?) -> NFKModelOption {
        let kind = NFKModelOption(key: key).kind
        let description = own ?? optionDescriptions[key]
        return NFKModelOption(key: key, kind: kind, title: description?.title, help: description?.help,
                              defaultValue: nil, minimum: nil, maximum: nil, step: nil, choices: nil, stage: nil,
                              group: nil, requirements: nil, propertyKeyPath: nil, memoryBytesPerUnit: nil,
                              required: required, extensions: nil)
    }

    /// The registry names `backend(named:)` builds, sorted.
    @objc public static var registryNames: [String] {
        engines.map(\.identifier).sorted()
    }

    /// A new backend for a descriptor's registry name, or nil for a name this registry does not know.
    @objc(backendNamed:)
    public static func backend(named registryName: String) -> (any NFKInferenceBackend)? {
        engines.first { $0.identifier == registryName }?.build()
    }

    /// The backend the catalog's build entry asks for: ``backend(named:)`` of the variant's registry name,
    /// else the descriptor's, with the request's property options set on it.
    ///
    /// - Throws: `NFKModelCatalogErrorUnknownModel` for a name this registry does not know, and
    ///   `NFKModelCatalogErrorInvalidOptions` when the backend has no property a property option names.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendForBuildRequest:error:)
    public static func backend(for request: NFKModelBuildRequest) throws -> any NFKInferenceBackend {
        let name = request.variant?.registryName ?? request.descriptor.registryName ?? request.descriptor.identifier
        guard let backend = backend(named: name) else {
            throw NSError(domain: NFKModelCatalogErrorDomain, code: NFKModelCatalogError.Code.unknownModel.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "The Apple Swift engines have no backend named \"\(name)\"."])
        }
        try request.applyPropertyOptions(toBackend: backend)
        return backend
    }

    /// The descriptors of this package's engines, sorted by identifier.
    @objc public static func modelDescriptors() -> [NFKModelDescriptor] {
        engines.sorted { $0.identifier < $1.identifier }.map(descriptor(of:))
    }

    private static func descriptor(of engine: Engine) -> NFKModelDescriptor {
        let backend = engine.build()
        let inputKeys = backend.supportedInputKeys ?? []
        let outputKeys = backend.supportedOutputKeys ?? []
        let parameterKeys = backend.supportedParameterKeys ?? []
        let inputs = inputKeys.sorted().map { NFKModelPort(key: $0, required: inputKeys.count == 1) }
        // Every engine here writes each key it declares on every result.
        let outputs = outputKeys.sorted().map { NFKModelPort.outputPort(withKey: $0, required: true) }
        let options = parameterKeys.sorted().map {
            option(for: $0, required: engine.requiredOptions.contains($0), description: engine.optionDescriptions[$0])
        }
        let source = NFKModelSource(kind: NFKModelSourceKindSystem, repository: nil, revision: nil, url: nil,
                                    layout: NFKModelSourceLayoutNone, files: nil, requiresConversion: false,
                                    extensions: nil)
        let variant = NFKModelVariant(identifier: "system", displayName: "System model", isDefault: true,
                                      registryName: engine.identifier, source: source, downloadByteCount: nil,
                                      residentByteCount: nil, peakByteCount: nil, parameterCount: nil,
                                      precision: nil, minimumMemoryByteCount: nil, extensions: nil)
        let license = NFKModelLicense(identifier: nil, name: "Apple system framework", url: nil, gated: false,
                                      restrictsCommercialUse: false,
                                      note: "The model is part of the operating system and runs under its license.",
                                      extensions: nil)
        let requirements = NFKModelRequirements(minimumOSVersions: ["macOS": "26.0", "iOS": "26.0"],
                                                requiresAppleSilicon: false, minimumMemoryByteCount: nil,
                                                extensions: nil)
        return NFKModelDescriptor(identifier: engine.identifier,
                                  displayName: engine.displayName,
                                  family: engine.family,
                                  summary: engine.summary,
                                  engine: NFKModelEngineAppleSwift,
                                  registryName: engine.identifier,
                                  providerClassName: engine.providerClassName,
                                  tasks: engine.tasks,
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
}
