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
        let build: () -> any NFKInferenceBackend
    }

    private static let engines: [Engine] = [
        Engine(identifier: "apple-speech-analyzer", displayName: "Apple Speech Analyzer", family: "apple-speech",
               summary: "Transcribes speech on device with Apple's speech analyzer, with a time range for each span.",
               tasks: [NFKModelTaskTranscription], providerClassName: "NFKSpeechAnalyzerProvider",
               build: { NFKSpeechAnalyzerBackend() }),
        Engine(identifier: "apple-translation", displayName: "Apple Translation", family: "apple-translation",
               summary: "Translates text on device with Apple's Translation framework.",
               tasks: [NFKModelTaskTranslation], providerClassName: "NFKTranslationProvider",
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

    /// The registry names `backend(named:)` builds, sorted.
    @objc public static var registryNames: [String] {
        engines.map(\.identifier).sorted()
    }

    /// A new backend for a descriptor's registry name, or nil for a name this registry does not know.
    @objc(backendNamed:)
    public static func backend(named registryName: String) -> (any NFKInferenceBackend)? {
        engines.first { $0.identifier == registryName }?.build()
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
        let options = parameterKeys.sorted().map { NFKModelOption(key: $0) }
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
