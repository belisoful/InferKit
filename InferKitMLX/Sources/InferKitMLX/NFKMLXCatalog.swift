//
//  NFKMLXCatalog.swift
//  InferKitMLX
//
//  The package's model descriptors: the builder each catalog entry is written with, and the provider
//  class the core's model catalog loads them from.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The descriptors of the models InferKitMLX ships, for `NFKModelCatalog`.
///
/// @discussion `NFKModelCatalog.sharedCatalog` looks this class up by name, so linking InferKitMLX adds
/// its models to the shared catalog with no call from the app. Each descriptor's `engine` is
/// `NFKModelEngineMLX`, and its `registryName` is the name ``NFKMLXModelRegistry`` builds the model under
/// once ``NFKMLXReferenceModels/registerAll()`` has run. The descriptors include the reference stand-ins,
/// marked `standIn`, which the catalog's `visibleDescriptors` leaves out. Each descriptor is read through
/// `+[NFKModelDescriptor descriptorWithDictionary:error:]`, the reader a plugin's manifest goes through.
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXModelDescriptors)
public final class NFKMLXModelDescriptors: NSObject, NFKModelDescriptorProvider {

    /// The catalog entries of every model the package ships, in the order the catalog receives them.
    static var entries: [NFKMLXCatalogEntry] {
        NFKMLXCatalogReference.entries
            + NFKMLXCatalogImage.entries
            + NFKMLXCatalogVideo.entries
            + NFKMLXCatalogLanguage.entries
            + NFKMLXCatalogAudio.entries
    }

    /// The descriptor of every model the package ships, with the sizes, digests, and memory figures
    /// `Tools/catalog/measure.py` measured. An entry the descriptor reader refuses is left out.
    /// Introduced in InferKit 0.4.0.
    public static func modelDescriptors() -> [NFKModelDescriptor] {
        entries.compactMap { entry in
            do {
                return try entry.measured().descriptor()
            } catch {
                assertionFailure("the catalog entry \(entry.identifier) is malformed: \(error)")
                return nil
            }
        }
    }
}

/// One model's catalog entry, written as Swift and read into an `NFKModelDescriptor` through the
/// manifest reader.
///
/// @discussion The defaults are the package's own: engine `NFKModelEngineMLX`, macOS 14 and iOS 17 on
/// Apple silicon, resident weights, and InferKit 0.4.0 as the introducing version. An entry names only
/// the fields that differ from them.
struct NFKMLXCatalogEntry {
    var identifier: String
    var displayName: String? = nil
    var family: String
    var summary: String? = nil
    var engine: String = NFKModelEngineMLX
    var registryName: String? = nil
    var tasks: [String]
    var inputs: [NFKModelPort] = []
    var outputs: [NFKModelPort] = []
    var parameters: [NFKModelOption] = []
    var variants: [NFKModelVariant] = []
    var license: NFKModelLicense? = nil
    var minimumOSVersions: [String: String] = ["macOS": "14.0", "iOS": "17.0"]
    var requiresAppleSilicon = true
    var minimumMemoryByteCount: Int64? = nil
    var residencies: [String] = [NFKModelResidencyResident]
    var standIn = false
    var introducedVersion: String? = "0.4.0"

    /// The entry as a manifest model dictionary, the shape `descriptorWithDictionary:error:` reads.
    var dictionaryRepresentation: [String: Any] {
        let memory = minimumMemoryByteCount.map(NSNumber.init(value:))
        let requirements = NFKModelRequirements(minimumOSVersions: minimumOSVersions,
                                                requiresAppleSilicon: requiresAppleSilicon,
                                                minimumMemoryByteCount: memory, extensions: nil)
        var dictionary: [String: Any] = [
            "id": identifier,
            "family": family,
            "engine": engine,
            "tasks": tasks,
            "inputs": inputs.map { $0.dictionaryRepresentation() },
            "outputs": outputs.map { $0.dictionaryRepresentation() },
            "parameters": parameters.map { $0.dictionaryRepresentation() },
            "variants": variants.map { $0.dictionaryRepresentation() },
            "requirements": requirements.dictionaryRepresentation(),
            "residencies": residencies,
            "standIn": standIn,
        ]
        dictionary["displayName"] = displayName
        dictionary["summary"] = summary
        dictionary["registryName"] = registryName
        dictionary["license"] = license?.dictionaryRepresentation()
        dictionary["introducedVersion"] = introducedVersion
        return dictionary
    }

    /// The descriptor the manifest reader makes of ``dictionaryRepresentation``. Throws the reader's
    /// `NFKModelCatalogErrorMalformedDescriptor` when a field is malformed.
    func descriptor() throws -> NFKModelDescriptor {
        try NFKModelDescriptor(dictionary: dictionaryRepresentation)
    }
}

// MARK: - Entry helpers

extension NFKModelPort {

    /// An input port under `key`, typed from `NFKModelValueTypeForKey` unless `valueType` names a type.
    static func input(_ key: String, required: Bool = false, valueType: String? = nil,
                      title: String? = nil, help: String? = nil) -> NFKModelPort {
        NFKModelPort(key: key, valueType: valueType ?? NFKModelValueTypeForKey(key), required: required,
                     title: title, help: help, extensions: nil)
    }

    /// An output port under `key`, typed from `NFKModelValueTypeForOutputKey` unless `valueType` names a
    /// type. `required` states that every result carries the key.
    static func output(_ key: String, required: Bool = false, valueType: String? = nil,
                       title: String? = nil, help: String? = nil) -> NFKModelPort {
        NFKModelPort(key: key, valueType: valueType ?? NFKModelValueTypeForOutputKey(key), required: required,
                     title: title, help: help, extensions: nil)
    }
}

extension NFKModelOption {

    /// An option under `key`. A nil `kind` follows the key's value type, as `optionWithKey:` does.
    /// `choices` pairs each value with its title.
    static func option(_ key: String, kind: String? = nil, title: String? = nil, help: String? = nil,
                       defaultValue: Any? = nil, minimum: NSNumber? = nil, maximum: NSNumber? = nil,
                       step: NSNumber? = nil, choices: [(value: Any, title: String)] = [],
                       stage: String = NFKModelOptionStageRequest, group: String? = nil) -> NFKModelOption {
        NFKModelOption(key: key, kind: kind ?? NFKModelOption(key: key).kind, title: title, help: help,
                       defaultValue: defaultValue, minimum: minimum, maximum: maximum, step: step,
                       choices: choices.map { NFKModelOptionChoice(value: $0.value, title: $0.title) },
                       stage: stage, group: group, extensions: nil)
    }
}

extension NFKModelLicense {

    /// A license with an SPDX identifier, or none, and the page its text is published on.
    static func license(_ identifier: String?, name: String, url: String?, gated: Bool = false,
                        restrictsCommercialUse: Bool = false, note: String? = nil) -> NFKModelLicense {
        NFKModelLicense(identifier: identifier, name: name, url: url.flatMap(URL.init(string:)), gated: gated,
                        restrictsCommercialUse: restrictsCommercialUse, note: note, extensions: nil)
    }

    /// The Apache License 2.0.
    static func apache2(note: String? = nil) -> NFKModelLicense {
        license("Apache-2.0", name: "Apache License 2.0", url: "https://www.apache.org/licenses/LICENSE-2.0",
                note: note)
    }

    /// The MIT License, published at `url`, the release's own copy.
    static func mit(url: String, note: String? = nil) -> NFKModelLicense {
        license("MIT", name: "MIT License", url: url, note: note)
    }

    /// InferKit's own MIT license, for a model that is InferKit code and downloads no weights.
    static func inferKit(note: String? = nil) -> NFKModelLicense {
        mit(url: "https://github.com/belisoful/InferKit/blob/main/LICENSE", note: note)
    }
}

extension NFKModelFile {

    /// A file a variant reads from another Hugging Face repository than its source's, at `revision`, or at
    /// the repository's default branch when nil.
    static func file(_ path: String, role: String, optional: Bool = false, repository: String,
                     revision: String? = nil) -> NFKModelFile {
        NFKModelFile(path: path, byteCount: nil, sha256: nil, role: role, optional: optional, repository: repository,
                     revision: revision, url: nil, extensions: nil)
    }
}

extension NFKModelSource {

    /// A Hugging Face repository read at `revision`, loaded as `layout`.
    static func hub(_ repository: String, revision: String? = nil, layout: String = NFKModelSourceLayoutDirectory,
                    files: [NFKModelFile] = [], requiresConversion: Bool = false) -> NFKModelSource {
        NFKModelSource(kind: NFKModelSourceKindHuggingFace, repository: repository, revision: revision, url: nil,
                       layout: layout, files: files, requiresConversion: requiresConversion, extensions: nil)
    }

    /// The source with a note a user should read about it. `publiclyServed` false marks a repository its
    /// publisher no longer serves to the public: a local copy of the release still builds the model, and
    /// the measurement does not size it.
    func noting(_ note: String, publiclyServed: Bool = true) -> NFKModelSource {
        var extensions = self.extensions
        extensions["note"] = note
        if !publiclyServed {
            extensions["publiclyServed"] = false
        }
        return NFKModelSource(kind: kind, repository: repository, revision: revision, url: url, layout: layout,
                              files: files, requiresConversion: requiresConversion, extensions: extensions)
    }

    /// A model that downloads nothing and loads no files of its own.
    static var noDownload: NFKModelSource {
        NFKModelSource(kind: NFKModelSourceKindNone, repository: nil, revision: nil, url: nil,
                       layout: NFKModelSourceLayoutNone, files: nil, requiresConversion: false, extensions: nil)
    }
}

extension NFKModelVariant {

    /// A release of a model. The byte and parameter counts are left out when nil. `license` is the
    /// variant's own, where it differs from its descriptor's.
    static func variant(_ identifier: String, displayName: String? = nil, isDefault: Bool = false,
                        registryName: String? = nil, source: NFKModelSource? = nil,
                        downloadByteCount: Int64? = nil, residentByteCount: Int64? = nil,
                        peakByteCount: Int64? = nil, parameterCount: Int64? = nil, precision: String? = nil,
                        minimumMemoryByteCount: Int64? = nil, license: NFKModelLicense? = nil) -> NFKModelVariant {
        NFKModelVariant(identifier: identifier, displayName: displayName, isDefault: isDefault,
                        registryName: registryName, source: source,
                        downloadByteCount: downloadByteCount.map(NSNumber.init(value:)),
                        residentByteCount: residentByteCount.map(NSNumber.init(value:)),
                        peakByteCount: peakByteCount.map(NSNumber.init(value:)),
                        parameterCount: parameterCount.map(NSNumber.init(value:)), precision: precision,
                        minimumMemoryByteCount: minimumMemoryByteCount.map(NSNumber.init(value:)),
                        license: license, extensions: nil)
    }
}
