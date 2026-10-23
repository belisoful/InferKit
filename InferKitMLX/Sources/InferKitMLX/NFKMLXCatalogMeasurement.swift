//
//  NFKMLXCatalogMeasurement.swift
//  InferKitMLX
//
//  How a catalog entry takes its measurements: the rows `Tools/catalog/measure.py` writes into
//  NFKMLXCatalogMeasurements.swift, the precision each model's loader holds its weights at, and the
//  memory figures read from the two.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// One measured file: its size and, for a file Git LFS stores, the SHA-256 the repository records.
struct NFKMLXMeasuredFile {
    /// The Hugging Face repository, or the download URL of a URL source.
    let location: String
    /// The revision the catalog names. nil for the default branch, and for a URL.
    let revision: String?
    /// The path inside the repository, or the file name of a URL.
    let path: String
    let bytes: Int64
    let sha256: String?
}

/// One variant's measurement, keyed by the descriptor's and the variant's identifiers.
struct NFKMLXMeasuredVariant {

    /// What the measurement reached.
    enum Status: String {
        /// Every file the download fetches was found and sized.
        case measured
        /// The repository refused the listing without an access token whose account accepted its terms.
        case gated
        /// The source is a page or a host that serves no size: a release page or a Google Drive folder.
        case unmeasurable
        /// A file the download requires is not served.
        case missing
    }

    let model: String
    let variant: String
    let status: Status
    /// Why a measurement is incomplete, or which file is missing.
    let note: String?
    /// The commit the source's revision resolved to when it was measured.
    let commit: String?
    /// The bytes the download transfers: the listed files the repository serves, the first weight
    /// candidate it serves, and the shards a shard index names.
    let downloadBytes: Int64?
    /// The elements of every safetensors tensor the download holds. A packed format (MXFP4 blocks,
    /// quantized words) counts its stored elements.
    let parameterCount: Int64?
    /// The safetensors tensor bytes as stored, by top-level directory ("" for the root) and dtype.
    let tensorBytes: [String: [String: Int64]]
    /// The bytes of weight files whose tensors the measurement does not read (a pickle, a GGUF, an
    /// archive), by top-level directory.
    let opaqueBytes: [String: Int64]
    /// The repeated layers of each top-level directory's largest layer container.
    var layers: [String: NFKMLXMeasuredLayers] = [:]
    /// The routed-expert tensor bytes as stored (names holding `.experts.`), by top-level directory and dtype.
    var expertBytes: [String: [String: Int64]] = [:]
    /// The elements a runtime weight quantization packs (rank-2 `.weight` tensors whose input width
    /// divides 64, outside the embeddings and the norms), by top-level directory.
    var quantizableElements: [String: Int64] = [:]
    /// The geometry the repository's `config.json` states, or nil where it has none or refuses it.
    var geometry: NFKMLXMeasuredGeometry? = nil
}

/// The repeated layers of one directory: how many, and the bytes of the largest as stored, by dtype.
struct NFKMLXMeasuredLayers {
    let count: Int
    let largestBytes: [String: Int64]
}

/// The fields of a release's `config.json` the key-value cache follows, read from its root and from
/// `text_config` where the release nests its decoder. A field the configuration does not state is nil.
struct NFKMLXMeasuredGeometry {
    var layerCount: Int? = nil
    var attentionHeads: Int? = nil
    var keyValueHeads: Int? = nil
    var headDimension: Int? = nil
    var hiddenSize: Int? = nil
    var maximumPositions: Int? = nil
    var slidingWindow: Int? = nil
    var slidingWindowPattern: Int? = nil
    var sharedKeyValueLayers: Int? = nil
    var globalHeadDimension: Int? = nil
    var globalKeyValueHeads: Int? = nil
    var vocabularySize: Int? = nil
    /// Each layer's kind, as `layer_types`, `layers_block_type`, or `hybrid_override_pattern` names it.
    var layerTypes: [String] = []
}

/// The measurements `Tools/catalog/measure.py` generated, looked up by file and by variant.
enum NFKMLXCatalogMeasurements {

    private static func fileKey(_ location: String, _ revision: String?, _ path: String) -> String {
        "\(location)|\(revision ?? "")|\(path)"
    }

    private static let filesByKey: [String: NFKMLXMeasuredFile] = Dictionary(
        measuredFiles.map { (fileKey($0.location, $0.revision, $0.path), $0) }, uniquingKeysWith: { first, _ in first })

    private static let variantsByKey: [String: NFKMLXMeasuredVariant] = Dictionary(
        measuredVariants.map { ("\($0.model)/\($0.variant)", $0) }, uniquingKeysWith: { first, _ in first })

    /// The measurement of a file of `source`, read where the file says it comes from.
    static func file(_ file: NFKModelFile, of source: NFKModelSource) -> NFKMLXMeasuredFile? {
        if let repository = file.repository {
            return filesByKey[fileKey(repository, file.revision, file.path)]
        }
        if let url = file.url ?? source.url, source.kind == NFKModelSourceKindURL || file.url != nil {
            return filesByKey[fileKey(url.absoluteString, nil, file.path)]
        }
        guard let repository = source.repository else { return nil }
        return filesByKey[fileKey(repository, file.revision ?? source.revision, file.path)]
    }

    /// The measurement of a descriptor's variant.
    static func variant(model: String, variant: String) -> NFKMLXMeasuredVariant? {
        variantsByKey["\(model)/\(variant)"]
    }
}

// MARK: - Load precision

/// The type a loader holds a group of weights in.
enum NFKMLXLoadPrecision {
    /// Each tensor's stored type.
    case stored
    /// Floating tensors widened or narrowed to float32; integer and packed tensors as stored.
    case float32
    /// Floating tensors held as bfloat16.
    case bfloat16
}

/// How a model's registered factory holds its weights: the precision of each top-level directory of
/// its release, and the directories each stage loads together where the model stages.
struct NFKMLXLoadPlan {
    /// The precision of a directory the plan does not name. nil where the loader's precision is not
    /// known, which estimates the stored bytes.
    var precision: NFKMLXLoadPrecision?
    var directories: [String: NFKMLXLoadPrecision] = [:]
    /// The directories each stage loads, in the order the stages run. Empty for a model that holds its
    /// weights together.
    var stages: [[String]] = []
    /// The directories a precision option converts. A stage's figure at a precision holds these at it and
    /// the stage's other directories as the plan holds them. Empty where every directory converts.
    var castDirectories: Set<String> = []

    static let stored = NFKMLXLoadPlan(precision: .stored)
    static let float32 = NFKMLXLoadPlan(precision: .float32)
    static let bfloat16 = NFKMLXLoadPlan(precision: .bfloat16)
    static let unknown = NFKMLXLoadPlan(precision: nil)

    func precision(ofDirectory directory: String) -> NFKMLXLoadPrecision? {
        directories[directory] ?? precision
    }

    /// The bytes `tensorBytes` (dtype → stored bytes) hold at `precision`.
    static func bytes(_ tensorBytes: [String: Int64], at precision: NFKMLXLoadPrecision) -> Int64 {
        tensorBytes.reduce(0) { total, entry in
            guard let size = floatingElementBytes[entry.key] else { return total + entry.value }
            switch precision {
            case .stored: return total + entry.value
            case .float32: return total + entry.value / size * 4
            case .bfloat16: return total + entry.value / size * 2
            }
        }
    }

    /// The element size of each floating safetensors dtype.
    static let floatingElementBytes: [String: Int64] = ["F64": 8, "F32": 4, "F16": 2, "BF16": 2,
                                                        "F8_E4M3": 1, "F8_E5M2": 1]

    /// The manifest's precision name of each floating safetensors dtype.
    static let precisionNames = ["F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
                                 "F8_E4M3": "float8_e4m3", "F8_E5M2": "float8_e5m2"]
}

extension NFKMLXLoadPlan {

    /// The plan of the model an entry describes, read from its registered factory's loader.
    static func plan(for entry: NFKMLXCatalogEntry) -> NFKMLXLoadPlan {
        if let plan = modelPlans[entry.identifier] {
            return plan
        }
        if NFKMLXCatalogImage.upcastingAutoencoderNames.contains(entry.identifier) {
            return upcastAutoencoder
        }
        return familyPlans[entry.family] ?? .stored
    }

    /// An SDXL release: the autoencoder at float32, which `force_upcast` widens it to under a 16-bit load,
    /// and the other networks as stored or as the precision option converts them.
    private static let upcastAutoencoder: NFKMLXLoadPlan = {
        var plan = NFKMLXLoadPlan(precision: .stored, directories: ["vae": .float32])
        plan.castDirectories = ["unet", "text_encoder", "text_encoder_2"]
        return plan
    }()

    /// The models whose plan differs from their family's.
    private static let modelPlans: [String: NFKMLXLoadPlan] = [
        NFKMLXTranslateGemma.modelName: .stored,
        NFKMLXQwen3Embedding.modelName: .float32,
        NFKMLXFluxControlNetGenerator.modelName: staged(textEncoders(clip: .stored, t5: .float32), denoiser: [:],
                                                        controlNet: .stored, vae: .stored),
        NFKMLXSD3ControlNetGenerator.modelName: sd3(controlNet: .stored),
    ]

    /// The families' plans. A family missing here loads its weights as stored.
    private static let familyPlans: [String: NFKMLXLoadPlan] = [
        "birefnet": .float32, "cosmos-tokenizer": .float32, "deberta-v3": .float32, "florence-2": .float32,
        "gemma4": .float32, "laya": .float32, "m2m100": .float32, "marian": .float32,
        "modernbert": .float32, "nllb": .float32, "pixtral": .float32, "qwen3-vl": .float32, "smollm2": .float32,
        "smolvlm": .float32, "trocr": .float32,
        "deepseek-v4": .bfloat16, "madlad": .bfloat16, "sa2va": .bfloat16,
        // The decoder loads as stored and the image and speech towers at float32, from the same shards.
        "phi-4-multimodal": .unknown,
        "flux": staged(textEncoders(clip: .stored, t5: .float32), denoiser: [:], controlNet: nil, vae: .stored),
        // The text encoder widens to float32 only where the budget holds it, so the plan counts it stored.
        "flux2": staged(["text_encoder": .stored], denoiser: [:], controlNet: nil, vae: .stored),
        "stable-diffusion-3": sd3(controlNet: nil),
        "sana": staged(["text_encoder": .float32], denoiser: ["transformer": .float32], controlNet: nil, vae: .stored),
        "z-image": staged(["text_encoder": .stored], denoiser: ["transformer": .bfloat16], controlNet: nil,
                          vae: .stored),
        "qwen-image": staged(["text_encoder": .stored], denoiser: [:], controlNet: nil, vae: .float32),
        "ltx-video": staged(["text_encoder": .float32], denoiser: ["transformer": .float32], controlNet: nil,
                            vae: .stored),
        // umT5 widens to float32 only where the budget holds it, so the plan counts it at bfloat16. The
        // precision option converts the transformer alone.
        "wan": {
            var plan = staged(["text_encoder": .bfloat16], denoiser: [:], controlNet: nil, vae: .float32)
            plan.castDirectories = ["transformer"]
            return plan
        }(),
        "minimax-music": NFKMLXLoadPlan(precision: .stored, directories: ["transformer": .float32],
                                        stages: [["language_model", "rvq_depth_decoder"],
                                                 ["condition_encoder", "transformer"], ["vocoder"]]),
        "stable-audio": NFKMLXLoadPlan(precision: .float32,
                                       stages: [["projection_model", "text_encoder"], ["transformer", "vae"]]),
    ]

    /// FLUX.1's two text encoders: CLIP-L in `text_encoder` and T5-XXL in `text_encoder_2`.
    private static func textEncoders(clip: NFKMLXLoadPrecision, t5: NFKMLXLoadPrecision) -> [String: NFKMLXLoadPrecision] {
        ["text_encoder": clip, "text_encoder_2": t5]
    }

    /// Stable Diffusion 3: two CLIP encoders at float32, T5-XXL at bfloat16 where float32 does not fit,
    /// the transformer at bfloat16, and the autoencoder at float32.
    private static func sd3(controlNet: NFKMLXLoadPrecision?) -> NFKMLXLoadPlan {
        staged(["text_encoder": .float32, "text_encoder_2": .float32, "text_encoder_3": .bfloat16],
               denoiser: ["transformer": .bfloat16], controlNet: controlNet, vae: .float32)
    }

    /// A generator whose text encoders load as the first stage and whose denoiser, ControlNet (the root of
    /// its own repository), and autoencoder load as the second.
    private static func staged(_ encoders: [String: NFKMLXLoadPrecision], denoiser: [String: NFKMLXLoadPrecision],
                               controlNet: NFKMLXLoadPrecision?, vae: NFKMLXLoadPrecision) -> NFKMLXLoadPlan {
        var directories = encoders.merging(denoiser) { first, _ in first }
        directories["vae"] = vae
        var second = ["transformer", "vae"]
        if let controlNet {
            directories[""] = controlNet
            second.append("")
        }
        return NFKMLXLoadPlan(precision: .stored, directories: directories,
                              stages: [encoders.keys.sorted(), second])
    }
}

// MARK: - Taking the measurements

extension NFKMLXCatalogEntry {

    /// The entry with each variant's measured file sizes and digests, download bytes, parameter count,
    /// and memory facts, and the load options, memory model, and presets its profile declares. A
    /// variant with memory facts takes its resident bytes and least working set from the estimate
    /// `NFKMLXModelDescriptors` derives; a variant without them takes the measured figures here.
    func measured() -> NFKMLXCatalogEntry {
        measured { NFKMLXCatalogMeasurements.variant(model: identifier, variant: $0.identifier) }
    }

    /// The entry measured as ``measured()`` describes, with each variant's measurement from `measurement`,
    /// loaded as `plan` stages it, else as its family's load plan does.
    func measured(by measurement: (NFKModelVariant) -> NFKMLXMeasuredVariant?,
                  plan planOverride: NFKMLXLoadPlan? = nil) -> NFKMLXCatalogEntry {
        let plan = planOverride ?? NFKMLXLoadPlan.plan(for: self)
        let profile = NFKMLXLoadProfile.profile(for: self)
        var entry = self
        var stagePrecisions = [String: String]()
        var maximumContextLength: Int?
        entry.variants = variants.map { variant in
            let (measuredVariant, precisions) = measured(variant, measurement: measurement(variant), plan: plan,
                                                         profile: profile)
            stagePrecisions.merge(precisions) { first, _ in first }
            if let length = measuredVariant.memoryFacts?.maximumContextLength?.intValue {
                maximumContextLength = min(maximumContextLength ?? length, length)
            }
            return measuredVariant
        }
        let declaration = (profile ?? .plain).declaration(for: entry, stagePrecisions: stagePrecisions,
                                                          maximumContextLength: maximumContextLength, drafts: [])
        entry.loadOptions = declaration.options
        entry.memoryModel = declaration.memory
        entry.presets = declaration.presets
        let positions = variants.compactMap { measurement($0)?.geometry?.maximumPositions }.min()
        if let limit = outputTokenLimit ?? positions {
            entry.parameters = entry.parameters.map { $0.key == NFKParameterMaxTokens && $0.maximum == nil ? $0.bounded(by: limit) : $0 }
        }
        return entry
    }

    /// The entry with the draft-model options for `drafts`, declared after every entry is measured.
    func declaringDrafts(_ drafts: [NFKMLXDraftChoice]) -> NFKMLXCatalogEntry {
        guard !drafts.isEmpty, let profile = NFKMLXLoadProfile.profile(for: self), profile.drafts else {
            return self
        }
        let maximumContextLength = variants.compactMap { $0.memoryFacts?.maximumContextLength?.intValue }.min()
        var entry = self
        let declaration = profile.declaration(for: entry, stagePrecisions: memoryModel?.stagePrecisions ?? [:],
                                              maximumContextLength: maximumContextLength, drafts: drafts)
        entry.loadOptions = declaration.options
        entry.memoryModel = declaration.memory
        entry.presets = declaration.presets
        return entry
    }

    private func measured(_ variant: NFKModelVariant, measurement: NFKMLXMeasuredVariant?, plan: NFKMLXLoadPlan,
                          profile: NFKMLXLoadProfile?) -> (NFKModelVariant, [String: String]) {
        guard let source = variant.source else {
            return (variant, [:])
        }
        var sourceExtensions = source.extensions
        if let commit = measurement?.commit, commit != source.revision {
            sourceExtensions["measuredRevision"] = commit
        }
        let files = source.files.map { file -> NFKModelFile in
            guard let measuredFile = NFKMLXCatalogMeasurements.file(file, of: source) else { return file }
            return NFKModelFile(path: file.path, byteCount: NSNumber(value: measuredFile.bytes),
                                sha256: measuredFile.sha256 ?? file.sha256, role: file.role, optional: file.isOptional,
                                repository: file.repository, revision: file.revision, url: file.url,
                                extensions: file.extensions)
        }
        let measuredSource = NFKModelSource(kind: source.kind, repository: source.repository, revision: source.revision,
                                            url: source.url, layout: source.layout, files: files,
                                            requiresConversion: source.requiresConversion,
                                            extensions: sourceExtensions)
        var extensions = variant.extensions
        var figures = NFKMLXVariantFigures()
        var facts: NFKMLXVariantFacts?
        if let measurement {
            if measurement.status != .measured {
                extensions["measurement"] = measurement.status.rawValue
            }
            if let note = measurement.note {
                extensions["measurementNote"] = note
            }
            figures = NFKMLXVariantFigures(measurement, plan: plan, converted: source.requiresConversion)
            if figures.estimatesStoredBytes {
                extensions["residentEstimate"] = "stored-dtype"
            }
            facts = NFKMLXVariantFacts(measurement, plan: plan, converted: source.requiresConversion,
                                       recipe: NFKMLXFactsRecipe(entry: self, profile: profile),
                                       estimatesStoredBytes: figures.estimatesStoredBytes)
        }
        let derived = variant.memoryFacts == nil && facts != nil
        let measuredVariant = NFKModelVariant(
            identifier: variant.identifier, displayName: variant.displayName, isDefault: variant.isDefault,
            registryName: variant.registryName, source: measuredSource,
            downloadByteCount: variant.downloadByteCount ?? figures.downloadBytes.map(NSNumber.init(value:)),
            residentByteCount: variant.residentByteCount ?? (derived ? nil : figures.residentBytes.map(NSNumber.init(value:))),
            peakByteCount: variant.peakByteCount,
            parameterCount: variant.parameterCount ?? figures.parameterCount.map(NSNumber.init(value:)),
            precision: variant.precision ?? figures.precision,
            minimumMemoryByteCount: variant.minimumMemoryByteCount
                ?? (derived ? nil : figures.minimumMemoryBytes.map(NSNumber.init(value:))),
            license: variant.license, memoryFacts: variant.memoryFacts ?? facts?.facts, extensions: extensions)
        return (measuredVariant, facts?.stagePrecisions ?? [:])
    }
}

// MARK: - Memory facts

/// Which figures a variant's memory facts carry, read from its entry's profile and residencies.
///
/// @discussion A figure a loader cannot act on is left out, so the residency planner never picks a
/// residency the model's factory does not offer: routed experts only where the entry pages, repeated
/// layers only where it streams, quantizable elements only where its factory quantizes, and the
/// key-value cache only where a context-length option sets it.
struct NFKMLXFactsRecipe {
    var stageIdentifier = "model"
    var quantizes = false
    var pagesExperts = false
    var streamsLayers = false
    var keyValue: NFKMLXKeyValueRule? = nil
    var widenedStages: [String: String] = [:]

    init(stageIdentifier: String = "model", quantizes: Bool = false, pagesExperts: Bool = false,
         streamsLayers: Bool = false, keyValue: NFKMLXKeyValueRule? = nil, widenedStages: [String: String] = [:]) {
        self.stageIdentifier = stageIdentifier
        self.quantizes = quantizes
        self.pagesExperts = pagesExperts
        self.streamsLayers = streamsLayers
        self.keyValue = keyValue
        self.widenedStages = widenedStages
    }

    init(entry: NFKMLXCatalogEntry, profile: NFKMLXLoadProfile?) {
        stageIdentifier = profile?.stageIdentifier ?? "model"
        quantizes = profile?.quantizes ?? false
        pagesExperts = profile != nil && entry.residencies.contains(NFKModelResidencyPaged)
        streamsLayers = profile != nil && entry.residencies.contains(NFKModelResidencyStreamed)
        keyValue = profile?.contextLength != nil ? profile?.keyValue : nil
        widenedStages = profile?.widenedStages ?? [:]
    }
}

/// A variant's memory facts and the precision each stage loads at.
struct NFKMLXVariantFacts {
    let facts: NFKModelMemoryFacts
    /// Each stage's precision under the load plan: an `NFKModelPrecision*` string, or
    /// ``loadedPrecision`` for a stage whose directories load at different precisions.
    let stagePrecisions: [String: String]

    /// The key of a stage's bytes as its directories load them, where they load at different precisions.
    static let loadedPrecision = "loaded"

    /// The facts of a measured variant, or nil where the measurement sized no weights.
    init?(_ measurement: NFKMLXMeasuredVariant, plan: NFKMLXLoadPlan, converted: Bool, recipe: NFKMLXFactsRecipe,
          estimatesStoredBytes: Bool) {
        let directories = Set(measurement.tensorBytes.keys).union(measurement.opaqueBytes.keys)
        guard measurement.downloadBytes != nil, !directories.isEmpty else {
            return nil
        }
        var stages = [NFKModelStageFacts]()
        var precisions = [String: String]()
        for split in plan.stageSplit(of: directories, singleStage: recipe.stageIdentifier) {
            let loads = split.directories.map { directory in
                (directory, converted ? NFKMLXLoadPrecision.stored : plan.precision(ofDirectory: directory) ?? .stored)
            }
            let bytes = Self.bytesByPrecision(loads, tensors: measurement.tensorBytes, opaque: measurement.opaqueBytes,
                                              converting: plan.castDirectories)
            let floating = Set(loads.filter { measurement.tensorBytes[$0.0]?.isEmpty == false }.map(\.1))
            let precision = floating.count > 1 ? Self.loadedPrecision : Self.name(of: floating.first ?? .stored)
            precisions[split.identifier] = precision
            let pageable = recipe.pagesExperts
                ? Self.bytesByPrecision(loads, tensors: measurement.expertBytes, opaque: [:]) : [:]
            var layerCount = 0
            var largestLayer = [String: NSNumber]()
            if recipe.streamsLayers,
               let (directory, layers) = loads.compactMap({ load in measurement.layers[load.0].map { (load, $0) } })
                .max(by: { $0.1.count < $1.1.count }) {
                layerCount = layers.count
                largestLayer = Self.bytesByPrecision([directory], tensors: [directory.0: layers.largestBytes], opaque: [:])
            }
            let quantizable = recipe.quantizes ? split.directories.reduce(Int64(0)) {
                $0 + (measurement.quantizableElements[$1] ?? 0)
            } : 0
            stages.append(NFKModelStageFacts(identifier: split.identifier, title: split.title,
                                             byteCountByPrecision: bytes, quantizableElementCount: Int(quantizable),
                                             pageableByteCountByPrecision: pageable, layerCount: layerCount,
                                             largestLayerByteCountByPrecision: largestLayer,
                                             widenedPrecision: recipe.widenedStages[split.identifier], extensions: nil))
        }
        let geometry = measurement.geometry
        let keyValue = geometry.flatMap { recipe.keyValue?.keyValueBytes(of: $0) }
        facts = NFKModelMemoryFacts(
            stages: stages, keyValueByteCountPerToken: keyValue.map { NSNumber(value: $0.perToken) },
            boundedKeyValueByteCount: keyValue.flatMap { $0.bounded > 0 ? NSNumber(value: $0.bounded) : nil },
            maximumContextLength: keyValue != nil ? geometry?.maximumPositions.map(NSNumber.init(value:)) : nil,
            vocabularySize: geometry?.vocabularySize.map(NSNumber.init(value:)), loadOverheadByteCount: 0,
            basis: estimatesStoredBytes ? NFKModelMemoryBasisStoredDType : NFKModelMemoryBasisMeasured, extensions: nil)
        stagePrecisions = precisions
    }

    /// The manifest's precision of a load precision.
    static func name(of precision: NFKMLXLoadPrecision) -> String {
        switch precision {
        case .stored: return NFKModelPrecisionCheckpoint
        case .float32: return NFKModelPrecisionFloat32
        case .bfloat16: return NFKModelPrecisionBFloat16
        }
    }

    /// The bytes of directories at the checkpoint's types, at float32, and at bfloat16, and, where the
    /// directories load at different precisions, as they load. Opaque bytes count as stored at every
    /// precision. A stage of opaque bytes alone states only the checkpoint's. Where `converting` names
    /// some of a stage's directories and not all, the figures at the three precisions hold the others as
    /// they load.
    static func bytesByPrecision(_ loads: [(String, NFKMLXLoadPrecision)], tensors: [String: [String: Int64]],
                                 opaque: [String: Int64], converting: Set<String> = []) -> [String: NSNumber] {
        let opaqueBytes = loads.reduce(Int64(0)) { $0 + (opaque[$1.0] ?? 0) }
        let held = loads.filter { tensors[$0.0]?.isEmpty == false }
        guard !held.isEmpty || opaqueBytes > 0 else {
            return [:]
        }
        let mixed = held.contains { converting.contains($0.0) } && !held.allSatisfy { converting.contains($0.0) }
        let converts = !mixed
        func total(_ precision: (NFKMLXLoadPrecision) -> NFKMLXLoadPrecision) -> NSNumber {
            NSNumber(value: held.reduce(opaqueBytes) { sum, load in
                let held = converts || converting.contains(load.0) ? precision(load.1) : load.1
                return sum + NFKMLXLoadPlan.bytes(tensors[load.0] ?? [:], at: held)
            })
        }
        var bytes = [NFKModelPrecisionCheckpoint: total { _ in .stored }]
        guard !held.isEmpty else {
            return bytes
        }
        bytes[NFKModelPrecisionFloat32] = total { _ in .float32 }
        bytes[NFKModelPrecisionBFloat16] = total { _ in .bfloat16 }
        if Set(held.map(\.1)).count > 1 {
            bytes[loadedPrecision] = total { $0 }
        }
        return bytes
    }
}

/// One stage of a measured variant: the directories it loads together.
struct NFKMLXStageSplit {
    let identifier: String
    let title: String?
    let directories: [String]
}

extension NFKMLXLoadPlan {

    /// The stages a variant's directories load in: the plan's stages, the directories they leave out
    /// held with the last, or one stage of every directory where the plan names no stages. A stage takes
    /// its first directory's name as its identifier, `model` for the root.
    func stageSplit(of directories: Set<String>, singleStage identifier: String) -> [NFKMLXStageSplit] {
        guard !stages.isEmpty else {
            return [NFKMLXStageSplit(identifier: identifier, title: nil, directories: directories.sorted())]
        }
        let named = Set(stages.joined())
        let leftOver = directories.subtracting(named).sorted()
        return stages.enumerated().map { index, stage in
            let held = stage + (index == stages.count - 1 ? leftOver : [])
            let first = stage.first ?? ""
            return NFKMLXStageSplit(identifier: first.isEmpty ? "model" : first,
                                    title: held.map { $0.isEmpty ? "root" : $0 }.joined(separator: ", "),
                                    directories: held)
        }
    }
}

/// The figures a variant's measurement and its model's load plan give.
struct NFKMLXVariantFigures {
    var downloadBytes: Int64?
    var parameterCount: Int64?
    /// The tensor bytes at the precision the loader holds them in.
    var residentBytes: Int64?
    /// ``NFKMLXResidencyBudget/leastWorkingSet(holding:)`` of the largest stage, or of the resident bytes.
    var minimumMemoryBytes: Int64?
    /// The floating type that holds the most stored bytes.
    var precision: String?
    /// True where the resident bytes are the stored bytes because the loader's precision, or the type of
    /// the weights it reads, is not known.
    var estimatesStoredBytes = false

    init() {}

    /// `converted` is true for a source a converter turns into the file the loader reads, whose type the
    /// measurement does not see.
    init(_ measurement: NFKMLXMeasuredVariant, plan: NFKMLXLoadPlan, converted: Bool) {
        downloadBytes = measurement.downloadBytes
        parameterCount = measurement.parameterCount
        let directories = Set(measurement.tensorBytes.keys).union(measurement.opaqueBytes.keys)
        guard measurement.downloadBytes != nil, !directories.isEmpty else { return }
        var stored = [String: Int64]()
        for dtypes in measurement.tensorBytes.values {
            stored.merge(dtypes, uniquingKeysWith: +)
        }
        precision = stored.filter { NFKMLXLoadPlan.precisionNames[$0.key] != nil }
            .max { ($0.value, $1.key) < ($1.value, $0.key) }
            .flatMap { NFKMLXLoadPlan.precisionNames[$0.key] }
        var directoryBytes = [String: Int64]()
        for directory in directories {
            let precision = converted ? nil : plan.precision(ofDirectory: directory)
            let tensors = measurement.tensorBytes[directory] ?? [:]
            let opaque = measurement.opaqueBytes[directory] ?? 0
            if precision == nil || (opaque > 0 && precision != .stored) {
                estimatesStoredBytes = true
            }
            directoryBytes[directory] = NFKMLXLoadPlan.bytes(tensors, at: precision ?? .stored) + opaque
        }
        let resident = directoryBytes.values.reduce(0, +)
        residentBytes = resident
        let stageBytes = plan.stages.map { stage in stage.reduce(Int64(0)) { $0 + (directoryBytes[$1] ?? 0) } }
        minimumMemoryBytes = NFKMLXResidencyBudget.leastWorkingSet(holding: stageBytes.max() ?? resident)
    }
}

extension NFKModelOption {

    /// The option with `maximum` as its greatest value, every other field as it is.
    func bounded(by maximum: Int) -> NFKModelOption {
        var dictionary = dictionaryRepresentation()
        dictionary["maximum"] = maximum
        return (try? NFKModelOption(dictionary: dictionary)) ?? self
    }
}
