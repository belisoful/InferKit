//
//  NFKMLXCatalogOptions.swift
//  InferKitMLX
//
//  The load options each catalog family declares, the memory model the estimate reads them through,
//  the key-value cache a family's geometry gives, the draft models a decoder takes, and the presets a
//  server offers for a model.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

// MARK: - Key-value cache geometry

/// How a family's key-value cache grows with the context, read from its configuration's geometry.
///
/// @discussion Each rule counts keys and values at 16 bits per element, the unit
/// `NFKModelMemoryFacts.keyValueByteCountPerToken` states.
enum NFKMLXKeyValueRule {
    /// Every layer caches every position at the key-value head count:
    /// ``NFKMLXModelSizing/keyValueBytesPerToken(of:precision:)`` at 16 bits. A sliding layer that masks
    /// its window and keeps every key (gpt-oss) counts here.
    case dense
    /// Gemma 3's hybrid cache: a full-attention layer caches every position and a sliding layer its
    /// window, so the sliding layers are bounded.
    case slidingWindow
    /// Gemma 3n and Gemma 4: the layers before the trailing `num_kv_shared_layers` cache every position,
    /// a full-attention layer at `num_global_key_value_heads` and `global_head_dim` where the
    /// configuration states them, and the shared layers cache nothing.
    case sharedLayers
    /// A hybrid of attention with state-space or linear-attention layers: only the attention layers
    /// cache. The fixed recurrent state is not counted.
    case attentionLayers

    /// The 16-bit cache bytes one position adds across the growing layers, and the bytes of the
    /// layers whose cache stops at a window. nil where the geometry lacks a field the rule reads.
    func keyValueBytes(of geometry: NFKMLXMeasuredGeometry) -> (perToken: Int, bounded: Int)? {
        guard let layers = geometry.layerCount, layers > 0,
              let heads = geometry.keyValueHeads ?? geometry.attentionHeads, heads > 0 else {
            return nil
        }
        var width = geometry.headDimension
        if width == nil, let hidden = geometry.hiddenSize, let queryHeads = geometry.attentionHeads, queryHeads > 0 {
            width = hidden / queryHeads
        }
        guard let width, width > 0 else {
            return nil
        }
        let perLayer = 2 * heads * width * 2
        let types = geometry.layerTypes.count == layers ? geometry.layerTypes : []
        switch self {
        case .dense:
            return (layers * perLayer, 0)
        case .slidingWindow:
            guard let window = geometry.slidingWindow else {
                return nil
            }
            let full = types.isEmpty
                ? NFKMLXGemma3Configuration.layerTypes(count: layers, pattern: geometry.slidingWindowPattern ?? 6)
                    .map { $0 == .full }
                : types.map { $0 == "full_attention" }
            let fullCount = full.filter { $0 }.count
            return (fullCount * perLayer, (layers - fullCount) * perLayer * window)
        case .sharedLayers:
            let caching = layers - min(max(geometry.sharedKeyValueLayers ?? 0, 0), layers)
            let globalLayer = 2 * (geometry.globalKeyValueHeads ?? heads) * (geometry.globalHeadDimension ?? width) * 2
            let perToken = (0 ..< caching).reduce(0) { total, index in
                total + (index < types.count && types[index] == "full_attention" ? globalLayer : perLayer)
            }
            return (perToken, 0)
        case .attentionLayers:
            guard !types.isEmpty else {
                return nil
            }
            let attention = types.filter { $0 == "full_attention" || $0 == "attention" }.count
            return (attention * perLayer, 0)
        }
    }
}

// MARK: - Load profiles

/// The load options a family's registered factory reads, and the memory facts its loader gives them.
///
/// @discussion A profile describes the options; ``declaration(for:stagePrecisions:maximumContextLength:drafts:)``
/// writes them for one entry. The entry's own `residencies` decide the residency option: an entry with
/// more than one takes `automatic` and each of them. ``NFKMLXCatalogEntry/load`` overrides the family's
/// profile for one entry.
struct NFKMLXLoadProfile {
    /// The primary tasks the profile applies to. An entry of the family whose first task is another
    /// takes no profile. Empty for every task.
    var tasks: Set<String> = []
    /// The precision choices, empty for a loader with one precision.
    var precisions: [String] = []
    /// The precision the registered factory loads at without the option.
    var defaultPrecision = NFKModelPrecisionCheckpoint
    /// The backend property the precision option sets by key-value coding, or nil for a factory
    /// argument.
    var precisionProperty: String? = nil
    /// True where the factory packs the decoder's weights to 8 or 4 bits at load.
    var quantizes = false
    /// The default context length of a decoder with a key-value cache, nil for a model without one.
    var contextLength: Int? = nil
    /// How the cache grows, nil where the family's geometry is not modeled.
    var keyValue: NFKMLXKeyValueRule? = nil
    /// True where the factory takes a draft model for speculative decoding.
    var drafts = false
    /// The stages that take a wider precision where the budget holds it, keyed by stage identifier.
    var widenedStages: [String: String] = [:]
    /// The identifier of the single stage of a model that does not stage.
    var stageIdentifier = "model"
    /// True where a suggested preset halves the sampling steps.
    var halvesSteps = false

    /// A profile that declares no options, for an entry its family's profile does not describe.
    static let plain = NFKMLXLoadProfile()

    /// A text decoder: its precision, the runtime weight quantization where `quantizes`, the key-value
    /// cache defaults, and a draft model where `drafts`. A mixture that pages its routed experts takes
    /// the residency option from its entry's residencies.
    static func decoder(precisions: [String] = [NFKModelPrecisionCheckpoint, NFKModelPrecisionFloat32],
                        defaultPrecision: String = NFKModelPrecisionFloat32, quantizes: Bool = false,
                        contextLength: Int? = 4096, keyValue: NFKMLXKeyValueRule? = .dense, drafts: Bool = false,
                        tasks: Set<String> = [NFKModelTaskTextGeneration]) -> NFKMLXLoadProfile {
        NFKMLXLoadProfile(tasks: tasks, precisions: precisions, defaultPrecision: defaultPrecision, quantizes: quantizes,
                          contextLength: contextLength, keyValue: keyValue, drafts: drafts, stageIdentifier: "decoder")
    }

    /// A decoder that streams the repeated layers it cannot hold: Gemma 3, MADLAD-400, TranslateGemma.
    static func streamedDecoder(precisions: [String], defaultPrecision: String, contextLength: Int?,
                                keyValue: NFKMLXKeyValueRule?, tasks: Set<String>) -> NFKMLXLoadProfile {
        decoder(precisions: precisions, defaultPrecision: defaultPrecision, contextLength: contextLength,
                keyValue: keyValue, tasks: tasks)
    }

    /// A decoder whose only load choice is how it holds its routed experts, with the cache defaults where
    /// `keyValue` models its geometry.
    static func pagedMixture(contextLength: Int? = nil, keyValue: NFKMLXKeyValueRule? = nil,
                             tasks: Set<String> = [NFKModelTaskTextGeneration]) -> NFKMLXLoadProfile {
        decoder(precisions: [], defaultPrecision: NFKModelPrecisionCheckpoint, contextLength: contextLength,
                keyValue: keyValue, tasks: tasks)
    }

    /// A generator whose stages load one after another: the residency from its entry, each stage's
    /// precision from its load plan, and the stages that widen.
    static func diffusionPipeline(widenedStages: [String: String] = [:]) -> NFKMLXLoadProfile {
        NFKMLXLoadProfile(widenedStages: widenedStages, halvesSteps: true)
    }

    /// A model of one stage whose precision is a choice.
    static func imageModule(precisions: [String] = [NFKModelPrecisionCheckpoint, NFKModelPrecisionFloat32],
                            defaultPrecision: String = NFKModelPrecisionCheckpoint, property: String? = nil,
                            halvesSteps: Bool = false, tasks: Set<String> = []) -> NFKMLXLoadProfile {
        NFKMLXLoadProfile(tasks: tasks, precisions: precisions, defaultPrecision: defaultPrecision,
                          precisionProperty: property, halvesSteps: halvesSteps)
    }

    /// The profile of an entry: its own, else its family's whose tasks include its primary task.
    static func profile(for entry: NFKMLXCatalogEntry) -> NFKMLXLoadProfile? {
        if let load = entry.load {
            return load
        }
        return familyProfiles[entry.family]?.first { profile in
            profile.tasks.isEmpty || entry.tasks.first.map(profile.tasks.contains) == true
        }
    }

    /// The families whose registered factories read load options. A family missing here declares none.
    static let familyProfiles: [String: [NFKMLXLoadProfile]] = [
        "qwen3": [.decoder(quantizes: true, drafts: true)],
        "mistral": [.decoder(quantizes: true, drafts: true)],
        "qwen3-moe": [.decoder(quantizes: true, drafts: true)],
        "qwen2-moe": [.decoder(quantizes: true, drafts: true)],
        "mixtral": [.decoder(quantizes: true, drafts: true)],
        // The MXFP4 experts load as stored, so the precision has one value.
        "gpt-oss": [.decoder(precisions: [], defaultPrecision: NFKModelPrecisionCheckpoint, quantizes: true, drafts: true)],
        "gemma3": [
            .streamedDecoder(precisions: [NFKModelPrecisionCheckpoint, NFKModelPrecisionFloat32],
                             defaultPrecision: NFKModelPrecisionFloat32, contextLength: nil, keyValue: nil,
                             tasks: [NFKModelTaskTextGeneration]),
            .streamedDecoder(precisions: [NFKModelPrecisionCheckpoint, NFKModelPrecisionFloat32],
                             defaultPrecision: NFKModelPrecisionCheckpoint, contextLength: nil, keyValue: nil,
                             tasks: [NFKModelTaskTranslation]),
        ],
        "madlad": [
            .streamedDecoder(precisions: [NFKModelPrecisionBFloat16, NFKModelPrecisionFloat32],
                             defaultPrecision: NFKModelPrecisionBFloat16, contextLength: nil, keyValue: nil,
                             tasks: [NFKModelTaskTranslation]),
        ],
        // These generation backends read no cache parameter, so they declare no context or cache option.
        "gemma3n": [.decoder(contextLength: nil, keyValue: nil)],
        "gemma4": [.pagedMixture()],
        "granite-4.0-h": [.pagedMixture()],
        "qwen3-vl": [.pagedMixture(tasks: [NFKModelTaskVisionQuestionAnswering])],
        "qwen4-exp": [.pagedMixture()],
        "deepseek-v4": [.pagedMixture()],
        "whisper": [.imageModule()],
        "stable-diffusion": [.imageModule(property: "precision", halvesSteps: true, tasks: [NFKModelTaskTextToImage])],
        "flux": [.diffusionPipeline()],
        // The text encoder and umT5 widen to float32 where the budget holds them.
        "flux2": [.diffusionPipeline(widenedStages: ["text_encoder": NFKModelPrecisionFloat32])],
        "wan": [.diffusionPipeline(widenedStages: ["text_encoder": NFKModelPrecisionFloat32])],
        "stable-diffusion-3": [.diffusionPipeline()],
        "sana": [.diffusionPipeline()],
        "z-image": [.diffusionPipeline()],
        "qwen-image": [.diffusionPipeline()],
        "ltx-video": [.diffusionPipeline()],
        "minimax-music": [.diffusionPipeline()],
    ]
}

/// The load options, memory model, and presets a profile writes for one entry.
struct NFKMLXLoadDeclaration {
    var options: [NFKModelOption] = []
    var memory: NFKModelMemoryModel? = nil
    /// The load-option keys the entry's registered factory reads.
    var consumes: [String] = []
    var presets: [NFKModelPreset] = []
}

/// One draft model a decoder takes: a smaller model of its family with its vocabulary.
struct NFKMLXDraftChoice {
    let identifier: String
    let title: String
}

extension NFKMLXLoadProfile {

    /// The options, memory model, and presets for `entry`.
    ///
    /// - Parameters:
    ///   - stagePrecisions: each stage's precision from the load plan, read by a profile without a
    ///     precision option.
    ///   - maximumContextLength: the least of the variants' `max_position_embeddings`, the context
    ///     option's maximum.
    ///   - drafts: the draft models the entry takes, empty for none.
    func declaration(for entry: NFKMLXCatalogEntry, stagePrecisions: [String: String], maximumContextLength: Int?,
                     drafts: [NFKMLXDraftChoice]) -> NFKMLXLoadDeclaration {
        var declaration = NFKMLXLoadDeclaration()
        let residency = entry.residencies.count > 1
        if residency {
            declaration.options.append(Self.residencyOption(entry.residencies))
        }
        let precision = precisions.count > 1
        if precision {
            declaration.options.append(precisionOption())
        }
        if quantizes {
            declaration.options.append(Self.bitsOption(NFKModelLoadOptionQuantizationBits, title: "Weight quantization",
                                                       help: "Packs the decoder's weights to 8 or 4 bits in groups of 64 as it loads. 8-bit and 4-bit are approximations of the published weights."))
        }
        let context = contextLength != nil
        if let contextLength {
            declaration.options += Self.cacheOptions(contextLength: contextLength, maximum: maximumContextLength)
        }
        let drafting = self.drafts && !drafts.isEmpty
        if drafting {
            declaration.options += Self.draftOptions(drafts)
        }
        declaration.consumes = declaration.options.map(\.key)
        let fixedPrecisions = precision ? [:] : stagePrecisions.filter { $0.value != NFKModelPrecisionCheckpoint }
        if residency || precision || quantizes || context || drafting || !fixedPrecisions.isEmpty {
            declaration.memory = NFKModelMemoryModel(
                precisionOptionKey: precision ? NFKModelLoadOptionPrecision : nil,
                quantizationBitsOptionKey: quantizes ? NFKModelLoadOptionQuantizationBits : nil,
                quantizationGroupSizeOptionKey: nil,
                contextLengthOptionKey: context ? NFKModelLoadOptionContextLength : nil,
                cacheBitsOptionKey: context ? NFKModelLoadOptionCacheBits : nil,
                cacheGroupSizeOptionKey: context ? NFKModelLoadOptionCacheGroupSize : nil,
                draftOptionKey: drafting ? NFKModelLoadOptionDraftModel : nil,
                draftBitsOptionKey: drafting ? NFKModelLoadOptionDraftBits : nil,
                residencyOptionKey: residency ? NFKModelLoadOptionResidency : nil,
                cacheStage: nil, draftStage: nil, stagePrecisions: fixedPrecisions, extensions: nil)
        }
        declaration.presets = presets(for: entry, options: declaration.options)
        return declaration
    }

    // MARK: Options

    private static let loadGroup = "Loading"
    private static let cacheGroup = "Cache"

    private static func loadOption(_ key: String, kind: String, title: String, help: String, defaultValue: Any?,
                                   minimum: NSNumber? = nil, maximum: NSNumber? = nil, step: NSNumber? = nil,
                                   choices: [(value: Any, title: String)] = [], group: String = loadGroup,
                                   property: String? = nil) -> NFKModelOption {
        NFKModelOption(key: key, kind: kind, title: title, help: help, defaultValue: defaultValue, minimum: minimum,
                       maximum: maximum, step: step,
                       choices: choices.map { NFKModelOptionChoice(value: $0.value, title: $0.title) },
                       stage: NFKModelOptionStageLoad, group: group, requirements: nil, propertyKeyPath: property,
                       memoryBytesPerUnit: nil, extensions: nil)
    }

    /// The residency choices: `automatic`, the default, and each residency the entry supports.
    static func residencyOption(_ residencies: [String]) -> NFKModelOption {
        let titles = [NFKModelResidencyResident: "Resident", NFKModelResidencyStaged: "Staged",
                      NFKModelResidencyPaged: "Paged", NFKModelResidencyStreamed: "Streamed"]
        let choices = [(value: NFKModelResidencyAutomatic as Any, title: "Automatic")]
            + residencies.compactMap { residency in titles[residency].map { (value: residency as Any, title: $0) } }
        return loadOption(NFKModelLoadOptionResidency, kind: NFKModelOptionKindChoice, title: "Residency",
                          help: "How the model holds its weights. Automatic holds every stage where the stages fit this Mac, and otherwise pages, streams, or stages them.",
                          defaultValue: NFKModelResidencyAutomatic, choices: choices)
    }

    private func precisionOption() -> NFKModelOption {
        let titles = [NFKModelPrecisionCheckpoint: "As published", NFKModelPrecisionFloat32: "Float32",
                      NFKModelPrecisionFloat16: "Float16", NFKModelPrecisionBFloat16: "BFloat16"]
        return Self.loadOption(NFKModelLoadOptionPrecision, kind: NFKModelOptionKindChoice, title: "Precision",
                               help: "The precision the weights load at. Float32 is the precision the parity records were measured at.",
                               defaultValue: defaultPrecision,
                               choices: precisions.map { (value: $0 as Any, title: titles[$0] ?? $0) },
                               property: precisionProperty)
    }

    /// A bit-width choice of none, 8, and 4, none the default.
    private static func bitsOption(_ key: String, title: String, help: String, group: String = loadGroup) -> NFKModelOption {
        loadOption(key, kind: NFKModelOptionKindChoice, title: title, help: help, defaultValue: 0,
                   choices: [(0, "None"), (8, "8-bit"), (4, "4-bit")], group: group)
    }

    /// The context length and the key-value cache's quantization, which a request may override.
    private static func cacheOptions(contextLength: Int, maximum: Int?) -> [NFKModelOption] {
        let defaultLength = maximum.map { min(contextLength, $0) } ?? contextLength
        return [
            loadOption(NFKModelLoadOptionContextLength, kind: NFKModelOptionKindInteger, title: "Context length",
                       help: "The positions the key-value cache holds by default. A request may set a shorter or longer window up to the maximum.",
                       defaultValue: defaultLength, minimum: 256, maximum: maximum.map(NSNumber.init(value:)), step: 256,
                       group: cacheGroup),
            bitsOption(NFKModelLoadOptionCacheBits, title: "Cache quantization",
                       help: "Stores the key-value cache quantized to 8 or 4 bits by default. A request may override it.",
                       group: cacheGroup),
            loadOption(NFKModelLoadOptionCacheGroupSize, kind: NFKModelOptionKindChoice, title: "Cache group size",
                       help: "The elements that share one scale and bias in a quantized key-value cache. A request may override it.",
                       defaultValue: 64, choices: [(32, "32"), (64, "64"), (128, "128")], group: cacheGroup),
        ]
    }

    /// The draft model, none by default, and its weight quantization.
    private static func draftOptions(_ drafts: [NFKMLXDraftChoice]) -> [NFKModelOption] {
        [
            loadOption(NFKModelLoadOptionDraftModel, kind: NFKModelOptionKindModelReference, title: "Draft model",
                       help: "A smaller model of the same family and vocabulary that proposes tokens for speculative decoding. None decodes without one.",
                       defaultValue: nil, choices: drafts.map { (value: $0.identifier as Any, title: $0.title) }),
            bitsOption(NFKModelLoadOptionDraftBits, title: "Draft quantization",
                       help: "Packs the draft model's weights to 8 or 4 bits as it loads. 8-bit and 4-bit are approximations of the published weights."),
        ]
    }

    // MARK: Presets

    /// The suggested presets: `fast` where a decoder quantizes its weights and its cache or a generator
    /// halves its steps, and `precise` where float32 is a precision choice other than the default.
    private func presets(for entry: NFKMLXCatalogEntry, options: [NFKModelOption]) -> [NFKModelPreset] {
        let keys = Set(options.map(\.key))
        var presets = [NFKModelPreset]()
        if keys.isSuperset(of: [NFKModelLoadOptionQuantizationBits, NFKModelLoadOptionCacheBits]) {
            presets.append(NFKModelPreset(identifier: "fast", title: "Fast",
                                          help: "4-bit weights and an 8-bit key-value cache, the least memory. Both are approximations of the published weights.",
                                          baseModelIdentifier: nil,
                                          loadOptions: [NFKModelLoadOptionQuantizationBits: 4, NFKModelLoadOptionCacheBits: 8],
                                          requestDefaults: nil, suggested: true, extensions: nil))
        }
        if halvesSteps, let steps = entry.parameters.first(where: { $0.key == NFKParameterSteps })?.defaultValue as? Int,
           steps >= 8 {
            presets.append(NFKModelPreset(identifier: "fast", title: "Fast",
                                          help: "Half the sampling steps, \(steps / 2) in place of \(steps): a faster result with less detail.",
                                          baseModelIdentifier: nil, loadOptions: nil,
                                          requestDefaults: [NFKParameterSteps: steps / 2], suggested: true,
                                          extensions: nil))
        }
        if precisions.contains(NFKModelPrecisionFloat32) && defaultPrecision != NFKModelPrecisionFloat32 {
            presets.append(NFKModelPreset(identifier: "precise", title: "Precise",
                                          help: "Loads the weights at float32, the precision the parity records were measured at.",
                                          baseModelIdentifier: nil,
                                          loadOptions: [NFKModelLoadOptionPrecision: NFKModelPrecisionFloat32],
                                          requestDefaults: nil, suggested: true, extensions: nil))
        }
        return presets
    }
}

// MARK: - Draft models

extension NFKMLXCatalogEntry {

    /// The draft models each measured entry takes, by entry identifier: the entries of its family whose
    /// profile takes drafts, with its default variant's vocabulary size and a smaller parameter count.
    /// The rule is the core's `draftDescriptorsForVariant:inCatalog:`, limited to the decoders whose
    /// factory loads a draft.
    static func draftChoices(_ entries: [NFKMLXCatalogEntry]) -> [String: [NFKMLXDraftChoice]] {
        let decoders = entries.filter { NFKMLXLoadProfile.profile(for: $0)?.drafts == true }
        var choices = [String: [NFKMLXDraftChoice]]()
        for entry in decoders {
            guard let drafted = entry.defaultVariant, let vocabulary = drafted.memoryFacts?.vocabularySize,
                  let parameters = drafted.parameterCount else {
                continue
            }
            let drafts = decoders.compactMap { candidate -> (Int64, NFKMLXDraftChoice)? in
                guard candidate.identifier != entry.identifier, candidate.family == entry.family,
                      let variant = candidate.defaultVariant, variant.memoryFacts?.vocabularySize == vocabulary,
                      let candidateParameters = variant.parameterCount,
                      candidateParameters.int64Value < parameters.int64Value else {
                    return nil
                }
                return (candidateParameters.int64Value,
                        NFKMLXDraftChoice(identifier: candidate.identifier, title: candidate.displayName ?? candidate.identifier))
            }
            if !drafts.isEmpty {
                choices[entry.identifier] = drafts.sorted { $0.0 < $1.0 }.map(\.1)
            }
        }
        return choices
    }

    /// The variant a caller gets without choosing one.
    var defaultVariant: NFKModelVariant? {
        variants.first(where: \.isDefault) ?? variants.first
    }
}
