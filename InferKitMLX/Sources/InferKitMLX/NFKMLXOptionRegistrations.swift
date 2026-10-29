//
//  NFKMLXOptionRegistrations.swift
//  InferKitMLX
//
//  The factories that build each model family with load options. ``NFKMLXReferenceModels/registerAll()``
//  calls ``NFKMLXOptionRegistrations/register()`` last, so these replace the factories the families and
//  the catalog registered under the same names, which read no options.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

/// Registers the model families that read load options, by the registry names the catalog entries give.
///
/// @discussion Each family's factory reads the `NFKModelLoadOption*` keys it lists as consumed, through
/// ``NFKMLXLoadOptions``, and keeps the family's own default for a key the options leave out:
/// - the dense and mixture decoders (``NFKMLXLanguage``): residency, precision, a runtime weight
///   quantization of 2 to 8 bits at group size 64, a draft model and its bits, and the context length and
///   cache settings as generation defaults;
/// - Gemma 3 and TranslateGemma: residency, precision, a draft model and its bits;
/// - Gemma 3n, Qwen3.5, Whisper, and Stable Diffusion (each release, and each released adapter merged into
///   its base): precision;
/// - Gemma 4, Qwen4-Exp, Granite 4.0-H, and MADLAD-400: residency and precision;
/// - Qwen3-VL and pplx-decider: residency, precision, and a runtime weight quantization of the decoder;
/// - DeepSeek V4: residency, and each other value whose key names an ``NFKMLXDeepSeekLoadOptions``
///   property, set by key-value coding;
/// - the staged generators (FLUX.1 and its ControlNet, FLUX.2, SD3 and its ControlNet, SANA, Z-Image,
///   Qwen-Image, Wan, LTX-Video, MiniMax Music 3, Stable Audio Open): residency.
///
/// A generator reports whether it holds its stages, so a backend's `modelInfo` names the residency an
/// automatic load resolved to under `NFKModelInfoResidency`. Introduced in InferKit 0.4.0.
enum NFKMLXOptionRegistrations {

    /// The keys a decoder reads that change only its generation defaults.
    static let generationKeys: Set<String> = [NFKModelLoadOptionContextLength, NFKModelLoadOptionCacheBits,
                                              NFKModelLoadOptionCacheGroupSize]

    /// The keys a runtime weight quantization reads.
    static let quantizationKeys: Set<String> = [NFKModelLoadOptionQuantizationBits,
                                                NFKModelLoadOptionQuantizationGroupSize]

    /// The keys a draft reads.
    static let draftKeys: Set<String> = [NFKModelLoadOptionDraftModel, NFKModelLoadOptionDraftBits]

    /// The keys every family shares, which the DeepSeek factory does not set by key-value coding.
    static let sharedKeys: Set<String> = generationKeys.union(quantizationKeys).union(draftKeys)
        .union([NFKModelLoadOptionVariant, NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                NFKModelLoadOptionComputeUnits, NFKModelLoadOptionIgnoreMemoryBudget])

    /// Registers every family's options factory, and the MLX engine's builder with the model catalog.
    static func register() {
        NFKMLXModelDescriptors.registerBackendBuilder()
        registerLanguageDecoders()
        registerGemma()
        registerOtherDecoders()
        registerTranslation()
        registerDeepSeek()
        registerImageGenerators()
        registerVideoAndMusicGenerators()
        registerWhisper()
    }

    // MARK: - Names

    /// The registry names of the catalog entries in `families` that `include` accepts: each entry's own
    /// name and its variants' names.
    static func registryNames(families: Set<String>,
                              where include: (NFKMLXCatalogEntry) -> Bool = { _ in true }) -> [String] {
        var names = [String]()
        for entry in NFKMLXModelDescriptors.entries where families.contains(entry.family) && include(entry) {
            let variantNames = entry.variants.compactMap(\.registryName)
            for name in [entry.registryName ?? (variantNames.isEmpty ? entry.identifier : nil)].compactMap({ $0 })
                + variantNames where !names.contains(name) {
                names.append(name)
            }
        }
        return names
    }

    /// Registers `factory` under each name, refusing a build without a release directory.
    private static func registerDirectoryModels(_ names: [String], consumes: Set<String>,
                                                factory: @escaping @Sendable (URL, String, NFKMLXLoadOptions) throws
                                                    -> any NFKInferenceBackend) {
        for name in names {
            NFKMLXModelRegistry.register(name: name, consumes: consumes) { url, options in
                guard let url else {
                    throw NFKMLXError.unsupportedConfiguration("\(name) builds from a release directory, not without weights")
                }
                return try factory(url, name, options)
            }
        }
    }

    /// Whether an entry lists `task`.
    private static func lists(_ task: String) -> (NFKMLXCatalogEntry) -> Bool {
        { $0.tasks.contains(task) }
    }

    // MARK: - Language

    /// The dense and mixture decoders ``NFKMLXLanguage`` loads.
    private static func registerLanguageDecoders() {
        let names = registryNames(families: ["qwen3", "qwen3-moe", "qwen2-moe", "mixtral", "gpt-oss", "mistral"],
                                  where: lists(NFKModelTaskTextGeneration))
        let consumes = generationKeys.union(quantizationKeys).union(draftKeys)
            .union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision])
        registerDirectoryModels(names, consumes: consumes) { url, _, options in
            try NFKMLXLanguage.backend(directoryURL: url, loadOptions: options)
        }
        for name in registryNames(families: ["smollm2"], where: lists(NFKModelTaskTextGeneration)) {
            NFKMLXModelRegistry.register(name: name, consumes: generationKeys) { url, options in
                guard let url else {
                    throw NFKMLXError.unsupportedConfiguration("\(name) builds from a GGUF file, not without weights")
                }
                return try NFKMLXLanguage.backend(ggufURL: url, options: options.generationOptions())
            }
        }
    }

    /// Gemma 3, Gemma 3n, and the Gemma 4 text decoders.
    private static func registerGemma() {
        let gemma3 = registryNames(families: ["gemma3"], where: lists(NFKModelTaskTextGeneration))
        registerDirectoryModels(gemma3, consumes: draftKeys.union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision])) {
            url, _, options in
            let precision = options.precision ?? .checkpoint
            guard let draftURL = options.draftURL else {
                return try NFKMLXGemma3.backend(directoryURL: url, precision: precision, residency: options.residency)
            }
            let model = try NFKMLXGemma3.model(directoryURL: url, draftDirectoryURL: draftURL, precision: precision,
                                               residency: options.residency, draftBits: options.draftBits(orDefault: 4))
            return NFKMLXGemma3Backend(model: model, identifier: NFKMLXGemma3.modelName,
                                       contextLength: NFKMLXModelDescription.contextLength(inDirectory: url),
                                       releaseSampling: NFKMLXReleaseSampling(directoryURL: url))
        }
        registerDirectoryModels(registryNames(families: ["gemma3n"]), consumes: [NFKModelLoadOptionPrecision]) {
            url, _, options in
            try NFKMLXGemma3n.backend(directoryURL: url, precision: options.precision ?? .checkpoint)
        }
        let gemma4 = registryNames(families: ["gemma4"]) { !$0.tasks.contains(NFKModelTaskTranscription) }
        registerDirectoryModels(gemma4, consumes: draftKeys.union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision])) {
            url, _, options in
            guard let draftURL = options.draftURL else {
                return try NFKMLXGemmaLanguage.backend(directoryURL: url, precision: options.precision ?? .float32,
                                                       residency: options.residency)
            }
            return try NFKMLXGemmaLanguage.backend(directoryURL: url, draftDirectoryURL: draftURL,
                                                   precision: options.precision ?? .float32, residency: options.residency,
                                                   draftBits: options.draftBits(orDefault: 4))
        }
    }

    /// Qwen3.5, Qwen4-Exp, Nemotron Nano 2, Granite 4.0-H, Qwen3-VL, pplx-decider, and Pixtral.
    private static func registerOtherDecoders() {
        registerDirectoryModels(registryNames(families: ["qwen3.5"]), consumes: [NFKModelLoadOptionPrecision]) {
            url, _, options in
            try NFKMLXHybridLanguage.backend(directoryURL: url, precision: options.precision ?? .checkpoint)
        }
        registerDirectoryModels(registryNames(families: ["qwen4-exp"]),
                                consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision]) { url, _, options in
            try NFKMLXQwen4Exp.backend(directoryURL: url, precision: options.precision ?? .checkpoint,
                                       residency: options.residency)
        }
        registerDirectoryModels(registryNames(families: ["nemotron-h"]),
                                consumes: draftKeys.union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision])) {
            url, name, options in
            try NFKMLXNemotronH.backend(directoryURL: url, precision: options.precision ?? .checkpoint,
                                        residency: options.residency, budget: NFKMLXResidencyBudget.current(),
                                        identifier: name, draftDirectoryURL: options.draftURL,
                                        draftBits: options.draftBits(orDefault: 4))
        }
        registerDirectoryModels(registryNames(families: ["granite-4.0-h"]),
                                consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision]) { url, name, options in
            try NFKMLXGraniteHybrid.backend(directoryURL: url, precision: options.precision ?? .checkpoint,
                                            residency: options.residency, identifier: name)
        }
        let qwen3VL = registryNames(families: ["qwen3-vl"], where: lists(NFKModelTaskVisionQuestionAnswering))
        registerDirectoryModels(qwen3VL, consumes: quantizationKeys.union([NFKModelLoadOptionResidency,
                                                                         NFKModelLoadOptionPrecision])) { url, _, options in
            try NFKMLXQwen3VL.model(directoryURL: url, loadOptions: options).inferenceBackend()
        }
        registerDirectoryModels(registryNames(families: ["pplx-decider"]),
                                consumes: quantizationKeys.union([NFKModelLoadOptionResidency,
                                                                  NFKModelLoadOptionPrecision])) { url, _, options in
            try NFKMLXPplxDecider.decider(directoryURL: url, loadOptions: options).makeBackend()
        }
        registerDirectoryModels(registryNames(families: ["pixtral"]),
                                consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision]) { url, _, options in
            try NFKMLXPixtral.backend(directoryURL: url, precision: options.precision ?? .float32,
                                      residency: options.residency)
        }
    }

    /// TranslateGemma, MADLAD-400, Seed-X, and HY-MT.
    private static func registerTranslation() {
        let translateGemma = registryNames(families: ["gemma3"], where: lists(NFKModelTaskTranslation))
        registerDirectoryModels(translateGemma,
                                consumes: draftKeys.union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision])) {
            url, _, options in
            let precision = options.precision ?? .checkpoint
            guard let draftURL = options.draftURL else {
                return try NFKMLXTranslateGemma.backend(directoryURL: url, precision: precision,
                                                        residency: options.residency)
            }
            return NFKMLXTranslationBackend(translator: try NFKMLXTranslateGemma.translator(
                directoryURL: url, draftDirectoryURL: draftURL, precision: precision, residency: options.residency,
                draftBits: options.draftBits(orDefault: 4)))
        }
        registerDirectoryModels(registryNames(families: ["madlad"]),
                                consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision]) { url, _, options in
            try NFKMLXMADLAD.backend(directoryURL: url, half: options.halfPrecision ?? true, residency: options.residency)
        }
        registerDirectoryModels(registryNames(families: ["seed-x"]),
                                consumes: quantizationKeys.union([NFKModelLoadOptionResidency,
                                                                  NFKModelLoadOptionPrecision])) { url, _, options in
            NFKMLXTranslationBackend(translator: try NFKMLXSeedX.translator(directoryURL: url, loadOptions: options))
        }
        for (name, release) in NFKMLXHYMT.registryNames {
            registerDirectoryModels([name],
                                    consumes: quantizationKeys.union([NFKModelLoadOptionResidency,
                                                                      NFKModelLoadOptionPrecision])) { url, _, options in
                NFKMLXTranslationBackend(translator: try NFKMLXHYMT.translator(directoryURL: url, release: release,
                                                                                loadOptions: options))
            }
        }
    }

    /// The DeepSeek V4 releases: the residency, and every other value whose key is an
    /// ``NFKMLXDeepSeekLoadOptions`` property.
    private static func registerDeepSeek() {
        let properties = deepSeekPropertyKeys
        registerDirectoryModels(registryNames(families: ["deepseek-v4"]),
                                consumes: properties.union([NFKModelLoadOptionResidency])) { url, _, options in
            try NFKMLXDeepSeek.deepSeekBackend(directoryURL: url, options: deepSeekLoadOptions(options))
        }
    }

    /// The writable properties of ``NFKMLXDeepSeekLoadOptions`` a load option may name.
    static let deepSeekPropertyKeys: Set<String> = ["paging", "expertCacheBytes", "speculates",
                                                    "quantizesActivations", "computesInFloat32"]

    /// The DeepSeek options the load options describe.
    static func deepSeekLoadOptions(_ options: NFKMLXLoadOptions) -> NFKMLXDeepSeekLoadOptions {
        let loadOptions = NFKMLXDeepSeekLoadOptions()
        loadOptions.residency = options.residency
        for key in deepSeekPropertyKeys.sorted() {
            if let value = options.values[key] {
                loadOptions.setValue(value, forKey: key)
            }
        }
        return loadOptions
    }

    // MARK: - Generators

    /// Stable Diffusion, FLUX.1 and its ControlNet, FLUX.2, SD3 and its ControlNet, SANA, Z-Image, and
    /// Qwen-Image.
    private static func registerImageGenerators() {
        for model in NFKMLXCatalogImage.stableDiffusionReleases {
            let offered: [String: DType] = NFKMLXCatalogImage.float16StableDiffusionReleases.contains(model)
                ? [NFKModelPrecisionFloat16: .float16] : [:]
            registerDirectoryModels([NFKMLXCatalogImage.stableDiffusionName(model)],
                                    consumes: [NFKModelLoadOptionPrecision]) { url, name, options in
                let cast = try options.cast(of: name, offering: offered, reading: [NFKModelPrecisionFloat32])
                return try NFKMLXTextToImage.backend(configuration: NFKMLXStableDiffusionRelease(model).configuration,
                                                     directoryURL: url, precision: options.precision ?? .checkpoint,
                                                     castingTo: cast)
            }
        }
        for adapter in NFKMLXCatalogImage.stableDiffusionAdapters {
            registerDirectoryModels([adapter.name], consumes: [NFKModelLoadOptionPrecision]) { url, name, options in
                _ = try options.cast(of: name, offering: [:], reading: [NFKModelPrecisionFloat32])
                return try adapter.backend(directoryURL: url, precision: options.precision ?? .checkpoint)
            }
        }
        let residency: Set<String> = [NFKModelLoadOptionResidency]
        registerDirectoryModels(registryNames(families: ["flux"], where: lists(NFKModelTaskTextToImage)),
                                consumes: residency) { url, name, options in
            let schedule: NFKMLXFlowMatchConfiguration = name == "flux.1-dev" ? .flux : .fluxSchnell
            let flux = try NFKMLXFlux.flux(directoryURL: url, schedule: schedule, residency: options.residency)
            report(flux.holdsStagesResident, to: options)
            return flux.inferenceBackend()
        }
        registerDirectoryModels(registryNames(families: ["flux"], where: lists(NFKModelTaskControlNet)),
                                consumes: residency) { url, _, options in
            let generator = try NFKMLXFluxControlNetGenerator.generator(
                directoryURL: url.appendingPathComponent("base"),
                controlNetDirectoryURL: url.appendingPathComponent("controlnet"), residency: options.residency)
            report(generator.holdsStagesResident, to: options)
            return generator.inferenceBackend()
        }
        registerDirectoryModels(registryNames(families: ["flux2"]), consumes: residency) { url, _, options in
            let flux = try NFKMLXFlux2.flux2(directoryURL: url, residency: options.residency)
            report(flux.holdsStagesResident, to: options)
            return flux.inferenceBackend()
        }
        registerDirectoryModels(registryNames(families: ["stable-diffusion-3"], where: lists(NFKModelTaskTextToImage)),
                                consumes: residency) { url, _, options in
            let generator = try NFKMLXSD3Generator.generator(directoryURL: url, residency: options.residency)
            if generator.streamedLayerCount > 0 {
                options.reportResidency(.streamed)
            } else {
                report(generator.holdsStagesResident, to: options)
            }
            return generator.inferenceBackend()
        }
        registerDirectoryModels(registryNames(families: ["stable-diffusion-3"], where: lists(NFKModelTaskControlNet)),
                                consumes: residency) { url, _, options in
            let generator = try NFKMLXSD3ControlNetGenerator.generator(
                directoryURL: url.appendingPathComponent("base"),
                controlNetDirectoryURL: url.appendingPathComponent("controlnet"), residency: options.residency)
            report(generator.holdsStagesResident, to: options)
            return generator.inferenceBackend()
        }
        registerDirectoryModels(registryNames(families: ["sana"]), consumes: residency) { url, _, options in
            let generator = try NFKMLXSANAGenerator.generator(directoryURL: url, residency: options.residency)
            report(generator.holdsStagesResident, to: options)
            return generator.inferenceBackend()
        }
        registerDirectoryModels(registryNames(families: ["z-image"]), consumes: residency) { url, _, options in
            let generator = try NFKMLXZImageGenerator.generator(directoryURL: url, residency: options.residency)
            report(generator.holdsStagesResident, to: options)
            return generator.inferenceBackend()
        }
        registerDirectoryModels(registryNames(families: ["qwen-image"]), consumes: residency) { url, _, options in
            let generator = try NFKMLXQwenImageGenerator.generator(directoryURL: url, residency: options.residency)
            report(generator.holdsStagesResident, to: options)
            return generator.inferenceBackend()
        }
    }

    /// Wan, LTX-Video, and MiniMax Music 3.
    private static func registerVideoAndMusicGenerators() {
        let residency: Set<String> = [NFKModelLoadOptionResidency]
        // The precision option converts Wan 2.1's float32 transformer to bfloat16 as it loads.
        registerDirectoryModels(registryNames(families: ["wan"]), consumes: residency.union([NFKModelLoadOptionPrecision])) {
            url, name, options in
            // Wan 2.2 TI2V 5B has no bfloat16 parity check of its own.
            let generator = try NFKMLXWanVideoGenerator.generator(
                directoryURL: url, residency: options.residency,
                transformerType: try options.bfloat16Cast(of: name, offersBFloat16: name != "wan2.2-ti2v-5b"))
            report(generator.holdsStagesResident, to: options)
            return generator.inferenceBackend()
        }
        registerDirectoryModels(registryNames(families: ["ltx-video"]), consumes: residency) { url, _, options in
            let generator = try NFKMLXLTXVideoGenerator.generator(directoryURL: url, residency: options.residency)
            report(generator.holdsStagesResident, to: options)
            return generator.inferenceBackend()
        }
        // Music 3 plans its residency at the start of each run, against its downloaded stages.
        registerDirectoryModels(registryNames(families: ["minimax-music"]), consumes: residency) { url, _, options in
            // Each run reads its stages: all of them at once where resident was asked for, else as the run's
            // plan stages them, so the load itself holds none.
            options.reportResidency(options.residency == .resident ? .resident : .staged)
            return try NFKMLXMusic3.backend(directoryURL: url, residency: options.residency)
        }
        registerDirectoryModels(registryNames(families: ["stable-audio"]), consumes: residency) { url, _, options in
            let generator = try NFKMLXStableAudioGenerator.generator(directoryURL: url, residency: options.residency)
            report(generator.holdsStagesResident, to: options)
            return generator.inferenceBackend()
        }
    }

    /// Records whether a generator holds its stages as the residency its load resolved to.
    private static func report(_ holdsStagesResident: Bool, to options: NFKMLXLoadOptions) {
        options.reportResidency(holdsStagesResident ? .resident : .staged)
    }

    // MARK: - Speech

    /// The Whisper sizes, by the release each registry name builds.
    static let whisperVariants: [String: NFKMLXWhisperVariant] = [
        "whisper-tiny": .tiny, "whisper-base": .base, "whisper-small": .small, "whisper-medium": .medium,
        "whisper-large": .large, "whisper-large-v1": .large, "whisper-large-v2": .large,
        "whisper-large-v3": .largeV3, "whisper-large-v3-turbo": .largeV3Turbo,
    ]

    /// Whisper at each released size, at the precision the options name, else as stored.
    private static func registerWhisper() {
        for name in registryNames(families: ["whisper"]) {
            guard let variant = whisperVariants[name] else {
                continue
            }
            let release = NFKMLXCatalogAudio.whisperRelease(named: name)
            let download = release.flatMap { URL(string: $0.url) }
            NFKMLXModelRegistry.register(name: name, consumes: [NFKModelLoadOptionPrecision]) { location, options in
                try NFKMLXWhisper.catalogBackend(variant: variant, location: location, weightsDownload: download,
                                                 repository: release?.repository, precision: options.precision ?? .checkpoint,
                                                 identifier: name)
            }
        }
    }
}
