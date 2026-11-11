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
///   quantization of 2 to 8 bits at group size 64, a draft model and its bits, the context length and
///   cache settings as generation defaults, and the adapters held beside its weights;
/// - Gemma 3: residency, precision, a draft model and its bits, and the adapters held beside its decoder;
/// - TranslateGemma: residency, precision, a draft model and its bits, and the adapters held beside its decoder;
/// - Nemotron-H: residency, precision, a draft model and its bits, and the adapters held beside its weights;
/// - Qwen3.5 (and 3.6 and 3.8): residency, precision, a runtime weight quantization of 2 to 8 bits held
///   resident, a draft model and its bits, the context length and cache settings as generation defaults,
///   and the adapters held beside its weights;
/// - Gemma 3n: precision, and the adapters held beside its decoder;
/// - Whisper: precision, and the adapters held beside its weights;
/// - Stable Diffusion (each release, and each released adapter merged into its base): precision, and the
///   adapters held beside its weights;
/// - MADLAD-400: residency, precision, and the adapters held beside its encoder and decoder;
/// - Seed-X and HY-MT: residency, precision, a runtime weight quantization of the decoder, and the adapters
///   held beside its weights;
/// - Gemma 4: residency, precision, a draft model and its bits, and the adapters held beside its decoder;
/// - Granite 4.0-H and Codestral Mamba: residency, precision, and the adapters held beside its weights;
/// - Qwen4-Exp and Open-Jev: residency and precision;
/// - Qwen3-VL: residency, precision, a runtime weight quantization of the decoder, and the adapters held
///   beside it;
/// - pplx-decider: residency, precision, and a runtime weight quantization of the decoder;
/// - Pixtral: residency, precision, and the adapters held beside its decoder;
/// - DeepSeek V4: residency, and each other value whose key names an ``NFKMLXDeepSeekLoadOptions``
///   property, set by key-value coding;
/// - FLUX.1: residency, and the adapters held beside its transformer;
/// - the staged generators (FLUX.1's ControlNet, FLUX.2, SD3 and its ControlNet, SANA, Z-Image,
///   Qwen-Image, Wan, LTX-Video, MiniMax Music 3, Stable Audio Open): residency.
/// - Wan 2.2 A14B: residency, precision, and a runtime weight quantization of both experts' blocks.
/// - Wan 2.2 Animate: residency, precision, and a runtime weight quantization of the transformer's blocks.
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
            .union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision, NFKModelLoadOptionAdapters])
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
        registerDirectoryModels(gemma3, consumes: draftKeys.union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                                                    NFKModelLoadOptionAdapters])) {
            url, _, options in
            let precision = options.precision ?? .checkpoint
            guard let draftURL = options.draftURL else {
                return try NFKMLXGemma3.backend(directoryURL: url, precision: precision, residency: options.residency,
                                                adapters: options.adapters())
            }
            let model = try NFKMLXGemma3.model(directoryURL: url, draftDirectoryURL: draftURL, precision: precision,
                                               residency: options.residency, draftBits: options.draftBits(orDefault: 4))
            return NFKMLXGemma3Backend(model: model, identifier: NFKMLXGemma3.modelName,
                                       contextLength: NFKMLXModelDescription.contextLength(inDirectory: url),
                                       releaseSampling: NFKMLXReleaseSampling(directoryURL: url),
                                       adapterStrengths: try NFKMLXGemma3.holdAdapters(options.adapters(), in: model))
        }
        registerDirectoryModels(registryNames(families: ["gemma3n"]),
                                consumes: [NFKModelLoadOptionPrecision, NFKModelLoadOptionAdapters]) { url, _, options in
            try NFKMLXGemma3n.backend(directoryURL: url, precision: options.precision ?? .checkpoint,
                                      adapters: options.adapters())
        }
        let gemma4 = registryNames(families: ["gemma4"]) { !$0.tasks.contains(NFKModelTaskTranscription) }
        registerDirectoryModels(gemma4, consumes: draftKeys.union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                                                    NFKModelLoadOptionAdapters])) {
            url, _, options in
            guard let draftURL = options.draftURL else {
                return try NFKMLXGemmaLanguage.backend(directoryURL: url, precision: options.precision ?? .float32,
                                                       residency: options.residency, adapters: options.adapters())
            }
            return try NFKMLXGemmaLanguage.backend(directoryURL: url, draftDirectoryURL: draftURL,
                                                   precision: options.precision ?? .float32, residency: options.residency,
                                                   draftBits: options.draftBits(orDefault: 4), adapters: options.adapters())
        }
    }

    /// Codestral Mamba, Qwen3.5, Qwen4-Exp, Nemotron Nano 2, Granite 4.0-H, Qwen3-VL, pplx-decider, and Pixtral.
    private static func registerOtherDecoders() {
        registerDirectoryModels(registryNames(families: ["mamba2"]),
                                consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                           NFKModelLoadOptionAdapters]) { url, _, options in
            try NFKMLXMamba.backend(directoryURL: url, precision: options.precision ?? .checkpoint,
                                    residency: options.residency, budget: NFKMLXResidencyBudget.current(),
                                    adapters: options.adapters())
        }
        registerDirectoryModels(registryNames(families: ["qwen3.5"]),
                                consumes: draftKeys.union(quantizationKeys).union(generationKeys)
                                    .union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                            NFKModelLoadOptionAdapters])) {
            url, name, options in
            try NFKMLXHybridLanguage.backend(directoryURL: url, precision: options.precision ?? .checkpoint,
                                             residency: options.residency, budget: NFKMLXResidencyBudget.current(),
                                             identifier: name, draftDirectoryURL: options.draftURL,
                                             draftBits: options.draftBits(orDefault: 4),
                                             quantizationBits: options.quantizationBits,
                                             quantizationGroupSize: options.quantizationGroupSize,
                                             generation: options.generationOptions(), adapters: options.adapters())
        }
        // An Open-Jev model's URL is the hub cache holding the release and its base.
        for variant in NFKMLXOpenJevVariant.allCases {
            registerDirectoryModels([variant.registryName],
                                    consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision]) { url, _, options in
                try NFKMLXOpenJev.backend(variant: variant, revision: nil, cacheDirectoryURL: url,
                                          precision: options.precision ?? .checkpoint, residency: options.residency)
            }
        }
        registerDirectoryModels(registryNames(families: ["qwen4-exp"]),
                                consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision]) { url, _, options in
            try NFKMLXQwen4Exp.backend(directoryURL: url, precision: options.precision ?? .checkpoint,
                                       residency: options.residency)
        }
        registerDirectoryModels(registryNames(families: ["nemotron-h"]),
                                consumes: draftKeys.union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                                           NFKModelLoadOptionAdapters])) {
            url, name, options in
            try NFKMLXNemotronH.backend(directoryURL: url, precision: options.precision ?? .checkpoint,
                                        residency: options.residency, budget: NFKMLXResidencyBudget.current(),
                                        identifier: name, draftDirectoryURL: options.draftURL,
                                        draftBits: options.draftBits(orDefault: 4), adapters: options.adapters())
        }
        registerDirectoryModels(registryNames(families: ["granite-4.0-h"]),
                                consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                           NFKModelLoadOptionAdapters]) { url, name, options in
            try NFKMLXGraniteHybrid.backend(directoryURL: url, precision: options.precision ?? .checkpoint,
                                            residency: options.residency, identifier: name, adapters: options.adapters())
        }
        let qwen3VL = registryNames(families: ["qwen3-vl"], where: lists(NFKModelTaskVisionQuestionAnswering))
        registerDirectoryModels(qwen3VL, consumes: quantizationKeys.union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                                                         NFKModelLoadOptionAdapters])) { url, _, options in
            // The adapters go in after the load quantization, which would replace an adapted layer with a plain one.
            let model = try NFKMLXQwen3VL.model(directoryURL: url, loadOptions: options)
            try model.holdAdapters(options.adapters())
            return model.inferenceBackend()
        }
        registerDirectoryModels(registryNames(families: ["pplx-decider"]),
                                consumes: quantizationKeys.union([NFKModelLoadOptionResidency,
                                                                  NFKModelLoadOptionPrecision])) { url, _, options in
            try NFKMLXPplxDecider.decider(directoryURL: url, loadOptions: options).makeBackend()
        }
        registerDirectoryModels(registryNames(families: ["pixtral"]),
                                consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                           NFKModelLoadOptionAdapters]) { url, _, options in
            let model = try NFKMLXPixtral.model(directoryURL: url, precision: options.precision ?? .float32,
                                                residency: options.residency)
            try model.holdAdapters(options.adapters())
            return model.inferenceBackend()
        }
    }

    /// TranslateGemma, MADLAD-400, Seed-X, and HY-MT.
    private static func registerTranslation() {
        let translateGemma = registryNames(families: ["gemma3"], where: lists(NFKModelTaskTranslation))
        registerDirectoryModels(translateGemma,
                                consumes: draftKeys.union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                                           NFKModelLoadOptionAdapters])) {
            url, _, options in
            let precision = options.precision ?? .checkpoint
            guard let draftURL = options.draftURL else {
                return try NFKMLXTranslateGemma.backend(directoryURL: url, precision: precision,
                                                        residency: options.residency, adapters: options.adapters())
            }
            let translator = try NFKMLXTranslateGemma.translator(
                directoryURL: url, draftDirectoryURL: draftURL, precision: precision, residency: options.residency,
                draftBits: options.draftBits(orDefault: 4))
            return try NFKMLXTranslationBackend.holding(options.adapters(), in: translator.model.decoder,
                                                        translator: translator, modulePath: NFKMLXGemma3.adapterModulePath)
        }
        registerDirectoryModels(registryNames(families: ["madlad"]),
                                consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                           NFKModelLoadOptionAdapters]) { url, _, options in
            try NFKMLXMADLAD.backend(directoryURL: url, half: options.halfPrecision ?? true, residency: options.residency,
                                     adapters: options.adapters())
        }
        registerDirectoryModels(registryNames(families: ["seed-x"]),
                                consumes: quantizationKeys.union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                                                  NFKModelLoadOptionAdapters])) { url, _, options in
            // The adapters go in after the load quantization, which would replace an adapted layer with a plain one.
            let translator = try NFKMLXSeedX.translator(directoryURL: url, loadOptions: options)
            return try NFKMLXTranslationBackend.holding(options.adapters(), in: translator.net, translator: translator)
        }
        for (name, release) in NFKMLXHYMT.registryNames {
            registerDirectoryModels([name],
                                    consumes: quantizationKeys.union([NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                                                      NFKModelLoadOptionAdapters])) { url, _, options in
                let translator = try NFKMLXHYMT.translator(directoryURL: url, release: release, loadOptions: options)
                return try NFKMLXTranslationBackend.holding(options.adapters(), in: translator.net, translator: translator)
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
                                    consumes: [NFKModelLoadOptionPrecision, NFKModelLoadOptionAdapters]) { url, name, options in
                let cast = try options.cast(of: name, offering: offered, reading: [NFKModelPrecisionFloat32])
                return try NFKMLXTextToImage.backend(configuration: NFKMLXStableDiffusionRelease(model).configuration,
                                                     directoryURL: url, precision: options.precision ?? .checkpoint,
                                                     castingTo: cast, adapters: options.adapters())
            }
        }
        for adapter in NFKMLXCatalogImage.stableDiffusionAdapters {
            registerDirectoryModels([adapter.name],
                                    consumes: [NFKModelLoadOptionPrecision, NFKModelLoadOptionAdapters]) { url, name, options in
                _ = try options.cast(of: name, offering: [:], reading: [NFKModelPrecisionFloat32])
                return try adapter.backend(directoryURL: url, precision: options.precision ?? .checkpoint,
                                           adapters: options.adapters())
            }
        }
        let residency: Set<String> = [NFKModelLoadOptionResidency]
        registerDirectoryModels(registryNames(families: ["flux"], where: lists(NFKModelTaskTextToImage)),
                                consumes: residency.union([NFKModelLoadOptionAdapters])) { url, name, options in
            let schedule: NFKMLXFlowMatchConfiguration = name == "flux.1-dev" ? .flux : .fluxSchnell
            let flux = try NFKMLXFlux.flux(directoryURL: url, schedule: schedule, residency: options.residency,
                                           adapters: options.adapters())
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

    /// Wan, Wan 2.2 A14B, Wan Animate, LTX-Video, and MiniMax Music 3.
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
        registerDirectoryModels(registryNames(families: ["wan-animate"]),
                                consumes: residency.union([NFKModelLoadOptionPrecision]).union(quantizationKeys)) {
            url, _, options in
            let generator = try NFKMLXWanAnimateGenerator.generator(
                directoryURL: url, residency: options.residency, precision: options.precision ?? .checkpoint,
                quantizationBits: options.quantizationBits ?? 0, budget: NFKMLXResidencyBudget.current(),
                groupSize: options.quantizationGroupSize)
            report(generator.holdsStagesResident, to: options)
            return generator.inferenceBackend()
        }
        registerDirectoryModels(registryNames(families: ["wan-a14b"]),
                                consumes: residency.union([NFKModelLoadOptionPrecision]).union(quantizationKeys)) {
            url, _, options in
            let generator = try NFKMLXWanVideoGenerator.generator(
                directoryURL: url, residency: options.residency, transformerType: nil,
                precision: options.precision ?? .checkpoint,
                bits: (options.quantizationBits ?? 0) == 0 ? nil : options.quantizationBits,
                groupSize: options.quantizationGroupSize)
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
            NFKMLXModelRegistry.register(name: name, consumes: [NFKModelLoadOptionPrecision, NFKModelLoadOptionAdapters]) {
                location, options in
                try NFKMLXWhisper.catalogBackend(variant: variant, location: location, weightsDownload: download,
                                                 repository: release?.repository, precision: options.precision ?? .checkpoint,
                                                 identifier: name, adapters: options.adapters())
            }
        }
    }
}
