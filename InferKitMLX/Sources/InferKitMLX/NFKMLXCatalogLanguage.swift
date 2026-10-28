//
//  NFKMLXCatalogLanguage.swift
//  InferKitMLX
//
//  The catalog entries of the language models: text generation, vision-language, translation, embedding,
//  reranking, forecasting, and typed decisions.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The catalog entries of the language models, and the registrations their registry names need.
///
/// @discussion A model that loads a release directory registers one name per validated release where its
/// factory reads one fixed geometry or names the size in its registry name; a model registered under a
/// family name lists each validated release as a variant of that name. The generic loaders (the dense and
/// mixture decoder, Gemma 3, Gemma 3n, Gemma 4, DeepSeek) register one name per release the parity
/// records cover with real weights.
enum NFKMLXCatalogLanguage {

    /// Whether the entry is a release ``NFKMLXLanguage`` loads from its directory, the entries whose loader
    /// judges another repository (``NFKMLXModelDescriptors/descriptor(for:matching:)``).
    static func readsReleases(of entry: NFKMLXCatalogEntry) -> Bool {
        languageReleases.contains { $0.name == entry.identifier }
    }

    /// The language models' entries, in the order ``NFKMLXModelDescriptors`` lists them.
    static var entries: [NFKMLXCatalogEntry] {
        languageReleases.map(languageEntry) + [ggufEntry]
            + [hybridEntry, qwen4ExpEntry] + graniteEntries + [nemotronEntry, nemotron3NanoEntry, mambaEntry]
            + gemma3Releases.map(gemma3Entry) + gemma3nReleases.map(gemma3nEntry) + gemma4Releases.map(gemma4Entry)
            + [gemma4MultimodalEntry]
            + deepSeekReleases.map(deepSeekEntry)
            + ([.medium] + SmolVLMSize.added).map(smolVLMEntry) + qwen3VLReleases.map(qwen3VLEntry)
            + [pixtralEntry, phi4MMEntry]
            + [opusMTEntry] + NFKMLXM2M100Variant.allCases.map(m2m100Entry) + NFKMLXNLLBVariant.allCases.map(nllbEntry)
            + madladReleases.map(madladEntry) + [translateGemmaEntry, seedXEntry]
            + [NFKMLXHYMTRelease.hyMT15, .hyMT2].map(hymtEntry)
            + [qwen3EmbeddingEntry, embeddingGemmaEntry, qwen3VLEmbeddingEntry]
            + [modernBERTRerankerEntry, qwen3VLRerankerEntry]
            + NFKMLXLayaVariant.allCases.map(layaEntry) + NFKMLXOpenJevVariant.allCases.map(openJevEntry)
            + [openJevDeBERTaEntry] + NFKMLXPplxDeciderVariant.allCases.map(pplxDeciderEntry)
            + [chronosEntry, timesFMEntry]
    }

    /// Registers the language models that have a factory and no registry name, under the names their entries
    /// give. ``NFKMLXReferenceModels/registerAll()`` calls it.
    static func registerAdditionalModels() {
        for release in languageReleases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) {
                try NFKMLXLanguage.backend(directoryURL: $0, residency: .automatic, options: NFKMLXGenerationOptions())
            }
        }
        NFKMLXModelRegistry.register(name: ggufRelease.name) { url in
            guard let url else {
                throw NFKMLXError.unsupportedConfiguration("\(ggufRelease.name) builds from a GGUF file, not without weights")
            }
            return try NFKMLXLanguage.backend(ggufURL: url, options: NFKMLXGenerationOptions())
        }
        for release in gemma3Releases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) {
                try NFKMLXGemma3.backend(directoryURL: $0, precision: .float32, residency: .automatic)
            }
        }
        NFKMLXGenerationBackend.registerDirectoryModel(NFKMLXNemotronH.nemotron3NanoModelName) {
            try NFKMLXNemotronH.backend(directoryURL: $0, precision: .checkpoint, residency: .automatic,
                                        budget: NFKMLXResidencyBudget.current(), identifier: NFKMLXNemotronH.nemotron3NanoModelName)
        }
        for release in gemma3nReleases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) {
                try NFKMLXGemma3n.backend(directoryURL: $0, precision: .float32)
            }
        }
        for release in gemma4Releases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) {
                try NFKMLXGemmaLanguage.gemmaBackend(directoryURL: $0)
            }
        }
        for release in deepSeekReleases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) {
                try NFKMLXDeepSeek.deepSeekBackend(directoryURL: $0)
            }
        }
        for size in SmolVLMSize.added {
            NFKMLXGenerationBackend.registerDirectoryModel(size.release.name) { try NFKMLXSmolVLM.backend(directoryURL: $0) }
        }
        for release in qwen3VLReleases.dropFirst() {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) { try NFKMLXQwen3VL.backend(directoryURL: $0) }
        }
        // NFKMLXM2M100.register() and NFKMLXNLLB.register() register the 418M, SMaLL-100, and distilled 600M.
        for variant in NFKMLXM2M100Variant.allCases where ![.m418M, .small100].contains(variant) {
            NFKMLXGenerationBackend.registerDirectoryModel(variant.registryName) {
                try NFKMLXM2M100.backend(variant: variant, directoryURL: $0)
            }
        }
        for variant in NFKMLXNLLBVariant.allCases where variant != .distilled600M {
            NFKMLXGenerationBackend.registerDirectoryModel(variant.name) {
                try NFKMLXNLLB.backend(variant: variant, directoryURL: $0)
            }
        }
        for release in madladReleases.dropFirst() {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) {
                try NFKMLXMADLAD.backend(directoryURL: $0, half: true)
            }
        }
        NFKMLXGenerationBackend.registerDirectoryModel(NFKMLXQwen3Embedding.modelName) {
            try NFKMLXQwen3Embedding.backend(directoryURL: $0, dimensions: nil)
        }
        NFKMLXGenerationBackend.registerDirectoryModel(NFKMLXEmbeddingGemma.modelName) {
            try NFKMLXEmbeddingGemma.backend(directoryURL: $0, dimensions: nil)
        }
        NFKMLXGenerationBackend.registerDirectoryModel(NFKMLXQwen3VLEmbedder.modelName) {
            try NFKMLXQwen3VLEmbedder.backend(directoryURL: $0)
        }
        for variant in NFKMLXLayaVariant.allCases {
            NFKMLXGenerationBackend.registerDirectoryModel(variant.registryName) { try NFKMLXLaya.backend(directoryURL: $0) }
        }
        // The registry's URL is the hub cache that holds the release and its base: an Open-Jev model reads
        // two repositories, and the registry passes one URL.
        for variant in NFKMLXOpenJevVariant.allCases {
            NFKMLXGenerationBackend.registerDirectoryModel(variant.registryName) {
                try NFKMLXOpenJev.backend(variant: variant, revision: nil, cacheDirectoryURL: $0)
            }
        }
        NFKMLXGenerationBackend.registerDirectoryModel(openJevDeBERTaName) {
            try NFKMLXOpenJevDeBERTa.backend(directoryURL: $0)
        }
        for variant in NFKMLXPplxDeciderVariant.allCases {
            NFKMLXGenerationBackend.registerDirectoryModel(variant.registryName) { try NFKMLXPplxDecider.backend(directoryURL: $0) }
        }
    }
}

// MARK: - Releases

/// One release a catalog entry describes: the registry name it builds under, the published name, and
/// the Hugging Face repository the original publisher serves it from.
private struct NFKMLXLanguageRelease {
    let name: String
    let displayName: String
    let repository: String
    var revision: String? = nil
    var family = ""
    var routesExperts = false
    var readsImages = false
    /// The `model_type` values of the configurations its loader reads, where a resolver matches other
    /// repositories to the release.
    var modelTypes: [String] = []
    /// The type the release stores its weights in, stated where no measurement reads it from the shards.
    var storedPrecision: String? = nil
    /// The sampling the release's `generation_config.json` sets, as the backend reads it, or nil where it
    /// decodes greedily.
    var sampling: NFKMLXReleaseSampling? = nil
}

extension NFKMLXCatalogLanguage {

    /// The dense and mixture releases ``NFKMLXLanguage`` loads, each measured on its own weights (whole, or
    /// cut to its first layers).
    fileprivate static var languageReleases: [NFKMLXLanguageRelease] {
        let qwen3Sampling = NFKMLXReleaseSampling(temperature: 0.6, topP: 0.95, topK: 20)
        return [
            NFKMLXLanguageRelease(name: "qwen3-0.6b", displayName: "Qwen3 0.6B", repository: "Qwen/Qwen3-0.6B",
                                  family: "qwen3", modelTypes: ["qwen3"], sampling: qwen3Sampling),
            NFKMLXLanguageRelease(name: "qwen3-1.7b", displayName: "Qwen3 1.7B", repository: "Qwen/Qwen3-1.7B",
                                  family: "qwen3", modelTypes: ["qwen3"], sampling: qwen3Sampling),
            NFKMLXLanguageRelease(name: "qwen3-4b", displayName: "Qwen3 4B", repository: "Qwen/Qwen3-4B",
                                  family: "qwen3", modelTypes: ["qwen3"], sampling: qwen3Sampling),
            NFKMLXLanguageRelease(name: "qwen3-14b", displayName: "Qwen3 14B", repository: "Qwen/Qwen3-14B",
                                  family: "qwen3", modelTypes: ["qwen3"], sampling: qwen3Sampling),
            NFKMLXLanguageRelease(name: "qwen3-32b", displayName: "Qwen3 32B", repository: "Qwen/Qwen3-32B",
                                  family: "qwen3", modelTypes: ["qwen3"], sampling: qwen3Sampling),
            NFKMLXLanguageRelease(name: "qwen3-30b-a3b", displayName: "Qwen3 30B-A3B", repository: "Qwen/Qwen3-30B-A3B",
                                  family: "qwen3-moe", routesExperts: true, modelTypes: ["qwen3_moe"],
                                  sampling: qwen3Sampling),
            NFKMLXLanguageRelease(name: "qwen1.5-moe-a2.7b", displayName: "Qwen1.5-MoE-A2.7B",
                                  repository: "Qwen/Qwen1.5-MoE-A2.7B", family: "qwen2-moe", routesExperts: true,
                                  modelTypes: ["qwen2_moe"]),
            NFKMLXLanguageRelease(name: "mixtral-8x7b-instruct", displayName: "Mixtral 8x7B Instruct",
                                  repository: "mistralai/Mixtral-8x7B-Instruct-v0.1", family: "mixtral",
                                  routesExperts: true, modelTypes: ["mixtral"]),
            NFKMLXLanguageRelease(name: "gpt-oss-20b", displayName: "gpt-oss-20b", repository: "openai/gpt-oss-20b",
                                  family: "gpt-oss", routesExperts: true, modelTypes: ["gpt_oss"],
                                  sampling: NFKMLXReleaseSampling(temperature: 1, topP: 1, topK: 50)),
            NFKMLXLanguageRelease(name: "mistral-small-24b-instruct-2501", displayName: "Mistral Small 3 24B Instruct",
                                  repository: "mistralai/Mistral-Small-24B-Instruct-2501", family: "mistral",
                                  modelTypes: ["mistral", "llama"],
                                  sampling: NFKMLXReleaseSampling(temperature: 0.15, topP: 1, topK: 50)),
        ]
    }

    /// The GGUF file the native reader is measured on.
    fileprivate static var ggufRelease: NFKMLXLanguageRelease {
        NFKMLXLanguageRelease(name: "smollm2-135m-instruct-gguf", displayName: "SmolLM2 135M Instruct (Q4_K_M GGUF)",
                              repository: "unsloth/SmolLM2-135M-Instruct-GGUF", family: "smollm2")
    }

    /// The Gemma 3 releases, by size.
    fileprivate static var gemma3Releases: [NFKMLXLanguageRelease] {
        [("270m", "270M", false), ("1b", "1B", false), ("4b", "4B", true), ("12b", "12B", true), ("27b", "27B", true)]
            .map { size, title, vision in
                // The gated releases are not measured. Their shards are bfloat16.
                NFKMLXLanguageRelease(name: "gemma-3-\(size)-it", displayName: "Gemma 3 \(title) IT",
                                      repository: "google/gemma-3-\(size)-it", family: "gemma3", readsImages: vision,
                                      modelTypes: [vision ? "gemma3" : "gemma3_text"],
                                      storedPrecision: NFKModelPrecisionBFloat16)
            }
    }

    /// The Gemma 3n releases, by size.
    fileprivate static var gemma3nReleases: [NFKMLXLanguageRelease] {
        ["E2B", "E4B"].map { size in
            // The gated releases are not measured. Their shards are bfloat16, so a float32 load takes twice
            // the stored bytes.
            NFKMLXLanguageRelease(name: "gemma-3n-\(size.lowercased())-it", displayName: "Gemma 3n \(size) IT",
                                  repository: "google/gemma-3n-\(size)-it", family: "gemma3n", modelTypes: ["gemma3n"],
                                  storedPrecision: NFKModelPrecisionBFloat16)
        }
    }

    /// The Gemma 4 releases ``NFKMLXGemmaLanguage`` runs as text, by size.
    fileprivate static var gemma4Releases: [NFKMLXLanguageRelease] {
        [("E2B", false), ("E4B", false), ("12B", false), ("26B-A4B", true), ("31B", false)].map { size, mixture in
            // The 12B is `gemma4_unified`, whose loader reads that one release.
            NFKMLXLanguageRelease(name: "gemma-4-\(size.lowercased())-it", displayName: "Gemma 4 \(size) IT",
                                  repository: "google/gemma-4-\(size)-it", family: "gemma4", routesExperts: mixture,
                                  modelTypes: size == "12B" ? [] : ["gemma4"])
        }
    }

    /// The DeepSeek V4 releases. V4.1 Flash carries an image tower.
    fileprivate static var deepSeekReleases: [NFKMLXLanguageRelease] {
        [
            NFKMLXLanguageRelease(name: "deepseek-v4.1-flash", displayName: "DeepSeek V4.1 Flash",
                                  repository: "deepseek-ai/DeepSeek-V4.1-Flash", family: "deepseek-v4",
                                  routesExperts: true, readsImages: true),
            NFKMLXLanguageRelease(name: "deepseek-v4-flash-0731", displayName: "DeepSeek V4 Flash (0731)",
                                  repository: "deepseek-ai/DeepSeek-V4-Flash-0731", family: "deepseek-v4",
                                  routesExperts: true,
                                  sampling: NFKMLXReleaseSampling(temperature: 1, topP: 1, topK: 50)),
            NFKMLXLanguageRelease(name: "deepseek-v4-pro-0813", displayName: "DeepSeek V4 Pro (0813)",
                                  repository: "deepseek-ai/DeepSeek-V4-Pro-0813", family: "deepseek-v4",
                                  routesExperts: true,
                                  sampling: NFKMLXReleaseSampling(temperature: 1, topP: 1, topK: 50)),
        ]
    }

    /// The Qwen3-VL releases, the registered `qwen3-vl-2b` first.
    fileprivate static var qwen3VLReleases: [NFKMLXLanguageRelease] {
        [
            NFKMLXLanguageRelease(name: NFKMLXQwen3VL.modelName, displayName: "Qwen3-VL 2B Instruct",
                                  repository: "Qwen/Qwen3-VL-2B-Instruct", family: "qwen3-vl", modelTypes: ["qwen3_vl"]),
            NFKMLXLanguageRelease(name: "qwen3-vl-8b", displayName: "Qwen3-VL 8B Instruct",
                                  repository: "Qwen/Qwen3-VL-8B-Instruct", family: "qwen3-vl", modelTypes: ["qwen3_vl"]),
            NFKMLXLanguageRelease(name: "qwen3-vl-32b", displayName: "Qwen3-VL 32B Instruct",
                                  repository: "Qwen/Qwen3-VL-32B-Instruct", family: "qwen3-vl", modelTypes: ["qwen3_vl"]),
            NFKMLXLanguageRelease(name: "qwen3-vl-30b-a3b", displayName: "Qwen3-VL 30B-A3B Instruct",
                                  repository: "Qwen/Qwen3-VL-30B-A3B-Instruct", family: "qwen3-vl", routesExperts: true,
                                  modelTypes: ["qwen3_vl_moe"]),
        ]
    }

    /// The MADLAD-400 releases, the registered 3B first. Both load at bfloat16.
    fileprivate static var madladReleases: [NFKMLXLanguageRelease] {
        [
            NFKMLXLanguageRelease(name: NFKMLXMADLAD.modelName, displayName: "MADLAD-400 3B-MT",
                                  repository: "google/madlad400-3b-mt"),
            NFKMLXLanguageRelease(name: "madlad400-7b-mt", displayName: "MADLAD-400 7B-MT",
                                  repository: "google/madlad400-7b-mt"),
        ]
    }

    /// The registry name of the DeBERTa decision model.
    fileprivate static let openJevDeBERTaName = "open-jev-deberta-v3-large"
}

/// The SmolVLM2 sizes. `smolvlm2-500m` is the registered one.
private enum SmolVLMSize {
    case small, medium, large

    static var added: [SmolVLMSize] { [.small, .large] }

    var release: NFKMLXLanguageRelease {
        switch self {
        case .small:
            return NFKMLXLanguageRelease(name: "smolvlm2-256m", displayName: "SmolVLM2 256M Video Instruct",
                                         repository: "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", family: "smolvlm")
        case .medium:
            return NFKMLXLanguageRelease(name: NFKMLXSmolVLM.modelName, displayName: "SmolVLM2 500M Video Instruct",
                                         repository: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", family: "smolvlm")
        case .large:
            return NFKMLXLanguageRelease(name: "smolvlm2-2.2b", displayName: "SmolVLM2 2.2B Instruct",
                                         repository: "HuggingFaceTB/SmolVLM2-2.2B-Instruct", family: "smolvlm")
        }
    }
}

extension NFKMLXM2M100Variant {

    /// The registry name the variant builds under.
    fileprivate var registryName: String {
        switch self {
        case .m418M: return NFKMLXM2M100.modelName
        case .m1_2B: return "m2m100-1.2b"
        case .small100: return "small100"
        }
    }
}

extension NFKMLXLayaVariant {

    /// The registry name the variant builds under.
    fileprivate var registryName: String {
        switch self {
        case .root: return NFKMLXLaya.modelName
        case .typedDecisions: return "laya-typed-decisions"
        case .multilingual: return "laya-multilingual"
        }
    }
}

extension NFKMLXOpenJevVariant {

    /// The registry name the variant builds under.
    fileprivate var registryName: String {
        "open-jev-" + (repository.split(separator: "/").last.map { String($0.dropFirst("Open-Jev-".count)) } ?? "")
            .lowercased()
    }
}

// MARK: - Sources and licenses

extension NFKMLXCatalogLanguage {

    /// The files a release download fetches: the required files, the optional ones, and each weights entry,
    /// optional where the download takes the first of several the repository serves.
    fileprivate static func releaseFiles(required: [String], optional: [String] = [],
                                         weights: [String]) -> [NFKModelFile] {
        required.map { NFKModelFile(path: $0, role: role(of: $0)) }
            + optional.map { NFKModelFile(path: $0, byteCount: nil, sha256: nil, role: role(of: $0), optional: true,
                                          extensions: nil) }
            + weights.map { NFKModelFile(path: $0, byteCount: nil, sha256: nil, role: NFKModelFileRoleWeights,
                                         optional: weights.count > 1, extensions: nil) }
    }

    /// The role a release file plays, read from its name.
    private static func role(of path: String) -> String {
        let name = (path as NSString).lastPathComponent
        if name.hasSuffix(".safetensors") || name.hasSuffix(".bin") || name.hasSuffix(".pt") || name.hasSuffix(".gguf")
            || name.hasSuffix(".index.json") {
            return NFKModelFileRoleWeights
        }
        if name.hasPrefix("tokenizer") || name.hasPrefix("vocab") || name.hasPrefix("merges") || name.hasSuffix(".spm")
            || name.hasSuffix(".model") || name == "added_tokens.json" || name == "special_tokens_map.json"
            || name == "chat_template.jinja" {
            return NFKModelFileRoleTokenizer
        }
        if name.hasSuffix(".json") {
            return NFKModelFileRoleConfig
        }
        return NFKModelFileRoleOther
    }

    /// The one variant of a release, registered under its own name.
    fileprivate static func variant(_ release: NFKMLXLanguageRelease, files: [NFKModelFile],
                                    layout: String = NFKModelSourceLayoutDirectory) -> NFKModelVariant {
        .variant(variantIdentifier(release.repository), displayName: release.displayName, isDefault: true,
                 registryName: release.name,
                 source: .hub(release.repository, revision: release.revision, layout: layout, files: files),
                 precision: release.storedPrecision)
    }

    /// A variant's identifier: the repository's own name, lowercased.
    fileprivate static func variantIdentifier(_ repository: String) -> String {
        (repository.split(separator: "/").last.map(String.init) ?? repository).lowercased()
    }

    /// The residencies a release supports through its registered factory: paged where it routes experts.
    fileprivate static func residencies(routesExperts: Bool) -> [String] {
        routesExperts ? [NFKModelResidencyResident, NFKModelResidencyPaged] : [NFKModelResidencyResident]
    }

    /// The residencies of a decoder that streams the layers it cannot hold.
    fileprivate static var streamedResidencies: [String] { [NFKModelResidencyResident, NFKModelResidencyStreamed] }

    /// A repository's own copy of its license.
    fileprivate static func repositoryLicenseURL(_ repository: String) -> String {
        "https://huggingface.co/\(repository)/blob/main/LICENSE"
    }

    /// A repository's model card, where a release states its license without a license file.
    fileprivate static func modelCardURL(_ repository: String) -> String {
        "https://huggingface.co/\(repository)"
    }

    /// The Gemma Terms of Use, which the Gemma 3 family is released under.
    fileprivate static var gemmaTerms: NFKModelLicense {
        .license(nil, name: "Gemma Terms of Use", url: "https://ai.google.dev/gemma/terms", gated: true,
                 note: "Gated: accept the Gemma terms on Hugging Face before the download. Use follows the Gemma Prohibited Use Policy.")
    }

    /// The Apache License 2.0 the Gemma 4 releases state.
    fileprivate static var gemma4License: NFKModelLicense {
        .license("Apache-2.0", name: "Apache License 2.0", url: "https://ai.google.dev/gemma/docs/gemma_4_license")
    }

    /// Creative Commons Attribution-NonCommercial 4.0, the NLLB-200 license.
    fileprivate static var creativeCommonsNonCommercial: NFKModelLicense {
        .license("CC-BY-NC-4.0", name: "Creative Commons Attribution-NonCommercial 4.0",
                 url: "https://creativecommons.org/licenses/by-nc/4.0/", restrictsCommercialUse: true,
                 note: "Non-commercial use only, with attribution.")
    }

    /// The license of a release, by its repository.
    fileprivate static func license(_ repository: String) -> NFKModelLicense {
        switch repository {
        case "Qwen/Qwen1.5-MoE-A2.7B":
            return .license(nil, name: "Tongyi Qianwen License Agreement", url: repositoryLicenseURL(repository),
                            restrictsCommercialUse: true,
                            note: "A commercial product or service with more than 100 million monthly active users requests a license from Alibaba Cloud.")
        case "Qwen/Qwen3.8-Flash-Next":
            return .license(nil, name: "Qwen Community License 1.0", url: repositoryLicenseURL(repository),
                            restrictsCommercialUse: true,
                            note: "A Model-as-a-Service or AI work-assistant business needs a separate license from Qwen for commercial use. A product past 100 million monthly active users or US$20 million monthly revenue displays the model name.")
        case "nvidia/NVIDIA-Nemotron-Nano-9B-v2":
            return .license(nil, name: "NVIDIA Open Model License",
                            url: "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/")
        case NFKMLXNemotronH.nemotron3NanoRepository:
            return .license(nil, name: "NVIDIA Nemotron Open Model License",
                            url: "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/")
        case "facebook/m2m100_418M", "facebook/m2m100_1.2B", "alirezamsh/small100":
            return .mit(url: modelCardURL(repository))
        case NFKMLXSeedXVariant.ppo.repository, NFKMLXSeedXVariant.instruct.repository:
            return .license(nil, name: "OpenMDW License Agreement 1.0", url: repositoryLicenseURL(repository))
        case NFKMLXHYMTVariant.hyMT15_1_8B.repository, NFKMLXHYMTVariant.hyMT15_7B.repository:
            return .license(nil, name: "Tencent HY Community License Agreement",
                            url: "https://huggingface.co/\(repository)/blob/main/License.txt", restrictsCommercialUse: true,
                            note: "The license does not cover the European Union, the United Kingdom, or South Korea. A licensee whose products passed 100 million monthly active users on the release date requests a license from Tencent. Neither the model nor its output may improve an AI model other than Tencent HY and its derivatives. Use follows Tencent's Acceptable Use Policy, and a distribution carries these use restrictions as an enforceable term.")
        default:
            break
        }
        if repository.hasPrefix("deepseek-ai/") || repository.hasPrefix("microsoft/") {
            return .mit(url: repositoryLicenseURL(repository))
        }
        if repository.hasPrefix("google/gemma-4-") {
            return gemma4License
        }
        if repository.hasPrefix("google/gemma-3") || repository.hasPrefix("google/translategemma")
            || repository == "google/embeddinggemma-300m" {
            return gemmaTerms
        }
        if repository.hasPrefix("facebook/nllb-200") {
            return creativeCommonsNonCommercial
        }
        return .apache2()
    }
}

// MARK: - Ports and options

extension NFKMLXCatalogLanguage {

    /// The defaults every generation backend starts a request from.
    private static var generationDefaults: NFKMLXGenerationOptions { NFKMLXGenerationOptions() }

    fileprivate static var promptInput: NFKModelPort {
        .input(NFKInputPrompt, title: "Prompt", help: "The text the model answers. A request carries it or a conversation.")
    }

    fileprivate static var messagesInput: NFKModelPort {
        .input(NFKInputMessages, title: "Conversation",
               help: "A conversation in the OpenAI message shape, rendered through the release's chat template.")
    }

    fileprivate static func imageInput(required: Bool = false, help: String = "An image the question is about.") -> NFKModelPort {
        .input(NFKInputImage, required: required, title: "Image", help: help)
    }

    fileprivate static var textOutput: NFKModelPort {
        .output(NFKOutputText, required: true, title: "Reply", help: "The model's reply.")
    }

    fileprivate static var usageOutput: NFKModelPort {
        .output(NFKOutputUsage, required: true, title: "Usage", help: "The prompt and reply token counts.")
    }

    /// The outputs of a backend that calls tools: the reply, the calls, and the assistant turn that holds them.
    fileprivate static var toolCallingOutputs: [NFKModelPort] {
        [
            textOutput,
            .output(NFKOutputToolCalls, title: "Tool calls", help: "The tools the model called, where a request offers tools."),
            .output(NFKOutputAssistantMessage, title: "Assistant turn",
                    help: "The assistant turn carrying the tool calls, ready to append to the conversation."),
        ]
    }

    fileprivate static var finishReasonOutput: NFKModelPort {
        .output(NFKOutputFinishReason, title: "Finish reason",
                help: "Why the reply stopped: stop at the end of its turn, length at its token limit, or tool_calls.")
    }

    fileprivate static var reasoningOutput: NFKModelPort {
        .output(NFKOutputReasoning, title: "Reasoning", help: "The reasoning the model shows before its answer.")
    }

    fileprivate static func temperatureOption(_ defaultValue: Float = generationDefaults.temperature) -> NFKModelOption {
        .option(NFKParameterTemperature, kind: NFKModelOptionKindNumber, title: "Temperature",
                help: "0 decodes greedily. A higher value divides the logits before sampling.",
                defaultValue: Double(defaultValue), minimum: 0)
    }

    fileprivate static func topPOption(_ defaultValue: Float = generationDefaults.topP) -> NFKModelOption {
        .option(NFKParameterTopP, kind: NFKModelOptionKindNumber, title: "Top P",
                help: "Samples from the smallest set of tokens whose probability passes this. Ignored at temperature 0.",
                defaultValue: Double(defaultValue), minimum: 0, maximum: 1)
    }

    fileprivate static func topKOption(_ defaultValue: Int = generationDefaults.topK) -> NFKModelOption {
        .option(NFKParameterTopK, kind: NFKModelOptionKindInteger, title: "Top K",
                help: "Samples from this many of the most probable tokens. 0 keeps every token. Ignored at temperature 0.",
                defaultValue: defaultValue, minimum: 0, step: 1)
    }

    fileprivate static func maxTokensOption(_ defaultValue: Int? = nil,
                                            help: String = "The most tokens the reply runs to. Unset, the reply runs to the room the context leaves.") -> NFKModelOption {
        .option(NFKParameterMaxTokens, kind: NFKModelOptionKindInteger, title: "Maximum tokens", help: help,
                defaultValue: defaultValue, minimum: 1, step: 1)
    }

    fileprivate static func seedOption() -> NFKModelOption {
        .option(NFKParameterSeed, kind: NFKModelOptionKindInteger, title: "Seed",
                help: "The seed of a sampled run. The same seed repeats the run.")
    }

    fileprivate static func toolsOption() -> NFKModelOption {
        .option(NFKParameterTools, kind: NFKModelOptionKindJSON, title: "Tools",
                help: "The tools the model may call, each a name, a description, and a JSON Schema of its parameters.")
    }

    fileprivate static func toolCallFormatOption() -> NFKModelOption {
        .option(NFKMLXGenerationParameterKey.toolCallFormat, kind: NFKModelOptionKindChoice, title: "Tool-call format",
                help: "The syntax the model writes a tool call in. Without it the format comes from the release's chat template.",
                choices: NFKMLXToolCallFormat.all.map { (value: $0.name as Any, title: $0.name) } + [(value: "none", title: "none")])
    }

    fileprivate static func reasoningEffortOption() -> NFKModelOption {
        .option(NFKParameterReasoningEffort, kind: NFKModelOptionKindChoice, title: "Reasoning effort",
                help: "How much the model reasons before it answers, bound to the release chat template's thinking switch.",
                choices: [(NFKReasoningEffortLight, "Light"), (NFKReasoningEffortModerate, "Moderate"),
                          (NFKReasoningEffortDeep, "Deep")])
    }

    fileprivate static func jsonSchemaOption() -> NFKModelOption {
        .option(NFKParameterJSONSchema, kind: NFKModelOptionKindJSON, title: "JSON Schema",
                help: "A JSON Schema the reply conforms to, enforced while sampling.")
    }

    fileprivate static func chatTemplateOption() -> NFKModelOption {
        .option(NFKMLXGenerationParameterKey.chatTemplate, kind: NFKModelOptionKindText, title: "Chat template",
                help: "A Jinja chat template in place of the release's own, or \"chatml\" for the ChatML template.")
    }

    fileprivate static func prefillChunkOption() -> NFKModelOption {
        .option(NFKMLXGenerationParameterKey.prefillChunkSize, kind: NFKModelOptionKindInteger, title: "Prefill chunk",
                help: "The most prompt tokens one prefill pass runs, which bounds the peak memory of a long prompt.",
                minimum: 1, step: 1)
    }

    fileprivate static func conversationKeyOption() -> NFKModelOption {
        .option(NFKParameterConversationKey, kind: NFKModelOptionKindText, title: "Conversation key",
                help: "A name for the conversation, whose prompt cache a later request of the same name continues from.")
    }

    fileprivate static func draftTokensOption() -> NFKModelOption {
        .option(NFKMLXGenerationParameterKey.draftTokens, kind: NFKModelOptionKindInteger, title: "Draft tokens",
                help: "The tokens a draft model proposes per round. 0 decodes without the draft.",
                defaultValue: generationDefaults.draftTokens, minimum: 0, step: 1)
    }

    /// The options of a backend whose decoder runs prefill-only and calls tools.
    /// Temperature, top-p, and top-k, defaulting to what a release's generation config sets, or greedy decoding.
    fileprivate static func samplingOptions(_ sampling: NFKMLXReleaseSampling?) -> [NFKModelOption] {
        let defaults = generationDefaults
        return [temperatureOption(sampling?.temperature ?? defaults.temperature), topPOption(sampling?.topP ?? defaults.topP),
                topKOption(sampling?.topK ?? defaults.topK)]
    }

    /// The sampling Gemma 3, 3n, and 4's generation configs set: temperature 1 (transformers' default where the
    /// file leaves it out), top-p 0.95, top-k 64.
    fileprivate static var gemmaSampling: NFKMLXReleaseSampling { NFKMLXReleaseSampling(temperature: 1, topP: 0.95, topK: 64) }

    fileprivate static func toolDecoderOptions(sampling: NFKMLXReleaseSampling? = nil) -> [NFKModelOption] {
        samplingOptions(sampling) + [maxTokensOption(), seedOption(), toolsOption(), toolCallFormatOption()]
    }

    /// The options ``NFKMLXLanguageBackend`` reads, sampling by default as `sampling` says.
    fileprivate static func languageOptions(sampling: NFKMLXReleaseSampling?) -> [NFKModelOption] {
        let defaults = generationDefaults
        return [
            temperatureOption(sampling?.temperature ?? defaults.temperature), topPOption(sampling?.topP ?? defaults.topP),
            topKOption(sampling?.topK ?? defaults.topK), maxTokensOption(), seedOption(), jsonSchemaOption(),
            .option(NFKParameterOutputFormat, kind: NFKModelOptionKindChoice, title: "Output format",
                    help: "Constrains the reply to well-formed JSON, a JSON object, or a JSON array.",
                    choices: [("json", "JSON"), ("json-object", "JSON object"), ("json-array", "JSON array")]),
            .option(NFKParameterChoices, kind: NFKModelOptionKindTextList, title: "Choices",
                    help: "The strings the reply must be exactly one of."),
            .option(NFKMLXGenerationParameterKey.contextWindow, kind: NFKModelOptionKindInteger, title: "Context window",
                    help: "The most cached positions kept; the oldest are dropped past it. Without it every position is kept.",
                    minimum: 1, step: 1, group: "Cache"),
            .option(NFKMLXGenerationParameterKey.cacheQuantizationBits, kind: NFKModelOptionKindChoice,
                    title: "Cache bits", help: "Stores the key-value cache quantized to this bit width. Without it the cache stays in full precision.",
                    choices: NFKMLXLoadProfile.bitWidths.map { ($0, "\($0)-bit") }, group: "Cache"),
            .option(NFKMLXGenerationParameterKey.cacheQuantizationGroupSize, kind: NFKModelOptionKindInteger,
                    title: "Cache group size", help: "The quantized cache's group size, which divides the head dimension.",
                    defaultValue: 64, minimum: 1, step: 1, group: "Cache"),
            .option(NFKMLXGenerationParameterKey.cacheQuantizationPerChannelKeys, kind: NFKModelOptionKindBoolean,
                    title: "Per-channel keys", help: "Groups each key channel along the sequence in the quantized cache.",
                    defaultValue: true, group: "Cache"),
            prefillChunkOption(), chatTemplateOption(), draftTokensOption(),
            .option(NFKMLXGenerationParameterKey.reusesPromptCache, kind: NFKModelOptionKindBoolean, title: "Reuse prompt cache",
                    help: "Keeps the key-value cache between requests, so a prompt that extends the last one prefills only its tail.",
                    defaultValue: defaults.reusesPromptCache, group: "Cache"),
            .option(NFKMLXGenerationParameterKey.outputFormat, kind: NFKModelOptionKindChoice, title: "Output format (MLX)",
                    help: "The output shape under the package's own key: well-formed JSON, a JSON object, or a JSON array.",
                    choices: [("json", "JSON"), ("json-object", "JSON object"), ("json-array", "JSON array")]),
            .option(NFKMLXGenerationParameterKey.choices, kind: NFKModelOptionKindTextList, title: "Choices (MLX)",
                    help: "The strings the reply must be one of, under the package's own key."),
            .option(NFKMLXGenerationParameterKey.reasoningFormat, kind: NFKModelOptionKindChoice, title: "Reasoning format",
                    help: "The markers the model wraps its reasoning in. Without it the markers come from the release's chat template.",
                    choices: [("think", "think tags"), ("harmony", "Harmony channels"), ("none", "none")]),
            toolCallFormatOption(), reasoningEffortOption(), conversationKeyOption(), toolsOption(),
        ]
    }

    /// The options ``NFKMLXTranslationBackend`` reads. `beams` is the release's own beam count, where it
    /// states one.
    fileprivate static func translationOptions(beams: Int? = nil) -> [NFKModelOption] {
        [
            .option(NFKParameterSourceLanguage, kind: NFKModelOptionKindText, title: "Source language",
                    help: "The BCP-47 language of the text. Without it the language is detected."),
            .option(NFKParameterTargetLanguage, kind: NFKModelOptionKindText, title: "Target language",
                    help: "The BCP-47 language to translate into. Every request names it."),
            .option(NFKParameterMaxTokens, kind: NFKModelOptionKindInteger, title: "Maximum tokens",
                    help: "The most tokens one translated paragraph runs to.", minimum: 1, step: 1),
            .option(NFKMLXTranslationParameterKey.beamCount, kind: NFKModelOptionKindInteger, title: "Beams",
                    help: "The beams the decoder searches. 1 decodes greedily.", defaultValue: beams, minimum: 1, step: 1),
            .option(NFKMLXTranslationParameterKey.lengthPenalty, kind: NFKModelOptionKindNumber, title: "Length penalty",
                    help: "Divides a finished hypothesis's log-probability by its length to this power.", defaultValue: 1.0),
            .option(NFKMLXTranslationParameterKey.splitsSentences, kind: NFKModelOptionKindBoolean, title: "Split sentences",
                    help: "Translates each sentence on its own. Off translates each paragraph whole.", defaultValue: false),
        ]
    }

    fileprivate static var translationInputs: [NFKModelPort] {
        [.input(NFKInputPrompt, required: true, title: "Text", help: "The text to translate, paragraph by paragraph.")]
    }

    fileprivate static var translationOutputs: [NFKModelPort] {
        [.output(NFKOutputText, required: true, title: "Translation", help: "The translated text.")]
    }

    /// The ports and option of a backend that answers one question about one image.
    fileprivate static var imageQuestionInputs: [NFKModelPort] {
        [imageInput(required: true),
         .input(NFKInputPrompt, required: true, title: "Question", help: "The question about the image.")]
    }

    fileprivate static var imageQuestionOptions: [NFKModelOption] {
        [maxTokensOption(NFKMLXGenerationBackend.defaultAnswerTokens, help: "The most tokens the answer runs to.")]
    }

    fileprivate static var imageQuestionOutputs: [NFKModelPort] {
        [.output(NFKOutputText, required: true, title: "Answer", help: "The answer.")]
    }

    fileprivate static var embeddingInputs: [NFKModelPort] {
        [.input(NFKInputPrompt, title: "Text", help: "The text to embed. A request carries it or a conversation."),
         .input(NFKInputMessages, title: "Conversation", help: "A conversation whose contents are joined and embedded.")]
    }

    fileprivate static var embeddingOutputs: [NFKModelPort] {
        [.output(NFKOutputEmbedding, required: true, title: "Embedding", help: "The normalized embedding.")]
    }

    fileprivate static var rerankOutputs: [NFKModelPort] {
        [.output(NFKOutputStructured, required: true, title: "Ranking",
                 help: "Each document's score in the documents' order under \"\(NFKMLXRerankKey.scores)\", and the indices from most relevant to least under \"\(NFKMLXRerankKey.rankedIndices)\".")]
    }

    fileprivate static var decisionInputs: [NFKModelPort] {
        [
            .input(NFKInputState, title: "State",
                   help: "The state the questions are about: a string, or a JSON object or array. The prompt or the conversation stands in for it."),
            .input(NFKInputQuestions, required: true, title: "Questions", help: "The typed questions, keyed by identifier."),
            .input(NFKInputPrompt, title: "Prompt", help: "The state as text, where the request carries no state."),
            .input(NFKInputMessages, title: "Conversation", help: "The state as a conversation, where the request carries neither."),
        ]
    }

    fileprivate static var decisionOutputs: [NFKModelPort] {
        [
            .output(NFKOutputAnswers, required: true, title: "Answers", help: "One typed answer per question."),
            .output(NFKOutputStructured, required: true, title: "Reply", help: "The answers in the hosted service's reply shape."),
            .output(NFKOutputUsage, required: true, title: "Usage", help: "The input tokens the answers cost."),
        ]
    }

    fileprivate static var forecastInputs: [NFKModelPort] {
        [.input(NFKMLXForecastKey.series, required: true, valueType: NFKModelValueTypeJSON, title: "Series",
                help: "The observed values, oldest first, as an array of numbers.")]
    }

    fileprivate static func forecastOutputs(mean: Bool) -> [NFKModelPort] {
        let fields = [NFKMLXForecastKey.median] + (mean ? [NFKMLXForecastKey.mean] : [])
            + [NFKMLXForecastKey.quantiles, NFKMLXForecastKey.quantileLevels]
        return [.output(NFKOutputStructured, required: true, title: "Forecast",
                        help: "The forecast fields: " + fields.map { "\"\($0)\"" }.joined(separator: ", ") + ".")]
    }

    fileprivate static func horizonOption(_ defaultValue: Int, maximum: Int? = nil) -> NFKModelOption {
        .option(NFKMLXForecastKey.horizon, kind: NFKModelOptionKindInteger, title: "Horizon",
                help: "The steps to forecast past the series.", defaultValue: defaultValue, minimum: 1,
                maximum: maximum.map { NSNumber(value: $0) }, step: 1)
    }
}

// MARK: - Text generation

extension NFKMLXCatalogLanguage {

    fileprivate static func languageEntry(_ release: NFKMLXLanguageRelease) -> NFKMLXCatalogEntry {
        let architecture = release.routesExperts ? "mixture-of-experts decoder" : "dense decoder"
        return NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: release.family,
            summary: "\(release.displayName), a \(architecture) that writes text, calls tools, and follows a JSON Schema.",
            registryName: release.name, tasks: [NFKModelTaskTextGeneration],
            inputs: [promptInput, messagesInput],
            outputs: toolCallingOutputs + [reasoningOutput, usageOutput,
                                           .output(NFKOutputStructured, title: "Parsed reply",
                                                   help: "The reply parsed as JSON, for a request constrained to JSON."),
                                           finishReasonOutput],
            parameters: languageOptions(sampling: release.sampling),
            variants: [variant(release, files: releaseFiles(required: NFKMLXLanguage.requiredFiles,
                                                            optional: NFKMLXLanguage.optionalFiles,
                                                            weights: NFKMLXLanguage.weightFiles))],
            license: license(release.repository),
            residencies: release.routesExperts ? residencies(routesExperts: true) : streamedResidencies,
            modelTypes: release.modelTypes)
    }

    fileprivate static var ggufEntry: NFKMLXCatalogEntry {
        let release = ggufRelease
        let file = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
        var entry = languageEntry(release)
        entry.residencies = [NFKModelResidencyResident]
        entry.summary = "SmolLM2 135M Instruct quantized to Q4_K_M, read from one GGUF file with its tokenizer."
        entry.variants = [variant(release, files: [NFKModelFile(path: file, role: NFKModelFileRoleWeights)],
                                  layout: NFKModelSourceLayoutFile)]
        entry.outputTokenLimit = 8192       // SmolLM2-135M-Instruct's max_position_embeddings
        return entry
    }

    /// A release-directory decoder that runs prefill-only and reads the release's chat template.
    private static func decoderEntry(name: String, displayName: String, family: String, summary: String,
                                     variants: [NFKModelVariant], license: NFKModelLicense, readsReasoning: Bool,
                                     outputs: [NFKModelPort], parameters: [NFKModelOption],
                                     residencies: [String] = [NFKModelResidencyResident]) -> NFKMLXCatalogEntry {
        NFKMLXCatalogEntry(identifier: name, displayName: displayName, family: family, summary: summary,
                           registryName: name, tasks: [NFKModelTaskTextGeneration],
                           inputs: [promptInput, messagesInput], outputs: outputs,
                           parameters: parameters + (readsReasoning ? [reasoningEffortOption()] : []),
                           variants: variants, license: license, residencies: residencies)
    }

    /// A variant of a family registry name, read from `repository`.
    private static func familyVariant(_ name: String, _ repository: String, displayName: String, isDefault: Bool = false,
                                      revision: String? = nil, files: [NFKModelFile]) -> NFKModelVariant {
        .variant(variantIdentifier(repository), displayName: displayName, isDefault: isDefault, registryName: name,
                 source: .hub(repository, revision: revision, files: files))
    }

    fileprivate static var hybridEntry: NFKMLXCatalogEntry {
        let name = NFKMLXHybridLanguage.modelName
        let files = releaseFiles(required: NFKMLXHybridLanguage.requiredFiles, optional: NFKMLXHybridLanguage.optionalFiles,
                                 weights: NFKMLXHybridLanguage.weightFiles)
        return decoderEntry(
            name: name, displayName: "Qwen3.5", family: "qwen3.5",
            summary: "Qwen3.5, Qwen3.6, and Qwen3.8, hybrid linear- and full-attention decoders that write text and call tools.",
            variants: [
                familyVariant(name, NFKMLXOpenJevVariant.twoB.baseRepository, displayName: "Qwen3.5 2B",
                              revision: NFKMLXOpenJevVariant.twoB.baseRevision, files: files),
                familyVariant(name, "Qwen/Qwen3.5-4B", displayName: "Qwen3.5 4B", isDefault: true, files: files),
                familyVariant(name, NFKMLXOpenJevVariant.nineB.baseRepository, displayName: "Qwen3.5 9B",
                              revision: NFKMLXOpenJevVariant.nineB.baseRevision, files: files),
                familyVariant(name, NFKMLXOpenJevVariant.twentySevenB.baseRepository, displayName: "Qwen3.8 27B",
                              revision: NFKMLXOpenJevVariant.twentySevenB.baseRevision, files: files),
            ],
            license: .apache2(), readsReasoning: true,
            outputs: toolCallingOutputs + [reasoningOutput, usageOutput, finishReasonOutput], parameters: toolDecoderOptions())
    }

    fileprivate static var qwen4ExpEntry: NFKMLXCatalogEntry {
        let name = NFKMLXQwen4Exp.modelName
        let repository = "Qwen/Qwen3.8-Flash-Next"
        return decoderEntry(
            name: name, displayName: "Qwen3.8-Flash-Next", family: "qwen4-exp",
            summary: "Qwen3.8-Flash-Next, a sparse mixture-of-experts decoder with hyper-connections that writes text and calls tools.",
            variants: [familyVariant(name, repository, displayName: "Qwen3.8-Flash-Next", isDefault: true,
                                     files: releaseFiles(required: NFKMLXQwen4Exp.requiredFiles,
                                                         optional: NFKMLXQwen4Exp.optionalFiles,
                                                         weights: NFKMLXQwen4Exp.weightFiles))],
            license: license(repository), readsReasoning: true,
            outputs: toolCallingOutputs + [reasoningOutput, usageOutput, finishReasonOutput], parameters: toolDecoderOptions(sampling: NFKMLXReleaseSampling(temperature: 1, topP: 0.95, topK: 20)),
            residencies: residencies(routesExperts: true))
    }

    /// Granite 4.0-H 1B, a dense decoder, and Granite 4.0-H Tiny, whose routed experts page where the
    /// release does not fit.
    fileprivate static var graniteEntries: [NFKMLXCatalogEntry] {
        let files = releaseFiles(required: NFKMLXGraniteHybrid.requiredFiles, optional: NFKMLXGraniteHybrid.optionalFiles,
                                 weights: NFKMLXGraniteHybrid.weightFiles)
        let releases: [(name: String, repository: String, displayName: String, summary: String, routesExperts: Bool)] = [
            (NFKMLXGraniteHybrid.graniteModelName, "ibm-granite/granite-4.0-h-1b", "Granite 4.0-H 1B",
             "IBM Granite 4.0-H 1B, a hybrid Mamba-2 and attention decoder that writes text and calls tools.", false),
            (NFKMLXGraniteHybrid.graniteTinyModelName, "ibm-granite/granite-4.0-h-tiny", "Granite 4.0-H Tiny",
             "IBM Granite 4.0-H Tiny, a hybrid Mamba-2 and attention decoder with routed experts that writes text and calls tools.",
             true),
        ]
        return releases.map { release in
            var entry = decoderEntry(
                name: release.name, displayName: release.displayName, family: "granite-4.0-h", summary: release.summary,
                variants: [familyVariant(release.name, release.repository, displayName: release.displayName,
                                         isDefault: true, files: files)],
                license: .apache2(), readsReasoning: false, outputs: toolCallingOutputs + [usageOutput, finishReasonOutput],
                parameters: toolDecoderOptions(), residencies: residencies(routesExperts: release.routesExperts))
            entry.modelTypes = ["granitemoehybrid"]
            return entry
        }
    }

    fileprivate static var nemotronEntry: NFKMLXCatalogEntry {
        let name = NFKMLXNemotronH.nemotronModelName
        let repository = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
        var entry = decoderEntry(
            name: name, displayName: "Nemotron Nano 2", family: "nemotron-h",
            summary: "NVIDIA Nemotron Nano 2, a hybrid Mamba-2, attention, and MLP decoder that writes text and calls tools.",
            variants: [familyVariant(name, repository, displayName: "Nemotron Nano 9B v2", isDefault: true,
                                     files: releaseFiles(required: NFKMLXNemotronH.requiredFiles,
                                                         optional: NFKMLXNemotronH.optionalFiles,
                                                         weights: NFKMLXNemotronH.weightFiles))],
            license: license(repository), readsReasoning: false, outputs: toolCallingOutputs + [usageOutput, finishReasonOutput],
            parameters: toolDecoderOptions(), residencies: streamedResidencies)
        entry.modelTypes = ["nemotron_h"]
        return entry
    }

    fileprivate static var nemotron3NanoEntry: NFKMLXCatalogEntry {
        let name = NFKMLXNemotronH.nemotron3NanoModelName
        let repository = NFKMLXNemotronH.nemotron3NanoRepository
        var entry = decoderEntry(
            name: name, displayName: "Nemotron 3 Nano 4B", family: "nemotron-h",
            summary: "NVIDIA Nemotron 3 Nano 4B, a hybrid Mamba-2, attention, and MLP decoder that writes text and calls tools.",
            variants: [familyVariant(name, repository, displayName: "Nemotron 3 Nano 4B", isDefault: true,
                                     files: releaseFiles(required: NFKMLXNemotronH.requiredFiles,
                                                         optional: NFKMLXNemotronH.optionalFiles,
                                                         weights: NFKMLXNemotronH.weightFiles))],
            license: license(repository), readsReasoning: false, outputs: toolCallingOutputs + [usageOutput, finishReasonOutput],
            parameters: toolDecoderOptions(), residencies: streamedResidencies)
        entry.modelTypes = ["nemotron_h"]
        return entry
    }

    fileprivate static var mambaEntry: NFKMLXCatalogEntry {
        let name = NFKMLXMamba.mambaModelName
        var entry = decoderEntry(
            name: name, displayName: "Codestral Mamba", family: "mamba2",
            summary: "Mistral's Codestral Mamba 7B, a Mamba-2 state-space decoder that writes code and text.",
            variants: [familyVariant(name, "mistralai/Mamba-Codestral-7B-v0.1", displayName: "Mamba-Codestral 7B v0.1",
                                     isDefault: true,
                                     files: releaseFiles(required: NFKMLXMamba.requiredFiles, optional: NFKMLXMamba.optionalFiles,
                                                         weights: NFKMLXMamba.weightFiles))],
            license: .apache2(), readsReasoning: false, outputs: [textOutput, usageOutput, finishReasonOutput],
            parameters: samplingOptions(nil) + [maxTokensOption(), seedOption()])
        // A state-space decoder has no position table; Mistral states retrieval tested to 256k tokens.
        entry.outputTokenLimit = 262_144
        return entry
    }
}

// MARK: - Gemma

extension NFKMLXCatalogLanguage {

    fileprivate static func gemma3Entry(_ release: NFKMLXLanguageRelease) -> NFKMLXCatalogEntry {
        let towerHelp = release.readsImages
            ? "An image the reply is about."
            : "An image, which this release refuses: it carries no vision tower."
        var entry = NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: release.family,
            summary: "\(release.displayName), a Gemma 3 decoder with a hybrid sliding-window cache"
                + (release.readsImages ? " and a SigLIP vision tower." : "."),
            registryName: release.name,
            tasks: [NFKModelTaskTextGeneration] + (release.readsImages ? [NFKModelTaskVisionQuestionAnswering] : []),
            inputs: [promptInput, messagesInput, imageInput(help: towerHelp)],
            outputs: [textOutput, usageOutput, finishReasonOutput],
            parameters: samplingOptions(gemmaSampling) + [maxTokensOption(), seedOption(), conversationKeyOption(),
                                                          draftTokensOption()],
            variants: [variant(release, files: releaseFiles(required: NFKMLXGemma3.requiredFiles,
                                                            optional: NFKMLXGemma3.optionalFiles,
                                                            weights: NFKMLXGemma3.weightFiles))],
            license: license(release.repository), residencies: streamedResidencies, modelTypes: release.modelTypes)
        // The gated sizes with a vision tower (4B, 12B, 27B) hold 128K positions, which no measurement reads.
        if release.readsImages {
            entry.outputTokenLimit = 131_072
        }
        return entry
    }

    fileprivate static func gemma3nEntry(_ release: NFKMLXLanguageRelease) -> NFKMLXCatalogEntry {
        NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: release.family,
            summary: "\(release.displayName), a Gemma 3n decoder that reads text, an image, and a recording.",
            registryName: release.name,
            tasks: [NFKModelTaskTextGeneration, NFKModelTaskVisionQuestionAnswering, NFKModelTaskTranscription],
            inputs: [promptInput, messagesInput, imageInput(help: "An image the reply is about."),
                     .input(NFKInputAudio, title: "Audio", help: "A recording the reply is about.")],
            outputs: [textOutput, usageOutput, finishReasonOutput],
            parameters: samplingOptions(gemmaSampling) + [maxTokensOption(), seedOption()],
            variants: [variant(release, files: releaseFiles(required: NFKMLXGemma3n.requiredFiles,
                                                            optional: NFKMLXGemma3n.optionalFiles,
                                                            weights: NFKMLXGemma3n.weightFiles))],
            license: license(release.repository), modelTypes: release.modelTypes)
    }

    fileprivate static func gemma4Entry(_ release: NFKMLXLanguageRelease) -> NFKMLXCatalogEntry {
        let kind = release.routesExperts ? "mixture-of-experts" : "dense"
        var entry = NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: release.family,
            summary: "\(release.displayName), a \(kind) Gemma 4 decoder that writes text and calls tools.",
            registryName: release.name, tasks: [NFKModelTaskTextGeneration],
            inputs: [promptInput, messagesInput], outputs: toolCallingOutputs + [usageOutput, finishReasonOutput],
            parameters: toolDecoderOptions(sampling: gemmaSampling),
            variants: [variant(release, files: releaseFiles(required: NFKMLXGemmaLanguage.requiredFiles,
                                                            optional: NFKMLXGemmaLanguage.optionalFiles,
                                                            weights: NFKMLXGemmaLanguage.weightFiles))],
            license: license(release.repository),
            residencies: release.routesExperts ? residencies(routesExperts: true) : streamedResidencies,
            modelTypes: release.modelTypes)
        if release.routesExperts {
            entry.load = .pagedMixture()
        }
        return entry
    }

    fileprivate static var gemma4MultimodalEntry: NFKMLXCatalogEntry {
        let name = NFKMLXGemmaLanguage.gemmaMultimodalModelName
        let files = releaseFiles(required: NFKMLXGemmaLanguage.requiredFiles, optional: NFKMLXGemmaLanguage.optionalFiles,
                                 weights: NFKMLXGemmaLanguage.weightFiles)
        return NFKMLXCatalogEntry(
            identifier: name, displayName: "Gemma 4 (image and audio)", family: "gemma4",
            summary: "A Gemma 4 decoder with the vision and audio towers its release carries, answering about an image or a recording.",
            registryName: name,
            tasks: [NFKModelTaskTextGeneration, NFKModelTaskVisionQuestionAnswering, NFKModelTaskTranscription],
            inputs: [promptInput, messagesInput,
                     imageInput(help: "An image the reply is about, for a release with a vision tower."),
                     .input(NFKInputAudio, title: "Audio", help: "A recording the reply is about, for a release with an audio tower.")],
            outputs: [textOutput, usageOutput, finishReasonOutput],
            parameters: samplingOptions(gemmaSampling) + [maxTokensOption(), seedOption()],
            variants: [familyVariant(name, "google/gemma-4-E2B-it", displayName: "Gemma 4 E2B IT", isDefault: true,
                                     files: files),
                       familyVariant(name, "google/gemma-4-E4B-it", displayName: "Gemma 4 E4B IT", files: files)],
            license: gemma4License, load: .plain)
    }

    fileprivate static var translateGemmaEntry: NFKMLXCatalogEntry {
        let name = NFKMLXTranslateGemma.modelName
        let files = releaseFiles(required: NFKMLXTranslateGemma.requiredFiles,
                                 optional: NFKMLXTranslateGemma.optionalFiles, weights: NFKMLXTranslateGemma.weightFiles)
        return NFKMLXCatalogEntry(
            identifier: name, displayName: "TranslateGemma", family: "gemma3",
            summary: "TranslateGemma, Gemma 3 fine-tuned to translate between the languages its chat template lists.",
            registryName: name, tasks: [NFKModelTaskTranslation],
            inputs: translationInputs, outputs: translationOutputs, parameters: translationOptions(),
            variants: ["4b", "12b", "27b"].map { size in
                familyVariant(name, "google/translategemma-\(size)-it", displayName: "TranslateGemma \(size.uppercased())",
                              isDefault: size == "4b", files: files)
            },
            license: gemmaTerms, residencies: streamedResidencies)
    }

    fileprivate static var seedXEntry: NFKMLXCatalogEntry {
        let name = NFKMLXSeedX.modelName
        let files = releaseFiles(required: NFKMLXSeedX.requiredFiles, optional: NFKMLXSeedX.optionalFiles,
                                 weights: NFKMLXSeedX.weightFiles)
        var entry = NFKMLXCatalogEntry(
            identifier: name, displayName: "Seed-X", family: "seed-x",
            summary: "Seed-X, a Mistral 7B decoder trained to translate among 28 languages.",
            registryName: name, tasks: [NFKModelTaskTranslation],
            inputs: translationInputs, outputs: translationOutputs, parameters: translationOptions(beams: 1),
            variants: [
                familyVariant(name, NFKMLXSeedXVariant.ppo.repository, displayName: "Seed-X PPO 7B", isDefault: true,
                              files: files),
                familyVariant(name, NFKMLXSeedXVariant.instruct.repository, displayName: "Seed-X Instruct 7B",
                              files: files),
            ],
            license: license(NFKMLXSeedXVariant.ppo.repository), residencies: streamedResidencies)
        entry.outputTokenLimit = 32_768     // the Mistral configuration's max_position_embeddings
        return entry
    }

    /// One HY-MT generation's translator, its 1.8B release the default and its 7B the other variant.
    fileprivate static func hymtEntry(_ release: NFKMLXHYMTRelease) -> NFKMLXCatalogEntry {
        let name = release.modelName
        let files = releaseFiles(required: NFKMLXHYMT.requiredFiles, optional: NFKMLXHYMT.optionalFiles,
                                 weights: NFKMLXHYMT.weightFiles)
        let (small, large): (NFKMLXHYMTVariant, NFKMLXHYMTVariant) = release == .hyMT15
            ? (.hyMT15_1_8B, .hyMT15_7B) : (.hyMT2_1_8B, .hyMT2_7B)
        let generation = release == .hyMT15 ? "HY-MT1.5" : "Hy-MT2"
        var entry = NFKMLXCatalogEntry(
            identifier: name, displayName: generation, family: "hy-mt",
            summary: "\(generation), dense Hunyuan decoders trained to translate among 38 languages.",
            registryName: name, tasks: [NFKModelTaskTranslation],
            inputs: translationInputs, outputs: translationOutputs, parameters: translationOptions(beams: 1),
            variants: [
                familyVariant(name, small.repository, displayName: "\(generation) 1.8B", isDefault: true, files: files),
                familyVariant(name, large.repository, displayName: "\(generation) 7B", files: files),
            ],
            license: license(small.repository), residencies: streamedResidencies,
            modelTypes: [NFKMLXHunyuanDense.modelType])
        entry.outputTokenLimit = 262_144    // the Hunyuan configuration's max_position_embeddings
        return entry
    }

    fileprivate static var embeddingGemmaEntry: NFKMLXCatalogEntry {
        let name = NFKMLXEmbeddingGemma.modelName
        let release = NFKMLXLanguageRelease(name: name, displayName: "EmbeddingGemma 300M",
                                            repository: "google/embeddinggemma-300m")
        return NFKMLXCatalogEntry(
            identifier: name, displayName: release.displayName, family: "gemma3",
            summary: "EmbeddingGemma 300M, a bidirectional Gemma 3 encoder that embeds text for retrieval.",
            registryName: name, tasks: [NFKModelTaskEmbedding],
            inputs: embeddingInputs, outputs: embeddingOutputs,
            parameters: [.option(NFKParameterEmbeddingDimensions, kind: NFKModelOptionKindChoice, title: "Dimensions",
                                 help: "The width of the embedding. The smaller widths are the full embedding truncated and normalized again, as the model was trained to be read.",
                                 defaultValue: 768, choices: [768, 512, 256, 128].map { ($0, "\($0)") })],
            variants: [variant(release, files: releaseFiles(required: NFKMLXEmbeddingGemma.requiredFiles,
                                                            optional: NFKMLXEmbeddingGemma.optionalFiles,
                                                            weights: NFKMLXEmbeddingGemma.weightFiles))],
            license: license(release.repository))
    }
}

// MARK: - DeepSeek

extension NFKMLXCatalogLanguage {

    fileprivate static func deepSeekEntry(_ release: NFKMLXLanguageRelease) -> NFKMLXCatalogEntry {
        NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: release.family,
            summary: "\(release.displayName), a mixture-of-experts decoder computed in bfloat16 as the release computes"
                + (release.readsImages ? ", with an image tower." : "."),
            registryName: release.name,
            tasks: [NFKModelTaskTextGeneration] + (release.readsImages ? [NFKModelTaskVisionQuestionAnswering] : []),
            inputs: [promptInput, messagesInput] + (release.readsImages ? [imageInput()] : []),
            outputs: toolCallingOutputs + [usageOutput, finishReasonOutput],
            parameters: samplingOptions(release.sampling) + [maxTokensOption(), seedOption(), jsonSchemaOption(),
                         chatTemplateOption(), prefillChunkOption(), conversationKeyOption(), toolsOption(),
                         toolCallFormatOption()],
            variants: [variant(release, files: releaseFiles(required: NFKMLXDeepSeek.deepSeekRequiredFiles,
                                                            optional: NFKMLXDeepSeek.deepSeekOptionalFiles,
                                                            weights: NFKMLXDeepSeek.deepSeekWeightFiles))],
            license: license(release.repository), residencies: residencies(routesExperts: true))
    }
}

// MARK: - Vision-language

extension NFKMLXCatalogLanguage {

    /// A model that answers a question about an image through ``NFKMLXGenerationBackend/imageQuestion(identifier:answer:)``.
    private static func imageQuestionEntry(_ release: NFKMLXLanguageRelease, summary: String, files: [NFKModelFile],
                                           residencies: [String] = [NFKModelResidencyResident]) -> NFKMLXCatalogEntry {
        NFKMLXCatalogEntry(identifier: release.name, displayName: release.displayName, family: release.family,
                           summary: summary, registryName: release.name, tasks: [NFKModelTaskVisionQuestionAnswering],
                           inputs: imageQuestionInputs, outputs: imageQuestionOutputs, parameters: imageQuestionOptions,
                           variants: [variant(release, files: files)], license: license(release.repository),
                           residencies: residencies)
    }

    fileprivate static func smolVLMEntry(_ size: SmolVLMSize) -> NFKMLXCatalogEntry {
        var entry = imageQuestionEntry(
            size.release,
            summary: "\(size.release.displayName), a SigLIP tower and a SmolLM2 decoder that answer a question about an image.",
            files: releaseFiles(required: NFKMLXSmolVLM.requiredFiles, optional: NFKMLXSmolVLM.optionalFiles,
                                weights: NFKMLXSmolVLM.weightFiles))
        entry.modelTypes = ["smolvlm"]
        return entry
    }

    fileprivate static func qwen3VLEntry(_ release: NFKMLXLanguageRelease) -> NFKMLXCatalogEntry {
        var entry = imageQuestionEntry(
            release, summary: "\(release.displayName), the Qwen3-VL vision tower and decoder answering a question about an image.",
            files: releaseFiles(required: NFKMLXQwen3VL.requiredFiles, optional: NFKMLXQwen3VL.optionalFiles,
                                weights: NFKMLXQwen3VL.weightFiles),
            residencies: release.routesExperts ? residencies(routesExperts: true) : streamedResidencies)
        // The mixture pages its routed experts, so its residency is its only load choice.
        if release.routesExperts {
            entry.load = .pagedMixture(tasks: [NFKModelTaskVisionQuestionAnswering])
        }
        entry.modelTypes = release.modelTypes
        return entry
    }

    fileprivate static var pixtralEntry: NFKMLXCatalogEntry {
        imageQuestionEntry(NFKMLXLanguageRelease(name: NFKMLXPixtral.modelName, displayName: "Pixtral 12B",
                                                 repository: "mistral-experimental/pixtral-12b", family: "pixtral"),
                           summary: "Pixtral 12B, a 2D-rotary vision encoder and a Mistral NeMo decoder answering a question about an image.",
                           files: releaseFiles(required: NFKMLXPixtral.requiredFiles, optional: NFKMLXPixtral.optionalFiles,
                                               weights: NFKMLXPixtral.weightFiles),
                           residencies: streamedResidencies)
    }

    fileprivate static var phi4MMEntry: NFKMLXCatalogEntry {
        let name = NFKMLXPhi4MM.modelName
        let release = NFKMLXLanguageRelease(name: name, displayName: "Phi-4-multimodal",
                                            repository: NFKMLXPhi4MM.releaseRepo, family: "phi-4-multimodal")
        return NFKMLXCatalogEntry(
            identifier: name, displayName: release.displayName, family: release.family,
            summary: "Phi-4-multimodal, a Phi-4-mini decoder with image and speech towers and a LoRA per modality.",
            registryName: name,
            tasks: [NFKModelTaskTextGeneration, NFKModelTaskVisionQuestionAnswering, NFKModelTaskTranscription],
            inputs: [promptInput, messagesInput,
                     imageInput(help: "An image the reply is about."),
                     .input(NFKInputImages, title: "Images", help: "Several images the reply is about."),
                     .input(NFKInputAudio, title: "Audio", help: "A recording the reply is about."),
                     .input(NFKInputAudios, title: "Recordings", help: "Several recordings the reply is about.")],
            outputs: [textOutput, usageOutput],
            parameters: [temperatureOption(), topPOption(), maxTokensOption(), seedOption()],
            variants: [variant(release, files: releaseFiles(required: NFKMLXPhi4MM.requiredFiles,
                                                            optional: NFKMLXPhi4MM.optionalFiles,
                                                            weights: NFKMLXPhi4MM.weightFiles))],
            license: license(release.repository))
    }
}

// MARK: - Translation

extension NFKMLXCatalogLanguage {

    /// The OPUS-MT language pairs the parity and language probes cover.
    private static let opusMTPairs = ["en-de", "ja-en", "zh-en", "en-zh", "ar-en", "vi-en", "th-en", "hi-en", "ko-en"]

    /// The pairs whose model cards state CC BY 4.0. The other listed pairs state Apache 2.0.
    private static let opusMTCreativeCommonsPairs: Set<String> = ["en-de", "zh-en"]

    fileprivate static var opusMTEntry: NFKMLXCatalogEntry {
        let name = NFKMLXMarian.modelName
        let files = releaseFiles(required: NFKMLXMarian.requiredFiles, optional: NFKMLXMarian.optionalFiles,
                                 weights: NFKMLXMarian.weightFiles)
        return NFKMLXCatalogEntry(
            identifier: name, displayName: "OPUS-MT", family: "marian",
            summary: "Helsinki-NLP's OPUS-MT, a Marian translator per language pair.",
            registryName: name, tasks: [NFKModelTaskTranslation],
            inputs: translationInputs, outputs: translationOutputs, parameters: translationOptions(beams: 4),
            variants: opusMTPairs.map { pair in
                let repository = "Helsinki-NLP/opus-mt-\(pair)"
                return .variant(variantIdentifier(repository), displayName: "OPUS-MT \(pair)", isDefault: pair == "en-de",
                                registryName: name, source: .hub(repository, files: files),
                                license: opusMTCreativeCommonsPairs.contains(pair)
                                    ? .license("CC-BY-4.0", name: "Creative Commons Attribution 4.0 International",
                                               url: modelCardURL(repository), note: "Attribution required.")
                                    : nil)
            },
            license: .license("Apache-2.0", name: "Apache License 2.0", url: "https://huggingface.co/Helsinki-NLP",
                              note: "The en-de and zh-en pairs are CC BY 4.0, which each of those variants states."),
            modelTypes: ["marian"])
    }

    fileprivate static func m2m100Entry(_ variant: NFKMLXM2M100Variant) -> NFKMLXCatalogEntry {
        let name = variant.registryName
        let displayName = ["m2m100": "M2M-100 418M", "m2m100-1.2b": "M2M-100 1.2B", "small100": "SMaLL-100"][name] ?? name
        let release = NFKMLXLanguageRelease(name: name, displayName: displayName, repository: variant.repo)
        return NFKMLXCatalogEntry(
            identifier: name, displayName: displayName, family: "m2m100",
            summary: "\(displayName), Meta's many-to-many translator between 100 languages.",
            registryName: name, tasks: [NFKModelTaskTranslation],
            inputs: translationInputs, outputs: translationOutputs, parameters: translationOptions(beams: 5),
            variants: [self.variant(release, files: releaseFiles(required: NFKMLXM2M100.requiredFiles,
                                                                 optional: NFKMLXM2M100.optionalFiles,
                                                                 weights: NFKMLXM2M100.weightFiles))],
            license: license(release.repository), modelTypes: ["m2m_100"])
    }

    fileprivate static func nllbEntry(_ variant: NFKMLXNLLBVariant) -> NFKMLXCatalogEntry {
        let size = (variant.repo.split(separator: "/").last.map(String.init) ?? variant.repo)
            .replacingOccurrences(of: "nllb-200-", with: "")
        let displayName = "NLLB-200 " + size.replacingOccurrences(of: "distilled-", with: "distilled ")
        let release = NFKMLXLanguageRelease(name: variant.name, displayName: displayName, repository: variant.repo)
        return NFKMLXCatalogEntry(
            identifier: variant.name, displayName: displayName, family: "nllb",
            summary: "\(displayName), Meta's No Language Left Behind translator between 202 languages.",
            registryName: variant.name, tasks: [NFKModelTaskTranslation],
            inputs: translationInputs, outputs: translationOutputs, parameters: translationOptions(),
            variants: [self.variant(release, files: releaseFiles(required: NFKMLXNLLB.requiredFiles,
                                                                 optional: NFKMLXNLLB.optionalFiles,
                                                                 weights: NFKMLXNLLB.weightFiles))],
            license: license(release.repository), modelTypes: ["m2m_100"])
    }

    fileprivate static func madladEntry(_ release: NFKMLXLanguageRelease) -> NFKMLXCatalogEntry {
        var entry = NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: "madlad",
            summary: "\(release.displayName), Google's T5 translator between more than 400 languages, loaded at bfloat16.",
            registryName: release.name, tasks: [NFKModelTaskTranslation],
            inputs: translationInputs, outputs: translationOutputs, parameters: translationOptions(beams: 1),
            variants: [variant(release, files: releaseFiles(required: NFKMLXMADLAD.requiredFiles,
                                                            optional: NFKMLXMADLAD.optionalFiles,
                                                            weights: NFKMLXMADLAD.weightFiles))],
            license: license(release.repository), residencies: streamedResidencies)
        entry.outputTokenLimit = 512        // the T5 configuration's n_positions, the length it was trained to
        return entry
    }
}

// MARK: - Embedding and reranking

extension NFKMLXCatalogLanguage {

    /// The embedding width a Matryoshka-trained model takes, from the range its model card states up to its
    /// full width, the default.
    fileprivate static func matryoshkaWidth(minimum: Int, maximum: Int) -> NFKModelOption {
        .option(NFKParameterEmbeddingDimensions, kind: NFKModelOptionKindInteger, title: "Dimensions",
                help: "The width of the embedding. A smaller width is the full embedding truncated and normalized again, as the model was trained to be read.",
                defaultValue: maximum, minimum: NSNumber(value: minimum), maximum: NSNumber(value: maximum), step: 1)
    }

    fileprivate static var qwen3EmbeddingEntry: NFKMLXCatalogEntry {
        let release = NFKMLXLanguageRelease(name: NFKMLXQwen3Embedding.modelName, displayName: "Qwen3-Embedding 0.6B",
                                            repository: "Qwen/Qwen3-Embedding-0.6B")
        return NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: "qwen3",
            summary: "Qwen3-Embedding 0.6B, the Qwen3 decoder pooled at its last token into a text embedding for retrieval.",
            registryName: release.name, tasks: [NFKModelTaskEmbedding],
            inputs: embeddingInputs, outputs: embeddingOutputs,
            parameters: [matryoshkaWidth(minimum: 32, maximum: 1024)],
            variants: [variant(release, files: releaseFiles(required: NFKMLXQwen3Embedding.requiredFiles,
                                                            optional: NFKMLXQwen3Embedding.optionalFiles,
                                                            weights: NFKMLXQwen3Embedding.weightFiles))],
            license: license(release.repository))
    }

    fileprivate static var qwen3VLEmbeddingEntry: NFKMLXCatalogEntry {
        let release = NFKMLXLanguageRelease(name: NFKMLXQwen3VLEmbedder.modelName, displayName: "Qwen3-VL-Embedding 2B",
                                            repository: "Qwen/Qwen3-VL-Embedding-2B")
        return NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: "qwen3-vl",
            summary: "Qwen3-VL-Embedding 2B, which embeds text and images in one space. The backend embeds text.",
            registryName: release.name, tasks: [NFKModelTaskEmbedding],
            inputs: embeddingInputs, outputs: embeddingOutputs,
            parameters: [matryoshkaWidth(minimum: 64, maximum: 2048)],
            variants: [variant(release, files: releaseFiles(required: NFKMLXQwen3VLEmbedder.requiredFiles,
                                                            optional: NFKMLXQwen3VLEmbedder.optionalFiles,
                                                            weights: NFKMLXQwen3VLEmbedder.weightFiles))],
            license: license(release.repository))
    }

    fileprivate static var modernBERTRerankerEntry: NFKMLXCatalogEntry {
        let release = NFKMLXLanguageRelease(name: NFKMLXModernBERTReranker.modelName,
                                            displayName: "gte-reranker-modernbert-base",
                                            repository: "Alibaba-NLP/gte-reranker-modernbert-base")
        return NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: "modernbert",
            summary: "Alibaba's GTE reranker, a ModernBERT cross-encoder that scores each document against a query.",
            registryName: release.name, tasks: [NFKModelTaskReranking],
            inputs: [.input(NFKInputPrompt, required: true, title: "Query", help: "The query the documents are ranked for."),
                     .input(NFKInputDocuments, required: true, title: "Documents", help: "The documents to rank, as strings.")],
            outputs: rerankOutputs, parameters: [],
            variants: [variant(release, files: releaseFiles(required: NFKMLXModernBERTReranker.requiredFiles,
                                                            optional: NFKMLXModernBERTReranker.optionalFiles,
                                                            weights: NFKMLXModernBERTReranker.weightFiles))],
            license: license(release.repository))
    }

    fileprivate static var qwen3VLRerankerEntry: NFKMLXCatalogEntry {
        let release = NFKMLXLanguageRelease(name: NFKMLXQwen3VLReranker.modelName, displayName: "Qwen3-VL-Reranker 2B",
                                            repository: "Qwen/Qwen3-VL-Reranker-2B")
        return NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: "qwen3-vl",
            summary: "Qwen3-VL-Reranker 2B, which scores text or image documents against a text or image query.",
            registryName: release.name, tasks: [NFKModelTaskReranking],
            inputs: [.input(NFKInputPrompt, title: "Query", help: "The query as text. A request carries it, an image, or both."),
                     imageInput(help: "The query as an image."),
                     .input(NFKInputDocuments, required: true, title: "Documents",
                            help: "The documents to rank, each a string or an image.")],
            outputs: rerankOutputs,
            parameters: [.option(NFKMLXRerankKey.instruction, kind: NFKModelOptionKindText, title: "Instruction",
                                 help: "The task relevance is judged for.",
                                 defaultValue: NFKMLXQwen3VLReranker.defaultInstruction)],
            variants: [variant(release, files: releaseFiles(required: NFKMLXQwen3VLReranker.requiredFiles,
                                                            optional: NFKMLXQwen3VLReranker.optionalFiles,
                                                            weights: NFKMLXQwen3VLReranker.weightFiles))],
            license: license(release.repository))
    }
}

// MARK: - Typed decisions

extension NFKMLXCatalogLanguage {

    fileprivate static func layaEntry(_ variant: NFKMLXLayaVariant) -> NFKMLXCatalogEntry {
        let name = variant.registryName
        let displayName = ["Laya", "Laya typed-decisions", "Laya multilingual"][variant.rawValue]
        let encoder = variant == .multilingual ? "an mmBERT-base encoder over 100-plus languages" : "a ModernBERT-large encoder"
        return NFKMLXCatalogEntry(
            identifier: name, displayName: displayName, family: "laya",
            summary: "\(displayName), \(encoder) with a decision head that answers typed questions about a state in one pass.",
            registryName: name, tasks: [NFKModelTaskDecision, NFKModelTaskClassification],
            inputs: decisionInputs, outputs: decisionOutputs, parameters: [],
            variants: [.variant(variant.folder.isEmpty ? "laya" : variant.folder, displayName: displayName,
                                isDefault: true, registryName: name,
                                source: .hub(NFKMLXLaya.repository, revision: NFKMLXLaya.measuredRevision,
                                             files: releaseFiles(required: NFKMLXLaya.releaseFiles(for: variant),
                                                                 weights: [])))],
            license: license(NFKMLXLaya.repository))
    }

    fileprivate static func openJevEntry(_ variant: NFKMLXOpenJevVariant) -> NFKMLXCatalogEntry {
        let name = variant.registryName
        let displayName = variant.repository.split(separator: "/").last.map(String.init) ?? name
        let base = variant.baseRepository.split(separator: "/").last.map(String.init) ?? variant.baseRepository
        let checkpoint = releaseFiles(required: NFKMLXOpenJev.checkpointFiles.map { "package/checkpoint/\($0)" },
                                      weights: [])
        let baseFiles = NFKMLXOpenJev.baseFiles.map { path in
            NFKModelFile.file(path, role: role(of: path), repository: variant.baseRepository, revision: variant.baseRevision)
        }
        return NFKMLXCatalogEntry(
            identifier: name, displayName: displayName, family: "open-jev",
            summary: "\(displayName), a LoRA adapter and scoring head over \(base) that answers typed questions about a state.",
            registryName: name, tasks: [NFKModelTaskDecision, NFKModelTaskClassification],
            inputs: decisionInputs, outputs: decisionOutputs, parameters: [],
            variants: [.variant(variantIdentifier(variant.repository), displayName: displayName, isDefault: true,
                                registryName: name,
                                source: .hub(variant.repository, revision: variant.measuredRevision,
                                             files: checkpoint + baseFiles))],
            license: .apache2(note: "The adapter reads \(variant.baseRepository) at \(variant.baseRevision), which the download fetches beside it."))
    }

    /// A pplx-decider release: the fine-tuned backbone, the readout, and the protocol files in one
    /// repository, read by its own `config.json` and `decision_config.json`.
    fileprivate static func pplxDeciderEntry(_ variant: NFKMLXPplxDeciderVariant) -> NFKMLXCatalogEntry {
        let name = variant.registryName
        let base = variant.baseRepository.split(separator: "/").last.map(String.init) ?? variant.baseRepository
        let files = releaseFiles(required: NFKMLXPplxDecider.requiredFiles, optional: NFKMLXPplxDecider.optionalFiles,
                                 weights: NFKMLXPplxDecider.weightFiles)
        return NFKMLXCatalogEntry(
            identifier: name, displayName: name, family: "pplx-decider",
            summary: "\(name), a \(base) fine-tune with a readout that answers every typed question about a state in one pass.",
            registryName: name, tasks: [NFKModelTaskDecision, NFKModelTaskClassification],
            inputs: decisionInputs, outputs: decisionOutputs, parameters: [],
            variants: [.variant(variantIdentifier(variant.repository), displayName: name, isDefault: true, registryName: name,
                                source: .hub(variant.repository, revision: variant.measuredRevision, files: files))],
            license: .apache2(note: "Derived from \(variant.baseRepository) under the Apache License 2.0, as the release's NOTICE states."),
            modelTypes: ["qwen3_5"])
    }

    fileprivate static var openJevDeBERTaEntry: NFKMLXCatalogEntry {
        let name = openJevDeBERTaName
        return NFKMLXCatalogEntry(
            identifier: name, displayName: "open-jev-deberta-v3-large", family: "deberta-v3",
            summary: "A DeBERTa-v3-large encoder with a scoring head that answers typed questions about a state.",
            registryName: name, tasks: [NFKModelTaskDecision, NFKModelTaskClassification],
            inputs: decisionInputs, outputs: decisionOutputs, parameters: [],
            variants: [.variant(variantIdentifier(NFKMLXOpenJevDeBERTa.repository), displayName: "open-jev-deberta-v3-large",
                                isDefault: true, registryName: name,
                                source: .hub(NFKMLXOpenJevDeBERTa.repository, revision: NFKMLXOpenJevDeBERTa.measuredRevision,
                                             files: releaseFiles(required: NFKMLXOpenJevDeBERTa.releaseFiles, weights: [])))],
            license: license(NFKMLXOpenJevDeBERTa.repository))
    }
}

// MARK: - Forecasting

extension NFKMLXCatalogLanguage {

    fileprivate static var chronosEntry: NFKMLXCatalogEntry {
        let release = NFKMLXLanguageRelease(name: NFKMLXChronos.modelName, displayName: "Chronos-Bolt (base)",
                                            repository: "amazon/chronos-bolt-base")
        let horizon = NFKMLXChronosConfiguration().predictionLength
        return NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: "chronos-bolt",
            summary: "Amazon's Chronos-Bolt, a patched T5 encoder-decoder that forecasts a series as nine quantiles.",
            registryName: release.name, tasks: [NFKModelTaskForecasting],
            inputs: forecastInputs, outputs: forecastOutputs(mean: false),
            parameters: [horizonOption(horizon, maximum: horizon)],
            variants: [variant(release, files: [NFKModelFile(path: "model.safetensors", role: NFKModelFileRoleWeights)],
                               layout: NFKModelSourceLayoutFile)],
            license: license(release.repository))
    }

    fileprivate static var timesFMEntry: NFKMLXCatalogEntry {
        let release = NFKMLXLanguageRelease(name: NFKMLXTimesFM.modelName, displayName: "TimesFM 2.5 200M",
                                            repository: "google/timesfm-2.5-200m-pytorch",
                                            revision: "1d952420fba87f3c6dee4f240de0f1a0fbc790e3")
        let options = NFKMLXTimesFMForecastOptions()
        return NFKMLXCatalogEntry(
            identifier: release.name, displayName: release.displayName, family: "timesfm",
            summary: "Google's TimesFM 2.5, a decoder-only forecaster with point, mean, and quantile heads.",
            registryName: release.name, tasks: [NFKModelTaskForecasting],
            inputs: forecastInputs, outputs: forecastOutputs(mean: true),
            parameters: [
                horizonOption(NFKMLXTimesFMConfiguration.v2_5.outputPatchLength),
                .option(NFKMLXTimesFMParameterKey.maximumContext, kind: NFKModelOptionKindInteger, title: "Context",
                        help: "The most recent steps the forecast reads, rounded up to a multiple of 32.",
                        defaultValue: options.maximumContext, minimum: 1, step: 1),
                .option(NFKMLXTimesFMParameterKey.normalizesInputs, kind: NFKModelOptionKindBoolean, title: "Normalize",
                        help: "Normalizes the context by its mean and standard deviation and restores the forecast after.",
                        defaultValue: options.normalizesInputs),
                .option(NFKMLXTimesFMParameterKey.usesContinuousQuantileHead, kind: NFKModelOptionKindBoolean,
                        title: "Continuous quantiles",
                        help: "Takes the quantiles from the continuous quantile head, which limits the horizon to 1024.",
                        defaultValue: options.usesContinuousQuantileHead),
                .option(NFKMLXTimesFMParameterKey.forcesFlipInvariance, kind: NFKModelOptionKindBoolean, title: "Flip invariance",
                        help: "Averages the forecast with the negated forecast of the negated series.",
                        defaultValue: options.forcesFlipInvariance),
                .option(NFKMLXTimesFMParameterKey.infersPositivity, kind: NFKModelOptionKindBoolean, title: "Infer positivity",
                        help: "Clamps the forecast at zero when every context value is nonnegative.",
                        defaultValue: options.infersPositivity),
                .option(NFKMLXTimesFMParameterKey.fixesQuantileCrossing, kind: NFKModelOptionKindBoolean,
                        title: "Fix quantile crossing", help: "Makes the quantiles monotonic outward from the median.",
                        defaultValue: options.fixesQuantileCrossing),
            ],
            variants: [variant(release, files: releaseFiles(required: NFKMLXTimesFM.requiredFiles,
                                                            weights: NFKMLXTimesFM.weightFiles))],
            license: license(release.repository), modelTypes: ["timesfm", "timesfm2_5"])
    }
}
