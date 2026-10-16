//
//  NFKMLXCatalogImage.swift
//  InferKitMLX
//
//  The catalog entries of the image models: image restoration, upscaling, depth, segmentation, matting,
//  detection, pose, embedding, vision-language reading, and image generation.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The catalog entries of the image models, and the registrations their registry names need.
///
/// @discussion Each entry is one registry name. A family of names is built by a loop over the variant
/// enum its `register()` loops over, so an entry's name is the name the registry holds. A single-file
/// model's source is the file the registry URL points at, as its publisher serves it; a release-directory
/// model's source lists the files the class's own download fetches.
enum NFKMLXCatalogImage {

    /// The image models' entries, in the order ``NFKMLXModelDescriptors`` lists them.
    static var entries: [NFKMLXCatalogEntry] {
        upscalers + restorers + colorizers + stylizers + depthEstimators + segmenters + matters
            + promptSegmenters + detectors + poseEstimators + embedders + readers + generators
    }

    /// Registers the image models that have a factory and no registry name, under the names their entries
    /// give. ``NFKMLXReferenceModels/registerAll()`` calls it.
    static func registerAdditionalModels() {
        for variant in swinIRReleases {
            NFKMLXModelRegistry.register(name: swinIRName(variant)) { weightsURL in
                try NFKMLXSwinIR.backend(variant: variant, weightsURL: weightsURL)
            }
        }
        for variant in nafNetAdditionalReleases {
            NFKMLXModelRegistry.register(name: nafNetName(variant)) { weightsURL in
                try NFKMLXNAFNet.backend(variant: variant, weightsURL: weightsURL)
            }
        }
        NFKMLXModelRegistry.register(name: codeFormerPhotoName) { weightsURL in
            try NFKMLXCodeFormer.photoBackend(fidelity: NFKMLXCodeFormerConfiguration.base.fidelity,
                                              weightsURL: weightsURL, detector: NFKMLXVisionFaceDetector())
        }
        for variant in depth3Releases {
            NFKMLXModelRegistry.register(name: depth3EstimatorName(variant)) { weightsURL in
                try NFKMLXDepth3Estimator.estimator(variant: variant, weightsURL: weightsURL).inferenceBackend()
            }
        }
        for variant in samReleases {
            NFKMLXModelRegistry.register(name: samName(variant)) { weightsURL in
                try NFKMLXSAM.backend(variant: variant, weightsURL: weightsURL)
            }
        }
        for release in sam2AdditionalReleases {
            NFKMLXModelRegistry.register(name: sam2Name(release)) { weightsURL in
                try NFKMLXSAM2.backend(variant: release.variant, release: release.release, weightsURL: weightsURL)
            }
        }
        NFKMLXGenerationBackend.registerDirectoryModel(sam31Name) { try NFKMLXSAM3.backend(directoryURL: $0) }
        for variant in yoloV8AdditionalReleases {
            NFKMLXModelRegistry.register(name: yoloV8Name(variant)) { weightsURL in
                try NFKMLXYOLO.backend(variant: variant, weightsURL: weightsURL, labels: nil)
            }
        }
        NFKMLXGenerationBackend.registerDirectoryModel(sigLIP2ProbeName) { try sigLIP2ProbeBackend(directoryURL: $0) }
        for release in florence2Releases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) { try NFKMLXFlorence2.backend(directoryURL: $0) }
        }
        for release in trOCRReleases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) { try NFKMLXTrOCR.backend(directoryURL: $0) }
        }
        for release in tableTransformerReleases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) {
                try NFKMLXTableTransformer.backend(directoryURL: $0)
            }
        }
        for release in sa2VAReleases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) { try NFKMLXSa2VA.backend(directoryURL: $0) }
        }
        for model in stableDiffusionReleases {
            NFKMLXGenerationBackend.registerDirectoryModel(stableDiffusionName(model)) {
                try NFKMLXTextToImage.backend(model: model, directoryURL: $0)
            }
        }
        for adapter in stableDiffusionAdapters {
            NFKMLXGenerationBackend.registerDirectoryModel(adapter.name) {
                try adapter.backend(directoryURL: $0, precision: .checkpoint)
            }
        }
        NFKMLXGenerationBackend.registerDirectoryModel(fluxDevName) {
            try NFKMLXFlux.backend(directoryURL: $0, schedule: .flux)
        }
        for release in flux2AdditionalReleases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) { try NFKMLXFlux2.backend(directoryURL: $0) }
        }
        for release in sd3AdditionalReleases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) {
                try NFKMLXSD3Generator.backend(directoryURL: $0)
            }
        }
        NFKMLXGenerationBackend.registerDirectoryModel(sana1600MName) { try NFKMLXSANAGenerator.backend(directoryURL: $0) }
        NFKMLXGenerationBackend.registerDirectoryModel(zImageBaseName) {
            try NFKMLXZImageGenerator.backend(directoryURL: $0)
        }
    }

    // MARK: - Entry helpers

    /// An entry registered under `name` with one default variant read from `source`.
    static func model(_ name: String, _ displayName: String, family: String, summary: String, tasks: [String],
                      inputs: [NFKModelPort], outputs: [NFKModelPort], parameters: [NFKModelOption] = [],
                      license: NFKModelLicense, source: NFKModelSource,
                      residencies: [String] = [NFKModelResidencyResident]) -> NFKMLXCatalogEntry {
        NFKMLXCatalogEntry(identifier: name, displayName: displayName, family: family, summary: summary,
                           registryName: name, tasks: tasks, inputs: inputs, outputs: outputs,
                           parameters: parameters,
                           variants: [.variant("default", displayName: displayName, isDefault: true,
                                               registryName: name, source: source)],
                           license: license, residencies: residencies)
    }

    /// The role a release file plays, read from its name.
    static func role(of path: String) -> String {
        let name = (path as NSString).lastPathComponent
        let weightSuffixes = [".safetensors", ".bin", ".pth", ".pt", ".pkl", ".ckpt", ".jit", ".zip",
                              ".safetensors.index.json"]
        if weightSuffixes.contains(where: name.hasSuffix) {
            return NFKModelFileRoleWeights
        }
        let tokenizerNames: Set<String> = ["vocab.json", "merges.txt", "spiece.model", "sentencepiece.bpe.model",
                                           "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"]
        if name.hasPrefix("tokenizer") || tokenizerNames.contains(name) {
            return NFKModelFileRoleTokenizer
        }
        if name.hasSuffix("config.json") || name == "model_index.json" || name.hasSuffix(".yaml") {
            return NFKModelFileRoleConfig
        }
        return NFKModelFileRoleOther
    }

    /// A file of a source, its role read from its name.
    static func file(_ path: String, optional: Bool = false) -> NFKModelFile {
        NFKModelFile(path: path, byteCount: nil, sha256: nil, role: role(of: path), optional: optional,
                     extensions: nil)
    }

    /// One file at `address`, the original download, loaded as a single checkpoint.
    static func download(_ address: String, requiresConversion: Bool) -> NFKModelSource {
        let url = URL(string: address)
        let files = url.map { [file($0.lastPathComponent)] } ?? []
        return NFKModelSource(kind: NFKModelSourceKindURL, repository: nil, revision: nil, url: url,
                              layout: NFKModelSourceLayoutFile, files: files,
                              requiresConversion: requiresConversion, extensions: nil)
    }

    /// A single checkpoint the publisher serves from no direct download address: `page` is where its
    /// release is described.
    static func releasePage(_ page: String, requiresConversion: Bool) -> NFKModelSource {
        NFKModelSource(kind: NFKModelSourceKindURL, repository: nil, revision: nil, url: URL(string: page),
                       layout: NFKModelSourceLayoutFile, files: [], requiresConversion: requiresConversion,
                       extensions: nil)
    }

    /// One file of a Hugging Face repository, loaded as a single checkpoint.
    static func hubFile(_ repository: String, _ path: String, revision: String? = nil,
                        requiresConversion: Bool) -> NFKModelSource {
        .hub(repository, revision: revision, layout: NFKModelSourceLayoutFile, files: [file(path)],
             requiresConversion: requiresConversion)
    }

    /// Several files of a Hugging Face repository that a conversion combines into the one checkpoint the
    /// registry URL points at.
    static func hubFiles(_ repository: String, _ paths: [String]) -> NFKModelSource {
        .hub(repository, layout: NFKModelSourceLayoutFile, files: paths.map { file($0) }, requiresConversion: true)
    }

    /// A release directory fetched as `NFKMLXReleaseDownload.directory(repo:revision:cacheDirectoryURL:required:optional:weights:)`
    /// fetches it. Weight candidates are alternatives, so several are each optional.
    static func hubRelease(_ repository: String, revision: String? = nil, required: [String],
                           optional: [String] = [], weights: [String]) -> NFKModelSource {
        let weightFiles = weights.map { file($0, optional: weights.count > 1) }
        return .hub(repository, revision: revision, layout: NFKModelSourceLayoutDirectory,
                    files: required.map { file($0) } + optional.map { file($0, optional: true) } + weightFiles)
    }

    /// A release directory fetched component by component, as `NFKMLXReleaseDownload.directory(repo:revision:cacheDirectoryURL:components:)`
    /// fetches it.
    static func hubRelease(_ repository: String, revision: String? = nil,
                           components: [NFKMLXReleaseComponent]) -> NFKModelSource {
        .hub(repository, revision: revision, layout: NFKModelSourceLayoutDirectory, files: componentFiles(components))
    }

    /// The files of a release's components: each required file, each optional one, and each weights
    /// entry, optional where the component lists several.
    static func componentFiles(_ components: [NFKMLXReleaseComponent]) -> [NFKModelFile] {
        components.flatMap { component -> [NFKModelFile] in
            component.required.map { file($0) } + component.optional.map { file($0, optional: true) }
                + component.weights.map { file($0, optional: component.weights.count > 1) }
        }
    }

    /// A ControlNet's release: its own repository's files, and the files of the base release it steers,
    /// each naming the base's repository.
    static func controlNetRelease(_ repository: String, components: [NFKMLXReleaseComponent], base: String,
                                  baseComponents: [NFKMLXReleaseComponent]) -> NFKModelSource {
        let baseFiles = componentFiles(baseComponents).map { file in
            NFKModelFile.file(file.path, role: file.role, optional: file.isOptional, repository: base)
        }
        return .hub(repository, layout: NFKModelSourceLayoutDirectory, files: componentFiles(components) + baseFiles)
    }

    /// A model built from files the app supplies, with no download of its own.
    static var appSupplied: NFKModelSource {
        NFKModelSource(kind: NFKModelSourceKindNone, repository: nil, revision: nil, url: nil,
                       layout: NFKModelSourceLayoutDirectory, files: nil, requiresConversion: false, extensions: nil)
    }

    // MARK: Ports

    /// The required source image.
    static func imageInput(_ help: String, title: String = "Image") -> NFKModelPort {
        .input(NFKInputImage, required: true, title: title, help: help)
    }

    /// The image every result carries.
    static func imageOutput(_ help: String, title: String = "Image") -> NFKModelPort {
        .output(NFKOutputImage, required: true, title: title, help: help)
    }

    /// The matting contract's hint, which the model does not read.
    static var unreadHint: NFKModelPort {
        .input(NFKInputMask, title: "Hint", help: "The matting contract's hint mask. This model does not read it.")
    }

    /// The straight foreground a matting backend writes.
    static var foregroundOutput: NFKModelPort {
        .output(NFKOutputImage, required: true, title: "Foreground",
                help: "The plate as straight foreground with the matte as its alpha.")
    }

    /// The matte a matting backend writes on its own.
    static var matteOutput: NFKModelPort {
        .output(NFKOutputMask, required: true, title: "Matte", help: "The alpha matte as a gray image.")
    }

    /// The gray label map a segmentation backend writes.
    static func labelMapOutput(classes: Int) -> NFKModelPort {
        imageOutput("The class of each pixel as a gray label map: the class index is round(gray × \(classes - 1)).",
                    title: "Label map")
    }

    // MARK: Options

    /// The output width, read by every image generator.
    static func widthOption(_ defaultValue: Int?, help: String) -> NFKModelOption {
        .option(NFKParameterWidth, title: "Width", help: help, defaultValue: defaultValue, minimum: 16, step: 16)
    }

    /// The output height, read by every image generator.
    static func heightOption(_ defaultValue: Int?, help: String) -> NFKModelOption {
        .option(NFKParameterHeight, title: "Height", help: help, defaultValue: defaultValue, minimum: 16, step: 16)
    }

    /// The seed of the initial noise.
    static func seedOption(_ defaultValue: Int?) -> NFKModelOption {
        .option(NFKParameterSeed, title: "Seed", help: "The seed of the initial noise.", defaultValue: defaultValue)
    }

    /// The denoising steps.
    static func stepsOption(_ defaultValue: Int) -> NFKModelOption {
        .option(NFKParameterSteps, title: "Steps", help: "The denoising steps the sampler runs.",
                defaultValue: defaultValue, minimum: 1, step: 1)
    }

    /// The guidance weight.
    static func guidanceOption(_ defaultValue: Double, help: String) -> NFKModelOption {
        .option(NFKParameterGuidanceScale, title: "Guidance", help: help, defaultValue: defaultValue, minimum: 0)
    }

    /// The share of a source image the sampler regenerates.
    static func strengthOption(_ defaultValue: Double, help: String) -> NFKModelOption {
        .option(NFKParameterStrength, title: "Strength", help: help, defaultValue: defaultValue, minimum: 0, maximum: 1)
    }

    /// The options every `NFKMLXDiffusionBackend` reads, at the model's own step count.
    static func diffusionOptions(steps: Int, guidance: Double, guidanceHelp: String) -> [NFKModelOption] {
        [stepsOption(steps), guidanceOption(guidance, help: guidanceHelp),
         strengthOption(1, help: "How much of the source the sampler regenerates. 1 starts from full noise."),
         .option(NFKParameterSeed, title: "Seed", help: "The seed of the initial noise. A run without one uses a fixed seed.")]
    }

    // MARK: Licenses

    static let bsd3RealESRGAN = NFKModelLicense.license("BSD-3-Clause", name: "BSD 3-Clause License",
                                                        url: "https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE")
    static let apacheSwinIR = NFKModelLicense.license("Apache-2.0", name: "Apache License 2.0",
                                                      url: "https://github.com/JingyunLiang/SwinIR/blob/main/LICENSE")
    static let apacheHAT = NFKModelLicense.license("Apache-2.0", name: "Apache License 2.0",
                                                   url: "https://github.com/XPixelGroup/HAT/blob/main/LICENSE")
    static let noLicense = NFKModelLicense.license(nil, name: "No license stated",
                                                   url: "https://github.com/Li-Chongyi/Zero-DCE",
                                                   restrictsCommercialUse: true,
                                                   note: "The publisher states no license, so copyright grants no use beyond what the law allows without one.")
    static let ccByNC4 = NFKModelLicense.license("CC-BY-NC-4.0", name: "Creative Commons Attribution-NonCommercial 4.0",
                                                 url: "https://creativecommons.org/licenses/by-nc/4.0/",
                                                 restrictsCommercialUse: true,
                                                 note: "Non-commercial use only, with attribution.")
    static let apacheSAM = NFKModelLicense.license("Apache-2.0", name: "Apache License 2.0",
                                                   url: "https://github.com/facebookresearch/segment-anything/blob/main/LICENSE")
    static let apacheSAM2 = NFKModelLicense.license("Apache-2.0", name: "Apache License 2.0",
                                                    url: "https://github.com/facebookresearch/sam2/blob/main/LICENSE")
    static let samLicense = NFKModelLicense.license(nil, name: "SAM License", url: "https://huggingface.co/facebook/sam3",
                                                    gated: true, note: "Gated: accept the terms on Hugging Face.")
    static let ultralytics = NFKModelLicense.license("AGPL-3.0-only", name: "GNU Affero General Public License v3.0",
                                                     url: "https://github.com/ultralytics/ultralytics/blob/main/LICENSE",
                                                     note: "Software that serves the weights over a network publishes its source under the AGPL; Ultralytics sells a separate enterprise license.")
    static let openAICLIP = NFKModelLicense.mit(url: "https://github.com/openai/CLIP/blob/main/LICENSE")
    static let stabilityCommunity = NFKModelLicense.license(nil, name: "Stability AI Community License",
                                                            url: "https://stability.ai/community-license-agreement",
                                                            restrictsCommercialUse: true,
                                                            note: "Free below US$1M in annual revenue; above it, Stability AI's enterprise license applies.")
    static let fluxNonCommercial = NFKModelLicense.license(nil, name: "FLUX Non-Commercial License",
                                                           url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md",
                                                           gated: true, restrictsCommercialUse: true,
                                                           note: "Gated: accept the terms on Hugging Face. Non-commercial use only.")
    static let fluxDevNonCommercial = NFKModelLicense.license(nil, name: "FLUX.1 [dev] Non-Commercial License",
                                                              url: "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md",
                                                              gated: true, restrictsCommercialUse: true,
                                                              note: "Gated: accept the terms on Hugging Face. Non-commercial use only.")

    /// The Apache License 2.0 on a repository whose terms are accepted on Hugging Face first.
    static var gatedApache2: NFKModelLicense {
        .license("Apache-2.0", name: "Apache License 2.0", url: "https://www.apache.org/licenses/LICENSE-2.0",
                 gated: true, note: "Gated: accept the terms on Hugging Face.")
    }

    // MARK: - Upscaling

    static var upscalers: [NFKMLXCatalogEntry] {
        realESRGANEntries + swinIREntries + hatEntries + [sdUpscaler]
    }

    static func upscalerPorts(scale: Int) -> (inputs: [NFKModelPort], outputs: [NFKModelPort]) {
        ([imageInput("The image to enlarge.")],
         [imageOutput("The image at \(scale) times the input's width and height.")])
    }

    /// The Real-ESRGAN releases, in the order `NFKMLXRealESRGAN.register()` registers them.
    static let realESRGANReleases: [NFKMLXRealESRGANVariant] = [.x4, .anime, .x2, .generalX4V3, .animeVideoV3]

    static var realESRGANEntries: [NFKMLXCatalogEntry] {
        realESRGANReleases.map { variant in
            let release: (name: String, displayName: String, address: String, scale: Int, converts: Bool,
                          summary: String)
            switch variant {
            case .x4:
                release = (NFKMLXRealESRGAN.modelName, "Real-ESRGAN ×4plus",
                           "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                           4, true, "A ×4 upscaler for real-world photographs with blur, noise, and compression.")
            case .anime:
                release = (NFKMLXRealESRGAN.animeModelName, "Real-ESRGAN ×4plus anime 6B",
                           "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                           4, true, "A six-block ×4 upscaler tuned for anime illustration.")
            case .x2:
                release = (NFKMLXRealESRGAN.x2ModelName, "Real-ESRGAN ×2plus",
                           "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                           2, true, "A ×2 upscaler for real-world photographs.")
            case .generalX4V3:
                release = (NFKMLXRealESRGAN.generalV3ModelName, "Real-ESRGAN general ×4 v3",
                           "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                           4, false, "A compact ×4 upscaler for general scenes.")
            case .animeVideoV3:
                release = (NFKMLXRealESRGAN.animeVideoV3ModelName, "Real-ESRGAN anime video ×4 v3",
                           "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
                           4, false, "A compact ×4 upscaler for anime video frames.")
            }
            let ports = upscalerPorts(scale: release.scale)
            return model(release.name, release.displayName, family: "real-esrgan", summary: release.summary,
                         tasks: [NFKModelTaskUpscaling], inputs: ports.inputs, outputs: ports.outputs,
                         license: bsd3RealESRGAN,
                         source: download(release.address, requiresConversion: release.converts))
        }
    }

    /// The SwinIR releases with a registry name of their own.
    static let swinIRReleases: [NFKMLXSwinIRVariant] = [
        .classicalX2, .classicalX3, .classicalX4, .classicalX8, .lightweightSRX2, .lightweightSRX3, .lightweightSRX4,
        .realWorldX4Medium, .realWorldX4Large,
    ]

    /// The registry name of a SwinIR release.
    static func swinIRName(_ variant: NFKMLXSwinIRVariant) -> String {
        swinIRRelease(variant).name
    }

    static func swinIRRelease(_ variant: NFKMLXSwinIRVariant)
        -> (name: String, displayName: String, file: String, scale: Int, summary: String) {
        switch variant {
        case .lightweightX2:
            return (NFKMLXSwinIR.modelName, "SwinIR (test geometry)", "", 2,
                    "SwinIR at its small default geometry, which no released checkpoint fits.")
        case .classicalX2:
            return ("swinir-classical-x2", "SwinIR-M classical ×2", "001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth", 2,
                    "A ×2 transformer upscaler trained on bicubic downsampling.")
        case .classicalX3:
            return ("swinir-classical-x3", "SwinIR-M classical ×3", "001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth", 3,
                    "A ×3 transformer upscaler trained on bicubic downsampling.")
        case .classicalX4:
            return ("swinir-classical-x4", "SwinIR-M classical ×4", "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth", 4,
                    "A ×4 transformer upscaler trained on bicubic downsampling.")
        case .classicalX8:
            return ("swinir-classical-x8", "SwinIR-M classical ×8", "001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth", 8,
                    "A ×8 transformer upscaler trained on bicubic downsampling.")
        case .lightweightSRX2:
            return ("swinir-lightweight-x2", "SwinIR-S lightweight ×2", "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth", 2,
                    "A small ×2 transformer upscaler.")
        case .lightweightSRX3:
            return ("swinir-lightweight-x3", "SwinIR-S lightweight ×3", "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth", 3,
                    "A small ×3 transformer upscaler.")
        case .lightweightSRX4:
            return ("swinir-lightweight-x4", "SwinIR-S lightweight ×4", "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth", 4,
                    "A small ×4 transformer upscaler.")
        case .realWorldX4Medium:
            return ("swinir-real-x4-m", "SwinIR-M real-world ×4", "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth", 4,
                    "A ×4 transformer upscaler for real-world degradations.")
        case .realWorldX4Large:
            return ("swinir-real-x4-l", "SwinIR-L real-world ×4", "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth", 4,
                    "A large ×4 transformer upscaler for real-world degradations.")
        }
    }

    static var swinIREntries: [NFKMLXCatalogEntry] {
        let testGeometry = swinIRRelease(.lightweightX2)
        let testPorts = upscalerPorts(scale: testGeometry.scale)
        var standIn = model(testGeometry.name, testGeometry.displayName, family: "swinir", summary: testGeometry.summary,
                            tasks: [NFKModelTaskUpscaling], inputs: testPorts.inputs, outputs: testPorts.outputs,
                            license: .inferKit(note: "Built with random weights; no released checkpoint fits this geometry."),
                            source: .noDownload)
        standIn.standIn = true
        return [standIn] + swinIRReleases.map { variant in
            let release = swinIRRelease(variant)
            let ports = upscalerPorts(scale: release.scale)
            return model(release.name, release.displayName, family: "swinir", summary: release.summary,
                         tasks: [NFKModelTaskUpscaling], inputs: ports.inputs, outputs: ports.outputs,
                         license: apacheSwinIR,
                         source: download("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/\(release.file)",
                                          requiresConversion: true))
        }
    }

    /// The HAT releases, in the order `NFKMLXHAT.register()` registers them.
    static let hatReleases: [NFKMLXHATVariant] = [.base, .large, .realWorld]

    static var hatEntries: [NFKMLXCatalogEntry] {
        hatReleases.map { variant in
            let ports = upscalerPorts(scale: 4)
            let name = NFKMLXHAT.specs(for: variant).name
            switch variant {
            case .base:
                return model(name, "HAT ×4", family: "hat",
                             summary: "A ×4 hybrid-attention transformer upscaler trained on bicubic downsampling.",
                             tasks: [NFKModelTaskUpscaling], inputs: ports.inputs, outputs: ports.outputs,
                             license: apacheHAT,
                             source: releasePage("https://github.com/XPixelGroup/HAT", requiresConversion: false))
            case .large:
                return model(name, "HAT-L ×4", family: "hat",
                             summary: "The large ×4 hybrid-attention transformer upscaler, pretrained on ImageNet.",
                             tasks: [NFKModelTaskUpscaling], inputs: ports.inputs, outputs: ports.outputs,
                             license: apacheHAT,
                             source: hubFile("anchuang/HAT-L_SRx4_ImageNet-pretrain", "HAT-L_SRx4_ImageNet-pretrain.pth",
                                             requiresConversion: false))
            case .realWorld:
                return model(name, "Real-HAT-GAN ×4", family: "hat",
                             summary: "A ×4 hybrid-attention transformer upscaler for real-world degradations.",
                             tasks: [NFKModelTaskUpscaling], inputs: ports.inputs, outputs: ports.outputs,
                             license: apacheHAT,
                             source: hubFile("anchuang/Real_HAT_GAN_SRx4_sharper", "Real_HAT_GAN_sharper.pth",
                                             requiresConversion: false))
            }
        }
    }

    static var sdUpscaler: NFKMLXCatalogEntry {
        model(NFKMLXSDUpscaler.modelName, "Stable Diffusion ×4 Upscaler", family: "stable-diffusion",
              summary: "A latent-diffusion ×4 upscaler conditioned on the low-resolution image.",
              tasks: [NFKModelTaskUpscaling],
              inputs: [imageInput("The image to enlarge."),
                       .input(NFKInputMask, title: "Mask", help: "The diffusion contract's mask. The upscaler does not read it.")],
              outputs: [imageOutput("The image at four times the input's width and height.")],
              parameters: diffusionOptions(steps: 20, guidance: 1,
                                           guidanceHelp: "The guidance weight the denoiser receives. The upscaler conditions on the image alone."),
              license: .license(nil, name: "CreativeML Open RAIL++-M License",
                                url: "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/blob/main/LICENSE.md",
                                note: "Use restrictions apply."),
              source: hubFiles("stabilityai/stable-diffusion-x4-upscaler",
                               ["unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"]))
    }

    // MARK: - Restoration

    static var restorers: [NFKMLXCatalogEntry] {
        nafNetEntries + [zeroDCE, zeroDCEPlus, lama, codeFormer, codeFormerPhoto, sdInpaint, taesd]
    }

    /// The NAFNet releases other than the SIDD width-32 release `nafnet` builds.
    static let nafNetAdditionalReleases: [NFKMLXNAFNetVariant] = [.goPro, .reds, .siddWidth64, .goProWidth64]

    static func nafNetName(_ variant: NFKMLXNAFNetVariant) -> String {
        nafNetRelease(variant).name
    }

    static func nafNetRelease(_ variant: NFKMLXNAFNetVariant)
        -> (name: String, displayName: String, file: String, summary: String) {
        switch variant {
        case .sidd:
            return (NFKMLXNAFNet.modelName, "NAFNet SIDD width 32", "NAFNet-SIDD-width32.pth",
                    "A denoiser for camera sensor noise, trained on SIDD.")
        case .goPro:
            return ("nafnet-gopro", "NAFNet GoPro width 32", "NAFNet-GoPro-width32.pth",
                    "A motion deblurrer trained on GoPro.")
        case .reds:
            return ("nafnet-reds", "NAFNet REDS width 64", "NAFNet-REDS-width64.pth",
                    "A deblurrer for compressed video frames, trained on REDS.")
        case .siddWidth64:
            return ("nafnet-sidd-width64", "NAFNet SIDD width 64", "NAFNet-SIDD-width64.pth",
                    "The wider denoiser for camera sensor noise, trained on SIDD.")
        case .goProWidth64:
            return ("nafnet-gopro-width64", "NAFNet GoPro width 64", "NAFNet-GoPro-width64.pth",
                    "The wider motion deblurrer, trained on GoPro.")
        }
    }

    static var nafNetEntries: [NFKMLXCatalogEntry] {
        ([NFKMLXNAFNetVariant.sidd] + nafNetAdditionalReleases).map { variant in
            let release = nafNetRelease(variant)
            return model(release.name, release.displayName, family: "nafnet", summary: release.summary,
                         tasks: [NFKModelTaskImageRestoration],
                         inputs: [imageInput("The image to restore.")],
                         outputs: [imageOutput("The restored image at the input's size.")],
                         license: .mit(url: "https://github.com/megvii-research/NAFNet/blob/main/LICENSE",
                                       note: "Served from a Hugging Face mirror of the authors' release."),
                         source: hubFile("nyanko7/nafnet-models", release.file, requiresConversion: true))
        }
    }

    static var zeroDCE: NFKMLXCatalogEntry {
        model(NFKMLXZeroDCE.modelName, "Zero-DCE", family: "zero-dce",
              summary: "A low-light enhancer that brightens an image through learned per-pixel curves.",
              tasks: [NFKModelTaskImageRestoration],
              inputs: [imageInput("The dark image to brighten.")],
              outputs: [imageOutput("The brightened image at the input's size.")],
              license: noLicense,
              source: download("https://github.com/Li-Chongyi/Zero-DCE/raw/master/Zero-DCE_code/snapshots/Epoch99.pth",
                               requiresConversion: true))
    }

    static var zeroDCEPlus: NFKMLXCatalogEntry {
        model(NFKMLXZeroDCEPlus.modelName, "Zero-DCE++", family: "zero-dce",
              summary: "A lighter low-light enhancer that estimates its curves at reduced resolution.",
              tasks: [NFKModelTaskImageRestoration],
              inputs: [imageInput("The dark image to brighten.")],
              outputs: [imageOutput("The brightened image at the input's size.")],
              license: .license(nil, name: "No license stated", url: "https://github.com/Li-Chongyi/Zero-DCE_extension",
                                restrictsCommercialUse: true,
                                note: "The publisher states no license, so copyright grants no use beyond what the law allows without one."),
              source: download("https://github.com/Li-Chongyi/Zero-DCE_extension/raw/main/Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B/Epoch99.pth",
                               requiresConversion: false))
    }

    static var lama: NFKMLXCatalogEntry {
        model(NFKMLXLaMa.modelName, "Big LaMa", family: "lama",
              summary: "An inpainter that fills a masked region with surrounding texture.",
              tasks: [NFKModelTaskInpainting],
              inputs: [imageInput("The plate to fill."),
                       .input(NFKInputMask, title: "Mask",
                              help: "White marks the region to fill. Without a mask the plate passes through.")],
              outputs: [imageOutput("The plate with the masked region filled.")],
              license: .license("Apache-2.0", name: "Apache License 2.0",
                                url: "https://github.com/advimman/lama/blob/main/LICENSE"),
              source: hubFile("smartywu/big-lama", "big-lama.zip", requiresConversion: true))
    }

    static let codeFormerLicense = NFKModelLicense.license(nil, name: "S-Lab License 1.0",
                                                           url: "https://github.com/sczhou/CodeFormer/blob/master/LICENSE",
                                                           restrictsCommercialUse: true,
                                                           note: "Non-commercial use only.")

    static let codeFormerSource = download("https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
                                           requiresConversion: true)

    static var codeFormer: NFKMLXCatalogEntry {
        model(NFKMLXCodeFormer.modelName, "CodeFormer", family: "codeformer",
              summary: "A face restorer that rebuilds an aligned 512×512 face from a learned codebook.",
              tasks: [NFKModelTaskImageRestoration],
              inputs: [imageInput("An aligned 512×512 face crop.", title: "Face")],
              outputs: [imageOutput("The restored face.", title: "Face")],
              license: codeFormerLicense, source: codeFormerSource)
    }

    /// The registry name of the whole-photograph CodeFormer backend.
    static let codeFormerPhotoName = "codeformer-photo"

    static var codeFormerPhoto: NFKMLXCatalogEntry {
        model(codeFormerPhotoName, "CodeFormer (whole photograph)", family: "codeformer",
              summary: "A face restorer that finds each face with Apple's Vision, restores it, and composites it back.",
              tasks: [NFKModelTaskImageRestoration, NFKModelTaskFaceDetection],
              inputs: [imageInput("The photograph whose faces to restore.")],
              outputs: [imageOutput("The photograph with every detected face restored. A photograph without a face passes through.")],
              license: codeFormerLicense, source: codeFormerSource)
    }

    static var sdInpaint: NFKMLXCatalogEntry {
        model(NFKMLXStableDiffusionInpaint.modelName, "Stable Diffusion 1.5 Inpainting", family: "stable-diffusion",
              summary: "A latent-diffusion inpainter that regenerates the masked region of a plate.",
              tasks: [NFKModelTaskInpainting],
              inputs: [imageInput("The plate to fill."),
                       .input(NFKInputMask, required: true, title: "Mask",
                              help: "White regenerates the region, and black keeps it.")],
              outputs: [imageOutput("The plate with the masked region regenerated.")],
              parameters: diffusionOptions(steps: NFKMLXSDInpaintConfiguration.stableDiffusion15.steps, guidance: 1,
                                           guidanceHelp: "The guidance weight the denoiser receives. The inpainter conditions on an empty prompt."),
              license: .license(nil, name: "CreativeML Open RAIL-M License",
                                url: "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting",
                                note: "Use restrictions apply."),
              source: hubFiles("stable-diffusion-v1-5/stable-diffusion-inpainting",
                               ["unet/diffusion_pytorch_model.bin", "vae/diffusion_pytorch_model.bin",
                                "text_encoder/pytorch_model.bin"]))
    }

    static var taesd: NFKMLXCatalogEntry {
        model(NFKMLXTAESD.modelName, "TAESD", family: "taesd",
              summary: "A tiny autoencoder that encodes an image to Stable Diffusion latents and decodes it back.",
              tasks: [NFKModelTaskImageToImage],
              inputs: [imageInput("The image to encode and decode.")],
              outputs: [imageOutput("The image decoded from its latent.")],
              license: .mit(url: "https://github.com/madebyollin/taesd/blob/main/LICENSE"),
              source: hubFiles("madebyollin/taesd", ["taesd_encoder.safetensors", "taesd_decoder.safetensors"]))
    }

    // MARK: - Colorization

    static var colorizers: [NFKMLXCatalogEntry] {
        [colorizerECCV16, colorizerSIGGRAPH17] + ddColorEntries
    }

    static let richZhangColorization = NFKModelLicense.license("BSD-2-Clause", name: "BSD 2-Clause License",
                                                               url: "https://github.com/richzhang/colorization/blob/master/LICENSE")

    static var colorizerECCV16: NFKMLXCatalogEntry {
        model(NFKMLXColorizer.modelName, "Colorful Image Colorization (ECCV 2016)", family: "colorization",
              summary: "A colorizer that predicts the color of a grayscale photograph.",
              tasks: [NFKModelTaskImageRestoration],
              inputs: [imageInput("The photograph to colorize. Only its lightness is read.")],
              outputs: [imageOutput("The colorized photograph.")],
              license: richZhangColorization,
              source: download("https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
                               requiresConversion: true))
    }

    static var colorizerSIGGRAPH17: NFKMLXCatalogEntry {
        model(NFKMLXSiggraphColorizer.modelName, "Interactive Colorization (SIGGRAPH 2017)", family: "colorization",
              summary: "A colorizer that predicts the color of a grayscale photograph, guided by optional color hints.",
              tasks: [NFKModelTaskImageRestoration],
              inputs: [imageInput("The photograph to colorize. Only its lightness is read."),
                       .input(NFKMLXSiggraphColorizer.hintKey, valueType: NFKModelValueTypeImage, title: "Hint",
                              help: "An image whose colors guide the colorization where the hint mask is set."),
                       .input(NFKMLXSiggraphColorizer.hintMaskKey, valueType: NFKModelValueTypeMask, title: "Hint mask",
                              help: "White marks the pixels whose hint colors apply.")],
              outputs: [imageOutput("The colorized photograph.")],
              license: richZhangColorization,
              source: download("https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth",
                               requiresConversion: true))
    }

    /// The DDColor releases, in the order `NFKMLXDDColor.register()` registers them.
    static let ddColorReleases: [NFKMLXDDColorVariant] = [.modelscope, .paper, .artistic]

    static var ddColorEntries: [NFKMLXCatalogEntry] {
        ddColorReleases.map { variant in
            let release: (displayName: String, file: String, summary: String)
            switch variant {
            case .modelscope:
                release = ("DDColor (ModelScope)", "ddcolor_modelscope.pth",
                           "A colorizer that decodes color through learned queries; the release the authors' demo runs.")
            case .paper:
                release = ("DDColor (paper)", "ddcolor_paper.pth",
                           "A colorizer that decodes color through learned queries; the weights the paper reports.")
            case .artistic:
                release = ("DDColor (artistic)", "ddcolor_artistic.pth",
                           "A colorizer that decodes color through learned queries, tuned for saturated color.")
            }
            return model(NFKMLXDDColor.specs(for: variant).name, release.displayName, family: "ddcolor",
                         summary: release.summary, tasks: [NFKModelTaskImageRestoration],
                         inputs: [imageInput("The photograph to colorize. Only its lightness is read.")],
                         outputs: [imageOutput("The colorized photograph.")],
                         license: .license("Apache-2.0", name: "Apache License 2.0",
                                           url: "https://github.com/piddnad/DDColor/blob/master/LICENSE"),
                         source: hubFile("piddnad/DDColor-models", release.file, requiresConversion: false))
        }
    }

    // MARK: - Style transfer

    static var stylizers: [NFKMLXCatalogEntry] {
        [fastStyleTransfer, adaIN]
    }

    static var fastStyleTransfer: NFKMLXCatalogEntry {
        model(NFKMLXStyleTransfer.modelName, "Fast Neural Style", family: "fast-neural-style",
              summary: "A stylizer that repaints an image in the one style its checkpoint was trained on.",
              tasks: [NFKModelTaskImageToImage],
              inputs: [imageInput("The image to stylize.")],
              outputs: [imageOutput("The stylized image.")],
              license: .license("BSD-3-Clause", name: "BSD 3-Clause License",
                                url: "https://github.com/pytorch/examples/blob/main/LICENSE"),
              source: releasePage("https://github.com/pytorch/examples/tree/main/fast_neural_style",
                                  requiresConversion: true))
    }

    static var adaIN: NFKMLXCatalogEntry {
        model(NFKMLXAdaIN.modelName, "AdaIN Style Transfer", family: "adain",
              summary: "A stylizer that repaints an image in the style of a second image.",
              tasks: [NFKModelTaskImageToImage],
              inputs: [imageInput("The content image to stylize.", title: "Content"),
                       .input(NFKInputControl, title: "Style",
                              help: "The image whose style is applied. Without one the content is reconstructed.")],
              outputs: [imageOutput("The stylized image.")],
              parameters: [strengthOption(1, help: "The blend between the content's features (0) and the style's (1).")],
              license: .mit(url: "https://github.com/naoto0804/pytorch-AdaIN/blob/master/LICENSE"),
              source: download("https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth",
                               requiresConversion: false))
    }

    // MARK: - Depth

    static var depthEstimators: [NFKMLXCatalogEntry] {
        depthAnythingEntries + depth3Entries + depth3EstimatorEntries + [marigold]
    }

    static var depthAnythingEntries: [NFKMLXCatalogEntry] {
        let releases: [(name: String, size: String, repository: String, file: String, license: NFKModelLicense)] = [
            (NFKMLXDepthAnything.modelName, "Small", "depth-anything/Depth-Anything-V2-Small",
             "depth_anything_v2_vits.pth", .apache2()),
            (NFKMLXDepthAnything.baseModelName, "Base", "depth-anything/Depth-Anything-V2-Base",
             "depth_anything_v2_vitb.pth", ccByNC4),
            (NFKMLXDepthAnything.largeModelName, "Large", "depth-anything/Depth-Anything-V2-Large",
             "depth_anything_v2_vitl.pth", ccByNC4),
        ]
        return releases.map { release in
            model(release.name, "Depth Anything V2 \(release.size)", family: "depth-anything",
                  summary: "A monocular estimator of relative depth.",
                  tasks: [NFKModelTaskDepthEstimation],
                  inputs: [imageInput("The image to estimate depth for.")],
                  outputs: [imageOutput("The relative depth as a gray image at the input's size, nearer brighter.",
                                        title: "Depth")],
                  license: release.license,
                  source: hubFile(release.repository, release.file, requiresConversion: true))
        }
    }

    /// The Depth Anything 3 releases, in the order `NFKMLXDepthAnything3.register()` registers them.
    static let depth3Releases: [NFKMLXDepth3Variant] = [.small, .base, .large]

    static func depth3Release(_ variant: NFKMLXDepth3Variant)
        -> (size: String, repository: String, license: NFKModelLicense) {
        switch variant {
        case .small: return ("Small", "depth-anything/DA3-SMALL", .apache2())
        case .base: return ("Base", "depth-anything/DA3-BASE", .apache2())
        case .large: return ("Large", "depth-anything/DA3-LARGE", ccByNC4)
        }
    }

    static var depth3Entries: [NFKMLXCatalogEntry] {
        depth3Releases.map { variant in
            let release = depth3Release(variant)
            return model(NFKMLXDepthAnything3.specs(for: variant).name, "Depth Anything 3 \(release.size)",
                         family: "depth-anything-3", summary: "A monocular estimator of depth.",
                         tasks: [NFKModelTaskDepthEstimation],
                         inputs: [imageInput("The image to estimate depth for.")],
                         outputs: [imageOutput("The depth as a gray image at the input's size, nearer brighter.", title: "Depth")],
                         license: release.license,
                         source: hubFile(release.repository, "model.safetensors", requiresConversion: false))
        }
    }

    /// The registry name of the Depth Anything 3 estimator at `variant`.
    static func depth3EstimatorName(_ variant: NFKMLXDepth3Variant) -> String {
        NFKMLXDepthAnything3.specs(for: variant).name + "-estimator"
    }

    static var depth3EstimatorEntries: [NFKMLXCatalogEntry] {
        depth3Releases.map { variant in
            let release = depth3Release(variant)
            return model(depth3EstimatorName(variant), "Depth Anything 3 \(release.size) (depth, camera, rays)",
                         family: "depth-anything-3",
                         summary: "A monocular estimator of depth, the camera, and the ray map.",
                         tasks: [NFKModelTaskDepthEstimation],
                         inputs: [imageInput("The image to estimate depth for.")],
                         outputs: [imageOutput("The depth as a gray image at the input's size, nearer brighter.", title: "Depth"),
                                   .output(NFKOutputStructured, required: true, title: "Geometry",
                                           help: "The camera, the ray map, and the ray map's confidence.")],
                         license: release.license,
                         source: hubFile(release.repository, "model.safetensors", requiresConversion: false))
        }
    }

    static var marigold: NFKMLXCatalogEntry {
        model(NFKMLXMarigold.modelName, "Marigold Depth v1.0", family: "marigold",
              summary: "A latent-diffusion estimator of relative depth fine-tuned from Stable Diffusion.",
              tasks: [NFKModelTaskDepthEstimation],
              inputs: [imageInput("The image to estimate depth for."),
                       .input(NFKInputMask, title: "Mask", help: "The diffusion contract's mask. Marigold does not read it.")],
              outputs: [imageOutput("The relative depth as a gray image.", title: "Depth")],
              parameters: diffusionOptions(steps: 10, guidance: 1,
                                           guidanceHelp: "The guidance weight the denoiser receives. Marigold does not apply it."),
              license: .license("Apache-2.0", name: "Apache License 2.0",
                                url: "https://huggingface.co/prs-eth/marigold-depth-v1-0/blob/main/LICENSE.txt"),
              source: hubFiles("prs-eth/marigold-depth-v1-0",
                               ["unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
                                "text_encoder/model.safetensors"]))
    }

    // MARK: - Semantic segmentation

    static var segmenters: [NFKMLXCatalogEntry] {
        [segFormer, deepLab, biSeNet, biSeNetV2]
    }

    static var segFormer: NFKMLXCatalogEntry {
        model(NFKMLXSegFormer.modelName, "SegFormer-B0 (ADE20K)", family: "segformer",
              summary: "A semantic segmenter that labels each pixel with one of the 150 ADE20K classes.",
              tasks: [NFKModelTaskSegmentation],
              inputs: [imageInput("The image to segment.")],
              outputs: [labelMapOutput(classes: 150)],
              license: .license(nil, name: "NVIDIA Source Code License for SegFormer",
                                url: "https://github.com/NVlabs/SegFormer/blob/master/LICENSE",
                                restrictsCommercialUse: true, note: "Research and evaluation use only."),
              source: hubFile("nvidia/segformer-b0-finetuned-ade-512-512", "pytorch_model.bin", requiresConversion: true))
    }

    static var deepLab: NFKMLXCatalogEntry {
        model(NFKMLXDeepLab.modelName, "DeepLabV3 ResNet-50", family: "deeplab",
              summary: "A semantic segmenter that labels each pixel with one of the 21 Pascal VOC classes.",
              tasks: [NFKModelTaskSegmentation],
              inputs: [imageInput("The image to segment.")],
              outputs: [labelMapOutput(classes: 21)],
              license: .license("BSD-3-Clause", name: "BSD 3-Clause License",
                                url: "https://github.com/pytorch/vision/blob/main/LICENSE"),
              source: download("https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
                               requiresConversion: true))
    }

    static let biSeNetLicense = NFKModelLicense.mit(url: "https://github.com/CoinCheung/BiSeNet/blob/master/LICENSE")

    static var biSeNet: NFKMLXCatalogEntry {
        model(NFKMLXBiSeNet.modelName, "BiSeNet V1 (Cityscapes)", family: "bisenet",
              summary: "A real-time semantic segmenter that labels each pixel with one of the 19 Cityscapes classes.",
              tasks: [NFKModelTaskSegmentation],
              inputs: [imageInput("The street scene to segment.")],
              outputs: [labelMapOutput(classes: 19)],
              license: biSeNetLicense,
              source: download("https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v1_city_new.pth",
                               requiresConversion: true))
    }

    static var biSeNetV2: NFKMLXCatalogEntry {
        model(NFKMLXBiSeNetV2.modelName, "BiSeNet V2 (Cityscapes)", family: "bisenet",
              summary: "A real-time semantic segmenter that labels each pixel with one of the 19 Cityscapes classes.",
              tasks: [NFKModelTaskSegmentation],
              inputs: [imageInput("The street scene to segment.")],
              outputs: [labelMapOutput(classes: 19)],
              license: biSeNetLicense,
              source: download("https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v2_city.pth",
                               requiresConversion: true))
    }

    // MARK: - Matting and background removal

    static var matters: [NFKMLXCatalogEntry] {
        [biRefNet, u2Net, u2NetP, isNet, modNet] + rvmEntries
    }

    static func matteEntry(_ name: String, _ displayName: String, family: String, summary: String,
                           license: NFKModelLicense, source: NFKModelSource) -> NFKMLXCatalogEntry {
        model(name, displayName, family: family, summary: summary, tasks: [NFKModelTaskMatting],
              inputs: [imageInput("The plate to matte.", title: "Plate"), unreadHint],
              outputs: [foregroundOutput, matteOutput], license: license, source: source)
    }

    static var biRefNet: NFKMLXCatalogEntry {
        matteEntry(NFKMLXBiRefNet.modelName, "BiRefNet", family: "birefnet",
                   summary: "A high-resolution background remover that separates the salient subject.",
                   license: .mit(url: "https://huggingface.co/ZhengPeng7/BiRefNet"),
                   source: hubFile("ZhengPeng7/BiRefNet", "model.safetensors", requiresConversion: false))
    }

    static let u2NetLicense = NFKModelLicense.license("Apache-2.0", name: "Apache License 2.0",
                                                      url: "https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE")

    static var u2Net: NFKMLXCatalogEntry {
        matteEntry(NFKMLXU2Net.modelName, "U²-Net (universal)", family: "u2net",
                   summary: "A background remover that separates the salient subject.",
                   license: .apache2(note: "Carve's universal fine-tune of U²-Net."),
                   source: hubFile("Carve/u2net-universal", "full_weights.pth", requiresConversion: true))
    }

    static var u2NetP: NFKMLXCatalogEntry {
        matteEntry(NFKMLXU2Net.lightModelName, "U²-NetP", family: "u2net",
                   summary: "The small background remover that separates the salient subject.",
                   license: u2NetLicense,
                   source: releasePage("https://github.com/xuebinqin/U-2-Net", requiresConversion: true))
    }

    static var isNet: NFKMLXCatalogEntry {
        matteEntry(NFKMLXISNet.modelName, "IS-Net (DIS general use)", family: "isnet",
                   summary: "A dichotomous segmenter that cuts out the foreground object with fine detail.",
                   license: .license("Apache-2.0", name: "Apache License 2.0",
                                     url: "https://github.com/xuebinqin/DIS/blob/main/LICENSE.md",
                                     note: "Served from a Hugging Face mirror of the authors' release."),
                   source: hubFile("NimaBoscarino/IS-Net_DIS-general-use", "isnet-general-use.pth",
                                   requiresConversion: false))
    }

    static var modNet: NFKMLXCatalogEntry {
        matteEntry(NFKMLXMODNet.modelName, "MODNet (photographic portrait)", family: "modnet",
                   summary: "A portrait matter that separates a person from the background without a trimap.",
                   license: .license("Apache-2.0", name: "Apache License 2.0",
                                     url: "https://github.com/ZHKKKe/MODNet/blob/master/LICENSE"),
                   source: releasePage("https://github.com/ZHKKKe/MODNet", requiresConversion: true))
    }

    /// The Robust Video Matting releases, in the order `NFKMLXRVM.register()` registers them.
    static let rvmReleases: [NFKMLXRVMVariant] = [.mobileNetV3, .resNet50]

    static let rvmLicense = NFKModelLicense.license("GPL-3.0-only", name: "GNU General Public License v3.0",
                                                    url: "https://github.com/PeterL1n/RobustVideoMatting/blob/master/LICENSE",
                                                    note: "Software that includes the weights is distributed under the GPL.")

    static func rvmRelease(_ variant: NFKMLXRVMVariant) -> (encoder: String, address: String) {
        switch variant {
        case .mobileNetV3:
            return ("MobileNetV3", "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth")
        case .resNet50:
            return ("ResNet-50", "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth")
        }
    }

    static var rvmEntries: [NFKMLXCatalogEntry] {
        rvmReleases.map { variant in
            let release = rvmRelease(variant)
            return matteEntry(NFKMLXRVM.specs(for: variant).name, "Robust Video Matting \(release.encoder) (one frame)",
                              family: "robust-video-matting",
                              summary: "A human matter that mattes one frame without the clip's recurrent state.",
                              license: rvmLicense, source: download(release.address, requiresConversion: true))
        }
    }

    // MARK: - Prompted segmentation and faces

    static var promptSegmenters: [NFKMLXCatalogEntry] {
        samEntries + sam2Entries + sam3Entries + [retinaFace]
    }

    static let samPointOption = NFKModelOption.option(
        NFKSAMPointKey, kind: NFKModelOptionKindJSON, title: "Point",
        help: "The click on the object, an array [x, y] in pixels of the plate. The plate's center when absent.")

    static func samEntry(_ name: String, _ displayName: String, family: String, summary: String,
                         license: NFKModelLicense, source: NFKModelSource) -> NFKMLXCatalogEntry {
        model(name, displayName, family: family, summary: summary, tasks: [NFKModelTaskSegmentation],
              inputs: [imageInput("The plate holding the object.", title: "Plate"), unreadHint],
              outputs: [.output(NFKOutputImage, required: true, title: "Cutout",
                                help: "The plate with the object's mask as its alpha."),
                        .output(NFKOutputMask, required: true, title: "Mask", help: "The object's mask as a gray image.")],
              parameters: [samPointOption], license: license, source: source)
    }

    /// The released SAM sizes, each under its own registry name.
    static let samReleases: [NFKMLXSAMVariant] = [.vitB, .vitL, .vitH]

    static func samName(_ variant: NFKMLXSAMVariant) -> String {
        samRelease(variant).name
    }

    static func samRelease(_ variant: NFKMLXSAMVariant) -> (name: String, size: String, file: String) {
        switch variant {
        case .compact: return (NFKMLXSAM.modelName, "compact", "")
        case .vitB: return ("sam-vit-b", "ViT-B", "sam_vit_b_01ec64.pth")
        case .vitL: return ("sam-vit-l", "ViT-L", "sam_vit_l_0b3195.pth")
        case .vitH: return ("sam-vit-h", "ViT-H", "sam_vit_h_4b8939.pth")
        }
    }

    static var samEntries: [NFKMLXCatalogEntry] {
        var compact = samEntry(NFKMLXSAM.modelName, "Segment Anything (test geometry)", family: "sam",
                               summary: "Segment Anything at its compact default geometry, which no released checkpoint fits.",
                               license: .inferKit(note: "Built with random weights; no released checkpoint fits this geometry."),
                               source: .noDownload)
        compact.standIn = true
        return [compact] + samReleases.map { variant in
            let release = samRelease(variant)
            return samEntry(release.name, "Segment Anything \(release.size)", family: "sam",
                            summary: "A promptable segmenter that cuts out the object under a click.",
                            license: apacheSAM,
                            source: download("https://dl.fbaipublicfiles.com/segment_anything/\(release.file)",
                                             requiresConversion: true))
        }
    }

    /// A SAM 2 size and release.
    struct SAM2Release {
        let variant: NFKMLXSAM2Variant
        let release: NFKMLXSAM2Release
    }

    /// The SAM 2 size and release `sam2` and `sam2-video` build: SAM 2.1 Hiera Tiny.
    static let sam2DefaultRelease = SAM2Release(variant: .tiny, release: .sam21)

    /// The SAM 2 sizes and releases other than the default, each under its own registry names.
    static let sam2AdditionalReleases: [SAM2Release] = [
        SAM2Release(variant: .tiny, release: .sam2), SAM2Release(variant: .small, release: .sam2),
        SAM2Release(variant: .basePlus, release: .sam2), SAM2Release(variant: .large, release: .sam2),
        SAM2Release(variant: .small, release: .sam21), SAM2Release(variant: .basePlus, release: .sam21),
        SAM2Release(variant: .large, release: .sam21),
    ]

    /// The stem SAM 2's publisher names a size and release by, such as `sam2.1-hiera-base-plus`.
    static func sam2Stem(_ release: SAM2Release) -> String {
        let size: String
        switch release.variant {
        case .tiny: size = "tiny"
        case .small: size = "small"
        case .basePlus: size = "base-plus"
        case .large: size = "large"
        }
        return (release.release == .sam21 ? "sam2.1" : "sam2") + "-hiera-" + size
    }

    /// The image backend's registry name for a SAM 2 size and release.
    static func sam2Name(_ release: SAM2Release) -> String {
        sam2Stem(release)
    }

    /// The display name of a SAM 2 size and release.
    static func sam2DisplayName(_ release: SAM2Release) -> String {
        let size: String
        switch release.variant {
        case .tiny: size = "Tiny"
        case .small: size = "Small"
        case .basePlus: size = "Base+"
        case .large: size = "Large"
        }
        return (release.release == .sam21 ? "SAM 2.1" : "SAM 2") + " Hiera " + size
    }

    /// The publisher's checkpoint of a SAM 2 size and release, as its Hugging Face repository serves it.
    static func sam2Source(_ release: SAM2Release) -> NFKModelSource {
        let stem = sam2Stem(release)
        return hubFile("facebook/\(stem)", stem.replacingOccurrences(of: "-", with: "_") + ".pt",
                       requiresConversion: true)
    }

    static var sam2Entries: [NFKMLXCatalogEntry] {
        ([sam2DefaultRelease] + sam2AdditionalReleases).map { release in
            let name = release.variant == sam2DefaultRelease.variant && release.release == sam2DefaultRelease.release
                ? NFKMLXSAM2.modelName : sam2Name(release)
            return samEntry(name, sam2DisplayName(release), family: "sam2",
                            summary: "A promptable segmenter that cuts out the object under a click.",
                            license: apacheSAM2, source: sam2Source(release))
        }
    }

    /// The registry name of SAM 3.1.
    static let sam31Name = "sam3.1"

    static var sam3Entries: [NFKMLXCatalogEntry] {
        let releases: [(name: String, displayName: String, repository: String)] = [
            (NFKMLXSAM3.modelName, "SAM 3", NFKMLXSAM3.releaseRepo),
            (sam31Name, "SAM 3.1", "facebook/sam3.1"),
        ]
        return releases.map { release in
            model(release.name, release.displayName, family: "sam3",
                  summary: "A concept segmenter that finds every instance a short phrase names.",
                  tasks: [NFKModelTaskSegmentation, NFKModelTaskObjectDetection],
                  inputs: [imageInput("The image to search."),
                           .input(NFKInputPrompt, required: true, title: "Concept",
                                  help: "A short noun phrase naming what to find.")],
                  outputs: [.output(NFKOutputDetections, required: true, title: "Instances",
                                    help: "Each instance above the score threshold, most confident first, boxed in 0...1 of the image."),
                            .output(NFKOutputMask, title: "Mask",
                                    help: "The union of the instances' masks at the image's size. Absent when no instance passes.")],
                  parameters: [.option(NFKMLXSAM3ParameterKey.scoreThreshold, kind: NFKModelOptionKindNumber,
                                       title: "Score threshold", help: "The score an instance must exceed to be returned.",
                                       defaultValue: 0.5, minimum: 0, maximum: 1),
                               .option(NFKMLXSAM3ParameterKey.maskThreshold, kind: NFKModelOptionKindNumber,
                                       title: "Mask threshold",
                                       help: "The probability a pixel must exceed to belong to an instance's mask.",
                                       defaultValue: 0.5, minimum: 0, maximum: 1)],
                  license: .license(nil, name: "SAM License", url: "https://huggingface.co/\(release.repository)",
                                    gated: true, note: "Gated: accept the terms on Hugging Face."),
                  source: release.name == sam31Name ? sam31Source : hubRelease(release.repository,
                                                                                components: NFKMLXSAM3.releaseComponents))
        }
    }

    /// The SAM 3.1 checkpoint `facebook/sam3.1` serves: `sam3.1_multiplex.pt`, saved by the `facebook/sam3`
    /// codebase with its Object Multiplex tracker. SAM 3's loader reads the transformers layout of
    /// `facebook/sam3`'s `model.safetensors`, whose names and video tracker differ, so the file is converted
    /// before it loads.
    static var sam31Source: NFKModelSource {
        let component = NFKMLXSAM3.releaseComponents[0]
        return NFKModelSource.hub("facebook/sam3.1", layout: NFKModelSourceLayoutDirectory,
                                  files: component.required.map { file($0) } + component.optional.map { file($0, optional: true) }
                                      + [file(sam31Checkpoint)],
                                  requiresConversion: true)
            .noting("The repository serves \(sam31Checkpoint) in the facebook/sam3 codebase's layout, with the Object Multiplex tracker. The registry builds SAM 3.1 from a directory whose model.safetensors is in the transformers layout facebook/sam3 serves; no converter ships yet.")
    }

    /// The checkpoint `facebook/sam3.1` serves.
    static let sam31Checkpoint = "sam3.1_multiplex.pt"

    static var retinaFace: NFKMLXCatalogEntry {
        model(NFKMLXRetinaFace.modelName, "RetinaFace MobileNet-0.25", family: "retinaface",
              summary: "A face detector that boxes every face in an image.",
              tasks: [NFKModelTaskFaceDetection],
              inputs: [imageInput("The image to search for faces.")],
              outputs: [.output(NFKOutputDetections, required: true, title: "Faces", help: "A box around each face found.")],
              license: .mit(url: "https://github.com/xinntao/facexlib/blob/master/LICENSE"),
              source: download("https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth",
                               requiresConversion: true))
    }

    // MARK: - Object detection

    static var detectors: [NFKMLXCatalogEntry] {
        yoloV8Entries + yoloGenerationEntries + rtDetrEntries + rfDetrEntries + rfDetrSegmentationEntries
            + tableTransformerEntries
    }

    static func detectionEntry(_ name: String, _ displayName: String, family: String, summary: String,
                               license: NFKModelLicense, source: NFKModelSource) -> NFKMLXCatalogEntry {
        model(name, displayName, family: family, summary: summary, tasks: [NFKModelTaskObjectDetection],
              inputs: [imageInput("The image to search.")],
              outputs: [.output(NFKOutputDetections, required: true, title: "Objects",
                                help: "A box, a class, and a score for each object found.")],
              license: license, source: source)
    }

    /// The YOLOv8 sizes other than the nano `yolo` builds, each under its own registry name.
    static let yoloV8AdditionalReleases: [NFKMLXYOLOVariant] = [.small, .medium, .large, .extraLarge]

    static func yoloV8Name(_ variant: NFKMLXYOLOVariant) -> String {
        switch variant {
        case .nano: return NFKMLXYOLO.modelName
        case .small: return "yolov8s"
        case .medium: return "yolov8m"
        case .large: return "yolov8l"
        case .extraLarge: return "yolov8x"
        }
    }

    static var yoloV8Entries: [NFKMLXCatalogEntry] {
        ([NFKMLXYOLOVariant.nano] + yoloV8AdditionalReleases).map { variant in
            let stem: String
            switch variant {
            case .nano: stem = "yolov8n"
            default: stem = yoloV8Name(variant)
            }
            return detectionEntry(yoloV8Name(variant), "YOLO" + stem.dropFirst(4), family: "yolo",
                                  summary: "A real-time detector of the 80 COCO object classes.",
                                  license: ultralytics,
                                  source: download("https://github.com/ultralytics/assets/releases/download/v8.3.0/\(stem).pt",
                                                   requiresConversion: true))
        }
    }

    static var yoloGenerationEntries: [NFKMLXCatalogEntry] {
        (0 ... NFKMLXYOLORelease.v26ExtraLarge.rawValue).compactMap { raw in
            guard let release = NFKMLXYOLORelease(rawValue: raw) else { return nil }
            let stem = NFKMLXYOLOGenerations.modelName(for: release)
            let tag = NFKMLXYOLOGenerations.specs(for: release).generation == .v26 ? "v8.4.0" : "v8.3.0"
            return detectionEntry(stem, "YOLO" + stem.dropFirst(4), family: "yolo",
                                  summary: "A real-time detector of the 80 COCO object classes.",
                                  license: ultralytics,
                                  source: download("https://github.com/ultralytics/assets/releases/download/\(tag)/\(stem).pt",
                                                   requiresConversion: false))
        }
    }

    /// The RT-DETR releases, in the order `NFKMLXRTDetr.register()` registers them.
    static let rtDetrReleases: [NFKMLXRTDetrVariant] = [.r50vd, .r18vd, .r34vd, .r101vd,
                                                       .v2R18VD, .v2R34VD, .v2R50VD, .v2R101VD]

    static var rtDetrEntries: [NFKMLXCatalogEntry] {
        rtDetrReleases.map { variant in
            let release: (repository: String, displayName: String)
            switch variant {
            case .r50vd: release = ("PekingU/rtdetr_r50vd", "RT-DETR R50vd")
            case .r18vd: release = ("PekingU/rtdetr_r18vd", "RT-DETR R18vd")
            case .r34vd: release = ("PekingU/rtdetr_r34vd", "RT-DETR R34vd")
            case .r101vd: release = ("PekingU/rtdetr_r101vd", "RT-DETR R101vd")
            case .v2R18VD: release = ("PekingU/rtdetr_v2_r18vd", "RT-DETRv2 R18vd")
            case .v2R34VD: release = ("PekingU/rtdetr_v2_r34vd", "RT-DETRv2 R34vd")
            case .v2R50VD: release = ("PekingU/rtdetr_v2_r50vd", "RT-DETRv2 R50vd")
            case .v2R101VD: release = ("PekingU/rtdetr_v2_r101vd", "RT-DETRv2 R101vd")
            }
            return detectionEntry(NFKMLXRTDetr.specs(for: variant).name, release.displayName, family: "rt-detr",
                                  summary: "A real-time transformer detector of the 80 COCO object classes.",
                                  license: .license("Apache-2.0", name: "Apache License 2.0",
                                                    url: "https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE"),
                                  source: hubFile(release.repository, "model.safetensors", requiresConversion: false))
        }
    }

    static let rfDetrLicense = NFKModelLicense.license("Apache-2.0", name: "Apache License 2.0",
                                                       url: "https://github.com/roboflow/rf-detr/blob/develop/LICENSE")

    /// The RF-DETR releases, in the order `NFKMLXRFDetr.register()` registers them.
    static let rfDetrReleases: [NFKMLXRFDetrVariant] = [.base, .nano, .small, .medium, .large]

    static var rfDetrEntries: [NFKMLXCatalogEntry] {
        rfDetrReleases.map { variant in
            let size: String
            switch variant {
            case .base: size = "base"
            case .nano: size = "nano"
            case .small: size = "small"
            case .medium: size = "medium"
            case .large: size = "large"
            }
            return detectionEntry(NFKMLXRFDetr.specs(for: variant).name, "RF-DETR \(size.capitalized)", family: "rf-detr",
                                  summary: "A real-time transformer detector of the 80 COCO object classes.",
                                  license: rfDetrLicense,
                                  source: hubFile("Roboflow/rf-detr-\(size)", "model.safetensors", requiresConversion: false))
        }
    }

    /// The RF-DETR segmentation releases, in the order `NFKMLXRFDetrSegmentation.register()` registers them.
    static let rfDetrSegmentationReleases: [NFKMLXRFDetrSegmentationVariant] = [
        .nano, .small, .preview, .medium, .large, .extraLarge, .extraExtraLarge,
    ]

    static var rfDetrSegmentationEntries: [NFKMLXCatalogEntry] {
        rfDetrSegmentationReleases.map { variant in
            let size: (stem: String, displayName: String)
            switch variant {
            case .nano: size = ("nano", "Nano")
            case .small: size = ("small", "Small")
            case .preview: size = ("preview", "Preview")
            case .medium: size = ("medium", "Medium")
            case .large: size = ("large", "Large")
            case .extraLarge: size = ("xlarge", "XLarge")
            case .extraExtraLarge: size = ("xxlarge", "2XLarge")
            }
            return model(NFKMLXRFDetrSegmentation.specs(for: variant).name, "RF-DETR Seg \(size.displayName)",
                         family: "rf-detr",
                         summary: "A real-time instance segmenter of the 80 COCO object classes.",
                         tasks: [NFKModelTaskSegmentation, NFKModelTaskObjectDetection],
                         inputs: [imageInput("The image to search.")],
                         outputs: [.output(NFKOutputDetections, required: true, title: "Objects",
                                           help: "A box, a class, a score, and a mask for each object found."),
                                   .output(NFKOutputMask, title: "Mask",
                                           help: "The union of the objects' masks. Absent when nothing is found.")],
                         license: rfDetrLicense,
                         source: hubFile("Roboflow/rf-detr-seg-\(size.stem)", "model.safetensors",
                                         requiresConversion: false))
        }
    }

    /// A release directory and the registry name it builds under.
    struct DirectoryRelease {
        let name: String
        let repository: String
        let displayName: String
    }

    /// The Table Transformer releases.
    static let tableTransformerReleases: [DirectoryRelease] = [
        DirectoryRelease(name: "table-transformer-detection", repository: "microsoft/table-transformer-detection",
                         displayName: "Table Transformer (detection)"),
        DirectoryRelease(name: NFKMLXTableTransformer.modelName,
                         repository: "microsoft/table-transformer-structure-recognition",
                         displayName: "Table Transformer (structure recognition)"),
        DirectoryRelease(name: "table-transformer-structure-recognition-v1.1-all",
                         repository: "microsoft/table-transformer-structure-recognition-v1.1-all",
                         displayName: "Table Transformer v1.1 (structure recognition, all)"),
        DirectoryRelease(name: "table-transformer-structure-recognition-v1.1-fin",
                         repository: "microsoft/table-transformer-structure-recognition-v1.1-fin",
                         displayName: "Table Transformer v1.1 (structure recognition, FinTabNet)"),
        DirectoryRelease(name: "table-transformer-structure-recognition-v1.1-pub",
                         repository: "microsoft/table-transformer-structure-recognition-v1.1-pub",
                         displayName: "Table Transformer v1.1 (structure recognition, PubTables-1M)"),
    ]

    static var tableTransformerEntries: [NFKMLXCatalogEntry] {
        tableTransformerReleases.map { release in
            let summary = release.name == "table-transformer-detection"
                ? "A detector of the tables on a document page."
                : "A detector of a table's rows, columns, headers, and spanning cells."
            return model(release.name, release.displayName, family: "table-transformer", summary: summary,
                         tasks: [NFKModelTaskObjectDetection],
                         inputs: [imageInput("The document page or table crop to read.")],
                         outputs: [.output(NFKOutputDetections, required: true, title: "Regions",
                                           help: "A box, a label from the release's own classes, and a score for each region.")],
                         license: .mit(url: "https://huggingface.co/\(release.repository)"),
                         source: hubRelease(release.repository, required: NFKMLXTableTransformer.requiredFiles,
                                            optional: NFKMLXTableTransformer.optionalFiles,
                                            weights: NFKMLXTableTransformer.weightFiles))
        }
    }

    // MARK: - Pose

    static var poseEstimators: [NFKMLXCatalogEntry] {
        [simpleBaselinePose] + vitPoseEntries
    }

    static func poseEntry(_ name: String, _ displayName: String, family: String, summary: String,
                          license: NFKModelLicense, source: NFKModelSource) -> NFKMLXCatalogEntry {
        model(name, displayName, family: family, summary: summary, tasks: [NFKModelTaskPoseEstimation],
              inputs: [imageInput("A crop around one person.")],
              outputs: [.output(NFKOutputPose, required: true, title: "Joints",
                                help: "The 17 COCO body joints, each with its position and confidence.")],
              license: license, source: source)
    }

    static var simpleBaselinePose: NFKMLXCatalogEntry {
        poseEntry(NFKMLXPose.modelName, "SimpleBaseline ResNet-50", family: "simplebaseline",
                  summary: "A top-down estimator of a person's 17 body joints.",
                  license: .license("Apache-2.0", name: "Apache License 2.0",
                                    url: "https://github.com/open-mmlab/mmpose/blob/main/LICENSE"),
                  source: download("https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192-04af38ce_20220923.pth",
                                   requiresConversion: true))
    }

    /// The ViTPose releases, in the order `NFKMLXVitPose.register()` registers them.
    static let vitPoseReleases: [NFKMLXVitPoseVariant] = [.baseSimple, .base]

    static var vitPoseEntries: [NFKMLXCatalogEntry] {
        vitPoseReleases.map { variant in
            let name = NFKMLXVitPose.specs(for: variant).name
            let displayName = variant == .baseSimple ? "ViTPose-B (simple decoder)" : "ViTPose-B"
            return poseEntry(name, displayName, family: "vitpose",
                             summary: "A top-down transformer estimator of a person's 17 body joints.",
                             license: .license("Apache-2.0", name: "Apache License 2.0",
                                               url: "https://huggingface.co/usyd-community/\(name)"),
                             source: hubFile("usyd-community/\(name)", "model.safetensors", requiresConversion: false))
        }
    }

    // MARK: - Embedding

    static var embedders: [NFKMLXCatalogEntry] {
        clipEntries + [clipProbe] + sigLIP2Entries + [sigLIP2Probe]
    }

    /// The CLIP releases, in the order `NFKMLXCLIP.register()` registers them.
    static let clipReleases: [NFKMLXCLIPVariant] = [.vitB32, .vitB16, .vitL14, .vitL14At336]

    static var clipEntries: [NFKMLXCatalogEntry] {
        clipReleases.map { variant in
            let release: (displayName: String, address: String)
            switch variant {
            case .vitB32:
                release = ("CLIP ViT-B/32", "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")
            case .vitB16:
                release = ("CLIP ViT-B/16", "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt")
            case .vitL14:
                release = ("CLIP ViT-L/14", "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt")
            case .vitL14At336:
                release = ("CLIP ViT-L/14@336px", "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt")
            }
            return model(NFKMLXCLIP.specs(for: variant).name, release.displayName, family: "clip",
                         summary: "An image encoder whose embeddings compare images with each other and with text.",
                         tasks: [NFKModelTaskImageEmbedding],
                         inputs: [.input(NFKInputImage, title: "Image", help: "The image to embed."),
                                  .input(NFKInputPrompt, title: "Text",
                                         help: "Text to embed. The registered backend has no tokenizer and embeds images only.")],
                         outputs: [.output(NFKOutputEmbedding, required: true, title: "Embedding",
                                           help: "The normalized embedding in CLIP's joint image-text space.")],
                         license: openAICLIP, source: download(release.address, requiresConversion: true))
        }
    }

    /// The ports of a linear probe over an image encoder.
    static func probePorts() -> (inputs: [NFKModelPort], outputs: [NFKModelPort]) {
        ([imageInput("The image to classify.")],
         [.output(NFKOutputClassifications, required: true, title: "Classes",
                  help: "The probe's classes ranked by probability, named from labels.json when the directory holds it.")])
    }

    static var clipProbe: NFKMLXCatalogEntry {
        let ports = probePorts()
        return model(NFKMLXCLIP.probeModelName, "CLIP ViT-B/32 linear probe", family: "clip",
                     summary: "An image classifier made of a probe the app trained over CLIP ViT-B/32 embeddings.",
                     tasks: [NFKModelTaskClassification], inputs: ports.inputs, outputs: ports.outputs,
                     license: .mit(url: "https://github.com/openai/CLIP/blob/main/LICENSE",
                                   note: "The license of the CLIP weights; the probe is the app's own."),
                     source: appSupplied)
    }

    static var sigLIP2Entries: [NFKMLXCatalogEntry] {
        NFKMLXSigLIP2.allVariants.map { variant in
            let name = NFKMLXSigLIP2.specs(for: variant).name
            return model(name, "SigLIP 2 " + name.dropFirst("siglip2-".count), family: "siglip2",
                         summary: "An image encoder whose embeddings compare images with each other and with text.",
                         tasks: [NFKModelTaskImageEmbedding],
                         inputs: [imageInput("The image to embed.")],
                         outputs: [.output(NFKOutputEmbedding, required: true, title: "Embedding",
                                           help: "The normalized image embedding in SigLIP 2's joint image-text space.")],
                         license: .license("Apache-2.0", name: "Apache License 2.0", url: "https://huggingface.co/google/\(name)"),
                         source: hubFile("google/\(name)", shardedSigLIP2Releases.contains(variant)
                                            ? "model.safetensors.index.json" : "model.safetensors",
                                         requiresConversion: true))
        }
    }

    /// The SigLIP 2 releases whose repositories shard their weights behind an index.
    static let shardedSigLIP2Releases: Set<NFKMLXSigLIP2Variant> = [.giantOptPatch16At256, .giantOptPatch16At384]

    /// The registry name of a SigLIP 2 probe directory.
    static let sigLIP2ProbeName = "siglip2-probe"

    /// A probe directory: `model.safetensors` (the SigLIP 2 base-patch16-224 checkpoint),
    /// `probe.safetensors`, and, where present, `labels.json` (an array of class names).
    static func sigLIP2ProbeBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        let labelsURL = directoryURL.appendingPathComponent("labels.json")
        var labels: [String]?
        if FileManager.default.fileExists(atPath: labelsURL.path) {
            guard let names = try JSONSerialization.jsonObject(with: Data(contentsOf: labelsURL)) as? [String] else {
                throw NFKMLXError.unsupportedConfiguration("labels.json is not an array of class names")
            }
            labels = names
        }
        let encoder = try NFKMLXSigLIP2.model(weightsURL: directoryURL.appendingPathComponent("model.safetensors"))
        return try encoder.probeBackend(probeURL: directoryURL.appendingPathComponent("probe.safetensors"), labels: labels)
    }

    static var sigLIP2Probe: NFKMLXCatalogEntry {
        let ports = probePorts()
        return model(sigLIP2ProbeName, "SigLIP 2 base-patch16-224 linear probe", family: "siglip2",
                     summary: "An image classifier made of a probe the app trained over SigLIP 2 embeddings.",
                     tasks: [NFKModelTaskClassification], inputs: ports.inputs, outputs: ports.outputs,
                     license: .license("Apache-2.0", name: "Apache License 2.0",
                                       url: "https://huggingface.co/google/siglip2-base-patch16-224",
                                       note: "The license of the SigLIP 2 weights; the probe is the app's own."),
                     source: appSupplied)
    }

    // MARK: - Reading and vision-language

    static var readers: [NFKMLXCatalogEntry] {
        florence2Entries + trOCREntries + sa2VAEntries
    }

    /// The Florence-2 releases.
    static let florence2Releases: [DirectoryRelease] = [
        DirectoryRelease(name: "florence-2-base", repository: "microsoft/Florence-2-base", displayName: "Florence-2 base"),
        DirectoryRelease(name: NFKMLXFlorence2.modelName, repository: "microsoft/Florence-2-large",
                         displayName: "Florence-2 large"),
        DirectoryRelease(name: "florence-2-base-ft", repository: "microsoft/Florence-2-base-ft",
                         displayName: "Florence-2 base (fine-tuned)"),
        DirectoryRelease(name: "florence-2-large-ft", repository: "microsoft/Florence-2-large-ft",
                         displayName: "Florence-2 large (fine-tuned)"),
    ]

    static var florence2Entries: [NFKMLXCatalogEntry] {
        florence2Releases.map { release in
            model(release.name, release.displayName, family: "florence-2",
                  summary: "A unified vision model that captions, reads text, and locates objects, chosen by a task token.",
                  tasks: [NFKModelTaskVisionQuestionAnswering, NFKModelTaskTextRecognition, NFKModelTaskObjectDetection],
                  inputs: [imageInput("The image to read."),
                           .input(NFKInputPrompt, title: "Task",
                                  help: "A task token such as <CAPTION>, <OCR>, or <OD>, with its text where the task takes one. <CAPTION> when absent.")],
                  outputs: [.output(NFKOutputText, required: true, title: "Text", help: "The task's answer as text."),
                            .output(NFKOutputDetections, title: "Regions",
                                    help: "The boxes a localization task answers with.")],
                  parameters: [.option(NFKParameterMaxTokens, title: "Maximum tokens",
                                       help: "The most tokens generated after the start token.",
                                       defaultValue: NFKMLXFlorence2.maximumTokens, minimum: 1, step: 1),
                               .option(NFKMLXTranslationParameterKey.beamCount, kind: NFKModelOptionKindInteger,
                                       title: "Beams", help: "The beams searched. 1 decodes greedily; absent, the release's own setting.",
                                       minimum: 1, step: 1)],
                  license: .mit(url: "https://huggingface.co/\(release.repository)/resolve/main/LICENSE"),
                  source: hubRelease(release.repository, required: NFKMLXFlorence2.requiredFiles,
                                     optional: NFKMLXFlorence2.optionalFiles, weights: NFKMLXFlorence2.weightFiles))
        }
    }

    /// The TrOCR releases.
    static let trOCRReleases: [DirectoryRelease] = ["small", "base", "large"].flatMap { size in
        ["handwritten", "printed", "str", "stage1"].compactMap { kind -> DirectoryRelease? in
            if size == "small" && kind == "str" {
                return nil
            }
            let name = "trocr-\(size)-\(kind)"
            let reading: String
            switch kind {
            case "handwritten": reading = "handwritten"
            case "printed": reading = "printed"
            case "str": reading = "scene text"
            default: reading = "pretrained, not fine-tuned"
            }
            return DirectoryRelease(name: name, repository: "microsoft/\(name)",
                                    displayName: "TrOCR \(size) (\(reading))")
        }
    }

    static var trOCREntries: [NFKMLXCatalogEntry] {
        trOCRReleases.map { release in
            model(release.name, release.displayName, family: "trocr",
                  summary: "A reader that transcribes one line of text from an image.",
                  tasks: [NFKModelTaskTextRecognition],
                  inputs: [imageInput("A crop holding one line of text.")],
                  outputs: [.output(NFKOutputText, required: true, title: "Text", help: "The transcribed line.")],
                  license: .mit(url: "https://huggingface.co/\(release.repository)"),
                  source: hubRelease(release.repository, required: NFKMLXTrOCR.requiredFiles,
                                     optional: NFKMLXTrOCR.optionalFiles, weights: NFKMLXTrOCR.weightFiles))
        }
    }

    /// The Sa2VA releases.
    static let sa2VAReleases: [DirectoryRelease] = [
        DirectoryRelease(name: "sa2va-1b", repository: "ByteDance/Sa2VA-1B", displayName: "Sa2VA 1B"),
        DirectoryRelease(name: NFKMLXSa2VA.modelName, repository: "ByteDance/Sa2VA-4B", displayName: "Sa2VA 4B"),
        DirectoryRelease(name: "sa2va-internvl3-2b", repository: "ByteDance/Sa2VA-InternVL3-2B",
                         displayName: "Sa2VA InternVL3 2B"),
        DirectoryRelease(name: "sa2va-qwen2.5-vl-3b", repository: "ByteDance/Sa2VA-Qwen2_5-VL-3B",
                         displayName: "Sa2VA Qwen2.5-VL 3B"),
        DirectoryRelease(name: "sa2va-qwen3-vl-2b", repository: "ByteDance/Sa2VA-Qwen3-VL-2B",
                         displayName: "Sa2VA Qwen3-VL 2B"),
        DirectoryRelease(name: "sa2va-qwen3-vl-4b", repository: "ByteDance/Sa2VA-Qwen3-VL-4B",
                         displayName: "Sa2VA Qwen3-VL 4B"),
        DirectoryRelease(name: "sa2va-qwen3-vl-4b-sam3", repository: "ByteDance/Sa2VA-Qwen3-VL-4B-SAM3",
                         displayName: "Sa2VA Qwen3-VL 4B (SAM 3 grounding)"),
        DirectoryRelease(name: "sa2va-llava-1.5-7b", repository: "ByteDance/Sa2VA-LLaVA-1.5-7B",
                         displayName: "Sa2VA LLaVA-1.5 7B"),
    ]

    /// The license that governs a Sa2VA release's weights. Every repository states Apache-2.0, and four
    /// carry a component whose own repository states a different license; those carry that license, with
    /// the repository's statement in the note.
    static func sa2VALicense(_ release: DirectoryRelease) -> NFKModelLicense {
        let stated = "The repository states Apache-2.0."
        switch release.repository {
        case "ByteDance/Sa2VA-4B":
            return .license(nil, name: "Qwen Research License",
                            url: "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE",
                            restrictsCommercialUse: true,
                            note: "\(stated) Its language model, Qwen2.5-3B-Instruct, is under the Qwen Research License: non-commercial use, and a commercial use requests a license from Alibaba Cloud.")
        case "ByteDance/Sa2VA-Qwen2_5-VL-3B":
            return .license(nil, name: "Qwen Research License",
                            url: "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE",
                            restrictsCommercialUse: true,
                            note: "\(stated) Its vision-language model, Qwen2.5-VL-3B-Instruct, is under the Qwen Research License: non-commercial use, and a commercial use requests a license from Alibaba Cloud.")
        case "ByteDance/Sa2VA-Qwen3-VL-4B-SAM3":
            return .license(nil, name: "SAM License", url: "https://huggingface.co/facebook/sam3/blob/main/LICENSE",
                            note: "\(stated) Its grounding model, SAM 3, is under the SAM License, which a redistribution or a derivative keeps.")
        case "ByteDance/Sa2VA-LLaVA-1.5-7B":
            return .license(nil, name: "Llama 2 Community License", url: "https://ai.meta.com/llama/license/",
                            restrictsCommercialUse: true,
                            note: "\(stated) Its language model, Vicuna-7B v1.5, is fine-tuned from Llama 2 and is under the Llama 2 Community License: a product with more than 700 million monthly active users requests a license from Meta, and use follows Meta's Acceptable Use Policy.")
        default:
            return .license("Apache-2.0", name: "Apache License 2.0", url: "https://huggingface.co/\(release.repository)")
        }
    }

    static var sa2VAEntries: [NFKMLXCatalogEntry] {
        sa2VAReleases.map { release in
            model(release.name, release.displayName, family: "sa2va",
                  summary: "A vision-language model that answers about an image and segments what its answer refers to.",
                  tasks: [NFKModelTaskVisionQuestionAnswering, NFKModelTaskSegmentation],
                  inputs: [imageInput("The image to discuss."),
                           .input(NFKInputPrompt, required: true, title: "Prompt",
                                  help: "The question or the referring expression, such as \"Please segment the person.\"")],
                  outputs: [.output(NFKOutputText, required: true, title: "Answer", help: "The model's answer."),
                            .output(NFKOutputMask, title: "Mask",
                                    help: "The mask of what the answer's [SEG] token refers to. Absent when the answer has none.")],
                  license: sa2VALicense(release),
                  source: hubRelease(release.repository, required: NFKMLXSa2VA.requiredFiles,
                                     optional: NFKMLXSa2VA.optionalFiles, weights: NFKMLXSa2VA.weightFiles))
        }
    }

    // MARK: - Image generation

    static var generators: [NFKMLXCatalogEntry] {
        stableDiffusionEntries + sd3Entries + fluxEntries + flux2Entries + [fluxControlNet, sd3ControlNet]
            + sanaEntries + zImageEntries + [qwenImage]
    }

    /// The generation residencies: the stages load together, or each loads for its turn.
    static let stagedResidencies = [NFKModelResidencyResident, NFKModelResidencyStaged]

    static var promptInput: NFKModelPort {
        .input(NFKInputPrompt, required: true, title: "Prompt", help: "What to draw.")
    }

    static var negativePromptInput: NFKModelPort {
        .input(NFKInputNegativePrompt, title: "Negative prompt", help: "What to steer away from.")
    }

    static var generatedImageOutput: NFKModelPort {
        imageOutput("The generated image.")
    }

    /// The Stable Diffusion text-to-image releases, each under its own registry name.
    static let stableDiffusionReleases: [NFKMLXStableDiffusionModel] = [.stableDiffusion15, .stableDiffusion21Base,
                                                                        .sdxlTurbo, .sdxlBase, .lcmDreamshaperV7]

    /// A release's registry name: its repository's name.
    static func stableDiffusionName(_ model: NFKMLXStableDiffusionModel) -> String {
        switch model {
        case .stableDiffusion15: return "stable-diffusion-v1-5"
        case .stableDiffusion21Base: return "stable-diffusion-2-1-base"
        case .sdxlTurbo: return "sdxl-turbo"
        case .sdxlBase: return "stable-diffusion-xl-base-1.0"
        case .lcmDreamshaperV7: return "lcm-dreamshaper-v7"
        }
    }

    /// A released Stable Diffusion LoRA merged into the base release it was trained on, under its own
    /// registry name. The registry directory is the base release's; the adapter file comes from its own
    /// repository.
    struct StableDiffusionAdapter {
        let name: String
        let displayName: String
        let summary: String
        let base: NFKMLXStableDiffusionModel
        let repository: String
        let weightsPath: String
        let sampler: NFKMLXSDSampler
        /// The license of the merged weights: the base release's, noting the adapter's.
        let license: NFKModelLicense

        /// The base release's configuration on the adapter's sampler, at the steps and guidance the
        /// adapter is trained for.
        var configuration: NFKMLXSDTextToImageConfiguration {
            NFKMLXStableDiffusionRelease(base).configuration.withSampler(sampler)
        }

        /// The adapter file: the copy in `directoryURL` where the directory holds one, else the copy in the
        /// default Hugging Face cache, downloaded when absent.
        func adapterURL(directoryURL: URL) throws -> URL {
            let local = directoryURL.appendingPathComponent(weightsPath)
            if FileManager.default.fileExists(atPath: local.path) {
                return local
            }
            return try NFKMLXDownload.weightsURL(repo: repository, weightsPath: weightsPath, revision: nil,
                                                 cacheDirectoryURL: nil)
        }

        /// The base release in `directoryURL` with the adapter merged in at its trained strength.
        func backend(directoryURL: URL, precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
            try NFKMLXTextToImage.backend(configuration: configuration, directoryURL: directoryURL,
                                          adapterURL: adapterURL(directoryURL: directoryURL), adapterScale: 1,
                                          precision: precision)
        }
    }

    /// The released adapters at reference parity on their base releases.
    static let stableDiffusionAdapters: [StableDiffusionAdapter] = [
        StableDiffusionAdapter(name: "lcm-lora-sdv1-5", displayName: "Stable Diffusion 1.5 with LCM-LoRA",
                               summary: "Stable Diffusion 1.5 with the LCM-LoRA adapter merged in, drawing 512-pixel images in four steps.",
                               base: .stableDiffusion15, repository: "latent-consistency/lcm-lora-sdv1-5",
                               weightsPath: "pytorch_lora_weights.safetensors", sampler: .lcm,
                               license: .license(nil, name: "CreativeML Open RAIL-M License",
                                                 url: "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5",
                                                 note: "Use restrictions apply. The adapter is under the CreativeML Open RAIL++-M License.")),
        StableDiffusionAdapter(name: "sdxl-lightning-4step-lora", displayName: "SDXL-Lightning 4-step",
                               summary: "SDXL base 1.0 with ByteDance's four-step SDXL-Lightning adapter merged in, drawing 1024-pixel images in four steps.",
                               base: .sdxlBase, repository: "ByteDance/SDXL-Lightning",
                               weightsPath: "sdxl_lightning_4step_lora.safetensors", sampler: .ddimTrailing,
                               license: openRAILPlusPlus(url: "https://huggingface.co/ByteDance/SDXL-Lightning/blob/main/LICENSE.md",
                                                         note: "Use restrictions apply. The base release and the adapter share the license.")),
    ]

    /// The CreativeML Open RAIL++-M License published at `url`.
    static func openRAILPlusPlus(url: String, note: String = "Use restrictions apply.") -> NFKModelLicense {
        .license(nil, name: "CreativeML Open RAIL++-M License", url: url, note: note)
    }

    /// The files a release's download fetches: the vocabularies and network configurations, then each
    /// weight file's candidates.
    static func stableDiffusionFiles(_ release: NFKMLXStableDiffusionRelease) -> [NFKModelFile] {
        let weights = release.weightFiles.flatMap { alternatives in
            alternatives.map { file($0, optional: alternatives.count > 1) }
        }
        return release.vocabularyFiles.map { file($0.path, optional: !$0.required) } + weights
    }

    static var stableDiffusionEntries: [NFKMLXCatalogEntry] {
        stableDiffusionReleaseEntries + stableDiffusionAdapterEntries
    }

    private static var stableDiffusionReleaseEntries: [NFKMLXCatalogEntry] {
        stableDiffusionReleases.map { sdModel in
            let release = NFKMLXStableDiffusionRelease(sdModel)
            let details: (displayName: String, summary: String, license: NFKModelLicense)
            switch sdModel {
            case .stableDiffusion15:
                details = ("Stable Diffusion 1.5", "A latent-diffusion generator of 512-pixel images from a prompt.",
                           .license(nil, name: "CreativeML Open RAIL-M License",
                                    url: "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5",
                                    note: "Use restrictions apply."))
            case .stableDiffusion21Base:
                details = ("Stable Diffusion 2.1 base", "A latent-diffusion generator of 512-pixel images from a prompt.",
                           openRAILPlusPlus(url: "https://huggingface.co/stabilityai/stable-diffusion-2-1-base"))
            case .sdxlTurbo:
                details = ("SDXL-Turbo", "A distilled latent-diffusion generator of 512-pixel images in one step.",
                           stabilityCommunity)
            case .sdxlBase:
                details = ("SDXL base 1.0", "A latent-diffusion generator of 1024-pixel images from a prompt.",
                           openRAILPlusPlus(url: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md"))
            case .lcmDreamshaperV7:
                details = ("LCM Dreamshaper v7",
                           "A latent consistency model distilled from a Stable Diffusion 1.5 fine-tune, drawing 512-pixel images in four steps.",
                           .mit(url: "https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7",
                                note: "Distilled from Dreamshaper v7, a Stable Diffusion 1.5 fine-tune under the CreativeML Open RAIL-M License."))
            }
            var source = NFKModelSource.hub(release.repository, layout: NFKModelSourceLayoutDirectory,
                                            files: stableDiffusionFiles(release))
            if sdModel == .stableDiffusion21Base {
                source = source.noting("The publisher no longer serves \(release.repository) publicly. A local copy of the release builds under this registry name.",
                                       publiclyServed: false)
            }
            return stableDiffusionEntry(stableDiffusionName(sdModel), details.displayName, summary: details.summary,
                                        configuration: release.configuration, license: details.license,
                                        source: source)
        }
    }

    private static var stableDiffusionAdapterEntries: [NFKMLXCatalogEntry] {
        stableDiffusionAdapters.map { adapter in
            let release = NFKMLXStableDiffusionRelease(adapter.base)
            let adapterFile = NFKModelFile.file(adapter.weightsPath, role: NFKModelFileRoleWeights,
                                                repository: adapter.repository)
            return stableDiffusionEntry(adapter.name, adapter.displayName, summary: adapter.summary,
                                        configuration: adapter.configuration, license: adapter.license,
                                        source: .hub(release.repository, layout: NFKModelSourceLayoutDirectory,
                                                     files: stableDiffusionFiles(release) + [adapterFile]))
        }
    }

    /// A Stable Diffusion entry at `configuration`'s size, steps, and guidance.
    private static func stableDiffusionEntry(_ name: String, _ displayName: String, summary: String,
                                             configuration: NFKMLXSDTextToImageConfiguration,
                                             license: NFKModelLicense, source: NFKModelSource) -> NFKMLXCatalogEntry {
        let side = configuration.sampleSize * configuration.latentStride
        let conditionsOnGuidance = configuration.unet.timeConditionProjectionDimensions != nil
        let guidanceHelp = conditionsOnGuidance
            ? "The guidance scale the model conditions on through its time embedding. It runs no classifier-free guidance."
            : "The classifier-free guidance weight. 0 runs without guidance."
        let negativePrompt = conditionsOnGuidance
            ? NFKModelPort.input(NFKInputNegativePrompt, title: "Negative prompt",
                                 help: "Not read: the model runs no classifier-free guidance.")
            : negativePromptInput
        return model(name, displayName, family: "stable-diffusion", summary: summary,
                     tasks: [NFKModelTaskTextToImage, NFKModelTaskImageToImage],
                     inputs: [promptInput, negativePrompt,
                              .input(NFKInputImage, title: "Source image",
                                     help: "An image to start from. Its presence runs image-to-image."),
                              .input(NFKInputMask, title: "Mask",
                                     help: "With a source image, white regenerates the region and black keeps it.")],
                     outputs: [generatedImageOutput],
                     parameters: [widthOption(side, help: "The image width in pixels, rounded down to a whole latent cell."),
                                  heightOption(side, help: "The image height in pixels, rounded down to a whole latent cell."),
                                  stepsOption(configuration.steps),
                                  guidanceOption(Double(configuration.guidanceScale), help: guidanceHelp),
                                  strengthOption(1, help: "How much of the source image the sampler regenerates. 1 starts from full noise."),
                                  .option(NFKParameterSeed, title: "Seed",
                                          help: "The seed of the initial noise. A run without one uses a fixed seed.")],
                     license: license, source: source)
    }

    /// The text-to-image generator options, at `steps` and `guidance`.
    static func generatorOptions(side: Int, multiple: Int, steps: Int, guidance: Double,
                                 guidanceHelp: String) -> [NFKModelOption] {
        [widthOption(side, help: "The image width in pixels, a multiple of \(multiple)."),
         heightOption(side, help: "The image height in pixels, a multiple of \(multiple)."),
         seedOption(0), stepsOption(steps), guidanceOption(guidance, help: guidanceHelp)]
    }

    /// A release of a generator that reads its geometry from the release directory.
    struct GeneratorRelease {
        let name: String
        let repository: String
        let displayName: String
        let license: NFKModelLicense
    }

    /// The Stable Diffusion 3 releases other than SD3.5 Medium, which `sd3` builds.
    static let sd3AdditionalReleases: [GeneratorRelease] = [
        GeneratorRelease(name: "sd3-medium", repository: "stabilityai/stable-diffusion-3-medium-diffusers",
                         displayName: "Stable Diffusion 3 Medium",
                         license: .license(nil, name: "Stability AI Non-Commercial Research Community License",
                                           url: "https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/blob/main/LICENSE",
                                           gated: true, restrictsCommercialUse: true,
                                           note: "Gated: accept the terms on Hugging Face. Non-commercial research use only.")),
        GeneratorRelease(name: "sd3.5-large", repository: "stabilityai/stable-diffusion-3.5-large",
                         displayName: "Stable Diffusion 3.5 Large", license: gatedStabilityCommunity),
    ]

    /// The Stability AI Community License on a repository whose terms are accepted on Hugging Face first.
    static var gatedStabilityCommunity: NFKModelLicense {
        .license(nil, name: "Stability AI Community License", url: "https://stability.ai/community-license-agreement",
                 gated: true, restrictsCommercialUse: true,
                 note: "Gated: accept the terms on Hugging Face. Free below US$1M in annual revenue; above it, Stability AI's enterprise license applies.")
    }

    static var sd3Entries: [NFKMLXCatalogEntry] {
        let releases = [GeneratorRelease(name: NFKMLXSD3Generator.modelName, repository: NFKMLXSD3Generator.releaseRepo,
                                         displayName: "Stable Diffusion 3.5 Medium", license: gatedStabilityCommunity)]
            + sd3AdditionalReleases
        return releases.map { release in
            model(release.name, release.displayName, family: "stable-diffusion-3",
                  summary: "A multimodal diffusion transformer that draws an image from a prompt.",
                  tasks: [NFKModelTaskTextToImage],
                  inputs: [promptInput, negativePromptInput], outputs: [generatedImageOutput],
                  parameters: generatorOptions(side: 1024, multiple: 16, steps: 28, guidance: 7,
                                               guidanceHelp: "The classifier-free guidance weight. Above 1 the image guides against the negative prompt."),
                  license: release.license,
                  source: hubRelease(release.repository, components: NFKMLXSD3Generator.releaseComponents),
                  residencies: stagedResidencies)
        }
    }

    /// The registry name of FLUX.1 [dev].
    static let fluxDevName = "flux.1-dev"

    static var fluxEntries: [NFKMLXCatalogEntry] {
        let releases: [(name: String, repository: String, displayName: String, steps: Int, license: NFKModelLicense,
                        summary: String)] = [
            (NFKMLXFlux.modelName, "black-forest-labs/FLUX.1-schnell", "FLUX.1 [schnell]", 4, gatedApache2,
             "A rectified-flow transformer distilled to draw an image from a prompt in four steps."),
            (fluxDevName, "black-forest-labs/FLUX.1-dev", "FLUX.1 [dev]", 4, fluxDevNonCommercial,
             "A guidance-distilled rectified-flow transformer that draws an image from a prompt."),
        ]
        return releases.map { release in
            model(release.name, release.displayName, family: "flux",
                  summary: release.summary, tasks: [NFKModelTaskTextToImage],
                  inputs: [promptInput], outputs: [generatedImageOutput],
                  parameters: generatorOptions(side: 1024, multiple: 16, steps: release.steps, guidance: 3.5,
                                               guidanceHelp: "The guidance a guidance-distilled release ([dev]) embeds. [schnell] ignores it."),
                  license: release.license,
                  source: hubRelease(release.repository, components: NFKMLXFlux.releaseComponents),
                  residencies: stagedResidencies)
        }
    }

    /// The FLUX.2 [klein] releases other than klein 4B, which `flux.2-klein-4b` builds.
    static let flux2AdditionalReleases: [GeneratorRelease] = [
        GeneratorRelease(name: "flux.2-klein-9b", repository: "black-forest-labs/FLUX.2-klein-9B",
                         displayName: "FLUX.2 [klein] 9B", license: fluxNonCommercial),
        GeneratorRelease(name: "flux.2-klein-base-4b", repository: "black-forest-labs/FLUX.2-klein-base-4B",
                         displayName: "FLUX.2 [klein] base 4B", license: .apache2()),
        GeneratorRelease(name: "flux.2-klein-base-9b", repository: "black-forest-labs/FLUX.2-klein-base-9B",
                         displayName: "FLUX.2 [klein] base 9B", license: fluxNonCommercial),
        GeneratorRelease(name: "flux.2-klein-9b-kv", repository: "black-forest-labs/FLUX.2-klein-9b-kv",
                         displayName: "FLUX.2 [klein] 9B KV",
                         license: .license(nil, name: "FLUX Non-Commercial License",
                                           url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv/blob/main/LICENSE",
                                           gated: true, restrictsCommercialUse: true,
                                           note: "Gated: accept the terms on Hugging Face. Non-commercial use only.")),
    ]

    static var flux2Entries: [NFKMLXCatalogEntry] {
        let releases = [GeneratorRelease(name: NFKMLXFlux2.modelName, repository: "black-forest-labs/FLUX.2-klein-4B",
                                         displayName: "FLUX.2 [klein] 4B", license: .apache2())]
            + flux2AdditionalReleases
        return releases.map { release in
            model(release.name, release.displayName, family: "flux2",
                  summary: "A rectified-flow transformer that draws, edits from reference images, and inpaints from a prompt.",
                  tasks: [NFKModelTaskTextToImage, NFKModelTaskImageToImage, NFKModelTaskInpainting],
                  inputs: [promptInput,
                           .input(NFKInputNegativePrompt, title: "Negative prompt",
                                  help: "What to steer away from. A distilled release ignores it."),
                           .input(NFKInputImages, title: "References",
                                  help: "The images the generation is conditioned on, each resampled to a multiple of 16 on each side."),
                           .input(NFKInputImage, title: "Image to inpaint",
                                  help: "The image to repaint where the mask is white. It needs the mask."),
                           .input(NFKInputMask, title: "Mask", help: "White repaints the region. It needs the image to inpaint.")],
                  outputs: [generatedImageOutput],
                  parameters: generatorOptions(side: 1024, multiple: 16, steps: 28, guidance: 4,
                                               guidanceHelp: "The guidance weight a base release reads. A distilled release ignores it.")
                      + [strengthOption(0.8, help: "How much of the image to inpaint the sampler regenerates. Read by inpainting only."),
                         .option(NFKMLXFlux2ParameterKey.cachesReferences, kind: NFKModelOptionKindBoolean,
                                 title: "Cache references",
                                 help: "Generate with the reference cache the 9B KV release is trained for.",
                                 defaultValue: false)],
                  license: release.license,
                  source: hubRelease(release.repository, components: NFKMLXFlux2.releaseComponents),
                  residencies: stagedResidencies)
        }
    }

    static var conditioningScaleOption: NFKModelOption {
        .option(NFKMLXControlNetParameterKey.conditioningScale, kind: NFKModelOptionKindNumber, title: "Control weight",
                help: "The weight the ControlNet's residuals carry.", defaultValue: 1, minimum: 0)
    }

    static func controlImageInput() -> NFKModelPort {
        .input(NFKInputControl, required: true, title: "Control",
               help: "The control image the output follows, resized to the output size.")
    }

    static func controlSizeOptions() -> [NFKModelOption] {
        [widthOption(nil, help: "The image width in pixels. The control image's width rounded down to a multiple of 16 when absent."),
         heightOption(nil, help: "The image height in pixels. The control image's height rounded down to a multiple of 16 when absent."),
         seedOption(0)]
    }

    static var fluxControlNet: NFKMLXCatalogEntry {
        let controlModes: [(value: Any, title: String)] = [(0, "Canny"), (1, "Tile"), (2, "Depth"), (3, "Blur"),
                                                           (4, "Pose"), (5, "Gray"), (6, "Low quality")]
        return model(NFKMLXFluxControlNetGenerator.modelName, "FLUX.1 [dev] ControlNet Union Pro", family: "flux",
                     summary: "FLUX.1 [dev] steered by a union ControlNet to follow a control image.",
                     tasks: [NFKModelTaskControlNet],
                     inputs: [promptInput, controlImageInput()], outputs: [generatedImageOutput],
                     parameters: controlSizeOptions()
                         + [stepsOption(28),
                            guidanceOption(3.5, help: "The guidance the guidance-distilled base embeds."),
                            conditioningScaleOption,
                            .option(NFKMLXControlNetParameterKey.controlMode, kind: NFKModelOptionKindChoice,
                                    title: "Control type",
                                    help: "What the control image holds. A union ControlNet requires it.",
                                    choices: controlModes)],
                     license: fluxDevNonCommercial,
                     source: controlNetRelease(NFKMLXFluxControlNetGenerator.controlNetReleaseRepo,
                                               components: NFKMLXFluxControlNetGenerator.controlNetComponents,
                                               base: NFKMLXFluxControlNetGenerator.releaseRepo,
                                               baseComponents: NFKMLXFlux.releaseComponents),
                     residencies: stagedResidencies)
    }

    static var sd3ControlNet: NFKMLXCatalogEntry {
        model(NFKMLXSD3ControlNetGenerator.modelName, "Stable Diffusion 3.5 Large ControlNet Canny",
              family: "stable-diffusion-3",
              summary: "Stable Diffusion 3.5 Large steered by a ControlNet to follow an edge map.",
              tasks: [NFKModelTaskControlNet],
              inputs: [promptInput, negativePromptInput, controlImageInput()], outputs: [generatedImageOutput],
              parameters: controlSizeOptions()
                  + [stepsOption(28),
                     guidanceOption(7, help: "The classifier-free guidance weight. Above 1 the image guides against the negative prompt."),
                     conditioningScaleOption],
              license: gatedStabilityCommunity,
              source: controlNetRelease(NFKMLXSD3ControlNetGenerator.controlNetReleaseRepo,
                                        components: NFKMLXSD3ControlNetGenerator.controlNetComponents,
                                        base: NFKMLXSD3ControlNetGenerator.releaseRepo,
                                        baseComponents: NFKMLXSD3Generator.releaseComponents),
              residencies: stagedResidencies)
    }

    /// The registry name of SANA 1600M.
    static let sana1600MName = "sana-1600m"

    static var sanaEntries: [NFKMLXCatalogEntry] {
        let releases: [(name: String, repository: String, displayName: String)] = [
            (NFKMLXSANAGenerator.modelName, NFKMLXSANAGenerator.releaseRepo, "SANA 600M 1024px"),
            (sana1600MName, "Efficient-Large-Model/Sana_1600M_1024px_diffusers", "SANA 1600M 1024px"),
        ]
        return releases.map { release in
            model(release.name, release.displayName, family: "sana",
                  summary: "A linear-attention diffusion transformer that draws an image from a prompt.",
                  tasks: [NFKModelTaskTextToImage],
                  inputs: [promptInput, negativePromptInput], outputs: [generatedImageOutput],
                  parameters: generatorOptions(side: 1024, multiple: 32, steps: 20, guidance: 4.5,
                                               guidanceHelp: "The classifier-free guidance weight."),
                  license: .license("Apache-2.0", name: "Apache License 2.0",
                                    url: "https://huggingface.co/\(release.repository)/blob/main/LICENSE"),
                  source: hubRelease(release.repository, components: NFKMLXSANAGenerator.releaseComponents),
                  residencies: stagedResidencies)
        }
    }

    /// The registry name of the Z-Image base release.
    static let zImageBaseName = "z-image-base"

    static var zImageEntries: [NFKMLXCatalogEntry] {
        let releases: [(name: String, repository: String, displayName: String, summary: String)] = [
            (NFKMLXZImageGenerator.modelName, NFKMLXZImageGenerator.releaseRepo, "Z-Image-Turbo",
             "A distilled single-stream diffusion transformer that draws an image from a prompt in nine steps."),
            (zImageBaseName, "Tongyi-MAI/Z-Image", "Z-Image",
             "A single-stream diffusion transformer that draws an image from a prompt."),
        ]
        return releases.map { release in
            model(release.name, release.displayName, family: "z-image", summary: release.summary,
                  tasks: [NFKModelTaskTextToImage],
                  inputs: [promptInput, negativePromptInput], outputs: [generatedImageOutput],
                  parameters: generatorOptions(side: 1024, multiple: 16, steps: 9, guidance: 0,
                                               guidanceHelp: "The classifier-free guidance weight. Turbo runs at 0; the base release is sampled with guidance."),
                  license: .license("Apache-2.0", name: "Apache License 2.0",
                                    url: "https://huggingface.co/\(release.repository)"),
                  source: hubRelease(release.repository, components: NFKMLXZImageGenerator.releaseComponents),
                  residencies: stagedResidencies)
        }
    }

    static var qwenImage: NFKMLXCatalogEntry {
        model(NFKMLXQwenImageGenerator.modelName, "Qwen-Image 2.1", family: "qwen-image",
              summary: "A diffusion transformer conditioned on Qwen3-VL that draws an image from a prompt.",
              tasks: [NFKModelTaskTextToImage],
              inputs: [promptInput, negativePromptInput], outputs: [generatedImageOutput],
              parameters: [widthOption(1024, help: "The image width in pixels, rounded down to a multiple of 32."),
                           heightOption(1024, help: "The image height in pixels, rounded down to a multiple of 32."),
                           seedOption(0), stepsOption(40),
                           guidanceOption(1, help: "The classifier-free guidance weight. It applies above 1 and only with a negative prompt.")],
              license: .license(nil, name: "Qwen Research License",
                                url: "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
                                restrictsCommercialUse: true, note: "Research and evaluation use only."),
              source: hubRelease(NFKMLXQwenImageGenerator.releaseRepo, components: NFKMLXQwenImageGenerator.releaseComponents),
              residencies: stagedResidencies)
    }
}
