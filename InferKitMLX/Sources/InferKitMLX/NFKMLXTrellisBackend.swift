//
//  NFKMLXTrellisBackend.swift
//  InferKitMLX
//
//  TRELLIS as an InferKit backend: an image (the image release) or a prompt (the text releases) in, a
//  mesh under NFKOutputMesh and Gaussians under NFKOutputGaussianSplats out.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// The TRELLIS releases. Introduced in InferKit 0.4.0.
@objc(NFKMLXTrellisVariant)
public enum NFKMLXTrellisVariant: Int, Sendable, CaseIterable {
    /// microsoft/TRELLIS-image-large: an image in, DINOv2-conditioned.
    case imageLarge
    /// microsoft/TRELLIS-text-base: a prompt in, CLIP-conditioned, 768-wide flows.
    case textBase
    /// microsoft/TRELLIS-text-large: 1024-wide flows.
    case textLarge
    /// microsoft/TRELLIS-text-xlarge: 1280-wide flows.
    case textXLarge
}

/// Request parameter keys a TRELLIS backend reads. Introduced in InferKit 0.4.0.
@objc(NFKMLXTrellisParameterKey)
public final class NFKMLXTrellisParameterKey: NSObject {
    /// NSNumber (integer): the structure stage's sampling steps (the release's 25 when absent).
    @objc public static let structureSteps = "trellisStructureSteps"
    /// NSNumber (integer): the structured-latent stage's sampling steps (25 when absent).
    @objc public static let latentSteps = "trellisLatentSteps"
    /// NSNumber (float): the structure stage's guidance strength (5 for the image release, 7.5 for text).
    @objc public static let structureGuidance = "trellisStructureGuidance"
    /// NSNumber (float): the structured-latent stage's guidance strength.
    @objc public static let latentGuidance = "trellisLatentGuidance"
    /// NSArray of output keys: which of NFKOutputMesh and NFKOutputGaussianSplats to decode (both when
    /// absent).
    @objc public static let outputs = "trellisOutputs"
}

/// TRELLIS as an InferKit backend. Build one through `NFKMLXTrellis`'s factories. Introduced in InferKit 0.4.0.
@objc(NFKMLXTrellisBackend)
public final class NFKMLXTrellisBackend: NSObject, NFKInferenceBackend {
    public let pipeline: NFKMLXTrellisPipeline
    private let identifier: String
    private let lock = NSLock()

    init(pipeline: NFKMLXTrellisPipeline, identifier: String) {
        self.pipeline = pipeline
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }
    /// Whether the networks stay loaded between runs (else each loads for its stage).
    @objc public var holdsStagesResident: Bool { pipeline.holdsStagesResident }

    @objc public var supportedInputKeys: Set<String> { pipeline.isImageConditioned ? [NFKInputImage] : [NFKInputPrompt] }
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputMesh, NFKOutputGaussianSplats] }
    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterSeed, NFKMLXTrellisParameterKey.structureSteps, NFKMLXTrellisParameterKey.latentSteps,
         NFKMLXTrellisParameterKey.structureGuidance, NFKMLXTrellisParameterKey.latentGuidance, NFKMLXTrellisParameterKey.outputs]
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let condition: (condition: MLXArray, unconditional: MLXArray)
        if pipeline.isImageConditioned {
            guard let value = request.input(forKey: NFKInputImage) else {
                throw NFKMLXError.unsupportedInput
            }
            let image = try NFKMLXImageBridge.rgbaBytes(from: value, colorSpace: CGColorSpaceCreateDeviceRGB())
            condition = try pipeline.condition(prepared: pipeline.preparedImage(rgba: image.bytes, width: image.width, height: image.height))
        } else {
            guard let prompt = request.input(forKey: NFKInputPrompt) as? String else {
                throw NFKMLXError.unsupportedInput
            }
            condition = try pipeline.condition(prompt: prompt)
        }
        func number(_ key: String) -> NSNumber? { request.parameter(forKey: key) as? NSNumber }
        var structure = pipeline.structureSampling, latent = pipeline.latentSampling
        if let steps = number(NFKMLXTrellisParameterKey.structureSteps) { structure.steps = max(1, steps.intValue) }
        if let steps = number(NFKMLXTrellisParameterKey.latentSteps) { latent.steps = max(1, steps.intValue) }
        if let guidance = number(NFKMLXTrellisParameterKey.structureGuidance) { structure.guidance = guidance.floatValue }
        if let guidance = number(NFKMLXTrellisParameterKey.latentGuidance) { latent.guidance = guidance.floatValue }
        var outputs = NFKMLXTrellisOutputs.all
        if let keys = request.parameter(forKey: NFKMLXTrellisParameterKey.outputs) as? [String] {
            outputs = []
            if keys.contains(NFKOutputMesh) { outputs.insert(.mesh) }
            if keys.contains(NFKOutputGaussianSplats) { outputs.insert(.gaussians) }
            guard !outputs.isEmpty else {
                throw NFKMLXError.unsupportedConfiguration("TRELLIS decodes \(NFKOutputMesh) and \(NFKOutputGaussianSplats), not \(keys)")
            }
        }
        let seed = number(NFKParameterSeed)?.uint64Value ?? 42
        let result = try lock.withLock {
            try pipeline.generate(condition: condition.condition, unconditional: condition.unconditional, seed: seed, outputs: outputs,
                                  structureSampling: structure, latentSampling: latent)
        }
        var values = [String: Any]()
        values[NFKOutputMesh] = result.mesh
        values[NFKOutputGaussianSplats] = result.gaussians
        return NFKInferenceResult(outputs: values)
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
            do {
                job.finish(with: try self.runInference(for: request))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

/// Delivers an asynchronous factory's result across the detached task.
private final class NFKTrellisCompletion: @unchecked Sendable {
    let handler: ((any NFKInferenceBackend)?, Error?) -> Void
    init(_ handler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) { self.handler = handler }
    func call(_ backend: (any NFKInferenceBackend)?, _ error: Error?) { handler(backend, error) }
}

/// TRELLIS's factories, registration, and release facts. Introduced in InferKit 0.4.0.
@objc(NFKMLXTrellisFactory)
public final class NFKMLXTrellisFactory: NSObject {

    struct Specification {
        let name: String
        let repo: String
        let revision: String
        let checkpoints: [String]
        var isImage: Bool { checkpoints.contains { $0.contains("_img_") } }
    }

    static let imageRepo = "microsoft/TRELLIS-image-large"
    static let imageRevision = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
    static let decoderCheckpoints = ["ss_dec_conv3d_16l8_fp16", "slat_dec_mesh_swin8_B_64l8m256c_fp16", "slat_dec_gs_swin8_B_64l8gs32_fp16"]
    static let clipRepo = "openai/clip-vit-large-patch14"
    static let clipRevision = "32bd64288804d66eefd0ccbe215aa642df71cc41"
    static let clipFiles = ["model.safetensors", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]
    /// DINOv2 ViT-L/14 with four registers, which the image release's code loads through `torch.hub`.
    @objc public static let encoderURL = URL(string: "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth")!
    @objc public static let encoderFileName = "dinov2_vitl14_reg4_pretrain.pth"

    static func specification(_ variant: NFKMLXTrellisVariant) -> Specification {
        switch variant {
        case .imageLarge:
            return Specification(name: "trellis-image-large", repo: imageRepo, revision: imageRevision,
                                 checkpoints: ["ss_flow_img_dit_L_16l8_fp16", "slat_flow_img_dit_L_64l8p2_fp16"] + decoderCheckpoints)
        case .textBase:
            return Specification(name: "trellis-text-base", repo: "microsoft/TRELLIS-text-base",
                                 revision: "f8e8cf00c40d53dea26b718e49169ce83cf24c67",
                                 checkpoints: ["ss_flow_txt_dit_B_16l8_fp16", "slat_flow_txt_dit_B_64l8p2_fp16"])
        case .textLarge:
            return Specification(name: "trellis-text-large", repo: "microsoft/TRELLIS-text-large",
                                 revision: "4aad9f4a110329a410974d7f41ce5333a9a1fc87",
                                 checkpoints: ["ss_flow_txt_dit_L_16l8_fp16", "slat_flow_txt_dit_L_64l8p2_fp16"])
        case .textXLarge:
            return Specification(name: "trellis-text-xlarge", repo: "microsoft/TRELLIS-text-xlarge",
                                 revision: "e0b00432b8e3a8ecee0df806ab1df9f7281f2be4",
                                 checkpoints: ["ss_flow_txt_dit_XL_16l8_fp16", "slat_flow_txt_dit_XL_64l8p2_fp16"])
        }
    }

    /// The registry name of a variant.
    @objc public static func modelName(for variant: NFKMLXTrellisVariant) -> String { specification(variant).name }

    // MARK: Local factories

    /// The image release from its directory and the DINOv2 checkpoint (`encoderFileName`).
    @objc(backendWithDirectoryURL:encoderWeightsURL:residency:error:)
    public static func backend(directoryURL: URL, encoderWeightsURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        NFKMLXTrellisBackend(pipeline: try .image(directoryURL: directoryURL, encoderWeightsURL: encoderWeightsURL, residency: residency),
                             identifier: specification(.imageLarge).name)
    }

    /// A release from its directory. The image release reads `encoderFileName` beside its files; a text
    /// release reads the image release's directory (its decoders) and CLIP ViT-L/14's (its text
    /// conditioner).
    @objc(backendWithVariant:directoryURL:imageReleaseDirectoryURL:textEncoderDirectoryURL:residency:error:)
    public static func backend(variant: NFKMLXTrellisVariant, directoryURL: URL, imageReleaseDirectoryURL: URL?,
                               textEncoderDirectoryURL: URL?, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        let spec = specification(variant)
        if spec.isImage {
            return try backend(directoryURL: directoryURL, encoderWeightsURL: directoryURL.appendingPathComponent(encoderFileName),
                               residency: residency)
        }
        guard let imageReleaseDirectoryURL, let textEncoderDirectoryURL else {
            throw NFKMLXError.weightsMismatch("\(spec.name) reads the image release's decoders and CLIP ViT-L/14; name both directories")
        }
        return NFKMLXTrellisBackend(pipeline: try .text(directoryURL: directoryURL, imageReleaseDirectoryURL: imageReleaseDirectoryURL,
                                                        textEncoderDirectoryURL: textEncoderDirectoryURL, residency: residency),
                                    identifier: spec.name)
    }

    /// The image release with fine-tuned flows: `structureFlowWeightsURL` and `latentFlowWeightsURL`, where
    /// set, replace the release's checkpoints (files `NFKMLXWeights.save` writes after
    /// `NFKMLXTrellisTraining`'s recipes).
    @objc(backendWithDirectoryURL:encoderWeightsURL:structureFlowWeightsURL:latentFlowWeightsURL:residency:error:)
    public static func backend(directoryURL: URL, encoderWeightsURL: URL, structureFlowWeightsURL: URL?, latentFlowWeightsURL: URL?,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        let flows = NFKMLXTrellisFlowWeights(structure: structureFlowWeightsURL, latent: latentFlowWeightsURL)
        return NFKMLXTrellisBackend(pipeline: try .image(directoryURL: directoryURL, encoderWeightsURL: encoderWeightsURL,
                                                         residency: residency, flows: flows),
                                    identifier: specification(.imageLarge).name)
    }

    /// An image backend of random networks at `configuration` (``NFKMLXTrellisPipeline/random(configuration:)``),
    /// reported as `trellis-image-large`. Swift-only: the configuration is a Swift structure.
    public static func backend(randomWeightsAt configuration: NFKMLXTrellisConfiguration) -> any NFKInferenceBackend {
        NFKMLXTrellisBackend(pipeline: .random(configuration: configuration), identifier: specification(.imageLarge).name)
    }

    // MARK: Download factories

    /// Downloads a release and what it reads (the DINOv2 checkpoint, or the image release's decoders and
    /// CLIP ViT-L/14) into the cache, then builds. Blocking on the network; run off the render thread. A nil
    /// `revision` reads the measured one.
    @objc(backendWithVariant:revision:cacheDirectoryURL:residency:error:)
    public static func backend(variant: NFKMLXTrellisVariant, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let files = try download(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
            return try backend(files: files, variant: variant, residency: residency, quantizationBits: nil)
        }
    }

    /// The asynchronous form of ``backend(variant:revision:cacheDirectoryURL:residency:)``: downloads on a
    /// background queue, then builds and delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithVariant:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func backend(variant: NFKMLXTrellisVariant, revision: String?, cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        let handler = NFKTrellisCompletion(completionHandler)
        Task.detached(priority: .userInitiated) {
            do {
                handler.call(try backend(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: residency), nil)
            } catch {
                handler.call(nil, error)
            }
        }
    }

    struct Files {
        let release: URL
        let imageRelease: URL
        let encoder: URL?
        let textEncoder: URL?
    }

    static func components(_ checkpoints: [String]) -> [NFKMLXReleaseComponent] {
        [NFKMLXReleaseComponent(required: ["pipeline.json"])]
            + checkpoints.map { NFKMLXReleaseComponent(required: ["ckpts/\($0).json", "ckpts/\($0).safetensors"]) }
    }

    static func download(variant: NFKMLXTrellisVariant, revision: String?, cacheDirectoryURL: URL?) throws -> Files {
        let spec = specification(variant)
        let release = try NFKMLXReleaseDownload.directory(repo: spec.repo, revision: revision ?? spec.revision,
                                                          cacheDirectoryURL: cacheDirectoryURL, components: components(spec.checkpoints))
        if spec.isImage {
            return Files(release: release, imageRelease: release, encoder: try downloadedEncoder(cacheDirectoryURL: cacheDirectoryURL),
                         textEncoder: nil)
        }
        let imageRelease = try NFKMLXReleaseDownload.directory(repo: imageRepo, revision: imageRevision, cacheDirectoryURL: cacheDirectoryURL,
                                                               components: components(decoderCheckpoints))
        let clip = try NFKMLXReleaseDownload.directory(repo: clipRepo, revision: clipRevision, cacheDirectoryURL: cacheDirectoryURL,
                                                       components: [NFKMLXReleaseComponent(required: clipFiles)])
        return Files(release: release, imageRelease: imageRelease, encoder: nil, textEncoder: clip)
    }

    /// The DINOv2 checkpoint in the cache's URL downloads, fetched first when it is not there.
    static func downloadedEncoder(cacheDirectoryURL: URL?) throws -> URL {
        let hub = NFKMLXHub.hub(cacheDirectoryURL: cacheDirectoryURL)
        guard let local = hub.localURL(for: encoderURL, fileName: nil) else {
            throw NFKMLXError.weightsMismatch("the cache has no place for \(encoderURL.absoluteString)")
        }
        if !FileManager.default.fileExists(atPath: local.path) {
            try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
            try hub.fetch(encoderURL, toFileURL: local)
        }
        return local
    }

    static func backend(files: Files, variant: NFKMLXTrellisVariant, residency: NFKMLXResidency, quantizationBits: Int?) throws
        -> NFKMLXTrellisBackend {
        let spec = specification(variant)
        let pipeline: NFKMLXTrellisPipeline
        if let encoder = files.encoder {
            pipeline = try .image(directoryURL: files.release, encoderWeightsURL: encoder, quantizationBits: quantizationBits,
                                  residency: residency)
        } else {
            pipeline = try .text(directoryURL: files.release, imageReleaseDirectoryURL: files.imageRelease,
                                 textEncoderDirectoryURL: files.textEncoder!, quantizationBits: quantizationBits, residency: residency)
        }
        return NFKMLXTrellisBackend(pipeline: pipeline, identifier: spec.name)
    }

    // MARK: Registry

    /// Registers the four releases with `NFKMLXModelRegistry` under `trellis-image-large`,
    /// `trellis-text-base`, `trellis-text-large`, and `trellis-text-xlarge`. The location is the release
    /// directory a catalog download fills; what the release reads beside it (the DINOv2 checkpoint, or the
    /// image release and CLIP ViT-L/14) comes from the cache, downloaded when missing.
    /// `NFKModelLoadOptionResidency` holds the stages resident or loads each for its stage, and
    /// `NFKModelLoadOptionQuantizationBits` packs the two flows' linear layers.
    @objc public static func register() {
        for variant in NFKMLXTrellisVariant.allCases {
            NFKMLXModelRegistry.register(name: specification(variant).name,
                                         consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionQuantizationBits]) { location, options in
                let spec = specification(variant)
                var files = try download(variant: variant, revision: nil, cacheDirectoryURL: nil)
                if let location {
                    let beside = location.appendingPathComponent(encoderFileName)
                    let encoder = spec.isImage && FileManager.default.fileExists(atPath: beside.path) ? beside : files.encoder
                    files = Files(release: location, imageRelease: spec.isImage ? location : files.imageRelease, encoder: encoder,
                                  textEncoder: files.textEncoder)
                }
                return try backend(files: files, variant: variant, residency: options.residency, quantizationBits: options.quantizationBits)
            }
        }
    }
}
