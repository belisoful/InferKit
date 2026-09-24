//
//  NFKMLXQwenImageGenerator.swift
//  InferKitMLX
//
//  Qwen-Image 2.1 text-to-image from a downloaded release directory, its two stages held as an
//  `NFKMLXResidency` says. The Qwen3-VL text encoder runs once per image and the transformer and
//  autoencoder after it, so a release too large to hold whole runs when the two take turns.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// The weights a release occupies at a load precision, read from its shard headers.
enum NFKMLXStageWeights {
    static func bytes(inDirectory directory: URL, precision: NFKMLXWeightPrecision) throws -> Int {
        try NFKMLXExpertInventory(inDirectory: directory, precision: precision) { _, _ in [] }.totalBytes
    }
}

/// The pipeline stage of a Qwen-Image release: the transformer, the autoencoder, and the latent
/// statistics that join them.
final class NFKQwenImagePipelineStage {
    let pipeline: NFKMLXQwenImagePipeline
    init(_ pipeline: NFKMLXQwenImagePipeline) { self.pipeline = pipeline }
}

/// Qwen-Image 2.1 text-to-image, assembled from a diffusers release directory.
///
/// @discussion The release layout is `text_encoder/` (Qwen3-VL at the 8B geometry), `processor/` (its
/// tokenizer), `transformer/`, `vae/`, and `scheduler/`. ``image(forPrompt:negativePrompt:width:height:seed:)``
/// encodes the prompt through the release's raw template, denoises over the flow schedule, and decodes
/// to an RGBA image, the four channels the release's autoencoder produces.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXQwenImageGenerator)
public final class NFKMLXQwenImageGenerator: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "qwen-image-2.1"

    /// The Hugging Face repository the download factories read.
    @objc public static let releaseRepo = "Qwen/Qwen-Image-2.1"

    private let staging: NFKMLXStagedModel
    private let textEncoderStage: NFKMLXStage<NFKMLXLanguageNet>
    private let pipelineStage: NFKMLXStage<NFKQwenImagePipelineStage>
    private let tokenizer: NFKTokenizer

    /// The denoising steps, 40 by the reference's default.
    @objc public var steps: Int = 40
    /// The true classifier-free guidance scale. It applies above 1 and only with a negative prompt,
    /// which is the reference's rule; the release is meant to be sampled without guidance, so the
    /// default is 1.
    @objc public var guidance: Float = 1

    /// Whether the text encoder and the transformer stay loaded together between images. False where
    /// the release is staged: each image then loads the encoder, encodes, releases it, and loads the
    /// transformer and the autoencoder.
    @objc public var holdsStagesResident: Bool { staging.resident }

    init(resident: Bool, tokenizer: NFKTokenizer,
         loadTextEncoder: @escaping () throws -> NFKMLXLanguageNet,
         loadPipeline: @escaping () throws -> NFKMLXQwenImagePipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textEncoderStage = NFKMLXStage(loadTextEncoder)
        pipelineStage = NFKMLXStage { NFKQwenImagePipelineStage(try loadPipeline()) }
        self.tokenizer = tokenizer
        super.init()
    }

    /// Loads both stages now, for a resident release: a bad release fails at construction rather than
    /// at the first image.
    func loadResident() throws {
        try staging.use(textEncoderStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    var isHoldingTextEncoder: Bool { staging.exclusively { textEncoderStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    // MARK: Factories

    /// Assembles the model from a downloaded release directory, holding it as
    /// ``NFKMLXResidency/automatic`` decides.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXQwenImageGenerator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    ///
    /// @discussion The text encoder and the transformer load at the bfloat16 the release stores them
    /// in, about 16 GB and 14 GB; the autoencoder loads at float32. A machine that cannot hold the two
    /// together stages under ``NFKMLXResidency/automatic``: a 32 GB Mac runs the release staged. The
    /// release has no routed experts, so ``NFKMLXResidency/paged`` holds it as
    /// ``NFKMLXResidency/staged`` does.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXQwenImageGenerator {
        let encoderDirectory = directoryURL.appendingPathComponent("text_encoder")
        let transformerDirectory = directoryURL.appendingPathComponent("transformer")
        let vaeDirectory = directoryURL.appendingPathComponent("vae")
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(
            inDirectory: directoryURL.appendingPathComponent("processor")) else {
            throw NFKMLXError.unsupportedConfiguration("the release's processor/ tokenizer could not be read")
        }
        let configuration = try NFKMLXQwenImage.configuration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json"))
        let vaeConfigURL = vaeDirectory.appendingPathComponent("config.json")
        let vaeConfiguration = try NFKMLXQwenImageVAE.configuration(fromHuggingFace: vaeConfigURL)
        let statistics = try NFKMLXQwenImageVAE.latentStatistics(fromHuggingFace: vaeConfigURL)

        let plan = try NFKMLXResidencyBudget.plan(
            [NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(inDirectory: encoderDirectory,
                                                                        precision: .checkpoint)),
             NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(inDirectory: transformerDirectory,
                                                                        precision: .checkpoint)
                                     + NFKMLXStageWeights.bytes(inDirectory: vaeDirectory, precision: .float32))],
            residency: residency, budget: NFKMLXResidencyBudget.current())
        let generator = NFKMLXQwenImageGenerator(
            resident: plan.holdsStagesResident, tokenizer: tokenizer,
            loadTextEncoder: { try NFKMLXQwen3VL.decoder(directoryURL: encoderDirectory, precision: .checkpoint) },
            loadPipeline: {
                let transformer = NFKMLXQwenImage.makeNet(configuration)
                try NFKMLXQwenImage.loadWeights(into: transformer, fromDirectory: transformerDirectory,
                                                precision: .checkpoint)
                let vae = NFKMLXQwenImageVAE.makeNet(vaeConfiguration)
                try NFKMLXQwenImageVAE.loadWeights(into: vae, fromDirectory: vaeDirectory)
                return NFKMLXQwenImagePipeline(transformer: transformer, vae: vae, latentMean: statistics.mean,
                                               latentStandardDeviation: statistics.standardDeviation)
            })
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    /// The component folders a download fetches.
    static let releaseComponents = [
        NFKMLXReleaseComponent(required: ["model_index.json", "scheduler/scheduler_config.json"]),
        NFKMLXReleaseComponent(required: ["text_encoder/config.json"],
                               weights: ["text_encoder/model.safetensors",
                                         "text_encoder/model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["processor/tokenizer.json", "processor/tokenizer_config.json"],
                               optional: ["processor/vocab.json", "processor/merges.txt",
                                          "processor/added_tokens.json", "processor/special_tokens_map.json"]),
        NFKMLXReleaseComponent(required: ["transformer/config.json"],
                               weights: ["transformer/diffusion_pytorch_model.safetensors",
                                         "transformer/diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]),
    ]

    /// Downloads the release (``releaseRepo`` when `repo` is nil) and assembles the model, holding it
    /// as `residency` says.
    ///
    /// @discussion The download is about 31 GB. A file already in the cache is not fetched again. The
    /// call blocks on the network, so run it off the main and render threads. The weights are under the
    /// Qwen Research License, which is non-commercial.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXQwenImageGenerator {
        let directory = try NFKMLXReleaseDownload.directory(repo: repo ?? releaseRepo, revision: revision,
                                                            cacheDirectoryURL: cacheDirectoryURL,
                                                            components: releaseComponents)
        return try generator(directoryURL: directory, residency: residency)
    }

    /// The asynchronous form of ``generator(repo:revision:cacheDirectoryURL:residency:)``. The handler
    /// runs on a background queue.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency,
                                 completionHandler: @escaping (NFKMLXQwenImageGenerator?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try generator(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                                residency: residency), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    // MARK: Generation

    /// The prompt features the transformer reads, `[tokens, 4096]`. A staged release loads the text
    /// encoder for the call and releases it after.
    public func encode(prompt: String) throws -> MLXArray {
        try staging.with(textEncoderStage) { decoder in
            NFKMLXQwenImagePipeline.promptEmbeddings(prompt, decoder: decoder, tokenizer: tokenizer)
        }
    }

    /// Generates an image for `prompt`, `[height, width, 4]` RGBA in `[0, 1]`.
    ///
    /// @discussion `width` and `height` round down to multiples of 32, as the reference rounds them. A
    /// negative prompt guides only where ``guidance`` is above 1. Both prompts encode from one load of
    /// the text encoder.
    public func image(forPrompt prompt: String, negativePrompt: String? = nil, width: Int = 1024,
                      height: Int = 1024, seed: UInt64 = 0) throws -> MLXArray {
        let (width, height) = (width / 32 * 32, height / 32 * 32)
        guard width > 0, height > 0 else {
            throw NFKMLXError.unsupportedConfiguration("an image is at least 32 pixels on a side")
        }
        let guides = guidance > 1 && negativePrompt != nil
        return try staging.exclusively {
            let prompts = guides ? [prompt, negativePrompt ?? ""] : [prompt]
            let embeddings = try staging.with(textEncoderStage) { decoder in
                prompts.map { NFKMLXQwenImagePipeline.promptEmbeddings($0, decoder: decoder, tokenizer: tokenizer) }
            }
            return try staging.with(pipelineStage) { stage in
                let image = stage.pipeline.generate(
                    promptEmbeddings: embeddings[0], negativeEmbeddings: guides ? embeddings[1] : nil,
                    height: height, width: width, steps: steps, guidance: guidance, seed: seed)
                return clip(image / 2 + 0.5, min: 0, max: 1)
            }
        }
    }

    /// Generates an image for `prompt` and returns it as an RGBA `CGImage`.
    @objc(imageForPrompt:negativePrompt:width:height:seed:error:)
    public func cgImage(forPrompt prompt: String, negativePrompt: String?, width: Int, height: Int,
                        seed: UInt64) throws -> CGImage {
        let array = try image(forPrompt: prompt, negativePrompt: negativePrompt, width: width, height: height,
                              seed: seed)
        return try NFKMLXImageBridge.cgImage(from: array, options: NFKMLXImageOptions())
    }
}
