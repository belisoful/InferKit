//
//  NFKMLXZImageGenerator.swift
//  InferKitMLX
//
//  Z-Image text-to-image from a downloaded diffusers release directory, its two stages held as an
//  `NFKMLXResidency` says: the Qwen3-4B text encoder runs once per image and the transformer and
//  autoencoder after it. The file also carries the release loader the transformer lacked: its
//  geometry from `transformer/config.json` and its schedule from `scheduler/scheduler_config.json`.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// Reading a diffusers Z-Image release's transformer and schedule.
enum NFKMLXZImageRelease {

    /// The DiT geometry a `transformer/config.json` describes. A release this port cannot build is
    /// refused: grouped key-value heads, attention without the query-key norm, or a patch size other
    /// than the 2×2×1 the module's patch embedder is keyed by.
    static func transformerConfiguration(fromHuggingFace url: URL) throws -> NFKMLXZImageConfiguration {
        let json = try NFKMLXWanRelease.json(url)
        let base = NFKMLXZImageConfiguration.base
        let heads = json["n_heads"] as? Int ?? base.heads
        let patch = json["all_patch_size"] as? [Int] ?? [base.patchSize]
        let framePatch = json["all_f_patch_size"] as? [Int] ?? [base.framePatchSize]
        guard patch == [2], framePatch == [1] else {
            throw NFKMLXError.unsupportedConfiguration("the transformer patches at \(patch)×\(framePatch); the port reads 2×1")
        }
        guard (json["n_kv_heads"] as? Int ?? heads) == heads, (json["qk_norm"] as? Bool ?? true) else {
            throw NFKMLXError.unsupportedConfiguration(
                "the transformer groups its key-value heads or omits the query-key norm, which the port does not implement")
        }
        return NFKMLXZImageConfiguration(
            inChannels: json["in_channels"] as? Int ?? base.inChannels, dim: json["dim"] as? Int ?? base.dim,
            layers: json["n_layers"] as? Int ?? base.layers,
            refinerLayers: json["n_refiner_layers"] as? Int ?? base.refinerLayers, heads: heads,
            normEps: (json["norm_eps"] as? NSNumber)?.floatValue ?? base.normEps,
            captionFeatureDim: json["cap_feat_dim"] as? Int ?? base.captionFeatureDim,
            ropeTheta: (json["rope_theta"] as? NSNumber)?.floatValue ?? base.ropeTheta,
            timestepScale: (json["t_scale"] as? NSNumber)?.floatValue ?? base.timestepScale,
            axesDims: json["axes_dims"] as? [Int] ?? base.axesDims)
    }

    /// Loads a release's `transformer/` weights, whose keys are the module's. The Turbo release stores
    /// float32 and loads at `dtype`.
    static func loadTransformer(into net: NFKMLXZImageTransformerNet, fromDirectory directory: URL,
                                dtype: DType) throws {
        try NFKMLXWeights.apply(try NFKMLXReleaseWeights.arrays(inDirectory: directory, converting: dtype),
                                to: net, verifyShapes: true)
    }

    /// The schedule a `scheduler/scheduler_config.json` states, over the ramp to sigma 0 the pipeline
    /// builds: a static shift (3 for Turbo, 6 for the base release), or the resolution-dependent one
    /// with the pipeline's own bounds where dynamic shifting is on.
    static func schedule(fromHuggingFace url: URL) throws -> NFKMLXFlowMatchConfiguration {
        let json = try NFKMLXWanRelease.json(url)
        let float = { (key: String, fallback: Float) in (json[key] as? NSNumber)?.floatValue ?? fallback }
        guard (json["use_dynamic_shifting"] as? Bool) == true else {
            var schedule = NFKMLXFlowMatchConfiguration.staticShiftToZero(float("shift", 1))
            schedule.trainTimesteps = json["num_train_timesteps"] as? Int ?? 1000
            return schedule
        }
        var schedule = NFKMLXFlowMatchConfiguration(
            trainTimesteps: json["num_train_timesteps"] as? Int ?? 1000, baseShift: float("base_shift", 0.5),
            maxShift: float("max_shift", 1.15), baseSequenceLength: json["base_image_seq_len"] as? Int ?? 256,
            maxSequenceLength: json["max_image_seq_len"] as? Int ?? 4096, shiftTerminal: nil,
            useDynamicShifting: true)
        schedule.rampEndsAtZero = true
        return schedule
    }
}

/// The text stage: the Qwen3 decoder read at its penultimate hidden state.
final class NFKZImageTextStage {
    let decoder: NFKMLXLanguageNet
    init(_ decoder: NFKMLXLanguageNet) { self.decoder = decoder }

    /// The caption features the transformer reads, `[tokens, hidden]`: the state the last layer reads,
    /// the reference's `hidden_states[-2]`, at the prompt's own tokens. The reference pads the batch on
    /// the right under a causal mask, so the unpadded encode is the same.
    func features(_ ids: [Int]) -> MLXArray {
        let states = decoder.layerStates(MLXArray(ids.map(Int32.init)).reshaped([1, ids.count]))
        return states[states.count - 2][0]
    }
}

/// Z-Image text-to-image, assembled from a diffusers release directory.
///
/// @discussion The release layout is `text_encoder/` (Qwen3-4B), `tokenizer/`, `transformer/`, `vae/`
/// (the Flux autoencoder), and `scheduler/`. The factories read `Tongyi-MAI/Z-Image-Turbo` and
/// `Tongyi-MAI/Z-Image`. ``image(forPrompt:negativePrompt:width:height:seed:)`` wraps the prompt in the
/// Qwen3 chat template with thinking enabled, reads the text encoder's penultimate hidden state,
/// denoises over the release's flow schedule, and decodes. The glue is measured against diffusers'
/// `ZImagePipeline`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXZImageGenerator)
public final class NFKMLXZImageGenerator: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "z-image"

    /// The Hugging Face repository the download factories read when none is named.
    @objc public static let releaseRepo = "Tongyi-MAI/Z-Image-Turbo"

    /// The caption length the reference pipeline truncates to.
    static let maximumSequenceLength = 512

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKZImageTextStage>
    private let pipelineStage: NFKMLXStage<NFKMLXZImagePipeline>
    private let tokenizer: NFKTokenizer

    /// The denoising steps. The default is Turbo's published setting, 9, of which the last lands on
    /// sigma 0 and evaluates nothing. The base release runs at the reference pipeline's 50.
    @objc public var steps: Int = 9
    /// The classifier-free guidance scale. Above 1 the image guides against the negative prompt, or an
    /// empty one where none is given. The default is Turbo's published 0; the base release runs at the
    /// reference pipeline's 5.
    @objc public var guidance: Float = 0

    /// Whether the text encoder and the transformer stay loaded together between images.
    @objc public var holdsStagesResident: Bool { staging.resident }

    init(resident: Bool, tokenizer: NFKTokenizer,
         loadTextEncoder: @escaping () throws -> NFKMLXLanguageNet,
         loadPipeline: @escaping () throws -> NFKMLXZImagePipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage { NFKZImageTextStage(try loadTextEncoder()) }
        pipelineStage = NFKMLXStage(loadPipeline)
        self.tokenizer = tokenizer
        super.init()
    }

    /// Loads both stages now, for a resident release.
    func loadResident() throws {
        try staging.use(textStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    var isHoldingTextEncoder: Bool { staging.exclusively { textStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    /// The token ids the text encoder reads: the prompt as the chat template's user turn, with the
    /// assistant turn opened and thinking left enabled, truncated to `maximumLength`.
    static func promptIds(_ prompt: String, tokenizer: NFKTokenizer,
                          maximumLength: Int = maximumSequenceLength) -> [Int] {
        let templated = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        return Array(tokenizer.encode(templated).map(\.intValue).prefix(maximumLength))
    }

    // MARK: Factories

    /// Assembles the model from a downloaded release directory, holding it as
    /// ``NFKMLXResidency/automatic`` decides.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXZImageGenerator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    ///
    /// @discussion The text encoder loads at the bfloat16 it is stored in, about 8 GB. The transformer
    /// loads at bfloat16, about 12 GB, the type the reference pipelines run it at; the Turbo release
    /// stores it at float32 and converts as it reads. The autoencoder loads as stored. A machine that
    /// cannot hold the two stages together stages under ``NFKMLXResidency/automatic``: a 32 GB Mac runs
    /// the release staged. The release has no routed experts, so ``NFKMLXResidency/paged`` holds it as
    /// ``NFKMLXResidency/staged`` does.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXZImageGenerator {
        let encoderDirectory = directoryURL.appendingPathComponent("text_encoder")
        let transformerDirectory = directoryURL.appendingPathComponent("transformer")
        let vaeDirectory = directoryURL.appendingPathComponent("vae")
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(
            inDirectory: directoryURL.appendingPathComponent("tokenizer")) else {
            throw NFKMLXError.unsupportedConfiguration("the release's tokenizer/ could not be read")
        }
        let encoderConfiguration = try NFKMLXLanguage.configuration(
            fromHuggingFace: encoderDirectory.appendingPathComponent("config.json"))
        let configuration = try NFKMLXZImageRelease.transformerConfiguration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json"))
        let vaeConfiguration = try NFKMLXStableDiffusionModels.vaeConfiguration(
            fromHuggingFace: vaeDirectory.appendingPathComponent("config.json"))
        let schedule = try NFKMLXZImageRelease.schedule(
            fromHuggingFace: directoryURL.appendingPathComponent("scheduler/scheduler_config.json"))

        let plan = try NFKMLXResidencyBudget.plan(
            [NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(inDirectory: encoderDirectory,
                                                                        precision: .checkpoint)),
             NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(inDirectory: transformerDirectory,
                                                                        holding: .bfloat16)
                                     + NFKMLXStageWeights.bytes(inDirectory: vaeDirectory, precision: .checkpoint))],
            residency: residency, budget: NFKMLXResidencyBudget.current())
        let generator = NFKMLXZImageGenerator(
            resident: plan.holdsStagesResident, tokenizer: tokenizer,
            loadTextEncoder: {
                let decoder = NFKMLXLanguage.makeNet(encoderConfiguration)
                try NFKMLXLanguage.loadWeights(into: decoder, fromDirectory: encoderDirectory, precision: .checkpoint)
                return decoder
            },
            loadPipeline: {
                let transformer = NFKMLXZImageTransformerNet(configuration)
                try NFKMLXZImageRelease.loadTransformer(into: transformer, fromDirectory: transformerDirectory,
                                                        dtype: .bfloat16)
                let vae = NFKMLXSDAutoencoder(configuration: vaeConfiguration)
                try NFKMLXStableDiffusionModels.loadVAEWeights(
                    into: vae, from: vaeDirectory.appendingPathComponent("diffusion_pytorch_model.safetensors"),
                    precision: .checkpoint)
                return NFKMLXZImagePipeline(transformer: transformer, vae: vae, schedule: schedule)
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
        NFKMLXReleaseComponent(required: ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"],
                               optional: ["tokenizer/vocab.json", "tokenizer/merges.txt"]),
        NFKMLXReleaseComponent(required: ["transformer/config.json"],
                               weights: ["transformer/diffusion_pytorch_model.safetensors",
                                         "transformer/diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]),
    ]

    /// Downloads the release (``releaseRepo`` when `repo` is nil) and assembles the model, holding it
    /// as `residency` says.
    ///
    /// @discussion The Turbo download is about 33 GB, its transformer stored at float32. A file already
    /// in the cache is not fetched again. The call blocks on the network, so run it off the main and
    /// render threads. The weights are under the Apache 2.0 license.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXZImageGenerator {
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
                                 completionHandler: @escaping (NFKMLXZImageGenerator?, Error?) -> Void) {
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

    /// The caption features the transformer reads, `[tokens, 2560]`. A staged release loads the text
    /// encoder for the call and releases it after.
    public func encode(prompt: String) throws -> MLXArray {
        try staging.with(textStage) { $0.features(Self.promptIds(prompt, tokenizer: tokenizer)) }
    }

    /// Generates an image for `prompt`, `[height, width, 3]` RGB in `[0, 1]`.
    ///
    /// @discussion `width` and `height` are multiples of 16, which the reference requires. Above a
    /// ``guidance`` of 1 the image guides against `negativePrompt`, or against an empty prompt where
    /// none is given; both prompts encode from one load of the text encoder.
    public func image(forPrompt prompt: String, negativePrompt: String? = nil, width: Int = 1024,
                      height: Int = 1024, seed: UInt64 = 0) throws -> MLXArray {
        guard width > 0, height > 0, width % 16 == 0, height % 16 == 0 else {
            throw NFKMLXError.unsupportedConfiguration("an image's sides are positive multiples of 16")
        }
        let guides = guidance > 1
        return try staging.exclusively {
            let prompts = guides ? [prompt, negativePrompt ?? ""] : [prompt]
            let features = try staging.with(textStage) { stage in
                prompts.map { stage.features(Self.promptIds($0, tokenizer: tokenizer)) }
            }
            return try staging.with(pipelineStage) { pipeline in
                let image = pipeline.generate(promptEmbeds: features[0], negativeEmbeds: guides ? features[1] : nil,
                                              latentHeight: height / pipeline.pixelsPerLatent,
                                              latentWidth: width / pipeline.pixelsPerLatent, steps: steps,
                                              guidance: guidance, seed: seed)
                return clip(image[0].asType(.float32) / 2 + 0.5, min: 0, max: 1)
            }
        }
    }

    /// Generates an image for `prompt` and returns it as a `CGImage`.
    @objc(imageForPrompt:negativePrompt:width:height:seed:error:)
    public func cgImage(forPrompt prompt: String, negativePrompt: String?, width: Int, height: Int,
                        seed: UInt64) throws -> CGImage {
        let array = try image(forPrompt: prompt, negativePrompt: negativePrompt, width: width, height: height,
                              seed: seed)
        return try NFKMLXImageBridge.cgImage(from: array, options: NFKMLXImageOptions())
    }
}
