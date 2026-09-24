//
//  NFKMLXFlux2Pipeline.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN
import MLXRandom

// The FLUX.2 text-to-image path, a prompt string in and an image out. The four stages are measured
// separately against diffusers — the transformer (`NFKMLXFlux2TransformerNet`), the autoencoder
// (`NFKMLXSDAutoencoder` in its `.flux2` preset) with its latent codec
// (`NFKMLXFlux2LatentCodec`), the text conditioning (`NFKMLXFlux2TextEncoder`), and the sigma
// schedule (`NFKMLXFlowMatchScheduler` in its `.flux2` preset) — and this chains them.
//
// The pipeline is FLUX.2 [klein]'s, whose text encoder is a Qwen3. FLUX.2 [dev] reads
// Mistral-Small 3, which `NFKMLXLanguageConfiguration.mistralSmall3` carries, but its repository is
// gated, so no [dev] release has been read here.

/// FLUX.2's denoising loop over a packed latent.
public final class NFKMLXFlux2Pipeline {
    public let transformer: NFKMLXFlux2TransformerNet
    public let autoencoder: NFKMLXSDAutoencoder
    public let codec: NFKMLXFlux2LatentCodec
    public var schedule: NFKMLXFlowMatchConfiguration

    public init(transformer: NFKMLXFlux2TransformerNet, autoencoder: NFKMLXSDAutoencoder,
                codec: NFKMLXFlux2LatentCodec, schedule: NFKMLXFlowMatchConfiguration = .flux2) {
        self.transformer = transformer
        self.autoencoder = autoencoder
        self.codec = codec
        self.schedule = schedule
    }

    /// Generates from a conditioning sequence. `promptEmbeds` is `[1, textTokens, jointAttentionDim]`
    /// and `negativeEmbeds` is supplied only where the caller wants classifier-free guidance, which
    /// the step-distilled klein releases do not use. `latentHeight` and `latentWidth` are in
    /// PATCHED units: the pixel size divided by the autoencoder's stride of 8 and again by the
    /// codec's 2×2 patch.
    public func generate(promptEmbeds: MLXArray, negativeEmbeds: MLXArray? = nil,
                         latentHeight: Int, latentWidth: Int, steps: Int = 28,
                         guidanceScale: Float = 4, seed: UInt64 = 0,
                         references: [MLXArray] = []) -> MLXArray {
        let channels = codec.runningMean.dim(0)
        MLXRandom.seed(seed)
        var latents = MLXRandom.normal([1, latentHeight * latentWidth, channels]).asType(latentType)
        let conditioning = referenceConditioning(references)
        let imageIds = concatenated(
            [NFKMLXFlux2TransformerNet.imageIds(height: latentHeight, width: latentWidth)]
                + conditioning.ids, axis: 0)

        var scheduler = NFKMLXFlowMatchScheduler(schedule)
        scheduler.setTimesteps(steps, sequenceLength: latentHeight * latentWidth)

        for step in 0 ..< steps {
            let predicted = velocity(latents, references: conditioning.tokens, imageIds: imageIds,
                                     promptEmbeds: promptEmbeds, negativeEmbeds: negativeEmbeds,
                                     timestep: scheduler.timesteps[step], guidanceScale: guidanceScale)
            latents = scheduler.step(velocity: predicted, sample: latents, index: step)
            eval(latents)
        }
        return decode(latents: latents, latentHeight: latentHeight, latentWidth: latentWidth)
    }

    /// The type the reference runs the latents and the conditioning in: the transformer's.
    private var latentType: DType { NFKReferenceRounding.parameterType(of: transformer) }

    /// Generates with FLUX.2 [klein] 9B KV's reference cache, as diffusers' `Flux2KleinKVPipeline` does.
    ///
    /// @discussion On the first step the reference latents LEAD the token sequence, and the transformer
    /// extracts every layer's reference keys and values
    /// (``NFKMLXFlux2TransformerNet/extractingReferences(_:referenceCount:encoderHidden:timestep:guidance:imageIds:textIds:referenceTimestep:)``);
    /// every later step runs the generated tokens alone against that cache, so the references cost
    /// one pass rather than one per step. Without references this is the plain loop. The weights must
    /// be trained for it: `FLUX.2-klein-9b-kv`'s are, and the same arithmetic over another release's
    /// weights computes a function that release was not trained for. There is no guidance, as the
    /// reference pipeline has none, and its default is 4 steps. `noise` is the packed starting latent
    /// `[1, tokens, channels]`, drawn from `seed` when nil.
    public func generateCachingReferences(promptEmbeds: MLXArray, latentHeight: Int, latentWidth: Int,
                                          steps: Int = 4, seed: UInt64 = 0, references: [MLXArray],
                                          referenceTimestep: Float = 0,
                                          noise: MLXArray? = nil) -> MLXArray {
        var latents: MLXArray
        if let noise {
            latents = noise.asType(latentType)
        } else {
            MLXRandom.seed(seed)
            latents = MLXRandom.normal([1, latentHeight * latentWidth, codec.runningMean.dim(0)]).asType(latentType)
        }
        let promptEmbeds = promptEmbeds.asType(latentType)
        let conditioning = referenceConditioning(references)
        let latentIds = NFKMLXFlux2TransformerNet.imageIds(height: latentHeight, width: latentWidth)
        let textIds = NFKMLXFlux2TransformerNet.textIds(length: promptEmbeds.dim(1))

        var scheduler = NFKMLXFlowMatchScheduler(schedule)
        scheduler.setTimesteps(steps, sequenceLength: latentHeight * latentWidth)

        var cache: NFKMLXFlux2ReferenceCache?
        for step in 0 ..< steps {
            let timestep = NFKReferenceRounding.flowFraction(scheduler.timesteps[step], dtype: latentType)
            let predicted: MLXArray
            if let cache {
                predicted = transformer(latents, encoderHidden: promptEmbeds, timestep: timestep,
                                        guidance: nil, imageIds: latentIds, textIds: textIds,
                                        referenceCache: cache)
            } else if !conditioning.tokens.isEmpty {
                let referenceTokens = concatenated(conditioning.tokens, axis: 1).asType(latentType)
                let extracted = transformer.extractingReferences(
                    concatenated([referenceTokens, latents], axis: 1),
                    referenceCount: referenceTokens.dim(1), encoderHidden: promptEmbeds,
                    timestep: timestep, guidance: nil,
                    imageIds: concatenated(conditioning.ids + [latentIds], axis: 0), textIds: textIds,
                    referenceTimestep: referenceTimestep)
                eval([extracted.velocity] + extracted.cache.arrays)
                cache = extracted.cache
                predicted = extracted.velocity
            } else {
                predicted = transformer(latents, encoderHidden: promptEmbeds, timestep: timestep,
                                        guidance: nil, imageIds: latentIds, textIds: textIds)
            }
            latents = scheduler.step(velocity: predicted, sample: latents, index: step)
            eval(latents)
        }
        return decode(latents: latents, latentHeight: latentHeight, latentWidth: latentWidth)
    }

    /// Repaints the masked part of an image and keeps the rest, as diffusers'
    /// `Flux2KleinInpaintPipeline` does.
    ///
    /// @discussion `image` is `[1, H, W, 3]` in `[-1, 1]` and `mask` is `[1, H, W, 1]`, where 1
    /// repaints and 0 keeps; H and W are multiples of the autoencoder's stride times the codec's patch
    /// (16 for a release). The mask is binarized at 0.5 and resampled bilinearly to the packed grid,
    /// so its edge cells blend the two regions.
    ///
    /// The image conditions the generation twice. Its latent is the first reference, at time
    /// coordinate 10, which is how FLUX.2 edits. It is also the starting point: the loop begins
    /// `strength` of the way into the schedule from the image noised to that step's sigma, and after
    /// every step the kept region is overwritten with the image noised to the NEXT sigma, with the
    /// noise drawn at the start, so the repainted region denoises against context at its own noise
    /// level. The last step blends with the clean image.
    ///
    /// `strength` is a `Double` because the reference computes the start step as
    /// `int(steps - steps · strength)` in double precision, and a single-precision strength widens to
    /// a different product. At the reference's own defaults, 50 steps and 0.8, the loop starts at
    /// step 10; a `Float` 0.8 widens to 0.80000001 and starts at step 9, one denoising step more.
    /// `noise` is the packed starting noise `[1, tokens, channels]`, drawn from `seed` when nil.
    /// `references` are further autoencoder latents, after the image's own.
    public func inpaint(promptEmbeds: MLXArray, negativeEmbeds: MLXArray? = nil, image: MLXArray,
                        mask: MLXArray, strength: Double = 0.8, steps: Int = 28,
                        guidanceScale: Float = 4, seed: UInt64 = 0, references: [MLXArray] = [],
                        noise: MLXArray? = nil) throws -> MLXArray {
        let stride = (1 << (autoencoder.configuration.blockChannels.count - 1)) * codec.patch
        guard image.dim(1) % stride == 0, image.dim(2) % stride == 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "inpainting takes an image whose sides are multiples of \(stride); this one is "
                + "\(image.dim(2))x\(image.dim(1))")
        }
        guard mask.dim(1) == image.dim(1), mask.dim(2) == image.dim(2) else {
            throw NFKMLXError.unsupportedConfiguration(
                "the mask is \(mask.dim(2))x\(mask.dim(1)) and the image \(image.dim(2))x\(image.dim(1))")
        }
        let start = Int(max(Double(steps) - min(Double(steps) * strength, Double(steps)), 0))
        guard start < steps else {
            throw NFKMLXError.unsupportedConfiguration(
                "a strength of \(strength) over \(steps) steps leaves no step to denoise")
        }

        let encoded = autoencoder.encode(image).mean
        let latentHeight = encoded.dim(1) / codec.patch, latentWidth = encoded.dim(2) / codec.patch
        let conditioning = referenceConditioning([encoded] + references)
        let original = conditioning.tokens[0].asType(latentType)
        let imageIds = concatenated(
            [NFKMLXFlux2TransformerNet.imageIds(height: latentHeight, width: latentWidth)]
                + conditioning.ids, axis: 0)
        let repaint = NFKMLXResample.resizeBilinear(
            (mask .>= MLXArray(Float(0.5))).asType(.float32), height: latentHeight, width: latentWidth)
            .reshaped([1, latentHeight * latentWidth, 1])

        var scheduler = NFKMLXFlowMatchScheduler(schedule)
        scheduler.setTimesteps(steps, sequenceLength: latentHeight * latentWidth)
        let startNoise: MLXArray
        if let noise {
            startNoise = noise.asType(latentType)
        } else {
            MLXRandom.seed(seed)
            startNoise = MLXRandom.normal(original.shape).asType(latentType)
        }
        func noised(atStep step: Int) -> MLXArray {
            let sigma = scheduler.sigmas[step]
            return NFKReferenceRounding.scaled(startNoise, by: sigma) + NFKReferenceRounding.scaled(original, by: 1 - sigma)
        }

        var latents = noised(atStep: start)
        for step in start ..< steps {
            let predicted = velocity(latents, references: conditioning.tokens, imageIds: imageIds,
                                     promptEmbeds: promptEmbeds, negativeEmbeds: negativeEmbeds,
                                     timestep: scheduler.timesteps[step], guidanceScale: guidanceScale)
            latents = scheduler.step(velocity: predicted, sample: latents, index: step)
            let kept = step < steps - 1 ? noised(atStep: step + 1) : original
            let blend = repaint.asType(latents.dtype)
            latents = (1 - blend) * kept + blend * latents
            eval(latents)
        }
        return decode(latents: latents, latentHeight: latentHeight, latentWidth: latentWidth)
    }

    /// The packed tokens and rotary ids of reference latents. Each reference is appended to the
    /// token sequence under its own time coordinate, so the rotary tells it from the generated image
    /// and from the other references.
    private func referenceConditioning(_ latents: [MLXArray]) -> (tokens: [MLXArray], ids: [MLXArray]) {
        var tokens = [MLXArray](), ids = [MLXArray]()
        for (index, latent) in latents.enumerated() {
            tokens.append(codec.encode(latent: latent))
            ids.append(NFKMLXFlux2TransformerNet.referenceImageIds(
                height: latent.dim(1) / codec.patch, width: latent.dim(2) / codec.patch, index: index))
        }
        return (tokens, ids)
    }

    /// The transformer's velocity for the generated tokens, with classifier-free guidance where
    /// `negativeEmbeds` is supplied. The references ride along in the sequence and are cut from the
    /// output. `timestep` is on the schedule's thousand scale and is passed as a fraction.
    private func velocity(_ latents: MLXArray, references: [MLXArray], imageIds: MLXArray,
                          promptEmbeds: MLXArray, negativeEmbeds: MLXArray?, timestep: Float,
                          guidanceScale: Float) -> MLXArray {
        let generated = latents.dim(1)
        let references = references.map { $0.asType(latentType) }
        let input = references.isEmpty ? latents : concatenated([latents] + references, axis: 1)
        let fraction = NFKReferenceRounding.flowFraction(timestep, dtype: latentType)
        func predict(_ embeds: MLXArray) -> MLXArray {
            let output = transformer(input, encoderHidden: embeds, timestep: fraction, guidance: nil,
                                     imageIds: imageIds,
                                     textIds: NFKMLXFlux2TransformerNet.textIds(length: embeds.dim(1)))
            return output[0..., 0 ..< generated, 0...]
        }
        let conditional = predict(promptEmbeds.asType(latentType))
        guard let negativeEmbeds else {
            return conditional
        }
        let unconditional = predict(negativeEmbeds.asType(latentType))
        return unconditional + guidanceScale * (conditional - unconditional)
    }

    /// The token sequence back to pixels: unpack, unwhiten, unpatchify, decode, and map the
    /// autoencoder's `[-1, 1]` to `[0, 1]`.
    public func decode(latents: MLXArray, latentHeight: Int, latentWidth: Int) -> MLXArray {
        let latent = codec.decode(tokens: latents, height: latentHeight, width: latentWidth)
        let image = autoencoder.decode(latent)
        return clip((image + 1) / 2, min: 0, max: 1)
    }
}

/// Building and running FLUX.2 [klein] text-to-image from a downloaded diffusers release directory.
///
/// The release layout is the diffusers one: `transformer/`, `vae/`, `text_encoder/` (a Qwen3),
/// `tokenizer/` (its byte-level BPE and chat template), and `scheduler/`. `image(forPrompt:)` runs
/// the prompt through the release's own chat template, encodes it, denoises, and decodes.
@objc(NFKMLXFlux2)
public final class NFKMLXFlux2: NSObject {

    /// A name for the model the factory produces.
    @objc public static let modelName = "flux.2-klein-4b"

    private let staging: NFKMLXStagedModel
    private let textEncoderStage: NFKMLXStage<NFKMLXFlux2TextEncoder>
    private let pipelineStage: NFKMLXStage<NFKMLXFlux2Pipeline>
    private let tokenizer: NFKTokenizer
    private let chatTemplate: String?

    /// The default number of denoising steps.
    @objc public var steps: Int = 28
    /// The classifier-free guidance scale. It applies where the release is not step distilled and
    /// the scale is above 1, which is the reference pipelines' own rule; a distilled release
    /// generates without guidance whatever this is set to.
    @objc public var guidance: Float = 4
    /// Whether the release is step distilled (`is_distilled` in its `model_index.json`). A distilled
    /// release (FLUX.2 [klein] 4B) ignores ``guidance`` and a negative prompt; the base releases
    /// guide, against an empty negative prompt when none is given.
    @objc public let isDistilled: Bool
    /// Whether generation with references uses FLUX.2 [klein] 9B KV's reference cache
    /// (``NFKMLXFlux2Pipeline/generateCachingReferences(promptEmbeds:latentHeight:latentWidth:steps:seed:references:referenceTimestep:noise:)``).
    /// Set it for `FLUX.2-klein-9b-kv`, whose weights are trained for it. Nothing in that release's
    /// files marks it: its `model_index.json` names the plain `Flux2KleinPipeline` and its transformer
    /// config is the base 9B's. It changes nothing without references, and inpainting keeps the plain
    /// path, since diffusers has no reference-cache inpainting pipeline.
    @objc public var cachesReferences = false

    /// Whether the text encoder and the transformer stay loaded together between images. False where
    /// the release is staged: each image then loads the encoder, encodes, releases it, and loads the
    /// transformer.
    @objc public var holdsStagesResident: Bool { staging.resident }
    /// Whether the text encoder runs at float32 rather than at its stored precision. It does where the
    /// encoder fits the working set at float32 in the placement chosen; FLUX.2 [klein] 9B's Qwen3-8B
    /// (30.5 GB at float32) runs as stored.
    @objc public let encodesInFloat32: Bool

    init(resident: Bool, loadTextEncoder: @escaping () throws -> NFKMLXFlux2TextEncoder,
         loadPipeline: @escaping () throws -> NFKMLXFlux2Pipeline, tokenizer: NFKTokenizer,
         chatTemplate: String?, isDistilled: Bool, encodesInFloat32: Bool) {
        self.staging = NFKMLXStagedModel(resident: resident)
        self.textEncoderStage = NFKMLXStage(loadTextEncoder)
        self.pipelineStage = NFKMLXStage(loadPipeline)
        self.tokenizer = tokenizer
        self.chatTemplate = chatTemplate
        self.isDistilled = isDistilled
        self.encodesInFloat32 = encodesInFloat32
        super.init()
    }

    /// A facade over components already loaded, held resident.
    convenience init(pipeline: NFKMLXFlux2Pipeline, textEncoder: NFKMLXFlux2TextEncoder,
                     tokenizer: NFKTokenizer, chatTemplate: String?, isDistilled: Bool = true,
                     encodesInFloat32: Bool = true) {
        self.init(resident: true, loadTextEncoder: { textEncoder }, loadPipeline: { pipeline },
                  tokenizer: tokenizer, chatTemplate: chatTemplate, isDistilled: isDistilled,
                  encodesInFloat32: encodesInFloat32)
    }

    /// Loads both stages now, for a resident release: a bad release fails at construction, as it
    /// always has, rather than at the first image.
    func loadResident() throws {
        try staging.use(textEncoderStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    var isHoldingTextEncoder: Bool { staging.exclusively { textEncoderStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    /// Assembles the whole model from a downloaded diffusers FLUX.2 release directory, holding it as
    /// ``NFKMLXResidency/automatic`` decides.
    @objc(flux2WithDirectoryURL:error:)
    public static func flux2(directoryURL: URL) throws -> NFKMLXFlux2 {
        try flux2(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the whole model from a downloaded diffusers FLUX.2 release directory.
    ///
    /// @discussion The text encoder runs once per image and the transformer on every step, so the two
    /// never need to be loaded together. A resident release holds both and loads them here; a staged
    /// one loads nothing here and, for each image, loads the encoder, encodes, releases it, then loads
    /// the transformer. `residency` decides which; ``NFKMLXResidency/automatic`` plans against the
    /// machine's working set, which is how FLUX.2 [klein] 9B (a 15.3 GB encoder and a
    /// 17 GB transformer) runs on a machine that cannot hold the two at once. Introduced in InferKit
    /// 0.4.0.
    @objc(flux2WithDirectoryURL:residency:error:)
    public static func flux2(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXFlux2 {
        let transformerDirectory = directoryURL.appendingPathComponent("transformer")
        let vaeDirectory = directoryURL.appendingPathComponent("vae")
        let encoderDirectory = directoryURL.appendingPathComponent("text_encoder")
        let tokenizerDirectory = directoryURL.appendingPathComponent("tokenizer")

        let configuration = try NFKMLXFlux2TransformerNet.configuration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json"))
        let encoderConfiguration = try NFKMLXLanguage.configuration(
            fromHuggingFace: encoderDirectory.appendingPathComponent("config.json"))
        // The text conditioning is three layers of the encoder concatenated. The release reads a
        // thirty-six layer Qwen3 at 9, 18 and 27; the indices scale with the encoder's depth, and the
        // transformer's own `jointAttentionDim` is what says how many layers it expects.
        let hidden = encoderConfiguration.hiddenSize
        let read = configuration.jointAttentionDim / hidden
        guard read > 0, configuration.jointAttentionDim % hidden == 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "the transformer reads a \(configuration.jointAttentionDim)-wide text sequence, which "
                + "is not a whole number of this encoder's \(hidden)-wide layers")
        }
        let layers = (1 ... read).map { encoderConfiguration.layerCount * $0 / (read + 1) }

        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: tokenizerDirectory) else {
            throw NFKMLXError.unsupportedConfiguration("the release's tokenizer/ could not be read")
        }
        let template = try? String(contentsOf: tokenizerDirectory.appendingPathComponent("chat_template.jinja"),
                                   encoding: .utf8)

        let placement = try plan(
            encoderStoredBytes: try NFKMLXReleaseWeights.weightBytes(inDirectory: encoderDirectory),
            pipelineStoredBytes: try NFKMLXReleaseWeights.weightBytes(inDirectory: transformerDirectory)
                + NFKMLXReleaseWeights.weightBytes(inDirectory: vaeDirectory),
            budget: NFKMLXResidencyBudget.current(), residency: residency)
        let flux = NFKMLXFlux2(
            resident: placement.resident,
            loadTextEncoder: {
                let (decoder, _) = try NFKMLXLanguage.loadedRelease(
                    at: encoderDirectory, precision: placement.encoderFloat32 ? .float32 : .checkpoint)
                return NFKMLXFlux2TextEncoder(decoder: decoder, layers: layers)
            },
            loadPipeline: {
                let transformer = NFKMLXFlux2TransformerNet(configuration)
                try NFKMLXFlux2TransformerNet.loadWeights(into: transformer, from: transformerDirectory,
                                                          precision: .checkpoint)
                let autoencoder = NFKMLXSDAutoencoder(configuration: .flux2)
                try NFKMLXStableDiffusionModels.loadVAEWeights(
                    into: autoencoder,
                    from: vaeDirectory.appendingPathComponent("diffusion_pytorch_model.safetensors"),
                    precision: .checkpoint)
                return NFKMLXFlux2Pipeline(transformer: transformer, autoencoder: autoencoder,
                                           codec: try NFKMLXFlux2LatentCodec.codec(fromReleaseDirectory: vaeDirectory))
            },
            tokenizer: tokenizer, chatTemplate: template,
            isDistilled: isDistilled(releaseDirectory: directoryURL),
            encodesInFloat32: placement.encoderFloat32)
        if placement.resident {
            try flux.loadResident()
        }
        return flux
    }

    /// How a release is held: whether the text encoder and the transformer stay loaded together, and
    /// whether the encoder runs at float32 or as stored.
    ///
    /// @discussion Every placement keeps ``NFKMLXResidencyBudget``'s reserve against `budget`, the rule
    /// every staged model here plans by. A resident placement prefers a
    /// float32 encoder and falls back to the stored precision, which is diffusers' own default of a
    /// bfloat16 pipeline; a staged one takes float32 where the encoder alone fits it. ``automatic``
    /// takes the resident placement where one is known to fit and the staged one otherwise, so a machine
    /// that reports no budget is staged with its encoder as stored. On a 32 GB machine (a 21.25 GiB
    /// budget) klein 4B is resident with a bfloat16 encoder, and klein 9B is staged with one, its 17 GB
    /// transformer stage 0.2 GiB inside the budget.
    static func plan(encoderStoredBytes encoder: Int, pipelineStoredBytes pipeline: Int, budget: Int,
                     residency: NFKMLXResidency) throws -> (resident: Bool, encoderFloat32: Bool) {
        let reserve = NFKMLXResidencyBudget.reserve
        let gib = NFKMLXResidencyBudget.gib
        func holds(_ bytes: Int) -> Bool { NFKMLXResidencyBudget.holds(bytes, budget: budget) }
        func admits(_ bytes: Int) -> Bool { NFKMLXResidencyBudget.admits(bytes, budget: budget) }
        switch residency {
        case .resident:
            if holds(2 * encoder + pipeline) {
                return (true, true)
            }
            if admits(encoder + pipeline) {
                return (true, false)
            }
            throw NFKMLXError.unsupportedConfiguration(
                "the text encoder and transformer need \(gib(encoder + pipeline)) together, plus a "
                + "\(gib(reserve)) reserve, against a \(gib(budget)) working set; load it staged")
        case .staged, .paged:
            // Neither stage has routed experts, so a paged release is held as a staged one.
            guard admits(pipeline) else {
                throw NFKMLXError.unsupportedConfiguration(
                    "the transformer alone needs \(gib(pipeline)) plus a \(gib(reserve)) reserve, "
                    + "against a \(gib(budget)) working set")
            }
            if holds(2 * encoder) {
                return (false, true)
            }
            if admits(encoder) {
                return (false, false)
            }
            throw NFKMLXError.unsupportedConfiguration(
                "the text encoder alone needs \(gib(encoder)) plus a \(gib(reserve)) reserve, "
                + "against a \(gib(budget)) working set")
        case .automatic:
            if holds(encoder + pipeline) {
                return try plan(encoderStoredBytes: encoder, pipelineStoredBytes: pipeline,
                                budget: budget, residency: .resident)
            }
            return try plan(encoderStoredBytes: encoder, pipelineStoredBytes: pipeline, budget: budget,
                            residency: .staged)
        }
    }

    /// The component folders ``flux2(directoryURL:)`` reads, as a download fetches them.
    /// `model_index.json` is optional to the build, which then reads the release as not distilled.
    static let releaseComponents = [
        NFKMLXReleaseComponent(required: [], optional: ["model_index.json"]),
        NFKMLXReleaseComponent(required: ["transformer/config.json"],
                               weights: ["transformer/diffusion_pytorch_model.safetensors",
                                         "transformer/diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["vae/diffusion_pytorch_model.safetensors"]),
        NFKMLXReleaseComponent(required: ["text_encoder/config.json"],
                               weights: ["text_encoder/model.safetensors",
                                         "text_encoder/model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"],
                               optional: ["tokenizer/vocab.json", "tokenizer/merges.txt",
                                          "tokenizer/added_tokens.json", "tokenizer/chat_template.jinja"]),
    ]

    /// Downloads the release's files into the hub cache and returns the snapshot directory.
    static func releaseDirectory(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> URL {
        try NFKMLXReleaseDownload.directory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                            components: releaseComponents)
    }

    /// Downloads a diffusers FLUX.2 [klein] release and assembles the model.
    ///
    /// @discussion The download is `model_index.json`, the transformer, the autoencoder, the Qwen3
    /// text encoder, and its tokenizer, about 16 GB for `black-forest-labs/FLUX.2-klein-4B` and about
    /// 35 GB for a 9B release. A file already in the cache is not fetched again. The call blocks on
    /// the network, so run it off the main and render threads. It serves
    /// `black-forest-labs/FLUX.2-klein-4B` and `black-forest-labs/FLUX.2-klein-base-4B`, which are
    /// public, and `black-forest-labs/FLUX.2-klein-9B`, `FLUX.2-klein-base-9B`, and
    /// `FLUX.2-klein-9b-kv`, which are gated: for those the caller accepts the license on Hugging
    /// Face and sets `NFKHFHub.defaultAccessToken` before the first download. Introduced in InferKit
    /// 0.4.0.
    @objc(flux2WithRepo:revision:cacheDirectoryURL:error:)
    public static func flux2(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXFlux2 {
        try flux2(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: .automatic)
    }

    /// ``flux2(repo:revision:cacheDirectoryURL:)`` held as `residency` says. Introduced in InferKit 0.4.0.
    @objc(flux2WithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func flux2(repo: String, revision: String?, cacheDirectoryURL: URL?,
                             residency: NFKMLXResidency) throws -> NFKMLXFlux2 {
        try flux2(directoryURL: try releaseDirectory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL),
                  residency: residency)
    }

    /// The asynchronous form of ``flux2(repo:revision:cacheDirectoryURL:)``. The handler runs on a
    /// background queue. Introduced in InferKit 0.4.0.
    @objc(flux2WithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func flux2(repo: String, revision: String?, cacheDirectoryURL: URL?,
                             completionHandler: @escaping (NFKMLXFlux2?, Error?) -> Void) {
        flux2(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: .automatic,
              completionHandler: completionHandler)
    }

    /// The asynchronous form of ``flux2(repo:revision:cacheDirectoryURL:residency:)``. The handler runs
    /// on a background queue. Introduced in InferKit 0.4.0.
    @objc(flux2WithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func flux2(repo: String, revision: String?, cacheDirectoryURL: URL?,
                             residency: NFKMLXResidency,
                             completionHandler: @escaping (NFKMLXFlux2?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try flux2(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                            residency: residency), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// The prompt as the release's own chat template renders it, tokenized.
    ///
    /// @discussion The reference wraps the prompt in a single user message, asks for the generation
    /// prompt, and passes `enable_thinking=False`, which makes a Qwen3 template append an EMPTY
    /// think block. A port that skipped the template would feed the encoder a bare prompt and read
    /// different hidden states, so the release's own `chat_template.jinja` is rendered rather than
    /// the wrapping being reimplemented.
    public func tokens(forPrompt prompt: String) throws -> [Int] {
        guard let chatTemplate else {
            return tokenizer.encode(prompt).map(\.intValue)
        }
        let text = try NFKMLXChatTemplateRenderer.render(
            chatTemplate, messages: [["role": "user", "content": prompt]],
            addGenerationPrompt: true, variables: ["enable_thinking": false])
        return tokenizer.encode(text).map(\.intValue)
    }

    /// The conditioning for a prompt: the chat template, the tokenizer, and the three-layer read. A
    /// staged release loads the text encoder for the call and releases it after.
    public func encode(prompt: String) throws -> MLXArray {
        let tokens = try tokens(forPrompt: prompt)
        return try staging.with(textEncoderStage) { [$0.encode(tokens: tokens)] }[0]
    }

    /// The prompt's conditioning and, where the release guides, the negative prompt's, from one load of
    /// the text encoder. The reference's negative prompt defaults to empty.
    private func conditioning(_ prompt: String, negativePrompt: String?) throws -> (MLXArray, MLXArray?) {
        let guides = Self.guides(isDistilled: isDistilled, guidance: guidance)
        let tokens = try (guides ? [prompt, negativePrompt ?? ""] : [prompt]).map { try self.tokens(forPrompt: $0) }
        let embeds = try staging.with(textEncoderStage) { encoder in tokens.map { encoder.encode(tokens: $0) } }
        return (embeds[0], guides ? embeds[1] : nil)
    }

    /// Generates an image for `prompt`, `[1, H, W, 3]` in `[0, 1]`. `width` and `height` are pixels
    /// and are multiples of 16: the autoencoder strides by 8 and the codec packs 2×2.
    ///
    /// `references` are images the generation is conditioned on, each `[1, H, W, 3]` in `[0, 1]`,
    /// which is how FLUX.2 edits and takes a subject rather than only a prompt. Each is encoded and
    /// appended to the token sequence under its own time coordinate.
    public func image(forPrompt prompt: String, negativePrompt: String? = nil,
                      width: Int = 1024, height: Int = 1024, seed: UInt64 = 0,
                      references: [MLXArray] = []) throws -> MLXArray {
        try staging.exclusively {
            let (embeds, negative) = try conditioning(prompt, negativePrompt: negativePrompt)
            return try staging.with(pipelineStage) { pipeline in
                let referenceLatents = references.map { pipeline.autoencoder.encode($0 * 2 - 1).mean }
                if cachesReferences, !referenceLatents.isEmpty {
                    return pipeline.generateCachingReferences(promptEmbeds: embeds, latentHeight: height / 16,
                                                              latentWidth: width / 16, steps: steps,
                                                              seed: seed, references: referenceLatents)
                }
                return pipeline.generate(promptEmbeds: embeds, negativeEmbeds: negative,
                                         latentHeight: height / 16, latentWidth: width / 16,
                                         steps: steps, guidanceScale: guidance, seed: seed,
                                         references: referenceLatents)
            }
        }
    }

    /// Repaints the masked part of `image` from `prompt` and keeps the rest, `[1, H, W, 3]` in
    /// `[0, 1]`.
    ///
    /// @discussion `image` is `[1, H, W, 3]` in `[0, 1]` and `mask` is `[1, H, W, 1]`, where 1
    /// repaints; H and W are multiples of 16. `strength` is how far into the schedule the loop starts
    /// from the image: 1 regenerates the masked region from noise, and lower values keep more of what
    /// was there. `references` are further images to condition on, as in
    /// ``image(forPrompt:negativePrompt:width:height:seed:references:)``. The step count and guidance
    /// are ``steps`` and ``guidance``; diffusers' inpainting pipeline defaults its guidance to 8 where
    /// its text-to-image pipeline defaults to 4.
    public func inpaint(prompt: String, negativePrompt: String? = nil, image: MLXArray,
                        mask: MLXArray, strength: Double = 0.8, seed: UInt64 = 0,
                        references: [MLXArray] = []) throws -> MLXArray {
        try staging.exclusively {
            let (embeds, negative) = try conditioning(prompt, negativePrompt: negativePrompt)
            return try staging.with(pipelineStage) { pipeline in
                let referenceLatents = references.map { pipeline.autoencoder.encode($0 * 2 - 1).mean }
                return try pipeline.inpaint(promptEmbeds: embeds, negativeEmbeds: negative,
                                            image: image * 2 - 1, mask: mask, strength: strength,
                                            steps: steps, guidanceScale: guidance, seed: seed,
                                            references: referenceLatents)
            }
        }
    }

    /// Generates an image for `prompt` and returns it as a `CGImage`.
    @objc(imageForPrompt:negativePrompt:width:height:seed:error:)
    public func cgImage(forPrompt prompt: String, negativePrompt: String?, width: Int, height: Int,
                        seed: UInt64) throws -> CGImage {
        try cgImage(forPrompt: prompt, negativePrompt: negativePrompt, references: [], width: width,
                    height: height, seed: seed)
    }

    /// Generates an image for `prompt` conditioned on `references` and returns it as a `CGImage`,
    /// which is how FLUX.2 edits an image or carries a subject.
    ///
    /// @discussion Each reference is a `CGImage` or an `MTLTexture`; Objective-C passes a
    /// `CGImageRef` as `(__bridge id)image`, since an array of a Core Foundation type is not an
    /// Objective-C collection. Each is resampled to a multiple of 16 on each side, the size the
    /// reference pipeline conditions at; the resampling is bilinear where diffusers' is Lanczos.
    @objc(imageForPrompt:negativePrompt:references:width:height:seed:error:)
    public func cgImage(forPrompt prompt: String, negativePrompt: String?, references: [Any],
                        width: Int, height: Int, seed: UInt64) throws -> CGImage {
        let arrays = try references.map { try Self.pixels($0) }
        let array = try image(forPrompt: prompt, negativePrompt: negativePrompt, width: width,
                              height: height, seed: seed, references: arrays)
        return try NFKMLXImageBridge.cgImage(from: array[0], options: NFKMLXImageOptions())
    }

    /// Repaints the white part of `mask` over `image` from `prompt` and returns a `CGImage` the size
    /// of `image` rounded down to a multiple of 16 on each side. The mask is read as luminance with
    /// the ITU-R 601-2 weights of PIL's `L` conversion, which is how the reference reads one, and
    /// binarized at one half.
    @objc(inpaintImage:mask:prompt:negativePrompt:strength:seed:error:)
    public func cgInpaint(image: CGImage, mask: CGImage, prompt: String, negativePrompt: String?,
                          strength: Double, seed: UInt64) throws -> CGImage {
        let height = image.height / 16 * 16, width = image.width / 16 * 16
        guard height > 0, width > 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "inpainting needs an image at least 16 pixels on each side")
        }
        let pixels = try Self.pixels(image, height: height, width: width)
        let weights = MLXArray([Float(0.299), 0.587, 0.114]).reshaped([1, 1, 1, 3])
        let luminance = (try Self.pixels(mask, height: height, width: width) * weights)
            .sum(axis: -1, keepDims: true)
        let array = try inpaint(prompt: prompt, negativePrompt: negativePrompt, image: pixels,
                                mask: luminance, strength: strength, seed: seed)
        return try NFKMLXImageBridge.cgImage(from: array[0], options: NFKMLXImageOptions())
    }

    /// The reference pipelines' guidance rule: a release guides where it is not step distilled and
    /// the scale is above 1.
    static func guides(isDistilled: Bool, guidance: Float) -> Bool {
        !isDistilled && guidance > 1
    }

    /// Whether a release directory's `model_index.json` marks it step distilled. The flag is the
    /// pipeline's; an absent file or flag is the reference's default of false, under which a
    /// release guides.
    static func isDistilled(releaseDirectory: URL) -> Bool {
        let index = (try? Data(contentsOf: releaseDirectory.appendingPathComponent("model_index.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        return (index?["is_distilled"] as? Bool) ?? false
    }

    /// An image (a `CGImage` or an `MTLTexture`) as `[1, H, W, 3]` in `[0, 1]`, resampled bilinearly
    /// to `height` × `width`, or to its own size rounded down to a multiple of 16 where those are nil.
    private static func pixels(_ image: Any, height: Int? = nil, width: Int? = nil) throws -> MLXArray {
        let rgb = try NFKMLXImageBridge.tensor(from: image, channels: 3,
                                               colorSpace: CGColorSpaceCreateDeviceRGB())
        let (sourceHeight, sourceWidth) = (rgb.dim(0), rgb.dim(1))
        let targetHeight = height ?? sourceHeight / 16 * 16
        let targetWidth = width ?? sourceWidth / 16 * 16
        let batched = rgb.reshaped([1, sourceHeight, sourceWidth, 3])
        guard targetHeight != sourceHeight || targetWidth != sourceWidth else {
            return batched
        }
        return NFKMLXResample.resizeBilinear(batched, height: targetHeight, width: targetWidth)
    }
}
