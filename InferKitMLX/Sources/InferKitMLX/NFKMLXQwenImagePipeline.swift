//
//  NFKMLXQwenImagePipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The Qwen-Image 2.1 text-to-image pipeline: it chains the vision-language text encoder, the
// block-causal DiT denoised over a flow schedule, and the single-frame VAE into an image. Each stage is
// validated at reference parity on its own; this is the glue — the prompt template, the latent packing,
// the sampling loop, true classifier-free guidance, and the latent normalization across the DiT and VAE.
//
// Two details of the glue are the model's rather than the pipeline's. The text encoder's features are
// read BEFORE its final normalization, which is what the transformer was trained on. And the joint
// sequence's image mask is the caption's own mask with one slot appended per 2x2 group of target
// latents, because the transformer expands every image slot four-fold.

/// Holds the pipeline stages for capture across an async boundary.
private final class NFKQwenImagePipelineHolder: @unchecked Sendable {
    let transformer: NFKMLXQwenImageNet
    let vae: NFKMLXWanVideoVAENet
    init(_ transformer: NFKMLXQwenImageNet, _ vae: NFKMLXWanVideoVAENet) {
        self.transformer = transformer
        self.vae = vae
    }
}

/// The Qwen-Image 2.1 text-to-image pipeline.
public final class NFKMLXQwenImagePipeline {

    /// The system turn the release conditions on.
    public static let systemPrompt = "Comprehend and analyze the provided prompt."

    /// The VAE compresses 16x spatially and the transformer consumes latents unpatched, so one latent
    /// token covers a 16x16 pixel tile.
    public static let scaleFactor = 16

    private let holder: NFKQwenImagePipelineHolder
    private let latentMean: MLXArray?
    private let latentStandardDeviation: MLXArray?
    private let schedule: NFKMLXFlowMatchConfiguration

    /// Builds a pipeline from the transformer and the autoencoder.
    ///
    /// `latentMean` and `latentStandardDeviation` are the release's per-channel latent statistics,
    /// shaped to broadcast over an NDHWC latent (``NFKMLXQwenImageVAE/latentStatistics(fromHuggingFace:)``).
    /// A nil pair passes the latent to the autoencoder unchanged, which is what a test at a tiny
    /// configuration does.
    public init(transformer: NFKMLXQwenImageNet, vae: NFKMLXWanVideoVAENet,
                latentMean: MLXArray? = nil, latentStandardDeviation: MLXArray? = nil,
                schedule: NFKMLXFlowMatchConfiguration = .qwenImage21) {
        self.holder = NFKQwenImagePipelineHolder(transformer, vae)
        self.latentMean = latentMean
        self.latentStandardDeviation = latentStandardDeviation
        self.schedule = schedule
    }

    /// The prompt the text encoder reads, which is a raw template rather than the chat template's
    /// rendering of it: the two tokenize differently and the release expects this one.
    public static func promptText(_ prompt: String) -> String {
        // An empty prompt leaves a Qwen encoder with nothing to read, because it has no beginning token.
        let body = prompt.isEmpty ? " " : prompt
        return "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\n\(body)<|im_end|>\n"
            + "<|im_start|>assistant\n"
    }

    /// How many leading tokens of the encoded prompt the system turn occupies, which the pipeline drops
    /// from the hidden states.
    public static func systemTokenCount(tokenizer: NFKTokenizer) -> Int {
        tokenizer.encode("<|im_start|>system\n\(systemPrompt)<|im_end|>\n").count
    }

    /// The prompt features the transformer reads: the text encoder's last layer BEFORE its final
    /// normalization, with the system turn's leading tokens dropped.
    ///
    /// @discussion The un-normalized states are what the transformer was trained on, and the reference
    /// neutralizes the encoder's final norm with a forward hook to obtain them. Taking the normalized
    /// states instead leaves a pipeline that runs and renders text badly.
    public static func promptEmbeddings(_ prompt: String, decoder: NFKMLXLanguageNet,
                                        tokenizer: NFKTokenizer) -> MLXArray {
        let ids = tokenizer.encode(promptText(prompt)).map(\.intValue)
        let hidden = NFKMLXQwen3VL.hiddenStates(decoder: decoder, inputIds: ids, visionFeatures: nil,
                                                deepstack: [], gridT: 1, gridH: 1, gridW: 1,
                                                applyFinalNorm: false)
        let drop = systemTokenCount(tokenizer: tokenizer)
        return hidden[0, drop...]
    }

    /// Packs a latent `[1, 1, height, width, channels]` into the transformer's token sequence
    /// `[height · width, channels]`, which for 2.1 is a plain spatial flatten.
    public static func pack(_ latent: MLXArray) -> MLXArray {
        latent.reshaped([latent.dim(2) * latent.dim(3), latent.dim(4)])
    }

    /// The inverse of ``pack(_:)``.
    public static func unpack(_ tokens: MLXArray, height: Int, width: Int) -> MLXArray {
        tokens.reshaped([1, 1, height, width, tokens.dim(1)])
    }

    /// Denoises a prompt into an image.
    ///
    /// - Parameters:
    ///   - promptEmbeddings: `[captionTokens, contextInDimensions]` from ``promptEmbeddings(_:decoder:tokenizer:)``.
    ///   - negativeEmbeddings: the unconditional features, for true classifier-free guidance, or nil
    ///     for none.
    ///   - height: the image height in pixels, a multiple of 32.
    ///   - width: the image width in pixels, a multiple of 32.
    ///   - steps: how many denoising steps to take.
    ///   - guidance: the true classifier-free guidance scale, applied when `negativeEmbeddings` is given.
    ///   - seed: the noise seed.
    ///   - latents: a starting latent in the packed layout, which replaces the seeded noise.
    /// - Returns: the decoded image `[height, width, channels]` in −1…1.
    ///
    /// Generation is multi-second; run it off the main thread.
    public func generate(promptEmbeddings: MLXArray, negativeEmbeddings: MLXArray? = nil,
                         height: Int, width: Int, steps: Int = 40, guidance: Float = 4,
                         seed: UInt64 = 0, latents: MLXArray? = nil) -> MLXArray {
        let denoised = denoise(promptEmbeddings: promptEmbeddings, negativeEmbeddings: negativeEmbeddings,
                               height: height, width: width, steps: steps, guidance: guidance,
                               seed: seed, latents: latents)
        return decode(denoised, height: height / Self.scaleFactor, width: width / Self.scaleFactor)
    }

    /// The denoising loop on its own, returning the packed latents ``decode(_:height:width:)`` reads.
    /// The arguments are ``generate(promptEmbeddings:negativeEmbeddings:height:width:steps:guidance:seed:latents:)``'s.
    public func denoise(promptEmbeddings: MLXArray, negativeEmbeddings: MLXArray? = nil,
                        height: Int, width: Int, steps: Int = 40, guidance: Float = 4,
                        seed: UInt64 = 0, latents: MLXArray? = nil) -> MLXArray {
        let latentHeight = height / Self.scaleFactor, latentWidth = width / Self.scaleFactor
        let channels = holder.transformer.configuration.inChannels
        // The reference runs the latents and the conditioning in the transformer's type.
        let dtype = NFKReferenceRounding.parameterType(of: holder.transformer)
        MLXRandom.seed(seed)
        var tokens = (latents ?? MLXRandom.normal([latentHeight * latentWidth, channels])).asType(dtype)
        let promptEmbeddings = promptEmbeddings.asType(dtype)
        let negativeEmbeddings = negativeEmbeddings?.asType(dtype)

        // For text to image the caption carries no image slot, so the joint sequence's image mask is
        // the caption's own false run followed by one slot per 2x2 group of target latents.
        let captionLength = promptEmbeddings.dim(0)
        let slots = [Bool](repeating: false, count: captionLength)
            + [Bool](repeating: true, count: tokens.dim(0) / 4)
        let shapes = [(frame: 1, height: latentHeight, width: latentWidth)]

        var scheduler = NFKMLXFlowMatchScheduler(schedule)
        scheduler.setTimesteps(steps, sequenceLength: tokens.dim(0))

        for index in 0 ..< scheduler.timesteps.count {
            let timestep = dtype == .float32
                ? scheduler.timesteps[index] / Float(scheduler.configuration.trainTimesteps)
                : NFKReferenceRounding.flowFraction(scheduler.timesteps[index], dtype: dtype).asType(.float32).item(Float.self)
            let conditioned = holder.transformer(latents: tokens, encoderHidden: promptEmbeddings,
                                                 timestep: timestep, imageShapes: shapes,
                                                 imageMask: slots, encoderMask: nil)
            var velocity = conditioned[(conditioned.dim(0) - tokens.dim(0))...]
            if let negativeEmbeddings {
                let negativeSlots = [Bool](repeating: false, count: negativeEmbeddings.dim(0))
                    + [Bool](repeating: true, count: tokens.dim(0) / 4)
                let unconditioned = holder.transformer(
                    latents: tokens, encoderHidden: negativeEmbeddings, timestep: timestep,
                    imageShapes: shapes, imageMask: negativeSlots, encoderMask: nil)
                let negative = unconditioned[(unconditioned.dim(0) - tokens.dim(0))...]
                velocity = negative + guidance * (velocity - negative)
            }
            tokens = scheduler.step(velocity: velocity, sample: tokens, index: index)
            eval(tokens)
        }
        return tokens
    }

    /// Decodes a packed latent into an image `[height · 16, width · 16, channels]`.
    public func decode(_ tokens: MLXArray, height: Int, width: Int) -> MLXArray {
        var latent = Self.unpack(tokens, height: height, width: width)
        if let latentMean, let latentStandardDeviation {
            latent = latent * latentStandardDeviation + latentMean
        }
        let image = holder.vae.decode(latent)
        eval(image)
        return image[0, 0]
    }
}
