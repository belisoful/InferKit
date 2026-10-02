//
//  NFKMLXWanPipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The Wan text-to-video glue after the text encoder: the DiT denoised over the UniPC flow schedule with
// classifier-free guidance, then the 3D causal VAE decode of the de-normalized latent. The glue is
// measured against diffusers' own `WanPipeline` (`run_reference.py wan_pipeline`); each model is
// measured against its own reference elsewhere.
//
// The sampler is Wan's released `UniPCMultistepScheduler` in its flow-prediction configuration
// (`NFKMLXUniPCScheduler`), whose shift the release's `scheduler_config.json` states. Wan 2.2 TI2V runs
// each token at its own timestep (`expand_timesteps`); for text to video every token carries the same
// step, which the reference measures within 2e-6 of the scalar timestep this passes.

/// Holds the pipeline stages for capture across an async boundary.
private final class NFKWanPipelineHolder: @unchecked Sendable {
    let transformer: NFKMLXWanTransformerNet
    let vae: NFKMLXWanVideoVAENet
    init(_ transformer: NFKMLXWanTransformerNet, _ vae: NFKMLXWanVideoVAENet) {
        self.transformer = transformer
        self.vae = vae
    }
}

/// The Wan text-to-video pipeline, from prompt features to a clip.
public final class NFKMLXWanPipeline {

    private let holder: NFKWanPipelineHolder
    private let inChannels: Int
    private let latentsMean: MLXArray?
    private let latentsStd: MLXArray?
    private let schedule: NFKMLXUniPCConfiguration

    /// Builds a pipeline from the DiT and the VAE.
    ///
    /// `latentsMean` and `latentsStd` are the release's per-channel latent statistics (length `zDim`), as
    /// `vae/config.json` states them; the decode reads `latent · std + mean`. A nil pair passes the latent
    /// to the VAE unchanged.
    public init(transformer: NFKMLXWanTransformerNet, vae: NFKMLXWanVideoVAENet,
                latentsMean: MLXArray? = nil, latentsStd: MLXArray? = nil,
                schedule: NFKMLXUniPCConfiguration = .wan) {
        holder = NFKWanPipelineHolder(transformer, vae)
        self.inChannels = transformer.config.inChannels
        self.latentsMean = latentsMean
        self.latentsStd = latentsStd
        self.schedule = schedule
    }

    /// Denoises prompt features into a latent `[inChannels, frames, height, width]`.
    ///
    /// - Parameters:
    ///   - textEmbeds: `[tokens, textDim]`, the umT5 features.
    ///   - negativeEmbeds: the negative prompt's features, or nil for no guidance.
    ///   - frames: the LATENT frame count.
    ///   - height: the latent height.
    ///   - width: the latent width.
    ///   - steps: the number of denoising steps.
    ///   - guidance: the classifier-free guidance scale, applied above 1 with a negative prompt.
    ///   - seed: the seed of the starting noise.
    ///   - latents: a starting latent, which replaces the seeded noise.
    public func denoise(textEmbeds: MLXArray, negativeEmbeds: MLXArray?, frames: Int, height: Int, width: Int,
                        steps: Int = 50, guidance: Float = 5, seed: UInt64 = 0,
                        latents: MLXArray? = nil) -> MLXArray {
        // The reference keeps the latents float32 and hands the transformer its own type.
        let dtype = NFKReferenceRounding.parameterType(of: holder.transformer)
        let textEmbeds = textEmbeds.asType(dtype), negativeEmbeds = negativeEmbeds?.asType(dtype)
        MLXRandom.seed(seed)
        var latent = latents ?? MLXRandom.normal([inChannels, frames, height, width])
        var scheduler = NFKMLXUniPCScheduler(schedule)
        scheduler.setTimesteps(steps)
        let guides = guidance > 1 && negativeEmbeds != nil

        for index in 0 ..< scheduler.timesteps.count {
            let step = MLXArray([scheduler.timesteps[index]])
            let conditioned = holder.transformer(latent.asType(dtype), text: textEmbeds, t: step)
            var velocity = conditioned
            if guides, let negativeEmbeds {
                let unconditioned = holder.transformer(latent.asType(dtype), text: negativeEmbeds, t: step)
                velocity = unconditioned + guidance * (conditioned - unconditioned)
            }
            latent = scheduler.step(velocity: velocity, sample: latent, index: index)
            eval(latent)
        }
        return latent
    }

    /// Decodes a latent `[C, F, H, W]` into a clip `[1, T, H, W, 3]` in −1…1, after de-normalizing it by
    /// the release's statistics.
    public func decode(_ latent: MLXArray) -> MLXArray {
        var channelsLast = latent.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
        if let latentsMean, let latentsStd {
            channelsLast = channelsLast * latentsStd.reshaped([1, 1, 1, 1, -1]) + latentsMean.reshaped([1, 1, 1, 1, -1])
        }
        let video = holder.vae.decode(channelsLast)
        eval(video)
        return video
    }

    /// ``denoise(textEmbeds:negativeEmbeds:frames:height:width:steps:guidance:seed:latents:)`` then
    /// ``decode(_:)``.
    public func generate(textEmbeds: MLXArray, negativeEmbeds: MLXArray?, frames: Int, height: Int,
                         width: Int, steps: Int = 20, guidance: Float = 5, seed: UInt64 = 0) -> MLXArray {
        decode(denoise(textEmbeds: textEmbeds, negativeEmbeds: negativeEmbeds, frames: frames, height: height,
                       width: width, steps: steps, guidance: guidance, seed: seed))
    }
}

// MARK: - Release

extension NFKMLXWanPipeline {

    /// Builds the pipeline from a diffusers Wan 2.1 or 2.2 text-to-video release directory: the DiT from
    /// `transformer/` at its stored precision, the autoencoder from `vae/` at float32 with the latent
    /// statistics its `config.json` states, and the flow shift from `scheduler/scheduler_config.json`.
    /// The caller supplies the umT5 prompt features (``NFKMLXT5Encoder``); ``NFKMLXWanVideoGenerator``
    /// runs the whole release, text encoder included. Blocking on the load; run it off the render
    /// thread. Introduced in InferKit 0.4.0.
    public static func pipeline(directoryURL: URL) throws -> NFKMLXWanPipeline {
        let transformerDirectory = directoryURL.appendingPathComponent("transformer")
        let vaeDirectory = directoryURL.appendingPathComponent("vae")
        let transformer = NFKMLXWanTransformerNet(try NFKMLXWanTransformerNet.configuration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json")))
        try NFKMLXWanTransformerNet.loadWeights(into: transformer, fromDirectory: transformerDirectory,
                                                precision: .checkpoint)
        let autoencoder = try NFKMLXWanVideoVAENet.configuration(
            fromHuggingFace: vaeDirectory.appendingPathComponent("config.json"))
        let vae = NFKMLXWanVideoVAENet(autoencoder.configuration)
        try NFKMLXWanVideoVAENet.loadWeights(into: vae, fromDirectory: vaeDirectory)
        let schedule = try NFKMLXWanRelease.schedule(
            fromHuggingFace: directoryURL.appendingPathComponent("scheduler/scheduler_config.json"))
        return NFKMLXWanPipeline(transformer: transformer, vae: vae, latentsMean: autoencoder.latentsMean,
                                 latentsStd: autoencoder.latentsStd, schedule: schedule)
    }
}

extension NFKMLXWanTransformerNet {

    /// The DiT geometry a diffusers `transformer/config.json` describes. An image-conditioned release
    /// (an `image_dim` or `added_kv_proj_dim`) is refused: this is the text-to-video transformer.
    /// Introduced in InferKit 0.4.0.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXWanConfiguration {
        try NFKMLXWanRelease.transformerConfiguration(fromHuggingFace: url)
    }

    /// Loads a release's `transformer/` directory (one file or a shard index). The module keys are the
    /// release's own; the patchify convolution moves from `[out, in, t, h, w]` to MLX's
    /// `[out, t, h, w, in]`. `.checkpoint` keeps the release's stored type. Introduced in InferKit 0.4.0.
    public static func loadWeights(into net: NFKMLXWanTransformerNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        try NFKMLXWanRelease.loadTransformer(into: net, fromDirectory: directory, precision: precision)
    }
}

extension NFKMLXWanVideoVAENet {

    /// The autoencoder geometry a diffusers `vae/config.json` describes, with the per-channel latent
    /// statistics the decode reads (nil where the release states none). Introduced in InferKit 0.4.0.
    public static func configuration(fromHuggingFace url: URL) throws
        -> (configuration: NFKMLXWanVAEConfiguration, latentsMean: MLXArray?, latentsStd: MLXArray?) {
        let read = try NFKMLXWanRelease.vaeConfiguration(fromHuggingFace: url)
        return (read.configuration, read.mean, read.std)
    }

    /// Loads a release's `vae/` directory: a 5-D causal convolution loads as `[out, t, h, w, in]`, a 4-D
    /// spatial one as `[out, h, w, in]`, and an RMS scale stored with trailing singleton axes flattens.
    /// Introduced in InferKit 0.4.0.
    public static func loadWeights(into net: NFKMLXWanVideoVAENet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        try NFKMLXWanRelease.loadVAE(into: net, fromDirectory: directory, precision: precision)
    }
}
