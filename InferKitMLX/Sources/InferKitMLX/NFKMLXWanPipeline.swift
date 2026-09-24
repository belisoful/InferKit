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
