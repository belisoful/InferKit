//
//  NFKMLXZImagePipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The Z-Image text-to-image pipeline: it chains the S3-DiT (denoised over the rectified-flow schedule
// with classifier-free guidance) and the Flux autoencoder into an image. The glue is measured against
// diffusers' `ZImagePipeline` (`NFKMLXZImagePipelineTests`, the `z_image_pipeline` oracle).
//
// The caller supplies the caption embedding (the Qwen3-4B penultimate hidden states) rather than a
// prompt; `NFKMLXZImageGenerator` runs that text step from a release. Above a guidance of 1, and with a
// negative embedding, the conditional and unconditional predictions combine
// `cond + guidance·(cond − uncond)`, and the flow velocity is negated before the Euler step, both the
// reference's own conventions. The latents stay float32 across the loop and enter the transformer in
// its parameter type, as the reference's do.

/// Holds the pipeline stages for capture across an async boundary.
private final class NFKZImagePipelineHolder: @unchecked Sendable {
    let transformer: NFKMLXZImageTransformerNet
    let vae: NFKMLXSDAutoencoder
    init(_ transformer: NFKMLXZImageTransformerNet, _ vae: NFKMLXSDAutoencoder) {
        self.transformer = transformer
        self.vae = vae
    }
}

/// The Z-Image text-to-image pipeline.
public final class NFKMLXZImagePipeline {

    private let holder: NFKZImagePipelineHolder
    private let inChannels: Int
    private let scaleFactor: Float
    private let shiftFactor: Float
    /// The flow schedule the loop samples.
    public let schedule: NFKMLXFlowMatchConfiguration
    /// The pixels a latent covers on a side: 8 for the Flux autoencoder.
    public let pixelsPerLatent: Int

    /// Builds a pipeline from the DiT and the autoencoder (each loaded through its own factory).
    init(transformer: NFKMLXZImageTransformerNet, vae: NFKMLXSDAutoencoder,
         schedule: NFKMLXFlowMatchConfiguration = .zImage) {
        holder = NFKZImagePipelineHolder(transformer, vae)
        self.inChannels = transformer.config.inChannels
        self.scaleFactor = vae.configuration.scaleFactor
        self.shiftFactor = vae.configuration.shiftFactor
        self.schedule = schedule
        self.pixelsPerLatent = 1 << (vae.configuration.blockChannels.count - 1)
    }

    /// Generates an image from a caption embedding. `latentHeight`/`latentWidth` are the LATENT grid
    /// dimensions (the VAE upsamples them by ``pixelsPerLatent``). A `negativeEmbeds` guides above a guidance of 1.
    public func generate(promptEmbeds: MLXArray, negativeEmbeds: MLXArray?, latentHeight: Int,
                         latentWidth: Int, steps: Int = 20, guidance: Float = 4, seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)
        let latent = MLXRandom.normal([inChannels, 1, latentHeight, latentWidth])
        return decode(denoise(latent, promptEmbeds: promptEmbeds, negativeEmbeds: negativeEmbeds,
                              steps: steps, guidance: guidance))
    }

    /// Image-to-image (the diffusers edit path): the Flux VAE encodes `image` `[1, H, W, 3]`, the latent
    /// is noised to `strength` (0 keeps the source, 1 is a fresh generation), and the denoise runs from
    /// the matching step. NOTE: SigLIP-conditioned editing is a separate original-repo model not present
    /// in the diffusers Z-Image; this is diffusers' own strength-based img2img.
    public func generate(image: MLXArray, promptEmbeds: MLXArray, negativeEmbeds: MLXArray?,
                         strength: Float = 0.6, steps: Int = 20, guidance: Float = 4, seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)
        let mean = holder.vae.encode(image).mean                          // [1, lh, lw, 16]
        let imageLatent = ((mean - shiftFactor) * scaleFactor)[0].transposed(2, 0, 1).expandedDimensions(axis: 1)
        let scheduler = self.scheduler(steps: steps, latent: imageLatent)
        let startIndex = steps - min(Int(Float(steps) * strength), steps)

        let noise = MLXRandom.normal(imageLatent.shape)
        let latent = scheduler.addNoise(imageLatent, noise: noise, sigma: scheduler.sigmas[startIndex])
        return decode(denoise(latent, from: startIndex, scheduler: scheduler, promptEmbeds: promptEmbeds,
                              negativeEmbeds: negativeEmbeds, guidance: guidance))
    }

    /// The denoising loop on its own, from `latent` `[C, 1, H, W]` (the reference's starting noise),
    /// returning the final float32 latent ``decode(_:)`` reads.
    public func denoise(_ latent: MLXArray, promptEmbeds: MLXArray, negativeEmbeds: MLXArray?,
                        steps: Int, guidance: Float) -> MLXArray {
        denoise(latent, from: 0, scheduler: scheduler(steps: steps, latent: latent),
                promptEmbeds: promptEmbeds, negativeEmbeds: negativeEmbeds, guidance: guidance)
    }

    private func scheduler(steps: Int, latent: MLXArray) -> NFKMLXFlowMatchScheduler {
        var scheduler = NFKMLXFlowMatchScheduler(schedule)
        scheduler.setTimesteps(steps, sequenceLength: (latent.dim(2) / 2) * (latent.dim(3) / 2))
        return scheduler
    }

    private func denoise(_ start: MLXArray, from startIndex: Int, scheduler: NFKMLXFlowMatchScheduler,
                         promptEmbeds: MLXArray, negativeEmbeds: MLXArray?, guidance: Float) -> MLXArray {
        let dtype = NFKReferenceRounding.parameterType(of: holder.transformer)
        let promptEmbeds = promptEmbeds.asType(dtype)
        let negativeEmbeds = guidance > 1 ? negativeEmbeds?.asType(dtype) : nil
        let train = Float(scheduler.configuration.trainTimesteps)
        var latent = start.asType(.float32)
        for index in startIndex ..< scheduler.timesteps.count {
            // A step onto the same sigma leaves the latent as it is, so its prediction is not computed.
            if scheduler.sigmas[index + 1] == scheduler.sigmas[index] {
                continue
            }
            let ditT = MLXArray([(train - scheduler.timesteps[index]) / train])
            let input = latent.asType(dtype)
            var prediction = holder.transformer(input, capFeats: promptEmbeds, t: ditT).asType(.float32)
            if let negativeEmbeds {
                let unconditioned = holder.transformer(input, capFeats: negativeEmbeds, t: ditT).asType(.float32)
                prediction = prediction + guidance * (prediction - unconditioned)
            }
            latent = scheduler.step(velocity: -prediction, sample: latent, index: index)
            eval(latent)
        }
        return latent
    }

    /// Decodes a latent `[C, 1, H, W]` into an image `[1, H, W, 3]` in `[-1, 1]` at ``pixelsPerLatent``. Flux latents are
    /// centered and scaled; the decode undoes both first.
    public func decode(_ latent: MLXArray) -> MLXArray {
        let planes = latent[0..., 0, 0..., 0...]                           // [C, H, W]
        let nhwc = (planes / scaleFactor + shiftFactor).transposed(1, 2, 0).expandedDimensions(axis: 0)
        let image = holder.vae.decode(nhwc.asType(NFKReferenceRounding.parameterType(of: holder.vae)))
        eval(image)
        return image
    }
}
