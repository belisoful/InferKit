//
//  NFKMLXZImagePipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The Z-Image text-to-image pipeline: it chains the S3-DiT (denoised over the rectified-flow schedule
// with classifier-free guidance) and the Flux autoencoder into an image. Each stage is validated at
// reference parity on its own; the pipeline is the glue (the sampling loop, guidance, the timestep and
// latent conventions).
//
// The caller supplies the caption embedding (the Qwen3-4B hidden states) rather than a prompt, as the
// SD pipeline takes a text context: Z-Image's text step is the shipped Qwen3 decoder read for its last
// hidden state, run and freed separately from the image stages. Above a guidance of 1 the conditional
// and unconditional predictions are combined `cond + guidance·(cond − uncond)`, and the flow velocity is
// negated before the Euler step, both the reference's own conventions.

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

    /// Builds a pipeline from the DiT and the autoencoder (each loaded through its own factory).
    init(transformer: NFKMLXZImageTransformerNet, vae: NFKMLXSDAutoencoder) {
        holder = NFKZImagePipelineHolder(transformer, vae)
        self.inChannels = transformer.config.inChannels
        self.scaleFactor = vae.configuration.scaleFactor
        self.shiftFactor = vae.configuration.shiftFactor
    }

    /// Generates an image from a caption embedding. `latentHeight`/`latentWidth` are the LATENT grid
    /// dimensions (the VAE upsamples them by 8×). `negativeEmbeds` enable classifier-free guidance.
    public func generate(promptEmbeds: MLXArray, negativeEmbeds: MLXArray?, latentHeight: Int,
                         latentWidth: Int, steps: Int = 20, guidance: Float = 4, seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)
        var latent = MLXRandom.normal([inChannels, 1, latentHeight, latentWidth])
        let sequence = (latentHeight / 2) * (latentWidth / 2)              // the DiT's patch size is 2
        var scheduler = NFKMLXFlowMatchScheduler(.zImage)
        scheduler.setTimesteps(steps, sequenceLength: sequence)
        return denoise(latent, from: 0, promptEmbeds: promptEmbeds, negativeEmbeds: negativeEmbeds,
                       guidance: guidance, scheduler: scheduler)
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
        let latentHeight = imageLatent.dim(2), latentWidth = imageLatent.dim(3)
        let sequence = (latentHeight / 2) * (latentWidth / 2)

        var scheduler = NFKMLXFlowMatchScheduler(.zImage)
        scheduler.setTimesteps(steps, sequenceLength: sequence)
        let startIndex = steps - min(Int(Float(steps) * strength), steps)

        let noise = MLXRandom.normal(imageLatent.shape)
        let latent = scheduler.addNoise(imageLatent, noise: noise, sigma: scheduler.sigmas[startIndex])
        return denoise(latent, from: startIndex, promptEmbeds: promptEmbeds, negativeEmbeds: negativeEmbeds,
                       guidance: guidance, scheduler: scheduler)
    }

    /// The shared denoise loop (starting at `startIndex`) and the centered-latent decode.
    private func denoise(_ start: MLXArray, from startIndex: Int, promptEmbeds: MLXArray,
                         negativeEmbeds: MLXArray?, guidance: Float,
                         scheduler: NFKMLXFlowMatchScheduler) -> MLXArray {
        var latent = start
        for index in startIndex ..< scheduler.timesteps.count {
            let ditT = MLXArray([1 - scheduler.sigmas[index]])            // (1000 − t)/1000 = 1 − σ
            let conditioned = holder.transformer(latent, capFeats: promptEmbeds, t: ditT)
            var prediction = conditioned
            if let negativeEmbeds {
                let unconditioned = holder.transformer(latent, capFeats: negativeEmbeds, t: ditT)
                prediction = conditioned + guidance * (conditioned - unconditioned)
            }
            latent = scheduler.step(velocity: -prediction, sample: latent, index: index)
            eval(latent)
        }

        // Flux latents are centered and scaled; undo both before the decode.
        let decoded = latent[0..., 0, 0..., 0...]                          // [C, H, W]
        let nhwc = (decoded / scaleFactor + shiftFactor).transposed(1, 2, 0).expandedDimensions(axis: 0)
        let image = holder.vae.decode(nhwc)
        eval(image)
        return image
    }
}
