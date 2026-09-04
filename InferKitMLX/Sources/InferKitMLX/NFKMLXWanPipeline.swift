//
//  NFKMLXWanPipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The Wan text-to-video pipeline: it chains the DiT (denoised over a flow schedule with classifier-free
// guidance) and the 3D causal VAE into a clip. Each stage is validated at reference parity on its own;
// the pipeline is the glue (the sampling loop, guidance, the latent normalization, the DiT↔VAE bridge).
//
// The caller supplies the text embedding (Wan's umT5 encoder, a T5-family model). The sampler is Wan's
// released `UniPCMultistepScheduler` in its flow-prediction configuration (`NFKMLXUniPCScheduler`, at
// reference parity against diffusers). The VAE latents are un-normalized by the release's per-channel
// mean/std before the decode.

/// Holds the pipeline stages for capture across an async boundary.
private final class NFKWanPipelineHolder: @unchecked Sendable {
    let transformer: NFKMLXWanTransformerNet
    let vae: NFKMLXWanVideoVAENet
    init(_ transformer: NFKMLXWanTransformerNet, _ vae: NFKMLXWanVideoVAENet) {
        self.transformer = transformer
        self.vae = vae
    }
}

/// The Wan text-to-video pipeline.
public final class NFKMLXWanPipeline {

    private let holder: NFKWanPipelineHolder
    private let inChannels: Int
    private let latentsMean: MLXArray?
    private let latentsStd: MLXArray?

    /// Builds a pipeline from the DiT and the VAE. `latentsMean`/`latentsStd` are the release's per-channel
    /// latent statistics (length `zDim`); when nil the latent is passed to the VAE unchanged.
    init(transformer: NFKMLXWanTransformerNet, vae: NFKMLXWanVideoVAENet,
         latentsMean: MLXArray? = nil, latentsStd: MLXArray? = nil) {
        holder = NFKWanPipelineHolder(transformer, vae)
        self.inChannels = transformer.config.inChannels
        self.latentsMean = latentsMean
        self.latentsStd = latentsStd
    }

    /// Generates a video from a text embedding. `frames`/`height`/`width` are the LATENT grid dimensions
    /// (the VAE upsamples them ×4 in time and ×16 in space). `negativeEmbeds` enable classifier-free
    /// guidance.
    public func generate(textEmbeds: MLXArray, negativeEmbeds: MLXArray?, frames: Int, height: Int,
                         width: Int, steps: Int = 20, guidance: Float = 5, seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)
        var latent = MLXRandom.normal([inChannels, frames, height, width])
        var scheduler = NFKMLXUniPCScheduler(.wan)
        scheduler.setTimesteps(steps)

        for index in 0 ..< scheduler.timesteps.count {
            let step = MLXArray([scheduler.timesteps[index]])
            let conditioned = holder.transformer(latent, text: textEmbeds, t: step)
            var velocity = conditioned
            if let negativeEmbeds {
                let unconditioned = holder.transformer(latent, text: negativeEmbeds, t: step)
                velocity = unconditioned + guidance * (conditioned - unconditioned)
            }
            latent = scheduler.step(velocity: velocity, sample: latent, index: index)
            eval(latent)
        }

        // Bridge [C, F, H, W] -> [1, F, H, W, C] and un-normalize the latent for the VAE.
        var nhwc = latent.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
        if let latentsMean, let latentsStd {
            nhwc = nhwc / latentsStd.reshaped([1, 1, 1, 1, -1]) + latentsMean.reshaped([1, 1, 1, 1, -1])
        }
        let video = holder.vae.decode(nhwc)
        eval(video)
        return video
    }
}
