//
//  NFKMLXSANAPipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The SANA text-to-image pipeline: it chains the linear-attention DiT (denoised over a flow schedule
// with classifier-free guidance) and the Deep-Compression Autoencoder into an image. Each stage is
// validated at reference parity on its own; the pipeline is the glue.
//
// The caller supplies the caption embedding (the Gemma text encoder's last hidden state), as the SD
// pipeline takes a text context. The sampler is SANA's released `DPMSolverMultistepScheduler` in its
// flow-prediction configuration (`NFKMLXDPMSolverScheduler`, at reference parity against diffusers).

/// Holds the pipeline stages for capture across an async boundary.
private final class NFKSANAPipelineHolder: @unchecked Sendable {
    let transformer: NFKMLXSANATransformerNet
    let vae: NFKMLXDCAutoencoderNet
    init(_ transformer: NFKMLXSANATransformerNet, _ vae: NFKMLXDCAutoencoderNet) {
        self.transformer = transformer
        self.vae = vae
    }
}

/// The SANA text-to-image pipeline.
public final class NFKMLXSANAPipeline {

    private let holder: NFKSANAPipelineHolder
    private let inChannels: Int
    private let scaleFactor: Float

    init(transformer: NFKMLXSANATransformerNet, vae: NFKMLXDCAutoencoderNet) {
        holder = NFKSANAPipelineHolder(transformer, vae)
        self.inChannels = transformer.config.inChannels
        self.scaleFactor = vae.configuration.scaleFactor
    }

    /// Generates an image from a caption embedding. `latentHeight`/`latentWidth` are the LATENT grid
    /// dimensions (the DC-AE upsamples them by 32×). `negativeEmbeds` enable classifier-free guidance.
    public func generate(promptEmbeds: MLXArray, negativeEmbeds: MLXArray?, latentHeight: Int,
                         latentWidth: Int, steps: Int = 20, guidance: Float = 4.5, seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)
        var latent = MLXRandom.normal([inChannels, latentHeight, latentWidth])
        var scheduler = NFKMLXDPMSolverScheduler(.sana)
        scheduler.setTimesteps(steps)

        for index in 0 ..< scheduler.timesteps.count {
            let step = MLXArray([scheduler.timesteps[index]])              // flow timestep (σ · 1000)
            let conditioned = holder.transformer(latent, capFeats: promptEmbeds, t: step)
            var velocity = conditioned
            if let negativeEmbeds {
                let unconditioned = holder.transformer(latent, capFeats: negativeEmbeds, t: step)
                velocity = unconditioned + guidance * (conditioned - unconditioned)
            }
            latent = scheduler.step(velocity: velocity, sample: latent, index: index)
            eval(latent)
        }

        let nhwc = (latent / scaleFactor).transposed(1, 2, 0).expandedDimensions(axis: 0) // [1, h, w, C]
        let image = holder.vae.decode(nhwc)
        eval(image)
        return image
    }
}
