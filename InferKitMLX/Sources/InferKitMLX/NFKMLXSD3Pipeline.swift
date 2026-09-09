//
//  NFKMLXSD3Pipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The Stable Diffusion 3 / 3.5 text-to-image pipeline: it chains the MMDiT (denoised over the
// rectified-flow schedule with classifier-free guidance) and the SD3 autoencoder into an image. Each
// stage is validated at reference parity on its own; the pipeline is the glue (the sampling loop,
// guidance, the timestep and latent conventions).
//
// The caller supplies the joint text embedding and the pooled projection rather than a prompt, as the
// SD pipeline takes a text context: SD3's text step is CLIP-L + CLIP-G + T5-XXL run separately (the two
// CLIP sequence outputs concatenated on channels and padded to T5's width, then concatenated with the
// T5 sequence; the two CLIP pooled outputs concatenated). The flow velocity is applied directly (SD3's
// scheduler steps `sample + (σ_next − σ)·v`, no negation).

/// Holds the pipeline stages for capture across an async boundary.
private final class NFKSD3PipelineHolder: @unchecked Sendable {
    let transformer: NFKMLXSD3TransformerNet
    let vae: NFKMLXSDAutoencoder
    init(_ transformer: NFKMLXSD3TransformerNet, _ vae: NFKMLXSDAutoencoder) {
        self.transformer = transformer
        self.vae = vae
    }
}

/// The Stable Diffusion 3 text-to-image pipeline.
public final class NFKMLXSD3Pipeline {

    private let holder: NFKSD3PipelineHolder
    private let inChannels: Int
    private let scaleFactor: Float
    private let shiftFactor: Float
    private let scheduler: NFKMLXFlowMatchScheduler

    /// Builds a pipeline from the MMDiT and the autoencoder (each loaded through its own factory).
    public init(transformer: NFKMLXSD3TransformerNet, vae: NFKMLXSDAutoencoder,
                schedule: NFKMLXFlowMatchConfiguration = .sd3) {
        holder = NFKSD3PipelineHolder(transformer, vae)
        self.inChannels = transformer.config.inChannels
        self.scaleFactor = vae.configuration.scaleFactor
        self.shiftFactor = vae.configuration.shiftFactor
        self.scheduler = NFKMLXFlowMatchScheduler(schedule)
    }

    /// Generates an image. `latentHeight`/`latentWidth` are the LATENT grid dimensions (the VAE upsamples
    /// them by 8×). `promptEmbeds` `[L, jointDim]`, `pooled` `[pooledDim]`; the negatives enable CFG.
    public func generate(promptEmbeds: MLXArray, pooled: MLXArray, negativeEmbeds: MLXArray?,
                         negativePooled: MLXArray?, latentHeight: Int, latentWidth: Int,
                         steps: Int = 28, guidance: Float = 7, seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)
        var latent = MLXRandom.normal([1, inChannels, latentHeight, latentWidth])
        let sequence = (latentHeight / 2) * (latentWidth / 2)
        var scheduler = self.scheduler
        scheduler.setTimesteps(steps, sequenceLength: sequence)

        let cond = promptEmbeds.expandedDimensions(axis: 0)
        let condPooled = pooled.expandedDimensions(axis: 0)
        let uncond = negativeEmbeds?.expandedDimensions(axis: 0)
        let uncondPooled = negativePooled?.expandedDimensions(axis: 0)

        for index in 0 ..< scheduler.timesteps.count {
            let t = MLXArray([scheduler.timesteps[index]])
            let conditioned = holder.transformer(latent, encoderHidden: cond, pooled: condPooled, timestep: t)
            var prediction = conditioned
            if let uncond, let uncondPooled {
                let unconditioned = holder.transformer(latent, encoderHidden: uncond, pooled: uncondPooled, timestep: t)
                prediction = unconditioned + guidance * (conditioned - unconditioned)
            }
            latent = scheduler.step(velocity: prediction, sample: latent, index: index)
            eval(latent)
        }

        // Undo the VAE's latent scaling and shift, then decode.
        let decoded = latent[0]                                            // [C, H, W]
        let nhwc = (decoded / scaleFactor + shiftFactor).transposed(1, 2, 0).expandedDimensions(axis: 0)
        let image = holder.vae.decode(nhwc)
        eval(image)
        return image
    }
}
