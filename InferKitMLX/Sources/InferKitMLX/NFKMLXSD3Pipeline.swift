//
//  NFKMLXSD3Pipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The Stable Diffusion 3 / 3.5 text-to-image pipeline: it chains the MMDiT (denoised over the
// rectified-flow schedule with classifier-free guidance) and the SD3 autoencoder into an image.
//
// The caller supplies the joint text embedding and the pooled projection rather than a prompt, as the
// SD pipeline takes a text context: SD3's text step is CLIP-L + CLIP-G + T5-XXL run separately (the two
// CLIP sequence outputs concatenated on channels and padded to T5's width, then concatenated with the
// T5 sequence; the two CLIP pooled outputs concatenated); `NFKMLXSD3Generator` runs that text step from
// a release. The flow velocity is applied directly (SD3's scheduler steps `sample + (σ_next − σ)·v`, no
// negation). The glue is measured against diffusers' `StableDiffusion3Pipeline` (`sd3_pipeline`).

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
    /// The flow schedule the loop samples.
    public let schedule: NFKMLXFlowMatchConfiguration
    /// The pixels a latent covers on a side: 8 for the released autoencoders.
    public let pixelsPerLatent: Int

    /// Builds a pipeline from the MMDiT and the autoencoder (each loaded through its own factory).
    public init(transformer: NFKMLXSD3TransformerNet, vae: NFKMLXSDAutoencoder,
                schedule: NFKMLXFlowMatchConfiguration = .sd3) {
        holder = NFKSD3PipelineHolder(transformer, vae)
        self.inChannels = transformer.config.inChannels
        self.scaleFactor = vae.configuration.scaleFactor
        self.shiftFactor = vae.configuration.shiftFactor
        self.schedule = schedule
        self.pixelsPerLatent = 1 << (vae.configuration.blockChannels.count - 1)
    }

    /// Generates an image. `latentHeight`/`latentWidth` are the LATENT grid dimensions (the VAE upsamples
    /// them by ``pixelsPerLatent``). `promptEmbeds` `[L, jointDim]`, `pooled` `[pooledDim]`; the
    /// negatives guide above a guidance of 1.
    public func generate(promptEmbeds: MLXArray, pooled: MLXArray, negativeEmbeds: MLXArray?,
                         negativePooled: MLXArray?, latentHeight: Int, latentWidth: Int,
                         steps: Int = 28, guidance: Float = 7, seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)
        let latent = MLXRandom.normal([1, inChannels, latentHeight, latentWidth])
        return decode(denoise(latent, promptEmbeds: promptEmbeds, pooled: pooled, negativeEmbeds: negativeEmbeds,
                              negativePooled: negativePooled, steps: steps, guidance: guidance))
    }

    /// The denoising loop on its own, from `latent` `[1, C, H, W]` (the reference's starting noise),
    /// returning the final float32 latent ``decode(_:)`` reads. The reference guides only above a
    /// guidance of 1, combining `uncond + guidance·(cond − uncond)`.
    public func denoise(_ latent: MLXArray, promptEmbeds: MLXArray, pooled: MLXArray, negativeEmbeds: MLXArray?,
                        negativePooled: MLXArray?, steps: Int, guidance: Float) -> MLXArray {
        var scheduler = NFKMLXFlowMatchScheduler(schedule)
        scheduler.setTimesteps(steps, sequenceLength: (latent.dim(2) / 2) * (latent.dim(3) / 2))
        let dtype = NFKReferenceRounding.parameterType(of: holder.transformer)
        let cond = promptEmbeds.expandedDimensions(axis: 0).asType(dtype)
        let condPooled = pooled.expandedDimensions(axis: 0).asType(dtype)
        let guides = guidance > 1 && negativeEmbeds != nil && negativePooled != nil
        let uncond = guides ? negativeEmbeds?.expandedDimensions(axis: 0).asType(dtype) : nil
        let uncondPooled = guides ? negativePooled?.expandedDimensions(axis: 0).asType(dtype) : nil

        var latent = latent.asType(.float32)
        for index in 0 ..< scheduler.timesteps.count {
            let t = MLXArray([scheduler.timesteps[index]])
            let input = latent.asType(dtype)
            var prediction = holder.transformer(input, encoderHidden: cond, pooled: condPooled, timestep: t)
                .asType(.float32)
            if let uncond, let uncondPooled {
                let unconditioned = holder.transformer(input, encoderHidden: uncond, pooled: uncondPooled, timestep: t)
                    .asType(.float32)
                prediction = unconditioned + guidance * (prediction - unconditioned)
            }
            latent = scheduler.step(velocity: prediction, sample: latent, index: index)
            eval(latent)
        }
        return latent
    }

    /// Decodes a latent `[1, C, H, W]` into an image `[1, H, W, 3]` in `[-1, 1]` at ``pixelsPerLatent``,
    /// undoing the autoencoder's latent scale and shift first.
    public func decode(_ latent: MLXArray) -> MLXArray {
        let nhwc = (latent[0] / scaleFactor + shiftFactor).transposed(1, 2, 0).expandedDimensions(axis: 0)
        let image = holder.vae.decode(nhwc.asType(NFKReferenceRounding.parameterType(of: holder.vae)))
        eval(image)
        return image
    }
}
