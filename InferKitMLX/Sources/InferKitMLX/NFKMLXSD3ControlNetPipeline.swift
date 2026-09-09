//
//  NFKMLXSD3ControlNetPipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The Stable Diffusion 3 ControlNet text-to-image pipeline: it chains the ControlNet
// (``NFKMLXSD3ControlNetNet``, run each step to produce per-block residuals), the base MMDiT
// (``NFKMLXSD3TransformerNet``, denoised over the rectified-flow schedule with the residuals injected),
// and the SD3 autoencoder. The control image is VAE-encoded once into the control latent the ControlNet
// reads at every step.
//
// The caller supplies the joint text embedding and the pooled projection, as the base pipeline does. The
// ControlNet runs once per classifier-free-guidance branch (the residuals depend on the text stream for
// the InstantX dual-stream shape), and each branch's base pass is offset by its own residuals. For the
// Stability 8B single-stream shape the ControlNet reads the pre-patch-embedded image tokens, which the
// base transformer's `pos_embed` supplies.

/// Holds the pipeline stages for capture across an async boundary.
private final class NFKSD3ControlNetHolder: @unchecked Sendable {
    let transformer: NFKMLXSD3TransformerNet
    let controlnet: NFKMLXSD3ControlNetNet
    let vae: NFKMLXSDAutoencoder
    init(_ transformer: NFKMLXSD3TransformerNet, _ controlnet: NFKMLXSD3ControlNetNet, _ vae: NFKMLXSDAutoencoder) {
        self.transformer = transformer
        self.controlnet = controlnet
        self.vae = vae
    }
}

/// The Stable Diffusion 3 ControlNet text-to-image pipeline.
public final class NFKMLXSD3ControlNetPipeline {

    private let holder: NFKSD3ControlNetHolder
    private let inChannels: Int
    private let scaleFactor: Float
    private let shiftFactor: Float
    private let usePosEmbed: Bool
    private let scheduler: NFKMLXFlowMatchScheduler

    /// Builds a pipeline from the base MMDiT, the ControlNet, and the autoencoder.
    public init(transformer: NFKMLXSD3TransformerNet, controlnet: NFKMLXSD3ControlNetNet,
                vae: NFKMLXSDAutoencoder, schedule: NFKMLXFlowMatchConfiguration = .sd3) {
        holder = NFKSD3ControlNetHolder(transformer, controlnet, vae)
        self.inChannels = transformer.config.inChannels
        self.scaleFactor = vae.configuration.scaleFactor
        self.shiftFactor = vae.configuration.shiftFactor
        self.usePosEmbed = controlnet.config.usePosEmbed
        self.scheduler = NFKMLXFlowMatchScheduler(schedule)
    }

    /// The VAE-encoded, scaled control latent `[1, C, h, w]` from a control image `[1, H, W, 3]`.
    private func controlLatent(from controlImage: MLXArray) -> MLXArray {
        let mean = holder.vae.encode(controlImage).mean                    // [1, h, w, C]
        return ((mean - shiftFactor) * scaleFactor).transposed(0, 3, 1, 2)
    }

    /// Runs the ControlNet for one guidance branch and returns its per-block residuals.
    private func residuals(latent: MLXArray, controlLatent: MLXArray, encoder: MLXArray, pooled: MLXArray,
                           timestep: MLXArray, scale: Float) -> [MLXArray] {
        let hidden = usePosEmbed ? latent : holder.transformer.posEmbed(latent)
        return holder.controlnet(hidden, controlnetCond: controlLatent, encoder: encoder, pooled: pooled,
                                 timestep: timestep, conditioningScale: scale)
    }

    /// Generates an image steered by `controlImage`. `latentHeight`/`latentWidth` are the LATENT grid
    /// dimensions (the VAE upsamples them by 8×). `controlImage` is `[1, H, W, 3]` in image space.
    /// `controlnetScale` weights the ControlNet's residuals; the negatives enable classifier-free
    /// guidance.
    public func generate(promptEmbeds: MLXArray, pooled: MLXArray, negativeEmbeds: MLXArray?,
                         negativePooled: MLXArray?, controlImage: MLXArray, controlnetScale: Float = 1,
                         latentHeight: Int, latentWidth: Int, steps: Int = 28, guidance: Float = 7,
                         seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)
        var latent = MLXRandom.normal([1, inChannels, latentHeight, latentWidth])
        let control = controlLatent(from: controlImage)
        let sequence = (latentHeight / 2) * (latentWidth / 2)
        var scheduler = self.scheduler
        scheduler.setTimesteps(steps, sequenceLength: sequence)

        let cond = promptEmbeds.expandedDimensions(axis: 0)
        let condPooled = pooled.expandedDimensions(axis: 0)
        let uncond = negativeEmbeds?.expandedDimensions(axis: 0)
        let uncondPooled = negativePooled?.expandedDimensions(axis: 0)

        for index in 0 ..< scheduler.timesteps.count {
            let t = MLXArray([scheduler.timesteps[index]])
            let condResiduals = residuals(latent: latent, controlLatent: control, encoder: cond,
                                          pooled: condPooled, timestep: t, scale: controlnetScale)
            let conditioned = holder.transformer(latent, encoderHidden: cond, pooled: condPooled,
                                                 timestep: t, blockControlnetHiddenStates: condResiduals)
            var prediction = conditioned
            if let uncond, let uncondPooled {
                let uncondResiduals = residuals(latent: latent, controlLatent: control, encoder: uncond,
                                                pooled: uncondPooled, timestep: t, scale: controlnetScale)
                let unconditioned = holder.transformer(latent, encoderHidden: uncond, pooled: uncondPooled,
                                                       timestep: t, blockControlnetHiddenStates: uncondResiduals)
                prediction = unconditioned + guidance * (conditioned - unconditioned)
            }
            latent = scheduler.step(velocity: prediction, sample: latent, index: index)
            eval(latent)
        }

        let decoded = latent[0]                                            // [C, H, W]
        let nhwc = (decoded / scaleFactor + shiftFactor).transposed(1, 2, 0).expandedDimensions(axis: 0)
        let image = holder.vae.decode(nhwc)
        eval(image)
        return image
    }
}
