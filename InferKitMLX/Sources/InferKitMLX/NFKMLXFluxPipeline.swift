//
//  NFKMLXFluxPipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The FLUX.1 text-to-image pipeline: it chains the FLUX transformer (denoised over the rectified-flow
// schedule) and the FLUX autoencoder into an image. Each stage is validated at reference parity on its
// own; the pipeline is the glue (the latent packing, the sampling loop, the guidance embedding, the
// timestep and latent conventions).
//
// The caller supplies the T5-XXL sequence embedding and the CLIP-L pooled embedding rather than a
// prompt, as the SD pipeline takes a text context. FLUX operates on a PACKED latent: a `[1, 16, H, W]`
// latent folds each 2×2 spatial block into the channel axis to `[1, (H/2)·(W/2), 64]` tokens, which the
// transformer denoises and the pipeline unpacks. The guidance-distilled `[dev]` model takes a scalar
// guidance embedding in place of classifier-free guidance; `[schnell]` takes none.

/// Holds the pipeline stages for capture across an async boundary.
private final class NFKFluxPipelineHolder: @unchecked Sendable {
    let transformer: NFKMLXFluxTransformerNet
    let vae: NFKMLXSDAutoencoder
    init(_ transformer: NFKMLXFluxTransformerNet, _ vae: NFKMLXSDAutoencoder) {
        self.transformer = transformer
        self.vae = vae
    }
}

/// The FLUX.1 text-to-image pipeline.
public final class NFKMLXFluxPipeline {

    private let holder: NFKFluxPipelineHolder
    private let latentChannels: Int
    private let scaleFactor: Float
    private let shiftFactor: Float
    private let guidanceEmbeds: Bool
    private let scheduler: NFKMLXFlowMatchScheduler

    /// Builds a pipeline from the transformer and the autoencoder. The schedule defaults to `[dev]`'s
    /// dynamic-shift flow; pass `.fluxSchnell` for the four-step distillation.
    public init(transformer: NFKMLXFluxTransformerNet, vae: NFKMLXSDAutoencoder,
                schedule: NFKMLXFlowMatchConfiguration = .flux) {
        holder = NFKFluxPipelineHolder(transformer, vae)
        self.latentChannels = vae.configuration.latentChannels
        self.scaleFactor = vae.configuration.scaleFactor
        self.shiftFactor = vae.configuration.shiftFactor
        self.guidanceEmbeds = transformer.config.guidanceEmbeds
        self.scheduler = NFKMLXFlowMatchScheduler(schedule)
    }

    /// Folds each 2×2 spatial block of a `[1, C, H, W]` latent into the channel axis: `[1, (H/2)·(W/2),
    /// C·4]`.
    static func pack(_ latent: MLXArray) -> MLXArray {
        let c = latent.dim(1), h = latent.dim(2), w = latent.dim(3)
        return latent.reshaped([1, c, h / 2, 2, w / 2, 2])
            .transposed(0, 2, 4, 1, 3, 5)
            .reshaped([1, (h / 2) * (w / 2), c * 4])
    }

    /// The inverse of `pack`, back to `[1, C, H, W]`.
    static func unpack(_ tokens: MLXArray, channels c: Int, height h: Int, width w: Int) -> MLXArray {
        tokens.reshaped([1, h / 2, w / 2, c, 2, 2])
            .transposed(0, 3, 1, 4, 2, 5)
            .reshaped([1, c, h, w])
    }

    /// Generates an image. `latentHeight`/`latentWidth` are the LATENT grid dimensions (the VAE upsamples
    /// them by 8×; both must be even). `promptEmbeds` `[L, jointDim]` is the T5 sequence, `pooled`
    /// `[pooledDim]` the CLIP-L pooled projection.
    public func generate(promptEmbeds: MLXArray, pooled: MLXArray, latentHeight: Int, latentWidth: Int,
                         steps: Int = 28, guidance: Float = 3.5, seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)
        let latent = MLXRandom.normal([1, latentChannels, latentHeight, latentWidth])
        var packed = Self.pack(latent)                                     // [1, seq, 64]
        let sequence = (latentHeight / 2) * (latentWidth / 2)
        let imageIds = NFKMLXFluxTransformerNet.imageIds(height: latentHeight / 2, width: latentWidth / 2)
        var scheduler = self.scheduler
        scheduler.setTimesteps(steps, sequenceLength: sequence)

        let cond = promptEmbeds.expandedDimensions(axis: 0)
        let condPooled = pooled.expandedDimensions(axis: 0)
        let guidanceValue = guidanceEmbeds ? MLXArray([guidance]) : nil

        for index in 0 ..< scheduler.timesteps.count {
            // The transformer scales the timestep by 1000 internally, so it takes the sigma directly.
            let t = MLXArray([scheduler.sigmas[index]])
            let prediction = holder.transformer(packed, encoderHidden: cond, pooled: condPooled,
                                                timestep: t, guidance: guidanceValue, imageIds: imageIds)
            packed = scheduler.step(velocity: prediction, sample: packed, index: index)
            eval(packed)
        }

        let unpacked = Self.unpack(packed, channels: latentChannels, height: latentHeight, width: latentWidth)
        let decoded = unpacked[0]                                          // [C, H, W]
        let nhwc = (decoded / scaleFactor + shiftFactor).transposed(1, 2, 0).expandedDimensions(axis: 0)
        let image = holder.vae.decode(nhwc)
        eval(image)
        return image
    }
}
