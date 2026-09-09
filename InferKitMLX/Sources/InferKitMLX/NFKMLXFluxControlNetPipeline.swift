//
//  NFKMLXFluxControlNetPipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The FLUX.1 ControlNet text-to-image pipeline: it chains the ControlNet (``NFKMLXFluxControlNetNet``,
// run each step to produce the double- and single-block residuals), the base FLUX transformer
// (``NFKMLXFluxTransformerNet``, denoised over the rectified-flow schedule with the residuals injected),
// and the FLUX autoencoder. The control image is VAE-encoded once and packed into the control latent the
// ControlNet reads at every step.
//
// The caller supplies the T5-XXL sequence embedding and the CLIP-L pooled embedding, as the base
// pipeline does. FLUX operates on a PACKED latent, and the guidance-distilled `[dev]` model takes a
// scalar guidance embedding in place of classifier-free guidance, so the ControlNet runs once per step.
// A union ControlNet additionally reads a control-type index (`controlnetMode`).

/// Holds the pipeline stages for capture across an async boundary.
private final class NFKFluxControlNetHolder: @unchecked Sendable {
    let transformer: NFKMLXFluxTransformerNet
    let controlnet: NFKMLXFluxControlNetNet
    let vae: NFKMLXSDAutoencoder
    init(_ transformer: NFKMLXFluxTransformerNet, _ controlnet: NFKMLXFluxControlNetNet, _ vae: NFKMLXSDAutoencoder) {
        self.transformer = transformer
        self.controlnet = controlnet
        self.vae = vae
    }
}

/// The FLUX.1 ControlNet text-to-image pipeline.
public final class NFKMLXFluxControlNetPipeline {

    private let holder: NFKFluxControlNetHolder
    private let latentChannels: Int
    private let scaleFactor: Float
    private let shiftFactor: Float
    private let guidanceEmbeds: Bool
    private let usesInputHint: Bool
    private let scheduler: NFKMLXFlowMatchScheduler

    /// Builds a pipeline from the base transformer, the ControlNet, and the autoencoder.
    public init(transformer: NFKMLXFluxTransformerNet, controlnet: NFKMLXFluxControlNetNet,
                vae: NFKMLXSDAutoencoder, schedule: NFKMLXFlowMatchConfiguration = .flux) {
        holder = NFKFluxControlNetHolder(transformer, controlnet, vae)
        self.latentChannels = vae.configuration.latentChannels
        self.scaleFactor = vae.configuration.scaleFactor
        self.shiftFactor = vae.configuration.shiftFactor
        self.guidanceEmbeds = transformer.config.guidanceEmbeds
        self.usesInputHint = controlnet.config.conditioningEmbeddingChannels != nil
        self.scheduler = NFKMLXFlowMatchScheduler(schedule)
    }

    /// The control input the ControlNet reads. An `input_hint_block` ControlNet takes the raw
    /// full-resolution control image `[1, C, H, W]` (NCHW, the pyramid downsamples it); otherwise the
    /// image is VAE-encoded, scaled, and packed into the control latent `[1, seq, 64]`.
    private func controlInput(from controlImage: MLXArray) -> MLXArray {
        if usesInputHint {
            return controlImage.transposed(0, 3, 1, 2)                      // NHWC → NCHW
        }
        let mean = holder.vae.encode(controlImage).mean                    // [1, h, w, C]
        let latent = ((mean - shiftFactor) * scaleFactor).transposed(0, 3, 1, 2)
        return NFKMLXFluxPipeline.pack(latent)
    }

    /// Generates an image steered by `controlImage`. `latentHeight`/`latentWidth` are the LATENT grid
    /// dimensions (the VAE upsamples them by 8×; both must be even). `controlImage` is `[1, H, W, 3]` in
    /// image space. `controlnetMode` selects the control type for a union ControlNet (nil otherwise).
    public func generate(promptEmbeds: MLXArray, pooled: MLXArray, controlImage: MLXArray,
                         controlnetScale: Float = 1, controlnetMode: Int? = nil, latentHeight: Int,
                         latentWidth: Int, steps: Int = 28, guidance: Float = 3.5, seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)
        let latent = MLXRandom.normal([1, latentChannels, latentHeight, latentWidth])
        var packed = NFKMLXFluxPipeline.pack(latent)                       // [1, seq, 64]
        let control = controlInput(from: controlImage)
        let sequence = (latentHeight / 2) * (latentWidth / 2)
        let imageIds = NFKMLXFluxTransformerNet.imageIds(height: latentHeight / 2, width: latentWidth / 2)
        var scheduler = self.scheduler
        scheduler.setTimesteps(steps, sequenceLength: sequence)

        let cond = promptEmbeds.expandedDimensions(axis: 0)
        let condPooled = pooled.expandedDimensions(axis: 0)
        let guidanceValue = guidanceEmbeds ? MLXArray([guidance]) : nil
        let mode = controlnetMode.map { MLXArray([Int32($0)]) }

        for index in 0 ..< scheduler.timesteps.count {
            let t = MLXArray([scheduler.sigmas[index]])                    // the transformer scales ×1000
            let (double, single) = holder.controlnet(packed, controlnetCond: control, encoder: cond,
                                                     pooled: condPooled, timestep: t, guidance: guidanceValue,
                                                     imageIds: imageIds, controlnetMode: mode,
                                                     conditioningScale: controlnetScale)
            let prediction = holder.transformer(packed, encoderHidden: cond, pooled: condPooled,
                                                timestep: t, guidance: guidanceValue, imageIds: imageIds,
                                                controlnetBlockSamples: double.isEmpty ? nil : double,
                                                controlnetSingleBlockSamples: single.isEmpty ? nil : single)
            packed = scheduler.step(velocity: prediction, sample: packed, index: index)
            eval(packed)
        }

        let unpacked = NFKMLXFluxPipeline.unpack(packed, channels: latentChannels, height: latentHeight,
                                                 width: latentWidth)
        let decoded = unpacked[0]                                          // [C, H, W]
        let nhwc = (decoded / scaleFactor + shiftFactor).transposed(1, 2, 0).expandedDimensions(axis: 0)
        let image = holder.vae.decode(nhwc)
        eval(image)
        return image
    }
}
