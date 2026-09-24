//
//  NFKMLXLTXPipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The LTX-Video 0.9.0 text-to-video glue after the text encoder: the DiT denoised over the rectified-flow
// schedule with classifier-free guidance, then the latent denormalization and the causal 3D VAE decode.
// The glue is measured against diffusers' own `LTXPipeline` (`run_reference.py ltx_pipeline`); each model
// is measured against its own reference elsewhere. Four details of it are the pipeline's rather than any
// model's:
//
// - The caption is padded to `max_sequence_length` and its padding is masked out of the cross-attention.
// - The sigma ramp is the pipeline's own `linspace(1, 1 / steps, steps)`, dynamically shifted by the
//   video's sequence length.
// - The rotary reads the latent grid at `(8 / frameRate, 32, 32)`, the VAE's compression over the frame
//   rate, so a frame index is a time.
// - The latents are denormalized by the autoencoder's stored per-channel mean and standard deviation
//   before the decode.
//
// The video latent is `[B, frames, height, width, latentChannels]` (the VAE's NDHWC latent); with the
// DiT's patch size of 1 the packing to and from the DiT's token sequence is a single reshape.

/// Holds the pipeline stages for capture across the async boundary.
private final class NFKLTXPipelineHolder: @unchecked Sendable {
    let transformer: NFKMLXLTXTransformerNet
    let vae: NFKMLXLTXVideoVAENet
    init(_ transformer: NFKMLXLTXTransformerNet, _ vae: NFKMLXLTXVideoVAENet) {
        self.transformer = transformer
        self.vae = vae
    }
}

/// The LTX-Video text-to-video pipeline, from prompt features to a clip.
public final class NFKMLXLTXPipeline {

    /// The VAE's spatial compression: a latent cell is 32×32 pixels.
    public static let spatialCompression = 32
    /// The VAE's temporal compression: a latent frame after the first is 8 video frames.
    public static let temporalCompression = 8

    private let holder: NFKLTXPipelineHolder
    private let schedule: NFKMLXFlowMatchConfiguration
    private let scalingFactor: Float

    /// Builds the pipeline from the DiT and the autoencoder, whose stored latent statistics the decode
    /// reads. `scalingFactor` is the autoencoder config's `scaling_factor`, 1 in the released 0.9.0.
    init(transformer: NFKMLXLTXTransformerNet, vae: NFKMLXLTXVideoVAENet, scalingFactor: Float = 1,
         schedule: NFKMLXFlowMatchConfiguration = .ltxVideoPipeline) {
        holder = NFKLTXPipelineHolder(transformer, vae)
        self.scalingFactor = scalingFactor
        self.schedule = schedule
    }

    private var latentChannels: Int { holder.transformer.configuration.inChannels }

    /// Denoises prompt features into packed latents `[1, frames · height · width, latentChannels]`.
    ///
    /// - Parameters:
    ///   - text: `[1, tokens, captionChannels]`, the T5 features of the padded prompt.
    ///   - textMask: `[1, tokens]`, 1 for a prompt token and 0 for padding.
    ///   - negativeText: the negative prompt's features, or nil for no guidance.
    ///   - negativeMask: the negative prompt's mask.
    ///   - frames: the LATENT frame count; the clip is `(frames − 1) · 8 + 1` frames long.
    ///   - height: the latent height; the clip is 32 times taller.
    ///   - width: the latent width; the clip is 32 times wider.
    ///   - steps: the number of denoising steps.
    ///   - guidance: the classifier-free guidance scale, applied above 1 with a negative prompt.
    ///   - seed: the seed of the starting noise.
    ///   - latents: a starting latent in the packed layout, which replaces the seeded noise.
    ///   - frameRate: the frame rate the rotary scales time by.
    public func denoise(text: MLXArray, textMask: MLXArray?, negativeText: MLXArray?, negativeMask: MLXArray?,
                        frames: Int, height: Int, width: Int, steps: Int = 50, guidance: Float = 3,
                        seed: UInt64 = 0, latents: MLXArray? = nil, frameRate: Float = 25) -> MLXArray {
        MLXRandom.seed(seed)
        let sequence = frames * height * width
        var latent = latents?.reshaped([1, sequence, latentChannels])
            ?? MLXRandom.normal([1, sequence, latentChannels])
        var scheduler = NFKMLXFlowMatchScheduler(schedule)
        scheduler.setTimesteps(steps, sequenceLength: sequence)
        let ropeScale = (Float(Self.temporalCompression) / frameRate, Float(Self.spatialCompression),
                         Float(Self.spatialCompression))
        let guides = guidance > 1 && negativeText != nil

        for (index, timestep) in scheduler.timesteps.enumerated() {
            let step = MLXArray([timestep])
            let conditioned = holder.transformer(latent, text: text, timestep: step, grid: (frames, height, width),
                                                 ropeScale: ropeScale, textMask: textMask)
            var velocity = conditioned
            if guides, let negativeText {
                let unconditioned = holder.transformer(latent, text: negativeText, timestep: step,
                                                       grid: (frames, height, width), ropeScale: ropeScale,
                                                       textMask: negativeMask)
                velocity = unconditioned + guidance * (conditioned - unconditioned)
            }
            latent = scheduler.step(velocity: velocity, sample: latent, index: index)
            eval(latent)
        }
        return latent
    }

    /// Decodes packed latents into a clip `[1, (frames − 1) · 8 + 1, height · 32, width · 32, 3]` in
    /// −1…1, after denormalizing them by the autoencoder's stored statistics.
    public func decode(_ latents: MLXArray, frames: Int, height: Int, width: Int) -> MLXArray {
        let vae = holder.vae
        let grid = latents.reshaped([1, frames, height, width, latentChannels])
        let denormalized = grid * vae.latentsStd.reshaped([1, 1, 1, 1, -1]) / scalingFactor
            + vae.latentsMean.reshaped([1, 1, 1, 1, -1])
        let video = vae.decode(denormalized.asType(grid.dtype))
        eval(video)
        return video
    }
}
