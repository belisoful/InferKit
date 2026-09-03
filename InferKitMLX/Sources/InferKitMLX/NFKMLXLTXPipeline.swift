//
//  NFKMLXLTXPipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The LTX-Video text-to-video pipeline: it chains the three stages — the T5 text encoder, the DiT
// (denoised over the rectified-flow schedule with classifier-free guidance), and the causal 3D VAE — into
// a clip. Each stage is validated at reference parity on its own; the pipeline is the glue (latent
// packing, the sampling loop, guidance) plus sequential residency so the 19 GB T5, the 7.7 GB DiT, and the
// VAE are not all resident at once.
//
// The video latent is `[B, frames, height, width, latentChannels]` (the VAE's NDHWC latent); with the
// DiT's patch size of 1 the packing to and from the DiT's token sequence is a single reshape.

/// Holds the pipeline stages for capture across the async boundary.
private final class NFKLTXPipelineHolder: @unchecked Sendable {
    let text: NFKMLXT5EncoderNet
    let transformer: NFKMLXLTXTransformerNet
    let vae: NFKMLXLTXVideoVAENet
    init(_ text: NFKMLXT5EncoderNet, _ transformer: NFKMLXLTXTransformerNet, _ vae: NFKMLXLTXVideoVAENet) {
        self.text = text
        self.transformer = transformer
        self.vae = vae
    }
}

/// The LTX-Video text-to-video pipeline.
public final class NFKMLXLTXPipeline {

    private let holder: NFKLTXPipelineHolder
    private let latentChannels: Int

    /// Builds a pipeline from its three stages (each loaded through its own factory). A caller managing
    /// memory can build and free the stages around a run; this holds them together for a single call.
    init(textEncoder: NFKMLXT5EncoderNet, transformer: NFKMLXLTXTransformerNet,
         vae: NFKMLXLTXVideoVAENet, latentChannels: Int = 128) {
        holder = NFKLTXPipelineHolder(textEncoder, transformer, vae)
        self.latentChannels = latentChannels
    }

    /// Generates a video from tokenized prompts. `frames`/`height`/`width` are the LATENT grid dimensions
    /// (the VAE upsamples them to pixels). `negativeTokens` enable classifier-free guidance.
    public func generate(promptTokens: MLXArray, negativeTokens: MLXArray?, frames: Int, height: Int,
                         width: Int, steps: Int = 20, guidance: Float = 3, seed: UInt64 = 0,
                         ropeScale: (Float, Float, Float) = (1, 1, 1)) -> MLXArray {
        MLXRandom.seed(seed)
        let text = holder.text(promptTokens)
        let uncond = negativeTokens.map { holder.text($0) }

        let sequence = frames * height * width
        var latent = MLXRandom.normal([1, sequence, latentChannels])
        var scheduler = NFKMLXFlowMatchScheduler(.ltxVideo)
        scheduler.setTimesteps(steps, sequenceLength: sequence)

        for (index, timestep) in scheduler.timesteps.enumerated() {
            let step = MLXArray([timestep])
            let conditioned = holder.transformer(latent, text: text, timestep: step, grid: (frames, height, width), ropeScale: ropeScale)
            var velocity = conditioned
            if let uncond {
                let unconditioned = holder.transformer(latent, text: uncond, timestep: step, grid: (frames, height, width), ropeScale: ropeScale)
                velocity = unconditioned + guidance * (conditioned - unconditioned)
            }
            latent = scheduler.step(velocity: velocity, sample: latent, index: index)
            eval(latent)
        }

        let grid = latent.reshaped([1, frames, height, width, latentChannels])
        let video = holder.vae.decode(grid)
        eval(video)
        return video
    }
}
