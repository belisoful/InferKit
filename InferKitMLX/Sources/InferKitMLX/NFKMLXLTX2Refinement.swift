//
//  NFKMLXLTX2Refinement.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

// LTX-2.5's decode-side paths around the joint denoising loop: the two-stage distilled recipe (stage 1 at
// half the size, the spatial latent upsampler, then a three-sigma refinement at the full size), the
// temporal latent upsampler, and the diffusion decoder.
//
// The upsamplers and the diffusion decoder read DENORMALIZED latents, `[B, F, H, W, C]`, the form the
// reference's `output_type="latent"` returns. The refinement's stage 2 normalizes and packs the upsampled
// video latent again and renoises it and stage 1's audio latent at `noise_scale`,
// `σ₀ · noise + (1 − σ₀) · latent`, before it samples its sigmas.

extension NFKMLXLTX2Pipeline {

    /// The second stage's sigmas in the two-stage distilled recipe, the reference's
    /// `STAGE_2_DISTILLED_SIGMA_VALUES`.
    public static let refinementSigmas: [Float] = [0.909375, 0.725, 0.421875]

    /// Packed normalized video latents `[B, F·H·W, C]` → the denormalized latent `[B, F, H, W, C]`.
    public func videoLatent(_ packed: MLXArray, frames: Int, height: Int, width: Int) -> MLXArray {
        packed.reshaped([packed.dim(0), frames, height, width, packed.dim(2)]) * videoStatistics.std
            + videoStatistics.mean
    }

    /// A denormalized latent `[B, F, H, W, C]` → packed normalized latents `[B, F·H·W, C]`.
    public func packedVideo(_ latent: MLXArray) -> MLXArray {
        ((latent - videoStatistics.mean) / videoStatistics.std).reshaped([latent.dim(0), -1, latent.dim(4)])
    }

    /// A denormalized latent `[B, F, H, W, C]` decoded by the convolutional autoencoder.
    public func decodeLatent(_ latent: MLXArray) -> MLXArray {
        videoDecoder(latent.asType(videoDecoder.convIn.conv.weight.dtype)).asType(.float32)
    }

    /// The reference's `_create_noised_state`: `scale · noise + (1 − scale) · latents`.
    public static func renoised(_ latents: MLXArray, noise: MLXArray, scale: Float) -> MLXArray {
        scale * noise + (1 - scale) * latents
    }

    /// A denormalized latent through `upsampler`, at the upsampler's type and back to float32.
    public static func upsampled(_ latent: MLXArray, by upsampler: NFKMLXLTX2LatentUpsamplerNet) -> MLXArray {
        upsampler(latent.asType(upsampler.initialConv.weight.dtype)).asType(.float32)
    }

    /// The two-stage recipe's second stage: stage 1's packed latents `video` (on the `frames × height ×
    /// width` grid) and `audio` → the refined packed latents on the upsampled grid, with that grid.
    ///
    /// @discussion The video latent is denormalized, upsampled by `upsampler`, normalized and packed
    /// again, and renoised with `videoNoise`; the audio latent is renoised with `audioNoise`; both at
    /// `noiseScale` (the first of `sigmas` where nil). Stage 2 then samples `sigmas` as `denoise` does,
    /// with the prompt streams and guidance of stage 1.
    public func refine(video: MLXArray, audio: MLXArray, frames: Int, height: Int, width: Int,
                       upsampler: NFKMLXLTX2LatentUpsamplerNet, videoNoise: MLXArray, audioNoise: MLXArray,
                       text: MLXArray, audioText: MLXArray, audioFrames: Int, frameRate: Float,
                       sigmas: [Float] = refinementSigmas, noiseScale: Float? = nil,
                       videoGuidance: NFKMLXLTX2Guidance, audioGuidance: NFKMLXLTX2Guidance, perturbedBlocks: Set<Int>,
                       schedule: NFKMLXFlowMatchConfiguration, onStep: ((Int) -> Void)? = nil) throws
        -> (video: MLXArray, audio: MLXArray, height: Int, width: Int) {
        let upsampledLatent = Self.upsampled(videoLatent(video, frames: frames, height: height, width: width),
                                             by: upsampler)
        eval(upsampledLatent)
        let scale = noiseScale ?? sigmas[0]
        let startVideo = Self.renoised(packedVideo(upsampledLatent), noise: videoNoise, scale: scale)
        let startAudio = Self.renoised(audio, noise: audioNoise, scale: scale)
        let (refinedHeight, refinedWidth) = (upsampledLatent.dim(2), upsampledLatent.dim(3))
        let refined = try denoise(
            video: startVideo, audio: startAudio, text: text, audioText: audioText, frames: frames,
            height: refinedHeight, width: refinedWidth, audioFrames: audioFrames, frameRate: frameRate,
            steps: sigmas.count, videoGuidance: videoGuidance, audioGuidance: audioGuidance,
            perturbedBlocks: perturbedBlocks, schedule: schedule, sigmas: sigmas, onStep: onStep)
        return (refined.video, refined.audio, refinedHeight, refinedWidth)
    }
}

extension NFKMLXLTX2DiffusionDecoder {
    /// A denormalized latent `[B, F, H, W, C]` decoded and returned at float32, as
    /// ``NFKMLXLTX2Pipeline/decodeLatent(_:)`` returns the autoencoder's decode.
    public func decodeLatent(_ latent: MLXArray, tiling: NFKMLXLTX2DecodeTiling? = .init(),
                             noise: ([Int]) -> MLXArray) -> MLXArray {
        decode(latent, tiling: tiling, noise: noise).asType(.float32)
    }
}
