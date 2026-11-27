//
//  NFKMLXWanVACEFun.swift
//  InferKitMLX
//
//  The conditions of Wan 2.2 VACE-Fun A14B (alibaba-pai, Apache-2.0), as VideoX-Fun's
//  `Wan2_2VaceFunPipeline` prepares them: a control video, or a video kept outside a mask, and subject
//  reference images, encoded into the 96-channel context its VACE blocks read.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX

/// The pictures one Wan 2.2 VACE-Fun clip is conditioned on, each at the clip's size in `0...1`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanVACEConditions {
    /// The control video `[F, H, W, 3]` (a pose, depth, or edge video) the clip follows. It takes the place of
    /// ``video``.
    public var controlVideo: MLXArray?
    /// The video `[F, H, W, 3]` kept outside ``mask``, where no control video is given.
    public var video: MLXArray?
    /// The mask `[F, H, W]`, 1 where the clip is generated; nil generates every pixel.
    public var mask: MLXArray?
    /// The subject reference images `[H, W, 3]`, each a latent frame of context ahead of the clip.
    public var subjectImages: [MLXArray]
    /// The weight of the VACE hints (`vace_context_scale`, 1).
    public var scale: Float

    public init(controlVideo: MLXArray? = nil, video: MLXArray? = nil, mask: MLXArray? = nil,
                subjectImages: [MLXArray] = [], scale: Float = 1) {
        self.controlVideo = controlVideo
        self.video = video
        self.mask = mask
        self.subjectImages = subjectImages
        self.scale = scale
    }
}

extension NFKMLXFlowMatchConfiguration {
    /// Wan 2.2 VACE-Fun's schedule: Euler at a static flow shift of 12 (`config/wan2.2/wan_civitai_t2v.yaml`).
    /// Introduced in InferKit 0.4.0.
    public static let wanVACEFun = NFKMLXFlowMatchConfiguration(baseShift: 12.0, shiftTerminal: nil, useDynamicShifting: false)
}

extension NFKMLXWanFunPipeline {
    /// The boundary of the VACE-Fun A14B experts: the high-noise expert at and above timestep 875
    /// (`config/wan2.2/wan_civitai_t2v.yaml`).
    static let vaceFunBoundary: Float = 0.875

    /// The prepared conditions of a VACE-Fun clip (`vace_encode_frames`, `vace_encode_masks`, `vace_latent`) and
    /// the reference frames ahead of the clip, which the denoised latents drop before the decode.
    ///
    /// @discussion The input is the control video, or the video where the mask keeps it and 0 where it is
    /// generated, both in `-1...1`. The mask binarizes at 0.5. The inactive part (the input outside the mask) and the
    /// reactive part (inside it) encode separately into 32 channels; each reference image encodes alone and takes
    /// zeros for its reactive half. The mask folds each 8 × 8 pixel block into 64 channels and resamples its
    /// frames to the latent frames by nearest-exact interpolation; the reference frames read zeros. The context is
    /// the latents then the mask, 96 channels.
    func prepareVACE(_ conditions: NFKMLXWanVACEConditions, frames: Int, height: Int,
                     width: Int) throws -> (prepared: Prepared, referenceFrames: Int) {
        let s = spatialCompression
        let mask = conditions.mask.map { $0 .>= 0.5 }.map { $0.asType(.float32) }
            ?? MLXArray.ones([frames, height, width])
        let input: MLXArray
        if let control = conditions.controlVideo {
            input = control * 2 - 1
        } else if let video = conditions.video {
            input = (video * 2 - 1) * (1 - mask).expandedDimensions(axis: -1)
        } else {
            throw NFKMLXError.unsupportedConfiguration("a VACE-Fun clip reads a control video or a video under a mask")
        }
        let m = mask.expandedDimensions(axis: -1)
        var latents = concatenated([encoded(input * (1 - m)), encoded(input * m)], axis: 0)
        let references = conditions.subjectImages.map { image -> MLXArray in
            let latent = encoded((image * 2 - 1).expandedDimensions(axis: 0))
            return concatenated([latent, MLXArray.zeros(like: latent)], axis: 0)
        }
        if !references.isEmpty {
            latents = concatenated(references + [latents], axis: 1)
        }
        let h = 2 * (height / (s * 2)), w = 2 * (width / (s * 2))
        let folded = mask[0..., 0 ..< (h * s), 0 ..< (w * s)].reshaped([frames, h, s, w, s])
            .transposed(2, 4, 0, 1, 3).reshaped([s * s, frames, h, w])
        let latentFrames = (frames + 3) / 4
        let scale = Float(frames) / Float(latentFrames)
        let sources = (0 ..< latentFrames).map { Int32(min(Int((Float($0) + 0.5) * scale), frames - 1)) }
        var maskLatents = folded.take(MLXArray(sources), axis: 1)
        if !references.isEmpty {
            maskLatents = concatenated([MLXArray.zeros([s * s, references.count, h, w]), maskLatents], axis: 1)
        }
        let context = concatenated([latents, maskLatents], axis: 0)
        eval(context)
        let z = latentChannels
        let prepared = Prepared(channels: MLXArray.zeros([0, context.dim(1), h, w]), reference: nil, camera: nil,
                                latentMask: nil, maskedLatents: MLXArray.zeros([z, context.dim(1), h, w]), blends: false,
                                vaceContext: context, vaceScale: conditions.scale)
        return (prepared, references.count)
    }
}
