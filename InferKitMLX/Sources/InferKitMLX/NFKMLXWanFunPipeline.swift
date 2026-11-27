//
//  NFKMLXWanFunPipeline.swift
//  InferKitMLX
//
//  The Wan 2.2 Fun releases' sampling loop (alibaba-pai, VideoX-Fun `Wan2_2FunControlPipeline` and
//  `Wan2_2FunInpaintPipeline`): the noise, a control video's latents, an inpainting mask and the masked
//  video's latents read together as the transformer's input channels; a reference image's latent as a
//  frame of tokens ahead of the video; a camera's Plücker embedding through the control adapter.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXRandom

/// The conditions a Wan 2.2 Fun clip reads, in pixels.
///
/// @discussion Every video is `[F, H, W, 3]` in `0...1` at the clip's size, and every image `[H, W, 3]`.
/// The inpainting mask is `[F, H, W]`, 1 where the clip is generated and 0 where `inpaintVideo` is kept;
/// a start image is `inpaintVideo` frame 0 under a mask of 0 there and 1 elsewhere, which
/// ``NFKMLXWanFunConditions/startImage(_:endImage:frames:)`` builds. No inpainting video generates every
/// frame, which is how VideoX-Fun's examples run a control release without a start image. Introduced in
/// InferKit 0.4.0.
public struct NFKMLXWanFunConditions {
    /// The control video (pose, depth, edges, …) a Control release follows.
    public var controlVideo: MLXArray?
    /// The video an inpainting mask keeps from.
    public var inpaintVideo: MLXArray?
    /// The inpainting mask, 1 where the clip is generated.
    public var inpaintMask: MLXArray?
    /// The reference image a Control release with `ref_conv` reads as a frame of tokens.
    public var referenceImage: MLXArray?
    /// The camera's Plücker embedding `[6, F, H, W]` a Control-Camera release reads.
    public var cameraEmbedding: MLXArray?

    public init(controlVideo: MLXArray? = nil, inpaintVideo: MLXArray? = nil, inpaintMask: MLXArray? = nil,
                referenceImage: MLXArray? = nil, cameraEmbedding: MLXArray? = nil) {
        self.controlVideo = controlVideo
        self.inpaintVideo = inpaintVideo
        self.inpaintMask = inpaintMask
        self.referenceImage = referenceImage
        self.cameraEmbedding = cameraEmbedding
    }

    /// The inpainting video and mask VideoX-Fun's `get_image_to_video_latent` builds from a start image
    /// and, where given, an end image: the start image repeated over every frame, the end image as the
    /// last, and a mask of 0 on the given frames and 1 on the rest.
    public static func startImage(_ start: MLXArray, endImage end: MLXArray? = nil, frames: Int)
        -> (video: MLXArray, mask: MLXArray) {
        var video = broadcast(start.expandedDimensions(axis: 0), to: [frames, start.dim(0), start.dim(1), 3])
        var keep = [Float](repeating: 1, count: frames)
        keep[0] = 0
        if let end {
            video = concatenated([video[0 ..< (frames - 1)], end.expandedDimensions(axis: 0)], axis: 0)
            keep[frames - 1] = 0
        }
        let mask = broadcast(MLXArray(keep).reshaped([frames, 1, 1]), to: [frames, start.dim(0), start.dim(1)])
        return (video, mask)
    }
}

/// The Wan 2.2 Fun sampler over one or two transformers and the autoencoder.
public final class NFKMLXWanFunPipeline {
    /// The transformer every step runs, or the low-noise expert of an A14B release; nil where a generator
    /// stages the experts apart and hands each phase its own.
    let transformer: NFKMLXWanTransformerNet?
    /// The high-noise expert of an A14B release, which runs every step at or above the boundary.
    let highNoiseTransformer: NFKMLXWanTransformerNet?
    /// The geometry of the transformers the conditions are prepared for.
    let configuration: NFKMLXWanConfiguration
    let vae: NFKMLXWanVideoVAENet
    let latentsMean: MLXArray
    let latentsStd: MLXArray
    /// The fraction of the train timesteps at and above which the high-noise expert runs (VideoX-Fun's
    /// `boundary`, 0.900 for the Fun A14B releases).
    let boundary: Float
    let schedule: NFKMLXFlowMatchConfiguration

    init(transformer: NFKMLXWanTransformerNet?, highNoiseTransformer: NFKMLXWanTransformerNet?,
         configuration: NFKMLXWanConfiguration? = nil,
         vae: NFKMLXWanVideoVAENet, latentsMean: MLXArray, latentsStd: MLXArray, boundary: Float = 0.9,
         schedule: NFKMLXFlowMatchConfiguration = .wan) {
        self.transformer = transformer
        self.highNoiseTransformer = highNoiseTransformer
        self.configuration = configuration ?? transformer?.config ?? highNoiseTransformer!.config
        self.vae = vae
        self.latentsMean = latentsMean
        self.latentsStd = latentsStd
        self.boundary = boundary
        self.schedule = schedule
    }

    var spatialCompression: Int { NFKMLXWanVideoGenerator.compression(of: vae.configuration).spatial }
    var latentChannels: Int { vae.configuration.zDim }

    // MARK: Conditions

    /// The prepared conditions of one clip, as the reference's pipeline holds them before its loop.
    struct Prepared {
        /// The channels after the noise: the control latents (none for a camera release), then the mask
        /// latents and the masked video's latents. `[C, F', h, w]`.
        var channels: MLXArray
        /// The reference image's latent `[z, h, w]`, for a transformer with `ref_conv`.
        var reference: MLXArray?
        /// The camera's embedding regrouped by latent frame, `[24, F', H, W]`.
        var camera: MLXArray?
        /// The per-latent mask `[F', h, w]` a 16× autoencoder's release reads (VideoX-Fun's `mask`): the
        /// tokens' timestep scale, and the blend that keeps the masked video's latents.
        var latentMask: MLXArray?
        /// The masked video's latents `[z, F', h, w]`.
        var maskedLatents: MLXArray
        /// Whether every step blends the masked video's latents back where the mask is 0 (the reference's
        /// `not mask[:, :, 0].any()`).
        var blends: Bool
        /// A VACE release's context `[96, R + F', h, w]` (``prepareVACE(_:frames:height:width:)``).
        var vaceContext: MLXArray? = nil
        /// The weight of the VACE hints (`vace_context_scale`).
        var vaceScale: Float = 1
    }

    /// The autoencoder's standardized mean of a clip in `-1...1`, `[F, H, W, 3]` → `[z, F', h, w]`.
    func encoded(_ clip: MLXArray) -> MLXArray {
        let dtype = NFKReferenceRounding.parameterType(of: vae)
        let latent = vae.encode(clip.expandedDimensions(axis: 0).asType(dtype))[0].asType(.float32)
        let standardized = ((latent - latentsMean) * (1 / latentsStd)).transposed(3, 0, 1, 2)
        eval(standardized)
        return standardized
    }

    func prepare(_ conditions: NFKMLXWanFunConditions, frames: Int, height: Int, width: Int) throws -> Prepared {
        let s = spatialCompression, z = latentChannels
        let latentFrames = (frames - 1) / 4 + 1, h = height / s, w = width / s
        let config = configuration

        var maskLatents = MLXArray.zeros([4, latentFrames, h, w])
        var maskedLatents = MLXArray.zeros([z, latentFrames, h, w])
        var latentMask: MLXArray? = s >= 16 ? MLXArray.ones([latentFrames, h, w]) : nil
        var blends = false
        if let video = conditions.inpaintVideo {
            guard let mask = conditions.inpaintMask else {
                throw NFKMLXError.unsupportedConfiguration("an inpainting video needs its mask")
            }
            let binary = which(mask .>= 0.5, MLXArray(Float(1)), MLXArray(Float(0)))       // [F, H, W]
            let kept = (video * 2 - 1) * (binary .< 0.5).asType(.float32).expandedDimensions(axis: -1)
            maskedLatents = encoded(kept)
            let grouped = Self.groupedByLatentFrame(binary.expandedDimensions(axis: 0))      // [4, F', H, W]
            maskLatents = Self.resizedFirstFrameApart(1 - grouped, height: h, width: w)
            if s >= 16 {
                let resized = Self.linearResized(grouped[0 ..< 1], size: [latentFrames, h, w], alignCorners: true)[0]
                let firstFrameKept = !(resized[0] .!= 0).any().item(Bool.self)
                if firstFrameKept {
                    latentMask = concatenated([resized[0 ..< 1], MLXArray.ones([latentFrames - 1, h, w])], axis: 0)
                    blends = true
                } else {
                    latentMask = resized
                }
            }
        }

        var channels = [MLXArray]()
        var camera: MLXArray?
        if let embedding = conditions.cameraEmbedding {
            guard config.controlAdapterChannels != nil else {
                throw NFKMLXError.unsupportedConfiguration("this release has no camera control adapter")
            }
            camera = Self.cameraLatents(embedding)
        } else if let control = conditions.controlVideo {
            channels.append(encoded(control * 2 - 1))
        } else if config.inChannels > 2 * z + 4 {
            channels.append(MLXArray.zeros([z, latentFrames, h, w]))
        }
        channels += [maskLatents, maskedLatents]

        var reference: MLXArray?
        if config.referenceChannels != nil {
            reference = try conditions.referenceImage.map { image -> MLXArray in
                guard image.dim(0) == height, image.dim(1) == width else {
                    throw NFKMLXError.unsupportedConfiguration("the reference image must be the clip's size")
                }
                return encoded((image * 2 - 1).expandedDimensions(axis: 0))[0..., 0]
            } ?? MLXArray.zeros([z, h, w])
        }
        let joined = concatenated(channels, axis: 0)
        guard joined.dim(0) + z == config.inChannels else {
            throw NFKMLXError.unsupportedConfiguration(
                "the conditions give \(joined.dim(0) + z) input channels where the transformer reads \(config.inChannels)")
        }
        eval(joined)
        return Prepared(channels: joined, reference: reference, camera: camera, latentMask: latentMask,
                        maskedLatents: maskedLatents, blends: blends)
    }

    /// A pixel-frame sequence `[C, F, H, W]` regrouped as the reference regroups a mask: the first frame
    /// repeated four times ahead of the rest, then every four frames as one latent frame's four channels,
    /// `[C, 4, F', H, W]` collapsed to `[C·4, F', H, W]` channel-major.
    static func groupedByLatentFrame(_ x: MLXArray) -> MLXArray {
        let c = x.dim(0), h = x.dim(2), w = x.dim(3)
        let padded = concatenated([repeated(x[0..., 0 ..< 1], count: 4, axis: 1), x[0..., 1...]], axis: 1)
        let latentFrames = padded.dim(1) / 4
        return padded.reshaped([c, latentFrames, 4, h, w]).transposed(0, 2, 1, 3, 4).reshaped([c * 4, latentFrames, h, w])
    }

    /// The camera embedding `[6, F, H, W]` as the control adapter reads it, `[24, F', H, W]`: the first
    /// frame repeated four times, and each latent frame's four pixel frames folded into the channels
    /// (`c · 4 + j`).
    static func cameraLatents(_ embedding: MLXArray) -> MLXArray {
        let c = embedding.dim(0), h = embedding.dim(2), w = embedding.dim(3)
        let padded = concatenated([repeated(embedding[0..., 0 ..< 1], count: 4, axis: 1), embedding[0..., 1...]], axis: 1)
        let latentFrames = padded.dim(1) / 4
        return padded.reshaped([c, latentFrames, 4, h, w]).transposed(0, 2, 1, 3, 4).reshaped([c * 4, latentFrames, h, w])
    }

    /// The reference's `resize_mask(…, process_first_frame_only=True)`: the first latent frame and the
    /// rest resized apart, each trilinearly with `align_corners=False`.
    static func resizedFirstFrameApart(_ mask: MLXArray, height: Int, width: Int) -> MLXArray {
        let first = linearResized(mask[0..., 0 ..< 1], size: [1, height, width], alignCorners: false)
        guard mask.dim(1) > 1 else { return first }
        let rest = linearResized(mask[0..., 1...], size: [mask.dim(1) - 1, height, width], alignCorners: false)
        return concatenated([first, rest], axis: 1)
    }

    /// torch's `F.interpolate(mode="trilinear")` over the last three axes of `[C, T, H, W]`, separable,
    /// with torch's source-index arithmetic for either `align_corners`.
    static func linearResized(_ x: MLXArray, size: [Int], alignCorners: Bool) -> MLXArray {
        var out = x.asType(.float32)
        for (offset, target) in size.enumerated() {
            let axis = offset + 1, source = out.dim(axis)
            guard source != target else { continue }
            var lower = [Int32](), upper = [Int32](), weight = [Float]()
            let scale = alignCorners ? (target > 1 ? Float(source - 1) / Float(target - 1) : 0) : Float(source) / Float(target)
            for index in 0 ..< target {
                var position = alignCorners ? scale * Float(index) : scale * (Float(index) + 0.5) - 0.5
                position = max(position, 0)
                let low = min(Int(position), source - 1)
                lower.append(Int32(low))
                upper.append(Int32(min(low + 1, source - 1)))
                weight.append(position - Float(low))
            }
            var shape = [Int](repeating: 1, count: out.ndim)
            shape[axis] = target
            let lambda = MLXArray(weight).reshaped(shape)
            out = take(out, MLXArray(lower), axis: axis) * (1 - lambda) + take(out, MLXArray(upper), axis: axis) * lambda
        }
        return out
    }

    // MARK: Sampling

    /// The timesteps of a schedule of `steps`.
    func timesteps(_ steps: Int) -> [Float] {
        var scheduler = NFKMLXFlowMatchScheduler(schedule)
        scheduler.setTimesteps(steps, sequenceLength: 0)
        return scheduler.timesteps
    }

    /// Whether the step at timestep `t` runs on the high-noise expert of an A14B release.
    func usesHighNoiseExpert(_ t: Float) -> Bool { t >= boundary * Float(schedule.trainTimesteps) }

    /// The latents `[z, F', h, w]` after the Euler steps `indices` (every one by default) of a `steps`-step
    /// schedule over the release's flow, from `latents`: the noise at the first step, the previous phase's
    /// latents after it.
    ///
    /// @discussion Each step forms the transformer's input as the latents followed by the prepared
    /// channels, runs the conditioned and unconditioned passes where `guidance` is above 1, combines them
    /// as `uncond + guidance · (cond − uncond)`, and steps. A 16× autoencoder's release scales each token's
    /// timestep by the latent mask, subsampled by the patch. An A14B release runs the high-noise expert on
    /// every step whose timestep is at least `boundary · 1000`; `net`, where given, hands the step at a
    /// timestep its transformer instead. `onStep` reports each step's latents.
    func denoise(text: MLXArray, negative: MLXArray?, prepared: Prepared, noise latents: MLXArray, steps: Int,
                 guidance: Float, indices: Range<Int>? = nil, net chosen: ((Float) -> NFKMLXWanTransformerNet)? = nil,
                 onStep: ((Int, MLXArray) -> Void)? = nil) throws -> MLXArray {
        var scheduler = NFKMLXFlowMatchScheduler(schedule)
        scheduler.setTimesteps(steps, sequenceLength: 0)
        let indices = indices ?? 0 ..< steps
        var latents = latents.asType(.float32)
        if indices.lowerBound == 0, prepared.blends, let mask = prepared.latentMask {
            latents = (1 - mask) * prepared.maskedLatents + mask * latents
        }
        let tokenScale = prepared.latentMask.map { mask -> (values: [Float], index: MLXArray) in
            let ph = configuration.patchSize[1], pw = configuration.patchSize[2]
            let sampled = mask[0..., .stride(by: ph), .stride(by: pw)].reshaped([-1])
            return Self.distinct(sampled)
        }
        for index in indices {
            let t = scheduler.timesteps[index]
            guard let net = chosen?(t)
                ?? (usesHighNoiseExpert(t) ? highNoiseTransformer ?? transformer : transformer ?? highNoiseTransformer) else {
                throw NFKMLXError.unsupportedConfiguration("the pipeline holds no transformer for timestep \(t)")
            }
            let dtype = NFKReferenceRounding.parameterType(of: net)
            let input = (prepared.channels.dim(0) == 0 ? latents : concatenated([latents, prepared.channels], axis: 0)).asType(dtype)
            var conditioning = NFKWanConditioning(
                reference: prepared.reference?.asType(dtype), cameraControl: prepared.camera?.asType(dtype),
                vaceContext: prepared.vaceContext?.asType(dtype), vaceScale: prepared.vaceScale)
            if let tokenScale {
                conditioning.tokenTimesteps = (tokenScale.values.map { Double($0 * t) }, tokenScale.index)
            }
            let conditioned = net(input, text: text.asType(dtype), t: Double(t), conditioning: conditioning)
            var velocity = conditioned
            if guidance > 1, let negative {
                let unconditioned = net(input, text: negative.asType(dtype), t: Double(t), conditioning: conditioning)
                velocity = NFKReferenceRounding.guided(unconditioned, conditioned, scale: guidance)
            }
            try net.verifyStream()
            latents = scheduler.step(velocity: velocity.asType(.float32), sample: latents, index: index)
            if prepared.blends, let mask = prepared.latentMask {
                latents = (1 - mask) * prepared.maskedLatents + mask * latents
            }
            eval(latents)
            onStep?(index, latents)
        }
        return latents
    }

    /// The distinct values of `x` and each element's index into them.
    static func distinct(_ x: MLXArray) -> (values: [Float], index: MLXArray) {
        let flat = x.asType(.float32).asArray(Float.self)
        var values = [Float](), positions = [Float: Int32]()
        let index = flat.map { value -> Int32 in
            if let known = positions[value] { return known }
            positions[value] = Int32(values.count)
            values.append(value)
            return Int32(values.count - 1)
        }
        return (values, MLXArray(index))
    }

    /// The clip `[F, H, W, 3]` in `0...1` the latents decode to.
    func decode(_ latents: MLXArray) -> MLXArray {
        let channelsLast = latents.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
        let unstandardized = channelsLast * latentsStd.reshaped([1, 1, 1, 1, -1]) + latentsMean.reshaped([1, 1, 1, 1, -1])
        let dtype = NFKReferenceRounding.parameterType(of: vae)
        let video = clip(vae.decode(unstandardized.asType(dtype))[0].asType(.float32) / 2 + 0.5, min: 0, max: 1)
        eval(video)
        return video
    }

    /// Standard-normal noise `[z, R + F', h, w]` for a clip with `leadingFrames` reference frames ahead of it,
    /// from `seed`.
    func noise(frames: Int, height: Int, width: Int, seed: UInt64, leadingFrames: Int = 0) -> MLXArray {
        let s = spatialCompression
        MLXRandom.seed(seed)
        return MLXRandom.normal([latentChannels, leadingFrames + (frames - 1) / 4 + 1, height / s, width / s])
    }
}

// MARK: - Release

/// A Wan 2.2 Fun release directory in the original layout: `config.json` and
/// `diffusion_pytorch_model.safetensors` (the 5B releases) or `high_noise_model/` and `low_noise_model/`
/// (the A14B releases), the autoencoder's `.pth`, umT5's `.pth`, and `google/umt5-xxl/`.
enum NFKMLXWanFunRelease {

    /// Whether the directory holds an A14B release's two experts.
    static func hasExperts(_ directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent("high_noise_model/config.json").path)
    }

    /// The transformer directories: the low-noise (or only) expert first, then the high-noise expert.
    static func transformerDirectories(_ directory: URL) -> [URL] {
        hasExperts(directory)
            ? [directory.appendingPathComponent("low_noise_model"), directory.appendingPathComponent("high_noise_model")]
            : [directory]
    }

    /// The autoencoder file the release holds.
    static func autoencoderURL(_ directory: URL) throws -> URL {
        for name in [NFKMLXWanOriginalRelease.wan22AutoencoderFile, NFKMLXWanOriginalRelease.wan21AutoencoderFile] {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) holds no Wan autoencoder")
    }

    /// The tokenizer the release's `tokenizer.json` describes: the SentencePiece model's Unigram pieces
    /// behind the fast tokenizer's own normalizer, which replaces every run of two or more spaces with one
    /// and applies no character map (``fastTokenizerText(_:)``).
    static func segmenter(_ directory: URL) throws -> NFKMLXSentencePieceSegmenter {
        var model = try NFKMLXSentencePieceModel(contentsOf: directory
            .appendingPathComponent(NFKMLXWanOriginalRelease.tokenizerDirectory).appendingPathComponent("spiece.model"))
        model.precompiledCharsMap = Data()
        model.appliesNFKC = false
        model.removeExtraWhitespace = false
        return NFKMLXSentencePieceSegmenter(model: model)
    }

    /// A prompt as the fast tokenizer's normalizer leaves it: each run of two or more spaces as one space.
    /// VideoX-Fun hands the prompt to the tokenizer as written, with no other cleanup.
    static func fastTokenizerText(_ text: String) -> String {
        text.replacingOccurrences(of: " {2,}", with: " ", options: .regularExpression)
    }

    /// The prompt as the encoder reads it.
    static func prompt(_ text: String, segmenter: NFKMLXSentencePieceSegmenter, length: Int = 512) -> NFKMLXT5Prompt {
        NFKMLXT5Prompt(fastTokenizerText(text), segmenter: segmenter, length: length)
    }

    /// The text features of a prompt, `[512, dim]`: the masked encode cut at the prompt's length and padded
    /// back with zeros, as the transformer reads the reference's per-prompt cut.
    static func features(_ text: String, encoder: NFKMLXT5EncoderNet, segmenter: NFKMLXSentencePieceSegmenter,
                         length: Int = 512) -> MLXArray {
        NFKWanTextStage(encoder).features(prompt(text, segmenter: segmenter, length: length))
    }

    /// The pipeline over the release's transformer(s) and autoencoder, every weight at `precision`.
    static func pipeline(directoryURL: URL, precision: NFKMLXWeightPrecision = .float32,
                         boundary: Float = 0.9) throws -> NFKMLXWanFunPipeline {
        let nets = try transformerDirectories(directoryURL).map { directory -> NFKMLXWanTransformerNet in
            let net = NFKMLXWanTransformerNet(try NFKMLXWanOriginalRelease.transformerConfiguration(
                fromConfigURL: directory.appendingPathComponent("config.json")))
            try NFKMLXWanOriginalRelease.loadTransformer(into: net, fromDirectory: directory, precision: precision)
            return net
        }
        let autoencoder = try NFKMLXWanOriginalRelease.autoencoder(from: try autoencoderURL(directoryURL), precision: precision)
        return NFKMLXWanFunPipeline(transformer: nets[0], highNoiseTransformer: nets.count > 1 ? nets[1] : nil,
                                    vae: autoencoder.net, latentsMean: autoencoder.mean, latentsStd: autoencoder.std,
                                    boundary: boundary)
    }
}
