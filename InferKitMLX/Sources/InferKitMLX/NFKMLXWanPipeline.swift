//
//  NFKMLXWanPipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The Wan glue after the text encoder: the DiT denoised over the UniPC flow schedule with
// classifier-free guidance, then the 3D causal VAE decode of the de-normalized latent. The glue is
// measured against diffusers' own `WanPipeline` (`run_reference.py wan_pipeline`) and, for the Wan 2.2
// A14B releases, `WanPipeline` and `WanImageToVideoPipeline` over two experts
// (`run_reference.py wan22_a14b_pipeline`); each model is measured against its own reference elsewhere.
//
// An A14B release splits the schedule between two transformers: the high-noise expert runs every step
// whose timestep is at least `boundary_ratio · 1000`, at the guidance scale, and the low-noise expert the
// rest, at the second guidance scale. Guidance applies on both only where the first scale is above 1, as
// the reference decides. I2V-A14B reads the noise, a mask, and the image's latents together (36 channels).
//
// The sampler is Wan's released `UniPCMultistepScheduler` in its flow-prediction configuration
// (`NFKMLXUniPCScheduler`), whose shift the release's `scheduler_config.json` states. Wan 2.2 TI2V runs
// each token at its own timestep (`expand_timesteps`); for text to video every token carries the same
// step, which the reference measures within 2e-6 of the scalar timestep this passes.

/// Holds the pipeline stages for capture across an async boundary.
private final class NFKWanPipelineHolder: @unchecked Sendable {
    let transformer: NFKMLXWanTransformerNet?
    let vae: NFKMLXWanVideoVAENet
    init(_ transformer: NFKMLXWanTransformerNet?, _ vae: NFKMLXWanVideoVAENet) {
        self.transformer = transformer
        self.vae = vae
    }
}

/// The Wan text-to-video pipeline, from prompt features to a clip.
public final class NFKMLXWanPipeline {

    private let holder: NFKWanPipelineHolder

    /// The transformer, or nil for a pipeline whose experts the generator holds apart.
    var transformer: NFKMLXWanTransformerNet? { holder.transformer }

    /// The autoencoder's latent channels, the starting noise's channel count.
    var latentChannels: Int { holder.vae.configuration.zDim }

    /// A clip `[T, H, W, 3]` in −1…1 encoded to the autoencoder's posterior, `(mean, logVariance)`, each
    /// `[zDim, T', H', W']` and normalized by the release's statistics as the trainers normalize both.
    func normalizedMoments(_ clip: MLXArray) -> (mean: MLXArray, logVariance: MLXArray) {
        let moments = holder.vae.encodeMoments(clip.expandedDimensions(axis: 0).asType(.float32))[0]
        let z = moments.dim(-1) / 2
        func normalized(_ x: MLXArray) -> MLXArray {
            guard let latentsMean, let latentsStd else { return x }
            return (x - latentsMean) * (1 / latentsStd)
        }
        let mean = normalized(moments[0..., 0..., 0..., 0 ..< z]), logVariance = normalized(moments[0..., 0..., 0..., z...])
        eval(mean, logVariance)
        return (mean.transposed(3, 0, 1, 2), logVariance.transposed(3, 0, 1, 2))
    }
    private let latentsMean: MLXArray?
    private let latentsStd: MLXArray?
    let schedule: NFKMLXUniPCConfiguration

    /// Builds a pipeline from the DiT and the VAE.
    ///
    /// `latentsMean` and `latentsStd` are the release's per-channel latent statistics (length `zDim`), as
    /// `vae/config.json` states them; the decode reads `latent · std + mean`. A nil pair passes the latent
    /// to the VAE unchanged.
    public init(transformer: NFKMLXWanTransformerNet, vae: NFKMLXWanVideoVAENet,
                latentsMean: MLXArray? = nil, latentsStd: MLXArray? = nil,
                schedule: NFKMLXUniPCConfiguration = .wan) {
        holder = NFKWanPipelineHolder(transformer, vae)
        self.latentsMean = latentsMean
        self.latentsStd = latentsStd
        self.schedule = schedule
    }

    /// A pipeline of the autoencoder alone, for a release whose experts are held apart from it.
    init(vae: NFKMLXWanVideoVAENet, latentsMean: MLXArray?, latentsStd: MLXArray?, schedule: NFKMLXUniPCConfiguration) {
        holder = NFKWanPipelineHolder(nil, vae)
        self.latentsMean = latentsMean
        self.latentsStd = latentsStd
        self.schedule = schedule
    }

    /// Denoises prompt features into a latent `[inChannels, frames, height, width]`.
    ///
    /// - Parameters:
    ///   - textEmbeds: `[tokens, textDim]`, the umT5 features.
    ///   - negativeEmbeds: the negative prompt's features, or nil for no guidance.
    ///   - frames: the LATENT frame count.
    ///   - height: the latent height.
    ///   - width: the latent width.
    ///   - steps: the number of denoising steps.
    ///   - guidance: the classifier-free guidance scale, applied above 1 with a negative prompt.
    ///   - seed: the seed of the starting noise.
    ///   - latents: a starting latent, which replaces the seeded noise.
    public func denoise(textEmbeds: MLXArray, negativeEmbeds: MLXArray?, frames: Int, height: Int, width: Int,
                        steps: Int = 50, guidance: Float = 5, seed: UInt64 = 0,
                        latents: MLXArray? = nil) -> MLXArray {
        guard let transformer = holder.transformer else {
            preconditionFailure("this pipeline's experts are held apart; denoise through NFKWanDenoiser")
        }
        var denoiser = NFKWanDenoiser(schedule: schedule, steps: steps, text: textEmbeds, negative: negativeEmbeds,
                                      guidance: guidance, noise: latents ?? Self.noise(
                                        channels: transformer.config.outChannels, frames: frames, height: height,
                                        width: width, seed: seed))
        for index in denoiser.timesteps.indices {
            denoiser.step(index, with: transformer)
        }
        return denoiser.latent
    }

    /// The seeded starting noise `[channels, frames, height, width]`.
    static func noise(channels: Int, frames: Int, height: Int, width: Int, seed: UInt64) -> MLXArray {
        MLXRandom.seed(seed)
        return MLXRandom.normal([channels, frames, height, width])
    }

    /// The image-to-video condition `[4 + zDim, F', H', W']` I2V-A14B reads beside the noise: a mask marking
    /// the given frames, and the standardized latent mean of the clip that holds `image` first (and
    /// `lastImage` last, where given) and zeros between.
    ///
    /// - Parameters:
    ///   - image: `[H, W, 3]` in −1…1, already at the clip's size.
    ///   - lastImage: the clip's last frame, the same size, or nil.
    ///   - frames: the clip's pixel frame count, `4k + 1`.
    func imageCondition(_ image: MLXArray, lastImage: MLXArray?, frames: Int) -> MLXArray {
        let height = image.dim(0), width = image.dim(1)
        var clip = [image.expandedDimensions(axis: 0).asType(.float32)]
        let between = frames - 1 - (lastImage == nil ? 0 : 1)
        if between > 0 {
            clip.append(MLXArray.zeros([between, height, width, 3]))
        }
        if let lastImage {
            clip.append(lastImage.expandedDimensions(axis: 0).asType(.float32))
        }
        let dtype = NFKReferenceRounding.parameterType(of: holder.vae)
        var latent = holder.vae.encode(concatenated(clip, axis: 0).expandedDimensions(axis: 0).asType(dtype))[0]
            .asType(.float32)
        if let latentsMean, let latentsStd {
            latent = (latent - latentsMean) * (1 / latentsStd)
        }
        let channelsFirst = latent.transposed(3, 0, 1, 2)
        let marked: Set<Int> = lastImage == nil ? [0] : [0, frames - 1]
        let mask = Self.imageMask(latentFrames: channelsFirst.dim(1), height: channelsFirst.dim(2),
                                  width: channelsFirst.dim(3), frames: frames, marked: marked)
        let condition = concatenated([mask, channelsFirst], axis: 0)
        eval(condition)
        return condition
    }

    /// The image-to-video mask `[4, latentFrames, height, width]`: the first latent frame stands for four
    /// copies of pixel frame 0, each later one for four pixel frames in turn, and a channel is 1 where its
    /// pixel frame is `marked`.
    static func imageMask(latentFrames: Int, height: Int, width: Int, frames: Int, marked: Set<Int>) -> MLXArray {
        var values = [Float](repeating: 0, count: 4 * latentFrames)
        for channel in 0 ..< 4 {
            for frame in 0 ..< latentFrames {
                let expanded = 4 * frame + channel
                let pixel = expanded < 4 ? 0 : expanded - 3
                values[channel * latentFrames + frame] = pixel < frames && marked.contains(pixel) ? 1 : 0
            }
        }
        return broadcast(MLXArray(values).reshaped([4, latentFrames, 1, 1]), to: [4, latentFrames, height, width])
    }

    /// Decodes a latent `[C, F, H, W]` into a clip `[1, T, H, W, 3]` in −1…1, after de-normalizing it by
    /// the release's statistics.
    public func decode(_ latent: MLXArray) -> MLXArray {
        var channelsLast = latent.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
        if let latentsMean, let latentsStd {
            channelsLast = channelsLast * latentsStd.reshaped([1, 1, 1, 1, -1]) + latentsMean.reshaped([1, 1, 1, 1, -1])
        }
        let video = holder.vae.decode(channelsLast)
        eval(video)
        return video
    }

    /// ``denoise(textEmbeds:negativeEmbeds:frames:height:width:steps:guidance:seed:latents:)`` then
    /// ``decode(_:)``.
    public func generate(textEmbeds: MLXArray, negativeEmbeds: MLXArray?, frames: Int, height: Int,
                         width: Int, steps: Int = 20, guidance: Float = 5, seed: UInt64 = 0) -> MLXArray {
        decode(denoise(textEmbeds: textEmbeds, negativeEmbeds: negativeEmbeds, frames: frames, height: height,
                       width: width, steps: steps, guidance: guidance, seed: seed))
    }
}

/// The state of one denoising run: the latent and the scheduler's history, advanced a step at a time by the
/// transformer the step belongs to, so a generator can load the second expert between steps.
struct NFKWanDenoiser {
    private(set) var latent: MLXArray
    private var scheduler: NFKMLXUniPCScheduler
    private let text: MLXArray
    private let negative: MLXArray?
    private let guidance: Float
    private let lowNoiseGuidance: Float
    private let boundary: Float?
    private let condition: MLXArray?

    /// - Parameters:
    ///   - boundary: the timestep below which the low-noise expert runs (`boundary_ratio · 1000`), or nil.
    ///   - lowNoiseGuidance: the low-noise expert's scale; nil takes `guidance`.
    ///   - condition: the channels read beside the latent (I2V-A14B's mask and image latents), or nil.
    init(schedule: NFKMLXUniPCConfiguration, steps: Int, text: MLXArray, negative: MLXArray?, guidance: Float,
         lowNoiseGuidance: Float? = nil, boundary: Float? = nil, condition: MLXArray? = nil, noise: MLXArray) {
        scheduler = NFKMLXUniPCScheduler(schedule)
        scheduler.setTimesteps(steps)
        self.text = text
        self.negative = negative
        self.guidance = guidance
        self.lowNoiseGuidance = lowNoiseGuidance ?? guidance
        self.boundary = boundary
        self.condition = condition
        latent = noise
    }

    var timesteps: [Float] { scheduler.timesteps }

    /// Whether step `index` belongs to the low-noise expert.
    func usesLowNoiseExpert(_ index: Int) -> Bool {
        boundary.map { scheduler.timesteps[index] < $0 } ?? false
    }

    /// Advances one step with `transformer`, the expert ``usesLowNoiseExpert(_:)`` names.
    mutating func step(_ index: Int, with transformer: NFKMLXWanTransformerNet) {
        // The reference keeps the latents float32 and hands the transformer its own type.
        let dtype = NFKReferenceRounding.parameterType(of: transformer)
        let input = (condition.map { concatenated([latent, $0], axis: 0) } ?? latent).asType(dtype)
        let step = MLXArray([scheduler.timesteps[index]])
        let conditioned = transformer(input, text: text.asType(dtype), t: step)
        var velocity = conditioned
        if guidance > 1, let negative {
            let unconditioned = transformer(input, text: negative.asType(dtype), t: step)
            velocity = NFKReferenceRounding.guided(unconditioned, conditioned,
                                                   scale: usesLowNoiseExpert(index) ? lowNoiseGuidance : guidance)
        }
        latent = scheduler.step(velocity: velocity, sample: latent, index: index)
        eval(latent)
    }
}

// MARK: - Release

extension NFKMLXWanPipeline {

    /// Builds the pipeline from a diffusers Wan 2.1 or 2.2 text-to-video release directory: the DiT from
    /// `transformer/` at its stored precision, the autoencoder from `vae/` at float32 with the latent
    /// statistics its `config.json` states, and the flow shift from `scheduler/scheduler_config.json`.
    /// The caller supplies the umT5 prompt features (``NFKMLXT5Encoder``); ``NFKMLXWanVideoGenerator``
    /// runs the whole release, text encoder included. Blocking on the load; run it off the render
    /// thread. Introduced in InferKit 0.4.0.
    public static func pipeline(directoryURL: URL) throws -> NFKMLXWanPipeline {
        let transformerDirectory = directoryURL.appendingPathComponent("transformer")
        let vaeDirectory = directoryURL.appendingPathComponent("vae")
        let transformer = NFKMLXWanTransformerNet(try NFKMLXWanTransformerNet.configuration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json")))
        try NFKMLXWanTransformerNet.loadWeights(into: transformer, fromDirectory: transformerDirectory,
                                                precision: .checkpoint)
        let autoencoder = try NFKMLXWanVideoVAENet.configuration(
            fromHuggingFace: vaeDirectory.appendingPathComponent("config.json"))
        let vae = NFKMLXWanVideoVAENet(autoencoder.configuration)
        try NFKMLXWanVideoVAENet.loadWeights(into: vae, fromDirectory: vaeDirectory)
        let schedule = try NFKMLXWanRelease.schedule(
            fromHuggingFace: directoryURL.appendingPathComponent("scheduler/scheduler_config.json"))
        return NFKMLXWanPipeline(transformer: transformer, vae: vae, latentsMean: autoencoder.latentsMean,
                                 latentsStd: autoencoder.latentsStd, schedule: schedule)
    }
}

extension NFKMLXWanTransformerNet {

    /// The DiT geometry a diffusers `transformer/config.json` describes. An image-conditioned release
    /// (an `image_dim` or `added_kv_proj_dim`) is refused: this is the text-to-video transformer.
    /// Introduced in InferKit 0.4.0.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXWanConfiguration {
        try NFKMLXWanRelease.transformerConfiguration(fromHuggingFace: url)
    }

    /// Loads a release's `transformer/` directory (one file or a shard index). The module keys are the
    /// release's own; the patchify convolution moves from `[out, in, t, h, w]` to MLX's
    /// `[out, t, h, w, in]`. `.checkpoint` keeps the release's stored type. Introduced in InferKit 0.4.0.
    public static func loadWeights(into net: NFKMLXWanTransformerNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        try NFKMLXWanRelease.loadTransformer(into: net, fromDirectory: directory, precision: precision)
    }
}

extension NFKMLXWanVideoVAENet {

    /// The autoencoder geometry a diffusers `vae/config.json` describes, with the per-channel latent
    /// statistics the decode reads (nil where the release states none). Introduced in InferKit 0.4.0.
    public static func configuration(fromHuggingFace url: URL) throws
        -> (configuration: NFKMLXWanVAEConfiguration, latentsMean: MLXArray?, latentsStd: MLXArray?) {
        let read = try NFKMLXWanRelease.vaeConfiguration(fromHuggingFace: url)
        return (read.configuration, read.mean, read.std)
    }

    /// Loads a release's `vae/` directory: a 5-D causal convolution loads as `[out, t, h, w, in]`, a 4-D
    /// spatial one as `[out, h, w, in]`, and an RMS scale stored with trailing singleton axes flattens.
    /// Introduced in InferKit 0.4.0.
    public static func loadWeights(into net: NFKMLXWanVideoVAENet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        try NFKMLXWanRelease.loadVAE(into: net, fromDirectory: directory, precision: precision)
    }
}
