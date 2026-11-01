//
//  NFKMLXLTX2GeneratorDecodePaths.swift
//  InferKitMLX
//
//  The LTX-2.5 decode-side paths of NFKMLXLTX2Generator: the two-stage distilled recipe through the
//  spatial latent upsampler, the temporal latent upsampler, and the diffusion decoder.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXRandom

/// The LTX-2.5 decode-side components a release directory holds, loaded at float32 when a path asks for
/// them, and the switches that turn the paths on.
final class NFKLTX2DecodePaths {
    var decodesWithDiffusionDecoder = false
    var refinesUpsampled = false
    var upsamplesTemporally = false
    var refinementSigmas = NFKMLXLTX2Pipeline.refinementSigmas
    var refinementNoiseScale: Float?
    var tiling: NFKMLXLTX2DecodeTiling? = NFKMLXLTX2DecodeTiling()

    let spatialUpsampler: NFKMLXStage<NFKMLXLTX2LatentUpsamplerNet>?
    let temporalUpsampler: NFKMLXStage<NFKMLXLTX2LatentUpsamplerNet>?
    let diffusionDecoder: NFKMLXStage<NFKMLXLTX2DiffusionDecoder>?

    /// The folders of `directory` the paths read; a folder the release lacks leaves its path unavailable.
    init(directory: URL?) {
        func present(_ name: String) -> URL? {
            guard let url = directory?.appendingPathComponent(name),
                  FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path) else {
                return nil
            }
            return url
        }
        spatialUpsampler = present("latent_upsampler").map { url in
            NFKMLXStage { try NFKMLXLTX2LatentUpsamplers.load(from: url) }
        }
        temporalUpsampler = present("temporal_latent_upsampler").map { url in
            NFKMLXStage { try NFKMLXLTX2LatentUpsamplers.load(from: url) }
        }
        diffusionDecoder = present("diffusion_decoder").map { url in
            NFKMLXStage { try NFKMLXLTX2DiffusionDecoders.load(from: url) }
        }
    }

    /// The bytes the upsamplers and the diffusion decoder hold at float32, for the residency plan.
    static func footprint(directory: URL) throws -> (upsamplers: Int, decoder: Int) {
        func bytes(_ name: String) throws -> Int {
            let url = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path) else { return 0 }
            return try NFKMLXStageWeights.bytes(inDirectory: url, precision: .float32)
        }
        return (try bytes("latent_upsampler") + bytes("temporal_latent_upsampler"), try bytes("diffusion_decoder"))
    }

    /// Throws when a switch is on whose component the release lacks.
    func verifyAvailable() throws {
        func refuse(_ path: String, _ folder: String) -> NFKMLXError {
            .unsupportedConfiguration("\(path) needs the release's \(folder)/, which this directory lacks")
        }
        if decodesWithDiffusionDecoder, diffusionDecoder == nil { throw refuse("the diffusion decode", "diffusion_decoder") }
        if refinesUpsampled, spatialUpsampler == nil { throw refuse("the two-stage refinement", "latent_upsampler") }
        if upsamplesTemporally, temporalUpsampler == nil {
            throw refuse("the temporal upsampling", "temporal_latent_upsampler")
        }
    }
}

/// The request parameters that turn the LTX-2.5 decode-side paths on. Introduced in InferKit 0.4.0.
extension NFKMLXLTX2ParameterKey {
    /// Whether the clip decodes through the diffusion decoder, as a boolean `NSNumber`. See
    /// ``NFKMLXLTX2Generator/decodesWithDiffusionDecoder``. Introduced in InferKit 0.4.0.
    @objc public static let diffusionDecode = "NFKMLXLTX2ParameterDiffusionDecode"
    /// Whether the clip samples by the two-stage distilled recipe, as a boolean `NSNumber`. See
    /// ``NFKMLXLTX2Generator/refinesUpsampled``. Introduced in InferKit 0.4.0.
    @objc public static let twoStageRefinement = "NFKMLXLTX2ParameterTwoStageRefinement"
    /// Whether the clip's latent is upsampled in time before it decodes, as a boolean `NSNumber`. See
    /// ``NFKMLXLTX2Generator/upsamplesTemporally``. Introduced in InferKit 0.4.0.
    @objc public static let temporalUpsampling = "NFKMLXLTX2ParameterTemporalUpsampling"
}

extension NFKMLXLTX2Generator {

    /// Whether the clip decodes through LTX-2.5's diffusion decoder (`diffusion_decoder/`, the default
    /// decode of its modular pipeline) rather than the convolutional autoencoder. Off by default. The
    /// decoder draws its own pixel noise from the clip's seed and decodes in the reference's tiles.
    /// Introduced in InferKit 0.4.0.
    @objc public var decodesWithDiffusionDecoder: Bool {
        get { decodePaths.decodesWithDiffusionDecoder }
        set { decodePaths.decodesWithDiffusionDecoder = newValue }
    }

    /// Whether the clip samples by LTX-2.5's two-stage distilled recipe: stage 1 at half the width and
    /// height, the spatial latent upsampler (`latent_upsampler/`), then ``refinementSigmas`` at the full
    /// size from the upsampled latent renoised at ``refinementNoiseScale``. Off by default. Introduced in
    /// InferKit 0.4.0.
    @objc public var refinesUpsampled: Bool {
        get { decodePaths.refinesUpsampled }
        set { decodePaths.refinesUpsampled = newValue }
    }

    /// Whether the final latent passes through LTX-2.5's temporal latent upsampler
    /// (`temporal_latent_upsampler/`) before it decodes: `F` latent frames become `2 F − 1`, so a clip of
    /// `n` frames decodes to `2 n − 1` at ``outputFrameRate``. Off by default. Introduced in InferKit 0.4.0.
    @objc public var upsamplesTemporally: Bool {
        get { decodePaths.upsamplesTemporally }
        set { decodePaths.upsamplesTemporally = newValue }
    }

    /// The second stage's sigmas in the two-stage recipe, the reference's `STAGE_2_DISTILLED_SIGMA_VALUES`
    /// by default. Introduced in InferKit 0.4.0.
    @objc public var refinementSigmas: [NSNumber] {
        get { decodePaths.refinementSigmas.map { NSNumber(value: $0) } }
        set { decodePaths.refinementSigmas = newValue.map(\.floatValue) }
    }

    /// The noise level the second stage renoises the upsampled latent at; nil, the default, takes the
    /// first of ``refinementSigmas``, as the reference's README passes. Introduced in InferKit 0.4.0.
    @objc public var refinementNoiseScale: NSNumber? {
        get { decodePaths.refinementNoiseScale.map { NSNumber(value: $0) } }
        set { decodePaths.refinementNoiseScale = newValue?.floatValue }
    }

    /// Whether the diffusion decoder decodes in the reference's overlapping tiles (768 × 768 pixels and 80
    /// frames, strides 704 and 56) where the clip exceeds one tile. On by default; untiled, a clip's
    /// memory grows with its size. Introduced in InferKit 0.4.0.
    @objc public var tilesDiffusionDecode: Bool {
        get { decodePaths.tiling != nil }
        set { decodePaths.tiling = newValue ? NFKMLXLTX2DecodeTiling() : nil }
    }

    /// Whether the release holds the diffusion decoder. Introduced in InferKit 0.4.0.
    @objc public var hasDiffusionDecoder: Bool { decodePaths.diffusionDecoder != nil }

    /// Whether the release holds the spatial latent upsampler the two-stage recipe reads. Introduced in
    /// InferKit 0.4.0.
    @objc public var hasSpatialUpsampler: Bool { decodePaths.spatialUpsampler != nil }

    /// Whether the release holds the temporal latent upsampler. Introduced in InferKit 0.4.0.
    @objc public var hasTemporalUpsampler: Bool { decodePaths.temporalUpsampler != nil }

    /// The frame rate the clip plays at: ``frameRate``, doubled where ``upsamplesTemporally``. Introduced in
    /// InferKit 0.4.0.
    @objc public var outputFrameRate: Float { upsamplesTemporally ? 2 * frameRate : frameRate }

    /// The parameters ``inferenceBackend()`` reads for the decode-side paths.
    static let decodePathParameterKeys: Set<String> = [
        NFKMLXLTX2ParameterKey.diffusionDecode, NFKMLXLTX2ParameterKey.twoStageRefinement,
        NFKMLXLTX2ParameterKey.temporalUpsampling,
    ]

    /// Applies a request's decode-path switches and returns the closure that restores the generator's.
    func applyingDecodePathParameters(_ parameter: (String) -> NSNumber?) -> () -> Void {
        let saved = (decodesWithDiffusionDecoder, refinesUpsampled, upsamplesTemporally)
        decodesWithDiffusionDecoder = parameter(NFKMLXLTX2ParameterKey.diffusionDecode)?.boolValue
            ?? decodesWithDiffusionDecoder
        refinesUpsampled = parameter(NFKMLXLTX2ParameterKey.twoStageRefinement)?.boolValue ?? refinesUpsampled
        upsamplesTemporally = parameter(NFKMLXLTX2ParameterKey.temporalUpsampling)?.boolValue ?? upsamplesTemporally
        return { [self] in
            (decodesWithDiffusionDecoder, refinesUpsampled, upsamplesTemporally) = saved
        }
    }

    /// The release components a download fetches for `variant`: LTX-2.5 adds the diffusion decoder, both
    /// latent upsamplers (2.1 GB), and the duration head, and ``NFKMLXLTX2Variant/ltx25Full`` reads the full
    /// transformer's folder in place of the distilled one.
    static func releaseComponents(for variant: NFKMLXLTX2Variant) -> [NFKMLXReleaseComponent] {
        guard variant != .ltx23 else { return releaseComponents }
        let components = releaseComponents.map {
            $0.required.contains("transformer/config.json") ? transformerComponent(variant.transformerFolder) : $0
        }
        return components + ["diffusion_decoder", "latent_upsampler", "temporal_latent_upsampler", "duration_head"].map {
            NFKMLXReleaseComponent(required: ["\($0)/config.json", "\($0)/diffusion_pytorch_model.safetensors"])
        }
    }

    /// The joint denoising of one clip inside the pipeline stage: stage 1, the two-stage refinement and the
    /// temporal upsampling where they are on. Returns the denormalized video latent `[1, F, H, W, C]` and
    /// the packed audio latents.
    func sampleLatents(_ pipeline: NFKMLXLTX2Pipeline, streams: [MLXArray], frames: Int, width: Int, height: Int,
                       seed: UInt64, videoGuidance: NFKMLXLTX2Guidance, audioGuidance: NFKMLXLTX2Guidance,
                       onStep: ((Int) -> Void)?) throws -> (video: MLXArray, audio: MLXArray) {
        try decodePaths.verifyAvailable()
        let refines = decodePaths.refinesUpsampled
        let grid = NFKMLXLTX2Pipeline.grid(frames: frames, width: refines ? width / 2 : width,
                                           height: refines ? height / 2 : height, frameRate: frameRate,
                                           video: video, audio: audio)
        guard frames >= 1, grid.height > 0, grid.width > 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                refines ? "a two-stage clip is at least one frame of 64×64 pixels" : "a clip is at least one frame of 32×32 pixels")
        }
        let (videoKey, audioKey) = MLXRandom.split(key: MLXRandom.key(seed))
        let videoNoise = MLXRandom.normal(
            [1, grid.frames * grid.height * grid.width, pipeline.transformer.config.inChannels], key: videoKey)
        let audioNoise = MLXRandom.normal([1, grid.audioFrames, pipeline.transformer.config.audioInChannels], key: audioKey)
        let dtype = pipeline.transformer.projIn.weight.dtype
        let (text, audioText) = (streams[0].asType(dtype), streams[1].asType(dtype))
        var (videoLatents, audioLatents) = try pipeline.denoise(
            video: videoNoise, audio: audioNoise, text: text, audioText: audioText, frames: grid.frames,
            height: grid.height, width: grid.width, audioFrames: grid.audioFrames, frameRate: frameRate, steps: steps,
            videoGuidance: videoGuidance, audioGuidance: audioGuidance, perturbedBlocks: perturbedBlocks,
            schedule: schedule, sigmas: sigmas?.map(\.floatValue), onStep: onStep)
        var (latentHeight, latentWidth) = (grid.height, grid.width)
        if refines, let stage = decodePaths.spatialUpsampler {
            let first = (sigmas?.count ?? steps)
            let refined = try staging.use(stage) { upsampler in
                let (videoKey, audioKey) = MLXRandom.split(key: MLXRandom.key(seed &+ 1))
                let side = upsampler.configuration.spatialScale
                let upsampledCells = grid.frames * Int(Double(grid.height) * side) * Int(Double(grid.width) * side)
                let result = try pipeline.refine(
                    video: videoLatents, audio: audioLatents, frames: grid.frames, height: grid.height,
                    width: grid.width, upsampler: upsampler,
                    videoNoise: MLXRandom.normal([1, upsampledCells, videoLatents.dim(2)], key: videoKey),
                    audioNoise: MLXRandom.normal(audioLatents.shape, key: audioKey), text: text, audioText: audioText,
                    audioFrames: grid.audioFrames, frameRate: frameRate, sigmas: decodePaths.refinementSigmas,
                    noiseScale: decodePaths.refinementNoiseScale, videoGuidance: videoGuidance,
                    audioGuidance: audioGuidance, perturbedBlocks: perturbedBlocks, schedule: schedule,
                    onStep: onStep.map { report in { report(first + $0) } })
                eval(result.video, result.audio)
                return result
            }
            (videoLatents, audioLatents, latentHeight, latentWidth) = refined
        }
        var latent = pipeline.videoLatent(videoLatents, frames: grid.frames, height: latentHeight, width: latentWidth)
        if decodePaths.upsamplesTemporally, let stage = decodePaths.temporalUpsampler {
            latent = try staging.use(stage) { upsampler in
                let upsampled = NFKMLXLTX2Pipeline.upsampled(latent, by: upsampler)
                eval(upsampled)
                return upsampled
            }
        }
        eval(latent, audioLatents)
        return (latent, audioLatents)
    }

    /// The denoising steps a clip reports progress over: stage 1's and, where the two-stage recipe runs,
    /// the refinement's.
    var reportedSteps: Int {
        (sigmas?.count ?? steps) + (refinesUpsampled ? decodePaths.refinementSigmas.count : 0)
    }

    /// The denormalized latent decoded by the diffusion decoder to the clip `[frames, height, width, 3]` in
    /// `[0, 1]`, its pixel noise drawn from `seed`.
    func diffusionDecodedClip(_ latent: MLXArray, seed: UInt64) throws -> MLXArray {
        guard let stage = decodePaths.diffusionDecoder else { try decodePaths.verifyAvailable(); return latent }
        var key = MLXRandom.key(seed &+ 2)
        let tiling = decodePaths.tiling
        return try staging.with(stage) { decoder in
            let decoded = decoder.decodeLatent(latent, tiling: tiling) { shape in
                let (next, draw) = MLXRandom.split(key: key)
                key = next
                return MLXRandom.normal(shape, key: draw)
            }
            return MLX.clip(decoded[0] / 2 + 0.5, min: 0, max: 1)
        }
    }
}
