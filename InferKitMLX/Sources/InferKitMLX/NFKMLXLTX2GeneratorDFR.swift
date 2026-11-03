//
//  NFKMLXLTX2GeneratorDFR.swift
//  InferKitMLX
//
//  LTX-2.5's Diffusion Fidelity Rendering recipe in NFKMLXLTX2Generator: the 1080p recipe of the diffusers
//  LTX-2 docs, opt-in.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXRandom

/// The request parameters of LTX-2.5's DFR recipe. Introduced in InferKit 0.4.0.
extension NFKMLXLTX2ParameterKey {
    /// Whether the clip renders by the DFR recipe, as a boolean `NSNumber`. See
    /// ``NFKMLXLTX2Generator/rendersWithDFR``. Introduced in InferKit 0.4.0.
    @objc public static let dfr = "NFKMLXLTX2ParameterDFR"
    /// The DFR temporal rounds, 0, 1, or 2, as an `NSNumber`. See ``NFKMLXLTX2Generator/dfrTemporalRounds``.
    /// Introduced in InferKit 0.4.0.
    @objc public static let dfrTemporalRounds = "NFKMLXLTX2ParameterDFRTemporalRounds"
    /// Whether DFR runs its third spatial stage, as a boolean `NSNumber`. See
    /// ``NFKMLXLTX2Generator/dfrEpilogue``. Introduced in InferKit 0.4.0.
    @objc public static let dfrEpilogue = "NFKMLXLTX2ParameterDFREpilogue"
}

extension NFKMLXLTX2Generator {

    /// The repository of LTX-2.5's spatial detailing IC-LoRA, which DFR's full-size passes attend with.
    @objc public static let detailingLoRARepo = "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"
    /// The IC-LoRA's file in ``detailingLoRARepo``.
    @objc public static let detailingLoRAFile = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"

    /// Whether the clip renders by LTX-2.5's Diffusion Fidelity Rendering recipe. Off by default.
    ///
    /// @discussion Stage 1 samples at half the width and height (each a multiple of 64) and generates a
    /// keyframe slot at the end of every 24- or 32-frame segment; the spatial upsampler doubles its video and
    /// its slots; stage 2 refines them at the full size over the stage-2 sigmas, attending to stage 1's video
    /// through the detailing IC-LoRA (``loadDetailingLoRA(at:)``) at ``detailingStrength``; each of
    /// ``dfrTemporalRounds`` doubles the frame count and the frame rate, inventing a slot mid-segment; and
    /// ``dfrEpilogue`` refines the result again at twice the size in overlapping tiles. The soundtrack is stage
    /// 1's, cut to the clip. Without the IC-LoRA loaded, stage 2 runs without it or its reference. Introduced in
    /// InferKit 0.4.0.
    @objc public var rendersWithDFR: Bool {
        get { decodePaths.rendersWithDFR }
        set { decodePaths.rendersWithDFR = newValue }
    }

    /// The DFR temporal rounds, 0 to 2, 1 by default; each doubles the frames and the frame rate. Introduced in
    /// InferKit 0.4.0.
    @objc public var dfrTemporalRounds: Int {
        get { decodePaths.dfrTemporalRounds }
        set { decodePaths.dfrTemporalRounds = newValue }
    }

    /// Whether DFR runs its third spatial stage, the output then twice the requested width and height. Off by
    /// default. Introduced in InferKit 0.4.0.
    @objc public var dfrEpilogue: Bool {
        get { decodePaths.dfrEpilogue }
        set { decodePaths.dfrEpilogue = newValue }
    }

    /// The detailing IC-LoRA's adapter weight in DFR's full-size passes, 0.5 as the reference's recipe sets
    /// it. Introduced in InferKit 0.4.0.
    @objc public var detailingStrength: Float {
        get { decodePaths.detailingStrength }
        set { decodePaths.detailingStrength = newValue }
    }

    /// The stills the DFR recipe conditions on, each at its latent frame and strength; empty for text-to-video.
    /// Each is resized to cover a pass, center-cropped, and encoded at that pass's size, without the reference's
    /// optional H.264 re-compression. Introduced in InferKit 0.4.0.
    public var imageConditions: [NFKMLXLTX2ImageCondition] {
        get { decodePaths.imageConditions }
        set { decodePaths.imageConditions = newValue }
    }

    /// The still the DFR recipe opens on, at its first frame and full strength, for Objective-C; nil for
    /// text-to-video. Setting it replaces ``imageConditions``. Introduced in InferKit 0.4.0.
    @objc public var firstFrameImage: CGImage? {
        get { nil }
        set { imageConditions = newValue.map { [NFKMLXLTX2ImageCondition(image: $0)] } ?? [] }
    }

    /// Whether the detailing IC-LoRA is loaded. Introduced in InferKit 0.4.0.
    @objc public var hasDetailingLoRA: Bool { decodePaths.detailingLoRA != nil }

    /// Loads the detailing IC-LoRA (``detailingLoRAFile``), which DFR's full-size passes then attend with.
    /// Introduced in InferKit 0.4.0.
    @objc(loadDetailingLoRAAtURL:error:)
    public func loadDetailingLoRA(at url: URL) throws {
        let arrays = try NFKMLXLTX2LoRA.arrays(at: url)
        guard arrays.keys.contains(where: { NFKMLXLTX2LoRA.modulePath($0) != nil }) else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) holds no LTX-2 transformer LoRA")
        }
        staging.exclusively { decodePaths.detailingLoRA = arrays }
    }

    /// Downloads ``detailingLoRARepo`` (`repo` where given; gated, its license accepted once on the Hub) and
    /// loads it. The call blocks on the network. Introduced in InferKit 0.4.0.
    @objc(loadDetailingLoRAFromRepo:revision:cacheDirectoryURL:error:)
    public func loadDetailingLoRA(repo: String?, revision: String?, cacheDirectoryURL: URL?) throws {
        let directory = try NFKMLXReleaseDownload.directory(
            repo: repo ?? Self.detailingLoRARepo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            components: [NFKMLXReleaseComponent(required: [Self.detailingLoRAFile])])
        try loadDetailingLoRA(at: directory.appendingPathComponent(Self.detailingLoRAFile))
    }

    /// Applies a request's `NFKInputImage`, the still the DFR recipe opens on, and returns the closure that
    /// restores the generator's conditions.
    func applyingDFRImage(_ request: NFKInferenceRequest) throws -> () -> Void {
        let saved = imageConditions
        if let image = try NFKMLXGenerationBackend.image(request, NFKInputImage) {
            imageConditions = [NFKMLXLTX2ImageCondition(image: image)]
        }
        return { [self] in imageConditions = saved }
    }

    /// The parameters ``inferenceBackend()`` reads for DFR.
    static let dfrParameterKeys: Set<String> = [
        NFKMLXLTX2ParameterKey.dfr, NFKMLXLTX2ParameterKey.dfrTemporalRounds, NFKMLXLTX2ParameterKey.dfrEpilogue,
    ]

    /// Applies a request's DFR switches and returns the closure that restores the generator's.
    func applyingDFRParameters(_ parameter: (String) -> NSNumber?) -> () -> Void {
        let saved = (rendersWithDFR, dfrTemporalRounds, dfrEpilogue)
        rendersWithDFR = parameter(NFKMLXLTX2ParameterKey.dfr)?.boolValue ?? rendersWithDFR
        dfrTemporalRounds = parameter(NFKMLXLTX2ParameterKey.dfrTemporalRounds)?.intValue ?? dfrTemporalRounds
        dfrEpilogue = parameter(NFKMLXLTX2ParameterKey.dfrEpilogue)?.boolValue ?? dfrEpilogue
        return { [self] in (rendersWithDFR, dfrTemporalRounds, dfrEpilogue) = saved }
    }

    /// The soundtrack cut to a DFR clip of `frames` requested frames at the output rate; any other clip's as
    /// given.
    func trimmedSoundtrack(_ waveform: MLXArray, frames: Int) -> MLXArray {
        guard rendersWithDFR else { return waveform }
        let outputFrames = (frames - 1) * (1 << dfrTemporalRounds) + 1
        let samples = Int((Double(outputFrames) / Double(outputFrameRate) * Double(audioSampleRate)).rounded(.toNearestOrEven))
        return waveform[0 ..< min(samples, waveform.dim(0))]
    }

    /// The DFR recipe inside the pipeline stage: the raw video latent trimmed to the requested frames and stage
    /// 1's packed normalized audio.
    func dfrLatents(_ pipeline: NFKMLXLTX2Pipeline, streams: [MLXArray], frames: Int, width: Int, height: Int,
                    seed: UInt64, onStep: ((Int) -> Void)?) throws -> (video: MLXArray, audio: MLXArray) {
        guard variant == .ltx25 else {
            throw NFKMLXError.unsupportedConfiguration("DFR generates keyframe slots, which LTX-2.5's transformer marks")
        }
        guard width % 64 == 0, height % 64 == 0, width >= 64, height >= 64 else {
            throw NFKMLXError.unsupportedConfiguration("DFR halves the clip for stage 1, so its sides are multiples of 64")
        }
        let paths = decodePaths
        let r = pipeline.temporalCompression
        let dtype = pipeline.transformer.projIn.weight.dtype
        let text = streams[0][(-1)...].asType(dtype), audioText = streams[1][(-1)...].asType(dtype)
        var counter: UInt64 = 0
        func draw(_ shape: [Int]) -> MLXArray {
            counter += 1
            return MLXRandom.normal(shape, key: MLXRandom.key(seed &+ counter &* 0x9E37_79B9))
        }
        var reported = 0
        func reporter(_ offset: Int) -> ((Int) -> Void)? { onStep.map { report in { report(offset + $0) } } }

        if let lora = paths.detailingLoRA, !pipeline.transformer.leafModules().flattened().contains(where: {
            $0.1 is NFKMLXLoRAAdapter
        }) {
            try NFKMLXLTX2LoRA.apply(lora, to: pipeline.transformer, strength: 0)
        }
        let detailing = paths.detailingLoRA != nil
        defer { NFKMLXLTX2LoRA.setStrength(0, in: pipeline.transformer) }
        NFKMLXLTX2LoRA.setStrength(0, in: pipeline.transformer)

        func conditions(width: Int, height: Int, frames: Int) throws -> [NFKMLXLTX2DFRCondition] {
            guard !paths.imageConditions.isEmpty, let stage = paths.videoEncoder else { return [] }
            return try staging.use(stage) { encoder in
                pipeline.encodedConditions(paths.imageConditions, width: width, height: height, frames: frames,
                                           encoder: encoder)
            }
        }
        let canvasFrames = try NFKMLXLTX2DFRLayout.canvas(frames: frames, temporalCompression: r).frames

        let stageOneSigmas = sigmas?.map(\.floatValue) ?? NFKMLXLTX2Generator.distilledSigmas
        let first = try pipeline.dfrPass(
            text: text, audioText: audioText, frames: frames, latentHeight: height / 2 / 32, latentWidth: width / 2 / 32,
            frameRate: frameRate, sigmas: stageOneSigmas,
            conditions: try conditions(width: width / 2, height: height / 2, frames: canvasFrames), schedule: schedule,
            sequenceNoise: draw, audioNoise: draw, onStep: reporter(reported))
        reported += stageOneSigmas.count
        let (video, slots) = try staging.use(paths.spatialUpsampler!) { upsampler -> (MLXArray, MLXArray) in
            let up = (NFKMLXLTX2Pipeline.upsampled(first.video, by: upsampler),
                      NFKMLXLTX2Pipeline.upsampled(first.keyframes!, by: upsampler))
            eval(up.0, up.1)
            return up
        }
        NFKMLXLTX2LoRA.setStrength(detailing ? paths.detailingStrength : 0, in: pipeline.transformer)
        var result = try pipeline.dfrPass(
            text: text, audioText: audioText, frames: frames, latentHeight: height / 32, latentWidth: width / 32,
            frameRate: frameRate, sigmas: NFKMLXLTX2Pipeline.refinementSigmas, latents: video, audio: first.audio,
            slotInitials: slots, slotPositions: first.positions, reference: detailing ? first.video : nil,
            conditions: try conditions(width: width, height: height, frames: canvasFrames), schedule: schedule,
            sequenceNoise: draw, audioNoise: draw, onStep: reporter(reported))
        reported += NFKMLXLTX2Pipeline.refinementSigmas.count
        NFKMLXLTX2LoRA.setStrength(0, in: pipeline.transformer)

        var fps = frameRate
        let sourceSeconds = Double(result.canvasFrames) / Double(frameRate)
        var lastPositions = result.positions
        if paths.dfrTemporalRounds > 0, let stage = paths.temporalUpsampler {
            let roundConditions = try conditions(width: width, height: height, frames: frames)
            for round in 1 ... paths.dfrTemporalRounds {
                lastPositions = result.positions
                let roundIndex = UInt64(round)
                result = try staging.use(stage) { upsampler in
                    try pipeline.temporalRound(
                        video: result.video, keyframes: result.keyframes!, positions: result.positions, audio: first.audio,
                        frames: result.canvasFrames, frameRate: fps, sourceSeconds: sourceSeconds, round: round,
                        upsampler: upsampler, text: text, audioText: audioText, conditions: roundConditions,
                        tileNoise: { _, shape in draw(shape) },
                        ancestralNoise: { tile in
                            var step: UInt64 = 0
                            return { shape in
                                step += 1
                                let key = MLXRandom.key(seed &+ 1000 &* roundIndex &+ UInt64(tile) &+ step &* 0x1_0000)
                                return MLXRandom.normal(shape, key: key)
                            }
                        }, onStep: reporter(reported))
                }
                reported += (1 << round) * NFKMLXLTX2DFR.temporalSigmas.count
                fps *= 2
            }
        }

        if paths.dfrEpilogue, let encoderStage = paths.videoEncoder {
            let upsampled = try staging.use(paths.spatialUpsampler!) { upsampler -> MLXArray in
                let up = NFKMLXLTX2Pipeline.upsampled(result.video, by: upsampler)
                eval(up)
                return up
            }
            let rebuilt = try staging.use(encoderStage) { encoder -> MLXArray in
                let keyframes = pipeline.rebuiltKeyframes(result.keyframes!, encoder: encoder)
                eval(keyframes)
                return keyframes
            }
            let latentFrames = (result.canvasFrames - 1) / r + 1
            let seams = paths.dfrTemporalRounds > 0 ? lastPositions.map { 2 * $0 / r } : []
            let tiles = try NFKMLXLTX2DFRLayout.epilogueTiles(
                latentShape: (latentFrames, upsampled.dim(2), upsampled.dim(3)), frameTiles: 1 << paths.dfrTemporalRounds,
                frameSeams: seams)
            NFKMLXLTX2LoRA.setStrength(detailing ? paths.detailingStrength : 0, in: pipeline.transformer)
            result = try pipeline.dfrPass(
                text: text, audioText: audioText, frames: result.canvasFrames, latentHeight: upsampled.dim(2),
                latentWidth: upsampled.dim(3), frameRate: fps, sigmas: NFKMLXLTX2Pipeline.refinementSigmas,
                latents: upsampled, audio: first.audio, freezesAudio: true, generatesSlots: false,
                reference: detailing ? result.video : nil,
                guidanceKeyframes: (rebuilt, result.positions, NFKMLXLTX2DFR.epilogueKeyframeStrength),
                conditions: try conditions(width: upsampled.dim(3) * 32, height: upsampled.dim(2) * 32,
                                           frames: result.canvasFrames), tiles: tiles,
                schedule: schedule, sequenceNoise: draw, audioNoise: draw, onStep: reporter(reported))
            NFKMLXLTX2LoRA.setStrength(0, in: pipeline.transformer)
        }

        let outputFrames = (frames - 1) * (1 << paths.dfrTemporalRounds) + 1
        let kept = NFKMLXLTX2DFRLayout.trimmedLatentFrames(frames: outputFrames, temporalCompression: r)
        let latent = result.video[0..., 0 ..< kept]
        let audio = pipeline.normalizedAudio(first.audio)
        eval(latent, audio)
        return (latent, audio)
    }
}
