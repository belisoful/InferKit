//
//  NFKMLXLTX2DFR.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

// LTX-2.5's Diffusion Fidelity Rendering (DFR) pass, the reference's `LTX2DFRPipeline` and the denoise loop
// its temporal refine round shares. One pass denoises a packed video sequence laid out
// `[grid | keyframes | slots | reference]`:
//
// - the grid is the latent video, seeded from a caller's latents where given;
// - a keyframe is an appended latent frame of given content (a carried seam keyframe, a condition placed
//   past frame 0), held near its clean content by its strength;
// - a slot is an appended latent frame the pass GENERATES, positioned at one pixel frame and marked with
//   the release's keyframe embedding: it buys one extra pixel frame of detail at that position;
// - the reference is the IC-LoRA's clean half-resolution latent, its spatial coordinates scaled up.
//
// A token held at strength `s` sees the noise level `(1 − s) · t`. Each step converts the velocity to x0
// with each token's own sigma, blends the held tokens back to their clean content, and steps the shared
// schedule. The temporal round steps an ancestral sampler that renoises every token, so the blend is
// applied again after the step. Every pass runs the audio stream too, for the cross-modal attention; a
// frozen audio stream stays at sigma 0 and takes no step.
//
// Latents here are VAE-NORMALIZED and channels-last: `[1, F, H, W, C]` unpacked, `[1, tokens, C]` packed.
// Every noise draw is the caller's, through a `([Int]) -> MLXArray` closure given the shape.

/// The Euler-ancestral rectified-flow sampler the DFR temporal round steps with, the reference's
/// `LTXEulerAncestralRFScheduler` (ComfyUI's `sample_euler_ancestral_RF`).
public struct NFKMLXLTX2AncestralScheduler: Sendable {
    /// The schedule, ending at 0.
    public let sigmas: [Float]
    /// `1000 σ`, one per sigma.
    public let timesteps: [Float]
    public let eta: Float
    public let noiseScale: Float

    /// The schedule `sigmas`, given without its terminal 0 (appended here) or with it.
    public init(sigmas given: [Float], eta: Float = 0.5, noiseScale: Float = 1) {
        let sigmas = given.last == 0 ? given : given + [0]
        self.sigmas = sigmas
        timesteps = sigmas.map { $0 * 1000 }
        self.eta = eta
        self.noiseScale = noiseScale
    }

    /// The steps a sampler takes: every sigma but the terminal one.
    public var steps: Int { sigmas.count - 1 }

    /// One step from `sample` along `velocity` at step `index`, in float32: the deterministic step to the
    /// lowered sigma, then fresh noise from `noise` brought back up to the next sigma.
    public func step(velocity: MLXArray, sample: MLXArray, index: Int, noise: ([Int]) -> MLXArray) -> MLXArray {
        let x = sample.asType(.float32)
        let sigma = sigmas[index], next = sigmas[index + 1]
        let denoised = x - sigma * velocity.asType(.float32)
        guard abs(next) >= 1e-8 else { return denoised }
        let downRatio = 1 + (next / sigma - 1) * eta
        let down = next * downRatio
        let alphaNext = 1 - next, alphaDown = 1 - down
        let ratio = down / sigma
        var result = ratio * x + (1 - ratio) * denoised
        if eta > 0, noiseScale > 0 {
            let renoise = max(next * next - down * down * alphaNext * alphaNext / (alphaDown * alphaDown + 1e-12), 0)
                .squareRoot()
            result = (alphaNext / (alphaDown + 1e-12)) * result + noise(x.shape) * renoise * noiseScale
        }
        return result
    }
}

/// A frame condition placed in a DFR pass: an encoded clip or still at a pixel frame, held at `strength`.
public struct NFKMLXLTX2DFRCondition {
    /// The pixel frame it sits at; 0 replaces the clip's first frames in place.
    public var pixelFrame: Int
    /// The normalized latent `[1, F, H, W, C]`.
    public var latent: MLXArray
    public var strength: Float
    /// The pixel frames the latent encodes; a single frame occupies one pixel frame of time.
    public var pixelFrames: Int

    public init(pixelFrame: Int, latent: MLXArray, strength: Float, pixelFrames: Int) {
        self.pixelFrame = pixelFrame
        self.latent = latent
        self.strength = strength
        self.pixelFrames = pixelFrames
    }
}

/// A packed DFR sequence and what each token is.
public struct NFKMLXLTX2DFRSequence {
    /// The noised tokens `[1, tokens, C]`, float32.
    public var latents: MLXArray
    /// Each token's conditioning strength `[1, tokens, 1]`: 0 denoises freely, 1 holds the clean content.
    public var conditioning: MLXArray
    /// The clean content held tokens are blended toward `[1, tokens, C]`.
    public var clean: MLXArray
    /// 1 where a token's latent frame is a single pixel frame (the grid's first frame and the slots) `[1, tokens]`.
    public var keyframes: MLXArray
    public var coordinates: NFKMLXLTX2VideoCoordinates
    /// The generated slot tokens, where the pass makes slots.
    public var slotTokens: Range<Int>?
    public var latentFrames: Int
    public var latentHeight: Int
    public var latentWidth: Int

    public var gridTokens: Int { latentFrames * latentHeight * latentWidth }
}

/// The DFR pass: laying out its sequence and denoising it.
public enum NFKMLXLTX2DFR {

    /// The distilled schedule's tail the temporal rounds sample, the reference's
    /// `TEMPORAL_ROUND_DISTILLED_SIGMA_VALUES`.
    public static let temporalSigmas: [Float] = Array(NFKMLXLTX2Generator.distilledSigmas[4...])
    /// The strength a carried seam keyframe is held at in a temporal round.
    public static let anchorStrength: Float = 0.95
    /// The strength the epilogue holds its rebuilt keyframes at.
    public static let epilogueKeyframeStrength: Float = 1
    /// The ancestral noise fraction of a temporal round.
    public static let temporalEta: Float = 0.5
    /// Above this frame rate the transformer is conditioned at ``maximumConditioningFPS``; the playback rate
    /// only sets the frame count.
    public static let snapConditioningFPSAbove: Float = 30
    public static let maximumConditioningFPS: Float = 60

    /// The frame rate a pass's rotary reads for a clip that plays at `fps`.
    public static func conditioningFPS(_ fps: Float) -> Float { fps > snapConditioningFPSAbove ? maximumConditioningFPS : fps }

    /// Packs `[1, F, H, W, C]` to `[1, F·H·W, C]`.
    static func packed(_ x: MLXArray) -> MLXArray { x.reshaped([x.dim(0), -1, x.dim(-1)]) }

    /// Lays out one pass's sequence on a `latentFrames × latentHeight × latentWidth` grid of `channels`, the
    /// reference's `prepare_latents`.
    ///
    /// @discussion The grid is seeded from `seed` `[1, F, H, W, C]`, or zeros. A condition at pixel frame 0
    /// replaces the first frames in place; the other conditions, then the `given` keyframes (pixel frame,
    /// `[1, 1, H, W, C]`, strength), append. `slots` are the pixel frames of generated slots, seeded from
    /// `slotInitials` `[1, K, H, W, C]` where given. `reference`, the IC-LoRA's latent, appends held clean, its
    /// spatial coordinates scaled by `referenceScale`. Frame coordinates are in seconds at `fps`. `noise` draws
    /// over the whole sequence, which starts at `noiseScale` with the held tokens blended back.
    public static func sequence(configuration c: NFKMLXLTX2Configuration, latentFrames: Int, latentHeight: Int,
                                latentWidth: Int, channels: Int, latents seed: MLXArray? = nil,
                                conditions: [NFKMLXLTX2DFRCondition] = [], keyframes given: [(Int, MLXArray, Float)] = [],
                                slots: [Int] = [], slotInitials: MLXArray? = nil, reference: MLXArray? = nil,
                                referenceScale: Float = 2, fps: Float, noiseScale: Float,
                                noise: ([Int]) -> MLXArray) -> NFKMLXLTX2DFRSequence {
        let perFrame = latentHeight * latentWidth
        let grid = latentFrames * perFrame
        var latents = packed(seed?.asType(.float32) ?? MLXArray.zeros([1, latentFrames, latentHeight, latentWidth, channels]))
        var conditioning = MLXArray.zeros([1, grid, 1])
        var clean = MLXArray.zeros(like: latents)

        var appended = [(tokens: MLXArray, coordinates: NFKMLXLTX2VideoCoordinates, strength: Float)]()
        for condition in conditions {
            let tokens = packed(condition.latent.asType(.float32))
            if condition.pixelFrame == 0 {
                let count = tokens.dim(1)
                conditioning = concatenated([MLXArray.ones([1, count, 1]) * condition.strength,
                                             conditioning[0..., count...]], axis: 1)
                clean = concatenated([tokens, clean[0..., count...]], axis: 1)
                continue
            }
            appended.append((tokens, .keyframe(frames: condition.latent.dim(1), height: condition.latent.dim(2),
                                               width: condition.latent.dim(3), pixelFrame: condition.pixelFrame,
                                               pixelFrames: condition.pixelFrames, configuration: c, fps: fps),
                             condition.strength))
        }
        for (pixelFrame, latent, strength) in given {
            appended.append((packed(latent.asType(.float32)),
                             .keyframe(frames: latent.dim(1), height: latent.dim(2), width: latent.dim(3),
                                       pixelFrame: pixelFrame, pixelFrames: 1, configuration: c, fps: fps), strength))
        }

        var marker = concatenated([MLXArray.ones([1, perFrame]), MLXArray.zeros([1, grid - perFrame])], axis: 1)
        var coordinates = [NFKMLXLTX2VideoCoordinates]()
        for (tokens, place, strength) in appended {
            latents = concatenated([latents, MLXArray.zeros(like: tokens)], axis: 1)
            clean = concatenated([clean, tokens], axis: 1)
            conditioning = concatenated([conditioning, MLXArray.ones([1, tokens.dim(1), 1]) * strength], axis: 1)
            marker = concatenated([marker, MLXArray.zeros([1, tokens.dim(1)])], axis: 1)
            coordinates.append(place)
        }

        var slotTokens: Range<Int>?
        if !slots.isEmpty {
            let tokens = slotInitials.map { packed($0.asType(.float32)) }
                ?? MLXArray.zeros([1, perFrame * slots.count, channels])
            slotTokens = latents.dim(1) ..< latents.dim(1) + tokens.dim(1)
            latents = concatenated([latents, tokens], axis: 1)
            clean = concatenated([clean, MLXArray.zeros(like: tokens)], axis: 1)
            conditioning = concatenated([conditioning, MLXArray.zeros([1, tokens.dim(1), 1])], axis: 1)
            marker = concatenated([marker, MLXArray.ones([1, tokens.dim(1)])], axis: 1)
            coordinates += slots.map {
                .keyframe(frames: 1, height: latentHeight, width: latentWidth, pixelFrame: $0, pixelFrames: 1,
                          configuration: c, fps: fps)
            }
        }

        if let reference {
            let tokens = packed(reference.asType(.float32))
            latents = concatenated([latents, MLXArray.zeros(like: tokens)], axis: 1)
            clean = concatenated([clean, tokens], axis: 1)
            conditioning = concatenated([conditioning, MLXArray.ones([1, tokens.dim(1), 1])], axis: 1)
            marker = concatenated([marker, MLXArray.zeros([1, tokens.dim(1)])], axis: 1)
            coordinates.append(NFKMLXLTX2VideoCoordinates.grid(
                frames: reference.dim(1), height: reference.dim(2), width: reference.dim(3), configuration: c, fps: fps)
                .spatiallyScaled(by: referenceScale))
        }

        let drawn = noise(latents.shape).asType(.float32)
        latents = drawn * noiseScale + latents * (1 - noiseScale)
        latents = clean * conditioning + latents * (1 - conditioning)
        return NFKMLXLTX2DFRSequence(
            latents: latents, conditioning: conditioning, clean: clean, keyframes: marker,
            coordinates: NFKMLXLTX2VideoCoordinates.grid(frames: latentFrames, height: latentHeight, width: latentWidth,
                                                         configuration: c, fps: fps).appending(coordinates),
            slotTokens: slotTokens, latentFrames: latentFrames, latentHeight: latentHeight, latentWidth: latentWidth)
    }

    /// The sampler a pass steps: the release's flow-match Euler (the DFR passes and the epilogue), or the
    /// ancestral sampler (the temporal rounds).
    public enum Sampler {
        case euler(NFKMLXFlowMatchConfiguration)
        case ancestral(eta: Float)
    }

    /// Denoises `sequence` over `sigmas` (without the terminal 0) and returns the packed video tokens and the
    /// packed audio latents, the reference's `denoise`.
    ///
    /// @discussion `audio` `[1, L, C·M]` is normalized; `freezesAudio` holds it at sigma 0 with no step.
    /// `text` and `audioText` are the positive prompt's streams: the distilled schedules sample unguided.
    /// `tilePlan`, where given, runs the transformer once a step for each tile of the grid and blends the
    /// predictions by the plan's weights. `noise` draws the ancestral sampler's renoise, given its shape.
    public static func denoise(_ sequence: NFKMLXLTX2DFRSequence, audio initialAudio: MLXArray,
                               transformer: NFKMLXLTX2TransformerNet, text: MLXArray, audioText: MLXArray,
                               sigmas: [Float], sampler: Sampler, freezesAudio: Bool, fps: Float,
                               tilePlan: [NFKMLXLTX2DFRLayout.TokenPlan]? = nil,
                               noise: ([Int]) -> MLXArray = { MLXRandom.normal($0) },
                               onStep: ((Int) -> Void)? = nil) throws -> (video: MLXArray, audio: MLXArray) {
        let scheduleSigmas: [Float], timesteps: [Float]
        var euler: NFKMLXFlowMatchScheduler?
        var ancestral: NFKMLXLTX2AncestralScheduler?
        switch sampler {
        case .euler(let configuration):
            var scheduler = NFKMLXFlowMatchScheduler(configuration)
            scheduler.setTimesteps(sigmas: sigmas, sequenceLength: sequence.latents.dim(1))
            euler = scheduler
            scheduleSigmas = scheduler.sigmas
            timesteps = scheduler.timesteps
        case .ancestral(let eta):
            guard freezesAudio else {
                throw NFKMLXError.unsupportedConfiguration("the ancestral DFR sampler steps a frozen audio stream only")
            }
            let scheduler = NFKMLXLTX2AncestralScheduler(sigmas: sigmas, eta: eta)
            ancestral = scheduler
            scheduleSigmas = scheduler.sigmas
            timesteps = Array(scheduler.timesteps.dropLast())
        }
        let dtype = transformer.projIn.weight.dtype
        var latents = sequence.latents, audio = initialAudio.asType(.float32)
        let held = sequence.conditioning, free = 1 - sequence.conditioning

        for index in timesteps.indices {
            let t = timesteps[index]
            let videoTimestep = t * free.squeezed(axis: -1)
            let audioTimestep = MLXArray([freezesAudio ? Float(0) : t])
            func pass(_ tokens: [Int]?) -> (video: MLXArray, audio: MLXArray) {
                let select: (MLXArray) -> MLXArray = { x in
                    tokens.map { x.take(MLXArray($0.map(Int32.init)), axis: 1) } ?? x
                }
                let one = NFKMLXLTX2Pass(
                    video: select(latents).asType(dtype), audio: audio.asType(dtype), text: text, audioText: audioText,
                    timestep: select(videoTimestep), audioTimestep: audioTimestep, sigma: MLXArray([t]),
                    audioSigma: audioTimestep, keyframes: select(sequence.keyframes))
                return transformer.predictions(
                    [one], frames: sequence.latentFrames, height: sequence.latentHeight, width: sequence.latentWidth,
                    audioFrames: audio.dim(1), fps: fps, crossTimestep: true,
                    videoCoordinates: tokens == nil ? sequence.coordinates : nil)[0]
            }
            var videoPrediction: MLXArray, audioPrediction: MLXArray
            if let tilePlan {
                videoPrediction = MLXArray.zeros(latents.shape, dtype: .float32)
                audioPrediction = MLXArray.zeros(audio.shape, dtype: .float32)
                for tile in tilePlan {
                    let tokens = MLXArray(tile.keep.map(Int32.init))
                    let one = NFKMLXLTX2Pass(
                        video: latents.take(tokens, axis: 1).asType(dtype), audio: audio.asType(dtype), text: text,
                        audioText: audioText, timestep: videoTimestep.take(tokens, axis: 1), audioTimestep: audioTimestep,
                        sigma: MLXArray([t]), audioSigma: audioTimestep, keyframes: sequence.keyframes.take(tokens, axis: 1))
                    let output = transformer.predictions(
                        [one], frames: sequence.latentFrames, height: sequence.latentHeight, width: sequence.latentWidth,
                        audioFrames: audio.dim(1), fps: fps, crossTimestep: true, videoCoordinates: tile.coordinates)[0]
                    let weighted = output.video.asType(.float32) * tile.weights.reshaped([1, -1, 1])
                    videoPrediction = videoPrediction.at[0..., tokens].add(weighted)
                    audioPrediction = audioPrediction + output.audio.asType(.float32) / Float(tilePlan.count)
                    eval(videoPrediction, audioPrediction)
                }
            } else {
                let output = pass(nil)
                videoPrediction = output.video.asType(.float32)
                audioPrediction = output.audio.asType(.float32)
            }
            try transformer.verifyStream()

            // x0 with each token's own sigma, the held tokens blended back to their clean content.
            let sigma = scheduleSigmas[index]
            let perToken = (videoTimestep / 1000).expandedDimensions(axis: -1)
            var denoised = latents - videoPrediction * perToken
            denoised = denoised * free + sequence.clean * held
            let velocity = (latents - denoised) / sigma
            if let euler {
                latents = euler.step(velocity: velocity, sample: latents, index: index)
            } else if let ancestral {
                latents = ancestral.step(velocity: velocity, sample: latents, index: index, noise: noise)
                latents = latents * free + sequence.clean * held
            }
            if !freezesAudio, let euler {
                let audioDenoised = audio - audioPrediction * sigma
                audio = euler.step(velocity: (audio - audioDenoised) / sigma, sample: audio, index: index)
            }
            eval(latents, audio)
            onStep?(index)
        }
        return (latents, audio)
    }
}
