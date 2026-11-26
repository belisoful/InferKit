//
//  NFKMLXWanS2VPipeline.swift
//  InferKitMLX
//
//  The Wan 2.2 S2V-14B sampler (Wan-Video/Wan2.2 `WanS2V.generate`): a speech track read by wav2vec2 into
//  per-frame audio buckets, and a clip generated segment by segment, each segment conditioned on a reference
//  image, the previous segment's last frames, and its share of the audio.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// A speech track as the S2V transformer reads it: wav2vec2's hidden states sampled once per video frame.
struct NFKMLXWanS2VAudio {
    /// The buckets `[layers, audioDim, segments · framesPerSegment]`, zeros past the track's end.
    var buckets: MLXArray
    /// The segments the track covers.
    var segments: Int
    /// The frames one segment generates.
    var framesPerSegment: Int

    /// Segment `index`'s buckets `[layers, audioDim, framesPerSegment]`.
    func segment(_ index: Int) -> MLXArray {
        buckets[0..., 0..., (index * framesPerSegment) ..< ((index + 1) * framesPerSegment)]
    }
}

/// The S2V sampler over a transformer and the Wan 2.1 autoencoder.
final class NFKMLXWanS2VPipeline {
    let net: NFKMLXWanS2VNet
    let vae: NFKMLXWanVideoVAENet
    let latentsMean: MLXArray
    let latentsStd: MLXArray

    /// The frames of the previous segment the next one reads (`motion_frames`).
    var motionFrames = 73
    /// The video frame rate the audio is sampled at (`sample_fps`).
    var fps = 16
    /// Whether the first segment drops its (empty) motion frames (`drop_first_motion`).
    var dropsFirstMotion = true

    /// wav2vec2's frame rate, and the rate its hidden states are interpolated to (`video_rate`).
    static let wav2vec2Rate = 50
    static let audioRate = 30

    var motionLatentFrames: Int { (motionFrames + 3) / 4 }

    init(net: NFKMLXWanS2VNet, vae: NFKMLXWanVideoVAENet, latentsMean: MLXArray, latentsStd: MLXArray) {
        self.net = net
        self.vae = vae
        self.latentsMean = latentsMean
        self.latentsStd = latentsStd
    }

    // MARK: Audio

    /// wav2vec2's hidden states `[layers + 1, frames, hidden]` of a normalized waveform `[1, samples]`, as
    /// transformers' `output_hidden_states` lists them for a stable-layer-norm encoder: each layer's input, then
    /// the final layer norm's output.
    static func hiddenStates(_ waveform: MLXArray, wav2vec2: NFKMLXWav2Vec2Net) -> MLXArray {
        let encoder = wav2vec2.encoder
        var hidden = wav2vec2.featureProjection(wav2vec2.extractFeatures(waveform))
        hidden = hidden + encoder.positionalConv(hidden)
        if !encoder.stable {
            hidden = encoder.layerNorm(hidden)
        }
        var states = [hidden]
        for (index, layer) in encoder.layers.enumerated() {
            hidden = layer(hidden)
            if index < encoder.layers.count - 1 || !encoder.stable {
                states.append(hidden)
            }
        }
        if encoder.stable {
            states.append(encoder.layerNorm(hidden))
        }
        return concatenated(states, axis: 0)
    }

    /// `features` `[L, T, C]` resampled along `T` from `inputRate` to `outputRate` (`linear_interpolation`:
    /// `F.interpolate(mode="linear", align_corners=True)` to `int(T / inputRate · outputRate)` frames, the source
    /// positions and weights in float32 as torch's CPU kernel forms them).
    static func interpolated(_ features: MLXArray, inputRate: Int, outputRate: Int) -> MLXArray {
        let input = features.dim(1)
        let output = Int(Double(input) / Double(inputRate) * Double(outputRate))
        let scale: Float = output > 1 ? Float(input - 1) / Float(output - 1) : 0
        var lower = [Int32](), upper = [Int32](), weights = [Float]()
        for index in 0 ..< output {
            let source = scale * Float(index)
            let floor = min(Int(source), input - 1)
            lower.append(Int32(floor))
            upper.append(Int32(floor + (floor < input - 1 ? 1 : 0)))
            weights.append(source - Float(floor))
        }
        let w1 = MLXArray(weights).reshaped([1, output, 1]), w0 = 1 - w1
        return features.take(MLXArray(lower), axis: 1) * w0 + features.take(MLXArray(upper), axis: 1) * w1
    }

    /// The per-frame buckets of the 30 fps features `[L, T, C]` for segments of `framesPerSegment` frames at `fps`
    /// (`get_audio_embed_bucket_fps` with `m` 0): enough segments to cover the track, each frame reading the
    /// feature frame nearest its time (NumPy's round half to even), zeros past the track's end.
    static func buckets(_ features: MLXArray, fps: Int, framesPerSegment: Int) -> NFKMLXWanS2VAudio {
        let frames = features.dim(1)
        let scale = Double(audioRate) / Double(fps)
        let segments = Int(Double(frames) / (Double(framesPerSegment) * scale)) + 1
        let count = segments * framesPerSegment
        let padded = Int((Double(count) / Double(fps) * Double(audioRate)).rounded(.up)) - frames
        let total = frames + padded
        let duration = Double(count) / Double(fps)
        var indices = [Int32](), valid = [Float]()
        for i in 0 ..< count {
            let time = Double(i) * (duration / Double(count))
            let index = min(max(Int((time * Double(audioRate)).rounded(.toNearestOrEven)), 0), total - 1)
            indices.append(Int32(min(index, frames - 1)))
            valid.append(index < frames ? 1 : 0)
        }
        let gathered = features.take(MLXArray(indices), axis: 1) * MLXArray(valid).reshaped([1, count, 1])
        return NFKMLXWanS2VAudio(buckets: gathered.transposed(0, 2, 1), segments: segments, framesPerSegment: framesPerSegment)
    }

    /// The buckets of a 16 kHz waveform through wav2vec2.
    static func audio(_ samples: [Float], wav2vec2: NFKMLXWav2Vec2Net, fps: Int, framesPerSegment: Int) -> NFKMLXWanS2VAudio {
        let input = NFKMLXWav2Vec2Processor.inputValues(samples, normalize: wav2vec2.configuration.normalizesInput)
        let states = hiddenStates(input.asType(NFKReferenceRounding.parameterType(of: wav2vec2)), wav2vec2: wav2vec2)
        let features = interpolated(states.asType(.float32), inputRate: wav2vec2Rate, outputRate: audioRate)
        let audio = buckets(features, fps: fps, framesPerSegment: framesPerSegment)
        eval(audio.buckets)
        return audio
    }

    // MARK: Latents

    /// The autoencoder's standardized mean of a clip `[F, H, W, 3]` in `-1...1`, `[16, F', h, w]`.
    func encoded(_ clip: MLXArray) -> MLXArray {
        let dtype = NFKReferenceRounding.parameterType(of: vae)
        let latent = vae.encode(clip.expandedDimensions(axis: 0).asType(dtype))[0].asType(.float32)
        let standardized = ((latent - latentsMean) * (1 / latentsStd)).transposed(3, 0, 1, 2)
        eval(standardized)
        return standardized
    }

    /// The clip `[F, H, W, 3]` in `-1...1` the latents `[16, F', h, w]` decode to.
    func decoded(_ latents: MLXArray) -> MLXArray {
        let channelsLast = latents.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
        let unstandardized = channelsLast * latentsStd.reshaped([1, 1, 1, 1, -1]) + latentsMean.reshaped([1, 1, 1, 1, -1])
        let dtype = NFKReferenceRounding.parameterType(of: vae)
        let clip = MLX.clip(vae.decode(unstandardized.asType(dtype))[0].asType(.float32), min: -1, max: 1)
        eval(clip)
        return clip
    }

    // MARK: Sampling

    /// What one segment's sampling records, for parity measurement: each guided step's latents and each
    /// transformer call's prediction (conditional, then unconditional).
    struct SegmentTrace {
        var steps: [MLXArray] = []
        var calls: [MLXArray] = []
    }

    /// The pose latents of each segment (`load_pose_cond`): the pose frames `[N, H, W, 3]` in `0...1`, the first
    /// `segments · frames` of them padded with frames of −1 to that count, split per segment, each
    /// encoded with its first frame ahead of it and that frame's latent dropped. Without pose frames, zeros.
    func poseLatents(_ pose: MLXArray?, segments: Int, frames: Int, latentShape: [Int]) -> [MLXArray] {
        guard let pose else {
            return Array(repeating: MLXArray.zeros(latentShape), count: segments)
        }
        var clip = pose[0 ..< min(pose.dim(0), segments * frames)] * 2 - 1
        if clip.dim(0) < segments * frames {
            clip = concatenated([clip, MLXArray.full([segments * frames - clip.dim(0), pose.dim(1), pose.dim(2), 3],
                                                     values: MLXArray(Float(-1)))], axis: 0)
        }
        return (0 ..< segments).map { segment in
            let chunk = clip[(segment * frames) ..< ((segment + 1) * frames)]
            return encoded(concatenated([chunk[0 ..< 1], chunk], axis: 0))[0..., 1...]
        }
    }

    /// Generates the clip `[N, H, W, 3]` in `0...1`, segment after segment (`WanS2V.generate`).
    ///
    /// @discussion Each segment generates `audio.framesPerSegment` frames from `noise(segment)` `[16, F', h, w]`
    /// over the original flow UniPC schedule at `shift`, guided where `negative` is given: the unconditional
    /// pass reads the negative prompt and silent audio. The transformer reads the reference image's latent, the
    /// latents of the last `motionFrames` frames so far, the segment's audio buckets, and the segment's pose
    /// latents (``poseLatents(_:segments:frames:latentShape:)``). The first segment's motion frames are blank and
    /// dropped: it decodes from the reference latent and drops its first three frames. `startsFromReference`
    /// places the reference image in the last six motion frames instead (`init_first_frame`), and the first
    /// segment keeps them. `text` and `negative` are umT5 features padded to 512 rows with zeros.
    func generate(text: MLXArray, negative: MLXArray?, reference: MLXArray, audio: NFKMLXWanS2VAudio, segments: Int? = nil,
                  pose poseFrames: MLXArray? = nil, startsFromReference: Bool = false,
                  noise: (Int) -> MLXArray, steps: Int, guidance: Float, shift: Double,
                  trace: ((Int, SegmentTrace) -> Void)? = nil, onStep: ((Int, Int) -> Void)? = nil) throws -> MLXArray {
        let frames = audio.framesPerSegment, height = reference.dim(0), width = reference.dim(1)
        guard frames % 4 == 0 else {
            throw NFKMLXError.unsupportedConfiguration("an S2V segment generates a multiple of 4 frames, not \(frames)")
        }
        let count = min(segments ?? audio.segments, audio.segments)
        let referencePixels = (reference * 2 - 1).expandedDimensions(axis: 0)
        let referenceLatent = encoded(referencePixels)
        var lastFrames = MLXArray.zeros([motionFrames, height, width, 3])
        if startsFromReference {
            let kept = min(6, motionFrames)
            lastFrames = concatenated([lastFrames[0 ..< (motionFrames - kept)], repeated(referencePixels, count: kept, axis: 0)],
                                      axis: 0)
        }
        var motion = encoded(lastFrames)
        let latentFrames = (frames + 3 + motionFrames) / 4 - motionLatentFrames
        let poses = poseLatents(poseFrames, segments: count, frames: frames,
                                latentShape: [net.configuration.conditionChannels, latentFrames, motion.dim(2), motion.dim(3)])
        var clips = [MLXArray]()
        for segment in 0 ..< count {
            let drops = dropsFirstMotion && !startsFromReference && segment == 0
            let segmentAudio = audio.segment(segment)
            let conditioned = NFKWanS2VInputs(
                reference: referenceLatent, motion: drops ? nil : motion, pose: poses[segment],
                audio: net.audioTokens(segmentAudio, motionFrames: motionFrames, motionLatentFrames: motionLatentFrames))
            var silent = conditioned
            if negative != nil {
                silent.audio = net.audioTokens(MLXArray.zeros(like: segmentAudio), motionFrames: motionFrames,
                                               motionLatentFrames: motionLatentFrames)
            }
            var scheduler = NFKMLXUniPCScheduler(NFKMLXUniPCConfiguration(flowShift: 1))
            scheduler.setOriginalTimesteps(steps, shift: shift)
            var latents = noise(segment)
            var record = SegmentTrace()
            for (index, timestep) in scheduler.timesteps.enumerated() {
                let t = Double(timestep)
                let cond = net(latents, text: text, t: t, inputs: conditioned)
                var guided = cond
                if let negative {
                    let uncond = net(latents, text: negative, t: t, inputs: silent)
                    guided = uncond + guidance * (cond - uncond)
                    if trace != nil {
                        record.calls += [cond, uncond]
                    }
                }
                latents = scheduler.step(velocity: guided, sample: latents, index: index)
                eval(latents)
                if trace != nil {
                    record.steps.append(latents)
                }
                onStep?(segment, index)
            }
            trace?(segment, record)
            let history = drops ? referenceLatent : motion
            var image = decoded(concatenated([history, latents], axis: 1))
            image = image[(image.dim(0) - frames)...]
            if drops {
                image = image[3...]
            }
            let overlap = min(motionFrames, image.dim(0))
            lastFrames = concatenated([lastFrames[overlap...], image[(image.dim(0) - overlap)...]], axis: 0)
            motion = encoded(lastFrames)
            clips.append(image)
        }
        return MLX.clip(concatenated(clips, axis: 0) / 2 + 0.5, min: 0, max: 1)
    }
}
