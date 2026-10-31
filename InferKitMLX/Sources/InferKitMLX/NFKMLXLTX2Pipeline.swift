//
//  NFKMLXLTX2Pipeline.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The LTX-2 text-to-video-and-audio glue (`LTX2Pipeline`, Lightricks): the connectors' two text streams,
// the joint denoising loop, and the two decodes. The parts it chains are measured on their own; this
// file holds what lies between them.
//
// A step runs the transformer up to three times. The guided pair is the negative and the positive
// prompt batched together; a spatio-temporal guidance pass perturbs the listed blocks' self-attentions;
// a modality-isolation pass skips the cross-modal attentions. The predictions combine in x0 space
// (`x − σ v`): classifier-free guidance `(g − 1)(x0₊ − x0₋)`, spatio-temporal guidance `s (x0₊ − x0ₛ)`,
// modality guidance `(m − 1)(x0₊ − x0ₘ)`, then the guidance rescale against the conditional prediction's
// unbiased standard deviation, and back to a velocity for the Euler step. Video and audio each carry
// their own scales, and two identical flow schedules advance them. Every pass's timestep is the step's
// `1000 σ`, shared by both modalities, and the cross-modal modulation reads the other modality's.
//
// Latents are packed as the transformer reads them. The video's `[B, F·H·W, C]` unpacks to the decoder's
// NDHWC and is denormalized by the autoencoder's per-channel statistics; the audio's `[B, L, C·M]` is
// denormalized PACKED (its statistics have that width), then unpacks channel-major to the audio
// decoder's `[B, L, M, C]`.

/// The guidance scales one modality samples with.
public struct NFKMLXLTX2Guidance: Sendable, Equatable {
    /// Classifier-free guidance; above 1 the step runs the negative prompt too.
    public var textScale: Float
    /// Spatio-temporal guidance; above 0 the step runs the perturbed pass.
    public var spatioTemporalScale: Float
    /// Modality guidance; above 1 the step runs the isolated pass.
    public var modalityScale: Float
    /// The rescale's mix toward the conditional prediction's spread; 0 turns it off.
    public var rescale: Float

    public init(textScale: Float, spatioTemporalScale: Float, modalityScale: Float, rescale: Float) {
        self.textScale = textScale
        self.spatioTemporalScale = spatioTemporalScale
        self.modalityScale = modalityScale
        self.rescale = rescale
    }

    /// The reference pipeline's video defaults: 3, 1, 3, 0.7.
    public static let video = NFKMLXLTX2Guidance(textScale: 3, spatioTemporalScale: 1, modalityScale: 3, rescale: 0.7)
    /// The reference pipeline's audio defaults: 7, 1, 3, 0.7.
    public static let audio = NFKMLXLTX2Guidance(textScale: 7, spatioTemporalScale: 1, modalityScale: 3, rescale: 0.7)
}

/// The LTX-2 transformer, connectors, and decoders held together for a generation. A caller that cannot
/// hold every stage at once (``NFKMLXLTX2Generator`` stages them) uses the pieces in turn.
public final class NFKMLXLTX2Pipeline {
    public let transformer: NFKMLXLTX2TransformerNet
    public let videoDecoder: NFKMLXLTX2VideoDecoderNet
    public let audioDecoder: NFKMLXLTX2AudioDecoderNet
    public let vocoder: NFKMLXLTX2VocoderNet
    let videoStatistics: (mean: MLXArray, std: MLXArray)
    let audioStatistics: (mean: MLXArray, std: MLXArray)

    public init(transformer: NFKMLXLTX2TransformerNet, videoDecoder: NFKMLXLTX2VideoDecoderNet,
                videoStatistics: (mean: MLXArray, std: MLXArray), audioDecoder: NFKMLXLTX2AudioDecoderNet,
                audioStatistics: (mean: MLXArray, std: MLXArray), vocoder: NFKMLXLTX2VocoderNet) {
        self.transformer = transformer
        self.videoDecoder = videoDecoder
        self.audioDecoder = audioDecoder
        self.vocoder = vocoder
        self.videoStatistics = videoStatistics
        self.audioStatistics = audioStatistics
    }

    /// The latent grid a clip of `frames` frames at `width × height` and `frameRate` occupies: video
    /// latent frames, rows, columns, and audio latent frames. The reference rounds the frame count down
    /// to one more than a multiple of the temporal compression and the size down to the spatial one; the
    /// audio covers the clip's duration at `sampleRate / hop / 4` latent frames a second.
    public static func grid(frames: Int, width: Int, height: Int, frameRate: Float,
                            video: NFKMLXLTX2VideoVAEConfiguration, audio: NFKMLXLTX2AudioVAEConfiguration)
        -> (frames: Int, height: Int, width: Int, audioFrames: Int) {
        let latentFrames = (frames - 1) / video.temporalCompression + 1
        let perSecond = Double(audio.sampleRate) / Double(audio.melHopLength)
            / Double(NFKMLXLTX2AudioVAEConfiguration.compression)
        let audioFrames = Int((Double(frames) / Double(frameRate) * perSecond).rounded(.toNearestOrEven))
        return (latentFrames, height / video.spatialCompression, width / video.spatialCompression, audioFrames)
    }

    /// One modality's guided x0 at a step: `conditional` and the optional unconditional, perturbed, and
    /// isolated predictions, all in x0 space.
    static func guided(conditional: MLXArray, unconditional: MLXArray?, perturbed: MLXArray?, isolated: MLXArray?,
                       guidance: NFKMLXLTX2Guidance) -> MLXArray {
        var combined = conditional
        if let unconditional { combined = combined + (guidance.textScale - 1) * (conditional - unconditional) }
        if let perturbed { combined = combined + guidance.spatioTemporalScale * (conditional - perturbed) }
        if let isolated { combined = combined + (guidance.modalityScale - 1) * (conditional - isolated) }
        guard guidance.rescale > 0 else { return combined }
        let rescaled = combined * (unbiasedStd(conditional) / unbiasedStd(combined))
        return guidance.rescale * rescaled + (1 - guidance.rescale) * combined
    }

    /// The standard deviation over every axis but the batch, with Bessel's correction (torch's default).
    static func unbiasedStd(_ x: MLXArray) -> MLXArray {
        let flat = x.reshaped([x.dim(0), -1])
        let mean = flat.mean(axis: 1, keepDims: true)
        let count = Float(flat.dim(1))
        let variance = (flat - mean).square().sum(axis: 1, keepDims: true) / (count - 1)
        return sqrt(variance).reshaped([x.dim(0)] + [Int](repeating: 1, count: x.ndim - 1))
    }

    /// Denoises from `video` `[1, F·H·W, C]` and `audio` `[1, L, C·M]` (normalized latents) over `steps`
    /// flow steps. `text` and `audioText` hold the positive prompt's streams, or the negative's and then the
    /// positive's when classifier-free guidance runs; `perturbedBlocks` are the blocks spatio-temporal
    /// guidance perturbs. `schedule` is the release's flow schedule, and `sigmas`, where given, is the
    /// ramp it warps in place of the pipeline's `linspace(1, 1 / steps, steps)`; a distilled release
    /// samples from its own sigmas and `steps` is then their count. `onStep` receives each completed
    /// step's index.
    public func denoise(video initialVideo: MLXArray, audio initialAudio: MLXArray, text: MLXArray,
                        audioText: MLXArray, frames: Int, height: Int, width: Int, audioFrames: Int,
                        frameRate: Float, steps: Int, videoGuidance: NFKMLXLTX2Guidance = .video,
                        audioGuidance: NFKMLXLTX2Guidance = .audio, perturbedBlocks: Set<Int>,
                        schedule: NFKMLXFlowMatchConfiguration = .ltxVideoPipeline, sigmas: [Float]? = nil,
                        onStep: ((Int) -> Void)? = nil) throws -> (video: MLXArray, audio: MLXArray) {
        let usesText = videoGuidance.textScale > 1 || audioGuidance.textScale > 1
        let usesPerturbation = (videoGuidance.spatioTemporalScale > 0 || audioGuidance.spatioTemporalScale > 0)
            && !perturbedBlocks.isEmpty
        let usesIsolation = videoGuidance.modalityScale > 1 || audioGuidance.modalityScale > 1
        let positiveText = usesText ? text[1 ..< 2] : text
        let positiveAudioText = usesText ? audioText[1 ..< 2] : audioText

        var scheduler = NFKMLXFlowMatchScheduler(schedule)
        if let sigmas {
            scheduler.setTimesteps(sigmas: sigmas, sequenceLength: initialVideo.dim(1))
        } else {
            scheduler.setTimesteps(steps, sequenceLength: initialVideo.dim(1))
        }
        let steps = scheduler.timesteps.count
        var video = initialVideo.asType(.float32), audio = initialAudio.asType(.float32)
        let dtype = transformer.projIn.weight.dtype

        for index in 0 ..< steps {
            let sigma = scheduler.sigmas[index]
            func pass(_ batch: Int, text: MLXArray, audioText: MLXArray, perturbed: Set<Int> = [],
                      isolates: Bool = false) -> NFKMLXLTX2Pass {
                let timestep = MLXArray([Float](repeating: scheduler.timesteps[index], count: batch))
                return NFKMLXLTX2Pass(
                    video: (batch == 1 ? video : concatenated([video, video], axis: 0)).asType(dtype),
                    audio: (batch == 1 ? audio : concatenated([audio, audio], axis: 0)).asType(dtype),
                    text: text, audioText: audioText, timestep: timestep, audioTimestep: timestep, sigma: timestep,
                    perturbedBlocks: perturbed, isolatesModalities: isolates)
            }
            var passes = [pass(usesText ? 2 : 1, text: text, audioText: audioText)]
            if usesPerturbation {
                passes.append(pass(1, text: positiveText, audioText: positiveAudioText, perturbed: perturbedBlocks))
            }
            if usesIsolation {
                passes.append(pass(1, text: positiveText, audioText: positiveAudioText, isolates: true))
            }
            // Each prediction in x0 space, `x − σ v`, against the latents the pass read.
            let x0 = transformer.predictions(passes, frames: frames, height: height, width: width,
                                             audioFrames: audioFrames, fps: frameRate, crossTimestep: true)
                .enumerated().map { offset, prediction in
                    let rows = passes[offset].video.dim(0)
                    let videoInput = rows == 1 ? video : concatenated([video, video], axis: 0)
                    let audioInput = rows == 1 ? audio : concatenated([audio, audio], axis: 0)
                    return (videoInput - prediction.video.asType(.float32) * sigma,
                            audioInput - prediction.audio.asType(.float32) * sigma)
                }
            let (videoPair, audioPair) = x0[0]
            let videoConditional = usesText ? videoPair[1 ..< 2] : videoPair
            let audioConditional = usesText ? audioPair[1 ..< 2] : audioPair
            let perturbed = usesPerturbation ? x0[1] : nil
            let isolated = usesIsolation ? x0[usesPerturbation ? 2 : 1] : nil

            func guidedVelocity(_ sample: MLXArray, conditional: MLXArray, unconditional: MLXArray?,
                                perturbed: MLXArray?, isolated: MLXArray?, guidance: NFKMLXLTX2Guidance) -> MLXArray {
                let x0 = Self.guided(
                    conditional: conditional,
                    unconditional: guidance.textScale > 1 ? unconditional : nil,
                    perturbed: guidance.spatioTemporalScale > 0 ? perturbed : nil,
                    isolated: guidance.modalityScale > 1 ? isolated : nil, guidance: guidance)
                return (sample - x0) / sigma
            }
            let videoVelocity = guidedVelocity(
                video, conditional: videoConditional, unconditional: usesText ? videoPair[0 ..< 1] : nil,
                perturbed: perturbed?.0, isolated: isolated?.0, guidance: videoGuidance)
            let audioVelocity = guidedVelocity(
                audio, conditional: audioConditional, unconditional: usesText ? audioPair[0 ..< 1] : nil,
                perturbed: perturbed?.1, isolated: isolated?.1, guidance: audioGuidance)
            video = scheduler.step(velocity: videoVelocity, sample: video, index: index)
            audio = scheduler.step(velocity: audioVelocity, sample: audio, index: index)
            eval(video, audio)
            try transformer.verifyStream()
            onStep?(index)
        }
        return (video, audio)
    }

    /// Packed video latents `[B, F·H·W, C]` → the decoded clip `[B, frames, height, width, 3]` in about
    /// `−1…1`.
    public func decodeVideo(_ latents: MLXArray, frames: Int, height: Int, width: Int) -> MLXArray {
        let unpacked = latents.reshaped([latents.dim(0), frames, height, width, latents.dim(2)])
        let denormalized = unpacked * videoStatistics.std + videoStatistics.mean
        let dtype = videoDecoder.convIn.conv.weight.dtype
        return videoDecoder(denormalized.asType(dtype)).asType(.float32)
    }

    /// Packed audio latents `[B, L, C·M]` → the stereo mel spectrogram `[B, 4L − 3, melBins, 2]`.
    public func decodeMel(_ latents: MLXArray) -> MLXArray {
        let denormalized = latents * audioStatistics.std + audioStatistics.mean
        let c = audioDecoder.configuration
        let unpacked = denormalized.reshaped([latents.dim(0), latents.dim(1), c.latentChannels, c.latentMelBins])
            .transposed(0, 1, 3, 2)
        let dtype = audioDecoder.convIn.conv.weight.dtype
        return audioDecoder(unpacked.asType(dtype)).asType(.float32)
    }

    /// Packed audio latents → the waveform `[B, samples, 2]` at the vocoder's output rate.
    public func decodeAudio(_ latents: MLXArray) -> MLXArray {
        let mel = decodeMel(latents)
        let dtype = vocoder.vocoder.convIn.weight.dtype
        return vocoder(mel.asType(dtype)).asType(.float32)
    }

    /// A Gemma 3 decoder's hidden state at every layer for one prompt's tokens, the embedding first and
    /// the final norm applied to the last, stacked on a trailing axis and flattened hidden-major: the
    /// connectors' input, `[1, tokens, hidden · (layers + 1)]`.
    ///
    /// @discussion The reference left-pads the prompt to its maximum length and masks the padding. A real
    /// token never attends to a padding one, so running the real tokens alone gives the same states up to
    /// the rotary's absolute offset, which the attention scores do not depend on.
    public static func textStates(_ encoder: NFKMLXGemma3Net, tokens: [Int]) -> MLXArray {
        let ids = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
        let states = encoder.layerStates(ids)
        let stacked = stacked(states, axis: -1)
        return stacked.reshaped([1, tokens.count, -1])
    }

    /// Normalizes packed audio latents by the autoencoder's statistics, as the reference does to latents a
    /// caller supplies.
    public func normalizedAudio(_ latents: MLXArray) -> MLXArray {
        (latents - audioStatistics.mean) / audioStatistics.std
    }
}
