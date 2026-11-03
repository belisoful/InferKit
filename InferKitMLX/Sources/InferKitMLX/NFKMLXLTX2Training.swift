//
//  NFKMLXLTX2Training.swift
//  InferKitMLX
//
//  Fine-tuning the LTX-2 audio-video transformer the way Lightricks' trainer does (`ltx-trainer` at
//  Lightricks/LTX-2 e98f6d9): LoRA adapters on the query, key, value, and output projections of all six
//  attentions in every block, over a base that may be quantized; flow matching with sigmas from a
//  stretched shifted-logit-normal sampler; the velocity target `noise − clean`; first-frame conditioning
//  that leaves the first latent frame clean, at timestep zero, and out of the loss; and AdamW with a linear
//  decay to a tenth of its rate. The trainer's model is `ltx_core`'s, which modulates the cross-modal
//  attentions as `gatesReadCrossSigma` describes, so every prediction here runs that way.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// The trainer's stretched shifted-logit-normal sigma sampler (`ShiftedLogitNormalTimestepSampler`).
///
/// @discussion A draw is `sigmoid(n·std + μ)` for a standard normal `n`, where `μ` rises linearly from
/// 0.95 at 1024 tokens to 2.05 at 4096 and continues on that line past both ends. The draw is stretched to
/// `[0, 1]` between the distribution's 0.5th and 99.9th percentiles, reflected about `eps` below it, and
/// clamped. With probability `uniformProbability` the sampler returns a uniform draw on `[eps, 1)` instead,
/// so long sequences keep low sigmas in the mix.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXLTX2SigmaSampler: Sendable {
    public var std: Float
    public var eps: Float
    public var uniformProbability: Float

    public init(std: Float = 1, eps: Float = 1e-3, uniformProbability: Float = 0.1) {
        self.std = std
        self.eps = eps
        self.uniformProbability = uniformProbability
    }

    /// The logit-normal's mean `μ` for a sequence of `tokens`.
    public static func shift(tokens: Int) -> Double {
        let slope = (2.05 - 0.95) / Double(4096 - 1024)
        return slope * Double(tokens) + (0.95 - slope * 1024)
    }

    /// The sigmas the sampler returns for the draws `normal` (standard normal), `uniform`, and `choice`
    /// (both on `[0, 1)`), each `[B]`, for a sequence of `tokens`.
    public func sigmas(normal: MLXArray, uniform: MLXArray, choice: MLXArray, tokens: Int) -> MLXArray {
        let mu = Self.shift(tokens: tokens)
        // The reference forms each bound as a float32 tensor from a double before its sigmoid.
        let upper = sigmoid(MLXArray(Float(mu + 3.0902 * Double(std))))
        let lower = sigmoid(MLXArray(Float(mu - 2.5758 * Double(std))))
        let raw = (sigmoid(normal * std + Float(mu)) - lower) / (upper - lower)
        let stretched = clip(MLX.where(raw .>= eps, raw, 2 * eps - raw), min: 0, max: 1)
        return MLX.where(choice .> uniformProbability, stretched, (1 - eps) * uniform + eps)
    }

    /// `count` sigmas for a sequence of `tokens`, drawn from `key`.
    public func sample(count: Int, tokens: Int, key: MLXArray) -> MLXArray {
        let keys = MLXRandom.split(key: key, into: 3)
        return sigmas(normal: MLXRandom.normal([count], key: keys[0]),
                      uniform: MLXRandom.uniform(low: 0, high: 1, [count], key: keys[1]),
                      choice: MLXRandom.uniform(low: 0, high: 1, [count], key: keys[2]), tokens: tokens)
    }
}

/// One training example: a clip's video and audio latents and its prompt's connector states, as the
/// trainer's preprocessing stores them.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXLTX2TrainingExample {
    /// The normalized video latents, token-major `[frames·height·width, channels]` (frame, then row, then
    /// column), the trainer's patchified layout.
    public var video: MLXArray
    /// The normalized, packed audio latents `[audioFrames, channels·melBins]`.
    public var audio: MLXArray
    /// The video and audio connector outputs for the prompt, `[textTokens, width]` each.
    public var text: MLXArray
    public var audioText: MLXArray
    /// The video latent grid.
    public var frames: Int
    public var height: Int
    public var width: Int
    /// The clip's frame rate, which scales the video's temporal positions.
    public var frameRate: Float

    public init(video: MLXArray, audio: MLXArray, text: MLXArray, audioText: MLXArray, frames: Int, height: Int,
                width: Int, frameRate: Float) {
        self.video = video
        self.audio = audio
        self.text = text
        self.audioText = audioText
        self.frames = frames
        self.height = height
        self.width = width
        self.frameRate = frameRate
    }
}

/// The objective Lightricks' trainer trains LTX-2 with.
///
/// @discussion For each example the sampler draws a sigma (one for both modalities, or one for each), and
/// the noised latents are `(1 − σ)·clean + σ·noise` with the velocity target `noise − clean`. Under
/// first-frame conditioning an example's first latent frame keeps its clean latent, reads timestep zero,
/// and leaves the loss. The loss of an example is the video's squared error averaged over its remaining
/// tokens and every channel, plus the audio's averaged over every token and channel; the batch's loss is
/// the mean over its examples.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXLTX2Objective: Sendable {
    /// Whether the two modalities share one sigma (`text_to_video`) or draw their own (`flexible`).
    public var sharesSigma: Bool
    /// The probability an example's first latent frame is conditioned.
    public var firstFrameConditioning: Float
    public var sampler: NFKMLXLTX2SigmaSampler

    public init(sharesSigma: Bool, firstFrameConditioning: Float, sampler: NFKMLXLTX2SigmaSampler = .init()) {
        self.sharesSigma = sharesSigma
        self.firstFrameConditioning = firstFrameConditioning
        self.sampler = sampler
    }

    /// The `flexible` strategy as the trainer's text-to-video recipes configure it (`t2v_lora.yaml`): both
    /// modalities generated, each with its own sigma, nothing conditioned.
    public static let reference = NFKMLXLTX2Objective(sharesSigma: false, firstFrameConditioning: 0)

    /// The `text_to_video` strategy with audio at its defaults: one sigma, the first frame conditioned for one
    /// example in ten.
    public static let textToVideo = NFKMLXLTX2Objective(sharesSigma: true, firstFrameConditioning: 0.1)

    /// A batch's noised inputs and targets.
    public struct Inputs {
        public var video: MLXArray, audio: MLXArray
        public var videoTargets: MLXArray, audioTargets: MLXArray
        /// Per-token sigmas `[B, tokens]`, zero on conditioning tokens, and `[B, audioFrames]`.
        public var timesteps: MLXArray, audioTimesteps: MLXArray
        public var sigma: MLXArray, audioSigma: MLXArray
        /// 1 on the video tokens the loss reads `[B, tokens]`.
        public var lossMask: MLXArray
    }

    /// Noises `video` `[B, tokens, C]` and `audio` `[B, audioFrames, C']` at `sigma` and `audioSigma` `[B]`
    /// with `noise` and `audioNoise`, leaving the first `tokensPerFrame` video tokens clean where
    /// `conditioned` `[B]` is 1.
    public static func inputs(video: MLXArray, audio: MLXArray, sigma: MLXArray, audioSigma: MLXArray,
                              noise: MLXArray, audioNoise: MLXArray, conditioned: MLXArray,
                              tokensPerFrame: Int) -> Inputs {
        let tokens = video.dim(1)
        let firstFrame = MLXArray((0 ..< tokens).map { Float($0 < tokensPerFrame && tokensPerFrame < tokens ? 1 : 0) })
        let conditioning = conditioned.asType(.float32).reshaped([-1, 1]) * firstFrame.reshaped([1, -1])
        let s = sigma.reshaped([-1, 1, 1]), audioS = audioSigma.reshaped([-1, 1, 1])
        let noisy = (1 - s) * video + s * noise
        let kept = conditioning.expandedDimensions(axis: -1) .> 0
        return Inputs(
            video: MLX.where(kept, video, noisy), audio: (1 - audioS) * audio + audioS * audioNoise,
            videoTargets: noise - video, audioTargets: audioNoise - audio,
            timesteps: (1 - conditioning) * sigma.reshaped([-1, 1]),
            audioTimesteps: MLX.broadcast(audioSigma.reshaped([-1, 1]), to: [audio.dim(0), audio.dim(1)]),
            sigma: sigma, audioSigma: audioSigma, lossMask: 1 - conditioning)
    }

    /// Each example's loss `[B]` for the velocity predictions against `inputs`.
    public static func exampleLosses(video: MLXArray, audio: MLXArray, inputs: Inputs) -> MLXArray {
        let mask = inputs.lossMask.asType(video.dtype).expandedDimensions(axis: -1)
        let videoLoss = ((video - inputs.videoTargets).square() * mask).mean(axes: [1, 2])
            / maximum(MLX.broadcast(mask, to: video.shape).mean(axes: [1, 2]), MLXArray(Float(1e-8)))
        return videoLoss + (audio - inputs.audioTargets).square().mean(axes: [1, 2])
    }

    /// The batch's loss: the mean of ``exampleLosses(video:audio:inputs:)``.
    public static func loss(video: MLXArray, audio: MLXArray, inputs: Inputs) -> MLXArray {
        exampleLosses(video: video, audio: audio, inputs: inputs).mean()
    }

    /// The velocity predictions of `net` for `inputs`, with the trainer's cross-modal modulation and no
    /// text mask (the connectors fill every padded position with a register). The latents and the text enter
    /// in the type of the transformer's weights, as the trainer's bfloat16 autocast runs them, so a bfloat16
    /// base computes in bfloat16; the predictions return at float32 for the loss.
    public static func predictions(_ net: NFKMLXLTX2TransformerNet, inputs: Inputs, text: MLXArray, audioText: MLXArray,
                                   frames: Int, height: Int, width: Int, frameRate: Float)
        -> (video: MLXArray, audio: MLXArray) {
        let scale = net.config.timestepScaleMultiplier
        let dtype = net.computeType
        let (video, audio) = net(video: inputs.video.asType(dtype), audio: inputs.audio.asType(dtype),
                                 text: text.asType(dtype), audioText: audioText.asType(dtype),
                                 timestep: inputs.timesteps * scale, audioTimestep: inputs.audioTimesteps * scale,
                                 sigma: inputs.sigma * scale, frames: frames, height: height, width: width,
                                 audioFrames: inputs.audio.dim(1), fps: frameRate, audioSigma: inputs.audioSigma * scale,
                                 gatesReadCrossSigma: true)
        return (video.asType(.float32), audio.asType(.float32))
    }

    /// Scores `net` on a batch from its draws: noises it as ``inputs(video:audio:sigma:audioSigma:noise:audioNoise:conditioned:tokensPerFrame:)``
    /// and returns ``loss(video:audio:inputs:)``.
    public static func score(_ net: NFKMLXLTX2TransformerNet, video: MLXArray, audio: MLXArray, text: MLXArray,
                             audioText: MLXArray, sigma: MLXArray, audioSigma: MLXArray, noise: MLXArray,
                             audioNoise: MLXArray, conditioned: MLXArray, frames: Int, height: Int, width: Int,
                             frameRate: Float) -> MLXArray {
        let batch = inputs(video: video, audio: audio, sigma: sigma, audioSigma: audioSigma, noise: noise,
                           audioNoise: audioNoise, conditioned: conditioned, tokensPerFrame: height * width)
        let (videoPrediction, audioPrediction) = predictions(net, inputs: batch, text: text, audioText: audioText,
                                                             frames: frames, height: height, width: width,
                                                             frameRate: frameRate)
        return loss(video: videoPrediction, audio: audioPrediction, inputs: batch)
    }

    /// The draws for a batch of `count` examples of `tokens` video tokens: the sigmas, then the first-frame
    /// conditioning flags `[B]`.
    func draws(count: Int, tokens: Int, audioTokens: Int, key: MLXArray)
        -> (sigma: MLXArray, audioSigma: MLXArray, conditioned: MLXArray) {
        let keys = MLXRandom.split(key: key, into: 3)
        let sigma = sampler.sample(count: count, tokens: tokens, key: keys[0])
        let audioSigma = sharesSigma ? sigma : sampler.sample(count: count, tokens: audioTokens, key: keys[1])
        let conditioned = (MLXRandom.uniform(low: 0, high: 1, [count], key: keys[2]) .< firstFrameConditioning)
            .asType(.float32)
        return (sigma, audioSigma, conditioned)
    }
}

/// Which parameters an LTX-2 fine-tune updates.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXLTX2Trainable: Sendable {
    /// LoRA adapters on the `to_q`, `to_k`, `to_v`, and `to_out.0` projections of every attention in every
    /// block, the trainer's `target_modules`; everything else frozen.
    case lora(rank: Int, alpha: Float)

    /// The trainer's text-to-video recipe (`t2v_lora.yaml`): rank 32, alpha 32.
    public static let reference = NFKMLXLTX2Trainable.lora(rank: 32, alpha: 32)

    /// Whether `path` names one of the trainer's target projections.
    static func targets(_ path: String) -> Bool {
        let parts = path.split(separator: ".")
        guard parts.first == "transformer_blocks", let last = parts.last else { return false }
        return ["to_q", "to_k", "to_v"].contains(last) || path.hasSuffix(".to_out.0")
    }
}

extension NFKMLXLTX2Generator {

    /// The trainer's optimizer settings: AdamW (torch's defaults, weight decay 0.01) at 1e-4, the gradient
    /// norm clipped to 1, and `LinearLR` from the full rate to a tenth of it over the run.
    ///
    /// Introduced in InferKit 0.4.0.
    public static let referenceLearningRate: Float = 1e-4
    public static let referenceWeightDecay: Float = 0.01
    public static let referenceGradientNorm: Float = 1

    /// Builds the transformer of a release directory, ready to fine-tune: `variant`'s transformer folder,
    /// `transformer/` where it is nil, and LTX-2.5's full transformer for ``NFKMLXLTX2Variant/ltx25Full``.
    /// `quantizationBits` loads it one block at a time, quantizing each block's linear layers as it arrives,
    /// so the float transformer is never resident whole; the adapters then train over the packed weights
    /// (QLoRA).
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, variant: NFKMLXLTX2Variant? = nil, quantizationBits: Int? = nil,
                               quantizationGroupSize: Int = 64) throws -> NFKMLXLTX2TransformerNet {
        let transformerDirectory = directoryURL.appendingPathComponent(variant?.transformerFolder ?? "transformer")
        let net = NFKMLXLTX2TransformerNet(try NFKMLXLTX2TransformerNet.configuration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json")))
        if let quantizationBits {
            try NFKMLXLTX2TransformerNet.loadQuantizedWeights(into: net, from: transformerDirectory,
                                                              bits: quantizationBits, groupSize: quantizationGroupSize)
        } else {
            try NFKMLXLTX2TransformerNet.loadWeights(into: net, from: transformerDirectory, precision: .checkpoint)
        }
        return net
    }

    /// Fine-tunes the LTX-2 transformer on a consumer's own clips, returning the loss from each step.
    ///
    /// The customization path is four calls: ``trainingExample(video:frameRate:audio:sampleRate:prompt:)``
    /// for each clip, ``network(directoryURL:variant:quantizationBits:quantizationGroupSize:)`` to build, this to
    /// train, and ``NFKMLXLoRA/adapterArrays(of:)`` with ``saveAdapters(of:to:)`` to write the adapters a
    /// generator applies with ``applyAdapters(at:)``.
    ///
    /// - Parameters:
    ///   - net: the transformer to train.
    ///   - examples: one batch per step; every example in a batch holds the same latent grid, audio length,
    ///     and text length.
    ///   - trainable: the reference's LoRA (rank 32, alpha 32). Adapters persist on `net`.
    ///   - objective: the reference's `flexible` text-to-video objective, or ``NFKMLXLTX2Objective/textToVideo``.
    ///   - optimizer: nil uses torch's AdamW (bias-corrected, betas 0.9 and 0.999) at `learningRate` and
    ///     `weightDecay`.
    ///   - learningRate: the reference optimizer's learning rate, the trainer's 1e-4 by default.
    ///   - weightDecay: the reference optimizer's decoupled weight decay, torch's 0.01 by default.
    ///   - steps: how many updates to run; the reference's linear decay spans them.
    ///   - seed: seeds the sigmas, the noise, and the conditioning.
    ///   - clipGradientNorm: the reference clips the global norm to 1.
    ///   - recomputesBlocksInBackward: recomputes each block's activations in the backward pass (gradient
    ///     checkpointing), as the trainer's recipes do; on by default. A step then holds one block's
    ///     activations at a time.
    ///   - accumulationSteps: how many batches each update averages.
    ///   - learningRateSchedule: nil uses the reference's `LinearLR` (1 to 0.1) with the reference optimizer.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The reference trains in bfloat16 autocast; this runs in the type the weights load at. A run is
    /// multi-second per step; call it off the render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXLTX2TransformerNet,
        examples: (Int) -> [NFKMLXLTX2TrainingExample],
        trainable: NFKMLXLTX2Trainable = .reference,
        objective: NFKMLXLTX2Objective = .reference,
        optimizer: Optimizer? = nil,
        learningRate: Float = referenceLearningRate,
        weightDecay: Float = referenceWeightDecay,
        steps: Int,
        seed: UInt64 = 0,
        clipGradientNorm: Float? = referenceGradientNorm,
        recomputesBlocksInBackward: Bool = true,
        accumulationSteps: Int = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        var key = MLXRandom.key(seed)
        var grid = (frames: 0, height: 0, width: 0, frameRate: Float(24))
        let recomputed = net.recomputesBlocksInBackward
        net.recomputesBlocksInBackward = recomputesBlocksInBackward
        defer { net.recomputesBlocksInBackward = recomputed }
        return try NFKMLXFineTune.run(
            net,
            freezing: {
                switch trainable {
                case .lora(let rank, let alpha):
                    try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha) { path, _ in
                        NFKMLXLTX2Trainable.targets(path)
                    }
                }
            },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: learningRate, weightDecay: weightDecay) },
            referenceSchedule: { .linearFactor(steps: steps, start: 1, end: 0.1) },
            steps: steps,
            arrays: { step in
                let batch = examples(step)
                guard let first = batch.first else { return [] }
                grid = (first.frames, first.height, first.width, first.frameRate)
                let video = MLX.stacked(batch.map(\.video)), audio = MLX.stacked(batch.map(\.audio))
                let keys = MLXRandom.split(key: key, into: 4)
                key = keys[0]
                let draws = objective.draws(count: batch.count, tokens: video.dim(1), audioTokens: audio.dim(1),
                                            key: keys[1])
                return [video, audio, MLX.stacked(batch.map(\.text)), MLX.stacked(batch.map(\.audioText)),
                        draws.sigma, draws.audioSigma, MLXRandom.normal(video.shape, key: keys[2]),
                        MLXRandom.normal(audio.shape, key: keys[3]), draws.conditioned]
            },
            loss: { net, arrays in
                NFKMLXLTX2Objective.score(net, video: arrays[0], audio: arrays[1], text: arrays[2],
                                          audioText: arrays[3], sigma: arrays[4], audioSigma: arrays[5],
                                          noise: arrays[6], audioNoise: arrays[7], conditioned: arrays[8],
                                          frames: grid.frames, height: grid.height, width: grid.width,
                                          frameRate: grid.frameRate)
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }

    /// Encodes one clip as the trainer's preprocessing does: the frames through the video encoder, the
    /// soundtrack through the trainer's log-mel front end and the audio encoder, and the prompt through the
    /// text encoder and the connectors at 1024 tokens.
    ///
    /// - Parameters:
    ///   - video: the frames `[8k + 1, 32h, 32w, 3]` in `[0, 1]`, already cut and sized as the trainer's
    ///     bucketing leaves them.
    ///   - frameRate: the clip's frame rate, which scales the video's temporal positions.
    ///   - audio: the soundtrack `[samples, channels]`. Mono is duplicated to stereo, 5.1 and 7.1 are mixed down
    ///     with the trainer's weights, and any other layout is averaged and duplicated.
    ///   - sampleRate: the soundtrack's rate; it is resampled to the encoder's.
    ///   - prompt: the clip's caption.
    /// - Throws: `NFKMLXError.unsupportedConfiguration` for a clip whose frame count or size the autoencoder
    ///   cannot encode, or a generator assembled without its release directory's encoders.
    ///
    /// Introduced in InferKit 0.4.0.
    public func trainingExample(video: MLXArray, frameRate: Float, audio: MLXArray, sampleRate: Int,
                                prompt: String) throws -> NFKMLXLTX2TrainingExample {
        let (frames, height, width) = (video.dim(0), video.dim(1), video.dim(2))
        guard frames % 8 == 1, height % 32 == 0, width % 32 == 0, height > 0, width > 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "a training clip is 8k + 1 frames of a multiple of 32 pixels on each side, not \(frames) of "
                + "\(width)×\(height)")
        }
        let tokens = tokens(prompt)
        return try staging.exclusively { () throws -> NFKMLXLTX2TrainingExample in
            let text = try connectorStates(tokens)
            let encoded = try encodedClip(video: video, audio: audio, sampleRate: sampleRate)
            let (latentFrames, latentHeight, latentWidth) = (encoded.video.dim(1), encoded.video.dim(2),
                                                             encoded.video.dim(3))
            return NFKMLXLTX2TrainingExample(
                video: encoded.video.reshaped([latentFrames * latentHeight * latentWidth, encoded.video.dim(4)]),
                audio: encoded.audio[0], text: text.video[0], audioText: text.audio[0], frames: latentFrames,
                height: latentHeight, width: latentWidth, frameRate: frameRate)
        }
    }

    /// The connector outputs for `tokens` at 1024 positions, video then audio.
    func connectorStates(_ tokens: [Int]) throws -> (video: MLXArray, audio: MLXArray) {
        let states: MLXArray = try staging.with(textStage) { stage in
            let states = stage.states(tokens)
            eval(states)
            try stage.verify()
            return states
        }
        let text: [MLXArray] = try staging.with(connectorStage) { connectors in
            let pair = connectors(validStates: states, length: Self.maximumSequenceLength)
            return [pair.video, pair.audio]
        }
        return (text[0], text[1])
    }

    /// The normalized video latents `[1, F', h, w, C]` and packed audio latents `[1, T', C']` of a clip.
    func encodedClip(video: MLXArray, audio: MLXArray, sampleRate: Int) throws -> (video: MLXArray, audio: MLXArray) {
        let encoded = try withEncoders { (encoders: NFKLTX2EncoderStage) -> [MLXArray] in
            let clip = video.asType(.float32).expandedDimensions(axis: 0) * 2 - 1
            let latents = encoders.video.latents(clip, statistics: encoders.videoStatistics)
            let frontEnd = NFKMLXLTX2MelFrontEnd(encoders.audio.configuration)
            let mel = frontEnd.logMel(frontEnd.resampled(Self.stereo(audio.asType(.float32)), from: sampleRate))
            return [latents, encoders.audio.latents(mel, statistics: encoders.audioStatistics)]
        }
        return (encoded[0], encoded[1])
    }

    /// `audio` `[samples, channels]` as the stereo `[2, samples]` the audio encoder reads, mixed as the trainer
    /// mixes it.
    static func stereo(_ audio: MLXArray) -> MLXArray {
        let channels = audio.transposed()
        switch channels.dim(0) {
        case 1:
            return concatenated([channels, channels], axis: 0)
        case 2:
            return channels
        case 6, 8:
            let (main, other): (Float, Float) = channels.dim(0) == 6 ? (1 / 2.414, 0.707 / 2.414)
                                                                     : (1 / 3.121, 0.707 / 3.121)
            var left = main * channels[0] + other * (channels[2] + channels[4])
            var right = main * channels[1] + other * (channels[2] + channels[5])
            if channels.dim(0) == 8 {
                left = left + other * channels[6]
                right = right + other * channels[7]
            }
            return MLX.stacked([left, right])
        default:
            let mono = channels.mean(axis: 0, keepDims: true)
            return concatenated([mono, mono], axis: 0)
        }
    }

    /// Writes `net`'s adapters to `url` as one safetensors, with their rank and alpha in its metadata. The
    /// base stays as the release ships it, so the file is the whole of the run.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func saveAdapters(of net: NFKMLXLTX2TransformerNet, to url: URL) throws {
        let arrays = NFKMLXLoRA.adapterArrays(of: net)
        guard let adapter = net.leafModules().flattened().lazy.compactMap({ $0.1 as? NFKMLXLoRAAdapter }).first else {
            throw NFKMLXError.loRANotApplicable("the transformer holds no adapter to save")
        }
        let rank = adapter.loraA.dim(1)
        try MLX.save(arrays: arrays, metadata: ["rank": "\(rank)", "alpha": "\(adapter.scale * Float(rank))"],
                     url: url)
    }
}
