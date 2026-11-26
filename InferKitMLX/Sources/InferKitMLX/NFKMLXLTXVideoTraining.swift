//
//  NFKMLXLTXVideoTraining.swift
//  InferKitMLX
//
//  Fine-tuning the LTX-Video 0.9.0 transformer the way Lightricks' trainer does (Lightricks/LTX-Video-Trainer
//  at e055182, `ltxv_2b_lora.yaml` with `LTXV_2B_0.9.0`): LoRA adapters on the query, key, value, and output
//  projections of both attentions in every block, over a float base or a quantized one; flow matching with
//  sigmas from a shifted logit-normal sampler, rounded to whole timesteps; the velocity target
//  `noise − clean`; first-frame conditioning that leaves the first latent frame clean, at timestep zero, and
//  out of the loss; and AdamW with a linear decay to a tenth of its rate.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// The trainer's shifted logit-normal sigma sampler (`ShiftedLogitNormalTimestepSampler`).
///
/// @discussion A draw is `sigmoid(n·std + μ)` for a standard normal `n`, where `μ` rises linearly from 0.95
/// at 1024 tokens to 2.05 at 4096 and continues on that line past both ends
/// (``NFKMLXLTX2SigmaSampler/shift(tokens:)``). The LTX-Video trainer neither stretches nor mixes in uniform
/// draws, which the LTX-2 trainer's sampler does.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXLTXVideoSigmaSampler: Sendable {
    public var std: Float

    public init(std: Float = 1) {
        self.std = std
    }

    /// The sigmas for the standard normal draws `normal` `[B]` and a sequence of `tokens`.
    public func sigmas(normal: MLXArray, tokens: Int) -> MLXArray {
        sigmoid(normal * std + Float(NFKMLXLTX2SigmaSampler.shift(tokens: tokens)))
    }

    /// `count` sigmas for a sequence of `tokens`, drawn from `key`.
    public func sample(count: Int, tokens: Int, key: MLXArray) -> MLXArray {
        sigmas(normal: MLXRandom.normal([count], key: key), tokens: tokens)
    }
}

/// One training example: a clip's latents and its caption's T5 states, as the trainer's preprocessing stores
/// them.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXLTXVideoTrainingExample {
    /// The normalized latents, token-major `[frames·height·width, channels]` (frame, then row, then column).
    public var latents: MLXArray
    /// The caption's T5 states `[textTokens, captionChannels]`, padded.
    public var text: MLXArray
    /// 1 on the caption's tokens and 0 on its padding, `[textTokens]`.
    public var textMask: MLXArray
    /// The latent grid.
    public var frames: Int
    public var height: Int
    public var width: Int
    /// The clip's frame rate, which scales the rotary's time axis; the trainer takes 24 where a clip states none.
    public var frameRate: Float

    public init(latents: MLXArray, text: MLXArray, textMask: MLXArray, frames: Int, height: Int, width: Int,
                frameRate: Float = 24) {
        self.latents = latents
        self.text = text
        self.textMask = textMask
        self.frames = frames
        self.height = height
        self.width = width
        self.frameRate = frameRate
    }
}

/// The objective Lightricks' trainer trains LTX-Video with (`StandardTrainingStrategy`).
///
/// @discussion A batch draws one sigma per example, and its noised latents are `(1 − σ)·clean + σ·noise`
/// with the velocity target `noise − clean`. The transformer reads a timestep per token: `round(1000·σ)`, and
/// zero on conditioning tokens. Under first-frame conditioning, drawn once for the whole batch, every
/// example's first latent frame keeps its clean latent and leaves the loss. The loss is the squared error
/// times the loss mask, averaged over every token and channel of the batch, divided by the mask's mean.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXLTXVideoObjective: Sendable {
    /// The probability a batch's first latent frame is conditioned.
    public var firstFrameConditioning: Float
    public var sampler: NFKMLXLTXVideoSigmaSampler

    public init(firstFrameConditioning: Float, sampler: NFKMLXLTXVideoSigmaSampler = .init()) {
        self.firstFrameConditioning = firstFrameConditioning
        self.sampler = sampler
    }

    /// The trainer's LoRA recipes (`ltxv_2b_lora.yaml`): the first frame conditioned for one batch in ten.
    public static let reference = NFKMLXLTXVideoObjective(firstFrameConditioning: 0.1)

    /// A batch's noised inputs and targets.
    public struct Inputs {
        public var latents: MLXArray
        public var targets: MLXArray
        /// Per-token timesteps `[B, tokens]`: whole numbers, zero on conditioning tokens.
        public var timesteps: MLXArray
        /// 1 on the tokens the loss reads, `[B, tokens]`.
        public var lossMask: MLXArray
    }

    /// Noises `latents` `[B, tokens, C]` at `sigma` `[B]` with `noise`, leaving the first `tokensPerFrame`
    /// tokens clean where `conditioned` `[B]` is 1.
    public static func inputs(latents: MLXArray, sigma: MLXArray, noise: MLXArray, conditioned: MLXArray,
                              tokensPerFrame: Int) -> Inputs {
        let tokens = latents.dim(1)
        let firstFrame = MLXArray((0 ..< tokens).map { Float($0 < tokensPerFrame && tokensPerFrame < tokens ? 1 : 0) })
        let conditioning = conditioned.asType(.float32).reshaped([-1, 1]) * firstFrame.reshaped([1, -1])
        let s = sigma.reshaped([-1, 1, 1])
        let noisy = (1 - s) * latents + s * noise
        let kept = conditioning.expandedDimensions(axis: -1) .> 0
        let timestep = MLX.round(sigma * 1000).reshaped([-1, 1])
        return Inputs(latents: MLX.where(kept, latents, noisy), targets: noise - latents,
                      timesteps: MLX.where(conditioning .> 0, MLXArray(Float(0)), timestep),
                      lossMask: 1 - conditioning)
    }

    /// The batch's loss for the velocity `prediction` `[B, tokens, C]` against `inputs`.
    public static func loss(prediction: MLXArray, inputs: Inputs) -> MLXArray {
        let mask = inputs.lossMask.asType(prediction.dtype).expandedDimensions(axis: -1)
        return ((prediction - inputs.targets).square() * mask).mean() / mask.mean()
    }

    /// The velocity prediction of `net` for `inputs`, its rotary scaled by `frameRate` as the trainer scales it
    /// (`8 / frameRate` in time, 32 in space) and its cross-attention masked by `textMask` `[B, textTokens]`.
    public static func prediction(_ net: NFKMLXLTXTransformerNet, inputs: Inputs, text: MLXArray, textMask: MLXArray,
                                  frames: Int, height: Int, width: Int, frameRate: Float) -> MLXArray {
        net(inputs.latents, text: text, timestep: inputs.timesteps, grid: (frames, height, width),
            ropeScale: (Float(NFKMLXLTXPipeline.temporalCompression) / frameRate,
                        Float(NFKMLXLTXPipeline.spatialCompression), Float(NFKMLXLTXPipeline.spatialCompression)),
            textMask: textMask)
    }

    /// Scores `net` on a batch from its draws: noises it as ``inputs(latents:sigma:noise:conditioned:tokensPerFrame:)``
    /// and returns ``loss(prediction:inputs:)``.
    public static func score(_ net: NFKMLXLTXTransformerNet, latents: MLXArray, text: MLXArray, textMask: MLXArray,
                             sigma: MLXArray, noise: MLXArray, conditioned: MLXArray, frames: Int, height: Int,
                             width: Int, frameRate: Float) -> MLXArray {
        let batch = inputs(latents: latents, sigma: sigma, noise: noise, conditioned: conditioned,
                           tokensPerFrame: height * width)
        return loss(prediction: prediction(net, inputs: batch, text: text, textMask: textMask, frames: frames,
                                           height: height, width: width, frameRate: frameRate),
                    inputs: batch)
    }

    /// The draws for a batch of `count` examples of `tokens` tokens: the sigmas, then the conditioning flags
    /// `[B]`, one draw for the whole batch.
    func draws(count: Int, tokens: Int, key: MLXArray) -> (sigma: MLXArray, conditioned: MLXArray) {
        let keys = MLXRandom.split(key: key, into: 2)
        let conditioned = MLXRandom.uniform(low: 0, high: 1, [1], key: keys[1]) .< firstFrameConditioning
        return (sampler.sample(count: count, tokens: tokens, key: keys[0]),
                MLX.broadcast(conditioned.asType(.float32), to: [count]))
    }
}

/// Which parameters an LTX-Video fine-tune updates.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXLTXVideoTrainable: Sendable {
    /// LoRA adapters on the `to_q`, `to_k`, `to_v`, and `to_out.0` projections of both attentions in every
    /// block, the trainer's `target_modules`; everything else frozen.
    case lora(rank: Int, alpha: Float)

    /// The trainer's LoRA recipes (`ltxv_2b_lora.yaml`): rank 32, alpha 32.
    public static let reference = NFKMLXLTXVideoTrainable.lora(rank: 32, alpha: 32)

    /// The trainer's target module names.
    static let targetModules = ["to_k", "to_q", "to_v", "to_out.0"]

    /// Whether `path` names one of the trainer's target projections.
    static func targets(_ path: String) -> Bool {
        let parts = path.split(separator: ".")
        guard parts.first == "transformer_blocks", let last = parts.last else { return false }
        return ["to_q", "to_k", "to_v"].contains(last) || path.hasSuffix(".to_out.0")
    }
}

extension NFKMLXLTXVideoGenerator {

    /// The trainer's optimizer settings (`ltxv_2b_lora.yaml`): AdamW (torch's defaults, weight decay 0.01) at
    /// 2e-4, the gradient norm clipped to 1, and `LinearLR` from the full rate to a tenth of it over the run.
    ///
    /// Introduced in InferKit 0.4.0.
    public static let referenceLearningRate: Float = 2e-4
    public static let referenceWeightDecay: Float = 0.01
    public static let referenceGradientNorm: Float = 1

    /// The caption length the trainer's preprocessing pads and truncates to.
    static let trainingSequenceLength = 256

    /// Builds the transformer of a release directory (`transformer/`), ready to fine-tune.
    /// `quantizationBits` loads it one block at a time, packing each block's linear layers as it arrives, so
    /// the float32 transformer is never resident whole; the adapters then train over the packed weights
    /// (QLoRA), as the trainer's `quantization` option trains over optimum-quanto's.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, quantizationBits: Int? = nil,
                               quantizationGroupSize: Int = 64) throws -> NFKMLXLTXTransformerNet {
        let transformerDirectory = directoryURL.appendingPathComponent("transformer")
        let net = NFKMLXLTXTransformer.makeNet(try transformerConfiguration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json")))
        if let quantizationBits {
            try NFKMLXLTXTransformer.loadQuantizedWeights(into: net, from: transformerDirectory, bits: quantizationBits,
                                                          groupSize: quantizationGroupSize)
        } else {
            try NFKMLXLTXTransformer.loadWeights(into: net, from: transformerDirectory)
        }
        return net
    }

    /// Fine-tunes the LTX-Video transformer on a consumer's own clips, returning the loss from each step.
    ///
    /// The customization path is four calls: ``trainingExample(video:frameRate:prompt:seed:)`` for each clip,
    /// ``network(directoryURL:quantizationBits:quantizationGroupSize:)`` to build, this to train, and
    /// ``saveAdapters(of:to:)`` to write the adapters in the trainer's own layout, which
    /// ``NFKMLXLTXVideoGenerator/backend(directoryURL:residency:adapterURLs:strengths:)`` holds beside the
    /// released transformer.
    ///
    /// - Parameters:
    ///   - net: the transformer to train.
    ///   - examples: one batch per step; every example in a batch holds the same latent grid and text length.
    ///   - trainable: the reference's LoRA (rank 32, alpha 32). Adapters persist on `net`.
    ///   - objective: the reference's objective, first-frame conditioning at 0.1.
    ///   - optimizer: nil uses torch's AdamW (bias-corrected, betas 0.9 and 0.999) at `learningRate` and
    ///     `weightDecay`.
    ///   - learningRate: the reference optimizer's learning rate, the trainer's 2e-4 by default.
    ///   - weightDecay: the reference optimizer's decoupled weight decay, torch's 0.01 by default.
    ///   - steps: how many updates to run; the reference's linear decay spans them.
    ///   - seed: seeds the sigmas, the noise, the conditioning, and the adapters' initialization.
    ///   - clipGradientNorm: the reference clips the global norm to 1.
    ///   - recomputesBlocksInBackward: recomputes each block's activations in the backward pass (gradient
    ///     checkpointing, which the trainer's low-memory recipe enables); on by default. The gradients are the
    ///     plain backward's.
    ///   - accumulationSteps: how many batches each update averages.
    ///   - learningRateSchedule: nil uses the reference's `LinearLR` (1 to 0.1) with the reference optimizer.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The reference trains a bfloat16 transformer under bfloat16 autocast; this runs in the type the weights
    /// load at, float32 for the release. A run is multi-second per step; call it off the render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXLTXTransformerNet,
        examples: (Int) -> [NFKMLXLTXVideoTrainingExample],
        trainable: NFKMLXLTXVideoTrainable = .reference,
        objective: NFKMLXLTXVideoObjective = .reference,
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
        return try NFKMLXBlockedAttention.training {
            try NFKMLXFineTune.run(
                net,
                freezing: {
                    switch trainable {
                    case .lora(let rank, let alpha):
                        try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha) { path, _ in
                            NFKMLXLTXVideoTrainable.targets(path)
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
                    let latents = MLX.stacked(batch.map(\.latents))
                    let keys = MLXRandom.split(key: key, into: 3)
                    key = keys[0]
                    let draws = objective.draws(count: batch.count, tokens: latents.dim(1), key: keys[1])
                    return [latents, MLX.stacked(batch.map(\.text)), MLX.stacked(batch.map(\.textMask)), draws.sigma,
                            MLXRandom.normal(latents.shape, key: keys[2]), draws.conditioned]
                },
                loss: { net, arrays in
                    NFKMLXLTXVideoObjective.score(net, latents: arrays[0], text: arrays[1], textMask: arrays[2],
                                                  sigma: arrays[3], noise: arrays[4], conditioned: arrays[5],
                                                  frames: grid.frames, height: grid.height, width: grid.width,
                                                  frameRate: grid.frameRate)
                },
                clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps,
                learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, seed: seed, observer: observer)
        }
    }

    /// Encodes one clip as the trainer's preprocessing does: the frames through the autoencoder, a draw from its
    /// posterior normalized by the stored latent statistics, and the caption through T5, padded to 256 tokens.
    ///
    /// - Parameters:
    ///   - video: the frames `[8k + 1, 32h, 32w, 3]` in `[0, 1]`, already cut and sized as the trainer's
    ///     resolution buckets leave them.
    ///   - frameRate: the clip's frame rate, which scales the rotary's time axis.
    ///   - prompt: the clip's caption.
    ///   - seed: seeds the posterior draw.
    /// - Throws: `NFKMLXError.unsupportedConfiguration` for a clip whose frame count or size the autoencoder
    ///   cannot encode.
    ///
    /// Introduced in InferKit 0.4.0.
    public func trainingExample(video: MLXArray, frameRate: Float, prompt: String,
                                seed: UInt64 = 0) throws -> NFKMLXLTXVideoTrainingExample {
        let (frames, height, width) = (video.dim(0), video.dim(1), video.dim(2))
        guard video.ndim == 4, frames % 8 == 1, height % 32 == 0, width % 32 == 0, height > 0, width > 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "a training clip is 8k + 1 frames of a multiple of 32 pixels on each side, not \(frames) of "
                + "\(width)×\(height)")
        }
        let caption = NFKMLXT5Prompt(prompt, segmenter: segmenter, length: Self.trainingSequenceLength)
        let clip = video.asType(.float32).expandedDimensions(axis: 0) * 2 - 1
        return try staging.exclusively { () throws -> NFKMLXLTXVideoTrainingExample in
            let text = try staging.with(textStage) { stage -> MLXArray in
                let states = stage.encoder(caption.tokens)[0]
                eval(states)
                return states
            }
            let encoded = try staging.use(pipelineStage) { pipeline in
                pipeline.trainingLatents(clip, key: MLXRandom.key(seed))
            }
            return NFKMLXLTXVideoTrainingExample(latents: encoded.latents, text: text,
                                                 textMask: caption.maskArray[0].asType(.float32),
                                                 frames: encoded.frames, height: encoded.height, width: encoded.width,
                                                 frameRate: frameRate)
        }
    }

    /// Writes `net`'s adapters to `url` in the trainer's layout: `transformer.<module>.lora_A.weight`
    /// `[rank, inputs]` and `lora_B.weight` `[outputs, rank]`, as `save_lora_weights` writes them, with the rank,
    /// alpha, and target modules in the `lora_adapter_metadata` diffusers reads back. The base stays as the
    /// release ships it, so the file is the whole of the run.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func saveAdapters(of net: NFKMLXLTXTransformerNet, to url: URL) throws {
        var arrays = [String: MLXArray]()
        var rank = 0, alpha: Float = 0
        for (path, layer) in net.leafModules().flattened() {
            guard let adapter = layer as? NFKMLXLoRAAdapter else { continue }
            let down = adapter.loraA.transposed(1, 0), up = adapter.loraB.transposed(1, 0)
            rank = down.dim(0)
            alpha = adapter.scale * Float(rank)
            arrays["transformer.\(path).lora_A.weight"] = down
            arrays["transformer.\(path).lora_B.weight"] = up
        }
        guard !arrays.isEmpty else {
            throw NFKMLXError.loRANotApplicable("the transformer holds no adapter to save")
        }
        let configuration: [String: Any] = ["transformer.r": rank, "transformer.lora_alpha": alpha,
                                            "transformer.target_modules": NFKMLXLTXVideoTrainable.targetModules]
        let text = String(decoding: try JSONSerialization.data(withJSONObject: configuration, options: [.sortedKeys]),
                          as: UTF8.self)
        try MLX.save(arrays: arrays, metadata: ["format": "pt", "lora_adapter_metadata": text], url: url)
    }
}
