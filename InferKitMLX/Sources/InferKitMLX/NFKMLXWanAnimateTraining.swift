//
//  NFKMLXWanAnimateTraining.swift
//  InferKitMLX
//
//  Fine-tuning the Wan 2.2 Animate 2 transformer with LoRA as DiffSynth-Studio trains it
//  (modelscope/DiffSynth-Studio at 1e8ae1d, `examples/wanvideo/model_training/lora/Wan2.2-Animate-2-14B.sh`).
//  The trainer runs both passes of the model in one forward (`forward_origin`): the reference pass over the
//  driving video's latents at timestep 1, whose keys and values the generation pass then attends over, and
//  the generation pass over the noised stack of the reference image's latent and the clip's latents. Both
//  passes carry the adapters, and the whole stack enters the loss.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// One Wan 2.2 Animate 2 training example: a clip, the character it shows, and the motion that drives it,
/// as the encoders prepare them.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanAnimateTrainingExample {
    /// The clean latents `[zDim, 1 + frames', H', W']`: the reference image's latent, then the clip's.
    public var latent: MLXArray
    /// The channels read beside them `[4 + zDim, 1 + frames', H', W']`: the reference image's mask and latent,
    /// then an empty continuation slot.
    public var condition: MLXArray
    /// The driving video's latents `[zDim, frames', H', W']`, which the reference pass reads.
    public var reference: MLXArray
    /// The channels read beside them `[4 + zDim, frames', H', W']`: a mask over every frame and the latents.
    public var referenceCondition: MLXArray
    /// The prompt's features `[textLength, textDim]`, zero-padded.
    public var text: MLXArray
    /// The reference pass's prompt features, the same shape.
    public var referenceText: MLXArray
    /// The image encoder's features of the reference image `[257, imageDim]`, or nil for a configuration
    /// without an image branch.
    public var imageFeatures: MLXArray?
    /// The image encoder's features of the driving video's first frame, the same shape.
    public var referenceImageFeatures: MLXArray?
    /// The clip's pixel frame count, which sizes the frame-wise attention over the reference.
    public var videoFrames: Int
    /// The clip's pixel area `height · width`.
    public var videoArea: Int

    public init(latent: MLXArray, condition: MLXArray, reference: MLXArray, referenceCondition: MLXArray,
                text: MLXArray, referenceText: MLXArray, imageFeatures: MLXArray?, referenceImageFeatures: MLXArray?,
                videoFrames: Int, videoArea: Int) {
        self.latent = latent
        self.condition = condition
        self.reference = reference
        self.referenceCondition = referenceCondition
        self.text = text
        self.referenceText = referenceText
        self.imageFeatures = imageFeatures
        self.referenceImageFeatures = referenceImageFeatures
        self.videoFrames = videoFrames
        self.videoArea = videoArea
    }
}

/// DiffSynth-Studio's Wan 2.2 Animate 2 LoRA recipe.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanAnimateTrainingPreset: Sendable {
    /// The noise-level draw and loss weight: DiffSynth-Studio's Wan table (``NFKMLXWanTrainingPreset/diffSynth``).
    public var objective: NFKMLXWanObjective
    public var rank: Int
    public var alpha: Float
    public var learningRate: Float
    public var weightDecay: Float
    public var warmupSteps: Int

    /// The script's settings: LoRA on the self-attention query, key, and value projections of every block,
    /// rank and alpha 32, the DiffSynth Wan objective, AdamW at 1e-4 with weight decay 0.01, and a 5-step warm-up.
    public static let diffSynth = NFKMLXWanAnimateTrainingPreset(
        objective: NFKMLXWanTrainingPreset.diffSynth.objective, rank: 32, alpha: 32, learningRate: 1e-4,
        weightDecay: 0.01, warmupSteps: 5)

    public init(objective: NFKMLXWanObjective, rank: Int, alpha: Float, learningRate: Float, weightDecay: Float,
                warmupSteps: Int) {
        self.objective = objective
        self.rank = rank
        self.alpha = alpha
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.warmupSteps = warmupSteps
    }

    /// Whether a module path is one the recipe adapts: a block's self-attention query, key, or value.
    func adapts(_ path: String) -> Bool {
        path.hasPrefix("blocks.") && ["self_attn.to_q", "self_attn.to_k", "self_attn.to_v"].contains { path.hasSuffix($0) }
    }

    var schedule: NFKMLXLearningRateSchedule {
        let warmup = warmupSteps
        guard warmup > 0 else { return .constant }
        return NFKMLXLearningRateSchedule { step in min(1, Float(step + 1) / Float(warmup)) }
    }
}

extension NFKMLXWanAnimate {

    /// The generation pass's prediction for `noisy` at `timestep`, after the reference pass over the
    /// example's driving latents at timestep 1, both run as the trainer's `forward_origin` runs them: block by
    /// block, the reference pass and then the generation pass.
    static func trainingPrediction(_ net: NFKMLXWanAnimateNet, example: NFKMLXWanAnimateTrainingExample,
                                   noisy: MLXArray, timestep: Float) throws -> MLXArray {
        let dtype = NFKReferenceRounding.parameterType(of: net)
        return net.trainingPrediction(
            reference: (example.reference.asType(dtype), example.referenceCondition.asType(dtype),
                        example.referenceText.asType(dtype), example.referenceImageFeatures?.asType(dtype)),
            latent: noisy.asType(dtype), condition: example.condition.asType(dtype), text: example.text.asType(dtype),
            imageEmbeddings: example.imageFeatures?.asType(dtype), timestep: MLXArray(timestep),
            videoFrames: example.videoFrames / 4 + 1, videoArea: example.videoArea / 256)
    }

    /// The loss of one example at `draw` with `noise`: `weight · mean((prediction − (noise − clean))²)`.
    static func score(_ net: NFKMLXWanAnimateNet, example: NFKMLXWanAnimateTrainingExample,
                      objective: NFKMLXWanObjective, draw: NFKMLXWanTrainingDraw, noise: MLXArray) throws -> MLXArray {
        let noisy = (1 - draw.sigma) * example.latent + draw.sigma * noise
        let prediction = try trainingPrediction(net, example: example, noisy: noisy, timestep: draw.timestep)
        return NFKMLXWanObjective.loss(prediction: prediction, clean: example.latent, noise: noise, weight: draw.weight)
    }

    /// Builds the transformer of a release directory, ready to fine-tune. A `quantizationBits` of 2 to 8 packs
    /// the blocks' linear layers as each loads, so the adapters train over a packed base (QLoRA).
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, quantizationBits: Int = 0) throws -> NFKMLXWanAnimateNet {
        let directory = directoryURL.appendingPathComponent("transformer")
        let net = NFKMLXWanAnimateNet(try NFKMLXWanAnimateRelease.transformerConfiguration(
            fromHuggingFace: directory.appendingPathComponent("config.json")))
        try NFKMLXWanAnimateRelease.loadTransformer(into: net, fromDirectory: directory, precision: .checkpoint,
                                                    bits: quantizationBits == 0 ? nil : quantizationBits)
        return net
    }

    /// Fine-tunes the Animate 2 transformer with LoRA as `preset`'s recipe does, returning each step's loss.
    ///
    /// The customization path is four calls: ``NFKMLXWanAnimateGenerator/trainingExample(referenceImage:drivingVideo:video:prompt:referencePrompt:)``
    /// for each clip, ``network(directoryURL:quantizationBits:)`` to build, this to train, and
    /// ``saveAdapters(of:to:)`` to write the adapters as DiffSynth-Studio does, which a generator applies with
    /// ``NFKMLXWanAnimateGenerator/addAdapter(at:scale:)``. A step is multi-second; call it off the render thread.
    /// `recomputesBlocksInBackward` (on by default, as the recipe's `--use_gradient_checkpointing`) recomputes
    /// each block in the backward pass (``NFKMLXWanAnimateNet/recomputesBlocksInBackward``) for the run, and the
    /// network's own setting returns after it.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXWanAnimateNet,
        examples: (Int) -> [NFKMLXWanAnimateTrainingExample],
        preset: NFKMLXWanAnimateTrainingPreset = .diffSynth,
        optimizer: Optimizer? = nil,
        steps: Int,
        seed: UInt64 = 0,
        accumulationSteps: Int = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        recomputesBlocksInBackward: Bool = true,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let recomputed = net.recomputesBlocksInBackward
        net.recomputesBlocksInBackward = recomputesBlocksInBackward
        defer { net.recomputesBlocksInBackward = recomputed }
        var key = MLXRandom.key(seed)
        var batch = [NFKMLXWanAnimateTrainingExample]()
        var draws = [NFKMLXWanTrainingDraw]()
        let objective = preset.objective
        let dtype = NFKReferenceRounding.parameterType(of: net)
        var failure: Error?
        let losses = try NFKMLXBlockedAttention.training {
            try NFKMLXFineTune.run(
                net,
                freezing: {
                    try NFKMLXLoRA.apply(to: net, rank: preset.rank, alpha: preset.alpha) { path, _ in preset.adapts(path) }
                },
                optimizer: optimizer,
                reference: {
                    NFKMLXReferenceOptimizers.adamW(learningRate: preset.learningRate, betas: (0.9, 0.999),
                                                    weightDecay: preset.weightDecay)
                },
                referenceSchedule: { preset.schedule },
                steps: steps,
                arrays: { step in
                    batch = examples(step)
                    let keys = MLXRandom.split(key: key, into: 3)
                    key = keys[0]
                    draws = objective.sampler.raw(count: batch.count, key: keys[1]).map {
                        objective.sampler.draw($0, weighting: objective.weighting, dtype: dtype)
                    }
                    let noiseKeys = MLXRandom.split(key: keys[2], into: batch.count)
                    return batch.indices.map { MLXRandom.normal(batch[$0].latent.shape, key: noiseKeys[$0]) }
                },
                loss: { net, arrays in
                    let losses = batch.indices.map { index -> MLXArray in
                        do {
                            return try score(net, example: batch[index], objective: objective, draw: draws[index],
                                             noise: arrays[index])
                        } catch {
                            failure = failure ?? error
                            return MLXArray(Float(0))
                        }
                    }
                    return MLX.stacked(losses).mean()
                },
                clipGradientNorm: nil, accumulationSteps: accumulationSteps,
                learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
        }
        if let failure {
            throw failure
        }
        return losses
    }

    /// Writes the adapters `net` carries as DiffSynth-Studio saves them: each layer's original name under PEFT's
    /// `default` adapter (`blocks.<n>.self_attn.q.lora_A.default.weight`), the scaling folded into the up
    /// weight, since its loader fuses a file at a strength of 1. ``NFKMLXWanLoRA`` reads the file back.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func saveAdapters(of net: NFKMLXWanAnimateNet, to url: URL) throws {
        var arrays = [String: MLXArray]()
        for (path, layer) in net.leafModules().flattened() {
            guard let adapter = layer as? NFKMLXLoRAAdapter else { continue }
            let name = try originalName(path)
            arrays["\(name).lora_A.default.weight"] = adapter.loraA.transposed(1, 0)
            arrays["\(name).lora_B.default.weight"] = adapter.loraB.transposed(1, 0) * adapter.scale
        }
        try MLX.save(arrays: arrays, url: url)
    }

    /// The release's original name of a block's self-attention projection, ``moduleKey(forRelease:)`` inverted.
    static func originalName(_ path: String) throws -> String {
        let parts = path.split(separator: ".").map(String.init)
        let names = ["to_q": "q", "to_k": "k", "to_v": "v", "to_out": "o"]
        guard parts.count >= 4, parts[0] == "blocks", let name = names[parts[3]] else {
            throw NFKMLXError.unsupportedConfiguration("\(path) is not a block attention layer a DiffSynth file names")
        }
        return "blocks.\(parts[1]).block.\(parts[2]).\(name)"
    }
}

extension NFKMLXWanAnimateGenerator {

    /// Encodes one training clip as DiffSynth-Studio's Animate 2 units prepare it, with this generator's own
    /// encoders: the reference image's latent stacked before the clip's, the condition of the reference image's
    /// mask and latent before an empty continuation slot, the driving video's latents under a mask over every
    /// frame, both prompts' umT5 features, and the image encoder's features of the reference image and of the
    /// driving video's first frame.
    ///
    /// - Parameters:
    ///   - referenceImage: the character `[H, W, 3]` in `[0, 1]`, at the clip's size.
    ///   - drivingVideo: the motion `[4k + 1, H, W, 3]` in `[0, 1]`, the clip's length and size.
    ///   - video: the clip to learn `[4k + 1, H, W, 3]` in `[0, 1]`; the sides are multiples of 16.
    ///   - text: the clip's caption.
    ///   - referencePrompt: the reference pass's prompt; nil takes the one generation reads.
    ///
    /// Introduced in InferKit 0.4.0.
    public func trainingExample(referenceImage: MLXArray, drivingVideo: MLXArray, video: MLXArray, prompt text: String,
                                referencePrompt: String? = nil) throws -> NFKMLXWanAnimateTrainingExample {
        let frames = video.dim(0), height = video.dim(1), width = video.dim(2)
        guard video.ndim == 4, (frames - 1) % 4 == 0, drivingVideo.shape == video.shape,
              referenceImage.shape == [height, width, 3] else {
            throw NFKMLXError.unsupportedConfiguration(
                "a training clip and its driving video are 4k + 1 frames of one size, and the reference image that size")
        }
        let prompts = [self.prompt(text), self.prompt(referencePrompt ?? Self.referencePrompt)]
        let clip = video * 2 - 1, driving = drivingVideo * 2 - 1, reference = referenceImage * 2 - 1
        let arrays = try usingStages { encoders, pipeline -> [MLXArray] in
            let features = prompts.map { encoders.text.features($0) }
                + [encoders.image.features(reference), encoders.image.features(driving[0])]
            let referenceLatent = pipeline.encode(reference.expandedDimensions(axis: 0))
            let clipLatent = pipeline.encode(clip)
            let empty = pipeline.encode(MLXArray.zeros(clip.shape))
            let drivingLatent = pipeline.encode(driving)
            let (h, w, latentFrames) = (clipLatent.dim(2), clipLatent.dim(3), clipLatent.dim(1))
            let condition = concatenated([
                concatenated([NFKMLXWanAnimatePipeline.conditionMask(latentFrames: 1, height: h, width: w, frames: 1),
                              referenceLatent], axis: 0),
                concatenated([NFKMLXWanAnimatePipeline.conditionMask(latentFrames: latentFrames, height: h, width: w,
                                                                     frames: 0), empty], axis: 0),
            ], axis: 1)
            let referenceCondition = concatenated([
                NFKMLXWanAnimatePipeline.conditionMask(latentFrames: latentFrames, height: h, width: w, frames: frames),
                drivingLatent,
            ], axis: 0)
            let result = [concatenated([referenceLatent, clipLatent], axis: 1), condition, drivingLatent,
                          referenceCondition] + features
            eval(result)
            return result
        }
        return NFKMLXWanAnimateTrainingExample(
            latent: arrays[0], condition: arrays[1], reference: arrays[2], referenceCondition: arrays[3],
            text: arrays[4], referenceText: arrays[5], imageFeatures: arrays[6], referenceImageFeatures: arrays[7],
            videoFrames: frames, videoArea: height * width)
    }
}
