//
//  NFKMLXWanDancerTraining.swift
//  InferKitMLX
//
//  LoRA fine-tuning of a Wan-Dancer transformer: the network built from a release file, a training example built
//  by the generator's own encoders, and DiffSynth-Studio's own Wan-Dancer recipe.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXOptimizers

/// One Wan-Dancer training clip as DiffSynth-Studio's units leave it.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanDancerTrainingExample {
    /// The clip's standardized latent mean `[16, F', h, w]`.
    public var latent: MLXArray
    /// The text features `[512, textDim]`, zeros past the prompt.
    public var text: MLXArray
    /// The pass inputs: the keyframe condition, the two CLIP feature sets, the music features, and the frame rate,
    /// the music encoder reading one span as DiffSynth-Studio trains it.
    public var inputs: NFKMLXWanDancerInputs

    public init(latent: MLXArray, text: MLXArray, inputs: NFKMLXWanDancerInputs) {
        self.latent = latent
        self.text = text
        self.inputs = inputs
    }
}

extension NFKMLXWanDancerGenerator {

    /// DiffSynth-Studio's Wan-Dancer LoRA recipe (`lora/Wan-Dancer-14B-local.sh` and `-global.sh` at 1e8ae1d):
    /// ``NFKMLXWanTrainingPreset/diffSynth`` (an index uniform over the shift-5 table, its Gaussian loss weight, AdamW at
    /// 1e-4) at rank 32 and alpha 32 on `q,k,v,o,ffn.0,ffn.2`, which in a Wan-Dancer transformer reaches both attentions
    /// and the feed-forward of every block and the music injectors' attentions. Introduced in InferKit 0.4.0.
    public static var diffSynthPreset: NFKMLXWanTrainingPreset { .diffSynth }

    /// Whether the recipe adapts the layer at `path`: a block's or a music injector's `q`, `k`, `v`, `o`, `ffn.0`,
    /// or `ffn.2`, as peft matches the module names' last components.
    static func adapts(_ path: String) -> Bool {
        let parts = path.split(separator: ".").map(String.init)
        guard parts.first == "blocks" || parts.first == "music_injector" else { return false }
        let tail = parts.count >= 2 ? parts.suffix(2).joined(separator: ".") : path
        return ["q", "k", "v", "o"].contains(parts.last!) || ["ffn.0", "ffn.2"].contains(tail)
    }

    /// Builds one of a release's transformers (`global_model.safetensors` or `local_model.safetensors`), ready to
    /// fine-tune, its weights at `precision` (the stored bfloat16 by default). Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, stage: String = "local",
                               precision: NFKMLXWeightPrecision = .checkpoint) throws -> NFKMLXWanDancerNet {
        let net = NFKMLXWanDancerNet(try NFKMLXWanDancerConfiguration.reading(directoryURL.appendingPathComponent("config.json")))
        let file = stage == "global" ? stageFiles.global : stageFiles.local
        try NFKMLXWanDancerNet.loadWeights(into: net, file: directoryURL.appendingPathComponent(file), precision: precision)
        return net
    }

    /// Encodes one clip as DiffSynth-Studio's units prepare it for training: the frames through the autoencoder to
    /// the standardized mean, the keyframes and their mask through the condition, the first keyframe and the
    /// reference image through CLIP, the prompt through umT5 at 512 tokens, and the clip's music, which the encoder
    /// reads in one span.
    ///
    /// - Parameters:
    ///   - video: the frames `[4k + 1, H, W, 3]` in `[0, 1]`, sides multiples of 16.
    ///   - prompt: the clip's caption, its frame-rate suffix included.
    ///   - clip: the keyframes, mask, reference image, music features, and frame rate.
    ///
    /// Introduced in InferKit 0.4.0.
    public func trainingExample(video: MLXArray, prompt: String, clip: NFKMLXWanDancerClip) throws -> NFKMLXWanDancerTrainingExample {
        guard video.dim(0) % 4 == 1, video.dim(1) % 16 == 0, video.dim(2) % 16 == 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "a training clip is 4k + 1 frames at sides that are multiples of 16, not \(video.shape)")
        }
        let text = try features(prompt)
        return try withPipeline(global: clip.fps != Self.framesPerSecond) { pipeline in
            var inputs = pipeline.inputs(clip)
            inputs.musicEncoderSpans = 1
            return NFKMLXWanDancerTrainingExample(latent: pipeline.encoded(video * 2 - 1).asType(.float32), text: text,
                                                  inputs: inputs)
        }
    }

    /// The velocity `net` predicts for an example at `draw` with `noise` (DiffSynth-Studio's `model_fn_wan_video`).
    static func prediction(_ net: NFKMLXWanDancerNet, example: NFKMLXWanDancerTrainingExample, draw: NFKMLXWanTrainingDraw,
                           noise: MLXArray) -> MLXArray {
        let dtype = NFKReferenceRounding.parameterType(of: net)
        let noisy = (1 - draw.sigma) * example.latent + draw.sigma * noise
        return net(noisy.asType(dtype), text: example.text.asType(dtype), t: Double(draw.timestep), inputs: example.inputs)
    }

    /// The loss of one example: `weight · mean((prediction − (noise − clean))²)`.
    static func loss(_ net: NFKMLXWanDancerNet, example: NFKMLXWanDancerTrainingExample, draw: NFKMLXWanTrainingDraw,
                     noise: MLXArray) -> MLXArray {
        NFKMLXWanObjective.loss(prediction: prediction(net, example: example, draw: draw, noise: noise),
                                clean: example.latent, noise: noise, weight: draw.weight)
    }

    /// Fine-tunes the transformer with LoRA as `preset`'s trainer does, returning each step's loss.
    ///
    /// @discussion The customization path is four calls: ``trainingExample(video:prompt:clip:)`` for each clip,
    /// ``network(directoryURL:stage:precision:)`` to build, this to train, and ``saveAdapters(of:to:preset:)`` to
    /// write the adapters, which a generator applies with ``addAdapter(at:scale:global:)``. Each step draws its index
    /// and noise from `seed`. A step is multi-second; call it off the render thread. Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(_ net: NFKMLXWanDancerNet, examples: (Int) -> [NFKMLXWanDancerTrainingExample],
                                preset: NFKMLXWanTrainingPreset = diffSynthPreset, optimizer: Optimizer? = nil,
                                steps: Int, seed: UInt64 = 0, accumulationSteps: Int = 1,
                                learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
                                checkpoint: NFKMLXTrainingCheckpoint? = nil, observer: NFKMLXTrainer.Observer? = nil,
                                recomputesBlocksInBackward: Bool = true) throws -> [Float] {
        let recomputed = net.recomputesBlocksInBackward
        net.recomputesBlocksInBackward = recomputesBlocksInBackward
        defer { net.recomputesBlocksInBackward = recomputed }
        var key = MLXRandom.key(seed)
        var batch = [NFKMLXWanDancerTrainingExample]()
        var draws = [NFKMLXWanTrainingDraw]()
        let objective = preset.objective
        let dtype = NFKReferenceRounding.parameterType(of: net)
        return try NFKMLXBlockedAttention.training {
            try NFKMLXFineTune.run(
                net,
                freezing: {
                    try NFKMLXLoRA.apply(to: net, rank: preset.rank, alpha: preset.alpha) { path, _ in adapts(path) }
                },
                optimizer: optimizer,
                reference: {
                    NFKMLXReferenceOptimizers.adamW(learningRate: preset.learningRate, betas: preset.betas,
                                                    eps: preset.epsilon, weightDecay: preset.weightDecay)
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
                    return batch.enumerated().map { index, example in
                        MLXRandom.normal(example.latent.shape, key: noiseKeys[index])
                    }
                },
                loss: { net, noise in
                    MLX.stacked((0 ..< batch.count).map { index in
                        loss(net, example: batch[index], draw: draws[index], noise: noise[index])
                    }).mean()
                },
                clipGradientNorm: preset.clipGradientNorm, accumulationSteps: accumulationSteps,
                learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, seed: seed, observer: observer)
        }
    }

    /// Writes the adapters `net` carries in DiffSynth-Studio's layout: the release's layer names, `lora_A` and
    /// `lora_B` under `.default`, the up projection carrying the scale. Introduced in InferKit 0.4.0.
    public static func saveAdapters(of net: NFKMLXWanDancerNet, to url: URL,
                                    preset: NFKMLXWanTrainingPreset = diffSynthPreset) throws {
        guard preset.format == .diffSynth else {
            throw NFKMLXError.unsupportedConfiguration("Wan-Dancer adapters are written in DiffSynth-Studio's layout")
        }
        var arrays = [String: MLXArray]()
        for (path, layer) in net.leafModules().flattened() {
            guard let adapter = layer as? NFKMLXLoRALinear else { continue }
            arrays["\(path).lora_A.default.weight"] = adapter.loraA.transposed(1, 0)
            arrays["\(path).lora_B.default.weight"] = adapter.loraB.transposed(1, 0) * adapter.scale
        }
        try MLX.save(arrays: arrays, url: url)
    }
}
