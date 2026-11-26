//
//  NFKMLXWanS2VTraining.swift
//  InferKitMLX
//
//  LoRA fine-tuning of Wan 2.2 S2V-14B: the network built from a release directory, a training example built by
//  the generator's own encoders, and DiffSynth-Studio's recipe (`lora/Wan2.2-S2V-14B.sh`).
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXOptimizers

/// One S2V training clip as DiffSynth-Studio's S2V units leave it.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanS2VTrainingExample {
    /// The normalized latent mean `[16, 1 + F', h, w]` of the clip's `4F' + 1` frames. Its first frame, the clip's
    /// first picture, is the reference image the transformer reads; the loss leaves it out.
    public var latent: MLXArray
    /// The text features `[512, textDim]`, zeros past the prompt.
    public var text: MLXArray
    /// The audio buckets of the clip's `4F'` generated frames, `[layers, audioDim, 4F']`.
    public var audio: MLXArray
    /// The pose latents `[16, F', h, w]`, or nil for zeros.
    public var pose: MLXArray?

    public init(latent: MLXArray, text: MLXArray, audio: MLXArray, pose: MLXArray? = nil) {
        self.latent = latent
        self.text = text
        self.audio = audio
        self.pose = pose
    }
}

extension NFKMLXWanS2VGenerator {

    /// DiffSynth-Studio's S2V LoRA recipe (`lora/Wan2.2-S2V-14B.sh` at 1e8ae1d): ``NFKMLXWanTrainingPreset/diffSynth``
    /// (an index uniform over the shift-5 table, its Gaussian loss weight, AdamW at 1e-4), rank 32 at alpha 32 on
    /// `q,k,v,o,ffn.0,ffn.2`, which in the S2V transformer reaches both attentions and the feed-forward of every
    /// block and the audio injector's attentions. Introduced in InferKit 0.4.0.
    public static var diffSynthPreset: NFKMLXWanTrainingPreset { .diffSynth }

    /// Whether a recipe adapts the S2V transformer's layer at `path`: the block layers `targets` names, and where
    /// it names the attention projections, those of the audio injector.
    static func adapts(_ path: String, targets: NFKMLXWanLoRATargets) -> Bool {
        if targets.contains(path) {
            return true
        }
        let parts = path.split(separator: ".")
        guard parts.count == 4, parts[0] == "audio_injector", parts[1] == "injector" else { return false }
        switch targets {
        case .attention, .attentionAndFeedForward: return ["q", "k", "v", "o"].contains(parts[3])
        case .queryKeyValueAndFeedForward: return ["q", "k", "v"].contains(parts[3])
        }
    }

    /// Builds the release's transformer, ready to fine-tune. A `quantizationBits` of 2 to 8 packs the blocks'
    /// linear layers as each loads, so the adapters train over a packed base (QLoRA); 0 keeps the weights at
    /// `precision`, the stored bfloat16 by default. Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, quantizationBits: Int = 0,
                               precision: NFKMLXWeightPrecision = .checkpoint) throws -> NFKMLXWanS2VNet {
        let net = NFKMLXWanS2VNet(try NFKMLXWanS2VConfiguration.reading(directoryURL.appendingPathComponent("config.json")))
        if quantizationBits != 0 {
            net.layerStream = try NFKMLXWanBlockedLoad.load(
                into: net, blocks: net.blocks, fromDirectory: directoryURL, precision: precision, bits: quantizationBits,
                rename: NFKMLXWanS2VNet.moduleKey(forRelease:), adapt: NFKMLXWanS2VNet.adapted)
        } else {
            try NFKMLXWanS2VNet.loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
        }
        return net
    }

    /// Encodes one clip as DiffSynth-Studio's S2V units prepare it for training: the frames through the
    /// autoencoder to the normalized mean (whose first frame is the reference image's latent), the prompt through
    /// umT5 at 512 tokens, the speech's first segment of buckets for the clip's `frames − 1` generated frames, and
    /// a pose video's latents where one is given.
    ///
    /// - Parameters:
    ///   - video: the frames `[4k + 1, H, W, 3]` in `[0, 1]`, sides multiples of 16.
    ///   - text: the clip's caption.
    ///   - audio: the clip's speech, mono 16 kHz samples.
    ///   - pose: the pose video's frames `[4k, H, W, 3]` in `[0, 1]` at the clip's size, or nil.
    ///
    /// Introduced in InferKit 0.4.0.
    public func trainingExample(video: MLXArray, prompt text: String, audio: [Float],
                                pose: MLXArray? = nil) throws -> NFKMLXWanS2VTrainingExample {
        let frames = video.dim(0)
        guard frames % 4 == 1, frames > 1, video.dim(1) % 16 == 0, video.dim(2) % 16 == 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "a training clip is 4k + 1 frames at sides that are multiples of 16, not \(video.shape)")
        }
        let features = try encodedFeatures([text])[0]
        let saved = framesPerSegment
        framesPerSegment = frames - 1
        defer { framesPerSegment = saved }
        let buckets = try encodedAudio(audio).segment(0)
        let latent = try withPipeline { pipeline -> (MLXArray, MLXArray?) in
            let encoded = pipeline.encoded(video * 2 - 1)
            let poses = pose.map {
                pipeline.poseLatents($0, segments: 1, frames: frames - 1, latentShape: [])[0]
            }
            return (encoded, poses)
        }
        return NFKMLXWanS2VTrainingExample(latent: latent.0, text: features, audio: buckets, pose: latent.1)
    }

    /// The velocity `net` predicts for an example's generated frames at `draw` with `noise`: the noised latent
    /// with its first frame kept clean (`first_frame_latents`), read as `model_fn_wans2v` reads it (the first frame
    /// as the reference, the blank motion clip dropped, the example's audio and pose).
    static func prediction(_ net: NFKMLXWanS2VNet, example: NFKMLXWanS2VTrainingExample, draw: NFKMLXWanTrainingDraw,
                           noise: MLXArray) -> MLXArray {
        let dtype = NFKReferenceRounding.parameterType(of: net)
        let x0 = example.latent
        let noisy = ((1 - draw.sigma) * x0 + draw.sigma * noise)[0..., 1...]
        let audio = net.audioTokens(example.audio.asType(dtype), motionFrames: 73, motionLatentFrames: 19)
        let inputs = NFKWanS2VInputs(reference: x0[0..., 0 ..< 1].asType(dtype), motion: nil,
                                     pose: (example.pose ?? MLXArray.zeros(noisy.shape)).asType(dtype), audio: audio)
        return net(noisy.asType(dtype), text: example.text.asType(dtype), t: Double(draw.timestep), inputs: inputs)
    }

    /// The loss of one example: `weight · mean((prediction − (noise − clean))²)` over the generated frames.
    static func loss(_ net: NFKMLXWanS2VNet, example: NFKMLXWanS2VTrainingExample, draw: NFKMLXWanTrainingDraw,
                     noise: MLXArray) -> MLXArray {
        let predicted = prediction(net, example: example, draw: draw, noise: noise)
        return NFKMLXWanObjective.loss(prediction: predicted, clean: example.latent[0..., 1...], noise: noise[0..., 1...],
                                       weight: draw.weight)
    }

    /// Fine-tunes the transformer with LoRA as `preset`'s trainer does, returning each step's loss.
    ///
    /// @discussion The customization path is four calls: ``trainingExample(video:prompt:audio:pose:)`` for each
    /// clip, ``network(directoryURL:quantizationBits:precision:)`` to build, this to train, and
    /// ``saveAdapters(of:to:preset:)`` to write the adapters, which a generator applies with
    /// ``addAdapter(at:scale:)``. Each step draws its index and noise from `seed`. The trainer runs bfloat16
    /// weights under autocast; this runs in the type the weights load at. A step is multi-second; call it off the
    /// render thread. Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(_ net: NFKMLXWanS2VNet, examples: (Int) -> [NFKMLXWanS2VTrainingExample],
                                preset: NFKMLXWanTrainingPreset = diffSynthPreset, optimizer: Optimizer? = nil,
                                steps: Int, seed: UInt64 = 0, accumulationSteps: Int = 1,
                                learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
                                checkpoint: NFKMLXTrainingCheckpoint? = nil, observer: NFKMLXTrainer.Observer? = nil,
                                recomputesBlocksInBackward: Bool = true) throws -> [Float] {
        let recomputed = net.recomputesBlocksInBackward
        net.recomputesBlocksInBackward = recomputesBlocksInBackward
        defer { net.recomputesBlocksInBackward = recomputed }
        var key = MLXRandom.key(seed)
        var batch = [NFKMLXWanS2VTrainingExample]()
        var draws = [NFKMLXWanTrainingDraw]()
        let objective = preset.objective
        let dtype = NFKReferenceRounding.parameterType(of: net)
        return try NFKMLXBlockedAttention.training {
            try NFKMLXFineTune.run(
                net,
                freezing: {
                    try NFKMLXLoRA.apply(to: net, rank: preset.rank, alpha: preset.alpha) { path, _ in
                        adapts(path, targets: preset.targets)
                    }
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
    public static func saveAdapters(of net: NFKMLXWanS2VNet, to url: URL,
                                    preset: NFKMLXWanTrainingPreset = diffSynthPreset) throws {
        guard preset.format == .diffSynth else {
            throw NFKMLXError.unsupportedConfiguration("Wan 2.2 S2V adapters are written in DiffSynth-Studio's layout")
        }
        var arrays = [String: MLXArray]()
        for (path, layer) in net.leafModules().flattened() {
            guard let adapter = layer as? NFKMLXLoRALinear else { continue }
            let name = path.hasPrefix("audio_injector.") ? path : try NFKMLXWanVideoGenerator.originalName(path)
            arrays["\(name).lora_A.default.weight"] = adapter.loraA.transposed(1, 0)
            arrays["\(name).lora_B.default.weight"] = adapter.loraB.transposed(1, 0) * adapter.scale
        }
        try MLX.save(arrays: arrays, url: url)
    }
}
