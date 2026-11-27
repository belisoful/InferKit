//
//  NFKMLXWanFunTraining.swift
//  InferKitMLX
//
//  LoRA fine-tuning of the Wan 2.2 Fun releases: the network built from a release directory, a training
//  example built by the generator's own encoders, and the trainers' presets.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXOptimizers

extension NFKMLXWanFunGenerator {

    /// The recipe of the trainer this package holds each release to: DiffSynth-Studio's for an A14B expert
    /// (``diffSynthPreset(expert:)``), VideoX-Fun's for a 5B release
    /// (``NFKMLXWanTrainingPreset/videoXFun(expert:)``), which DiffSynth-Studio does not train. Introduced in
    /// InferKit 0.4.0.
    public static func referencePreset(variant: NFKMLXWanFunVariant, expert: NFKMLXWanExpert) -> NFKMLXWanTrainingPreset {
        if variant.isVACE {
            return vaceDiffSynthPreset(expert: expert)
        }
        return variant.hasExperts ? diffSynthPreset(expert: expert) : .videoXFun()
    }

    /// DiffSynth-Studio's LoRA recipe for one expert of Wan 2.2 VACE-Fun A14B (`lora/Wan2.2-VACE-Fun-A14B.sh` at
    /// 1e8ae1d): ``diffSynthPreset(expert:)`` (the index from the expert's side of 0.358, AdamW at 1e-4, rank 32) on
    /// the VACE blocks alone (`--lora_base_model vace`, ``NFKMLXWanLoRATargets/vaceAttentionAndFeedForward``).
    /// Introduced in InferKit 0.4.0.
    public static func vaceDiffSynthPreset(expert: NFKMLXWanExpert) -> NFKMLXWanTrainingPreset {
        var preset = diffSynthPreset(expert: expert)
        preset.targets = .vaceAttentionAndFeedForward
        return preset
    }

    /// DiffSynth-Studio's LoRA recipe for one expert of a Fun A14B release (`lora/Wan2.2-Fun-A14B-Control.sh`,
    /// `-InP.sh`, and `-Control-Camera.sh` at 1e8ae1d): ``NFKMLXWanTrainingPreset/diffSynth`` with the index drawn
    /// from the expert's side of the boundary 0.358 (timestep 900), rank 32 on `q,k,v,o,ffn.0,ffn.2`, AdamW at
    /// 1e-4. Introduced in InferKit 0.4.0.
    public static func diffSynthPreset(expert: NFKMLXWanExpert) -> NFKMLXWanTrainingPreset {
        NFKMLXWanTrainingPreset.diffSynth(expert: expert, imageToVideo: true)
    }

    /// Builds one transformer of a release directory, ready to fine-tune: a 5B release's only one, or an
    /// A14B release's `high_noise_model/` (``NFKMLXWanExpert/highNoise``) or `low_noise_model/`. A
    /// `quantizationBits` of 2 to 8 packs the blocks' linear layers as each loads, so the adapters train over a
    /// packed base (QLoRA); 0 keeps the weights at `precision`, the stored bfloat16 by default. Introduced in
    /// InferKit 0.4.0.
    public static func network(directoryURL: URL, expert: NFKMLXWanExpert = .highNoise, quantizationBits: Int = 0,
                               precision: NFKMLXWeightPrecision = .checkpoint) throws -> NFKMLXWanTransformerNet {
        let directory = NFKMLXWanFunRelease.hasExperts(directoryURL)
            ? directoryURL.appendingPathComponent(expert == .highNoise ? "high_noise_model" : "low_noise_model")
            : directoryURL
        let net = NFKMLXWanTransformerNet(try NFKMLXWanOriginalRelease.transformerConfiguration(
            fromConfigURL: directory.appendingPathComponent("config.json")))
        if quantizationBits != 0 {
            net.layerStream = try NFKMLXWanBlockedLoad.load(
                into: net, blocks: net.blocks, fromDirectory: directory, precision: precision, bits: quantizationBits,
                rename: NFKMLXWanOriginalRelease.transformerModuleKey(forRelease:),
                adapt: NFKMLXWanOriginalRelease.adaptedTransformer)
        } else {
            try NFKMLXWanOriginalRelease.loadTransformer(into: net, fromDirectory: directory, precision: precision)
        }
        return net
    }

    /// Encodes one clip as DiffSynth-Studio's Fun units prepare it for training: the frames through the
    /// autoencoder to the normalized mean, the prompt through umT5 at 512 tokens (zeros past the prompt), and the
    /// conditions as the sampler prepares them (``NFKMLXWanFunConditions``: the control video, the start image
    /// under its mask, the reference image, the camera), at the clip's size.
    ///
    /// - Parameters:
    ///   - video: the frames `[4k + 1, H, W, 3]` in `[0, 1]`, at a size the generator's tile divides.
    ///   - text: the clip's caption, read as written.
    ///   - conditions: the clip's conditions, each at the clip's size.
    ///
    /// Introduced in InferKit 0.4.0.
    public func trainingExample(video: MLXArray, prompt text: String,
                                conditions: NFKMLXWanFunConditions = NFKMLXWanFunConditions()) throws -> NFKMLXWanTrainingExample {
        let frames = video.dim(0), height = video.dim(1), width = video.dim(2)
        let size = try clipSize(frames: frames, width: width, height: height)
        guard size.frames == frames, size.width == width, size.height == height else {
            throw NFKMLXError.unsupportedConfiguration(
                "a training clip is 4k + 1 frames at sides the tile divides, not \(video.shape)")
        }
        let features = try encodedFeatures([text])[0]
        let (latent, prepared) = try withPipeline { pipeline in
            (pipeline.encoded(video * 2 - 1),
             try pipeline.prepare(conditions, frames: frames, height: height, width: width))
        }
        return NFKMLXWanTrainingExample(latent: latent, text: features, condition: prepared.channels,
                                        reference: prepared.reference, camera: prepared.camera,
                                        latentMask: prepared.latentMask)
    }

    /// Encodes one clip as DiffSynth-Studio's VACE units prepare it for training: the frames through the autoencoder
    /// to the normalized mean, led by each subject image's latent (`WanVideoUnit_InputVideoEmbedder` with
    /// `vace_reference_image`), the prompt through umT5 at 512 tokens, and the VACE context the sampler prepares
    /// from `conditions` (`WanVideoUnit_VACE`), at the clip's size.
    ///
    /// - Parameters:
    ///   - video: the frames `[4k + 1, H, W, 3]` in `[0, 1]`, at a size the generator's tile divides.
    ///   - text: the clip's caption, read as written.
    ///   - conditions: the clip's control video (or video and mask) and subject images, at the clip's size.
    ///
    /// Introduced in InferKit 0.4.0.
    public func trainingExample(video: MLXArray, prompt text: String,
                                vace conditions: NFKMLXWanVACEConditions) throws -> NFKMLXWanTrainingExample {
        let frames = video.dim(0), height = video.dim(1), width = video.dim(2)
        let size = try clipSize(frames: frames, width: width, height: height)
        guard size.frames == frames, size.width == width, size.height == height else {
            throw NFKMLXError.unsupportedConfiguration(
                "a training clip is 4k + 1 frames at sides the tile divides, not \(video.shape)")
        }
        let features = try encodedFeatures([text])[0]
        let (latent, context) = try withPipeline { pipeline -> (MLXArray, MLXArray) in
            let references = conditions.subjectImages.map { pipeline.encoded(($0 * 2 - 1).expandedDimensions(axis: 0)) }
            let latent = concatenated(references + [pipeline.encoded(video * 2 - 1)], axis: 1)
            let prepared = try pipeline.prepareVACE(conditions, frames: frames, height: height, width: width).prepared
            return (latent, prepared.vaceContext!)
        }
        return NFKMLXWanTrainingExample(latent: latent, text: features, vaceContext: context)
    }

    /// Fine-tunes one transformer with LoRA as `preset`'s trainer does, returning each step's loss: the Wan
    /// recipe (``NFKMLXWanVideoGenerator/fineTune(_:examples:preset:optimizer:steps:seed:accumulationSteps:learningRateSchedule:checkpoint:observer:recomputesBlocksInBackward:)``)
    /// over examples ``trainingExample(video:prompt:conditions:)`` builds, the transformer's original forward
    /// reading their conditions.
    ///
    /// The customization path is four calls: ``trainingExample(video:prompt:conditions:)`` for each clip,
    /// ``network(directoryURL:expert:quantizationBits:precision:)`` to build, this to train, and
    /// ``saveAdapters(of:to:preset:)`` to write the adapters, which a generator applies with
    /// ``addAdapter(at:scale:expert:)``. Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(_ net: NFKMLXWanTransformerNet, examples: (Int) -> [NFKMLXWanTrainingExample],
                                preset: NFKMLXWanTrainingPreset, optimizer: Optimizer? = nil, steps: Int,
                                seed: UInt64 = 0, accumulationSteps: Int = 1,
                                learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
                                checkpoint: NFKMLXTrainingCheckpoint? = nil, observer: NFKMLXTrainer.Observer? = nil,
                                recomputesBlocksInBackward: Bool = true) throws -> [Float] {
        try NFKMLXWanVideoGenerator.fineTune(net, examples: examples, preset: preset, optimizer: optimizer, steps: steps,
                                             seed: seed, accumulationSteps: accumulationSteps,
                                             learningRateSchedule: learningRateSchedule, checkpoint: checkpoint,
                                             observer: observer, recomputesBlocksInBackward: recomputesBlocksInBackward)
    }

    /// Writes the adapters `net` carries in `preset`'s trainer's layout. Introduced in InferKit 0.4.0.
    public static func saveAdapters(of net: NFKMLXWanTransformerNet, to url: URL, preset: NFKMLXWanTrainingPreset) throws {
        try NFKMLXWanVideoGenerator.saveAdapters(of: net, to: url, preset: preset)
    }
}
