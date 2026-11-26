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
    /// packed base (QLoRA); 0 keeps the stored type. Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, expert: NFKMLXWanExpert = .highNoise,
                               quantizationBits: Int = 0) throws -> NFKMLXWanTransformerNet {
        let directory = NFKMLXWanFunRelease.hasExperts(directoryURL)
            ? directoryURL.appendingPathComponent(expert == .highNoise ? "high_noise_model" : "low_noise_model")
            : directoryURL
        let net = NFKMLXWanTransformerNet(try NFKMLXWanOriginalRelease.transformerConfiguration(
            fromConfigURL: directory.appendingPathComponent("config.json")))
        if quantizationBits != 0 {
            net.layerStream = try NFKMLXWanBlockedLoad.load(
                into: net, blocks: net.blocks, fromDirectory: directory, precision: .checkpoint, bits: quantizationBits,
                rename: NFKMLXWanOriginalRelease.transformerModuleKey(forRelease:),
                adapt: NFKMLXWanOriginalRelease.adaptedTransformer)
        } else {
            try NFKMLXWanOriginalRelease.loadTransformer(into: net, fromDirectory: directory, precision: .checkpoint)
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
                                        reference: prepared.reference, camera: prepared.camera)
    }

    /// Fine-tunes one transformer with LoRA as `preset`'s trainer does, returning each step's loss: the Wan
    /// recipe (``NFKMLXWanVideoGenerator/fineTune(_:examples:preset:optimizer:steps:seed:accumulationSteps:learningRateSchedule:checkpoint:observer:recomputesBlocksInBackward:)``)
    /// over examples ``trainingExample(video:prompt:conditions:)`` builds, the transformer's original forward
    /// reading their conditions.
    ///
    /// The customization path is four calls: ``trainingExample(video:prompt:conditions:)`` for each clip,
    /// ``network(directoryURL:expert:quantizationBits:)`` to build, this to train, and
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
