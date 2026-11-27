//
//  NFKMLXWanFunGenerator.swift
//  InferKitMLX
//
//  The Wan 2.2 Fun releases (alibaba-pai, Apache-2.0) end to end from their original-layout directories:
//  umT5 from its `.pth`, the transformer (or the two A14B experts), and the autoencoder, staged as an
//  `NFKMLXResidency` says.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// The Wan 2.2 Fun releases the factories read.
@objc(NFKMLXWanFunVariant)
public enum NFKMLXWanFunVariant: Int, CaseIterable, Sendable {
    /// `alibaba-pai/Wan2.2-Fun-5B-Control`: a control video (pose, depth, edges) and an optional reference
    /// image over Wan 2.2 TI2V-5B's geometry and autoencoder.
    case control5B = 0
    /// `alibaba-pai/Wan2.2-Fun-5B-InP`: a start and an end image, or a masked video, over the 5B geometry.
    case inpaint5B = 1
    /// `alibaba-pai/Wan2.2-Fun-5B-Control-Camera`: a camera trajectory over the 5B geometry.
    case camera5B = 2
    /// `alibaba-pai/Wan2.2-Fun-A14B-Control`: two 14B experts over the Wan 2.1 autoencoder.
    case controlA14B = 3
    /// `alibaba-pai/Wan2.2-Fun-A14B-InP`.
    case inpaintA14B = 4
    /// `alibaba-pai/Wan2.2-Fun-A14B-Control-Camera`.
    case cameraA14B = 5
    /// `alibaba-pai/Wan2.2-VACE-Fun-A14B`: two 14B experts with VACE blocks, which read a control video or a
    /// video kept outside a mask, and subject reference images (``NFKMLXWanVACEConditions``).
    case vaceA14B = 6

    /// The registry name.
    public var modelName: String {
        ["wan2.2-fun-5b-control", "wan2.2-fun-5b-inp", "wan2.2-fun-5b-control-camera",
         "wan2.2-fun-a14b-control", "wan2.2-fun-a14b-inp", "wan2.2-fun-a14b-control-camera",
         "wan2.2-vace-fun-a14b"][rawValue]
    }

    /// The Hugging Face repository the download factories read.
    public var releaseRepo: String {
        ["alibaba-pai/Wan2.2-Fun-5B-Control", "alibaba-pai/Wan2.2-Fun-5B-InP", "alibaba-pai/Wan2.2-Fun-5B-Control-Camera",
         "alibaba-pai/Wan2.2-Fun-A14B-Control", "alibaba-pai/Wan2.2-Fun-A14B-InP",
         "alibaba-pai/Wan2.2-Fun-A14B-Control-Camera", "alibaba-pai/Wan2.2-VACE-Fun-A14B"][rawValue]
    }

    /// Whether the release holds two 14B experts.
    public var hasExperts: Bool { rawValue >= 3 }

    /// Whether the release is VACE-Fun, conditioned through its VACE blocks.
    public var isVACE: Bool { self == .vaceA14B }

    /// The conditions a Fun release reads beside the noise: 0 a control video, 1 an inpainting video, 2 a camera
    /// trajectory; nil for VACE-Fun.
    var conditionKind: Int? { isVACE ? nil : rawValue % 3 }

    /// The flow-match schedule the release samples over: Euler at a shift of 5, or 12 for VACE-Fun.
    var schedule: NFKMLXFlowMatchConfiguration { isVACE ? .wanVACEFun : .wan }

    /// The fraction of the train timesteps at and above which an A14B release's high-noise expert runs: 0.900,
    /// or 0.875 for VACE-Fun.
    var boundary: Float { isVACE ? NFKMLXWanFunPipeline.vaceFunBoundary : 0.9 }

    /// The component files a download fetches.
    var releaseComponents: [NFKMLXReleaseComponent] {
        let shared = [
            NFKMLXReleaseComponent(required: [NFKMLXWanOriginalRelease.textEncoderFile]),
            NFKMLXReleaseComponent(required: ["google/umt5-xxl/spiece.model"],
                                   optional: ["google/umt5-xxl/tokenizer.json", "google/umt5-xxl/tokenizer_config.json",
                                              "google/umt5-xxl/special_tokens_map.json"]),
        ]
        guard hasExperts else {
            return shared + [
                NFKMLXReleaseComponent(required: ["config.json", NFKMLXWanOriginalRelease.wan22AutoencoderFile],
                                       weights: ["diffusion_pytorch_model.safetensors"]),
            ]
        }
        return shared + [NFKMLXReleaseComponent(required: [NFKMLXWanOriginalRelease.wan21AutoencoderFile])]
            + ["high_noise_model", "low_noise_model"].map {
                NFKMLXReleaseComponent(required: ["\($0)/config.json"], weights: ["\($0)/diffusion_pytorch_model.safetensors"])
            }
    }
}

/// A Wan 2.2 Fun release, assembled from its directory.
///
/// @discussion The release layout is the original Wan one (`NFKMLXWanOriginalRelease`): umT5 as
/// `models_t5_umt5-xxl-enc-bf16.pth` with its tokenizer under `google/umt5-xxl/`, the autoencoder as
/// `Wan2.2_VAE.pth` (5B) or `Wan2.1_VAE.pth` (A14B), and the transformer as `config.json` and
/// `diffusion_pytorch_model.safetensors`, or `high_noise_model/` and `low_noise_model/` for an A14B release.
/// ``video(forPrompt:negativePrompt:conditions:frames:width:height:seed:)`` samples as VideoX-Fun's
/// `Wan2_2FunControlPipeline` and `Wan2_2FunInpaintPipeline` do (``NFKMLXWanFunPipeline``), with the
/// defaults of VideoX-Fun's `examples/wan2.2_fun/predict_*.py`: Euler at a flow shift of 5, 50 steps,
/// guidance 6, and for an A14B release the high-noise expert at and above timestep 900.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXWanFunGenerator)
public final class NFKMLXWanFunGenerator: NSObject {

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKMLXT5EncoderNet>
    /// The autoencoder, and a 5B release's transformer.
    private let pipelineStage: NFKMLXStage<NFKMLXWanFunPipeline>
    /// An A14B release's experts, high-noise first; empty for a 5B release.
    private let expertStages: [NFKMLXStage<NFKMLXWanTransformerNet>]
    private let segmenter: NFKMLXSentencePieceSegmenter
    /// Each transformer's adapters: a 5B release's one, or an A14B release's high-noise then low-noise.
    private let adapters: [NFKWanAdapterList]
    private let mergesOnLoad: Bool

    /// The release the generator was read from.
    @objc public let variant: NFKMLXWanFunVariant

    /// The denoising steps, 50 by VideoX-Fun's examples.
    @objc public var steps: Int = 50
    /// The classifier-free guidance scale, 6 by VideoX-Fun's examples. Above 1 the clip guides against the
    /// negative prompt, or an empty one where none is given.
    @objc public var guidance: Float = 6

    /// The frame rate the release is trained at: 24 for the 5B releases, 16 for A14B.
    @objc public var framesPerSecond: Double { variant.hasExperts ? 16 : 24 }

    /// The bits the transformers' block weights are packed to, or 0.
    @objc public internal(set) var quantizationBits = 0

    /// How many transformer blocks are read from the release on every pass, across the experts.
    @objc public internal(set) var streamedLayerCount = 0

    /// Whether the text encoder and the transformers stay loaded together between clips.
    @objc public var holdsStagesResident: Bool { staging.resident }

    init(variant: NFKMLXWanFunVariant, resident: Bool, segmenter: NFKMLXSentencePieceSegmenter,
         adapters: [NFKWanAdapterList], mergesOnLoad: Bool,
         loadTextEncoder: @escaping () throws -> NFKMLXT5EncoderNet,
         loadPipeline: @escaping () throws -> NFKMLXWanFunPipeline,
         loadExperts: [() throws -> NFKMLXWanTransformerNet] = []) {
        self.variant = variant
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage(loadTextEncoder)
        pipelineStage = NFKMLXStage(loadPipeline)
        expertStages = loadExperts.map { NFKMLXStage($0) }
        self.segmenter = segmenter
        self.adapters = adapters
        self.mergesOnLoad = mergesOnLoad
        super.init()
    }

    func loadResident() throws {
        try staging.use(textStage) { _ in }
        NFKMLXGPU.clearCache()
        try staging.use(pipelineStage) { _ in }
        for stage in expertStages {
            try staging.use(stage) { _ in }
        }
    }

    /// The text features of `prompts` from one load of umT5.
    func encodedFeatures(_ prompts: [String]) throws -> [MLXArray] {
        try staging.exclusively {
            try staging.with(textStage) { encoder in
                prompts.map { NFKMLXWanFunRelease.features($0, encoder: encoder, segmenter: segmenter) }
            }
        }
    }

    /// Runs `body` with the pipeline stage (the autoencoder, and a 5B release's transformer).
    func withPipeline<T>(_ body: (NFKMLXWanFunPipeline) throws -> T) throws -> T {
        try staging.exclusively { try staging.use(pipelineStage, body) }
    }

    var isHoldingTextEncoder: Bool { staging.exclusively { textStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }
    func isHoldingExpert(_ expert: NFKMLXWanExpert) -> Bool {
        staging.exclusively { expertStages.indices.contains(expert.rawValue) && expertStages[expert.rawValue].isHeld }
    }

    /// The variant a release directory holds, read from its transformer configuration and its autoencoder.
    static func variant(ofDirectory directory: URL) throws -> NFKMLXWanFunVariant {
        let experts = NFKMLXWanFunRelease.hasExperts(directory)
        let configuration = try NFKMLXWanOriginalRelease.transformerConfiguration(
            fromConfigURL: NFKMLXWanFunRelease.transformerDirectories(directory)[0].appendingPathComponent("config.json"))
        if configuration.vaceLayers != nil {
            guard experts else {
                throw NFKMLXError.unsupportedConfiguration("a Wan 2.2 VACE-Fun release holds two experts")
            }
            return .vaceA14B
        }
        let latent = experts ? 16 : 48
        let kind: Int
        if configuration.controlAdapterChannels != nil {
            kind = 2
        } else if configuration.inChannels == 3 * latent + 4 {
            kind = 0
        } else if configuration.inChannels == 2 * latent + 4 {
            kind = 1
        } else {
            throw NFKMLXError.unsupportedConfiguration(
                "a Wan 2.2 Fun transformer reading \(configuration.inChannels) channels is no release this reads")
        }
        return NFKMLXWanFunVariant(rawValue: kind + (experts ? 3 : 0))!
    }

    // MARK: Factories

    /// Assembles the model from a downloaded release directory, holding it as ``NFKMLXResidency/automatic``
    /// decides.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXWanFunGenerator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    ///
    /// @discussion umT5-XXL is stored at bfloat16 (11.4 GB) and runs at float32 where the plan affords it.
    /// A 5B transformer is 10 GB at bfloat16 and shares its stage with the autoencoder; an A14B release's two
    /// 14B experts (28.6 GB each at bfloat16) form that stage together, so a 32 GB machine packs them
    /// (``generator(directoryURL:residency:precision:quantizationBits:)``) or streams their blocks. The
    /// release has no routed experts, so ``NFKMLXResidency/paged`` holds it as ``NFKMLXResidency/staged``
    /// does.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXWanFunGenerator {
        try generator(directoryURL: directoryURL, residency: residency, precision: .checkpoint, bits: nil)
    }

    /// Assembles the model holding it as `residency` says, every stage's weights at `precision`, and the
    /// transformers' block weights packed to `quantizationBits` (2 to 8; 0 packs nothing) as each block loads.
    @objc(generatorWithDirectoryURL:residency:precision:quantizationBits:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                                 quantizationBits: Int) throws -> NFKMLXWanFunGenerator {
        try generator(directoryURL: directoryURL, residency: residency, precision: precision,
                      bits: quantizationBits == 0 ? nil : quantizationBits)
    }

    static func generator(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                          bits: Int?, groupSize: Int = NFKMLXLoadOptions.defaultQuantizationGroupSize,
                          budget: Int? = nil, streamedBlocks: Set<Int>? = nil) throws -> NFKMLXWanFunGenerator {
        let variant = try variant(ofDirectory: directoryURL)
        let segmenter = try NFKMLXWanFunRelease.segmenter(directoryURL)
        let textURL = directoryURL.appendingPathComponent(NFKMLXWanOriginalRelease.textEncoderFile)
        let vaeURL = try NFKMLXWanFunRelease.autoencoderURL(directoryURL)
        // High-noise first, the order the schedule reaches them in.
        let directories = Array(NFKMLXWanFunRelease.transformerDirectories(directoryURL).reversed())
        let configurations = try directories.map {
            try NFKMLXWanOriginalRelease.transformerConfiguration(fromConfigURL: $0.appendingPathComponent("config.json"))
        }

        let textBytes = try fileBytes(textURL)
        let text = NFKMLXStageFootprint(bytes: textBytes, widenedBytes: 2 * textBytes)
        let vaeBytes = try fileBytes(vaeURL)
        let transformers = try directories.map { directory -> NFKMLXStageFootprint in
            if let bits {
                return NFKMLXStageFootprint(bytes: try NFKMLXWanBlockedLoad.quantizedFootprint(
                    directory: directory, precision: precision, bits: bits, groupSize: groupSize).bytes)
            }
            return try NFKMLXWanBlockedLoad.footprint(directory: directory, precision: precision)
        }
        // A 5B release's transformer shares its stage with the autoencoder; an A14B release's experts are
        // stages of their own, after the autoencoder's.
        let stages = variant.hasExperts
            ? [text, NFKMLXStageFootprint(bytes: vaeBytes)] + transformers
            : [text, NFKMLXStageFootprint(bytes: transformers[0].bytes + vaeBytes, layerBytes: transformers[0].layerBytes)]
        let plan = try NFKMLXResidencyBudget.plan(stages, residency: residency,
                                                  budget: budget ?? NFKMLXResidencyBudget.current())
        let firstTransformerStage = variant.hasExperts ? 2 : 1
        let streamed = directories.indices.map { index -> Set<Int> in
            if let streamedBlocks { return streamedBlocks }
            let stage = firstTransformerStage + index
            guard plan.streams(stage) else { return [] }
            return Set(plan.heldLayers(stage, of: stages[stage]).upperBound ..< transformers[index].layerBytes.count)
        }
        let textInFloat32 = plan.widens(0)
        let adapters = directories.map { _ in NFKWanAdapterList() }
        func loadTransformer(_ index: Int) throws -> NFKMLXWanTransformerNet {
            let net = NFKMLXWanTransformerNet(configurations[index])
            let deltas = try adapters[index].deltas(naming: NFKMLXWanLoRA.textToVideoPath(original:))
            if bits != nil || !streamed[index].isEmpty {
                net.layerStream = try NFKMLXWanBlockedLoad.load(
                    into: net, blocks: net.blocks, fromDirectory: directories[index], precision: precision,
                    streamed: streamed[index], bits: bits, groupSize: groupSize, adapters: deltas,
                    rename: NFKMLXWanOriginalRelease.transformerModuleKey(forRelease:),
                    adapt: NFKMLXWanOriginalRelease.adaptedTransformer)
                return net
            }
            try NFKMLXWanOriginalRelease.loadTransformer(into: net, fromDirectory: directories[index], precision: precision)
            for delta in deltas {
                try NFKMLXWanLoRA.apply(delta, to: net)
            }
            return net
        }
        let generator = NFKMLXWanFunGenerator(
            variant: variant, resident: plan.holdsStagesResident, segmenter: segmenter, adapters: adapters,
            mergesOnLoad: bits != nil || streamed.contains { !$0.isEmpty },
            loadTextEncoder: {
                try NFKMLXWanOriginalRelease.textEncoder(from: textURL, precision: textInFloat32 ? .float32 : .checkpoint)
            },
            loadPipeline: {
                // The autoencoder is stored at float32 and runs at it, as the reference runs it.
                let autoencoder = try NFKMLXWanOriginalRelease.autoencoder(from: vaeURL)
                return NFKMLXWanFunPipeline(transformer: variant.hasExperts ? nil : try loadTransformer(0),
                                            highNoiseTransformer: nil, configuration: configurations[0],
                                            vae: autoencoder.net, latentsMean: autoencoder.mean,
                                            latentsStd: autoencoder.std, boundary: variant.boundary,
                                            schedule: variant.schedule)
            },
            loadExperts: variant.hasExperts ? directories.indices.map { index in { try loadTransformer(index) } } : [])
        generator.quantizationBits = bits ?? 0
        if variant.isVACE {
            generator.guidance = 5
        }
        generator.streamedLayerCount = streamed.reduce(0) { $0 + $1.count }
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    private static func fileBytes(_ url: URL) throws -> Int {
        (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue ?? 0
    }

    /// Downloads `variant`'s release (or `repo`, where given) and assembles the model, holding it as
    /// `residency` says.
    ///
    /// @discussion A 5B release is about 24 GB and an A14B release about 70 GB. A file already in the cache
    /// is not fetched again. The call blocks on the network, so run it off the main and render threads.
    @objc(generatorWithVariant:repo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(variant: NFKMLXWanFunVariant, repo: String?, revision: String?,
                                 cacheDirectoryURL: URL?, residency: NFKMLXResidency) throws -> NFKMLXWanFunGenerator {
        try NFKMLXBuildHolds.keeping {
            let directory = try NFKMLXReleaseDownload.directory(repo: repo ?? variant.releaseRepo, revision: revision,
                                                                cacheDirectoryURL: cacheDirectoryURL,
                                                                components: variant.releaseComponents)
            let generator = try generator(directoryURL: directory, residency: residency)
            guard generator.variant == variant else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(repo ?? variant.releaseRepo) holds \(generator.variant.modelName), not \(variant.modelName)")
            }
            return generator
        }
    }

    /// The asynchronous form of ``generator(variant:repo:revision:cacheDirectoryURL:residency:)``. The handler
    /// runs on a background queue.
    @objc(generatorWithVariant:repo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func generator(variant: NFKMLXWanFunVariant, repo: String?, revision: String?,
                                 cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                                 completionHandler: @escaping (NFKMLXWanFunGenerator?, Error?) -> Void) {
        NFKMLXReleaseDownload.asyncValue(completionHandler) {
            try generator(variant: variant, repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                          residency: residency)
        }
    }

    // MARK: Adapters

    /// Merges a released LoRA (``NFKMLXWanLoRA``) into a transformer at `scale`, now where it is loaded and on
    /// every later load. An A14B release takes the adapter into `expert`; a 5B release into its one
    /// transformer, under ``NFKMLXWanExpert/highNoise``. A packed or streamed transformer merges an adapter only
    /// as it loads, so a loaded one is released and loads again with it.
    @objc(addAdapterAtURL:scale:expert:error:)
    public func addAdapter(at url: URL, scale: Float, expert: NFKMLXWanExpert) throws {
        let index = expertStages.isEmpty ? 0 : expert.rawValue
        try staging.exclusively {
            let deltas = try NFKMLXWanLoRA.deltas(adapterURL: url, naming: NFKMLXWanLoRA.textToVideoPath(original:),
                                                  scale: scale).deltas
            let stageHeld = expertStages.isEmpty ? pipelineStage.isHeld : expertStages[index].isHeld
            if stageHeld && mergesOnLoad {
                if expertStages.isEmpty {
                    pipelineStage.release()
                } else {
                    expertStages[index].release()
                }
                NFKMLXGPU.clearCache()
            } else if stageHeld {
                if expertStages.isEmpty {
                    _ = try staging.use(pipelineStage) { try NFKMLXWanLoRA.apply(deltas, to: $0.transformer!) }
                } else {
                    _ = try staging.use(expertStages[index]) { try NFKMLXWanLoRA.apply(deltas, to: $0) }
                }
            }
            adapters[index].add(url, scale: scale)
            if stageHeld && mergesOnLoad && staging.resident {
                try loadResident()
            }
        }
    }

    /// Removes every adapter. A loaded transformer is released and loads again without them.
    @objc public func removeAdapters() {
        staging.exclusively {
            adapters.forEach { $0.removeAll() }
            pipelineStage.release()
            expertStages.forEach { $0.release() }
            NFKMLXGPU.clearCache()
        }
    }

    /// The adapters ``addAdapter(at:scale:expert:)`` added, across the transformers.
    @objc public var adapterCount: Int { staging.exclusively { adapters.reduce(0) { $0 + $1.entries.count } } }

    // MARK: Generation

    /// Generates a clip for `prompt` under `conditions`, `[frames, height, width, 3]` in `[0, 1]`.
    ///
    /// @discussion `frames` rounds down to one more than a multiple of 4, and `width` and `height` down to
    /// multiples of the autoencoder's compression times the patch (32 for the 5B releases, 16 for A14B). Every
    /// video and image in `conditions` is at that size; ``NFKMLXWanFunConditions`` describes them. A Control
    /// release reads a control video (zeros without one) and, where it reads one, a reference image; a
    /// Control-Camera release requires a camera embedding; an InP release reads only the inpainting video and
    /// mask.
    public func video(forPrompt prompt: String, negativePrompt: String? = nil,
                      conditions: NFKMLXWanFunConditions = NFKMLXWanFunConditions(), frames: Int = 81,
                      width: Int = 832, height: Int = 480, seed: UInt64 = 0) throws -> MLXArray {
        try generate(prompt: prompt, negativePrompt: negativePrompt, conditions: conditions, frames: frames,
                     width: width, height: height, seed: seed).clip
    }

    /// Generates a Wan 2.2 VACE-Fun clip for `prompt` under `conditions`, `[frames, height, width, 3]` in `[0, 1]`.
    ///
    /// @discussion The release follows the control video, or keeps the video outside the mask, and reads each
    /// subject image as a frame of context ahead of the clip (``NFKMLXWanVACEConditions``), every picture at the
    /// clip's size. `frames` rounds down to one more than a multiple of 4, and `width` and `height` down to
    /// multiples of 16. Introduced in InferKit 0.4.0.
    public func video(forPrompt prompt: String, negativePrompt: String? = nil, vace conditions: NFKMLXWanVACEConditions,
                      frames: Int = 81, width: Int = 832, height: Int = 480, seed: UInt64 = 0) throws -> MLXArray {
        try generate(prompt: prompt, negativePrompt: negativePrompt, conditions: NFKMLXWanFunConditions(), vace: conditions,
                     frames: frames, width: width, height: height, seed: seed).clip
    }

    /// The clip's size after rounding: frames to `4k + 1`, sides to the tile.
    func clipSize(frames: Int, width: Int, height: Int) throws -> (frames: Int, width: Int, height: Int) {
        let tile = variant.hasExperts ? 16 : 32
        let size = ((max(frames, 1) - 1) / 4 * 4 + 1, width / tile * tile, height / tile * tile)
        guard size.1 > 0, size.2 > 0 else {
            throw NFKMLXError.unsupportedConfiguration("a clip is at least \(tile) × \(tile) pixels")
        }
        return size
    }

    func generate(prompt: String, negativePrompt: String?, conditions: NFKMLXWanFunConditions,
                  vace: NFKMLXWanVACEConditions? = nil, frames: Int, width: Int, height: Int, seed: UInt64,
                  noise: MLXArray? = nil, progress: ((Double) -> Void)? = nil) throws -> (clip: MLXArray, latent: MLXArray) {
        let size = try clipSize(frames: frames, width: width, height: height)
        if variant.conditionKind == 2, conditions.cameraEmbedding == nil {
            throw NFKMLXError.unsupportedConfiguration("a Control-Camera release needs a camera trajectory")
        }
        if variant.isVACE, vace == nil {
            throw NFKMLXError.unsupportedConfiguration("a VACE-Fun release reads NFKMLXWanVACEConditions")
        }
        let guides = guidance > 1
        let prompts = [prompt] + (guides ? [negativePrompt ?? ""] : [])
        return try staging.exclusively {
            let features = try staging.with(textStage) { encoder in
                prompts.map { NFKMLXWanFunRelease.features($0, encoder: encoder, segmenter: segmenter) }
            }
            NFKMLXGPU.clearCache()
            let (prepared, references, start, timesteps) = try staging.use(pipelineStage) { pipeline in
                let (prepared, references) = try vace.map {
                    try pipeline.prepareVACE($0, frames: size.frames, height: size.height, width: size.width)
                } ?? (try pipeline.prepare(conditions, frames: size.frames, height: size.height, width: size.width), 0)
                return (prepared, references,
                        noise ?? pipeline.noise(frames: size.frames, height: size.height, width: size.width, seed: seed,
                                                leadingFrames: references),
                        pipeline.timesteps(steps))
            }
            let stepCount = steps, guidance = self.guidance
            let report: (Int, MLXArray) -> Void = { index, _ in progress?(Double(index + 1) / Double(stepCount + 1)) }
            var latent = start
            if expertStages.isEmpty {
                latent = try staging.use(pipelineStage) { pipeline in
                    try pipeline.denoise(text: features[0], negative: guides ? features[1] : nil, prepared: prepared,
                                         noise: latent, steps: steps, guidance: guidance, onStep: report)
                }
            } else {
                // The schedule reaches the high-noise expert's steps first; each expert is held only for its own.
                let usesHigh = try staging.use(pipelineStage) { pipeline in timesteps.map(pipeline.usesHighNoiseExpert) }
                let high = usesHigh.prefix { $0 }.count
                for (stage, indices) in [(expertStages[0], 0 ..< high), (expertStages[1], high ..< steps)] where !indices.isEmpty {
                    let denoiser = try staging.use(pipelineStage) { $0 }
                    latent = try staging.use(stage) { net in
                        try denoiser.denoise(text: features[0], negative: guides ? features[1] : nil, prepared: prepared,
                                             noise: latent, steps: steps, guidance: guidance, indices: indices,
                                             net: { _ in net }, onStep: report)
                    }
                }
            }
            // A VACE-Fun clip's reference frames lead the latents and are dropped before the decode.
            let decoded = latent[0..., references...]
            return (try staging.use(pipelineStage) { $0.decode(decoded) }, decoded)
        }
    }
}
