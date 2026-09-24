//
//  NFKMLXWanVideoGenerator.swift
//  InferKitMLX
//
//  Wan text-to-video from a downloaded diffusers release directory, its two stages held as an
//  `NFKMLXResidency` says: umT5-XXL runs once per clip and the transformer and autoencoder after it.
//  The file also carries the release loaders the three models lacked: the transformer's and the
//  autoencoder's geometry from their configs, and their weights in the module's layout.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// Reading a diffusers Wan release's transformer and autoencoder.
enum NFKMLXWanRelease {

    static func json(_ url: URL) throws -> [String: Any] {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not an object")
        }
        return json
    }

    /// The DiT geometry a `transformer/config.json` describes. An image-conditioned release (an
    /// `image_dim` or `added_kv_proj_dim`) is refused: this is the text-to-video transformer.
    static func transformerConfiguration(fromHuggingFace url: URL) throws -> NFKMLXWanConfiguration {
        let json = try json(url)
        if !(json["image_dim"] is NSNull || json["image_dim"] == nil)
            || !(json["added_kv_proj_dim"] is NSNull || json["added_kv_proj_dim"] == nil) {
            throw NFKMLXError.unsupportedConfiguration(
                "the transformer conditions on an image, which the text-to-video port does not implement")
        }
        let base = NFKMLXWanConfiguration.base
        return NFKMLXWanConfiguration(
            inChannels: json["in_channels"] as? Int ?? base.inChannels,
            heads: json["num_attention_heads"] as? Int ?? base.heads,
            headDim: json["attention_head_dim"] as? Int ?? base.headDim,
            layers: json["num_layers"] as? Int ?? base.layers,
            ffnDim: json["ffn_dim"] as? Int ?? base.ffnDim,
            textDim: json["text_dim"] as? Int ?? base.textDim,
            freqDim: json["freq_dim"] as? Int ?? base.freqDim,
            patchSize: json["patch_size"] as? [Int] ?? base.patchSize,
            eps: (json["eps"] as? NSNumber)?.floatValue ?? base.eps)
    }

    /// Loads a release's `transformer/` weights. The module keys are the release's; the patchify
    /// `Conv3d` weight is stored `[out, in, t, h, w]` and loads as MLX's `[out, t, h, w, in]`.
    static func loadTransformer(into net: NFKMLXWanTransformerNet, fromDirectory directory: URL,
                                precision: NFKMLXWeightPrecision) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision)
        try NFKMLXWeights.apply(arrays.map { key, value in
            (key, value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value)
        }, to: net, verifyShapes: true)
    }

    /// The autoencoder geometry a `vae/config.json` describes, with its latent statistics.
    ///
    /// @discussion Wan 2.2's `in_channels` counts the patchified channels (12 for RGB at patch 2), where
    /// this configuration counts the image's own; Wan 2.1 states neither.
    static func vaeConfiguration(fromHuggingFace url: URL) throws
        -> (configuration: NFKMLXWanVAEConfiguration, mean: MLXArray?, std: MLXArray?) {
        let json = try json(url)
        let patch = json["patch_size"] as? Int ?? 1
        let residual = (json["is_residual"] as? NSNumber)?.boolValue ?? false
        let baseDim = json["base_dim"] as? Int ?? 96
        let configuration = NFKMLXWanVAEConfiguration(
            baseDim: baseDim, decoderBaseDim: json["decoder_base_dim"] as? Int ?? baseDim,
            zDim: json["z_dim"] as? Int ?? 16, dimMult: json["dim_mult"] as? [Int] ?? [1, 2, 4, 4],
            numResBlocks: json["num_res_blocks"] as? Int ?? 2,
            temporalDownsample: json["temperal_downsample"] as? [Bool] ?? [false, true, true],
            patchSize: patch, inChannels: (json["in_channels"] as? Int).map { $0 / (patch * patch) } ?? 3,
            isResidual: residual)
        let mean = (json["latents_mean"] as? [Double]).map { MLXArray($0.map(Float.init)) }
        let std = (json["latents_std"] as? [Double]).map { MLXArray($0.map(Float.init)) }
        return (configuration, mean, std)
    }

    /// Loads a release's `vae/` weights: a 5-D causal convolution loads as `[out, t, h, w, in]`, a 4-D
    /// spatial one as `[out, h, w, in]`, and an RMS scale stored with trailing singleton axes flattens.
    static func loadVAE(into net: NFKMLXWanVideoVAENet, fromDirectory directory: URL,
                        precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision)
        try NFKMLXWeights.apply(arrays.map(adaptedVAE), to: net, verifyShapes: true)
    }

    static func adaptedVAE(key: String, value: MLXArray) -> (String, MLXArray) {
        if key.hasSuffix(".gamma") { return (key, value.reshaped([-1])) }
        if value.ndim == 5 { return (key, value.transposed(0, 2, 3, 4, 1)) }
        if value.ndim == 4 { return (key, value.transposed(0, 2, 3, 1)) }
        return (key, value)
    }

    /// The flow shift a `scheduler/scheduler_config.json` states: 3 for Wan 2.1 at 480p, 5 for Wan 2.2.
    static func schedule(fromHuggingFace url: URL) throws -> NFKMLXUniPCConfiguration {
        let json = try json(url)
        return NFKMLXUniPCConfiguration(flowShift: (json["flow_shift"] as? NSNumber)?.floatValue ?? 5,
                                        solverOrder: json["solver_order"] as? Int ?? 2,
                                        trainTimesteps: json["num_train_timesteps"] as? Int ?? 1000)
    }
}

/// The text stage: the umT5 encoder.
final class NFKWanTextStage {
    let encoder: NFKMLXT5EncoderNet
    init(_ encoder: NFKMLXT5EncoderNet) { self.encoder = encoder }

    /// The features the transformer reads: the masked encode, cut at the prompt's length and padded
    /// back with zeros, which is how the reference hands the padding to the transformer.
    func features(_ prompt: NFKMLXT5Prompt) -> MLXArray {
        let encoded = encoder(prompt.tokens, mask: prompt.maskArray)[0]
        let length = prompt.mask.reduce(0, +)
        let zeros = MLXArray.zeros([prompt.ids.count - length, encoded.dim(-1)], dtype: encoded.dtype)
        return concatenated([encoded[0 ..< length], zeros], axis: 0)
    }
}

/// Wan text-to-video, assembled from a diffusers release directory.
///
/// @discussion The release layout is `text_encoder/` (umT5-XXL), `tokenizer/` (its SentencePiece model),
/// `transformer/`, `vae/`, and `scheduler/`. The factories read the text-to-video releases: Wan 2.1 T2V
/// (1.3B and 14B, the Wan 2.1 autoencoder) and Wan 2.2 TI2V-5B (the Wan 2.2 autoencoder).
/// ``video(forPrompt:negativePrompt:frames:width:height:seed:)`` encodes the prompt through the masked
/// umT5, denoises over the UniPC flow schedule with classifier-free guidance, and decodes. The glue is
/// measured against diffusers' `WanPipeline`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXWanVideoGenerator)
public final class NFKMLXWanVideoGenerator: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "wan"

    /// The Hugging Face repository the download factories read when none is named: Wan 2.1 T2V 1.3B,
    /// the size that runs on a 32 GB machine.
    @objc public static let releaseRepo = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    /// The caption length the pipeline pads and truncates to.
    static let maximumSequenceLength = 512

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKWanTextStage>
    private let pipelineStage: NFKMLXStage<NFKMLXWanPipeline>
    private let segmenter: NFKMLXSentencePieceSegmenter
    private let compression: (spatial: Int, temporal: Int)
    private let sequenceLength: Int

    /// The denoising steps, 50 by the reference's default.
    @objc public var steps: Int = 50
    /// The classifier-free guidance scale, 5 by the reference's default. Above 1 the clip guides against
    /// the negative prompt, or an empty one where none is given.
    @objc public var guidance: Float = 5

    /// Whether the text encoder and the transformer stay loaded together between clips.
    @objc public var holdsStagesResident: Bool { staging.resident }

    /// Whether umT5 runs at float32 rather than at bfloat16. It does where the stored float32 encoder
    /// fits the working set on its own.
    @objc public internal(set) var encodesInFloat32 = true

    init(resident: Bool, segmenter: NFKMLXSentencePieceSegmenter, compression: (spatial: Int, temporal: Int),
         sequenceLength: Int = NFKMLXWanVideoGenerator.maximumSequenceLength,
         loadTextEncoder: @escaping () throws -> NFKMLXT5EncoderNet,
         loadPipeline: @escaping () throws -> NFKMLXWanPipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage { NFKWanTextStage(try loadTextEncoder()) }
        pipelineStage = NFKMLXStage(loadPipeline)
        self.segmenter = segmenter
        self.compression = compression
        self.sequenceLength = sequenceLength
        super.init()
    }

    /// Loads both stages now, for a resident release.
    func loadResident() throws {
        try staging.use(textStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    var isHoldingTextEncoder: Bool { staging.exclusively { textStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    /// The autoencoder's compression: `2^downsamplings · patch` spatially and `2^temporal downsamplings`
    /// in time.
    static func compression(of vae: NFKMLXWanVAEConfiguration) -> (spatial: Int, temporal: Int) {
        ((1 << vae.temporalDownsample.count) * vae.patchSize, 1 << vae.temporalDownsample.filter { $0 }.count)
    }

    // MARK: Factories

    /// Assembles the model from a downloaded release directory, holding it as
    /// ``NFKMLXResidency/automatic`` decides.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXWanVideoGenerator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    ///
    /// @discussion Wan 2.1 T2V 1.3B stores umT5-XXL at float32 (22.7 GB) beside a 5.7 GB transformer.
    /// The encoder runs at float32 where that fits the working set on its own and at bfloat16, the
    /// reference pipeline's precision, otherwise; the transformer and the autoencoder load as stored.
    /// The 14B transformer alone is beyond a 32 GB machine in any placement. A machine that cannot hold the stages together stages under
    /// ``NFKMLXResidency/automatic``. The release has no routed experts, so ``NFKMLXResidency/paged``
    /// holds it as ``NFKMLXResidency/staged`` does.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXWanVideoGenerator {
        let textDirectory = directoryURL.appendingPathComponent("text_encoder")
        let transformerDirectory = directoryURL.appendingPathComponent("transformer")
        let vaeDirectory = directoryURL.appendingPathComponent("vae")
        let segmenter = NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: directoryURL.appendingPathComponent("tokenizer/spiece.model")))
        let textConfiguration = try NFKMLXT5Encoder.configuration(
            fromHuggingFace: textDirectory.appendingPathComponent("config.json"))
        let transformerConfiguration = try NFKMLXWanRelease.transformerConfiguration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json"))
        let vae = try NFKMLXWanRelease.vaeConfiguration(fromHuggingFace: vaeDirectory.appendingPathComponent("config.json"))
        let schedule = try NFKMLXWanRelease.schedule(
            fromHuggingFace: directoryURL.appendingPathComponent("scheduler/scheduler_config.json"))

        // umT5-XXL is stored at float32 (22.7 GB). It runs at float32 where that is known to fit the
        // working set on its own, and at the bfloat16 the reference pipeline runs it at otherwise.
        let budget = NFKMLXResidencyBudget.current()
        let storedText = try NFKMLXStageWeights.bytes(inDirectory: textDirectory, precision: .checkpoint)
        let textInFloat32 = NFKMLXResidencyBudget.holds(storedText, budget: budget)
        let plan = try NFKMLXResidencyBudget.plan(
            [NFKMLXStageFootprint(bytes: textInFloat32 ? storedText : storedText / 2),
             NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(inDirectory: transformerDirectory, precision: .checkpoint)
                                     + NFKMLXStageWeights.bytes(inDirectory: vaeDirectory, precision: .float32))],
            residency: residency, budget: budget)
        let generator = NFKMLXWanVideoGenerator(
            resident: plan.holdsStagesResident, segmenter: segmenter,
            compression: compression(of: vae.configuration),
            loadTextEncoder: {
                let encoder = NFKMLXT5Encoder.makeNet(textConfiguration)
                try NFKMLXT5Encoder.loadWeights(into: encoder, from: textDirectory,
                                                dtype: textInFloat32 ? nil : .bfloat16)
                return encoder
            },
            loadPipeline: {
                let transformer = NFKMLXWanTransformerNet(transformerConfiguration)
                try NFKMLXWanRelease.loadTransformer(into: transformer, fromDirectory: transformerDirectory,
                                                     precision: .checkpoint)
                let autoencoder = NFKMLXWanVideoVAENet(vae.configuration)
                try NFKMLXWanRelease.loadVAE(into: autoencoder, fromDirectory: vaeDirectory)
                return NFKMLXWanPipeline(transformer: transformer, vae: autoencoder, latentsMean: vae.mean,
                                         latentsStd: vae.std, schedule: schedule)
            })
        generator.encodesInFloat32 = textInFloat32
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    /// The component folders a download fetches.
    static let releaseComponents = [
        NFKMLXReleaseComponent(required: ["model_index.json", "scheduler/scheduler_config.json"]),
        NFKMLXReleaseComponent(required: ["text_encoder/config.json"],
                               weights: ["text_encoder/model.safetensors", "text_encoder/model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["tokenizer/spiece.model"],
                               optional: ["tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json"]),
        NFKMLXReleaseComponent(required: ["transformer/config.json"],
                               weights: ["transformer/diffusion_pytorch_model.safetensors",
                                         "transformer/diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]),
    ]

    /// Downloads a release (``releaseRepo`` when `repo` is nil, or `Wan-AI/Wan2.2-TI2V-5B-Diffusers`)
    /// and assembles the model, holding it as `residency` says.
    ///
    /// @discussion Wan 2.1 T2V 1.3B is about 29 GB. A file already in the cache is not fetched again. The
    /// call blocks on the network, so run it off the main and render threads.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXWanVideoGenerator {
        let directory = try NFKMLXReleaseDownload.directory(repo: repo ?? releaseRepo, revision: revision,
                                                            cacheDirectoryURL: cacheDirectoryURL,
                                                            components: releaseComponents)
        return try generator(directoryURL: directory, residency: residency)
    }

    /// The asynchronous form of ``generator(repo:revision:cacheDirectoryURL:residency:)``. The handler
    /// runs on a background queue.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency,
                                 completionHandler: @escaping (NFKMLXWanVideoGenerator?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try generator(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                                residency: residency), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    // MARK: Generation

    /// The prompt's whitespace collapsed and trimmed, the reference's `prompt_clean`. The reference also
    /// runs `ftfy.fix_text` and unescapes HTML entities, which change only mis-decoded or escaped text.
    static func cleaned(_ prompt: String) -> String {
        prompt.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// The prompt as the encoder reads it.
    func prompt(_ text: String) -> NFKMLXT5Prompt {
        NFKMLXT5Prompt(Self.cleaned(text), segmenter: segmenter, length: sequenceLength)
    }

    /// Generates a clip for `prompt`, `[frames, height, width, 3]` in `[0, 1]`.
    ///
    /// @discussion `frames` rounds down to one more than a multiple of the autoencoder's temporal
    /// compression (4), and `width` and `height` down to multiples of its spatial compression times the
    /// transformer's patch (16 for Wan 2.1, 32 for Wan 2.2). Both prompts encode from one load of umT5.
    public func video(forPrompt prompt: String, negativePrompt: String? = nil, frames: Int = 81,
                      width: Int = 832, height: Int = 480, seed: UInt64 = 0) throws -> MLXArray {
        let tile = compression.spatial * 2
        let grid = ((frames - 1) / compression.temporal + 1, height / tile * 2, width / tile * 2)
        guard frames >= 1, grid.1 > 0, grid.2 > 0 else {
            throw NFKMLXError.unsupportedConfiguration("a clip is at least one frame of \(tile)×\(tile) pixels")
        }
        let guides = guidance > 1
        let prompts = [self.prompt(prompt)] + (guides ? [self.prompt(negativePrompt ?? "")] : [])
        return try staging.exclusively {
            let features = try staging.with(textStage) { stage in prompts.map { stage.features($0) } }
            return try staging.with(pipelineStage) { pipeline in
                let latent = pipeline.denoise(textEmbeds: features[0], negativeEmbeds: guides ? features[1] : nil,
                                              frames: grid.0, height: grid.1, width: grid.2, steps: steps,
                                              guidance: guidance, seed: seed)
                return clip(pipeline.decode(latent)[0] / 2 + 0.5, min: 0, max: 1)
            }
        }
    }

    /// Generates a clip for `prompt` and returns its frames as `CGImage`s. Objective-C reads each element
    /// as `(__bridge CGImageRef)`, since an array of a Core Foundation type is not an Objective-C
    /// collection.
    @objc(framesForPrompt:negativePrompt:frames:width:height:seed:error:)
    public func frames(forPrompt prompt: String, negativePrompt: String?, frames: Int, width: Int, height: Int,
                       seed: UInt64) throws -> [Any] {
        let clip = try video(forPrompt: prompt, negativePrompt: negativePrompt, frames: frames, width: width,
                             height: height, seed: seed)
        return try (0 ..< clip.dim(0)).map {
            try NFKMLXImageBridge.cgImage(from: clip[$0], options: NFKMLXImageOptions())
        }
    }
}
