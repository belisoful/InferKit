//
//  NFKMLXLTXVideoGenerator.swift
//  InferKitMLX
//
//  LTX-Video 0.9.0 text-to-video from a downloaded diffusers release directory, its two stages held as
//  an `NFKMLXResidency` says. T5-XXL runs once per clip and the transformer and autoencoder after it,
//  so a release too large to hold whole runs when the two take turns.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// A T5 prompt as the diffusers pipelines hand it to the encoder: SentencePiece pieces, the end token,
/// padding to a fixed length, and the mask that marks the real tokens.
struct NFKMLXT5Prompt {
    let ids: [Int]
    let mask: [Int]

    /// Tokenizes `prompt`, truncating to `length − 1` pieces so the end token always fits, and pads.
    init(_ prompt: String, segmenter: NFKMLXSentencePieceSegmenter, length: Int, endToken: Int = 1,
         padToken: Int = 0) {
        let pieces = Array(segmenter.encode(prompt, dummyPrefix: true).prefix(length - 1)) + [endToken]
        ids = pieces + [Int](repeating: padToken, count: length - pieces.count)
        mask = [Int](repeating: 1, count: pieces.count) + [Int](repeating: 0, count: length - pieces.count)
    }

    var tokens: MLXArray { MLXArray(ids.map(Int32.init)).reshaped([1, ids.count]) }
    var maskArray: MLXArray { MLXArray(mask.map(Int32.init)).reshaped([1, mask.count]) }
}

/// The text stage: the T5 encoder and the tokenizer that feeds it.
final class NFKLTXTextStage {
    let encoder: NFKMLXT5EncoderNet
    init(_ encoder: NFKMLXT5EncoderNet) { self.encoder = encoder }
}

/// LTX-Video text-to-video, assembled from a diffusers release directory.
///
/// @discussion The release layout is `text_encoder/` (T5-XXL), `tokenizer/` (its SentencePiece model),
/// `transformer/`, `vae/`, and `scheduler/`. ``video(forPrompt:negativePrompt:frames:width:height:seed:)``
/// pads the prompt to 128 tokens, masks the padding out of the transformer's cross-attention, denoises
/// over the flow schedule with classifier-free guidance, and decodes. The glue is measured against
/// diffusers' `LTXPipeline`. The factories read LTX-Video 0.9.0; a 0.9.1 or later autoencoder conditions
/// its decode on a timestep this port does not implement, and is refused.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXLTXVideoGenerator)
public final class NFKMLXLTXVideoGenerator: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "ltx-video-0.9.0"

    /// The Hugging Face repository the download factories read.
    @objc public static let releaseRepo = "Lightricks/LTX-Video"

    /// The caption length the pipeline pads and truncates to.
    static let maximumSequenceLength = 128

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKLTXTextStage>
    private let pipelineStage: NFKMLXStage<NFKMLXLTXPipeline>
    private let segmenter: NFKMLXSentencePieceSegmenter

    /// The denoising steps, 50 by the reference's default.
    @objc public var steps: Int = 50
    /// The classifier-free guidance scale, 3 by the reference's default. Above 1 the clip guides against
    /// the negative prompt, or an empty one where none is given.
    @objc public var guidance: Float = 3
    /// The frame rate the rotary scales time by, 25 by the reference's default.
    @objc public var frameRate: Float = 25

    /// Whether the text encoder and the transformer stay loaded together between clips. False where the
    /// release is staged: each clip then loads T5, encodes, releases it, and loads the transformer and
    /// the autoencoder.
    @objc public var holdsStagesResident: Bool { staging.resident }

    init(resident: Bool, segmenter: NFKMLXSentencePieceSegmenter,
         loadTextEncoder: @escaping () throws -> NFKMLXT5EncoderNet,
         loadPipeline: @escaping () throws -> NFKMLXLTXPipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage { NFKLTXTextStage(try loadTextEncoder()) }
        pipelineStage = NFKMLXStage(loadPipeline)
        self.segmenter = segmenter
        super.init()
    }

    /// Loads both stages now, for a resident release.
    func loadResident() throws {
        try staging.use(textStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    var isHoldingTextEncoder: Bool { staging.exclusively { textStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    // MARK: Factories

    /// Assembles the model from a downloaded release directory, holding it as
    /// ``NFKMLXResidency/automatic`` decides.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXLTXVideoGenerator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    ///
    /// @discussion The release stores T5-XXL and the transformer at float32, about 19 GB and 7.7 GB, and
    /// both load as stored. A machine that cannot hold them together stages under
    /// ``NFKMLXResidency/automatic``. The release has no routed experts, so ``NFKMLXResidency/paged``
    /// holds it as ``NFKMLXResidency/staged`` does.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXLTXVideoGenerator {
        let textDirectory = directoryURL.appendingPathComponent("text_encoder")
        let transformerDirectory = directoryURL.appendingPathComponent("transformer")
        let vaeDirectory = directoryURL.appendingPathComponent("vae")
        let segmenter = NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: directoryURL.appendingPathComponent("tokenizer/spiece.model")))
        let textConfiguration = try NFKMLXT5Encoder.configuration(
            fromHuggingFace: textDirectory.appendingPathComponent("config.json"))
        let transformerConfiguration = try self.transformerConfiguration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json"))
        let (vaeConfiguration, scalingFactor) = try self.vaeConfiguration(
            fromHuggingFace: vaeDirectory.appendingPathComponent("config.json"))

        let plan = try NFKMLXResidencyBudget.plan(
            [NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(inDirectory: textDirectory, precision: .float32)),
             NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(inDirectory: transformerDirectory, precision: .float32)
                                     + NFKMLXStageWeights.bytes(inDirectory: vaeDirectory, precision: .checkpoint))],
            residency: residency, budget: NFKMLXResidencyBudget.current())
        let generator = NFKMLXLTXVideoGenerator(
            resident: plan.holdsStagesResident, segmenter: segmenter,
            loadTextEncoder: {
                let encoder = NFKMLXT5Encoder.makeNet(textConfiguration)
                try NFKMLXT5Encoder.loadWeights(into: encoder, from: textDirectory)
                return encoder
            },
            loadPipeline: {
                let transformer = NFKMLXLTXTransformer.makeNet(transformerConfiguration)
                try NFKMLXLTXTransformer.loadWeights(into: transformer, from: transformerDirectory)
                let vae = NFKMLXLTXVideoVAE.makeNet(vaeConfiguration)
                try NFKMLXLTXVideoVAE.loadWeights(
                    into: vae, from: vaeDirectory.appendingPathComponent("diffusion_pytorch_model.safetensors"))
                return NFKMLXLTXPipeline(transformer: transformer, vae: vae, scalingFactor: scalingFactor)
            })
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

    /// Downloads the release (``releaseRepo`` when `repo` is nil) and assembles the model, holding it
    /// as `residency` says.
    ///
    /// @discussion The download is about 28 GB. A file already in the cache is not fetched again. The
    /// call blocks on the network, so run it off the main and render threads.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXLTXVideoGenerator {
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
                                 completionHandler: @escaping (NFKMLXLTXVideoGenerator?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try generator(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                                residency: residency), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    // MARK: Configuration

    private static func json(_ url: URL) throws -> [String: Any] {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not an object")
        }
        return json
    }

    /// The DiT geometry a `transformer/config.json` describes.
    static func transformerConfiguration(fromHuggingFace url: URL) throws -> NFKMLXLTXTransformerConfiguration {
        let json = try json(url)
        let base = NFKMLXLTXTransformerConfiguration.base
        guard (json["patch_size"] as? Int ?? 1) == 1, (json["patch_size_t"] as? Int ?? 1) == 1 else {
            throw NFKMLXError.unsupportedConfiguration("the transformer patchifies its latent, which this port does not")
        }
        return NFKMLXLTXTransformerConfiguration(
            inChannels: json["in_channels"] as? Int ?? base.inChannels,
            heads: json["num_attention_heads"] as? Int ?? base.heads,
            headDim: json["attention_head_dim"] as? Int ?? base.headDim,
            layers: json["num_layers"] as? Int ?? base.layers,
            crossAttentionDim: json["cross_attention_dim"] as? Int ?? base.crossAttentionDim,
            captionChannels: json["caption_channels"] as? Int ?? base.captionChannels)
    }

    /// The autoencoder geometry a `vae/config.json` describes, and its `scaling_factor`.
    static func vaeConfiguration(fromHuggingFace url: URL) throws -> (NFKMLXLTXVAEConfiguration, Float) {
        let json = try json(url)
        if (json["timestep_conditioning"] as? Bool) == true {
            throw NFKMLXError.unsupportedConfiguration(
                "the autoencoder conditions its decode on a timestep (LTX-Video 0.9.1 and later), which this port "
                + "does not implement; use the 0.9.0 release")
        }
        let base = NFKMLXLTXVAEConfiguration.base
        let configuration = NFKMLXLTXVAEConfiguration(
            inChannels: json["in_channels"] as? Int ?? base.inChannels,
            latentChannels: json["latent_channels"] as? Int ?? base.latentChannels,
            blockOutChannels: json["block_out_channels"] as? [Int] ?? base.blockOutChannels,
            layersPerBlock: json["layers_per_block"] as? [Int] ?? base.layersPerBlock,
            spatioTemporalScaling: json["spatio_temporal_scaling"] as? [Bool] ?? base.spatioTemporalScaling,
            patchSize: json["patch_size"] as? Int ?? base.patchSize,
            patchSizeT: json["patch_size_t"] as? Int ?? base.patchSizeT)
        return (configuration, (json["scaling_factor"] as? NSNumber)?.floatValue ?? 1)
    }

    // MARK: Generation

    /// The prompt as the encoder reads it: ids padded to 128 and the mask of the real tokens.
    func prompt(_ text: String) -> NFKMLXT5Prompt {
        NFKMLXT5Prompt(text, segmenter: segmenter, length: Self.maximumSequenceLength)
    }

    /// Generates a clip for `prompt`, `[frames, height, width, 3]` in `[0, 1]`.
    ///
    /// @discussion `frames` rounds down to one more than a multiple of 8 and `width` and `height` down to
    /// multiples of 32, the autoencoder's compression, as the reference rounds them. Both prompts encode
    /// from one load of T5.
    public func video(forPrompt prompt: String, negativePrompt: String? = nil, frames: Int = 121,
                      width: Int = 704, height: Int = 480, seed: UInt64 = 0) throws -> MLXArray {
        let spatial = NFKMLXLTXPipeline.spatialCompression, temporal = NFKMLXLTXPipeline.temporalCompression
        let grid = ((frames - 1) / temporal + 1, height / spatial, width / spatial)
        guard frames >= 1, grid.1 > 0, grid.2 > 0 else {
            throw NFKMLXError.unsupportedConfiguration("a clip is at least one frame of 32×32 pixels")
        }
        let guides = guidance > 1
        let prompts = [self.prompt(prompt)] + (guides ? [self.prompt(negativePrompt ?? "")] : [])
        return try staging.exclusively {
            let features = try staging.with(textStage) { stage in prompts.map { stage.encoder($0.tokens) } }
            return try staging.with(pipelineStage) { pipeline in
                let latents = pipeline.denoise(
                    text: features[0], textMask: prompts[0].maskArray,
                    negativeText: guides ? features[1] : nil, negativeMask: guides ? prompts[1].maskArray : nil,
                    frames: grid.0, height: grid.1, width: grid.2, steps: steps, guidance: guidance, seed: seed,
                    frameRate: frameRate)
                let video = pipeline.decode(latents, frames: grid.0, height: grid.1, width: grid.2)
                return clip(video[0] / 2 + 0.5, min: 0, max: 1)
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
