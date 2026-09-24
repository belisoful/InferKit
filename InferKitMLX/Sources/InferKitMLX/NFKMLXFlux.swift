//
//  NFKMLXFlux.swift
//  InferKitMLX
//
//  The end-to-end FLUX.1 text-to-image path: a prompt string in, an image out. The denoising transformer
//  (`NFKMLXFluxTransformerNet`), the flow-match sampler, and the autoencoder (`NFKMLXSDAutoencoder` in its
//  `.flux` preset) are the pieces `NFKMLXFluxPipeline` already chains from pre-computed embeddings; this
//  facade adds the text front end — CLIP-L for the pooled projection and T5-XXL for the sequence
//  (`NFKMLXFluxTextEncoder`) — and the release-directory assembly, so a consumer goes from a diffusers
//  FLUX release and a prompt to a picture without wiring the two text encoders by hand.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

/// FLUX.1's text front end: CLIP-L for the pooled projection and T5-XXL for the conditioning sequence.
/// Split from the pipeline so it can be loaded and measured without the 24 GB transformer.
public final class NFKMLXFluxTextEncoder {
    private let clip: NFKMLXSDTextEncoderNet
    private let clipTokenizer: NFKMLXSDPromptTokenizer
    private let t5: NFKMLXT5Encoder
    private let t5Segmenter: NFKMLXSentencePieceSegmenter
    private let t5Context: Int
    private let t5EndToken: Int
    private let t5PadToken: Int

    init(clip: NFKMLXSDTextEncoderNet, clipTokenizer: NFKMLXSDPromptTokenizer, t5: NFKMLXT5Encoder,
         t5Segmenter: NFKMLXSentencePieceSegmenter, t5Context: Int = 256, t5EndToken: Int = 1,
         t5PadToken: Int = 0) {
        self.clip = clip
        self.clipTokenizer = clipTokenizer
        self.t5 = t5
        self.t5Segmenter = t5Segmenter
        self.t5Context = t5Context
        self.t5EndToken = t5EndToken
        self.t5PadToken = t5PadToken
    }

    /// Loads CLIP-L (`text_encoder/` + `tokenizer/`) and T5-XXL (`text_encoder_2/` + `tokenizer_2/`) from a
    /// diffusers FLUX release directory. schnell caps the T5 sequence at 256; [dev] at 512.
    public static func textEncoder(directoryURL: URL, t5Context: Int = 256) throws -> NFKMLXFluxTextEncoder {
        let clip = try NFKMLXSDTextEncoder.net(
            configuration: .stableDiffusion15,
            weightsURL: directoryURL.appendingPathComponent("text_encoder/model.safetensors"),
            precision: .checkpoint)
        let clipTokenizer = try NFKMLXSDPromptTokenizer(
            directoryURL: directoryURL.appendingPathComponent("tokenizer"))
        let t5 = try NFKMLXT5Encoder.encoder(configuration: .xxl,
                                             directory: directoryURL.appendingPathComponent("text_encoder_2"))
        let t5Model = try NFKMLXSentencePieceModel(
            contentsOf: directoryURL.appendingPathComponent("tokenizer_2/spiece.model"))
        return NFKMLXFluxTextEncoder(clip: clip, clipTokenizer: clipTokenizer, t5: t5,
                                     t5Segmenter: NFKMLXSentencePieceSegmenter(model: t5Model),
                                     t5Context: t5Context)
    }

    /// Encodes a prompt into the two conditioning tensors FLUX reads: the T5-XXL sequence
    /// `[t5Context, 4096]` and the CLIP-L pooled projection `[768]`.
    ///
    /// @discussion CLIP truncates to 77 tokens and the pooled embedding is the end-of-text token's hidden
    /// state after the final normalization, with no projection (FLUX uses `CLIPTextModel`, not the
    /// projection variant); a causal model, so padding past the end-of-text token does not change it. T5
    /// pads to `t5Context` with the pad token and encodes the whole padded sequence, matching diffusers,
    /// which passes no attention mask.
    public func encode(prompt: String) -> (promptEmbeds: MLXArray, pooled: MLXArray) {
        let clipIds = clipTokenizer.tokens(for: prompt, contextLength: 77)  // start + prompt + end + padding
        let clipEnd = clipIds.firstIndex(of: clipTokenizer.endTokenId) ?? (clipIds.count - 1)
        let clipHidden = clip.encode(clipIds).hidden                       // [1, 77, 768], lnFinal applied
        let pooled = clipHidden[0, clipEnd]                                // the end-of-text token's hidden state

        var t5Ids = t5Segmenter.encode(prompt, dummyPrefix: true) + [t5EndToken]
        if t5Ids.count > t5Context {
            t5Ids = Array(t5Ids.prefix(t5Context - 1)) + [t5EndToken]
        }
        while t5Ids.count < t5Context { t5Ids.append(t5PadToken) }
        let t5Tokens = MLXArray(t5Ids.map(Int32.init)).reshaped([1, t5Context])
        let embeds = t5.encode(t5Tokens)[0]                                // [t5Context, 4096]
        return (embeds, pooled)
    }
}

/// Building and running FLUX.1 text-to-image from a downloaded diffusers release directory.
///
/// The release layout is the diffusers one: `transformer/`, `vae/`, `text_encoder/` (CLIP-L),
/// `text_encoder_2/` (T5-XXL), `tokenizer/` (CLIP byte-level BPE), and `tokenizer_2/` (the T5
/// SentencePiece model). FLUX.1 [schnell] is the four-step distilled variant with no guidance embedding;
/// FLUX.1 [dev] is guidance-distilled and read the same way, its schedule set by the caller.
@objc(NFKMLXFlux)
public final class NFKMLXFlux: NSObject {

    /// A name for the model the factory produces.
    @objc public static let modelName = "flux.1-schnell"

    private let staging: NFKMLXStagedModel
    private let textEncoderStage: NFKMLXStage<NFKMLXFluxTextEncoder>
    private let pipelineStage: NFKMLXStage<NFKMLXFluxPipeline>

    /// The default number of denoising steps. schnell is distilled to four.
    @objc public var steps: Int = 4
    /// The guidance scale a guidance-distilled release ([dev]) reads; schnell ignores it.
    @objc public var guidance: Float = 3.5

    /// Whether the text encoders and the transformer stay loaded together between images. False where
    /// the release is staged: each image then loads CLIP-L and T5-XXL, encodes, releases them, and
    /// loads the transformer and the autoencoder. Introduced in InferKit 0.4.0.
    @objc public var holdsStagesResident: Bool { staging.resident }

    init(resident: Bool, loadTextEncoder: @escaping () throws -> NFKMLXFluxTextEncoder,
         loadPipeline: @escaping () throws -> NFKMLXFluxPipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textEncoderStage = NFKMLXStage(loadTextEncoder)
        pipelineStage = NFKMLXStage(loadPipeline)
        super.init()
    }

    /// A facade over components already loaded, held resident.
    convenience init(pipeline: NFKMLXFluxPipeline, textEncoder: NFKMLXFluxTextEncoder) {
        self.init(resident: true, loadTextEncoder: { textEncoder }, loadPipeline: { pipeline })
    }

    /// Loads both stages now, for a resident release: a bad release fails at construction rather than
    /// at the first image.
    func loadResident() throws {
        try staging.use(textEncoderStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    var isHoldingTextEncoder: Bool { staging.exclusively { textEncoderStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    /// Assembles the whole model from a downloaded diffusers FLUX release directory with the four-step
    /// schnell schedule, holding it as ``NFKMLXResidency/automatic`` decides.
    @objc(fluxWithDirectoryURL:error:)
    public static func flux(directoryURL: URL) throws -> NFKMLXFlux {
        try flux(directoryURL: directoryURL, schedule: .fluxSchnell, residency: .automatic)
    }

    /// ``flux(directoryURL:)`` holding the release as `residency` says. Introduced in InferKit 0.4.0.
    @objc(fluxWithDirectoryURL:residency:error:)
    public static func flux(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXFlux {
        try flux(directoryURL: directoryURL, schedule: .fluxSchnell, residency: residency)
    }

    /// The component folders ``flux(directoryURL:)`` reads, as a download fetches them.
    static let releaseComponents = [
        NFKMLXReleaseComponent(required: ["transformer/config.json"],
                               weights: ["transformer/diffusion_pytorch_model.safetensors",
                                         "transformer/diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["vae/diffusion_pytorch_model.safetensors"]),
        NFKMLXReleaseComponent(required: ["text_encoder/model.safetensors"]),
        NFKMLXReleaseComponent(required: [],
                               weights: ["text_encoder_2/model.safetensors",
                                         "text_encoder_2/model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["tokenizer/vocab.json", "tokenizer/merges.txt"],
                               optional: ["tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json"]),
        NFKMLXReleaseComponent(required: ["tokenizer_2/spiece.model"]),
    ]

    /// Downloads the release's files into the hub cache and returns the snapshot directory.
    static func releaseDirectory(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> URL {
        try NFKMLXReleaseDownload.directory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                            components: releaseComponents)
    }

    /// Downloads a diffusers FLUX.1 release and assembles the model with the four-step schnell schedule.
    ///
    /// @discussion The download is the transformer, the autoencoder, CLIP-L, T5-XXL, and both
    /// tokenizers, about 34 GB for `black-forest-labs/FLUX.1-schnell`. A file already in the cache is
    /// not fetched again. The call blocks on the network, so run it off the main and render threads.
    /// It serves `black-forest-labs/FLUX.1-schnell` and `black-forest-labs/FLUX.1-dev`; both are gated,
    /// so the caller accepts the license on Hugging Face and sets `NFKHFHub.defaultAccessToken` before
    /// the first download. Introduced in InferKit 0.4.0.
    @objc(fluxWithRepo:revision:cacheDirectoryURL:error:)
    public static func flux(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXFlux {
        try flux(directoryURL: try releaseDirectory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL))
    }

    /// The asynchronous form of ``flux(repo:revision:cacheDirectoryURL:)``. The handler runs on a
    /// background queue. Introduced in InferKit 0.4.0.
    @objc(fluxWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func flux(repo: String, revision: String?, cacheDirectoryURL: URL?,
                            completionHandler: @escaping (NFKMLXFlux?, Error?) -> Void) {
        flux(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: .automatic,
             completionHandler: completionHandler)
    }

    /// ``flux(repo:revision:cacheDirectoryURL:)`` holding the release as `residency` says. Introduced
    /// in InferKit 0.4.0.
    @objc(fluxWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func flux(repo: String, revision: String?, cacheDirectoryURL: URL?,
                            residency: NFKMLXResidency) throws -> NFKMLXFlux {
        try flux(directoryURL: try releaseDirectory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL),
                 residency: residency)
    }

    /// The asynchronous form of ``flux(repo:revision:cacheDirectoryURL:residency:)``. The handler runs
    /// on a background queue. Introduced in InferKit 0.4.0.
    @objc(fluxWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func flux(repo: String, revision: String?, cacheDirectoryURL: URL?,
                            residency: NFKMLXResidency,
                            completionHandler: @escaping (NFKMLXFlux?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try flux(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                           residency: residency), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// Assembles the whole model from a downloaded diffusers FLUX release directory. `schedule` is
    /// `.fluxSchnell` for the four-step distillation or `.flux` for the [dev] dynamic-shift schedule.
    ///
    /// @discussion The text encoders run once per image and the transformer on every step, so the two
    /// stages never need to be loaded together. A resident release holds both and loads them here; a
    /// staged one loads nothing here and, for each image, loads CLIP-L and T5-XXL, encodes, releases
    /// them, then loads the transformer and the autoencoder. The plan weighs T5-XXL at the float32 it
    /// runs at (about 19 GB) and the transformer at its stored bfloat16 (about 24 GB), so a machine that
    /// cannot hold the 43 GB together stages under ``NFKMLXResidency/automatic``. FLUX.1 has no routed
    /// experts, so ``NFKMLXResidency/paged`` holds it as ``NFKMLXResidency/staged`` does.
    /// ``holdsStagesResident`` reports the placement. Introduced in InferKit 0.4.0.
    public static func flux(directoryURL: URL, schedule: NFKMLXFlowMatchConfiguration,
                            residency: NFKMLXResidency = .automatic) throws -> NFKMLXFlux {
        let transformerDirectory = directoryURL.appendingPathComponent("transformer")
        let configuration = try NFKMLXFluxTransformerNet.configuration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json"))
        let vaeURL = directoryURL.appendingPathComponent("vae/diffusion_pytorch_model.safetensors")
        let fileBytes = { (url: URL) in
            (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        }
        // CLIP-L loads as stored; T5-XXL loads at float32, twice its stored bfloat16.
        let encoderBytes = fileBytes(directoryURL.appendingPathComponent("text_encoder/model.safetensors"))
            + 2 * (try NFKMLXReleaseWeights.weightBytes(inDirectory: directoryURL.appendingPathComponent("text_encoder_2")))
        let pipelineBytes = try NFKMLXReleaseWeights.weightBytes(inDirectory: transformerDirectory) + fileBytes(vaeURL)
        let plan = try NFKMLXResidencyBudget.plan(
            [NFKMLXStageFootprint(bytes: encoderBytes), NFKMLXStageFootprint(bytes: pipelineBytes)],
            residency: residency, budget: NFKMLXResidencyBudget.current())

        let flux = NFKMLXFlux(
            resident: plan.holdsStagesResident,
            loadTextEncoder: { try NFKMLXFluxTextEncoder.textEncoder(directoryURL: directoryURL) },
            loadPipeline: {
                let transformer = NFKMLXFluxTransformerNet(configuration)
                try NFKMLXFluxTransformerNet.loadWeights(into: transformer, from: transformerDirectory,
                                                         precision: .checkpoint)
                let vae = NFKMLXSDAutoencoder(configuration: .flux)
                try NFKMLXStableDiffusionModels.loadVAEWeights(into: vae, from: vaeURL, precision: .checkpoint)
                return NFKMLXFluxPipeline(transformer: transformer, vae: vae, schedule: schedule)
            })
        if plan.holdsStagesResident {
            try flux.loadResident()
        }
        return flux
    }

    /// Encodes a prompt into FLUX's conditioning: the T5 sequence and the CLIP pooled projection. A
    /// staged release loads the text encoders for the call and releases them after.
    public func encode(prompt: String) throws -> (promptEmbeds: MLXArray, pooled: MLXArray) {
        let encoded = try staging.with(textEncoderStage) { encoder -> [MLXArray] in
            let (embeds, pooled) = encoder.encode(prompt: prompt)
            return [embeds, pooled]
        }
        return (encoded[0], encoded[1])
    }

    /// Generates an image for `prompt`. `width` and `height` are the pixel dimensions (multiples of 16; the
    /// VAE downsamples by 8 and the transformer packs 2×2). Returns the image as `[1, H, W, 3]`.
    public func image(forPrompt prompt: String, width: Int = 1024, height: Int = 1024,
                      seed: UInt64 = 0) throws -> MLXArray {
        try staging.exclusively {
            let (embeds, pooled) = try encode(prompt: prompt)
            return try staging.with(pipelineStage) { pipeline in
                pipeline.generate(promptEmbeds: embeds, pooled: pooled, latentHeight: height / 8,
                                  latentWidth: width / 8, steps: steps, guidance: guidance, seed: seed)
            }
        }
    }

    /// Generates an image for `prompt` and returns it as a `CGImage`.
    @objc(imageForPrompt:width:height:seed:error:)
    public func cgImage(forPrompt prompt: String, width: Int, height: Int, seed: UInt64) throws -> CGImage {
        let array = try image(forPrompt: prompt, width: width, height: height, seed: seed)
        return try NFKMLXImageBridge.cgImage(from: array[0], options: NFKMLXImageOptions())
    }
}
