//
//  NFKMLXSD3ControlNetGenerator.swift
//  InferKitMLX
//
//  Stable Diffusion 3 and 3.5 steered by a ControlNet, from a downloaded diffusers SD3 release
//  directory and a ControlNet release directory: the three text towers run once per image, then the
//  MMDiT, the ControlNet, and the autoencoder, the two stages held as an `NFKMLXResidency` says.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// Stable Diffusion 3 and 3.5 text-to-image steered by a control image through an SD3 ControlNet.
///
/// @discussion The base release is the diffusers layout ``NFKMLXSD3Generator`` reads, and its text towers
/// encode the prompt the same way. The ControlNet release is an `SD3ControlNetModel` directory
/// (`config.json` and its weights): the InstantX dual-stream ControlNets for SD3 Medium, or the
/// Stability single-stream ones for SD3.5 Large. The glue is measured against diffusers'
/// `StableDiffusion3ControlNetPipeline`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXSD3ControlNetGenerator)
public final class NFKMLXSD3ControlNetGenerator: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "sd3-controlnet"

    /// The base repository the download factories read when none is named. It is gated: its license is
    /// accepted once on the Hub, and the token is `NFKHFHub.defaultAccessToken`.
    @objc public static let releaseRepo = "stabilityai/stable-diffusion-3.5-large"

    /// The ControlNet repository the download factories read when none is named.
    @objc public static let controlNetReleaseRepo = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKSD3TextStage>
    private let pipelineStage: NFKMLXStage<NFKMLXSD3ControlNetPipeline>
    private let tokenizerL: NFKMLXSDPromptTokenizer
    private let tokenizerG: NFKMLXSDPromptTokenizer
    private let segmenter: NFKMLXSentencePieceSegmenter?
    private let t5Length: Int

    /// The pixels one latent cell decodes to: 8 for the released autoencoders.
    @objc public let pixelsPerLatent: Int

    /// The denoising steps, 28 by the reference's default.
    @objc public var steps: Int = 28
    /// The classifier-free guidance scale, 7 by the reference's default. Above 1 the image guides
    /// against the negative prompt, or against an empty one where none is given.
    @objc public var guidance: Float = 7
    /// The weight the ControlNet's residuals carry, 1 by the reference's default.
    @objc public var conditioningScale: Float = 1

    /// Whether the text towers and the transformer stay loaded together between images.
    @objc public var holdsStagesResident: Bool { staging.resident }

    init(resident: Bool, tokenizerL: NFKMLXSDPromptTokenizer, tokenizerG: NFKMLXSDPromptTokenizer,
         segmenter: NFKMLXSentencePieceSegmenter?, t5Length: Int = NFKMLXSD3Generator.maximumSequenceLength,
         pixelsPerLatent: Int,
         loadTextEncoders: @escaping () throws -> NFKSD3TextStage,
         loadPipeline: @escaping () throws -> NFKMLXSD3ControlNetPipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage(loadTextEncoders)
        pipelineStage = NFKMLXStage(loadPipeline)
        self.tokenizerL = tokenizerL
        self.tokenizerG = tokenizerG
        self.segmenter = segmenter
        self.t5Length = t5Length
        self.pixelsPerLatent = pixelsPerLatent
        super.init()
    }

    /// Loads both stages now, for a resident release.
    func loadResident() throws {
        try staging.use(textStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    var isHoldingTextEncoders: Bool { staging.exclusively { textStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    /// `text` tokenized for each tower, as ``NFKMLXSD3Generator`` tokenizes it.
    func prompt(_ text: String) -> NFKSD3Prompt {
        NFKSD3Prompt(clipL: tokenizerL.tokens(for: text, contextLength: NFKMLXSD3Generator.clipLength),
                     clipG: tokenizerG.tokens(for: text, contextLength: NFKMLXSD3Generator.clipLength),
                     t5: segmenter.map { NFKMLXT5Prompt(text, segmenter: $0, length: t5Length) })
    }

    // MARK: Factories

    /// Assembles the model from an SD3 release directory and a ControlNet release directory, holding it
    /// as ``NFKMLXResidency/automatic`` decides.
    @objc(generatorWithDirectoryURL:controlNetDirectoryURL:error:)
    public static func generator(directoryURL: URL, controlNetDirectoryURL: URL) throws -> NFKMLXSD3ControlNetGenerator {
        try generator(directoryURL: directoryURL, controlNetDirectoryURL: controlNetDirectoryURL, residency: .automatic)
    }

    /// Assembles the model from an SD3 release directory and a ControlNet release directory, holding it
    /// as `residency` says.
    ///
    /// @discussion The CLIP towers and the autoencoder load at float32. T5-XXL runs at float32 where the
    /// residency plan affords it and at bfloat16 otherwise. The transformer loads at bfloat16 and the
    /// ControlNet as stored. A release without `text_encoder_3/` conditions on zeros in T5's place, as
    /// the reference does. A machine that cannot hold the two stages together stages under
    /// ``NFKMLXResidency/automatic``. The release has no routed experts, so ``NFKMLXResidency/paged``
    /// holds it as ``NFKMLXResidency/staged`` does.
    @objc(generatorWithDirectoryURL:controlNetDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, controlNetDirectoryURL: URL,
                                 residency: NFKMLXResidency) throws -> NFKMLXSD3ControlNetGenerator {
        let directory = { (name: String) in directoryURL.appendingPathComponent(name) }
        let tokenizerL = try NFKMLXSDPromptTokenizer(directoryURL: directory("tokenizer"))
        let tokenizerG = try NFKMLXSDPromptTokenizer(directoryURL: directory("tokenizer_2"))
        let clipL = try NFKMLXSDTextEncoder.configuration(
            fromHuggingFace: directory("text_encoder/config.json"), output: .penultimateHiddenState)
        let clipG = try NFKMLXSDTextEncoder.configuration(
            fromHuggingFace: directory("text_encoder_2/config.json"), output: .penultimateHiddenState)
        let t5Directory = directory("text_encoder_3")
        let hasT5 = FileManager.default.fileExists(atPath: t5Directory.appendingPathComponent("config.json").path)
        let t5 = hasT5 ? try NFKMLXT5Encoder.configuration(fromHuggingFace: t5Directory.appendingPathComponent("config.json")) : nil
        let segmenter = hasT5 ? NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: directory("tokenizer_3/spiece.model"))) : nil
        let configuration = try NFKMLXSD3TransformerNet.configuration(fromHuggingFace: directory("transformer/config.json"))
        let controlConfiguration = try NFKMLXSD3ControlNetNet.configuration(
            fromHuggingFace: controlNetDirectoryURL.appendingPathComponent("config.json"))
        let vaeConfiguration = try NFKMLXStableDiffusionModels.vaeConfiguration(fromHuggingFace: directory("vae/config.json"))
        let schedule = try NFKMLXSD3Release.schedule(fromHuggingFace: directory("scheduler/scheduler_config.json"))
        let clipLURL = try NFKMLXReleaseWeights.files(inDirectory: directory("text_encoder"))
        let clipGURL = try NFKMLXReleaseWeights.files(inDirectory: directory("text_encoder_2"))
        guard clipLURL.count == 1, clipGURL.count == 1 else {
            throw NFKMLXError.unsupportedConfiguration("each CLIP tower is read from one weight file")
        }
        let vaeURL = directory("vae/diffusion_pytorch_model.safetensors")

        // T5-XXL takes float32 as its wider precision over bfloat16; the CLIP towers are float32 in both.
        let clipBytes = try NFKMLXStageWeights.bytes(inDirectory: directory("text_encoder"), holding: .float32)
            + NFKMLXStageWeights.bytes(inDirectory: directory("text_encoder_2"), holding: .float32)
        let t5Bytes = { (dtype: DType) in hasT5 ? try NFKMLXStageWeights.bytes(inDirectory: t5Directory, holding: dtype) : 0 }
        let plan = try NFKMLXResidencyBudget.plan(
            [NFKMLXStageFootprint(bytes: clipBytes + (try t5Bytes(.bfloat16)),
                                  widenedBytes: hasT5 ? clipBytes + (try t5Bytes(.float32)) : nil),
             NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(inDirectory: directory("transformer"), holding: .bfloat16)
                                     + NFKMLXStageWeights.bytes(inDirectory: controlNetDirectoryURL, precision: .checkpoint)
                                     + NFKMLXStageWeights.bytes(inDirectory: directory("vae"), holding: .float32))],
            residency: residency, budget: NFKMLXResidencyBudget.current())
        let t5InFloat32 = plan.widens(0)
        let generator = NFKMLXSD3ControlNetGenerator(
            resident: plan.holdsStagesResident, tokenizerL: tokenizerL, tokenizerG: tokenizerG, segmenter: segmenter,
            pixelsPerLatent: 1 << (vaeConfiguration.blockChannels.count - 1),
            loadTextEncoders: {
                let encoderL = try NFKMLXSDTextEncoder.net(configuration: clipL, weightsURL: clipLURL[0])
                let encoderG = try NFKMLXSDTextEncoder.net(configuration: clipG, weightsURL: clipGURL[0])
                let encoder3 = try t5.map { configuration -> NFKMLXT5EncoderNet in
                    let encoder = NFKMLXT5Encoder.makeNet(configuration)
                    try NFKMLXT5Encoder.loadWeights(into: encoder, from: t5Directory,
                                                    dtype: t5InFloat32 ? .float32 : .bfloat16)
                    return encoder
                }
                return NFKSD3TextStage(clipL: encoderL, clipG: encoderG, t5: encoder3,
                                       jointDimensions: configuration.jointAttentionDim,
                                       t5Length: NFKMLXSD3Generator.maximumSequenceLength)
            },
            loadPipeline: {
                let transformer = NFKMLXSD3TransformerNet(configuration)
                try NFKMLXSD3Release.loadTransformer(into: transformer, fromDirectory: directory("transformer"),
                                                     dtype: .bfloat16)
                let controlnet = NFKMLXSD3ControlNetNet(controlConfiguration)
                try NFKMLXSD3ControlNetNet.loadWeights(into: controlnet, from: controlNetDirectoryURL,
                                                       precision: .checkpoint)
                let vae = NFKMLXSDAutoencoder(configuration: vaeConfiguration)
                try NFKMLXStableDiffusionModels.loadVAEWeights(into: vae, from: vaeURL, precision: .float32)
                return NFKMLXSD3ControlNetPipeline(transformer: transformer, controlnet: controlnet, vae: vae,
                                                   schedule: schedule)
            })
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    /// The files a ControlNet download fetches.
    static let controlNetComponents = [
        NFKMLXReleaseComponent(required: ["config.json"],
                               weights: ["diffusion_pytorch_model.safetensors",
                                         "diffusion_pytorch_model.safetensors.index.json"]),
    ]

    /// Downloads the base release (``releaseRepo`` when `repo` is nil) and the ControlNet
    /// (``controlNetReleaseRepo`` when `controlNetRepo` is nil), and assembles the model, holding it as
    /// `residency` says.
    ///
    /// @discussion `revision` names the base release's revision; the ControlNet is read at its main
    /// revision. A file already in the cache is not fetched again. The call blocks on the network, so
    /// run it off the main and render threads. The weights are under the Stability AI Community License.
    @objc(generatorWithRepo:controlNetRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, controlNetRepo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXSD3ControlNetGenerator {
        let base = try NFKMLXReleaseDownload.directory(repo: repo ?? releaseRepo, revision: revision,
                                                       cacheDirectoryURL: cacheDirectoryURL,
                                                       components: NFKMLXSD3Generator.releaseComponents)
        let control = try NFKMLXReleaseDownload.directory(repo: controlNetRepo ?? controlNetReleaseRepo, revision: nil,
                                                          cacheDirectoryURL: cacheDirectoryURL,
                                                          components: controlNetComponents)
        return try generator(directoryURL: base, controlNetDirectoryURL: control, residency: residency)
    }

    /// The asynchronous form of ``generator(repo:controlNetRepo:revision:cacheDirectoryURL:residency:)``.
    /// The handler runs on a background queue.
    @objc(generatorWithRepo:controlNetRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func generator(repo: String?, controlNetRepo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency,
                                 completionHandler: @escaping (NFKMLXSD3ControlNetGenerator?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try generator(repo: repo, controlNetRepo: controlNetRepo, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL, residency: residency), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    // MARK: Generation

    /// Generates an image for `prompt` steered by `controlImage`, `[height, width, 3]` RGB in `[0, 1]`.
    ///
    /// @discussion `controlImage` is `[1, height, width, 3]` in `[-1, 1]`. `width` and `height` are
    /// positive multiples of 16. Above a ``guidance`` of 1 the image guides against `negativePrompt`, or
    /// against an empty prompt where none is given; the ControlNet runs once per guidance branch.
    public func image(forPrompt prompt: String, negativePrompt: String? = nil, controlImage: MLXArray,
                      width: Int, height: Int, seed: UInt64 = 0) throws -> MLXArray {
        let multiple = 2 * pixelsPerLatent
        guard width > 0, height > 0, width % multiple == 0, height % multiple == 0 else {
            throw NFKMLXError.unsupportedConfiguration("an image's sides are positive multiples of \(multiple)")
        }
        guard controlImage.ndim == 4, controlImage.dim(1) == height, controlImage.dim(2) == width else {
            throw NFKMLXError.unsupportedConfiguration("the control image is [1, height, width, 3] at the output size")
        }
        let guides = guidance > 1
        let prompts = [self.prompt(prompt)] + (guides ? [self.prompt(negativePrompt ?? "")] : [])
        return try staging.exclusively {
            // Sequence and pooled projection per prompt, in that order.
            let embeddings = try staging.with(textStage) { stage in
                prompts.flatMap { prompt -> [MLXArray] in
                    let embedding = stage.embeddings(prompt)
                    return [embedding.sequence, embedding.pooled]
                }
            }
            return try staging.with(pipelineStage) { pipeline in
                let image = pipeline.generate(
                    promptEmbeds: embeddings[0], pooled: embeddings[1],
                    negativeEmbeds: guides ? embeddings[2] : nil, negativePooled: guides ? embeddings[3] : nil,
                    controlImage: controlImage, controlnetScale: conditioningScale,
                    latentHeight: height / pixelsPerLatent, latentWidth: width / pixelsPerLatent,
                    steps: steps, guidance: guidance, seed: seed)
                return clip(image[0].asType(.float32) / 2 + 0.5, min: 0, max: 1)
            }
        }
    }

    /// Generates an image for `prompt` steered by `controlImage`, resized to `width` × `height`, and
    /// returns it as a `CGImage`.
    @objc(imageForPrompt:negativePrompt:controlImage:width:height:seed:error:)
    public func cgImage(forPrompt prompt: String, negativePrompt: String?, controlImage: CGImage, width: Int,
                        height: Int, seed: UInt64) throws -> CGImage {
        let control = try NFKMLXControlImage.pixels(controlImage, width: width, height: height)
        let array = try image(forPrompt: prompt, negativePrompt: negativePrompt, controlImage: control,
                              width: width, height: height, seed: seed)
        return try NFKMLXImageBridge.cgImage(from: array, options: NFKMLXImageOptions())
    }
}
