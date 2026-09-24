//
//  NFKMLXSD3Generator.swift
//  InferKitMLX
//
//  Stable Diffusion 3 and 3.5 text-to-image from a downloaded diffusers release directory, its two
//  stages held as an `NFKMLXResidency` says: the three text towers (CLIP-L, CLIP-G, T5-XXL) run once per
//  image and the MMDiT and autoencoder after them.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// Reading a diffusers Stable Diffusion 3 release's schedule and transformer.
enum NFKMLXSD3Release {

    /// The schedule a `scheduler/scheduler_config.json` states, over the scheduler's own ramp, which the
    /// SD3 pipelines leave in place: a static shift (3.0 in the released configs), or the
    /// resolution-dependent one with the pipeline's own bounds where dynamic shifting is on.
    static func schedule(fromHuggingFace url: URL) throws -> NFKMLXFlowMatchConfiguration {
        let json = try NFKMLXWanRelease.json(url)
        let float = { (key: String, fallback: Float) in (json[key] as? NSNumber)?.floatValue ?? fallback }
        let dynamic = (json["use_dynamic_shifting"] as? Bool) == true
        return NFKMLXFlowMatchConfiguration(
            trainTimesteps: json["num_train_timesteps"] as? Int ?? 1000,
            baseShift: dynamic ? float("base_shift", 0.5) : float("shift", 1),
            maxShift: float("max_shift", 1.16), baseSequenceLength: json["base_image_seq_len"] as? Int ?? 256,
            maxSequenceLength: json["max_image_seq_len"] as? Int ?? 4096,
            shiftTerminal: (json["shift_terminal"] as? NSNumber)?.floatValue, useDynamicShifting: dynamic)
    }

    /// Loads a release's `transformer/` weights at `dtype`. The patch-embed convolution is stored
    /// `[out, in, h, w]` and loads as MLX's `[out, h, w, in]`.
    static func loadTransformer(into net: NFKMLXSD3TransformerNet, fromDirectory directory: URL,
                                dtype: DType) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, converting: dtype)
        try NFKMLXWeights.apply(arrays.map { key, value in (key, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value) },
                                to: net, verifyShapes: true)
    }
}

/// A prompt as the three text towers read it.
struct NFKSD3Prompt {
    let clipL: [Int]
    let clipG: [Int]
    let t5: NFKMLXT5Prompt?
}

/// The text stage: the two CLIP towers and, where the release ships it, T5-XXL.
final class NFKSD3TextStage {
    let clipL: NFKMLXSDTextEncoderNet
    let clipG: NFKMLXSDTextEncoderNet
    let t5: NFKMLXT5EncoderNet?
    let jointDimensions: Int
    let t5Length: Int

    init(clipL: NFKMLXSDTextEncoderNet, clipG: NFKMLXSDTextEncoderNet, t5: NFKMLXT5EncoderNet?,
         jointDimensions: Int, t5Length: Int) {
        self.clipL = clipL
        self.clipG = clipG
        self.t5 = t5
        self.jointDimensions = jointDimensions
        self.t5Length = t5Length
    }

    /// The joint sequence `[77 + t5Length, jointDimensions]` and the pooled projection the transformer
    /// reads: the two CLIP penultimate states concatenated on channels and zero-padded to T5's width,
    /// followed by the T5 sequence (zeros where the release ships no T5), and the two pooled
    /// projections concatenated.
    func embeddings(_ prompt: NFKSD3Prompt) -> (sequence: MLXArray, pooled: MLXArray) {
        let l = clipL.encode(prompt.clipL), g = clipG.encode(prompt.clipG)
        let clip = concatenated([l.hidden[0], g.hidden[0]], axis: -1)
        let padded = padded(clip, widths: [.init((0, 0)), .init((0, jointDimensions - clip.dim(-1)))])
        var t5Sequence = MLXArray.zeros([t5Length, jointDimensions])
        if let t5, let tokens = prompt.t5?.tokens {
            t5Sequence = t5(tokens)[0]
        }
        let pooled = concatenated([l.pooled ?? MLXArray.zeros([1, 0]), g.pooled ?? MLXArray.zeros([1, 0])], axis: -1)[0]
        return (concatenated([padded, t5Sequence.asType(padded.dtype)], axis: 0), pooled)
    }
}

/// Stable Diffusion 3 and 3.5 text-to-image, assembled from a diffusers release directory.
///
/// @discussion The release layout is `text_encoder/` (CLIP-L with its projection), `text_encoder_2/`
/// (OpenCLIP bigG), `text_encoder_3/` (T5-XXL, optional), their `tokenizer/`, `tokenizer_2/` and
/// `tokenizer_3/`, `transformer/`, `vae/`, and `scheduler/`. The factories read SD3 Medium and SD3.5
/// Medium and Large; the MMDiT-X dual-attention layers and the query-key norm are read from the config.
/// ``image(forPrompt:negativePrompt:width:height:seed:)`` encodes the prompt through the three towers,
/// denoises over the release's flow schedule, and decodes. The glue is measured against diffusers'
/// `StableDiffusion3Pipeline`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXSD3Generator)
public final class NFKMLXSD3Generator: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "sd3"

    /// The Hugging Face repository the download factories read when none is named. The repository is
    /// gated: its license is accepted once on the Hub, and the token is `NFKHubClient`'s.
    @objc public static let releaseRepo = "stabilityai/stable-diffusion-3.5-medium"

    /// The CLIP context length and the T5 length the reference pipeline pads to.
    static let clipLength = 77
    static let maximumSequenceLength = 256

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKSD3TextStage>
    private let pipelineStage: NFKMLXStage<NFKMLXSD3Pipeline>
    private let tokenizerL: NFKMLXSDPromptTokenizer
    private let tokenizerG: NFKMLXSDPromptTokenizer
    private let segmenter: NFKMLXSentencePieceSegmenter?
    private let t5Length: Int

    /// The denoising steps, 28 by the reference's default.
    @objc public var steps: Int = 28
    /// The classifier-free guidance scale, 7 by the reference's default. Above 1 the image guides
    /// against the negative prompt, or against an empty one where none is given.
    @objc public var guidance: Float = 7

    /// Whether the text towers and the transformer stay loaded together between images.
    @objc public var holdsStagesResident: Bool { staging.resident }

    /// Whether T5-XXL runs at float32 rather than at bfloat16. It does where the residency plan affords
    /// the float32 encoder. False also where the release ships no T5.
    @objc public internal(set) var encodesInFloat32 = true

    init(resident: Bool, tokenizerL: NFKMLXSDPromptTokenizer, tokenizerG: NFKMLXSDPromptTokenizer,
         segmenter: NFKMLXSentencePieceSegmenter?, t5Length: Int = NFKMLXSD3Generator.maximumSequenceLength,
         loadTextEncoders: @escaping () throws -> NFKSD3TextStage,
         loadPipeline: @escaping () throws -> NFKMLXSD3Pipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage(loadTextEncoders)
        pipelineStage = NFKMLXStage(loadPipeline)
        self.tokenizerL = tokenizerL
        self.tokenizerG = tokenizerG
        self.segmenter = segmenter
        self.t5Length = t5Length
        super.init()
    }

    /// Loads both stages now, for a resident release.
    func loadResident() throws {
        try staging.use(textStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    var isHoldingTextEncoders: Bool { staging.exclusively { textStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    /// `text` tokenized for each tower: CLIP padded to 77 (bigG's tokenizer pads with `!`), T5 padded
    /// to the sequence length with its end token.
    func prompt(_ text: String) -> NFKSD3Prompt {
        NFKSD3Prompt(clipL: tokenizerL.tokens(for: text, contextLength: Self.clipLength),
                     clipG: tokenizerG.tokens(for: text, contextLength: Self.clipLength),
                     t5: segmenter.map { NFKMLXT5Prompt(text, segmenter: $0, length: t5Length) })
    }

    // MARK: Factories

    /// Assembles the model from a downloaded release directory, holding it as
    /// ``NFKMLXResidency/automatic`` decides.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXSD3Generator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    ///
    /// @discussion The CLIP towers and the autoencoder load at float32. T5-XXL runs at float32 where the
    /// residency plan affords it (beside the transformer when resident, on its own when staged), and at
    /// bfloat16 otherwise. The transformer loads at bfloat16, the
    /// type the releases are published to run at. A release without `text_encoder_3/` conditions on zeros
    /// in T5's place, as the reference does. A machine that cannot hold the two stages together stages
    /// under ``NFKMLXResidency/automatic``. The release has no routed experts, so
    /// ``NFKMLXResidency/paged`` holds it as ``NFKMLXResidency/staged`` does.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXSD3Generator {
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
                                     + NFKMLXStageWeights.bytes(inDirectory: directory("vae"), holding: .float32))],
            residency: residency, budget: NFKMLXResidencyBudget.current())
        let t5InFloat32 = plan.widens(0)
        let generator = NFKMLXSD3Generator(
            resident: plan.holdsStagesResident, tokenizerL: tokenizerL, tokenizerG: tokenizerG, segmenter: segmenter,
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
                                       t5Length: maximumSequenceLength)
            },
            loadPipeline: {
                let transformer = NFKMLXSD3TransformerNet(configuration)
                try NFKMLXSD3Release.loadTransformer(into: transformer, fromDirectory: directory("transformer"),
                                                     dtype: .bfloat16)
                let vae = NFKMLXSDAutoencoder(configuration: vaeConfiguration)
                try NFKMLXStableDiffusionModels.loadVAEWeights(into: vae, from: vaeURL, precision: .float32)
                return NFKMLXSD3Pipeline(transformer: transformer, vae: vae, schedule: schedule)
            })
        generator.encodesInFloat32 = t5InFloat32
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    /// The component folders a download fetches. T5-XXL is fetched with the rest; a release without it
    /// is read from a directory.
    static let releaseComponents = [
        NFKMLXReleaseComponent(required: ["model_index.json", "scheduler/scheduler_config.json"]),
        NFKMLXReleaseComponent(required: ["text_encoder/config.json", "text_encoder/model.safetensors"]),
        NFKMLXReleaseComponent(required: ["text_encoder_2/config.json", "text_encoder_2/model.safetensors"]),
        NFKMLXReleaseComponent(required: ["text_encoder_3/config.json"],
                               weights: ["text_encoder_3/model.safetensors",
                                         "text_encoder_3/model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["tokenizer/vocab.json", "tokenizer/merges.txt"],
                               optional: ["tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json"]),
        NFKMLXReleaseComponent(required: ["tokenizer_2/vocab.json", "tokenizer_2/merges.txt"],
                               optional: ["tokenizer_2/tokenizer_config.json", "tokenizer_2/special_tokens_map.json"]),
        NFKMLXReleaseComponent(required: ["tokenizer_3/spiece.model"]),
        NFKMLXReleaseComponent(required: ["transformer/config.json"],
                               weights: ["transformer/diffusion_pytorch_model.safetensors",
                                         "transformer/diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]),
    ]

    /// Downloads the release (``releaseRepo`` when `repo` is nil) and assembles the model, holding it
    /// as `residency` says.
    ///
    /// @discussion SD3.5 Medium is about 20 GB with T5-XXL. A file already in the cache is not fetched
    /// again. The call blocks on the network, so run it off the main and render threads. The weights are
    /// under the Stability AI Community License.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXSD3Generator {
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
                                 completionHandler: @escaping (NFKMLXSD3Generator?, Error?) -> Void) {
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

    /// Generates an image for `prompt`, `[height, width, 3]` RGB in `[0, 1]`.
    ///
    /// @discussion `width` and `height` are multiples of 16, which the reference requires. Above a
    /// ``guidance`` of 1 the image guides against `negativePrompt`, or against an empty prompt where
    /// none is given; both prompts encode from one load of the text towers.
    public func image(forPrompt prompt: String, negativePrompt: String? = nil, width: Int = 1024,
                      height: Int = 1024, seed: UInt64 = 0) throws -> MLXArray {
        guard width > 0, height > 0, width % 16 == 0, height % 16 == 0 else {
            throw NFKMLXError.unsupportedConfiguration("an image's sides are positive multiples of 16")
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
                    latentHeight: height / pipeline.pixelsPerLatent, latentWidth: width / pipeline.pixelsPerLatent,
                    steps: steps, guidance: guidance, seed: seed)
                return clip(image[0].asType(.float32) / 2 + 0.5, min: 0, max: 1)
            }
        }
    }

    /// Generates an image for `prompt` and returns it as a `CGImage`.
    @objc(imageForPrompt:negativePrompt:width:height:seed:error:)
    public func cgImage(forPrompt prompt: String, negativePrompt: String?, width: Int, height: Int,
                        seed: UInt64) throws -> CGImage {
        let array = try image(forPrompt: prompt, negativePrompt: negativePrompt, width: width, height: height,
                              seed: seed)
        return try NFKMLXImageBridge.cgImage(from: array, options: NFKMLXImageOptions())
    }
}
