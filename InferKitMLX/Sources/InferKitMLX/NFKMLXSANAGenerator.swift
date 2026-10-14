//
//  NFKMLXSANAGenerator.swift
//  InferKitMLX
//
//  SANA text-to-image from a downloaded diffusers release directory: the Gemma 2 text encoder and its
//  tokenizer, then the linear-attention DiT and the Deep-Compression Autoencoder, the two stages held
//  as an `NFKMLXResidency` says. The prompt is encoded the way diffusers' `SanaPipeline` encodes it.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// The text stage: Gemma 2 read at its normalized last hidden state.
final class NFKSANATextStage {
    let encoder: NFKMLXGemma2Net
    init(_ encoder: NFKMLXGemma2Net) { self.encoder = encoder }

    /// The caption features `[kept, hidden]`: the encoder's states at `ids`, read at the positions
    /// `kept` names. The reference pads on the right under a causal mask, so the unpadded encode is the
    /// same at every real position, and the padding it masks out is never kept.
    func features(_ encoding: NFKMLXSANAPrompt.Encoding) -> MLXArray {
        let states = encoder(MLXArray(encoding.ids.map(Int32.init)))
        return states.take(MLXArray(encoding.kept.map(Int32.init)), axis: 0)
    }
}

/// The prompt as diffusers' `SanaPipeline` encodes it.
///
/// @discussion The prompt is lowercased and stripped (`clean_caption` off). A positive prompt follows
/// the pipeline's default complex human instruction, and its encoding keeps the begin-of-sequence state
/// and the states from the instruction's last token on, up to the 300 the pipeline selects. A negative
/// prompt is encoded alone, up to 300 tokens.
enum NFKMLXSANAPrompt {

    /// The token ids the encoder reads and the positions of its output the transformer reads.
    struct Encoding {
        let ids: [Int]
        let kept: [Int]
    }

    /// The caption length the reference pipeline selects.
    static let maximumSequenceLength = 300

    /// The reference pipeline's default `complex_human_instruction`, joined by newlines.
    static let humanInstruction = [
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ].joined(separator: "\n")

    /// `prompt` encoded for the text encoder. `instructionTokens` is the instruction's own token
    /// count with the begin-of-sequence marker, or nil to encode the prompt alone.
    static func encoding(_ prompt: String, tokenizer: NFKMLXGemmaTokenizer, instructionTokens: Int?) -> Encoding {
        let begin = tokenizer.id(forToken: "<bos>") ?? 2
        let text = prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard let instructionTokens else {
            let ids = Array(([begin] + tokenizer.encode(text)).prefix(maximumSequenceLength))
            return Encoding(ids: ids, kept: Array(0 ..< ids.count))
        }
        let ids = Array(([begin] + tokenizer.encode(humanInstruction + text))
            .prefix(instructionTokens + maximumSequenceLength - 2))
        // The reference keeps position 0 and the last 299 of the padded length, which start at the
        // instruction's last token.
        let first = max(instructionTokens - 1, 1)
        return Encoding(ids: ids, kept: [0] + (first < ids.count ? Array(first ..< ids.count) : []))
    }

    /// The instruction's token count with the begin-of-sequence marker, as the reference counts it.
    static func instructionTokenCount(tokenizer: NFKMLXGemmaTokenizer) -> Int {
        1 + tokenizer.encode(humanInstruction).count
    }
}

/// SANA text-to-image, assembled from a diffusers release directory.
///
/// @discussion The release layout is `text_encoder/` (Gemma 2), `tokenizer/`, `transformer/`, `vae/`
/// (the Deep-Compression Autoencoder), and `scheduler/`. The factories read
/// `Efficient-Large-Model/Sana_600M_1024px_diffusers` and the 1600M release. ``image(forPrompt:negativePrompt:width:height:seed:)``
/// encodes the prompt through Gemma 2 behind the reference pipeline's complex human instruction,
/// denoises over the flow-prediction DPM-Solver schedule with classifier-free guidance, and decodes.
/// The image is generated at the requested size: the reference pipeline's resolution binning, which
/// generates at the nearest trained aspect ratio and resizes back, is not applied.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXSANAGenerator)
public final class NFKMLXSANAGenerator: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "sana"

    /// The Hugging Face repository the download factories read when none is named.
    @objc public static let releaseRepo = "Efficient-Large-Model/Sana_600M_1024px_diffusers"

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKSANATextStage>
    private let pipelineStage: NFKMLXStage<NFKMLXSANAPipeline>
    private let tokenizer: NFKMLXGemmaTokenizer
    private let instructionTokens: Int

    /// The pixels one latent cell decodes to: 32 for the released autoencoder.
    @objc public let pixelsPerLatent: Int

    /// The denoising steps, 20 by the reference's default.
    @objc public var steps: Int = 20
    /// The classifier-free guidance scale, 4.5 by the reference's default. Above 1 the image guides
    /// against the negative prompt, or against an empty one where none is given.
    @objc public var guidance: Float = 4.5
    /// Whether a positive prompt follows the reference pipeline's complex human instruction. True by
    /// the reference's default.
    @objc public var followsHumanInstruction = true

    /// Whether the text encoder and the transformer stay loaded together between images.
    @objc public var holdsStagesResident: Bool { staging.resident }

    init(resident: Bool, tokenizer: NFKMLXGemmaTokenizer, pixelsPerLatent: Int,
         loadTextEncoder: @escaping () throws -> NFKMLXGemma2Net,
         loadPipeline: @escaping () throws -> NFKMLXSANAPipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage { NFKSANATextStage(try loadTextEncoder()) }
        pipelineStage = NFKMLXStage(loadPipeline)
        self.tokenizer = tokenizer
        self.pixelsPerLatent = pixelsPerLatent
        instructionTokens = NFKMLXSANAPrompt.instructionTokenCount(tokenizer: tokenizer)
        super.init()
    }

    /// Loads both stages now, for a resident release.
    func loadResident() throws {
        try staging.use(textStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    var isHoldingTextEncoder: Bool { staging.exclusively { textStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    /// Loads a release's Gemma 2 text encoder. The SANA release saves the bare `Gemma2Model`, whose
    /// names carry no `model.` prefix; a causal-language-model save carries it, and both load.
    static func loadTextEncoder(fromDirectory directory: URL) throws -> NFKMLXGemma2Net {
        let net = NFKMLXGemma2Net(try NFKMLXGemma2Configuration.configuration(
            fromHuggingFace: directory.appendingPathComponent("config.json")))
        let mapped = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, remap: { key in
            if key.hasPrefix("lm_head.") { return nil }
            return key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key
        })
        try NFKMLXWeights.apply(mapped, to: net)
        return net
    }

    // MARK: Factories

    /// Assembles the model from a downloaded release directory, holding it as
    /// ``NFKMLXResidency/automatic`` decides.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXSANAGenerator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    ///
    /// @discussion Gemma 2 loads at float32, about 10 GB. The transformer and the autoencoder load at
    /// float32. A machine that cannot hold the two stages together stages under
    /// ``NFKMLXResidency/automatic``. The release has no routed experts, so ``NFKMLXResidency/paged``
    /// holds it as ``NFKMLXResidency/staged`` does.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXSANAGenerator {
        let encoderDirectory = directoryURL.appendingPathComponent("text_encoder")
        guard let tokenizer = NFKMLXGemmaTokenizer(
            directoryURL: directoryURL.appendingPathComponent("tokenizer")) else {
            throw NFKMLXError.unsupportedConfiguration("the release's tokenizer/ carries no readable tokenizer.json")
        }
        let plan = try NFKMLXResidencyBudget.plan(
            [NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(inDirectory: encoderDirectory, holding: .float32)),
             NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(
                inDirectory: directoryURL.appendingPathComponent("transformer"), holding: .float32)
                + NFKMLXStageWeights.bytes(inDirectory: directoryURL.appendingPathComponent("vae"), holding: .float32))],
            residency: residency, budget: NFKMLXResidencyBudget.current())
        let generator = NFKMLXSANAGenerator(
            resident: plan.holdsStagesResident, tokenizer: tokenizer,
            pixelsPerLatent: 1 << (NFKMLXDCAEConfiguration.sana.blockOutChannels.count - 1),
            loadTextEncoder: { try loadTextEncoder(fromDirectory: encoderDirectory) },
            loadPipeline: { try NFKMLXSANAPipeline.pipeline(directoryURL: directoryURL) })
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    /// The component folders a download fetches.
    static let releaseComponents = [
        NFKMLXReleaseComponent(required: ["model_index.json", "scheduler/scheduler_config.json"]),
        NFKMLXReleaseComponent(required: ["text_encoder/config.json"],
                               weights: ["text_encoder/model.safetensors.index.json",
                                         "text_encoder/model.safetensors"]),
        NFKMLXReleaseComponent(required: ["tokenizer/tokenizer.json"],
                               optional: ["tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json"]),
        NFKMLXReleaseComponent(required: ["transformer/config.json"],
                               weights: ["transformer/diffusion_pytorch_model.safetensors",
                                         "transformer/diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]),
    ]

    /// Downloads the release (``releaseRepo`` when `repo` is nil) and assembles the model, holding it
    /// as `residency` says.
    ///
    /// @discussion The 600M release is about 14 GB at the float32 it is stored in. A file already in the
    /// cache is not fetched again. The call blocks on the network, so run it off the main and render
    /// threads. The repositories are not gated.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXSANAGenerator {
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
                                 completionHandler: @escaping (NFKMLXSANAGenerator?, Error?) -> Void) {
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

    /// The prompt's encoding: behind the human instruction where `instructed` is true.
    func encoding(_ prompt: String, instructed: Bool) -> NFKMLXSANAPrompt.Encoding {
        NFKMLXSANAPrompt.encoding(prompt, tokenizer: tokenizer, instructionTokens: instructed ? instructionTokens : nil)
    }

    /// The caption features the transformer reads, `[tokens, 2304]`, for a positive prompt. A staged
    /// release loads the text encoder for the call and releases it after.
    public func encode(prompt: String) throws -> MLXArray {
        let encoded = encoding(prompt, instructed: followsHumanInstruction)
        return try staging.with(textStage) { $0.features(encoded) }
    }

    /// Generates an image for `prompt`, `[height, width, 3]` RGB in `[0, 1]`.
    ///
    /// @discussion `width` and `height` are positive multiples of ``pixelsPerLatent``. Above a
    /// ``guidance`` of 1 the image guides against `negativePrompt`, or against an empty prompt where
    /// none is given; both prompts encode from one load of the text encoder.
    public func image(forPrompt prompt: String, negativePrompt: String? = nil, width: Int = 1024,
                      height: Int = 1024, seed: UInt64 = 0) throws -> MLXArray {
        guard width > 0, height > 0, width % pixelsPerLatent == 0, height % pixelsPerLatent == 0 else {
            throw NFKMLXError.unsupportedConfiguration("an image's sides are positive multiples of \(pixelsPerLatent)")
        }
        let guides = guidance > 1
        let encodings = [encoding(prompt, instructed: followsHumanInstruction)]
            + (guides ? [encoding(negativePrompt ?? "", instructed: false)] : [])
        return try staging.exclusively {
            let features = try staging.with(textStage) { stage in encodings.map { stage.features($0) } }
            return try staging.with(pipelineStage) { pipeline in
                let image = pipeline.generate(promptEmbeds: features[0], negativeEmbeds: guides ? features[1] : nil,
                                              latentHeight: height / pixelsPerLatent,
                                              latentWidth: width / pixelsPerLatent, steps: steps,
                                              guidance: guidance, seed: seed)
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
