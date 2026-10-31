//
//  NFKMLXLTX2Generator.swift
//  InferKitMLX
//
//  LTX-2 text-to-video-and-audio from a downloaded diffusers release directory, its three stages held as
//  an `NFKMLXResidency` says: the Gemma text encoder, the text connectors, and the transformer with the
//  video decoder, audio decoder, and vocoder.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

/// The LTX-2 releases the factories read.
@objc(NFKMLXLTX2Variant)
public enum NFKMLXLTX2Variant: Int, CaseIterable, Sendable {
    /// LTX-2.3, `diffusers/LTX-2.3-Diffusers`: the 22B transformer, Gemma 3 12B as the text encoder.
    case ltx23 = 0
    /// LTX-2.5, `Lightricks/LTX-2.5-Diffusers` (gated): the distilled 22B transformer without the video
    /// feed-forward's bias and with the keyframe embedding, Gemma 4 12B as the text encoder.
    case ltx25 = 1

    /// The registry name.
    public var modelName: String {
        switch self {
        case .ltx23: return "ltx-2.3"
        case .ltx25: return "ltx-2.5"
        }
    }

    /// The Hugging Face repository the download factories read.
    public var releaseRepo: String {
        switch self {
        case .ltx23: return "diffusers/LTX-2.3-Diffusers"
        case .ltx25: return "Lightricks/LTX-2.5-Diffusers"
        }
    }

    /// The release's display name.
    public var displayName: String {
        switch self {
        case .ltx23: return "LTX-2.3"
        case .ltx25: return "LTX-2.5"
        }
    }
}

/// The text encoder of an LTX-2 release: a Gemma decoder whose hidden state at every layer conditions the
/// connectors, and the tokenizer that feeds it.
final class NFKLTX2TextStage {
    /// The stacked hidden states `[1, tokens, hidden · (layers + 1)]` for token ids.
    let states: ([Int]) -> MLXArray
    /// Throws the first failed read of a streamed layer since the last check.
    let verify: () throws -> Void

    init(states: @escaping ([Int]) -> MLXArray, verify: @escaping () throws -> Void) {
        self.states = states
        self.verify = verify
    }
}

/// The text encoders an LTX-2 release names, by their `model_type`.
enum NFKLTX2TextEncoder {
    case gemma3(NFKMLXGemma3Configuration)
    case gemma4(NFKMLXGemmaConfiguration)
    case gemma4Unified(NFKMLXGemmaConfiguration)

    /// Reads `text_encoder/config.json`: Gemma 3 (LTX-2.3) and Gemma 4, dense or unified (LTX-2.5).
    static func reading(_ url: URL) throws -> NFKLTX2TextEncoder {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        let text = (json["text_config"] as? [String: Any]) ?? json
        let kind = (text["model_type"] as? String) ?? (json["model_type"] as? String) ?? ""
        switch kind {
        case "gemma3", "gemma3_text":
            return .gemma3(try NFKMLXGemma3Language.configuration(fromJSON: json))
        case "gemma4", "gemma4_text":
            return .gemma4(try NFKMLXGemmaLanguage.configuration(fromHuggingFace: url))
        case "gemma4_unified", "gemma4_unified_text":
            return .gemma4Unified(try NFKMLXGemmaLanguage.unifiedConfiguration(fromHuggingFace: url))
        default:
            throw NFKMLXError.unsupportedConfiguration(
                "an LTX-2 release conditions on a Gemma 3 or Gemma 4 text encoder; this one names \(kind)")
        }
    }

    /// The hidden states the connectors stack: the embedding and every layer.
    var stateCount: Int {
        switch self {
        case .gemma3(let c): return c.layerCount + 1
        case .gemma4(let c), .gemma4Unified(let c): return c.layerCount + 1
        }
    }

    /// The decoder's module key for a release tensor, or nil for a tensor outside it.
    func decoderKey(_ key: String) -> String? {
        switch self {
        case .gemma3: return NFKMLXGemma3Language.decoderName(of: key)
        case .gemma4, .gemma4Unified: return NFKMLXGemmaLanguage.decoderKey(of: key)
        }
    }

    /// The decoder's footprint at `dtype`, each layer a repeated layer.
    func footprint(directory: URL, converting dtype: DType?) throws -> NFKMLXStageFootprint {
        try NFKMLXLayerStream.footprint(directory: directory, precision: .checkpoint, converting: dtype,
                                        includes: { decoderKey($0) != nil },
                                        layerOf: { decoderKey($0).flatMap { NFKMLXLayerStream.layer(ofDecoderName: $0) }?.layer })
    }

    /// Builds and loads the decoder at `dtype` (the release's own type where nil), leaving the `streamed`
    /// layers in the release.
    func load(directory: URL, converting dtype: DType?, streamed: Set<Int>) throws -> NFKLTX2TextStage {
        let layerOf: (String) -> (layer: Int, name: String)? = { key in
            decoderKey(key).flatMap { NFKMLXLayerStream.layer(ofDecoderName: $0) }
        }
        func arrays(_ held: @escaping (String) -> String?) throws -> [(String, MLXArray)] {
            try dtype.map { try NFKMLXReleaseWeights.arrays(inDirectory: directory, converting: $0, remap: held) }
                ?? NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: .checkpoint, remap: held)
        }
        func install(_ net: Module, layers: [Module]) throws -> NFKMLXLayerStream? {
            let vacated = try NFKMLXLayerStream.vacating(streamed, of: layers, prefix: "layers.", directory: directory,
                                                         precision: .checkpoint, converting: dtype, layerOf: layerOf)
            let held: (String) -> String? = { key in
                guard let name = decoderKey(key) else { return nil }
                return NFKMLXLayerStream.layer(ofDecoderName: name).map { streamed.contains($0.layer) } == true ? nil : name
            }
            try NFKMLXWeights.apply(try arrays(held) + (vacated?.placeholders ?? []), to: net, verifyShapes: true)
            vacated?.stream.prime()
            return vacated?.stream
        }
        switch self {
        case .gemma3(let c):
            let net = NFKMLXGemma3Net(c)
            net.layerStream = try install(net, layers: net.layers)
            return NFKLTX2TextStage(states: { NFKMLXLTX2Pipeline.textStates(net, tokens: $0) },
                                    verify: { try net.verifyStream() })
        case .gemma4(let c):
            let net = NFKMLXGemmaNet(c)
            net.layerStream = try install(net, layers: net.layers)
            return NFKLTX2TextStage(states: { tokens in
                NFKLTX2TextEncoder.stacked(net.hiddenStates(MLXArray(tokens.map(Int32.init)).reshaped([1, -1])))
            }, verify: { try net.verifyStream() })
        case .gemma4Unified(let c):
            let net = NFKMLXGemma4UnifiedNet(c)
            net.layerStream = try install(net, layers: net.layers)
            return NFKLTX2TextStage(states: { tokens in
                NFKLTX2TextEncoder.stacked(net.hiddenStates(MLXArray(tokens.map(Int32.init)).reshaped([1, -1])))
            }, verify: { try net.verifyStream() })
        }
    }

    /// Per-layer states `[1, tokens, hidden]` stacked on a trailing axis and flattened hidden-major.
    static func stacked(_ states: [MLXArray]) -> MLXArray {
        let joined = MLX.stacked(states, axis: -1)
        return joined.reshaped([1, joined.dim(1), -1])
    }
}

/// The transformer, the decoders, and the vocoder, which run after the text stages.
final class NFKLTX2PipelineStage {
    let pipeline: NFKMLXLTX2Pipeline
    init(_ pipeline: NFKMLXLTX2Pipeline) { self.pipeline = pipeline }
}

/// LTX-2 text-to-video-and-audio, assembled from a diffusers release directory.
///
/// @discussion The release layout is `text_encoder/` (Gemma 3 12B for LTX-2.3, Gemma 4 12B for LTX-2.5),
/// `tokenizer/`, `connectors/`, `transformer/`, `vae/`, `audio_vae/`, `vocoder/`, and `scheduler/`.
/// ``clip(forPrompt:negativePrompt:frames:width:height:seed:onStep:)`` tokenizes the prompt, runs the text
/// encoder over its real tokens, fills the connectors' registers to 1024 tokens, denoises the joint video
/// and audio latents over the flow schedule with the reference's guidance (classifier-free,
/// spatio-temporal, and modality, with the rescale), and decodes both. The glue is measured against
/// diffusers' `LTX2Pipeline`.
///
/// The stages load one after another where the machine cannot hold them together: the text encoder, the
/// connectors, then the transformer with the decoders. Where a stage does not fit even alone,
/// ``NFKMLXResidency/automatic`` and ``NFKMLXResidency/streamed`` hold its first layers and read the rest
/// from the release on every pass, the transformer's blocks and the text encoder's layers alike.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXLTX2Generator)
public final class NFKMLXLTX2Generator: NSObject {

    /// The text length the connectors fill to: the reference pads and truncates prompts to 1024 tokens.
    static let maximumSequenceLength = 1024

    /// The release this generator was assembled from, read from its transformer.
    @objc public let variant: NFKMLXLTX2Variant

    /// The registry name of ``variant``.
    @objc public var modelName: String { variant.modelName }

    /// The registry name of `variant`, ``NFKMLXLTX2Variant/modelName`` for Objective-C. Introduced in InferKit
    /// 0.4.0.
    @objc(modelNameForVariant:)
    public static func modelName(for variant: NFKMLXLTX2Variant) -> String { variant.modelName }

    /// The repository the download factories read for `variant`, ``NFKMLXLTX2Variant/releaseRepo`` for
    /// Objective-C. Introduced in InferKit 0.4.0.
    @objc(releaseRepoForVariant:)
    public static func releaseRepo(for variant: NFKMLXLTX2Variant) -> String { variant.releaseRepo }

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKLTX2TextStage>
    private let connectorStage: NFKMLXStage<NFKMLXLTX2ConnectorsNet>
    private let pipelineStage: NFKMLXStage<NFKLTX2PipelineStage>
    private let tokenizer: NFKMLXGemmaTokenizer
    private let video: NFKMLXLTX2VideoVAEConfiguration
    private let audio: NFKMLXLTX2AudioVAEConfiguration
    private let layerCount: Int

    /// The denoising steps, 30 by the reference's default. Unread where ``sigmas`` is set.
    @objc public var steps: Int = 30
    /// The sigma ramp to sample from in place of `linspace(1, 1 / steps, steps)`, warped by the release's
    /// schedule. LTX-2.5's distilled transformer samples from ``distilledSigmas``, its default; nil
    /// elsewhere.
    @objc public var sigmas: [NSNumber]?
    /// The release's flow schedule.
    let schedule: NFKMLXFlowMatchConfiguration
    /// The video's classifier-free guidance scale, 3 by the reference's default.
    @objc public var guidance: Float = NFKMLXLTX2Guidance.video.textScale
    /// The audio's classifier-free guidance scale, 7 by the reference's default.
    @objc public var audioGuidance: Float = NFKMLXLTX2Guidance.audio.textScale
    /// The spatio-temporal guidance scale of both modalities, 1 by the reference's default; 0 skips the
    /// perturbed pass.
    @objc public var spatioTemporalGuidance: Float = 1
    /// The modality guidance scale of both modalities, 3 by the reference's default; 1 skips the isolated
    /// pass.
    @objc public var modalityGuidance: Float = 3
    /// The guidance rescale of both modalities, 0.7 by the reference's default; 0 turns it off.
    @objc public var guidanceRescale: Float = 0.7
    /// The frame rate the rotary scales time by and the clip is written at, 24 by the reference's default.
    @objc public var frameRate: Float = 24
    /// The blocks spatio-temporal guidance perturbs, block 28 by the reference's default.
    public var perturbedBlocks: Set<Int> = [28]

    /// Whether the stages stay loaded together between clips.
    @objc public var holdsStagesResident: Bool { staging.resident }

    /// The transformer blocks a streamed load reads from the release on every pass; zero where every block
    /// is held.
    @objc public private(set) var streamedBlockCount = 0

    /// The text encoder layers a streamed load reads from the release on every pass.
    @objc public private(set) var streamedTextLayerCount = 0

    init(variant: NFKMLXLTX2Variant, resident: Bool, tokenizer: NFKMLXGemmaTokenizer,
         video: NFKMLXLTX2VideoVAEConfiguration, audio: NFKMLXLTX2AudioVAEConfiguration, layerCount: Int,
         audioSampleRate: Int, schedule: NFKMLXFlowMatchConfiguration = .ltxVideoPipeline, loadText: @escaping () throws -> NFKLTX2TextStage,
         loadConnectors: @escaping () throws -> NFKMLXLTX2ConnectorsNet,
         loadPipeline: @escaping () throws -> NFKMLXLTX2Pipeline) {
        self.variant = variant
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage(loadText)
        connectorStage = NFKMLXStage(loadConnectors)
        pipelineStage = NFKMLXStage { NFKLTX2PipelineStage(try loadPipeline()) }
        self.tokenizer = tokenizer
        self.video = video
        self.audio = audio
        self.layerCount = layerCount
        self.audioSampleRate = audioSampleRate
        self.schedule = schedule
        super.init()
        perturbedBlocks = [min(28, layerCount - 1)]
        if variant == .ltx25 {
            // The distilled transformer LTX-2.5's `transformer/` holds samples its own sigmas unguided.
            sigmas = Self.distilledSigmas.map { NSNumber(value: $0) }
            (guidance, audioGuidance, spatioTemporalGuidance, modalityGuidance) = (1, 1, 0, 1)
        }
    }

    /// Loads every stage now, for a resident release.
    func loadResident() throws {
        try staging.use(textStage) { _ in }
        try staging.use(connectorStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    // MARK: Factories

    /// Assembles the model from a downloaded release directory, holding it as ``NFKMLXResidency/automatic``
    /// decides. Introduced in InferKit 0.4.0.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXLTX2Generator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    ///
    /// @discussion The transformer loads as published (bfloat16, about 38 GB for the 22B release), the
    /// connectors as published, the text encoder at bfloat16 and at float32 where the plan affords it, and
    /// the decoders and vocoder at float32. ``NFKMLXResidency/paged`` holds the release as
    /// ``NFKMLXResidency/staged`` does: it has no routed experts. Introduced in InferKit 0.4.0.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXLTX2Generator {
        try generator(directoryURL: directoryURL, residency: residency, options: NFKMLXLoadOptions())
    }

    /// ``generator(directoryURL:residency:)`` with the precision and quantization `options` name, planned
    /// against `budget` bytes.
    static func generator(directoryURL: URL, residency: NFKMLXResidency, options: NFKMLXLoadOptions,
                          budget: Int = NFKMLXResidencyBudget.current()) throws -> NFKMLXLTX2Generator {
        let directory = { (name: String) in directoryURL.appendingPathComponent(name) }
        let transformerConfiguration = try NFKMLXLTX2TransformerNet.configuration(
            fromHuggingFace: directory("transformer/config.json"))
        let variant = Self.variant(of: transformerConfiguration)
        let text = try NFKLTX2TextEncoder.reading(directory("text_encoder/config.json"))
        let connectorConfiguration = try NFKMLXLTX2Connectors.configuration(
            fromHuggingFace: directory("connectors/config.json"))
        guard connectorConfiguration.layerCount == text.stateCount else {
            throw NFKMLXError.unsupportedConfiguration(
                "the connectors read \(connectorConfiguration.layerCount) hidden states; the text encoder gives "
                + "\(text.stateCount)")
        }
        let videoConfiguration = try NFKMLXLTX2VideoVAE.configuration(fromHuggingFace: directory("vae/config.json"))
        let audioConfiguration = try NFKMLXLTX2AudioVAE.configuration(
            fromHuggingFace: directory("audio_vae/config.json"))
        let vocoderConfiguration = try NFKMLXLTX2Vocoder.configuration(
            fromHuggingFace: directory("vocoder/config.json"))
        let tokenizerDirectory = FileManager.default.fileExists(atPath: directory("tokenizer/tokenizer.json").path)
            ? directory("tokenizer") : directory("processor")
        guard let tokenizer = NFKMLXGemmaTokenizer(directoryURL: tokenizerDirectory) else {
            throw NFKMLXError.unsupportedConfiguration("the LTX-2 release has no readable tokenizer.json")
        }

        let precision: NFKMLXWeightPrecision = options.precision ?? .checkpoint
        let textType: DType? = { if case .gemma3 = text { return .bfloat16 } else { return nil } }()
        let textNarrow = try text.footprint(directory: directory("text_encoder"), converting: textType)
        let textWide = textType == nil ? nil : try text.footprint(directory: directory("text_encoder"), converting: .float32)
        let connectorBytes = try NFKMLXStageWeights.bytes(inDirectory: directory("connectors"), precision: .checkpoint)
        let transformer = try NFKMLXLTX2TransformerNet.footprint(directory: directory("transformer"), precision: precision)
        let decoderBytes = try ["vae", "audio_vae", "vocoder"].reduce(0) {
            $0 + (try NFKMLXStageWeights.bytes(inDirectory: directory($1), precision: .float32))
        }
        let textFootprint = NFKMLXStageFootprint(bytes: textNarrow.bytes, widenedBytes: textWide?.bytes,
                                                 layerBytes: textNarrow.layerBytes)
        let pipelineFootprint = NFKMLXStageFootprint(bytes: transformer.bytes + decoderBytes,
                                                     layerBytes: transformer.layerBytes)
        let plan = try NFKMLXResidencyBudget.plan(
            [textFootprint, NFKMLXStageFootprint(bytes: connectorBytes), pipelineFootprint],
            residency: residency, budget: budget)
        let textStreamed = plan.streams(0)
            ? Set(plan.heldLayers(0, of: textFootprint).upperBound ..< textNarrow.layerBytes.count) : []
        let streamed = plan.streams(2)
            ? Set(plan.heldLayers(2, of: pipelineFootprint).upperBound ..< transformer.layerBytes.count) : []
        let textLoadType: DType? = plan.widens(0) ? .float32 : textType
        if options.quantizationBits != nil, !streamed.isEmpty {
            throw NFKMLXError.unsupportedConfiguration(
                "\(variant.modelName) quantizes a transformer it holds whole; this budget streams "
                + "\(streamed.count) of its blocks, so load it unquantized or with a residency that holds it")
        }
        let quantization = options.quantizationBits.map { ($0, options.quantizationGroupSize) }

        let generator = NFKMLXLTX2Generator(
            variant: variant, resident: plan.holdsStagesResident, tokenizer: tokenizer, video: videoConfiguration,
            audio: audioConfiguration, layerCount: transformerConfiguration.numLayers,
            audioSampleRate: vocoderConfiguration.outputSampleRate,
            schedule: try schedule(fromHuggingFace: directory("scheduler/scheduler_config.json")), loadText: {
                try text.load(directory: directory("text_encoder"), converting: textLoadType, streamed: textStreamed)
            },
            loadConnectors: {
                let net = NFKMLXLTX2ConnectorsNet(connectorConfiguration)
                try NFKMLXLTX2Connectors.loadWeights(into: net, from: directory("connectors"), precision: .checkpoint)
                return net
            },
            loadPipeline: {
                let net = NFKMLXLTX2TransformerNet(transformerConfiguration)
                try NFKMLXLTX2TransformerNet.loadWeights(into: net, from: directory("transformer"), precision: precision,
                                                         streamed: streamed)
                if let (bits, groupSize) = quantization {
                    try NFKMLXQuantization.quantize(module: net, bits: bits, groupSize: groupSize)
                }
                let videoDecoder = NFKMLXLTX2VideoDecoderNet(videoConfiguration)
                let videoStatistics = try NFKMLXLTX2VideoVAE.loadWeights(into: videoDecoder, from: directory("vae"))
                let audioDecoder = NFKMLXLTX2AudioDecoderNet(audioConfiguration)
                let audioStatistics = try NFKMLXLTX2AudioVAE.loadWeights(into: audioDecoder, from: directory("audio_vae"))
                let vocoder = NFKMLXLTX2VocoderNet(vocoderConfiguration)
                try NFKMLXLTX2Vocoder.loadWeights(into: vocoder, from: directory("vocoder"))
                return NFKMLXLTX2Pipeline(transformer: net, videoDecoder: videoDecoder, videoStatistics: videoStatistics,
                                          audioDecoder: audioDecoder, audioStatistics: audioStatistics, vocoder: vocoder)
            })
        generator.streamedBlockCount = streamed.count
        generator.streamedTextLayerCount = textStreamed.count
        options.reportResidency(plan.holdsStagesResident ? .resident
                                    : (streamed.isEmpty && textStreamed.isEmpty ? .staged : .streamed))
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    /// The release a transformer configuration describes: LTX-2.5 adds the keyframe embedding and drops the
    /// video feed-forward's bias.
    static func variant(of configuration: NFKMLXLTX2Configuration) -> NFKMLXLTX2Variant {
        configuration.usesKeyframeEmbedding && !configuration.feedForwardBias ? .ltx25 : .ltx23
    }

    /// The sigmas LTX-2.5's distilled transformer samples from, the reference's `DISTILLED_SIGMA_VALUES`.
    public static let distilledSigmas: [Float] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]

    /// Reads `scheduler/scheduler_config.json`: dynamic shifting from its bounds, else the static `shift`,
    /// the terminal stretch where it names one, over the pipeline's `1 / steps` ramp.
    static func schedule(fromHuggingFace url: URL) throws -> NFKMLXFlowMatchConfiguration {
        var schedule = try NFKMLXSD3Release.schedule(fromHuggingFace: url)
        schedule.rampEndsAtStepFraction = true
        return schedule
    }

    /// The component folders a download fetches.
    static let releaseComponents = [
        NFKMLXReleaseComponent(required: ["model_index.json", "scheduler/scheduler_config.json"]),
        NFKMLXReleaseComponent(required: ["text_encoder/config.json"],
                               weights: ["text_encoder/model.safetensors", "text_encoder/model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["tokenizer/tokenizer.json"],
                               optional: ["tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json"]),
        NFKMLXReleaseComponent(required: ["connectors/config.json"],
                               weights: ["connectors/diffusion_pytorch_model.safetensors",
                                         "connectors/diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["transformer/config.json"],
                               weights: ["transformer/diffusion_pytorch_model.safetensors",
                                         "transformer/diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]),
        NFKMLXReleaseComponent(required: ["audio_vae/config.json", "audio_vae/diffusion_pytorch_model.safetensors"]),
        NFKMLXReleaseComponent(required: ["vocoder/config.json", "vocoder/diffusion_pytorch_model.safetensors"]),
    ]

    /// Downloads `variant`'s release (its ``NFKMLXLTX2Variant/releaseRepo`` when `repo` is nil) and assembles
    /// the model, holding it as `residency` says.
    ///
    /// @discussion LTX-2.3 is about 95 GB: the 48.8 GB float32 text encoder, the 38 GB transformer, and the
    /// 6.3 GB connectors; LTX-2.5 about 70 GB. LTX-2.5's repository is gated:
    /// its license is accepted once on the Hub, and the token is `NFKHFHub.defaultAccessToken`. A file
    /// already in the cache is not fetched again. The call blocks on the network, so run it off the main
    /// and render threads. Introduced in InferKit 0.4.0.
    @objc(generatorWithVariant:repo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(variant: NFKMLXLTX2Variant, repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXLTX2Generator {
        try NFKMLXBuildHolds.keeping {
            let directory = try NFKMLXReleaseDownload.directory(repo: repo ?? variant.releaseRepo, revision: revision,
                                                                cacheDirectoryURL: cacheDirectoryURL,
                                                                components: releaseComponents)
            let generator = try generator(directoryURL: directory, residency: residency)
            guard generator.variant == variant else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(repo ?? variant.releaseRepo) holds \(generator.variant.displayName), not \(variant.displayName)")
            }
            return generator
        }
    }

    /// The asynchronous form of ``generator(variant:repo:revision:cacheDirectoryURL:residency:)``. The handler
    /// runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(generatorWithVariant:repo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func generator(variant: NFKMLXLTX2Variant, repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency,
                                 completionHandler: @escaping (NFKMLXLTX2Generator?, Error?) -> Void) {
        NFKMLXReleaseDownload.asyncValue(completionHandler) {
            try generator(variant: variant, repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                          residency: residency)
        }
    }

    // MARK: Generation

    /// The prompt's token ids as the reference's tokenizer gives them: the text stripped, `<bos>` first,
    /// truncated to 1024. The padding the reference adds is never attended to, so it is not formed.
    func tokens(_ prompt: String) -> [Int] {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let bos = tokenizer.id(forToken: "<bos>") ?? 2
        return Array(([bos] + tokenizer.encode(text)).prefix(Self.maximumSequenceLength))
    }

    /// The guidance a modality samples with at the generator's settings.
    func guidance(textScale: Float) -> NFKMLXLTX2Guidance {
        NFKMLXLTX2Guidance(textScale: textScale, spatioTemporalScale: spatioTemporalGuidance,
                           modalityScale: modalityGuidance, rescale: guidanceRescale)
    }

    /// Generates a clip for `prompt`: its frames `[frames, height, width, 3]` in `[0, 1]` and its stereo
    /// waveform `[samples, 2]` at ``audioSampleRate``.
    ///
    /// @discussion `frames` rounds down to one more than a multiple of 8, and `width` and `height` down to
    /// multiples of 32, the autoencoder's compression, as the reference rounds them. The soundtrack covers
    /// the clip's duration at the frame rate. Both prompts encode from one load of the text encoder.
    public func clip(forPrompt prompt: String, negativePrompt: String? = nil, frames: Int = 121, width: Int = 768,
                     height: Int = 512, seed: UInt64 = 0,
                     onStep: ((Int) -> Void)? = nil) throws -> (video: MLXArray, audio: MLXArray) {
        let grid = NFKMLXLTX2Pipeline.grid(frames: frames, width: width, height: height, frameRate: frameRate,
                                           video: video, audio: audio)
        guard frames >= 1, grid.height > 0, grid.width > 0 else {
            throw NFKMLXError.unsupportedConfiguration("a clip is at least one frame of 32×32 pixels")
        }
        let videoGuidance = guidance(textScale: guidance)
        let audioGuidanceScales = guidance(textScale: audioGuidance)
        let guides = guidance > 1 || audioGuidance > 1
        let prompts = (guides ? [tokens(negativePrompt ?? "")] : []) + [tokens(prompt)]
        return try staging.exclusively {
            let states = try staging.with(textStage) { stage -> [MLXArray] in
                let states = prompts.map { stage.states($0) }
                eval(states)
                try stage.verify()
                return states
            }
            let streams = try staging.with(connectorStage) { connectors -> [MLXArray] in
                let pairs = states.map { connectors(validStates: $0, length: Self.maximumSequenceLength) }
                return [concatenated(pairs.map(\.video), axis: 0), concatenated(pairs.map(\.audio), axis: 0)]
            }
            let result = try staging.with(pipelineStage) { stage -> [MLXArray] in
                let pipeline = stage.pipeline
                let key = MLXRandom.key(seed)
                let (videoKey, audioKey) = MLXRandom.split(key: key)
                let videoNoise = MLXRandom.normal(
                    [1, grid.frames * grid.height * grid.width, pipeline.transformer.config.inChannels], key: videoKey)
                let audioNoise = MLXRandom.normal(
                    [1, grid.audioFrames, pipeline.transformer.config.audioInChannels], key: audioKey)
                let dtype = pipeline.transformer.projIn.weight.dtype
                let (videoLatents, audioLatents) = try pipeline.denoise(
                    video: videoNoise, audio: audioNoise, text: streams[0].asType(dtype),
                    audioText: streams[1].asType(dtype), frames: grid.frames, height: grid.height, width: grid.width,
                    audioFrames: grid.audioFrames, frameRate: frameRate, steps: steps, videoGuidance: videoGuidance,
                    audioGuidance: audioGuidanceScales, perturbedBlocks: perturbedBlocks, schedule: schedule,
                    sigmas: sigmas?.map(\.floatValue), onStep: onStep)
                let clip = MLX.clip(pipeline.decodeVideo(videoLatents, frames: grid.frames, height: grid.height,
                                                         width: grid.width)[0] / 2 + 0.5, min: 0, max: 1)
                eval(clip)
                return [clip, pipeline.decodeAudio(audioLatents)[0]]
            }
            return (result[0], result[1])
        }
    }

    /// The soundtrack's sample rate, the vocoder's output rate (48 kHz in both releases).
    @objc public let audioSampleRate: Int

    /// Generates a clip for `prompt` and returns its frames as `CGImage`s. Objective-C reads each element
    /// as `(__bridge CGImageRef)`, since an array of a Core Foundation type is not an Objective-C
    /// collection. The soundtrack is dropped; ``inferenceBackend()`` returns both. Introduced in InferKit
    /// 0.4.0.
    @objc(framesForPrompt:negativePrompt:frames:width:height:seed:error:)
    public func frames(forPrompt prompt: String, negativePrompt: String?, frames: Int, width: Int, height: Int,
                       seed: UInt64) throws -> [Any] {
        let clip = try clip(forPrompt: prompt, negativePrompt: negativePrompt, frames: frames, width: width,
                            height: height, seed: seed).video
        return try (0 ..< clip.dim(0)).map {
            try NFKMLXImageBridge.cgImage(from: clip[$0], options: NFKMLXImageOptions())
        }
    }
}
