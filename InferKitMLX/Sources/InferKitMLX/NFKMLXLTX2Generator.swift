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
    /// LTX-2.5's full (SFT) transformer, `transformer_full/` of the same repository: the distilled one's
    /// arrangement, sampled guided over the release's schedule with dynamic shifting and a terminal stretch
    /// to 0.1, the arrangement Lightricks' trainer fine-tunes.
    case ltx25Full = 2

    /// The registry name.
    public var modelName: String {
        switch self {
        case .ltx23: return "ltx-2.3"
        case .ltx25: return "ltx-2.5"
        case .ltx25Full: return "ltx-2.5-full"
        }
    }

    /// The Hugging Face repository the download factories read.
    public var releaseRepo: String {
        switch self {
        case .ltx23: return "diffusers/LTX-2.3-Diffusers"
        case .ltx25, .ltx25Full: return "Lightricks/LTX-2.5-Diffusers"
        }
    }

    /// The release's display name.
    public var displayName: String {
        switch self {
        case .ltx23: return "LTX-2.3"
        case .ltx25: return "LTX-2.5"
        case .ltx25Full: return "LTX-2.5 (full transformer)"
        }
    }

    /// The release folder the transformer loads from.
    public var transformerFolder: String { self == .ltx25Full ? "transformer_full" : "transformer" }
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

/// The two encoders a fine-tune's data preparation runs, with the statistics each normalizes by.
final class NFKLTX2EncoderStage {
    let video: NFKMLXLTX2VideoEncoderNet
    let videoStatistics: (mean: MLXArray, std: MLXArray)
    let audio: NFKMLXLTX2AudioEncoderNet
    let audioStatistics: (mean: MLXArray, std: MLXArray)

    init(directory: URL) throws {
        video = try NFKMLXLTX2VideoEncoderNet(try NFKMLXLTX2VideoVAE.encoderConfiguration(
            fromHuggingFace: directory.appendingPathComponent("vae/config.json")))
        videoStatistics = try NFKMLXLTX2VideoVAE.loadEncoderWeights(into: video,
                                                                    from: directory.appendingPathComponent("vae"))
        audio = NFKMLXLTX2AudioEncoderNet(try NFKMLXLTX2AudioVAE.configuration(
            fromHuggingFace: directory.appendingPathComponent("audio_vae/config.json")))
        audioStatistics = try NFKMLXLTX2AudioVAE.loadEncoderWeights(into: audio,
                                                                    from: directory.appendingPathComponent("audio_vae"))
    }
}

/// The LoRA adapters a generator places over its transformer each time the transformer loads.
final class NFKLTX2AdapterSet {
    var arrays: [String: MLXArray]?
    var rank = 0
    var alpha: Float = 0

    func apply(to net: NFKMLXLTX2TransformerNet) throws {
        guard let arrays else { return }
        try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha) { path, _ in NFKMLXLTX2Trainable.targets(path) }
        try NFKMLXLoRA.loadAdapters(arrays, into: net)
        net.train(false)
        eval(net)
    }
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

    let staging: NFKMLXStagedModel
    let textStage: NFKMLXStage<NFKLTX2TextStage>
    let connectorStage: NFKMLXStage<NFKMLXLTX2ConnectorsNet>
    private let pipelineStage: NFKMLXStage<NFKLTX2PipelineStage>
    private let adapters = NFKLTX2AdapterSet()
    private let encoderStage: NFKMLXStage<NFKLTX2EncoderStage>?
    private let durationStage: NFKMLXStage<NFKMLXLTX2DurationHeadNet>?
    private let enhancerStage: NFKMLXStage<NFKMLXLTX2PromptEnhancer>?

    /// Whether the release directory holds LTX-2.5's prompt enhancer (`prompt_enhancer/` and `processor/`).
    @objc public var canEnhancePrompts: Bool { enhancerStage != nil }

    /// Whether a clip's prompt is first rewritten by the prompt enhancer
    /// (``NFKMLXLTX2PromptEnhancer/enhancedPrompt(_:)``), off by the reference's default. Setting it on a
    /// release without the enhancer makes generation throw.
    @objc public var enhancesPrompts = false

    /// Whether the release ships a duration head (LTX-2.5), so a clip without a frame count takes the
    /// length its prompt describes.
    @objc public var predictsDuration: Bool { durationStage != nil }

    /// The window a predicted duration is clamped to, 1 to 20 seconds by the reference's default.
    @objc public var minimumSeconds: Double = 1
    @objc public var maximumSeconds: Double = 20
    private let tokenizer: NFKMLXGemmaTokenizer
    let video: NFKMLXLTX2VideoVAEConfiguration
    let audio: NFKMLXLTX2AudioVAEConfiguration
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

    /// The release's decode-side components and their switches (`NFKMLXLTX2GeneratorDecodePaths.swift`).
    var decodePaths = NFKLTX2DecodePaths(directory: nil)

    init(variant: NFKMLXLTX2Variant, resident: Bool, tokenizer: NFKMLXGemmaTokenizer,
         video: NFKMLXLTX2VideoVAEConfiguration, audio: NFKMLXLTX2AudioVAEConfiguration, layerCount: Int,
         audioSampleRate: Int, schedule: NFKMLXFlowMatchConfiguration = .ltxVideoPipeline, loadText: @escaping () throws -> NFKLTX2TextStage,
         loadConnectors: @escaping () throws -> NFKMLXLTX2ConnectorsNet,
         loadPipeline: @escaping () throws -> NFKMLXLTX2Pipeline,
         loadEncoders: (() throws -> NFKLTX2EncoderStage)? = nil,
         loadDurationHead: (() throws -> NFKMLXLTX2DurationHeadNet)? = nil,
         loadEnhancer: (() throws -> NFKMLXLTX2PromptEnhancer)? = nil) {
        encoderStage = loadEncoders.map { NFKMLXStage($0) }
        enhancerStage = loadEnhancer.map { NFKMLXStage($0) }
        durationStage = loadDurationHead.map { NFKMLXStage($0) }
        self.variant = variant
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage(loadText)
        connectorStage = NFKMLXStage(loadConnectors)
        let adapters = adapters
        pipelineStage = NFKMLXStage {
            let pipeline = try loadPipeline()
            try adapters.apply(to: pipeline.transformer)
            return NFKLTX2PipelineStage(pipeline)
        }
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

    /// Places LoRA adapters a fine-tune saved (``saveAdapters(of:to:)``) over the transformer, now where it
    /// is loaded and on every later load. The file's metadata names the rank and alpha. Applying a file
    /// again replaces the adapters; a file of another rank throws. Introduced in InferKit 0.4.0.
    ///
    /// - Throws: `NFKMLXError.loRANotApplicable` for a file without its rank and alpha or one naming a
    ///   layer the trainer does not adapt, and `NFKMLXError.unsupportedConfiguration` for a streamed
    ///   transformer, whose blocks are read from the release on every pass.
    @objc(applyAdaptersAtURL:error:)
    public func applyAdapters(at url: URL) throws {
        guard streamedBlockCount == 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(variant.modelName) streams \(streamedBlockCount) transformer blocks from the release, which "
                + "carries no adapters; load it with a residency that holds the transformer")
        }
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)
        guard let rank = metadata["rank"].flatMap(Int.init), let alpha = metadata["alpha"].flatMap(Float.init) else {
            throw NFKMLXError.loRANotApplicable("\(url.lastPathComponent) names no rank and alpha in its metadata")
        }
        adapters.arrays = arrays
        adapters.rank = rank
        adapters.alpha = alpha
        if pipelineStage.isHeld {
            try staging.use(pipelineStage) { try adapters.apply(to: $0.pipeline.transformer) }
        }
    }

    /// Runs `body` on the two encoders, loading their stage where the generator stages.
    func withEncoders(_ body: (NFKLTX2EncoderStage) throws -> [MLXArray]) throws -> [MLXArray] {
        guard let encoderStage else {
            throw NFKMLXError.unsupportedConfiguration("this generator was assembled without its release's encoders")
        }
        return try staging.with(encoderStage, body)
    }

    /// Runs `body` on the transformer, loading its stage where the generator stages.
    func withTransformer<T>(_ body: (NFKMLXLTX2TransformerNet) throws -> T) throws -> T {
        try staging.use(pipelineStage) { try body($0.pipeline.transformer) }
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

    /// Assembles `variant` from a downloaded release directory, holding it as `residency` says:
    /// ``NFKMLXLTX2Variant/ltx25Full`` reads `transformer_full/` and samples it guided over the dynamically
    /// shifted schedule; the other variants read `transformer/` and must be what it holds. Introduced in
    /// InferKit 0.4.0.
    @objc(generatorWithDirectoryURL:variant:residency:error:)
    public static func generator(directoryURL: URL, variant: NFKMLXLTX2Variant,
                                 residency: NFKMLXResidency) throws -> NFKMLXLTX2Generator {
        try generator(directoryURL: directoryURL, residency: residency, options: NFKMLXLoadOptions(), variant: variant)
    }

    /// ``generator(directoryURL:variant:residency:)`` with the load `options`, refusing a directory that
    /// holds another release than `variant`.
    static func generator(directoryURL: URL, residency: NFKMLXResidency, options: NFKMLXLoadOptions,
                          variant: NFKMLXLTX2Variant) throws -> NFKMLXLTX2Generator {
        let generator = try generator(directoryURL: directoryURL, residency: residency, options: options,
                                      transformer: variant)
        guard generator.variant == variant else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(directoryURL.lastPathComponent) holds \(generator.variant.displayName), not \(variant.displayName)")
        }
        return generator
    }

    /// ``generator(directoryURL:residency:)`` with the precision and quantization `options` name, planned
    /// against `budget` bytes.
    static func generator(directoryURL: URL, residency: NFKMLXResidency, options: NFKMLXLoadOptions,
                          budget: Int = NFKMLXResidencyBudget.current(),
                          transformer requested: NFKMLXLTX2Variant? = nil) throws -> NFKMLXLTX2Generator {
        let directory = { (name: String) in directoryURL.appendingPathComponent(name) }
        let full = requested == .ltx25Full
        let transformerFolder = full ? NFKMLXLTX2Variant.ltx25Full.transformerFolder : "transformer"
        let transformerConfiguration = try NFKMLXLTX2TransformerNet.configuration(
            fromHuggingFace: directory("\(transformerFolder)/config.json"))
        var variant = Self.variant(of: transformerConfiguration)
        if full {
            guard variant == .ltx25 else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(directoryURL.lastPathComponent)'s transformer_full is not in LTX-2.5's arrangement")
            }
            variant = .ltx25Full
        }
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
        let transformer = try NFKMLXLTX2TransformerNet.footprint(directory: directory(transformerFolder),
                                                                 precision: precision)
        let decoderBytes = try ["vae", "audio_vae", "vocoder"].reduce(0) {
            $0 + (try NFKMLXStageWeights.bytes(inDirectory: directory($1), precision: .float32))
        }
        let textFootprint = NFKMLXStageFootprint(bytes: textNarrow.bytes, widenedBytes: textWide?.bytes,
                                                 layerBytes: textNarrow.layerBytes)
        let decodePathBytes = try NFKLTX2DecodePaths.footprint(directory: directoryURL)
        let pipelineFootprint = NFKMLXStageFootprint(bytes: transformer.bytes + decoderBytes + decodePathBytes.upsamplers,
                                                     layerBytes: transformer.layerBytes)
        let plan = try NFKMLXResidencyBudget.plan(
            [textFootprint, NFKMLXStageFootprint(bytes: connectorBytes), pipelineFootprint]
                + (decodePathBytes.decoder > 0 ? [NFKMLXStageFootprint(bytes: decodePathBytes.decoder)] : []),
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
        let durationDirectory = directory("duration_head")
        let durationHead: (() throws -> NFKMLXLTX2DurationHeadNet)? =
            FileManager.default.fileExists(atPath: durationDirectory.appendingPathComponent("config.json").path) ? {
                let head = NFKMLXLTX2DurationHeadNet(try NFKMLXLTX2DurationHead.configuration(
                    fromHuggingFace: durationDirectory.appendingPathComponent("config.json")))
                try NFKMLXLTX2DurationHead.loadWeights(into: head, from: durationDirectory)
                return head
            } : nil

        let generator = NFKMLXLTX2Generator(
            variant: variant, resident: plan.holdsStagesResident, tokenizer: tokenizer, video: videoConfiguration,
            audio: audioConfiguration, layerCount: transformerConfiguration.numLayers,
            audioSampleRate: vocoderConfiguration.outputSampleRate,
            schedule: try schedule(fromHuggingFace: directory("scheduler/scheduler_config.json"), fullTransformer: full),
            loadText: {
                try text.load(directory: directory("text_encoder"), converting: textLoadType, streamed: textStreamed)
            },
            loadConnectors: {
                let net = NFKMLXLTX2ConnectorsNet(connectorConfiguration)
                try NFKMLXLTX2Connectors.loadWeights(into: net, from: directory("connectors"), precision: .checkpoint)
                return net
            },
            loadPipeline: {
                let net = NFKMLXLTX2TransformerNet(transformerConfiguration)
                try NFKMLXLTX2TransformerNet.loadWeights(into: net, from: directory(transformerFolder), precision: precision,
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
            },
            loadEncoders: { try NFKLTX2EncoderStage(directory: directoryURL) },
            loadDurationHead: durationHead,
            loadEnhancer: ["prompt_enhancer/config.json", "processor/tokenizer.json"].allSatisfy {
                FileManager.default.fileExists(atPath: directory($0).path)
            } ? { try NFKMLXLTX2PromptEnhancer.enhancer(directoryURL: directoryURL) } : nil)
        generator.decodePaths = NFKLTX2DecodePaths(directory: directoryURL)
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
    /// the terminal stretch where it names one, over the pipeline's `1 / steps` ramp. `fullTransformer` turns
    /// dynamic shifting on from the configuration's `base_shift` with a terminal stretch to 0.1, the override
    /// LTX-2.5's `transformer_full/` samples with over the distilled transformer's static scheduler.
    static func schedule(fromHuggingFace url: URL, fullTransformer: Bool = false) throws -> NFKMLXFlowMatchConfiguration {
        var schedule = try NFKMLXSD3Release.schedule(fromHuggingFace: url)
        schedule.rampEndsAtStepFraction = true
        if fullTransformer {
            schedule.baseShift = (try NFKMLXWanRelease.json(url)["base_shift"] as? NSNumber)?.floatValue ?? 0.95
            schedule.useDynamicShifting = true
            schedule.shiftTerminal = 0.1
        }
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
        transformerComponent("transformer"),
        NFKMLXReleaseComponent(required: ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]),
        NFKMLXReleaseComponent(required: ["audio_vae/config.json", "audio_vae/diffusion_pytorch_model.safetensors"]),
        NFKMLXReleaseComponent(required: ["vocoder/config.json", "vocoder/diffusion_pytorch_model.safetensors"]),
    ]

    static func transformerComponent(_ folder: String) -> NFKMLXReleaseComponent {
        NFKMLXReleaseComponent(required: ["\(folder)/config.json"],
                               weights: ["\(folder)/diffusion_pytorch_model.safetensors",
                                         "\(folder)/diffusion_pytorch_model.safetensors.index.json"])
    }

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
                                                                components: releaseComponents(for: variant))
            return try generator(directoryURL: directory, variant: variant, residency: residency)
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
    /// multiples of 32, the autoencoder's compression, as the reference rounds them. Where `frames` is nil, a
    /// release with a duration head (``predictsDuration``) takes the length the prompt describes, clamped to
    /// ``minimumSeconds`` and ``maximumSeconds`` at ``frameRate``; another release takes 121 frames. The
    /// soundtrack covers the clip's duration at the frame rate. Both prompts encode from one load of the text
    /// encoder.
    public func clip(forPrompt prompt: String, negativePrompt: String? = nil, frames requestedFrames: Int? = nil,
                     width: Int = 768, height: Int = 512, seed: UInt64 = 0,
                     onStep: ((Int) -> Void)? = nil) throws -> (video: MLXArray, audio: MLXArray) {
        guard requestedFrames.map({ $0 >= 1 }) ?? true else {
            throw NFKMLXError.unsupportedConfiguration("a clip is at least one frame")
        }
        let videoGuidance = guidance(textScale: guidance)
        let audioGuidanceScales = guidance(textScale: audioGuidance)
        let guides = guidance > 1 || audioGuidance > 1
        let prompt = enhancesPrompts ? try enhancedPrompt(prompt) : prompt
        let prompts = (guides ? [tokens(negativePrompt ?? "")] : []) + [tokens(prompt)]
        return try staging.exclusively {
            try decodePaths.verifyAvailable()
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
            // The prompt's own streams are the last row; a guided batch puts the negative prompt first.
            let frames = try requestedFrames ?? predictedFrames(video: streams[0][(-1)...], audio: streams[1][(-1)...])
            let result = try staging.with(pipelineStage) { stage -> [MLXArray] in
                let pipeline = stage.pipeline
                let (latent, audioLatents) = try sampleLatents(
                    pipeline, streams: streams, frames: frames, width: width, height: height, seed: seed,
                    videoGuidance: videoGuidance, audioGuidance: audioGuidanceScales, onStep: onStep)
                let waveform = pipeline.decodeAudio(audioLatents)[0]
                if decodesWithDiffusionDecoder { return [latent, waveform] }
                let clip = MLX.clip(pipeline.decodeLatent(latent)[0] / 2 + 0.5, min: 0, max: 1)
                eval(clip)
                return [clip, waveform]
            }
            let clip = decodesWithDiffusionDecoder ? try diffusionDecodedClip(result[0], seed: seed) : result[0]
            return (clip, result[1])
        }
    }

    /// The frame count the duration head predicts from a prompt's connector streams, or 121 for a release
    /// without one.
    func predictedFrames(video: MLXArray, audio: MLXArray) throws -> Int {
        guard let durationStage else { return 121 }
        let seconds = try staging.with(durationStage) { head in head(video: video, audio: audio) }
        return NFKMLXLTX2DurationHeadNet.frames(
            forSeconds: Double(seconds.asType(.float32).item(Float.self)), frameRate: Double(frameRate),
            compression: self.video.temporalCompression, minimumSeconds: minimumSeconds,
            maximumSeconds: maximumSeconds)
    }

    /// The frame count a clip of `prompt` takes where none is given: the duration head's prediction for the
    /// prompt, clamped to ``minimumSeconds`` and ``maximumSeconds`` and snapped to the autoencoder's grid, or
    /// 121 for a release without a head. Introduced in InferKit 0.4.0.
    @objc(predictedFrameCountForPrompt:error:)
    public func predictedFrameCount(forPrompt prompt: String) throws -> NSNumber {
        let tokens = tokens(prompt)
        return NSNumber(value: try staging.exclusively { () throws -> Int in
            let text = try connectorStates(tokens)
            return try predictedFrames(video: text.video, audio: text.audio)
        })
    }

    /// `prompt` rewritten by the release's prompt enhancer, which loads for the call where the generator
    /// stages. Introduced in InferKit 0.4.0.
    @objc(enhancedPrompt:error:)
    public func enhancedPrompt(_ prompt: String) throws -> String {
        guard let enhancerStage else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(variant.modelName)'s release directory holds no prompt_enhancer/ and processor/")
        }
        var enhanced = ""
        _ = try staging.with(enhancerStage) { enhancer -> MLXArray in
            enhanced = try enhancer.enhancedPrompt(prompt)
            return MLXArray(0)
        }
        return enhanced
    }

    /// The soundtrack's sample rate, the vocoder's output rate (48 kHz in both releases).
    @objc public let audioSampleRate: Int

    /// Generates a clip for `prompt` and returns its frames as `CGImage`s. Objective-C reads each element
    /// as `(__bridge CGImageRef)`, since an array of a Core Foundation type is not an Objective-C
    /// collection. The soundtrack is dropped; ``inferenceBackend()`` returns both. Introduced in InferKit
    /// 0.4.0.
    ///
    /// A `frames` of 0 takes the length ``clip(forPrompt:negativePrompt:frames:width:height:seed:onStep:)``
    /// gives a nil one.
    @objc(framesForPrompt:negativePrompt:frames:width:height:seed:error:)
    public func frames(forPrompt prompt: String, negativePrompt: String?, frames: Int, width: Int, height: Int,
                       seed: UInt64) throws -> [Any] {
        let clip = try clip(forPrompt: prompt, negativePrompt: negativePrompt, frames: frames == 0 ? nil : frames,
                            width: width, height: height, seed: seed).video
        return try (0 ..< clip.dim(0)).map {
            try NFKMLXImageBridge.cgImage(from: clip[$0], options: NFKMLXImageOptions())
        }
    }
}
