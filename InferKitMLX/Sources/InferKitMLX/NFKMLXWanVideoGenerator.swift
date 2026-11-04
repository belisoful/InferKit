//
//  NFKMLXWanVideoGenerator.swift
//  InferKitMLX
//
//  Wan text-to-video and image-to-video from a downloaded diffusers release directory, its stages held as
//  an `NFKMLXResidency` says: umT5-XXL runs once per clip and the transformer and autoencoder after it. A
//  Wan 2.2 A14B release holds two transformers, a high-noise and a low-noise expert, each a stage of its
//  own, so a machine that cannot hold both loads the second when the schedule reaches it.
//  The file also carries the release loaders the three models lacked: the transformer's and the
//  autoencoder's geometry from their configs, and their weights in the module's layout.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// The two transformers of a Wan 2.2 A14B release.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXWanExpert)
public enum NFKMLXWanExpert: Int, Sendable {
    /// `transformer/`: the steps whose timestep is at least the release's boundary, and the only
    /// transformer of a single-transformer release.
    case highNoise
    /// `transformer_2/`: the steps below the boundary.
    case lowNoise

    /// The release folder the expert loads from.
    var folder: String { self == .highNoise ? "transformer" : "transformer_2" }
}

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
            eps: (json["eps"] as? NSNumber)?.floatValue ?? base.eps,
            outChannels: json["out_channels"] as? Int)
    }

    /// The timestep fraction at which an A14B release hands the schedule to its low-noise expert
    /// (`boundary_ratio` in `model_index.json`: 0.875 for T2V-A14B, 0.9 for I2V-A14B), or nil for a release
    /// with one transformer.
    static func boundaryRatio(ofDirectory directoryURL: URL) throws -> Float? {
        let url = directoryURL.appendingPathComponent("model_index.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let ratio = (try json(url)["boundary_ratio"] as? NSNumber)?.floatValue
        let hasSecond = FileManager.default.fileExists(
            atPath: directoryURL.appendingPathComponent("transformer_2/config.json").path)
        guard let ratio, hasSecond else { return nil }
        return ratio
    }

    /// Loads a release's `transformer/` weights. The module keys are the release's; the patchify
    /// `Conv3d` weight is stored `[out, in, t, h, w]` and loads as MLX's `[out, t, h, w, in]`.
    static func loadTransformer(into net: NFKMLXWanTransformerNet, fromDirectory directory: URL,
                                precision: NFKMLXWeightPrecision) throws {
        let arrays = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision,
                                                                 transform: adaptedTransformer)
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
    }

    /// A release transformer tensor in the module's layout: the patchify `Conv3d` weight moves from
    /// `[out, in, t, h, w]` to `[out, t, h, w, in]`, and every other tensor loads as stored.
    static func adaptedTransformer(key: String, value: MLXArray) -> (String, MLXArray) {
        (key, value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value)
    }

    /// Whether a transformer key names a parameter diffusers keeps float32 under a half-precision load
    /// (`WanTransformer3DModel._keep_in_fp32_modules`): one whose dotted path has a `time_embedder`,
    /// `scale_shift_table`, `norm1`, `norm2`, or `norm3` component.
    static func keepsFloat32(_ key: String) -> Bool {
        key.split(separator: ".").contains { float32Modules.contains(String($0)) }
    }

    private static let float32Modules: Set<String> = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]

    /// Loads a release's `transformer/` weights as diffusers loads them at a half precision: every floating
    /// tensor converted to `dtype` as it is read, so the stored float32 never sits whole beside the converted
    /// copy, except the parameters ``keepsFloat32(_:)`` names, which load at float32. `remap` renames a key,
    /// or drops it with nil.
    static func loadTransformer(into net: NFKMLXWanTransformerNet, fromDirectory directory: URL,
                                converting dtype: DType, remap: (String) -> String? = { $0 }) throws {
        let converted = try NFKMLXReleaseWeights.arrays(inDirectory: directory, converting: dtype) { key in
            keepsFloat32(key) ? nil : remap(key)
        }
        let kept = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: .float32,
                                                               remap: { keepsFloat32($0) ? remap($0) : nil })
        try NFKMLXWeights.apply((converted + kept).map(adaptedTransformer), to: net, verifyShapes: true)
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
        let arrays = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision,
                                                                 transform: adaptedVAE)
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
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

/// Wan text-to-video and image-to-video, assembled from a diffusers release directory.
///
/// @discussion The release layout is `text_encoder/` (umT5-XXL), `tokenizer/` (its SentencePiece model),
/// `transformer/`, `vae/`, and `scheduler/`, and for a Wan 2.2 A14B release `transformer_2/` with the
/// boundary in `model_index.json`. The factories read Wan 2.1 T2V (1.3B and 14B, the Wan 2.1
/// autoencoder), Wan 2.2 TI2V-5B (the Wan 2.2 autoencoder) for text to video, and Wan 2.2 T2V-A14B and
/// I2V-A14B (two experts, the Wan 2.1 autoencoder).
/// ``video(forPrompt:negativePrompt:frames:width:height:seed:)`` encodes the prompt through the masked
/// umT5, denoises over the UniPC flow schedule with classifier-free guidance, and decodes;
/// ``video(forPrompt:image:lastImage:negativePrompt:frames:width:height:seed:)`` conditions I2V-A14B on a
/// first frame, and a last one where given. The glue is measured against diffusers' `WanPipeline` and
/// `WanImageToVideoPipeline`.
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
    /// An A14B release's experts, high-noise first; empty where the pipeline holds the one transformer.
    private let expertStages: [NFKMLXStage<NFKMLXWanTransformerNet>]
    private let boundaryRatio: Float?
    private let segmenter: NFKMLXSentencePieceSegmenter
    private let compression: (spatial: Int, temporal: Int)
    private let sequenceLength: Int
    /// Each expert's adapters, high-noise first.
    private let adapters: [NFKWanAdapterList]
    /// Whether a transformer merges an adapter only as it loads: a packed or streamed one.
    private let mergesOnLoad: Bool

    /// The denoising steps, 50 by the reference's default.
    @objc public var steps: Int = 50
    /// The classifier-free guidance scale, 5 by the reference's default. Above 1 the clip guides against
    /// the negative prompt, or an empty one where none is given.
    @objc public var guidance: Float = 5

    /// The low-noise expert's guidance scale (the reference's `guidance_scale_2`); nil takes ``guidance``.
    /// Guidance applies on both experts only where ``guidance`` is above 1, as the reference decides.
    /// Introduced in InferKit 0.4.0.
    public var lowNoiseGuidance: Float?

    /// ``lowNoiseGuidance`` for Objective-C. Introduced in InferKit 0.4.0.
    @objc(lowNoiseGuidance) public var lowNoiseGuidanceNumber: NSNumber? {
        get { lowNoiseGuidance.map { NSNumber(value: $0) } }
        set { lowNoiseGuidance = newValue?.floatValue }
    }

    /// Whether the release holds two experts (Wan 2.2 A14B). Introduced in InferKit 0.4.0.
    @objc public var hasExperts: Bool { !expertStages.isEmpty }

    /// Whether the release conditions on an image (Wan 2.2 I2V-A14B), which
    /// ``video(forPrompt:image:lastImage:negativePrompt:frames:width:height:seed:)`` then requires.
    /// Introduced in InferKit 0.4.0.
    @objc public let isImageToVideo: Bool

    /// The bits the transformers' block weights are packed to, or 0. Introduced in InferKit 0.4.0.
    @objc public internal(set) var quantizationBits = 0

    /// How many transformer blocks are read from the release on every pass, across the experts.
    /// Introduced in InferKit 0.4.0.
    @objc public internal(set) var streamedLayerCount = 0

    /// Whether the text encoder and the transformer stay loaded together between clips.
    @objc public var holdsStagesResident: Bool { staging.resident }

    /// Whether umT5 is released once a clip's prompts are encoded, and loaded again for the next clip.
    /// Off by default.
    ///
    /// @discussion A resident generator holds umT5 (10.8 GB at bfloat16) through denoising and decoding so
    /// the next clip reuses it. Releasing it lowers MLX's working memory for each clip by that much and
    /// costs a umT5 load per clip. The transformer and the autoencoder stay as the residency plan holds
    /// them.
    ///
    /// Pair it with `NFKMLXGPU.applyStandingLimits()`. Under MLX's default cache limit the freed buffers
    /// return to MLX's cache, which trims only near the recommended working set, so the process footprint
    /// barely falls. Measured on the Wan 2.1 T2V 1.3B release at 17×480×832 on a 32 GB Mac:
    /// - neither → a 24.9 GB footprint peak;
    /// - the release alone → 24.4 GB;
    /// - the standing limits alone → 23.9 GB;
    /// - both → 17.8 GB, with 6.4 GB held between clips.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc public var releasesTextEncoderAfterEncoding = false

    /// Whether umT5 runs at float32 rather than at bfloat16. It does where the residency plan affords
    /// the float32 encoder.
    @objc public internal(set) var encodesInFloat32 = true

    /// The frame rate the release is trained at: 24 for Wan 2.2 TI2V-5B, 16 for Wan 2.1. The factories read
    /// it from the release's `vae/config.json`, where TI2V-5B's autoencoder is the residual one. Introduced in
    /// InferKit 0.4.0.
    @objc public let framesPerSecond: Double

    init(resident: Bool, segmenter: NFKMLXSentencePieceSegmenter, compression: (spatial: Int, temporal: Int),
         framesPerSecond: Double = 16,
         sequenceLength: Int = NFKMLXWanVideoGenerator.maximumSequenceLength,
         adapters: [NFKWanAdapterList] = [NFKWanAdapterList()],
         imageToVideo: Bool = false, boundaryRatio: Float? = nil, mergesOnLoad: Bool = false,
         loadTextEncoder: @escaping () throws -> NFKMLXT5EncoderNet,
         loadPipeline: @escaping () throws -> NFKMLXWanPipeline,
         loadExperts: [() throws -> NFKMLXWanTransformerNet] = []) {
        staging = NFKMLXStagedModel(resident: resident)
        self.adapters = adapters
        textStage = NFKMLXStage { NFKWanTextStage(try loadTextEncoder()) }
        pipelineStage = NFKMLXStage(loadPipeline)
        expertStages = loadExperts.map { NFKMLXStage($0) }
        self.boundaryRatio = boundaryRatio
        self.isImageToVideo = imageToVideo
        self.mergesOnLoad = mergesOnLoad
        self.segmenter = segmenter
        self.compression = compression
        self.framesPerSecond = framesPerSecond
        self.sequenceLength = sequenceLength
        super.init()
    }

    /// Loads both stages now, for a resident release.
    func loadResident() throws {
        try staging.use(textStage) { _ in }
        // umT5's load leaves its converted-away groups in MLX's buffer cache, which the pipeline's load
        // would otherwise sit on top of.
        NFKMLXGPU.clearCache()
        try staging.use(pipelineStage) { _ in }
        for stage in expertStages {
            try staging.use(stage) { _ in }
        }
    }

    var isHoldingTextEncoder: Bool { staging.exclusively { textStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }
    func isHoldingExpert(_ expert: NFKMLXWanExpert) -> Bool {
        staging.exclusively { expertStages.indices.contains(expert.rawValue) && expertStages[expert.rawValue].isHeld }
    }

    /// Runs `body` with the text stage.
    func usingText<T>(_ body: (NFKWanTextStage) throws -> T) throws -> T {
        try staging.use(textStage, body)
    }

    /// Runs `body` with the pipeline (the autoencoder, and a single-transformer release's transformer).
    func usingPipeline<T>(_ body: (NFKMLXWanPipeline) throws -> T) throws -> T {
        try staging.use(pipelineStage, body)
    }

    /// Runs `body` with `expert`'s transformer: the pipeline's own for a single-transformer release.
    private func withExpert<T>(_ expert: NFKMLXWanExpert, _ body: (NFKMLXWanTransformerNet) throws -> T) throws -> T {
        guard !expertStages.isEmpty else {
            return try staging.use(pipelineStage) { pipeline in
                guard let transformer = pipeline.transformer else {
                    throw NFKMLXError.unsupportedConfiguration("the pipeline holds no transformer")
                }
                return try body(transformer)
            }
        }
        return try staging.use(expertStages[expert.rawValue], body)
    }

    /// The autoencoder's compression: `2^downsamplings · patch` spatially and `2^temporal downsamplings`
    /// in time.
    static func compression(of vae: NFKMLXWanVAEConfiguration) -> (spatial: Int, temporal: Int) {
        ((1 << vae.temporalDownsample.count) * vae.patchSize, 1 << vae.temporalDownsample.filter { $0 }.count)
    }

    /// The frame rate a release is trained at, told by its autoencoder: Wan 2.2's residual autoencoder
    /// belongs to TI2V-5B (24 frames per second), and Wan 2.1's to the 16 frames-per-second releases.
    static func framesPerSecond(of vae: NFKMLXWanVAEConfiguration) -> Double {
        vae.isResidual ? 24 : 16
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
    /// The encoder runs at float32 where the plan affords it (beside the transformer when resident,
    /// on its own when staged) and at bfloat16, the reference pipeline's precision, otherwise; the
    /// transformer and the autoencoder load as stored. A 14B transformer is 28 GB at bfloat16 and the
    /// A14B releases store each expert at float32 (57 GB); ``generator(directoryURL:residency:precision:quantizationBits:)``
    /// packs them. A machine that cannot hold the stages together stages under
    /// ``NFKMLXResidency/automatic``. The release has no routed experts, so ``NFKMLXResidency/paged``
    /// holds it as ``NFKMLXResidency/staged`` does.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXWanVideoGenerator {
        try generator(directoryURL: directoryURL, residency: residency, transformerType: nil)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says, the
    /// transformers' weights at `precision`, and their block weights packed to `quantizationBits`.
    ///
    /// @discussion A `quantizationBits` of 2 to 8 packs every linear layer inside a transformer's blocks as
    /// each block loads, so the stored form is never held whole (4 bits holds a 14B expert in about
    /// 8 GB); 0 packs nothing. Where an unpacked transformer and the autoencoder do not fit, the plan holds
    /// the first blocks that fit and reads the others from the release on every pass
    /// (``streamedLayerCount``). An A14B release's experts are stages of their own: a machine that cannot
    /// hold both loads the low-noise expert when the schedule reaches it. Introduced in InferKit 0.4.0.
    @objc(generatorWithDirectoryURL:residency:precision:quantizationBits:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                                 quantizationBits: Int) throws -> NFKMLXWanVideoGenerator {
        try generator(directoryURL: directoryURL, residency: residency, transformerType: nil, precision: precision,
                      bits: quantizationBits == 0 ? nil : quantizationBits)
    }

    /// Assembles the model as ``generator(directoryURL:residency:precision:quantizationBits:)`` does, with the
    /// transformers' floating weights converted to `transformerType` as they load (held whole), or held at
    /// `precision` where it is nil. `streamedBlocks`, where given, replaces the blocks the plan streams in
    /// each transformer.
    static func generator(directoryURL: URL, residency: NFKMLXResidency, transformerType: DType?,
                          precision: NFKMLXWeightPrecision = .checkpoint, bits: Int? = nil,
                          budget: Int? = nil, groupSize: Int = NFKMLXLoadOptions.defaultQuantizationGroupSize,
                          streamedBlocks: Set<Int>? = nil) throws -> NFKMLXWanVideoGenerator {
        guard transformerType == nil || bits == nil else {
            throw NFKMLXError.unsupportedConfiguration("a transformer is converted to one type or packed, not both")
        }
        let textDirectory = directoryURL.appendingPathComponent("text_encoder")
        let vaeDirectory = directoryURL.appendingPathComponent("vae")
        let segmenter = NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: directoryURL.appendingPathComponent("tokenizer/spiece.model")))
        let textConfiguration = try NFKMLXT5Encoder.configuration(
            fromHuggingFace: textDirectory.appendingPathComponent("config.json"))
        let boundaryRatio = try NFKMLXWanRelease.boundaryRatio(ofDirectory: directoryURL)
        let experts: [NFKMLXWanExpert] = boundaryRatio == nil ? [.highNoise] : [.highNoise, .lowNoise]
        let transformerDirectories = experts.map { directoryURL.appendingPathComponent($0.folder) }
        let configurations = try transformerDirectories.map {
            try NFKMLXWanRelease.transformerConfiguration(fromHuggingFace: $0.appendingPathComponent("config.json"))
        }
        let vae = try NFKMLXWanRelease.vaeConfiguration(fromHuggingFace: vaeDirectory.appendingPathComponent("config.json"))
        let schedule = try NFKMLXWanRelease.schedule(
            fromHuggingFace: directoryURL.appendingPathComponent("scheduler/scheduler_config.json"))
        let imageToVideo = configurations[0].inChannels != configurations[0].outChannels

        // umT5-XXL (stored at float32, 22.7 GB) takes float32 as its wider precision over the bfloat16
        // the reference pipeline runs it at.
        let text = NFKMLXStageFootprint(
            bytes: try NFKMLXStageWeights.bytes(inDirectory: textDirectory, holding: .bfloat16),
            widenedBytes: try NFKMLXStageWeights.bytes(inDirectory: textDirectory, holding: .float32))
        let vaeBytes = try NFKMLXStageWeights.bytes(inDirectory: vaeDirectory, precision: .float32)
        let transformers = try transformerDirectories.map {
            try transformerFootprint($0, holding: transformerType, precision: precision, bits: bits, groupSize: groupSize)
        }
        // A single transformer shares its stage with the autoencoder; an A14B release's experts are stages of
        // their own, after the autoencoder's.
        let stages = boundaryRatio == nil
            ? [text, NFKMLXStageFootprint(bytes: transformers[0].bytes + vaeBytes, layerBytes: transformers[0].layerBytes)]
            : [text, NFKMLXStageFootprint(bytes: vaeBytes)] + transformers
        let plan = try NFKMLXResidencyBudget.plan(stages, residency: residency,
                                                  budget: budget ?? NFKMLXResidencyBudget.current())
        let firstExpertStage = boundaryRatio == nil ? 1 : 2
        let streamed = experts.indices.map { index -> Set<Int> in
            if let streamedBlocks { return streamedBlocks }
            let stage = firstExpertStage + index
            guard plan.streams(stage) else { return [] }
            return Set(plan.heldLayers(stage, of: stages[stage]).upperBound ..< transformers[index].layerBytes.count)
        }
        let textInFloat32 = plan.widens(0)
        let adapters = experts.map { _ in NFKWanAdapterList() }
        func loadExpert(_ index: Int) throws -> NFKMLXWanTransformerNet {
            try loadTransformer(configurations[index], directory: transformerDirectories[index],
                                transformerType: transformerType, precision: precision, bits: bits,
                                groupSize: groupSize, streamed: streamed[index],
                                adapters: try adapters[index].deltas(naming: NFKMLXWanLoRA.textToVideoPath(original:)))
        }
        func loadAutoencoder() throws -> NFKMLXWanVideoVAENet {
            let autoencoder = NFKMLXWanVideoVAENet(vae.configuration)
            try NFKMLXWanRelease.loadVAE(into: autoencoder, fromDirectory: vaeDirectory)
            return autoencoder
        }
        let generator = NFKMLXWanVideoGenerator(
            resident: plan.holdsStagesResident, segmenter: segmenter,
            compression: compression(of: vae.configuration),
            framesPerSecond: framesPerSecond(of: vae.configuration), adapters: adapters,
            imageToVideo: imageToVideo, boundaryRatio: boundaryRatio,
            mergesOnLoad: bits != nil || streamed.contains { !$0.isEmpty },
            loadTextEncoder: {
                let encoder = NFKMLXT5Encoder.makeNet(textConfiguration)
                try NFKMLXT5Encoder.loadWeights(into: encoder, from: textDirectory,
                                                dtype: textInFloat32 ? nil : .bfloat16)
                return encoder
            },
            loadPipeline: {
                guard boundaryRatio == nil else {
                    return NFKMLXWanPipeline(vae: try loadAutoencoder(), latentsMean: vae.mean, latentsStd: vae.std,
                                             schedule: schedule)
                }
                let transformer = try loadExpert(0)
                return NFKMLXWanPipeline(transformer: transformer, vae: try loadAutoencoder(), latentsMean: vae.mean,
                                         latentsStd: vae.std, schedule: schedule)
            },
            loadExperts: boundaryRatio == nil ? [] : experts.indices.map { index in { try loadExpert(index) } })
        generator.encodesInFloat32 = textInFloat32
        generator.quantizationBits = bits ?? 0
        generator.streamedLayerCount = streamed.reduce(0) { $0 + $1.count }
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    /// Loads one transformer: packed or streamed block by block, converted to `transformerType` whole, or
    /// held at `precision`, with `adapters` merged in order.
    static func loadTransformer(_ configuration: NFKMLXWanConfiguration, directory: URL, transformerType: DType?,
                                precision: NFKMLXWeightPrecision, bits: Int?, groupSize: Int, streamed: Set<Int>,
                                adapters: [[String: NFKMLXWanLoRA.Delta]]) throws -> NFKMLXWanTransformerNet {
        let transformer = NFKMLXWanTransformerNet(configuration)
        if bits != nil || !streamed.isEmpty {
            transformer.layerStream = try NFKMLXWanBlockedLoad.load(
                into: transformer, blocks: transformer.blocks, fromDirectory: directory, precision: precision,
                streamed: streamed, bits: bits, groupSize: groupSize, adapters: adapters)
            return transformer
        }
        if let transformerType {
            try NFKMLXWanRelease.loadTransformer(into: transformer, fromDirectory: directory, converting: transformerType)
        } else {
            try NFKMLXWanRelease.loadTransformer(into: transformer, fromDirectory: directory, precision: precision)
        }
        for deltas in adapters {
            try NFKMLXWanLoRA.apply(deltas, to: transformer)
        }
        return transformer
    }

    /// What one transformer holds: packed, converted to `dtype` (held whole), or at `precision` with each
    /// block counted as a layer the plan may stream.
    private static func transformerFootprint(_ directory: URL, holding dtype: DType?, precision: NFKMLXWeightPrecision,
                                             bits: Int?, groupSize: Int) throws -> NFKMLXStageFootprint {
        if let bits {
            let packed = try NFKMLXWanBlockedLoad.quantizedFootprint(directory: directory, precision: precision,
                                                                     bits: bits, groupSize: groupSize)
            return NFKMLXStageFootprint(bytes: packed.bytes)
        }
        if let dtype {
            return NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(
                inDirectory: directory, holding: dtype, float32: NFKMLXWanRelease.keepsFloat32))
        }
        return try NFKMLXWanBlockedLoad.footprint(directory: directory, precision: precision)
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

    /// The component folders an A14B release's download fetches: the single-transformer set and `transformer_2/`.
    static let expertReleaseComponents = releaseComponents + [
        NFKMLXReleaseComponent(required: ["transformer_2/config.json"],
                               weights: ["transformer_2/diffusion_pytorch_model.safetensors",
                                         "transformer_2/diffusion_pytorch_model.safetensors.index.json"]),
    ]

    /// The Wan 2.2 A14B releases, Apache-2.0: two 14B experts at float32 (57 GB each), umT5-XXL, and the
    /// Wan 2.1 autoencoder.
    @objc public static let textToVideoExpertsRepo = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    /// The image-to-video A14B release. Introduced in InferKit 0.4.0.
    @objc public static let imageToVideoExpertsRepo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    /// Downloads a release (``releaseRepo`` when `repo` is nil, `Wan-AI/Wan2.2-TI2V-5B-Diffusers`,
    /// ``textToVideoExpertsRepo``, or ``imageToVideoExpertsRepo``) and assembles the model, holding it as
    /// `residency` says.
    ///
    /// @discussion Wan 2.1 T2V 1.3B is about 29 GB, an A14B release about 126 GB. A file already in the cache is not fetched again. The
    /// call blocks on the network, so run it off the main and render threads.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXWanVideoGenerator {
        try NFKMLXBuildHolds.keeping {
            let name = repo ?? releaseRepo
            let experts = [textToVideoExpertsRepo, imageToVideoExpertsRepo].contains(name)
            let directory = try NFKMLXReleaseDownload.directory(repo: name, revision: revision,
                                                                cacheDirectoryURL: cacheDirectoryURL,
                                                                components: experts ? expertReleaseComponents : releaseComponents)
            return try generator(directoryURL: directory, residency: residency)
        }
    }

    /// The asynchronous form of ``generator(repo:revision:cacheDirectoryURL:residency:)``. The handler
    /// runs on a background queue.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency,
                                 completionHandler: @escaping (NFKMLXWanVideoGenerator?, Error?) -> Void) {
        NFKMLXReleaseDownload.asyncValue(completionHandler) {
            try generator(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                          residency: residency)
        }
    }

    // MARK: Training

    /// Encodes one clip as `preset`'s trainer prepares it: the frames through the autoencoder to the
    /// posterior's normalized mean and log-variance, and the prompt through umT5 at 512 tokens — zeros past
    /// the prompt, or the encoder's outputs at the padded positions where the trainer keeps them
    /// (finetrainers encodes the padded prompt with no mask). An image-to-video release adds the condition its
    /// experts read, built from the clip's first frame.
    ///
    /// - Parameters:
    ///   - video: the frames `[4k + 1, H, W, 3]` in `[0, 1]`, already cut and sized as the trainer's bucketing
    ///     leaves them; the sides are multiples of 16.
    ///   - text: the clip's caption, read as written.
    ///   - preset: the trainer whose text encoding the example follows, and for an image-to-video release whose
    ///     condition it carries.
    ///
    /// Introduced in InferKit 0.4.0.
    public func trainingExample(video: MLXArray, prompt text: String,
                                preset: NFKMLXWanTrainingPreset = .finetrainers) throws -> NFKMLXWanTrainingExample {
        guard video.ndim == 4, (video.dim(0) - 1) % compression.temporal == 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "a training clip is \(compression.temporal)k + 1 frames of [H, W, 3], not \(video.shape)")
        }
        let tokens = NFKMLXT5Prompt(text, segmenter: segmenter, length: sequenceLength)
        return try staging.exclusively {
            let features = try staging.with(textStage) { stage -> MLXArray in
                preset.keepsPaddedText ? stage.encoder(tokens.tokens)[0] : stage.features(tokens)
            }
            let moments = try staging.with(pipelineStage) { pipeline -> [MLXArray] in
                let clip = video * 2 - 1
                let encoded = pipeline.normalizedMoments(clip)
                // An image-to-video expert trains from the clip's own first frame, as DiffSynth-Studio's does.
                let condition = isImageToVideo ? [pipeline.imageCondition(clip[0], lastImage: nil, frames: clip.dim(0))] : []
                return [encoded.mean, encoded.logVariance] + condition
            }
            return NFKMLXWanTrainingExample(latent: moments[0], logVariance: moments[1], text: features,
                                            condition: moments.count > 2 ? moments[2] : nil)
        }
    }

    // MARK: Adapters

    /// Merges a released LoRA (``NFKMLXWanLoRA``) into the transformer at `scale`, now where the transformer
    /// is loaded and on every later load. Adapters merge in the order they are added. In an A14B release
    /// the adapter merges into the high-noise expert, as diffusers' `load_lora_weights` does by default.
    ///
    /// @discussion The file is read and its layers named now; a layer this transformer does not have, or a
    /// shape it does not match, throws now where the transformer is loaded and at its next load otherwise.
    /// Introduced in InferKit 0.4.0.
    @objc(addAdapterAtURL:scale:error:)
    public func addAdapter(at url: URL, scale: Float) throws {
        try addAdapter(at: url, scale: scale, expert: .highNoise)
    }

    /// Merges a released LoRA into one expert of an A14B release: Wan 2.2 LoRAs ship as a high-noise and a
    /// low-noise file. A single-transformer release takes ``NFKMLXWanExpert/highNoise`` alone. A packed or
    /// streamed transformer merges an adapter only as it loads, so a loaded one is released and loads again
    /// with it. Introduced in InferKit 0.4.0.
    @objc(addAdapterAtURL:scale:expert:error:)
    public func addAdapter(at url: URL, scale: Float, expert: NFKMLXWanExpert) throws {
        guard adapters.indices.contains(expert.rawValue) else {
            throw NFKMLXError.unsupportedConfiguration("this release has one transformer; it has no low-noise expert")
        }
        try staging.exclusively {
            let deltas = try NFKMLXWanLoRA.deltas(adapterURL: url, naming: NFKMLXWanLoRA.textToVideoPath(original:),
                                                  scale: scale).deltas
            let stageHeld = expertStages.isEmpty ? pipelineStage.isHeld : expertStages[expert.rawValue].isHeld
            if stageHeld && mergesOnLoad {
                if expertStages.isEmpty {
                    pipelineStage.release()
                } else {
                    expertStages[expert.rawValue].release()
                }
                NFKMLXGPU.clearCache()
            } else if stageHeld {
                _ = try withExpert(expert) { try NFKMLXWanLoRA.apply(deltas, to: $0) }
            }
            adapters[expert.rawValue].add(url, scale: scale)
            if stageHeld && mergesOnLoad && staging.resident {
                _ = try withExpert(expert) { _ in }
            }
        }
    }

    /// Removes every adapter. A loaded transformer is released and loads again without them. Introduced in
    /// InferKit 0.4.0.
    @objc public func removeAdapters() {
        staging.exclusively {
            adapters.forEach { $0.removeAll() }
            pipelineStage.release()
            expertStages.forEach { $0.release() }
            NFKMLXGPU.clearCache()
        }
    }

    /// The adapters ``addAdapter(at:scale:)`` added, across the experts. Introduced in InferKit 0.4.0.
    @objc public var adapterCount: Int { staging.exclusively { adapters.reduce(0) { $0 + $1.entries.count } } }

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
        try video(forPrompt: prompt, image: nil, lastImage: nil, negativePrompt: negativePrompt, frames: frames,
                  width: width, height: height, seed: seed)
    }

    /// Generates a clip for `prompt` that starts at `image` (and ends at `lastImage`, where given),
    /// `[frames, height, width, 3]` in `[0, 1]`. An image-to-video release (``isImageToVideo``) requires the
    /// image and a text-to-video release refuses one.
    ///
    /// @discussion Each image is resized to the clip's `width` × `height` by PIL's 8-bit Lanczos filter, as
    /// the reference's image processor resizes a picture, then encoded with zeros between as the clip's
    /// first frame (and last) by the autoencoder. Introduced in InferKit 0.4.0.
    public func video(forPrompt prompt: String, image: CGImage?, lastImage: CGImage? = nil,
                      negativePrompt: String? = nil, frames: Int = 81, width: Int = 832, height: Int = 480,
                      seed: UInt64 = 0) throws -> MLXArray {
        try generate(prompt: prompt, image: image, lastImage: lastImage, negativePrompt: negativePrompt,
                     frames: frames, width: width, height: height, seed: seed).clip
    }

    /// ``video(forPrompt:image:lastImage:negativePrompt:frames:width:height:seed:)`` with its final latent,
    /// from `noise` in place of the seeded noise where given.
    func generate(prompt: String, image: CGImage?, lastImage: CGImage?, negativePrompt: String?, frames: Int,
                  width: Int, height: Int, seed: UInt64, noise: MLXArray? = nil) throws -> (clip: MLXArray, latent: MLXArray) {
        let tile = compression.spatial * 2
        let grid = ((frames - 1) / compression.temporal + 1, height / tile * 2, width / tile * 2)
        guard frames >= 1, grid.1 > 0, grid.2 > 0 else {
            throw NFKMLXError.unsupportedConfiguration("a clip is at least one frame of \(tile)×\(tile) pixels")
        }
        guard isImageToVideo == (image != nil), lastImage == nil || image != nil else {
            throw NFKMLXError.unsupportedConfiguration(isImageToVideo
                ? "this release starts a clip from an image; pass one"
                : "this release generates from text alone; it takes no image")
        }
        let pixelFrames = (grid.0 - 1) * compression.temporal + 1
        let size = (width: grid.2 * compression.spatial, height: grid.1 * compression.spatial)
        let pictures = try [image, lastImage].map { picture in
            try picture.map { try Self.preparedImage($0, width: size.width, height: size.height) }
        }
        let guides = guidance > 1
        let prompts = [self.prompt(prompt)] + (guides ? [self.prompt(negativePrompt ?? "")] : [])
        return try staging.exclusively {
            let features = try staging.with(textStage) { stage in prompts.map { stage.features($0) } }
            if releasesTextEncoderAfterEncoding {
                textStage.release()
            }
            // The encode's buffers, and umT5's own where it was released, stay in MLX's cache until cleared.
            NFKMLXGPU.clearCache()
            func denoiser(_ schedule: NFKMLXUniPCConfiguration, channels: Int, condition: MLXArray?) -> NFKWanDenoiser {
                NFKWanDenoiser(schedule: schedule, steps: steps, text: features[0], negative: guides ? features[1] : nil,
                               guidance: guidance, lowNoiseGuidance: lowNoiseGuidance,
                               boundary: boundaryRatio.map { $0 * Float(schedule.trainTimesteps) }, condition: condition,
                               noise: noise ?? NFKMLXWanPipeline.noise(channels: channels, frames: grid.0, height: grid.1,
                                                                       width: grid.2, seed: seed))
            }
            func decoded(_ pipeline: NFKMLXWanPipeline, _ latent: MLXArray) -> MLXArray {
                let clip = MLX.clip(pipeline.decode(latent)[0] / 2 + 0.5, min: 0, max: 1)
                eval(clip)
                return clip
            }
            guard !expertStages.isEmpty else {
                // One transformer: the pipeline stage holds it beside the autoencoder, loaded once per clip.
                return try staging.use(pipelineStage) { pipeline in
                    guard let transformer = pipeline.transformer else {
                        throw NFKMLXError.unsupportedConfiguration("the pipeline holds no transformer")
                    }
                    var run = denoiser(pipeline.schedule, channels: transformer.config.inChannels, condition: nil)
                    for index in run.timesteps.indices {
                        run.step(index, with: transformer)
                        try transformer.verifyStream()
                    }
                    return (decoded(pipeline, run.latent), run.latent)
                }
            }
            let (schedule, channels, condition) = try staging.use(pipelineStage) { pipeline in
                let condition = pictures[0].map { pipeline.imageCondition($0, lastImage: pictures[1], frames: pixelFrames) }
                return (pipeline.schedule, pipeline.latentChannels, condition)
            }
            var run = denoiser(schedule, channels: channels, condition: condition)
            for expert in [NFKMLXWanExpert.highNoise, .lowNoise] {
                let indices = run.timesteps.indices.filter { run.usesLowNoiseExpert($0) == (expert == .lowNoise) }
                guard !indices.isEmpty else { continue }
                try staging.use(expertStages[expert.rawValue]) { transformer in
                    for index in indices {
                        run.step(index, with: transformer)
                        try transformer.verifyStream()
                    }
                }
            }
            let latent = run.latent
            return (try staging.use(pipelineStage) { decoded($0, latent) }, latent)
        }
    }

    /// A picture resized to `width` × `height` by PIL's 8-bit Lanczos filter, `[height, width, 3]` in −1…1.
    static func preparedImage(_ image: CGImage, width: Int, height: Int) throws -> MLXArray {
        let (rgb, sourceWidth, sourceHeight) = try NFKMLXImageBridge.rgbBytes(
            from: image, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let resized = sourceWidth == width && sourceHeight == height ? rgb
            : NFKMLXPILResample.resampled(rgb, width: sourceWidth, height: sourceHeight, toWidth: width,
                                          toHeight: height, filter: .lanczos)
        return MLXArray(resized.map { Float($0) / 255 * 2 - 1 }).reshaped([height, width, 3])
    }

    /// Generates a clip for `prompt` and returns its frames as `CGImage`s. Objective-C reads each element
    /// as `(__bridge CGImageRef)`, since an array of a Core Foundation type is not an Objective-C
    /// collection.
    @objc(framesForPrompt:negativePrompt:frames:width:height:seed:error:)
    public func frames(forPrompt prompt: String, negativePrompt: String?, frames: Int, width: Int, height: Int,
                       seed: UInt64) throws -> [Any] {
        try self.frames(forPrompt: prompt, image: nil, lastImage: nil, negativePrompt: negativePrompt, frames: frames,
                        width: width, height: height, seed: seed)
    }

    /// ``video(forPrompt:image:lastImage:negativePrompt:frames:width:height:seed:)`` as `CGImage`s, read in
    /// Objective-C as ``frames(forPrompt:negativePrompt:frames:width:height:seed:)`` returns them.
    /// Introduced in InferKit 0.4.0.
    @objc(framesForPrompt:image:lastImage:negativePrompt:frames:width:height:seed:error:)
    public func frames(forPrompt prompt: String, image: CGImage?, lastImage: CGImage?, negativePrompt: String?,
                       frames: Int, width: Int, height: Int, seed: UInt64) throws -> [Any] {
        let clip = try video(forPrompt: prompt, image: image, lastImage: lastImage, negativePrompt: negativePrompt,
                             frames: frames, width: width, height: height, seed: seed)
        return try (0 ..< clip.dim(0)).map {
            try NFKMLXImageBridge.cgImage(from: clip[$0], options: NFKMLXImageOptions())
        }
    }
}
