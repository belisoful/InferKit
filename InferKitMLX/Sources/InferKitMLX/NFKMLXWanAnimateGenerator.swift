//
//  NFKMLXWanAnimateGenerator.swift
//  InferKitMLX
//
//  Wan 2.2 Animate 2 character animation from a downloaded diffusers release directory: a reference image
//  and a driving video in, a clip of the reference's character moving as the driving video does out. The
//  encoders (umT5-XXL and the CLIP image encoder) run once per clip and the transformer and autoencoder
//  after them, the two stages held as an `NFKMLXResidency` says.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXRandom

/// An RGB image, eight bits a channel, three interleaved channels a pixel.
struct NFKWanAnimateRGB {
    let pixels: [UInt8]
    let width: Int
    let height: Int

    /// The image's colors, its alpha dropped, as Core Graphics draws it in `colorSpace`.
    init(_ image: CGImage, colorSpace: CGColorSpace) {
        let (rgba, width, height) = NFKMLXImageBridge.rgbaBytes(from: image, colorSpace: colorSpace)
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0 ..< (width * height) {
            rgb[pixel * 3] = rgba[pixel * 4]
            rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1]
            rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2]
        }
        self.init(pixels: rgb, width: width, height: height)
    }

    init(pixels: [UInt8], width: Int, height: Int) {
        self.pixels = pixels
        self.width = width
        self.height = height
    }
}

/// The encoder stage: umT5 and the CLIP image encoder.
final class NFKWanAnimateEncoderStage {
    let text: NFKWanTextStage
    let image: NFKMLXWanAnimateImageEncoderNet
    init(text: NFKMLXT5EncoderNet, image: NFKMLXWanAnimateImageEncoderNet) {
        self.text = NFKWanTextStage(text)
        self.image = image
    }
}

/// Wan 2.2 Animate 2 character animation, assembled from a diffusers release directory.
///
/// @discussion The release layout is `text_encoder/` (umT5-XXL), `tokenizer/`, `image_encoder/` (CLIP
/// ViT-H/14), `transformer/` (the 14B Animate transformer), `vae/` (the Wan 2.1 autoencoder), and
/// `scheduler/`. ``video(referenceImage:drivingFrames:drivingFramesPerSecond:prompt:negativePrompt:seed:)``
/// letterboxes both inputs to one frame size, divides the driving video into overlapping segments, and
/// denoises each in turn, the next conditioned on the last frames the one before it decoded. The glue is
/// measured against diffusers' modular `WanAnimate2Blocks` over a small random release in the released
/// layout.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXWanAnimateGenerator)
public final class NFKMLXWanAnimateGenerator: NSObject {

    /// The registry name of the standard release.
    @objc public static let modelName = "wan2.2-animate-14b"

    /// The registry name of the distilled release.
    @objc public static let distilledModelName = "wan2.2-animate-14b-distilled"

    /// The Hugging Face repository the download factories read for the standard release.
    @objc public static let releaseRepo = NFKMLXWanAnimateVariant.standard.repository

    /// The prompt the reference pipeline conditions the extract pass on: "a reference video of a person's
    /// movements".
    static let referencePrompt = "人物动作的参考视频"

    /// The caption length the pipeline pads and truncates to.
    static let maximumSequenceLength = 512

    private let staging: NFKMLXStagedModel
    private let encoderStage: NFKMLXStage<NFKWanAnimateEncoderStage>
    private let pipelineStage: NFKMLXStage<NFKMLXWanAnimatePipeline>
    private let segmenter: NFKMLXSentencePieceSegmenter
    private let sequenceLength: Int

    /// The release the generator reads.
    @objc public let variant: NFKMLXWanAnimateVariant

    /// The denoising steps of each segment: 40 for the standard release, 10 for the distilled one.
    @objc public var steps: Int
    /// The classifier-free guidance scale: 3 for the standard release, 1 (none) for the distilled one.
    @objc public var guidance: Float
    /// The frames each segment denoises, 81 by the reference's default. One more than a multiple of four
    /// keeps every latent frame whole.
    @objc public var segmentLength = 81
    /// The frames each segment after the first carries over from the one before it, 1 by the reference's
    /// default.
    @objc public var carriedFrames = 1
    /// The frame rate the model generates at, 24. A driving video at another rate is resampled to it.
    @objc public var framesPerSecond: Double = 24
    /// The target frame width, 640 by the reference's default. With ``height`` it sets the frame's area;
    /// the reference image sets its aspect ratio.
    @objc public var width = 640
    /// The target frame height, 800 by the reference's default.
    @objc public var height = 800

    /// Whether the encoders and the transformer stay loaded together between clips.
    @objc public var holdsStagesResident: Bool { staging.resident }

    /// How many transformer blocks the model leaves in the release, read in their turn on every pass; 0
    /// where every block is held.
    @objc public internal(set) var streamedLayerCount = 0

    /// The bytes each transformer pass reads from the release for the streamed blocks; 0 where every block
    /// is held. A guided step runs both of its passes on one read.
    @objc public internal(set) var streamedBytesPerPass = 0

    /// The bits each of the transformer's packed linear weights holds; 0 where none are packed.
    @objc public internal(set) var quantizationBits = 0

    /// Each segment's starting noise for its index and shape, in place of the seeded draw. Tests set it
    /// to the reference's noise.
    var noiseSource: ((Int, [Int]) -> MLXArray)?

    init(variant: NFKMLXWanAnimateVariant, resident: Bool, segmenter: NFKMLXSentencePieceSegmenter,
         sequenceLength: Int = NFKMLXWanAnimateGenerator.maximumSequenceLength,
         loadEncoders: @escaping () throws -> NFKWanAnimateEncoderStage,
         loadPipeline: @escaping () throws -> NFKMLXWanAnimatePipeline) {
        self.variant = variant
        steps = variant.defaultSteps
        guidance = variant.defaultGuidance
        staging = NFKMLXStagedModel(resident: resident)
        encoderStage = NFKMLXStage(loadEncoders)
        pipelineStage = NFKMLXStage(loadPipeline)
        self.segmenter = segmenter
        self.sequenceLength = sequenceLength
        super.init()
    }

    /// Loads both stages now, for a resident release.
    func loadResident() throws {
        try staging.use(encoderStage) { _ in }
        NFKMLXGPU.clearCache()
        try staging.use(pipelineStage) { _ in }
    }

    /// Runs `body` with both stages loaded, the encoders' first.
    func usingStages<T>(_ body: (NFKWanAnimateEncoderStage, NFKMLXWanAnimatePipeline) throws -> T) throws -> T {
        try staging.exclusively {
            try staging.use(encoderStage) { encoders in
                try staging.use(pipelineStage) { pipeline in try body(encoders, pipeline) }
            }
        }
    }

    var isHoldingEncoders: Bool { staging.exclusively { encoderStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    // MARK: Factories

    /// Assembles the model from a downloaded release directory, holding it as
    /// ``NFKMLXResidency/automatic`` decides.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXWanAnimateGenerator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXWanAnimateGenerator {
        try generator(directoryURL: directoryURL, residency: residency, precision: .checkpoint, quantizationBits: 0)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says, every
    /// stage's weights at `precision`, and the transformer's block weights packed to `quantizationBits`.
    ///
    /// @discussion The released transformer is 32.8 GB at bfloat16, umT5-XXL 11.4 GB, the image encoder 1.3 GB,
    /// and the autoencoder 0.5 GB. `.checkpoint` keeps each stage's stored type; `.float32` widens all of
    /// them, which is what the parity measurement runs. A `quantizationBits` of 2 to 8 packs every linear
    /// layer inside the transformer's blocks as each block loads (4 bits holds the transformer in about
    /// 9.6 GB); 0 packs nothing. A machine that cannot hold the stages together stages under
    /// ``NFKMLXResidency/automatic``. Where the transformer and the autoencoder do not fit, an unpacked
    /// transformer under ``NFKMLXResidency/automatic`` or ``NFKMLXResidency/streamed`` holds the first
    /// blocks that fit and reads the others from the release on every pass (``streamedLayerCount``). The
    /// release has no routed experts, so ``NFKMLXResidency/paged`` holds it as ``NFKMLXResidency/staged``
    /// does. Introduced in InferKit 0.4.0.
    @objc(generatorWithDirectoryURL:residency:precision:quantizationBits:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                                 quantizationBits: Int) throws -> NFKMLXWanAnimateGenerator {
        try generator(directoryURL: directoryURL, residency: residency, precision: precision,
                      quantizationBits: quantizationBits, budget: NFKMLXResidencyBudget.current())
    }

    /// ``generator(directoryURL:residency:precision:quantizationBits:)`` planned against `budget` bytes.
    /// `streamedBlocks`, where given, replaces the blocks the plan streams.
    static func generator(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                          quantizationBits: Int, budget: Int,
                          groupSize: Int = NFKMLXLoadOptions.defaultQuantizationGroupSize,
                          streamedBlocks: Set<Int>? = nil) throws -> NFKMLXWanAnimateGenerator {
        let directory = { (name: String) in directoryURL.appendingPathComponent(name) }
        let bits: Int? = quantizationBits == 0 ? nil : quantizationBits
        let variant = try Self.variant(ofDirectory: directoryURL)
        let segmenter = NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: directory("tokenizer/spiece.model")))
        let textConfiguration = try NFKMLXT5Encoder.configuration(fromHuggingFace: directory("text_encoder/config.json"))
        let imageConfiguration = try NFKMLXWanAnimateImageEncoderConfiguration.configuration(
            fromHuggingFace: directory("image_encoder/config.json"))
        let transformerConfiguration = try NFKMLXWanAnimateRelease.transformerConfiguration(
            fromHuggingFace: directory("transformer/config.json"))
        let vae = try NFKMLXWanRelease.vaeConfiguration(fromHuggingFace: directory("vae/config.json"))
        let schedule = try NFKMLXWanAnimateSchedule.reading(directory("scheduler/scheduler_config.json"))

        let encoderBytes = try NFKMLXStageWeights.bytes(inDirectory: directory("text_encoder"), precision: precision)
            + NFKMLXStageWeights.bytes(inDirectory: directory("image_encoder"), precision: precision)
        let transformer = try bits.map {
            try NFKMLXWanAnimateRelease.quantizedTransformerFootprint(directory: directory("transformer"),
                                                                     precision: precision, bits: $0, groupSize: groupSize)
        } ?? NFKMLXWanAnimateRelease.transformerFootprint(directory: directory("transformer"), precision: precision)
        let pipelineFootprint = NFKMLXStageFootprint(
            bytes: transformer.bytes + (try NFKMLXStageWeights.bytes(inDirectory: directory("vae"), precision: precision)),
            layerBytes: bits == nil ? transformer.layerBytes : [])
        let plan = try NFKMLXResidencyBudget.plan([NFKMLXStageFootprint(bytes: encoderBytes), pipelineFootprint],
                                                  residency: residency, budget: budget)
        let streamed = streamedBlocks ?? (plan.streams(1)
            ? Set(plan.heldLayers(1, of: pipelineFootprint).upperBound ..< transformer.layerBytes.count) : [])

        let generator = NFKMLXWanAnimateGenerator(
            variant: variant, resident: plan.holdsStagesResident, segmenter: segmenter,
            loadEncoders: {
                let text = NFKMLXT5Encoder.makeNet(textConfiguration)
                try NFKMLXT5Encoder.loadWeights(into: text, from: directory("text_encoder"), precision: precision)
                let image = NFKMLXWanAnimateImageEncoderNet(imageConfiguration)
                try NFKMLXWanAnimateImageEncoderNet.loadWeights(into: image, fromDirectory: directory("image_encoder"),
                                                               precision: precision)
                return NFKWanAnimateEncoderStage(text: text, image: image)
            },
            loadPipeline: {
                let net = NFKMLXWanAnimateNet(transformerConfiguration)
                try NFKMLXWanAnimateRelease.loadTransformer(into: net, fromDirectory: directory("transformer"),
                                                            precision: precision, streamed: streamed, bits: bits,
                                                            groupSize: groupSize)
                let autoencoder = NFKMLXWanVideoVAENet(vae.configuration)
                try NFKMLXWanRelease.loadVAE(into: autoencoder, fromDirectory: directory("vae"), precision: precision)
                return NFKMLXWanAnimatePipeline(transformer: net, vae: autoencoder, latentsMean: vae.mean,
                                                latentsStd: vae.std, schedule: schedule)
            })
        generator.streamedLayerCount = streamed.count
        generator.streamedBytesPerPass = streamed.reduce(0) { $0 + transformer.layerBytes[$1] }
        generator.quantizationBits = bits ?? 0
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    /// The release a directory holds, read from its scheduler: the distilled release samples with Euler.
    static func variant(ofDirectory directoryURL: URL) throws -> NFKMLXWanAnimateVariant {
        let json = try NFKMLXWanAnimateSchedule.configuration(directoryURL.appendingPathComponent("scheduler/scheduler_config.json"))
        return (json["_class_name"] as? String) == "FlowMatchEulerDiscreteScheduler" ? .distilled : .standard
    }

    /// The component folders a download fetches.
    static let releaseComponents = [
        NFKMLXReleaseComponent(required: ["model_index.json", "scheduler/scheduler_config.json"],
                               optional: ["modular_model_index.json"]),
        NFKMLXReleaseComponent(required: ["text_encoder/config.json"],
                               weights: ["text_encoder/model.safetensors", "text_encoder/model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["tokenizer/spiece.model"],
                               optional: ["tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json"]),
        NFKMLXReleaseComponent(required: ["image_encoder/config.json"],
                               weights: ["image_encoder/model.safetensors", "image_encoder/model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["transformer/config.json"],
                               weights: ["transformer/diffusion_pytorch_model.safetensors",
                                         "transformer/diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]),
    ]

    /// Downloads a release (`variant`'s repository when `repo` is nil) and assembles the model, holding it
    /// as `residency` says.
    ///
    /// @discussion Each release is about 50 GB. A file already in the cache is not fetched again. The call
    /// blocks on the network, so run it off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(generatorWithVariant:repo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(variant: NFKMLXWanAnimateVariant, repo: String?, revision: String?,
                                 cacheDirectoryURL: URL?, residency: NFKMLXResidency) throws -> NFKMLXWanAnimateGenerator {
        try NFKMLXBuildHolds.keeping {
            let directory = try NFKMLXReleaseDownload.directory(repo: repo ?? variant.repository, revision: revision,
                                                                cacheDirectoryURL: cacheDirectoryURL,
                                                                components: releaseComponents)
            return try generator(directoryURL: directory, residency: residency)
        }
    }

    /// The asynchronous form of ``generator(variant:repo:revision:cacheDirectoryURL:residency:)``. The
    /// handler runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(generatorWithVariant:repo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func generator(variant: NFKMLXWanAnimateVariant, repo: String?, revision: String?,
                                 cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                                 completionHandler: @escaping (NFKMLXWanAnimateGenerator?, Error?) -> Void) {
        NFKMLXReleaseDownload.asyncValue(completionHandler) {
            try generator(variant: variant, repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                          residency: residency)
        }
    }

    // MARK: Generation

    /// The prompt as the encoder reads it. The reference passes the prompt to the tokenizer as written.
    func prompt(_ text: String) -> NFKMLXT5Prompt {
        NFKMLXT5Prompt(text, segmenter: segmenter, length: sequenceLength)
    }

    /// Animates `referenceImage` with the motion of `drivingFrames`, `[frames, height, width, 3]` in `[0, 1]`.
    ///
    /// @discussion The frame size is ``width`` × ``height`` in area, shaped to the reference image's aspect
    /// ratio and rounded down to multiples of 16; both inputs are letterboxed into it and the letterbox is
    /// cropped off the result. A `drivingFramesPerSecond` above 0 resamples the driving video from that rate
    /// to ``framesPerSecond`` by the nearest frame; 0 takes the frames as they are. The clip has one frame
    /// per driving frame after resampling.
    public func video(referenceImage: CGImage, drivingFrames: [CGImage], drivingFramesPerSecond: Double,
                      prompt: String, negativePrompt: String? = nil, seed: UInt64 = 0) throws -> MLXArray {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        return try video(reference: NFKWanAnimateRGB(referenceImage, colorSpace: colorSpace),
                         driving: drivingFrames.map { NFKWanAnimateRGB($0, colorSpace: colorSpace) },
                         drivingRate: drivingFramesPerSecond > 0 ? drivingFramesPerSecond : nil,
                         prompt: prompt, negativePrompt: negativePrompt, seed: seed)
    }

    /// ``video(referenceImage:drivingFrames:drivingFramesPerSecond:prompt:negativePrompt:seed:)`` over RGB
    /// pixels.
    func video(reference: NFKWanAnimateRGB, driving: [NFKWanAnimateRGB], drivingRate: Double?, prompt text: String,
               negativePrompt: String?, seed: UInt64, progress: (Double) -> Void = { _ in }) throws -> MLXArray {
        let frame = NFKWanAnimatePreprocess.frameSize(imageWidth: reference.width, imageHeight: reference.height,
                                                      width: width, height: height)
        guard frame.height > 0, frame.width > 0 else {
            throw NFKMLXError.unsupportedConfiguration("a frame is at least 16×16 pixels")
        }
        let crop = NFKWanAnimatePreprocess.cropRegion(imageWidth: reference.width, imageHeight: reference.height,
                                                      frameHeight: frame.height, frameWidth: frame.width)
        let kept = drivingRate.map {
            NFKWanAnimatePreprocess.frameIndices(count: driving.count, sourceRate: $0, targetRate: framesPerSecond)
        } ?? Array(driving.indices)
        let segments = try NFKWanAnimateSegments(realFrames: kept.count, segmentLength: segmentLength,
                                                 carriedFrames: carriedFrames)
        let referencePixels = NFKWanAnimatePreprocess.letterboxed(
            reference.pixels, width: reference.width, height: reference.height, toWidth: frame.width,
            toHeight: frame.height, filter: .bicubic)
        let letterboxed = kept.map { index -> MLXArray in
            let source = driving[index]
            return NFKWanAnimatePreprocess.letterboxed(source.pixels, width: source.width, height: source.height,
                                                       toWidth: frame.width, toHeight: frame.height, filter: .bilinear)
        }
        let drivingPixels = stacked(segments.paddedIndices.map { letterboxed[$0] }, axis: 0)

        let guides = guidance != 1
        let prompts = [self.prompt(text), self.prompt(Self.referencePrompt)]
            + (guides ? [self.prompt(negativePrompt ?? "")] : [])
        return try staging.exclusively {
            let features = try staging.with(encoderStage) { stage -> [MLXArray] in
                prompts.map { stage.text.features($0) }
                    + [stage.image.features(referencePixels), stage.image.features(drivingPixels[0])]
            }
            NFKMLXGPU.clearCache()
            let count = prompts.count
            return try staging.with(pipelineStage) { pipeline in
                MLXRandom.seed(seed)
                let noise = noiseSource ?? { _, shape in MLXRandom.normal(shape) }
                let clip = try pipeline.animate(
                    reference: referencePixels, driving: drivingPixels, segments: segments, prompt: features[0],
                    negativePrompt: guides ? features[2] : nil, referencePrompt: features[1],
                    imageEmbeddings: features[count], drivingImageEmbeddings: features[count + 1], steps: steps,
                    guidance: guidance, noise: noise, progress: progress)
                let cropped = clip[0..., crop.top ..< (crop.top + crop.height), crop.left ..< (crop.left + crop.width)]
                return MLX.clip(cropped / 2 + 0.5, min: 0, max: 1)
            }
        }
    }

    /// Animates `referenceImage` with `drivingFrames` and returns the clip's frames as `CGImage`s.
    /// Objective-C reads each element as `(__bridge CGImageRef)`, since an array of a Core Foundation type
    /// is not an Objective-C collection.
    @objc(framesForReferenceImage:drivingFrames:drivingFramesPerSecond:prompt:negativePrompt:seed:error:)
    public func frames(referenceImage: CGImage, drivingFrames: [Any], drivingFramesPerSecond: Double, prompt: String,
                       negativePrompt: String?, seed: UInt64) throws -> [Any] {
        let images = try drivingFrames.map { frame -> CGImage in
            guard CFGetTypeID(frame as CFTypeRef) == CGImage.typeID else {
                throw NFKMLXError.unsupportedConfiguration("every driving frame is a CGImage")
            }
            return frame as! CGImage
        }
        let clip = try video(referenceImage: referenceImage, drivingFrames: images,
                             drivingFramesPerSecond: drivingFramesPerSecond, prompt: prompt,
                             negativePrompt: negativePrompt, seed: seed)
        return try (0 ..< clip.dim(0)).map {
            try NFKMLXImageBridge.cgImage(from: clip[$0], options: NFKMLXImageOptions())
        }
    }
}
