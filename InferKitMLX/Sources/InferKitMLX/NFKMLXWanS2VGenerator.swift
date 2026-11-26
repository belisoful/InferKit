//
//  NFKMLXWanS2VGenerator.swift
//  InferKitMLX
//
//  Wan 2.2 S2V-14B (Wan-AI, Apache-2.0) end to end from its release directory: umT5 from its `.pth`,
//  wav2vec2-large-xlsr-53-english, the speech-to-video transformer, and the Wan 2.1 autoencoder, staged as an
//  `NFKMLXResidency` says.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// Wan 2.2 S2V-14B, assembled from its release directory.
///
/// @discussion The release layout is the original Wan one (`NFKMLXWanOriginalRelease`): umT5 as
/// `models_t5_umt5-xxl-enc-bf16.pth` with its tokenizer under `google/umt5-xxl/`, the autoencoder as
/// `Wan2.1_VAE.pth`, the transformer as `config.json` and its safetensors shards, and the speech encoder under
/// `wav2vec2-large-xlsr-53-english/`. ``video(forPrompt:negativePrompt:referenceImage:audio:poseFrames:startsFromReference:segments:seed:)``
/// samples as Wan-Video/Wan2.2's `WanS2V.generate` does (`NFKMLXWanS2VPipeline`), with the defaults of its
/// `generate.py`: segments of 80 frames at 16 per second, as many as the speech covers, each over the original
/// flow UniPC at a shift of 3 for 40 steps with guidance 4.5, the size the reference image's aspect gives within
/// 1280 × 720 pixels.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXWanS2VGenerator)
public final class NFKMLXWanS2VGenerator: NSObject {

    /// The registry name.
    @objc public static let modelName = "wan2.2-s2v-14b"
    /// The Hugging Face repository the download factories read.
    @objc public static let releaseRepo = "Wan-AI/Wan2.2-S2V-14B"
    /// The release's speech encoder directory.
    static let wav2vec2Directory = "wav2vec2-large-xlsr-53-english"
    /// The negative prompt `generate()` reads when none is given (`s2v_14B.sample_neg_prompt`).
    @objc public static let defaultNegativePrompt = "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKMLXT5EncoderNet>
    private let audioStage: NFKMLXStage<NFKMLXWav2Vec2Net>
    /// The transformer and the autoencoder, which every segment reads in turn.
    private let pipelineStage: NFKMLXStage<NFKMLXWanS2VPipeline>
    private let segmenter: NFKMLXSentencePieceSegmenter
    private let adapters: NFKWanAdapterList
    private let mergesOnLoad: Bool

    /// The denoising steps of each segment, 40 by the release's configuration.
    @objc public var steps: Int = 40
    /// The classifier-free guidance scale, 4.5 by the release's configuration. Above 1 each step also runs the
    /// negative prompt over silent audio.
    @objc public var guidance: Float = 4.5
    /// The flow shift of the UniPC schedule, 3 by the release's configuration.
    @objc public var shift: Double = 3
    /// The frames each segment generates, a multiple of 4 (80 by `generate.py`).
    @objc public var framesPerSegment: Int = 80
    /// The largest pixel area a clip takes (`max_area`, 1280 × 720 by `generate.py`).
    @objc public var maximumArea: Int = 1280 * 720

    /// The frame rate the release is trained at and the speech is sampled at.
    @objc public var framesPerSecond: Double { 16 }

    /// The bits the transformer's block weights are packed to, or 0.
    @objc public internal(set) var quantizationBits = 0

    /// How many transformer blocks are read from the release on every pass.
    @objc public internal(set) var streamedLayerCount = 0

    /// Whether the encoders and the transformer stay loaded together between clips.
    @objc public var holdsStagesResident: Bool { staging.resident }

    init(resident: Bool, segmenter: NFKMLXSentencePieceSegmenter, adapters: NFKWanAdapterList, mergesOnLoad: Bool,
         loadTextEncoder: @escaping () throws -> NFKMLXT5EncoderNet,
         loadAudioEncoder: @escaping () throws -> NFKMLXWav2Vec2Net,
         loadPipeline: @escaping () throws -> NFKMLXWanS2VPipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage(loadTextEncoder)
        audioStage = NFKMLXStage(loadAudioEncoder)
        pipelineStage = NFKMLXStage(loadPipeline)
        self.segmenter = segmenter
        self.adapters = adapters
        self.mergesOnLoad = mergesOnLoad
        super.init()
    }

    func loadResident() throws {
        try staging.use(textStage) { _ in }
        try staging.use(audioStage) { _ in }
        NFKMLXGPU.clearCache()
        try staging.use(pipelineStage) { _ in }
    }

    var isHoldingTextEncoder: Bool { staging.exclusively { textStage.isHeld } }
    var isHoldingAudioEncoder: Bool { staging.exclusively { audioStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    /// Runs `body` with the pipeline stage (the transformer and the autoencoder).
    func withPipeline<T>(_ body: (NFKMLXWanS2VPipeline) throws -> T) throws -> T {
        try staging.exclusively { try staging.use(pipelineStage, body) }
    }

    /// The text features of `prompts` from one load of umT5, each cleaned as the official tokenizer cleans it.
    func encodedFeatures(_ prompts: [String]) throws -> [MLXArray] {
        try staging.exclusively {
            try staging.with(textStage) { encoder in
                prompts.map {
                    NFKMLXWanFunRelease.features(NFKMLXWanPromptCleaning.cleaned($0), encoder: encoder, segmenter: segmenter)
                }
            }
        }
    }

    /// The speech buckets of 16 kHz mono samples for segments of ``framesPerSegment`` frames.
    func encodedAudio(_ samples: [Float]) throws -> NFKMLXWanS2VAudio {
        guard framesPerSegment > 0, framesPerSegment % 4 == 0 else {
            throw NFKMLXError.unsupportedConfiguration("an S2V segment generates a multiple of 4 frames, not \(framesPerSegment)")
        }
        guard !samples.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration("an S2V clip needs speech")
        }
        let frames = framesPerSegment
        return try staging.exclusively {
            try staging.use(audioStage) { wav2vec2 in
                NFKMLXWanS2VPipeline.audio(samples, wav2vec2: wav2vec2, fps: Int(framesPerSecond), framesPerSegment: frames)
            }
        }
    }

    // MARK: Factories

    /// Assembles the model from a downloaded release directory, holding it as ``NFKMLXResidency/automatic``
    /// decides.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXWanS2VGenerator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    ///
    /// @discussion umT5-XXL is stored at bfloat16 (11.4 GB) and runs at float32 where the plan affords it;
    /// wav2vec2 is 1.3 GB at float32. The transformer is 32.6 GB at bfloat16 and shares its stage with the
    /// autoencoder, so a 32 GB machine packs it (``generator(directoryURL:residency:precision:quantizationBits:)``)
    /// or streams its blocks. The release has no routed experts, so ``NFKMLXResidency/paged`` holds it as
    /// ``NFKMLXResidency/staged`` does.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXWanS2VGenerator {
        try generator(directoryURL: directoryURL, residency: residency, precision: .checkpoint, bits: nil)
    }

    /// Assembles the model holding it as `residency` says, every stage's weights at `precision`, and the
    /// transformer's block weights packed to `quantizationBits` (2 to 8; 0 packs nothing) as each block loads.
    @objc(generatorWithDirectoryURL:residency:precision:quantizationBits:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                                 quantizationBits: Int) throws -> NFKMLXWanS2VGenerator {
        try generator(directoryURL: directoryURL, residency: residency, precision: precision,
                      bits: quantizationBits == 0 ? nil : quantizationBits)
    }

    static func generator(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                          bits: Int?, groupSize: Int = NFKMLXLoadOptions.defaultQuantizationGroupSize,
                          budget: Int? = nil, streamedBlocks: Set<Int>? = nil) throws -> NFKMLXWanS2VGenerator {
        let configuration = try NFKMLXWanS2VConfiguration.reading(directoryURL.appendingPathComponent("config.json"))
        let segmenter = try NFKMLXWanFunRelease.segmenter(directoryURL)
        let textURL = directoryURL.appendingPathComponent(NFKMLXWanOriginalRelease.textEncoderFile)
        let vaeURL = directoryURL.appendingPathComponent(NFKMLXWanOriginalRelease.wan21AutoencoderFile)
        let audioDirectory = directoryURL.appendingPathComponent(wav2vec2Directory)
        let audioURL = try NFKMLXWav2Vec2Net.weightsURL(inDirectory: audioDirectory)

        let textBytes = try fileBytes(textURL)
        let transformer = bits.map { bits in
            Result { NFKMLXStageFootprint(bytes: try NFKMLXWanBlockedLoad.quantizedFootprint(
                directory: directoryURL, precision: precision, bits: bits, groupSize: groupSize).bytes) }
        } ?? Result { try NFKMLXWanBlockedLoad.footprint(directory: directoryURL, precision: precision) }
        let transformerFootprint = try transformer.get()
        let stages = [NFKMLXStageFootprint(bytes: textBytes, widenedBytes: 2 * textBytes),
                      NFKMLXStageFootprint(bytes: try fileBytes(audioURL)),
                      NFKMLXStageFootprint(bytes: transformerFootprint.bytes + (try fileBytes(vaeURL)),
                                           layerBytes: transformerFootprint.layerBytes)]
        let plan = try NFKMLXResidencyBudget.plan(stages, residency: residency,
                                                  budget: budget ?? NFKMLXResidencyBudget.current())
        let streamed = streamedBlocks ?? (plan.streams(2)
            ? Set(plan.heldLayers(2, of: stages[2]).upperBound ..< transformerFootprint.layerBytes.count) : [])
        let textInFloat32 = plan.widens(0)
        let adapters = NFKWanAdapterList()
        let generator = NFKMLXWanS2VGenerator(
            resident: plan.holdsStagesResident, segmenter: segmenter, adapters: adapters,
            mergesOnLoad: bits != nil || !streamed.isEmpty,
            loadTextEncoder: {
                try NFKMLXWanOriginalRelease.textEncoder(from: textURL, precision: textInFloat32 ? .float32 : .checkpoint)
            },
            loadAudioEncoder: {
                let net = try NFKMLXWav2Vec2Net(configurationURL: audioDirectory.appendingPathComponent("config.json"))
                try net.loadWeights(fromDirectory: audioDirectory)
                return net
            },
            loadPipeline: {
                let net = NFKMLXWanS2VNet(configuration)
                let deltas = try adapters.deltas(naming: NFKMLXWanLoRA.speechToVideoPath(original:))
                if bits != nil || !streamed.isEmpty {
                    net.layerStream = try NFKMLXWanBlockedLoad.load(
                        into: net, blocks: net.blocks, fromDirectory: directoryURL, precision: precision,
                        streamed: streamed, bits: bits, groupSize: groupSize, adapters: deltas,
                        rename: NFKMLXWanS2VNet.moduleKey(forRelease:), adapt: NFKMLXWanS2VNet.adapted)
                } else {
                    try NFKMLXWanS2VNet.loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
                    for delta in deltas {
                        try NFKMLXWanLoRA.apply(delta, to: net)
                    }
                }
                // The autoencoder is stored at float32 and runs at it, as the reference runs it.
                let autoencoder = try NFKMLXWanOriginalRelease.autoencoder(from: vaeURL)
                return NFKMLXWanS2VPipeline(net: net, vae: autoencoder.net, latentsMean: autoencoder.mean,
                                            latentsStd: autoencoder.std)
            })
        generator.quantizationBits = bits ?? 0
        generator.streamedLayerCount = streamed.count
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    private static func fileBytes(_ url: URL) throws -> Int {
        (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue ?? 0
    }

    /// The files a download fetches.
    static let releaseComponents: [NFKMLXReleaseComponent] = [
        NFKMLXReleaseComponent(required: [NFKMLXWanOriginalRelease.textEncoderFile]),
        NFKMLXReleaseComponent(required: ["google/umt5-xxl/spiece.model"],
                               optional: ["google/umt5-xxl/tokenizer.json", "google/umt5-xxl/tokenizer_config.json",
                                          "google/umt5-xxl/special_tokens_map.json"]),
        NFKMLXReleaseComponent(required: ["config.json", NFKMLXWanOriginalRelease.wan21AutoencoderFile],
                               weights: ["diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["\(wav2vec2Directory)/config.json", "\(wav2vec2Directory)/preprocessor_config.json"],
                               optional: ["\(wav2vec2Directory)/vocab.json", "\(wav2vec2Directory)/special_tokens_map.json"],
                               weights: ["\(wav2vec2Directory)/model.safetensors", "\(wav2vec2Directory)/pytorch_model.bin"]),
    ]

    /// Downloads the release (``releaseRepo``, or `repo` where given) and assembles the model, holding it as
    /// `residency` says.
    ///
    /// @discussion The release is about 49 GB. A file already in the cache is not fetched again. The call
    /// blocks on the network, so run it off the main and render threads.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXWanS2VGenerator {
        try NFKMLXBuildHolds.keeping {
            let directory = try NFKMLXReleaseDownload.directory(repo: repo ?? releaseRepo, revision: revision,
                                                                cacheDirectoryURL: cacheDirectoryURL,
                                                                components: releaseComponents)
            return try generator(directoryURL: directory, residency: residency)
        }
    }

    /// The asynchronous form of ``generator(repo:revision:cacheDirectoryURL:residency:)``. The handler runs on a
    /// background queue.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                                 completionHandler: @escaping (NFKMLXWanS2VGenerator?, Error?) -> Void) {
        NFKMLXReleaseDownload.asyncValue(completionHandler) {
            try generator(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: residency)
        }
    }

    // MARK: Adapters

    /// Merges a released LoRA (``NFKMLXWanLoRA``) into the transformer at `scale`, now where it is loaded and on
    /// every later load. A packed or streamed transformer merges an adapter only as it loads, so a loaded one is
    /// released and loads again with it.
    @objc(addAdapterAtURL:scale:error:)
    public func addAdapter(at url: URL, scale: Float) throws {
        try staging.exclusively {
            let deltas = try NFKMLXWanLoRA.deltas(adapterURL: url, naming: NFKMLXWanLoRA.speechToVideoPath(original:),
                                                  scale: scale).deltas
            let held = pipelineStage.isHeld
            if held && mergesOnLoad {
                pipelineStage.release()
                NFKMLXGPU.clearCache()
            } else if held {
                _ = try staging.use(pipelineStage) { try NFKMLXWanLoRA.apply(deltas, to: $0.net) }
            }
            adapters.add(url, scale: scale)
            if held && mergesOnLoad && staging.resident {
                try loadResident()
            }
        }
    }

    /// Removes every adapter. A loaded transformer is released and loads again without them.
    @objc public func removeAdapters() {
        staging.exclusively {
            adapters.removeAll()
            pipelineStage.release()
            NFKMLXGPU.clearCache()
        }
    }

    /// The adapters ``addAdapter(at:scale:)`` added.
    @objc public var adapterCount: Int { staging.exclusively { adapters.entries.count } }

    // MARK: Pictures

    /// The clip size a reference image of `width` × `height` gives within `maximumArea` (`get_size_less_than_area`):
    /// the largest scale of the image whose sides, each padded up to a multiple of 64, stay within the area,
    /// searched over 100 steps between the scale that pads most and the one that pads none; failing that, the
    /// image's aspect at the area, each side a multiple of 64.
    static func clipSize(imageWidth width: Int, imageHeight height: Int, maximumArea: Int) -> (width: Int, height: Int) {
        let h = Double(height), w = Double(width), target = Double(maximumArea)
        var minimum = 0.1, maximum = 1.0
        if h * w > target {
            let d = 63.0, b = d * (h + w), a = h * w, c = d * d - target
            minimum = (-b + (b * b - 2 * a * c).squareRoot()) / (2 * a)
            maximum = (target / (h * w)).squareRoot()
        }
        for i in 0 ..< 100 {
            let scale = maximum - (maximum - minimum) * Double(i) / 100
            let scaledHeight = Int(h * scale), scaledWidth = Int(w * scale)
            let paddedHeight = scaledHeight + (64 - scaledHeight % 64) % 64
            let paddedWidth = scaledWidth + (64 - scaledWidth % 64) % 64
            if paddedHeight * paddedWidth <= maximumArea {
                return (paddedWidth, paddedHeight)
            }
        }
        let aspect = w / h
        var targetWidth = Int(((target * aspect).squareRoot() / 64).rounded(.down) * 64)
        var targetHeight = Int(((target / aspect).squareRoot() / 64).rounded(.down) * 64)
        if targetWidth >= width || targetHeight >= height {
            targetWidth = width / 64 * 64
            targetHeight = height / 64 * 64
        }
        return (targetWidth, targetHeight)
    }

    /// A picture at the clip's size, `[height, width, 3]` in `[0, 1]`, as torchvision's `Resize(min(height, width))`
    /// and `CenterCrop((height, width))` leave a PIL image: the shorter side scaled to the clip's shorter side by
    /// PIL's 8-bit bilinear filter, the longer to `int(side · long / short)`, then the center cut out (rounding
    /// half to even), padded with black where the scaled picture is smaller than the clip.
    static func preparedPicture(_ image: CGImage, width: Int, height: Int) throws -> MLXArray {
        let (rgb, sourceWidth, sourceHeight) = try NFKMLXImageBridge.rgbBytes(
            from: image, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let side = min(width, height)
        let short = min(sourceWidth, sourceHeight), long = max(sourceWidth, sourceHeight)
        let scaledLong = Int(Double(side) * Double(long) / Double(short))
        let (scaledWidth, scaledHeight) = sourceWidth <= sourceHeight ? (side, scaledLong) : (scaledLong, side)
        let scaled = scaledWidth == sourceWidth && scaledHeight == sourceHeight ? rgb
            : NFKMLXPILResample.resampled(rgb, width: sourceWidth, height: sourceHeight, toWidth: scaledWidth,
                                          toHeight: scaledHeight, filter: .bilinear)
        // `CenterCrop` pads a picture smaller than the crop by half the shortfall on each side (the extra pixel
        // after), then crops from `round((size − crop) / 2)`.
        let padLeft = max(width - scaledWidth, 0) / 2, padTop = max(height - scaledHeight, 0) / 2
        let paddedWidth = max(scaledWidth, width), paddedHeight = max(scaledHeight, height)
        let left = Int((Double(paddedWidth - width) / 2).rounded(.toNearestOrEven))
        let top = Int((Double(paddedHeight - height) / 2).rounded(.toNearestOrEven))
        var pixels = [Float](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            let sourceY = y + top - padTop
            guard sourceY >= 0, sourceY < scaledHeight else { continue }
            for x in 0 ..< width {
                let sourceX = x + left - padLeft
                guard sourceX >= 0, sourceX < scaledWidth else { continue }
                for channel in 0 ..< 3 {
                    pixels[(y * width + x) * 3 + channel] = Float(scaled[(sourceY * scaledWidth + sourceX) * 3 + channel]) / 255
                }
            }
        }
        return MLXArray(pixels).reshaped([height, width, 3])
    }

    // MARK: Generation

    /// Generates a talking clip of `referenceImage` driven by `audio`, `[frames, height, width, 3]` in `[0, 1]`.
    ///
    /// @discussion `audio` is mono 16 kHz samples. The clip's size follows the reference image's aspect within
    /// ``maximumArea`` and the picture is fitted to it (`preparedPicture(_:width:height:)`). The clip is as many
    /// segments of ``framesPerSegment`` frames as the speech covers, or `segments` where fewer; the first segment
    /// drops its first three frames, so the clip holds `segments · framesPerSegment − 3` frames.
    /// `startsFromReference` starts the clip from the reference image (`--start_from_ref`) and keeps every
    /// frame. `poseFrames`, a pose video's frames at 16 per second (`--pose_video`), drives the body; each is
    /// fitted to the clip as the reference is. The negative prompt defaults to ``defaultNegativePrompt``.
    public func video(forPrompt prompt: String, negativePrompt: String? = nil, referenceImage: CGImage, audio: [Float],
                      poseFrames: [CGImage]? = nil, startsFromReference: Bool = false, segments: Int? = nil,
                      seed: UInt64 = 0) throws -> MLXArray {
        let size = Self.clipSize(imageWidth: referenceImage.width, imageHeight: referenceImage.height, maximumArea: maximumArea)
        let reference = try Self.preparedPicture(referenceImage, width: size.width, height: size.height)
        let pose = try poseFrames.map { frames in
            stacked(try frames.map { try Self.preparedPicture($0, width: size.width, height: size.height) })
        }
        return try generate(prompt: prompt, negativePrompt: negativePrompt, reference: reference, audio: audio, pose: pose,
                            startsFromReference: startsFromReference, segments: segments, seed: seed)
    }

    /// Segment `index`'s noise `[16, F', h, w]` from `seed + index`, as the reference seeds each segment.
    static func noise(seed: UInt64, segment: Int, shape: [Int]) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(seed &+ UInt64(segment)))
    }

    /// Generates from a reference picture already at the clip's size, `[height, width, 3]` in `[0, 1]`, and pose
    /// frames at that size.
    func generate(prompt: String, negativePrompt: String?, reference: MLXArray, audio samples: [Float], pose: MLXArray?,
                  startsFromReference: Bool, segments: Int?, seed: UInt64, noise: ((Int) -> MLXArray)? = nil,
                  progress: ((Double) -> Void)? = nil) throws -> MLXArray {
        guard reference.dim(0) % 16 == 0, reference.dim(1) % 16 == 0 else {
            throw NFKMLXError.unsupportedConfiguration("an S2V clip's sides are multiples of 16, not \(reference.shape)")
        }
        let guides = guidance > 1
        let texts = [prompt] + (guides ? [negativePrompt ?? Self.defaultNegativePrompt] : [])
        let features = try encodedFeatures(texts)
        let audio = try encodedAudio(samples)
        NFKMLXGPU.clearCache()
        let count = min(segments ?? audio.segments, audio.segments)
        let steps = self.steps, guidance = self.guidance, shift = self.shift, frames = framesPerSegment
        let latentShape = [16, frames / 4, reference.dim(0) / 8, reference.dim(1) / 8]
        return try staging.exclusively {
            try staging.use(pipelineStage) { pipeline in
                let clip = try pipeline.generate(
                    text: features[0], negative: guides ? features[1] : nil, reference: reference, audio: audio,
                    segments: count, pose: pose, startsFromReference: startsFromReference,
                    noise: noise ?? { Self.noise(seed: seed, segment: $0, shape: latentShape) }, steps: steps,
                    guidance: guidance, shift: shift,
                    onStep: { segment, step in progress?(Double(segment * steps + step + 1) / Double(count * steps + 1)) })
                eval(clip)
                return clip
            }
        }
    }
}
