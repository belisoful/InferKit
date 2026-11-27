//
//  NFKMLXWanDancerGenerator.swift
//  InferKitMLX
//
//  Wan-Dancer-14B (Wan-AI, Apache-2.0) end to end from its release directory, as Wan-Video/Wan-Dancer's
//  `gen_video_global.py` and `gen_video_local.py` run it: a global stage that plans keyframes over the whole
//  track at a reduced frame rate, then a local stage that renders the track at 30 frames a second in 149-frame
//  segments conditioned on those keyframes.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

/// An 8-bit RGB picture.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanDancerPicture {
    public var rgb: [UInt8]
    public var width: Int
    public var height: Int

    public init(rgb: [UInt8], width: Int, height: Int) {
        self.rgb = rgb
        self.width = width
        self.height = height
    }

    /// The picture `[H, W, 3]` in `0...1`.
    var array: MLXArray { MLXArray(rgb.map { Float($0) / 255 }, [height, width, 3]) }
}

/// The global stage's keyframe video: its frames at the clip size before the crop, and the box the picture fills.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanDancerKeyframeVideo {
    /// The frames `[N, H, W, 3]`, 8-bit as the reference saves them.
    public var frames: [NFKMLXWanDancerPicture]
    /// The box `(x0, y0, x1, y1)` the reference image fills within the clip.
    public var box: (Int, Int, Int, Int)
}

/// Wan-Dancer-14B, assembled from its release directory.
///
/// @discussion The release layout is the original Wan one: umT5 as `models_t5_umt5-xxl-enc-bf16.pth` with its
/// tokenizer under `google/umt5-xxl/`, the autoencoder as `Wan2.1_VAE.pth`, the image encoder as open-clip's
/// `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, and the two transformers as
/// `global_model.safetensors` and `local_model.safetensors` beside one `config.json`.
/// ``video(prompt:localPrompt:negativePrompt:reference:music:sampleRate:seed:)`` runs both stages as the released
/// scripts run them: 1280 × 720 clips of 149 frames, 48 global and 24 local steps of DiffSynth-Studio's flow
/// match at shift 5 with guidance 5, the unconditional pass skipping block 9, and the music encoder read in the
/// eight spans of the scripts' eight-way sequence parallelism.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXWanDancerGenerator)
public final class NFKMLXWanDancerGenerator: NSObject {

    /// The registry name.
    @objc public static let modelName = "wan-dancer-14b"
    /// The Hugging Face repository the download factories read.
    @objc public static let releaseRepo = "Wan-AI/Wan-Dancer-14B"
    /// The negative prompt the released scripts pass.
    @objc public static let defaultNegativePrompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    static let imageEncoderFile = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    static let stageFiles = (global: "global_model.safetensors", local: "local_model.safetensors")
    /// The frames of a clip, which the released keyframe planning is written for.
    @objc public static let framesPerClip = 149
    /// The frame rate of the final video and of the music features.
    @objc public static let framesPerSecond = 30.0

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKMLXT5EncoderNet>
    private let globalStage: NFKMLXStage<NFKMLXWanDancerPipeline>
    private let localStage: NFKMLXStage<NFKMLXWanDancerPipeline>
    private let segmenter: NFKMLXSentencePieceSegmenter
    private let globalAdapters: NFKWanAdapterList
    private let localAdapters: NFKWanAdapterList

    /// The global stage's steps, 48 by the released script.
    @objc public var globalSteps: Int = 48
    /// The local stage's steps, 24 by the released script.
    @objc public var localSteps: Int = 24
    /// The classifier-free guidance scale, 5 by the released scripts.
    @objc public var guidance: Float = 5
    /// The flow shift, 5 by the released scripts.
    @objc public var shift: Float = 5
    /// The clip's width and height, 720 × 1280 by the released scripts.
    @objc public var width: Int = 720
    @objc public var height: Int = 1280
    /// Whether the unconditional pass skips block 9, as both released scripts set it.
    @objc public var skipsBlock = true

    /// Whether the encoders and both transformers stay loaded together between videos.
    @objc public var holdsStagesResident: Bool { staging.resident }

    init(resident: Bool, segmenter: NFKMLXSentencePieceSegmenter, adapters: (global: NFKWanAdapterList, local: NFKWanAdapterList),
         loadTextEncoder: @escaping () throws -> NFKMLXT5EncoderNet,
         loadGlobal: @escaping () throws -> NFKMLXWanDancerPipeline, loadLocal: @escaping () throws -> NFKMLXWanDancerPipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage(loadTextEncoder)
        globalStage = NFKMLXStage(loadGlobal)
        localStage = NFKMLXStage(loadLocal)
        self.segmenter = segmenter
        globalAdapters = adapters.global
        localAdapters = adapters.local
        super.init()
    }

    // MARK: Factories

    /// The bits the transformers' block weights are packed to, or 0.
    @objc public internal(set) var quantizationBits = 0

    /// How many transformer blocks each stage reads from the release on every pass.
    @objc public internal(set) var streamedLayerCount = 0

    func loadResident() throws {
        try staging.use(textStage) { _ in }
        NFKMLXGPU.clearCache()
        try staging.use(globalStage) { _ in }
        try staging.use(localStage) { _ in }
    }

    /// Assembles the model from a downloaded release directory, holding it as ``NFKMLXResidency/automatic`` decides.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXWanDancerGenerator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    ///
    /// @discussion umT5-XXL is stored at bfloat16 (11.4 GB) and runs at float32 where the plan affords it. Each
    /// transformer is 34.5 GB at bfloat16 and shares its stage with the autoencoder; the CLIP vision tower (2.5 GB at
    /// float32) is one copy both stages read. ``NFKMLXResidency/staged`` holds one stage at a time, so a run loads the
    /// global transformer, then the local one. A 64 GB machine packs the transformers
    /// (``generator(directoryURL:residency:precision:quantizationBits:)``) or streams their blocks. The release has
    /// no routed experts, so ``NFKMLXResidency/paged`` holds it as ``NFKMLXResidency/staged`` does.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXWanDancerGenerator {
        try generator(directoryURL: directoryURL, residency: residency, precision: .checkpoint, bits: nil)
    }

    /// Assembles the model holding it as `residency` says, every stage's weights at `precision`, and the
    /// transformers' block weights packed to `quantizationBits` (2 to 8; 0 packs nothing) as each block loads.
    @objc(generatorWithDirectoryURL:residency:precision:quantizationBits:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                                 quantizationBits: Int) throws -> NFKMLXWanDancerGenerator {
        try generator(directoryURL: directoryURL, residency: residency, precision: precision,
                      bits: quantizationBits == 0 ? nil : quantizationBits)
    }

    /// Assembles the model, every stage loaded as it is used, at `precision`.
    @objc(generatorWithDirectoryURL:precision:error:)
    public static func generator(directoryURL: URL, precision: NFKMLXWeightPrecision) throws -> NFKMLXWanDancerGenerator {
        try generator(directoryURL: directoryURL, residency: .staged, precision: precision, bits: nil)
    }

    static func generator(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                          bits: Int?, groupSize: Int = NFKMLXLoadOptions.defaultQuantizationGroupSize,
                          budget: Int? = nil, streamedBlocks: Set<Int>? = nil) throws -> NFKMLXWanDancerGenerator {
        let configuration = try NFKMLXWanDancerConfiguration.reading(directoryURL.appendingPathComponent("config.json"))
        let segmenter = try NFKMLXWanFunRelease.segmenter(directoryURL)
        let textURL = directoryURL.appendingPathComponent(NFKMLXWanOriginalRelease.textEncoderFile)
        let vaeURL = directoryURL.appendingPathComponent(NFKMLXWanOriginalRelease.wan21AutoencoderFile)
        let imageURL = directoryURL.appendingPathComponent(imageEncoderFile)
        let files = (global: directoryURL.appendingPathComponent(stageFiles.global),
                     local: directoryURL.appendingPathComponent(stageFiles.local))
        let footprint = try (bits.map { bits in
            Result { NFKMLXStageFootprint(bytes: try NFKMLXWanBlockedLoad.quantizedFootprint(
                directory: files.local, precision: precision, bits: bits, groupSize: groupSize).bytes) }
        } ?? Result { try NFKMLXWanBlockedLoad.footprint(directory: files.local, precision: precision) }).get()
        let textBytes = try fileBytes(textURL)
        let shared = try fileBytes(vaeURL) + (try fileBytes(imageURL)) / 2
        let transformerStage = NFKMLXStageFootprint(bytes: footprint.bytes + shared, layerBytes: footprint.layerBytes)
        let stages = [NFKMLXStageFootprint(bytes: textBytes, widenedBytes: 2 * textBytes), transformerStage, transformerStage]
        let plan = try NFKMLXResidencyBudget.plan(stages, residency: residency, budget: budget ?? NFKMLXResidencyBudget.current())
        let streamed = streamedBlocks ?? (plan.streams(1)
            ? Set(plan.heldLayers(1, of: transformerStage).upperBound ..< footprint.layerBytes.count) : [])
        let textInFloat32 = plan.widens(0)
        let adapters = (global: NFKWanAdapterList(), local: NFKWanAdapterList())
        let imageEncoder = NFKWanDancerSharedEncoder { try NFKMLXWanAnimateImageEncoderNet.fromOpenCLIP(imageURL, precision: precision) }
        func pipeline(_ file: URL, _ list: NFKWanAdapterList) -> () throws -> NFKMLXWanDancerPipeline {
            {
                let net = NFKMLXWanDancerNet(configuration)
                let deltas = try list.deltas(naming: { $0 })
                if bits != nil || !streamed.isEmpty {
                    net.layerStream = try NFKMLXWanBlockedLoad.load(
                        into: net, blocks: net.blocks, fromDirectory: file, precision: precision, streamed: streamed,
                        bits: bits, groupSize: groupSize, adapters: deltas, adapt: NFKMLXWanDancerNet.adapted)
                } else {
                    try NFKMLXWanDancerNet.loadWeights(into: net, file: file, precision: precision)
                    for delta in deltas {
                        try NFKMLXWanLoRA.apply(delta, to: net)
                    }
                }
                // The autoencoder is stored at float32 and runs at it, as the reference holds it.
                let autoencoder = try NFKMLXWanOriginalRelease.autoencoder(from: vaeURL)
                return NFKMLXWanDancerPipeline(net: net, vae: autoencoder.net, latentsMean: autoencoder.mean,
                                               latentsStd: autoencoder.std, imageEncoder: try imageEncoder.value())
            }
        }
        let generator = NFKMLXWanDancerGenerator(
            resident: plan.holdsStagesResident, segmenter: segmenter, adapters: adapters,
            loadTextEncoder: {
                try NFKMLXWanOriginalRelease.textEncoder(from: textURL, precision: textInFloat32 ? .float32 : .checkpoint)
            },
            loadGlobal: pipeline(files.global, adapters.global), loadLocal: pipeline(files.local, adapters.local))
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
        NFKMLXReleaseComponent(required: ["config.json", NFKMLXWanOriginalRelease.wan21AutoencoderFile, imageEncoderFile]),
        NFKMLXReleaseComponent(required: [stageFiles.global]),
        NFKMLXReleaseComponent(required: [stageFiles.local]),
    ]

    /// Downloads the release (``releaseRepo``, or `repo` where given) and assembles the model, holding it as
    /// `residency` says.
    ///
    /// @discussion The release is about 85 GB. A file already in the cache is not fetched again. The call
    /// blocks on the network, so run it off the main and render threads.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXWanDancerGenerator {
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
                                 completionHandler: @escaping (NFKMLXWanDancerGenerator?, Error?) -> Void) {
        NFKMLXReleaseDownload.asyncValue(completionHandler) {
            try generator(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: residency)
        }
    }

    // MARK: Adapters

    /// Merges a LoRA in DiffSynth-Studio's layout (``saveAdapters(of:to:preset:)``) into the global or the local
    /// transformer at `scale`, now where it is loaded and on every later load. Introduced in InferKit 0.4.0.
    @objc(addAdapterAtURL:scale:global:error:)
    public func addAdapter(at url: URL, scale: Float, global: Bool) throws {
        try staging.exclusively {
            let stage = global ? globalStage : localStage
            if stage.isHeld {
                let deltas = try NFKMLXWanLoRA.deltas(adapterURL: url, naming: { $0 }, scale: scale).deltas
                _ = try staging.use(stage) { try NFKMLXWanLoRA.apply(deltas, to: $0.net) }
            }
            (global ? globalAdapters : localAdapters).add(url, scale: scale)
        }
    }

    /// Removes every adapter. A loaded transformer is released and loads again without them.
    @objc public func removeAdapters() {
        staging.exclusively {
            globalAdapters.removeAll()
            localAdapters.removeAll()
            globalStage.release()
            localStage.release()
            NFKMLXGPU.clearCache()
        }
    }

    /// The adapters ``addAdapter(at:scale:global:)`` added to both stages.
    @objc public var adapterCount: Int { staging.exclusively { globalAdapters.entries.count + localAdapters.entries.count } }

    /// Runs `body` with the global or the local stage's sampler.
    func withPipeline<T>(global: Bool, _ body: (NFKMLXWanDancerPipeline) throws -> T) throws -> T {
        try staging.exclusively { try staging.use(global ? globalStage : localStage, body) }
    }

    // MARK: Pictures

    /// `crop_and_resize`: the picture scaled to fit `width` × `height` (Pillow's bilinear filter, the sides
    /// rounded half to even), centered on mid-gray (127), and the box it fills.
    static func fitted(_ picture: NFKMLXWanDancerPicture, width: Int, height: Int) -> (NFKMLXWanDancerPicture, (Int, Int, Int, Int)) {
        let scale = Swift.min(Double(width) / Double(picture.width), Double(height) / Double(picture.height))
        let resizedHeight = Int((Double(picture.height) * scale).rounded(.toNearestOrEven))
        let resizedWidth = Int((Double(picture.width) * scale).rounded(.toNearestOrEven))
        let resized = resizedWidth == picture.width && resizedHeight == picture.height ? picture.rgb
            : NFKMLXPILResample.resampled(picture.rgb, width: picture.width, height: picture.height, toWidth: resizedWidth,
                                          toHeight: resizedHeight, filter: .bilinear, channels: 3)
        var canvas = [UInt8](repeating: 127, count: width * height * 3)
        let left = (width - resizedWidth) / 2, top = (height - resizedHeight) / 2
        for y in 0 ..< resizedHeight {
            for x in 0 ..< resizedWidth {
                for c in 0 ..< 3 {
                    canvas[((top + y) * width + left + x) * 3 + c] = resized[(y * resizedWidth + x) * 3 + c]
                }
            }
        }
        return (NFKMLXWanDancerPicture(rgb: canvas, width: width, height: height),
                (left, top, left + resizedWidth, top + resizedHeight))
    }

    /// Frames `[N, H, W, 3]` in `-1...1` as the reference saves them: `(x + 1) · 127.5` clipped and truncated to 8 bits.
    static func pictures(_ frames: MLXArray) -> [NFKMLXWanDancerPicture] {
        let bytes = clip((frames + 1) * 127.5, min: 0, max: 255).asType(.uint8)
        eval(bytes)
        return (0 ..< frames.dim(0)).map {
            NFKMLXWanDancerPicture(rgb: bytes[$0].asArray(UInt8.self), width: frames.dim(2), height: frames.dim(1))
        }
    }

    /// A picture's crop to `box`.
    static func cropped(_ picture: NFKMLXWanDancerPicture, to box: (Int, Int, Int, Int)) -> NFKMLXWanDancerPicture {
        let width = box.2 - box.0, height = box.3 - box.1
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                for c in 0 ..< 3 {
                    rgb[(y * width + x) * 3 + c] = picture.rgb[((box.1 + y) * picture.width + box.0 + x) * 3 + c]
                }
            }
        }
        return NFKMLXWanDancerPicture(rgb: rgb, width: width, height: height)
    }

    // MARK: Planning

    /// The global stage's frame rate for `featureFrames` music frames: 30 over the whole frames per keyframe
    /// clip frame, rounded half up, at four decimals.
    static func globalFrameRate(featureFrames: Int) -> Double {
        let rate = framesPerSecond / Double(Int(Double(featureFrames) / Double(framesPerClip) + 0.5))
        return (rate * 10000).rounded(.toNearestOrEven) / 10000
    }

    /// The local stage's keyframe masks (`process_global_video_firstlastframe`) for a track of `totalFrames` at 30
    /// per second planned by `keyframeCount` global frames, before each segment's last frame is shared with the
    /// next segment's first: every `totalFrames / keyframeCount` frames of each 149-frame segment, and the last
    /// segment's final frame.
    static func keyframeMasks(totalFrames: Int, keyframeCount: Int) -> [[Bool]] {
        let segments = Int((Double(totalFrames) / Double(framesPerClip)).rounded(.up))
        let interval = Double(totalFrames) / Double(keyframeCount)
        return (0 ..< segments).map { segment -> [Bool] in
            var mask = [Bool](repeating: false, count: framesPerClip)
            let end: Double
            if segment != segments - 1 {
                end = Double(framesPerClip)
            } else {
                let endIndex = totalFrames - framesPerClip * segment - 1
                mask[endIndex] = true
                end = Double(endIndex)
            }
            var count = 0
            while Double(count) * interval < end - interval {
                mask[Int((interval * Double(count)).rounded(.up))] = true
                count += 1
            }
            return mask
        }
    }

    /// The track's duration as the container reports it, to the hundredth of a second.
    static func containerDuration(samples: Int, sampleRate: Int) -> Double {
        ((Double(samples) / Double(sampleRate) + 0.005) * 100).rounded(.down) / 100
    }

    /// The local stage's music clips (`get_music_clip_149f`): 149 frames of the track at a time, the last cut at the
    /// track's end, none starting within 0.2 s of it, each written and read as 16-bit PCM as libsndfile converts
    /// (`floor(x · 32768)`, clipped).
    static func musicClips(_ samples: [Float], sampleRate: Int) -> [[Float]] {
        let total = containerDuration(samples: samples.count, sampleRate: sampleRate)
        let duration = Double(framesPerClip) / framesPerSecond
        var clips = [[Float]](), t = 0.0
        while t + 0.2 < total {
            let end = Swift.min(t + duration, total)
            let start = Int(t * Double(sampleRate)), stop = Swift.min(Int(end * Double(sampleRate)), samples.count)
            clips.append(samples[start ..< Swift.max(start, stop)].map {
                Float(Swift.min(32767, Swift.max(-32768, ($0 * 32768).rounded(.down)))) / 32768
            })
            t += duration
        }
        return clips
    }

    // MARK: Stages

    func features(_ text: String) throws -> MLXArray {
        try staging.exclusively {
            try staging.with(textStage) {
                NFKMLXWanFunRelease.features(NFKMLXWanPromptCleaning.cleaned(text), encoder: $0, segmenter: segmenter)
            }
        }
    }

    /// Standard-normal noise `[16, F', h, w]` for a clip of `frames` from `seed`.
    func noise(frames: Int, seed: UInt64) -> MLXArray {
        MLXRandom.seed(seed)
        return MLXRandom.normal([16, (frames - 1) / 4 + 1, height / 8, width / 8])
    }

    /// Music features at the bfloat16 the released scripts hand them over at.
    static func scriptMusic(_ samples: [Float], sampleRate: Int) -> MLXArray {
        NFKMLXWanDancerMusic.features(samples: samples, sampleRate: sampleRate).asType(.bfloat16).asType(.float32)
    }

    /// The global stage (`gen_video_global.py`): the reference image as the first keyframe of one 149-frame clip at
    /// the track's reduced frame rate, its prompt ending in that rate (`帧率是…`), each latent frame decoded alone.
    public func keyframeVideo(prompt: String, negativePrompt: String? = nil, reference: NFKMLXWanDancerPicture,
                              music: [Float], sampleRate: Int, seed: UInt64) throws -> NFKMLXWanDancerKeyframeVideo {
        let features = Self.scriptMusic(music, sampleRate: sampleRate)
        let fps = Self.globalFrameRate(featureFrames: features.dim(0))
        let positive = try self.features(prompt + String(format: "帧率是%.4f", fps))
        let negative = try self.features(negativePrompt ?? Self.defaultNegativePrompt)
        let (image, box) = Self.fitted(reference, width: width, height: height)
        let keyframes = MLXArray.zeros([Self.framesPerClip, height, width, 3])
        keyframes[0] = image.array
        let mask = [true] + [Bool](repeating: false, count: Self.framesPerClip - 1)
        let clip = NFKMLXWanDancerClip(keyframes: keyframes, keyframeMask: mask, reference: image.array, music: features, fps: fps)
        let frames = try sample(globalStage, clip: clip, positive: positive, negative: negative, steps: globalSteps,
                                seed: seed, framewise: true)
        return NFKMLXWanDancerKeyframeVideo(frames: Self.pictures(frames), box: box)
    }

    /// The local stage (`gen_video_local.py`): the track in 149-frame segments at 30 frames a second, each keyed by
    /// the global frames its mask names and its own music clip, its prompt ending `, 帧率是30fps。`, seeded
    /// `10 · segment + seed`. Returns the frames `[N, h, w, 3]` in `0...1` within the reference image's box, trimmed to
    /// the track's duration less 0.2 s.
    public func danceVideo(prompt: String, negativePrompt: String? = nil, reference: NFKMLXWanDancerPicture,
                           keyframes global: NFKMLXWanDancerKeyframeVideo, music: [Float], sampleRate: Int,
                           seed: UInt64) throws -> MLXArray {
        let duration = Self.containerDuration(samples: music.count, sampleRate: sampleRate)
        let totalFrames = Int(duration * Self.framesPerSecond)
        var masks = Self.keyframeMasks(totalFrames: totalFrames, keyframeCount: global.frames.count)
        var plannedKeyframes = [[NFKMLXWanDancerPicture?]]()
        var next = 0
        for mask in masks {
            var segment = [NFKMLXWanDancerPicture?](repeating: nil, count: Self.framesPerClip)
            for j in 0 ..< Self.framesPerClip where mask[j] {
                guard next < global.frames.count else {
                    throw NFKMLXError.unsupportedConfiguration("the keyframe plan reads past the global video's \(global.frames.count) frames")
                }
                segment[j] = Self.fitted(Self.cropped(global.frames[next], to: global.box), width: width, height: height).0
                next += 1
            }
            plannedKeyframes.append(segment)
        }
        for segment in 0 ..< Swift.max(0, plannedKeyframes.count - 1) {
            plannedKeyframes[segment][Self.framesPerClip - 1] = plannedKeyframes[segment + 1][0]
            masks[segment][Self.framesPerClip - 1] = true
        }
        let (image, box) = Self.fitted(reference, width: width, height: height)
        if Swift.min(height, width) < 512 {
            plannedKeyframes[0][0] = image
        }
        let positive = try features(prompt + ", 帧率是30fps。")
        let negative = try features(negativePrompt ?? Self.defaultNegativePrompt)
        var clips = [MLXArray]()
        for (index, samples) in Self.musicClips(music, sampleRate: sampleRate).enumerated() where index < masks.count {
            let keyframes = stacked(plannedKeyframes[index].map { $0?.array ?? MLXArray.zeros([height, width, 3]) })
            let clip = NFKMLXWanDancerClip(keyframes: keyframes, keyframeMask: masks[index], reference: image.array,
                                           music: Self.scriptMusic(samples, sampleRate: sampleRate), fps: Self.framesPerSecond)
            let frames = try sample(localStage, clip: clip, positive: positive, negative: negative, steps: localSteps,
                                    seed: UInt64(index * 10) &+ seed, framewise: false)
            clips.append(frames[0..., box.1 ..< box.3, box.0 ..< box.2])
        }
        let kept = Int(((duration - 0.2) * Self.framesPerSecond).rounded(.up))
        let video = concatenated(clips, axis: 0)
        return ((video[0 ..< Swift.min(kept, video.dim(0))] + 1) / 2)
    }

    /// Both stages: the keyframe video, then the dance video it plans.
    public func video(prompt: String, localPrompt: String? = nil, negativePrompt: String? = nil,
                      reference: NFKMLXWanDancerPicture, music: [Float], sampleRate: Int, seed: UInt64) throws -> MLXArray {
        let keyframes = try keyframeVideo(prompt: prompt, negativePrompt: negativePrompt, reference: reference, music: music,
                                          sampleRate: sampleRate, seed: seed)
        return try danceVideo(prompt: localPrompt ?? prompt, negativePrompt: negativePrompt, reference: reference,
                              keyframes: keyframes, music: music, sampleRate: sampleRate, seed: seed)
    }

    /// One clip through a stage's sampler, its frames `[F, H, W, 3]` in `-1...1`.
    private func sample(_ stage: NFKMLXStage<NFKMLXWanDancerPipeline>, clip: NFKMLXWanDancerClip, positive: MLXArray,
                        negative: MLXArray, steps: Int, seed: UInt64, framewise: Bool) throws -> MLXArray {
        try staging.exclusively {
            try staging.use(stage) { pipeline in
                pipeline.steps = steps
                pipeline.guidance = guidance
                pipeline.shift = shift
                pipeline.skipsBlock = skipsBlock
                let inputs = pipeline.inputs(clip)
                let latents = pipeline.denoise(noise(frames: clip.keyframes.dim(0), seed: seed), positive: positive,
                                               negative: negative, inputs: inputs)
                return framewise ? pipeline.decodedFramewise(latents) : pipeline.decoded(latents)
            }
        }
    }
}

/// One image encoder both stages read, loaded when the first stage loads.
final class NFKWanDancerSharedEncoder {
    private let load: () throws -> NFKMLXWanAnimateImageEncoderNet
    private var loaded: NFKMLXWanAnimateImageEncoderNet?
    private let lock = NSLock()

    init(_ load: @escaping () throws -> NFKMLXWanAnimateImageEncoderNet) { self.load = load }

    func value() throws -> NFKMLXWanAnimateImageEncoderNet {
        try lock.withLock {
            if let loaded { return loaded }
            let net = try load()
            loaded = net
            return net
        }
    }
}
