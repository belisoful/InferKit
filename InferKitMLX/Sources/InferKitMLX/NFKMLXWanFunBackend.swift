//
//  NFKMLXWanFunBackend.swift
//  InferKitMLX
//
//  The Wan 2.2 Fun releases behind the InferKit contract, and the Objective-C form of their conditions.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

extension NFKMLXVideoInputKey {
    /// The reference image a Wan 2.2 Fun Control release reads as a frame of tokens ahead of the clip (a
    /// `CGImage`, `CVPixelBuffer`, or texture). Introduced in InferKit 0.4.0.
    @objc public static let referenceImage = "NFKMLXInputReferenceImage"
    /// The video an inpainting mask keeps from (an `NFKVideoAsset` with a file URL). Introduced in InferKit
    /// 0.4.0.
    @objc public static let inpaintVideo = "NFKMLXInputInpaintVideo"
    /// The inpainting mask as a video (an `NFKVideoAsset` with a file URL): white where the clip is
    /// generated, black where ``inpaintVideo`` is kept. Introduced in InferKit 0.4.0.
    @objc public static let inpaintMask = "NFKMLXInputInpaintMask"
    /// A camera trajectory: an ``NFKMLXWanCameraTrajectory``, or the text of a CameraCtrl pose file.
    /// Introduced in InferKit 0.4.0.
    @objc public static let cameraTrajectory = "NFKMLXInputCameraTrajectory"
}

extension NFKMLXWanFunGenerator {
    /// The request inputs the backend reads.
    static func backendInputKeys(_ variant: NFKMLXWanFunVariant) -> Set<String> {
        var keys: Set<String> = [NFKInputPrompt, NFKInputNegativePrompt, NFKInputImage, NFKMLXVideoInputKey.lastImage,
                                 NFKMLXVideoInputKey.inpaintVideo, NFKMLXVideoInputKey.inpaintMask]
        switch variant.rawValue % 3 {
        case 0: keys.formUnion([NFKInputVideo, NFKMLXVideoInputKey.referenceImage])
        case 2: keys.insert(NFKMLXVideoInputKey.cameraTrajectory)
        default: break
        }
        return keys
    }

    /// The request parameters the backend reads.
    static let backendParameterKeys: Set<String> = [
        NFKParameterFrameCount, NFKParameterWidth, NFKParameterHeight, NFKParameterFramesPerSecond,
        NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale,
    ]

    /// The default clip size and length VideoX-Fun's examples run: 121 frames of 704 × 1280 for a 5B
    /// release, 81 of 480 × 832 for A14B.
    var defaultClip: (frames: Int, width: Int, height: Int) {
        variant.hasExperts ? (81, 832, 480) : (121, 1280, 704)
    }

    /// An image at the clip's size, `[height, width, 3]` in `[0, 1]`: resized by PIL's 8-bit bicubic filter,
    /// as VideoX-Fun's example helpers resize an image (`Image.resize`).
    static func preparedImage(_ image: CGImage, width: Int, height: Int) throws -> MLXArray {
        try prepared(image, width: width, height: height, filter: .bicubic)
    }

    static func prepared(_ image: CGImage, width: Int, height: Int, filter: NFKMLXPILResample.Filter) throws -> MLXArray {
        let (rgb, sourceWidth, sourceHeight) = try NFKMLXImageBridge.rgbBytes(
            from: image, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let resized = sourceWidth == width && sourceHeight == height ? rgb
            : NFKMLXPILResample.resampled(rgb, width: sourceWidth, height: sourceHeight, toWidth: width,
                                          toHeight: height, filter: filter)
        return MLXArray(resized.map { Float($0) / 255 }).reshaped([height, width, 3])
    }

    /// A video's frames at the clip's rate and size, `[frames, height, width, 3]` in `[0, 1]`: the source frame
    /// nearest each target time (VideoX-Fun's `get_video_to_video_latent`), the first `frames` of them, each
    /// resized by PIL's 8-bit bilinear filter. The helper resizes with OpenCV's bilinear `cv2.resize`, which
    /// does not filter on reduction; the frames agree where the source is at the clip's size. A video shorter
    /// than the clip is refused.
    static func preparedVideo(_ clip: NFKMLXVideoFile.Clip, frames: Int, framesPerSecond: Double, width: Int,
                              height: Int) throws -> MLXArray {
        let indices = NFKWanAnimatePreprocess.frameIndices(count: clip.frames.count, sourceRate: clip.framesPerSecond,
                                                            targetRate: framesPerSecond)
        guard indices.count >= frames else {
            throw NFKMLXError.unsupportedConfiguration(
                "the video holds \(indices.count) frames at \(framesPerSecond) per second; the clip needs \(frames)")
        }
        return stacked(try indices.prefix(frames).map {
            try prepared(clip.frames[$0], width: width, height: height, filter: .bilinear)
        })
    }

    static func video(_ request: NFKInferenceRequest, _ key: String) throws -> NFKMLXVideoFile.Clip? {
        guard let value = request.input(forKey: key) else { return nil }
        guard let asset = value as? NFKVideoAsset, let url = asset.fileURL else {
            throw NFKMLXError.unsupportedConfiguration("\(key) is an NFKVideoAsset with a file URL")
        }
        return try NFKMLXVideoFile.read(url)
    }

    static func trajectory(_ value: Any?) throws -> NFKMLXWanCameraTrajectory? {
        switch value {
        case let trajectory as NFKMLXWanCameraTrajectory: return trajectory
        case let text as String: return try NFKMLXWanCameraTrajectory(poseFileContents: text)
        case nil: return nil
        default: throw NFKMLXError.unsupportedConfiguration("a camera trajectory is an NFKMLXWanCameraTrajectory or pose text")
        }
    }

    /// The conditions of one clip from its pictures, each at the clip's size.
    func conditions(control: MLXArray?, start: MLXArray?, end: MLXArray?, reference: MLXArray?,
                    inpaint: (video: MLXArray, mask: MLXArray)?, trajectory: NFKMLXWanCameraTrajectory?,
                    frames: Int, width: Int, height: Int) throws -> NFKMLXWanFunConditions {
        var conditions = NFKMLXWanFunConditions(controlVideo: control, referenceImage: reference)
        if let inpaint {
            conditions.inpaintVideo = inpaint.video
            conditions.inpaintMask = inpaint.mask
        } else if let start {
            let built = NFKMLXWanFunConditions.startImage(start, endImage: end, frames: frames)
            conditions.inpaintVideo = built.video
            conditions.inpaintMask = built.mask
        } else if end != nil {
            throw NFKMLXError.unsupportedConfiguration("an end image needs a start image")
        }
        conditions.cameraEmbedding = try trajectory?.plucker(width: width, height: height, frames: frames)
        return conditions
    }

    /// The generator behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt` (required) and `NFKInputNegativePrompt`; a start image
    /// under `NFKInputImage` and an end image under ``NFKMLXVideoInputKey/lastImage``, or a video and its mask
    /// under ``NFKMLXVideoInputKey/inpaintVideo`` and ``NFKMLXVideoInputKey/inpaintMask``; for a Control release
    /// the control video under `NFKInputVideo` and a reference image under
    /// ``NFKMLXVideoInputKey/referenceImage``; for a Control-Camera release the trajectory under
    /// ``NFKMLXVideoInputKey/cameraTrajectory`` (required). Every picture is resized to the clip, and every
    /// video resampled to the clip's rate, as ``preparedImage(_:width:height:)`` and
    /// ``preparedVideo(_:frames:framesPerSecond:width:height:)`` describe. It returns an `NFKVideoAsset` (a
    /// QuickTime movie in the temporary directory) under `NFKOutputVideo`. Parameters: `NFKParameterFrameCount`,
    /// `NFKParameterWidth`, `NFKParameterHeight` (``defaultClip``), `NFKParameterFramesPerSecond`
    /// (``framesPerSecond``), `NFKParameterSeed` (0), `NFKParameterSteps` and `NFKParameterGuidanceScale`
    /// (``steps`` and ``guidance``). A request's values apply to that request only. Introduced in InferKit
    /// 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        let identifier = variant.modelName
        return NFKMLXGenerationBackend(identifier: identifier, inputKeys: Self.backendInputKeys(variant),
                                       parameterKeys: Self.backendParameterKeys,
                                       outputKeys: [NFKOutputVideo]) { [self] request, job in
            let prompt = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            let negative = NFKMLXGenerationBackend.text(request, NFKInputNegativePrompt)
            func parameter(_ key: String) -> NSNumber? { NFKMLXGenerationBackend.number(request, key) }
            let defaults = defaultClip
            let size = try clipSize(frames: parameter(NFKParameterFrameCount)?.intValue ?? defaults.frames,
                                    width: parameter(NFKParameterWidth)?.intValue ?? defaults.width,
                                    height: parameter(NFKParameterHeight)?.intValue ?? defaults.height)
            let rate = parameter(NFKParameterFramesPerSecond)?.doubleValue ?? framesPerSecond
            func image(_ key: String) throws -> MLXArray? {
                try NFKMLXGenerationBackend.image(request, key).map {
                    try Self.preparedImage($0, width: size.width, height: size.height)
                }
            }
            func clip(_ key: String) throws -> MLXArray? {
                try Self.video(request, key).map {
                    try Self.preparedVideo($0, frames: size.frames, framesPerSecond: rate, width: size.width,
                                           height: size.height)
                }
            }
            var inpaint: (MLXArray, MLXArray)?
            if let video = try clip(NFKMLXVideoInputKey.inpaintVideo) {
                guard let mask = try clip(NFKMLXVideoInputKey.inpaintMask) else {
                    throw NFKMLXError.unsupportedConfiguration("an inpainting video needs its mask video")
                }
                inpaint = (video, mask.mean(axis: -1))
            }
            let conditions = try self.conditions(
                control: try clip(NFKInputVideo), start: try image(NFKInputImage),
                end: try image(NFKMLXVideoInputKey.lastImage), reference: try image(NFKMLXVideoInputKey.referenceImage),
                inpaint: inpaint, trajectory: try Self.trajectory(request.input(forKey: NFKMLXVideoInputKey.cameraTrajectory)),
                frames: size.frames, width: size.width, height: size.height)
            let saved = (steps, guidance)
            defer { (steps, guidance) = saved }
            steps = parameter(NFKParameterSteps)?.intValue ?? steps
            guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? guidance
            job.reportProgress(0)
            let result = try generate(prompt: prompt, negativePrompt: negative, conditions: conditions,
                                      frames: size.frames, width: size.width, height: size.height,
                                      seed: NFKMLXGenerationBackend.seed(request),
                                      progress: { job.reportProgress(0.99 * $0) }).clip
            if job.status == .cancelled { return nil }
            let output = try NFKMLXGenerationBackend.videoResult(result, framesPerSecond: rate, identifier: identifier)
            job.reportProgress(1)
            return output
        }
    }

    /// Generates a clip and returns its frames as `CGImage`s, the conditions given as pictures: the control
    /// frames (a Control release), a start and an end image, a reference image, and a camera trajectory (a
    /// Control-Camera release), each optional and resized to the clip. Objective-C reads each element as
    /// `(__bridge CGImageRef)`. Introduced in InferKit 0.4.0.
    @objc(framesForPrompt:negativePrompt:controlFrames:startImage:endImage:referenceImage:cameraTrajectory:frames:width:height:seed:error:)
    public func frames(forPrompt prompt: String, negativePrompt: String?, controlFrames: [Any]?, startImage: CGImage?,
                       endImage: CGImage?, referenceImage: CGImage?, cameraTrajectory: NFKMLXWanCameraTrajectory?,
                       frames: Int, width: Int, height: Int, seed: UInt64) throws -> [Any] {
        let size = try clipSize(frames: frames, width: width, height: height)
        func picture(_ image: CGImage?) throws -> MLXArray? {
            try image.map { try Self.preparedImage($0, width: size.width, height: size.height) }
        }
        let control = try controlFrames.map { values -> MLXArray in
            let images = try values.map { try NFKMLXGenerationBackend.cgImage(fromInput: $0) }
            guard images.count >= size.frames else {
                throw NFKMLXError.unsupportedConfiguration("the control video holds \(images.count) frames; the clip needs \(size.frames)")
            }
            return stacked(try images.prefix(size.frames).map {
                try Self.prepared($0, width: size.width, height: size.height, filter: .bilinear)
            })
        }
        let conditions = try self.conditions(control: control, start: try picture(startImage), end: try picture(endImage),
                                             reference: try picture(referenceImage), inpaint: nil,
                                             trajectory: cameraTrajectory, frames: size.frames, width: size.width,
                                             height: size.height)
        let clip = try video(forPrompt: prompt, negativePrompt: negativePrompt, conditions: conditions,
                             frames: size.frames, width: size.width, height: size.height, seed: seed)
        return try (0 ..< clip.dim(0)).map { try NFKMLXImageBridge.cgImage(from: clip[$0], options: NFKMLXImageOptions()) }
    }

    // MARK: Backend factories

    /// Assembles the model from a downloaded release directory, held as ``NFKMLXResidency/automatic``
    /// decides, behind the InferKit contract (``inferenceBackend()``). Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL).inferenceBackend()
    }

    /// ``backend(directoryURL:)`` holding the stages as `residency` says. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL, residency: residency).inferenceBackend()
    }

    /// ``backend(directoryURL:residency:)`` with the weights at `precision` and the transformers' block weights
    /// packed to `quantizationBits`, as ``generator(directoryURL:residency:precision:quantizationBits:)`` reads
    /// them. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:precision:quantizationBits:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                               quantizationBits: Int) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL, residency: residency, precision: precision,
                      quantizationBits: quantizationBits).inferenceBackend()
    }

    /// Downloads `variant`'s release (or `repo`) and returns the model behind the InferKit contract.
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithVariant:repo:revision:cacheDirectoryURL:residency:error:)
    public static func backend(variant: NFKMLXWanFunVariant, repo: String?, revision: String?,
                               cacheDirectoryURL: URL?, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try generator(variant: variant, repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                          residency: residency).inferenceBackend()
        }
    }

    /// The asynchronous form of ``backend(variant:repo:revision:cacheDirectoryURL:residency:)``. The handler
    /// runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithVariant:repo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func backend(variant: NFKMLXWanFunVariant, repo: String?, revision: String?,
                               cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(variant: variant, repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                        residency: residency)
        }
    }

    /// Registers every variant's backend under its ``NFKMLXWanFunVariant/modelName``. The registry's URL is the
    /// release directory; nil is refused, and a directory holding another variant is refused. Introduced in
    /// InferKit 0.4.0.
    @objc public static func register() {
        for variant in NFKMLXWanFunVariant.allCases {
            NFKMLXGenerationBackend.registerDirectoryModel(variant.modelName) { directory in
                let generator = try generator(directoryURL: directory)
                guard generator.variant == variant else {
                    throw NFKMLXError.unsupportedConfiguration(
                        "\(directory.lastPathComponent) holds \(generator.variant.modelName), not \(variant.modelName)")
                }
                return generator.inferenceBackend()
            }
        }
    }
}
