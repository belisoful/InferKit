//
//  NFKMLXWanS2VBackend.swift
//  InferKitMLX
//
//  Wan 2.2 S2V-14B speech-to-video behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

extension NFKMLXGenerationParameterKey {
    /// The most segments a segmented video generator generates (`NSNumber` integer). Wan 2.2 S2V reads it
    /// (`--num_clip`); absent, it generates as many as the speech covers. Introduced in InferKit 0.4.0.
    @objc public static let segmentCount = "NFKMLXParameterSegmentCount"
    /// Whether a video generator starts the clip from its reference image (`NSNumber` Boolean). Wan 2.2 S2V reads
    /// it (`--start_from_ref`); its default is false. Introduced in InferKit 0.4.0.
    @objc public static let startsFromReference = "NFKMLXParameterStartsFromReference"
    /// The flow shift of a flow-matching sampler's schedule (`NSNumber` float). Wan 2.2 S2V reads it; its default
    /// is ``NFKMLXWanS2VGenerator/shift``. Introduced in InferKit 0.4.0.
    @objc public static let flowShift = "NFKMLXParameterFlowShift"
}

extension NFKMLXWanS2VGenerator {

    /// The request inputs the backend reads.
    static let backendInputKeys: Set<String> = [NFKInputPrompt, NFKInputNegativePrompt, NFKInputImage, NFKInputAudio,
                                                NFKInputVideo]

    /// The request parameters the backend reads.
    static let backendParameterKeys: Set<String> = [
        NFKParameterWidth, NFKParameterHeight, NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale,
        NFKMLXGenerationParameterKey.segmentLength, NFKMLXGenerationParameterKey.segmentCount,
        NFKMLXGenerationParameterKey.startsFromReference, NFKMLXGenerationParameterKey.flowShift,
    ]

    /// Mono 16 kHz samples from an audio input (an `NFKAudioAsset` or WAV `Data`, or an `AVAudioPCMBuffer`), the
    /// channels averaged as `librosa.load` averages them. Audio at another rate is refused.
    static func speechSamples(_ value: Any?) throws -> [Float] {
        guard let (samples, rate, channels) = NFKMLXLiveAudio.samples(from: value) else {
            throw NFKMLXError.unsupportedConfiguration("an S2V request needs speech under NFKInputAudio")
        }
        guard rate == NFKMLXWav2Vec2Processor.sampleRate else {
            throw NFKMLXError.unsupportedConfiguration(
                "the speech is at \(rate) Hz; Wan 2.2 S2V reads 16 kHz audio")
        }
        guard channels > 1 else { return samples }
        return (0 ..< samples.count / channels).map { frame in
            (0 ..< channels).reduce(Float(0)) { $0 + samples[frame * channels + $1] } / Float(channels)
        }
    }

    /// The first `count` frames of a pose video at 16 frames per second as the reference reads them
    /// (`read_last_n_frames(reverse=True)`): every `round(rate / 16)`-th frame from the first.
    static func poseFrames(_ clip: NFKMLXVideoFile.Clip, count: Int) -> [CGImage] {
        let interval = max(1, Int((clip.framesPerSecond / 16).rounded(.toNearestOrEven)))
        return stride(from: 0, to: clip.frames.count, by: interval).prefix(count).map { clip.frames[$0] }
    }

    /// The generator behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt` (required), `NFKInputNegativePrompt`, the reference image
    /// under `NFKInputImage` (required), the speech under `NFKInputAudio` (16 kHz, required, `speechSamples(_:)`),
    /// and a pose video under `NFKInputVideo` (an `NFKVideoAsset` with a file URL, `poseFrames(_:count:)`). It
    /// returns an `NFKVideoAsset` (a QuickTime movie at 16 frames per second in the temporary directory) under
    /// `NFKOutputVideo`. Parameters: `NFKParameterWidth` × `NFKParameterHeight` bound the clip's area
    /// (``maximumArea``), `NFKParameterSeed` (0), `NFKParameterSteps` and `NFKParameterGuidanceScale` (``steps``
    /// and ``guidance``), ``NFKMLXGenerationParameterKey/segmentLength`` (``framesPerSegment``),
    /// ``NFKMLXGenerationParameterKey/segmentCount``, ``NFKMLXGenerationParameterKey/startsFromReference``, and
    /// ``NFKMLXGenerationParameterKey/flowShift`` (``shift``). A request's values apply to that request only.
    /// Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        let identifier = Self.modelName
        return NFKMLXGenerationBackend(identifier: identifier, inputKeys: Self.backendInputKeys,
                                       parameterKeys: Self.backendParameterKeys,
                                       outputKeys: [NFKOutputVideo]) { [self] request, job in
            let prompt = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            let negative = NFKMLXGenerationBackend.text(request, NFKInputNegativePrompt)
            guard let image = try NFKMLXGenerationBackend.image(request, NFKInputImage) else {
                throw NFKMLXError.unsupportedConfiguration("an S2V request needs a reference image under NFKInputImage")
            }
            let speech = try Self.speechSamples(request.input(forKey: NFKInputAudio))
            func parameter(_ key: String) -> NSNumber? { NFKMLXGenerationBackend.number(request, key) }
            let saved = (steps, guidance, shift, framesPerSegment, maximumArea)
            defer { (steps, guidance, shift, framesPerSegment, maximumArea) = saved }
            steps = parameter(NFKParameterSteps)?.intValue ?? steps
            guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? guidance
            shift = parameter(NFKMLXGenerationParameterKey.flowShift)?.doubleValue ?? shift
            framesPerSegment = parameter(NFKMLXGenerationParameterKey.segmentLength)?.intValue ?? framesPerSegment
            if let width = parameter(NFKParameterWidth)?.intValue, let height = parameter(NFKParameterHeight)?.intValue {
                maximumArea = width * height
            }
            let segments = parameter(NFKMLXGenerationParameterKey.segmentCount)?.intValue
            let size = Self.clipSize(imageWidth: image.width, imageHeight: image.height, maximumArea: maximumArea)
            let reference = try Self.preparedPicture(image, width: size.width, height: size.height)
            var pose: MLXArray?
            if let value = request.input(forKey: NFKInputVideo) {
                guard let asset = value as? NFKVideoAsset, let url = asset.fileURL else {
                    throw NFKMLXError.unsupportedConfiguration("the pose video is an NFKVideoAsset with a file URL")
                }
                let count = framesPerSegment * (segments ?? Int.max / max(framesPerSegment, 1))
                pose = stacked(try Self.poseFrames(try NFKMLXVideoFile.read(url), count: count).map {
                    try Self.preparedPicture($0, width: size.width, height: size.height)
                })
            }
            job.reportProgress(0)
            let clip = try generate(prompt: prompt, negativePrompt: negative, reference: reference, audio: speech,
                                    pose: pose,
                                    startsFromReference: parameter(NFKMLXGenerationParameterKey.startsFromReference)?.boolValue ?? false,
                                    segments: segments, seed: NFKMLXGenerationBackend.seed(request),
                                    progress: { job.reportProgress(0.99 * $0) })
            if job.status == .cancelled { return nil }
            let output = try NFKMLXGenerationBackend.videoResult(clip, framesPerSecond: framesPerSecond, identifier: identifier)
            job.reportProgress(1)
            return output
        }
    }

    /// Generates a clip and returns its frames as `CGImage`s: `referenceImage` speaking `audio` (mono 16 kHz
    /// samples as `NSNumber` floats or one WAV `NSData`), optionally driven by `poseFrames`, as
    /// ``video(forPrompt:negativePrompt:referenceImage:audio:poseFrames:startsFromReference:segments:seed:)``
    /// describes; `segments` 0 generates as many as the speech covers. Objective-C reads each element as
    /// `(__bridge CGImageRef)`. Introduced in InferKit 0.4.0.
    @objc(framesForPrompt:negativePrompt:referenceImage:audio:poseFrames:startsFromReference:segments:seed:error:)
    public func frames(forPrompt prompt: String, negativePrompt: String?, referenceImage: CGImage, audio: Any,
                       poseFrames: [Any]?, startsFromReference: Bool, segments: Int, seed: UInt64) throws -> [Any] {
        let samples: [Float]
        if let numbers = audio as? [NSNumber] {
            samples = numbers.map(\.floatValue)
        } else {
            samples = try Self.speechSamples(audio)
        }
        let poses = try poseFrames?.map { try NFKMLXGenerationBackend.cgImage(fromInput: $0) }
        let clip = try video(forPrompt: prompt, negativePrompt: negativePrompt, referenceImage: referenceImage,
                             audio: samples, poseFrames: poses, startsFromReference: startsFromReference,
                             segments: segments > 0 ? segments : nil, seed: seed)
        return try (0 ..< clip.dim(0)).map { try NFKMLXImageBridge.cgImage(from: clip[$0], options: NFKMLXImageOptions()) }
    }

    // MARK: Backend factories

    /// Assembles the model from a downloaded release directory, held as ``NFKMLXResidency/automatic`` decides,
    /// behind the InferKit contract (``inferenceBackend()``). Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL).inferenceBackend()
    }

    /// ``backend(directoryURL:)`` holding the stages as `residency` says. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL, residency: residency).inferenceBackend()
    }

    /// ``backend(directoryURL:residency:)`` with the weights at `precision` and the transformer's block weights
    /// packed to `quantizationBits`, as ``generator(directoryURL:residency:precision:quantizationBits:)`` reads
    /// them. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:precision:quantizationBits:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                               quantizationBits: Int) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL, residency: residency, precision: precision,
                      quantizationBits: quantizationBits).inferenceBackend()
    }

    /// Downloads the release (or `repo`) and returns the model behind the InferKit contract. Introduced in
    /// InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func backend(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try generator(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                          residency: residency).inferenceBackend()
        }
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:residency:)``. The handler runs on a
    /// background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func backend(repo: String?, revision: String?, cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: residency)
        }
    }

    /// Registers the backend under ``modelName``. The registry's URL is the release directory; nil is refused.
    /// Introduced in InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXGenerationBackend.registerDirectoryModel(modelName) { directory in
            try generator(directoryURL: directory).inferenceBackend()
        }
    }
}
