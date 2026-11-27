//
//  NFKMLXWanDancerBackend.swift
//  InferKitMLX
//
//  Wan-Dancer-14B music-to-dance behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

extension NFKMLXVideoInputKey {
    /// The prompt of a two-stage generator's second stage (an `NSString`). Wan-Dancer's local stage reads it
    /// (the release's `*_local.txt` style prompts); absent, it reads `NFKInputPrompt`. Introduced in InferKit 0.4.0.
    @objc public static let localPrompt = "NFKMLXInputLocalPrompt"
}

extension NFKMLXGenerationParameterKey {
    /// The denoising steps of a two-stage generator's keyframe stage (`NSNumber` integer). Wan-Dancer's global
    /// stage reads it; its default is ``NFKMLXWanDancerGenerator/globalSteps``. Introduced in InferKit 0.4.0.
    @objc public static let keyframeSteps = "NFKMLXParameterKeyframeSteps"
}

extension NFKMLXWanDancerGenerator {

    /// The request inputs the backend reads.
    static let backendInputKeys: Set<String> = [NFKInputPrompt, NFKInputNegativePrompt, NFKInputImage, NFKInputAudio,
                                                NFKMLXVideoInputKey.localPrompt]

    /// The request parameters the backend reads.
    static let backendParameterKeys: Set<String> = [
        NFKParameterWidth, NFKParameterHeight, NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale,
        NFKMLXGenerationParameterKey.flowShift, NFKMLXGenerationParameterKey.keyframeSteps,
    ]

    /// Mono samples and their rate from an audio input (an `NFKAudioAsset`, WAV `Data`, or an `AVAudioPCMBuffer`),
    /// the channels averaged as `librosa.load` averages them.
    static func music(_ value: Any?) throws -> (samples: [Float], sampleRate: Int) {
        guard let (samples, rate, channels) = NFKMLXLiveAudio.samples(from: value) else {
            throw NFKMLXError.unsupportedConfiguration("a Wan-Dancer request needs music under NFKInputAudio")
        }
        guard channels > 1 else { return (samples, rate) }
        let mono = (0 ..< samples.count / channels).map { frame in
            (0 ..< channels).reduce(Float(0)) { $0 + samples[frame * channels + $1] } / Float(channels)
        }
        return (mono, rate)
    }

    static func picture(_ image: Any) throws -> NFKMLXWanDancerPicture {
        let (rgb, width, height) = try NFKMLXImageBridge.rgbBytes(from: image, colorSpace: CGColorSpaceCreateDeviceRGB())
        return NFKMLXWanDancerPicture(rgb: rgb, width: width, height: height)
    }

    /// The generator behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt` (required, the global stage's style prompt),
    /// ``NFKMLXVideoInputKey/localPrompt`` (the local stage's, defaulting to the global one),
    /// `NFKInputNegativePrompt`, the dancer's image under `NFKInputImage` (required), and the music under
    /// `NFKInputAudio` (required, at any rate). It returns an `NFKVideoAsset` (a QuickTime movie at 30 frames per
    /// second in the temporary directory, the music not muxed) under `NFKOutputVideo`. Parameters:
    /// `NFKParameterWidth` × `NFKParameterHeight` (``width`` × ``height``), `NFKParameterSeed` (0), `NFKParameterSteps`
    /// (``localSteps``), ``NFKMLXGenerationParameterKey/keyframeSteps`` (``globalSteps``), `NFKParameterGuidanceScale`
    /// (``guidance``), and ``NFKMLXGenerationParameterKey/flowShift`` (``shift``). A request's values apply to that
    /// request only. Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        let identifier = Self.modelName
        return NFKMLXGenerationBackend(identifier: identifier, inputKeys: Self.backendInputKeys,
                                       parameterKeys: Self.backendParameterKeys,
                                       outputKeys: [NFKOutputVideo]) { [self] request, job in
            let prompt = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            let localPrompt = NFKMLXGenerationBackend.text(request, NFKMLXVideoInputKey.localPrompt)
            let negative = NFKMLXGenerationBackend.text(request, NFKInputNegativePrompt)
            guard let image = request.input(forKey: NFKInputImage) else {
                throw NFKMLXError.unsupportedConfiguration("a Wan-Dancer request needs the dancer's image under NFKInputImage")
            }
            let reference = try Self.picture(image)
            let (samples, rate) = try Self.music(request.input(forKey: NFKInputAudio))
            func parameter(_ key: String) -> NSNumber? { NFKMLXGenerationBackend.number(request, key) }
            let saved = (localSteps, globalSteps, guidance, shift, width, height)
            defer { (localSteps, globalSteps, guidance, shift, width, height) = saved }
            localSteps = parameter(NFKParameterSteps)?.intValue ?? localSteps
            globalSteps = parameter(NFKMLXGenerationParameterKey.keyframeSteps)?.intValue ?? globalSteps
            guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? guidance
            shift = parameter(NFKMLXGenerationParameterKey.flowShift)?.floatValue ?? shift
            width = parameter(NFKParameterWidth)?.intValue ?? width
            height = parameter(NFKParameterHeight)?.intValue ?? height
            job.reportProgress(0)
            let clip = try video(prompt: prompt, localPrompt: localPrompt, negativePrompt: negative, reference: reference,
                                 music: samples, sampleRate: rate, seed: NFKMLXGenerationBackend.seed(request))
            if job.status == .cancelled { return nil }
            let output = try NFKMLXGenerationBackend.videoResult(clip, framesPerSecond: Self.framesPerSecond,
                                                                 identifier: identifier)
            job.reportProgress(1)
            return output
        }
    }

    /// Generates a dance video and returns its frames as `CGImage`s: the dancer in `referenceImage` dancing to
    /// `music` (mono samples as `NSNumber` floats at `sampleRate`, or one WAV `NSData`, whose own rate is read), as
    /// ``video(prompt:localPrompt:negativePrompt:reference:music:sampleRate:seed:)`` describes. Objective-C reads
    /// each element as `(__bridge CGImageRef)`. Introduced in InferKit 0.4.0.
    @objc(framesForPrompt:localPrompt:negativePrompt:referenceImage:music:sampleRate:seed:error:)
    public func frames(forPrompt prompt: String, localPrompt: String?, negativePrompt: String?, referenceImage: CGImage,
                       music: Any, sampleRate: Int, seed: UInt64) throws -> [Any] {
        let samples: [Float], rate: Int
        if let numbers = music as? [NSNumber] {
            (samples, rate) = (numbers.map(\.floatValue), sampleRate)
        } else {
            (samples, rate) = try Self.music(music)
        }
        let clip = try video(prompt: prompt, localPrompt: localPrompt, negativePrompt: negativePrompt,
                             reference: try Self.picture(referenceImage), music: samples, sampleRate: rate, seed: seed)
        return try (0 ..< clip.dim(0)).map { try NFKMLXImageBridge.cgImage(from: clip[$0], options: NFKMLXImageOptions()) }
    }

    // MARK: Backend factories

    /// Assembles the model from a downloaded release directory behind the InferKit contract
    /// (``inferenceBackend()``). Introduced in InferKit 0.4.0.
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
    /// packed to `quantizationBits`, as ``generator(directoryURL:residency:precision:quantizationBits:)`` reads them.
    /// Introduced in InferKit 0.4.0.
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
