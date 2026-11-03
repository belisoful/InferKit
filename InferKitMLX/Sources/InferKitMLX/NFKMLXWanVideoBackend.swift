//
//  NFKMLXWanVideoBackend.swift
//  InferKitMLX
//
//  Wan 2.1 T2V, Wan 2.2 TI2V-5B, and Wan 2.2 T2V-A14B text-to-video, and Wan 2.2 I2V-A14B image-to-video,
//  behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit

extension NFKMLXGenerationParameterKey {
    /// The guidance scale of a two-expert video model's low-noise expert (`NSNumber` float), the
    /// reference's `guidance_scale_2`. Wan 2.2 A14B reads it; absent, it takes `NFKParameterGuidanceScale`.
    /// Introduced in InferKit 0.4.0.
    @objc public static let lowNoiseGuidanceScale = "NFKMLXParameterLowNoiseGuidanceScale"
}

/// Request inputs a video generator reads beside the core's.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXVideoInputKey)
public final class NFKMLXVideoInputKey: NSObject {
    /// The clip's last frame (a `CGImage`, `CVPixelBuffer`, or texture) beside its first under `NFKInputImage`.
    /// Wan 2.2 I2V-A14B reads it. Introduced in InferKit 0.4.0.
    @objc public static let lastImage = "NFKMLXInputLastImage"
}

extension NFKMLXWanVideoGenerator {

    /// The request inputs a text-to-video release's backend reads.
    static let backendInputKeys: Set<String> = [NFKInputPrompt, NFKInputNegativePrompt]

    /// The request inputs an image-to-video release's backend reads.
    static let imageToVideoInputKeys = backendInputKeys.union([NFKInputImage, NFKMLXVideoInputKey.lastImage])

    /// The request parameters a single-transformer release's backend reads.
    static let backendParameterKeys: Set<String> = [
        NFKParameterFrameCount, NFKParameterWidth, NFKParameterHeight, NFKParameterFramesPerSecond,
        NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale,
    ]

    /// The request parameters a two-expert release's backend reads.
    static let expertParameterKeys = backendParameterKeys.union([NFKMLXGenerationParameterKey.lowNoiseGuidanceScale])

    /// The generator behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt` (required) and `NFKInputNegativePrompt`, and for an
    /// image-to-video release the first frame under `NFKInputImage` (required) and the last under
    /// ``NFKMLXVideoInputKey/lastImage``. It returns an `NFKVideoAsset` (a QuickTime movie in the temporary
    /// directory) under `NFKOutputVideo`. A two-expert release reads
    /// ``NFKMLXGenerationParameterKey/lowNoiseGuidanceScale`` (``lowNoiseGuidance``). Parameters:
    /// `NFKParameterFrameCount` (81), `NFKParameterWidth` (832), `NFKParameterHeight` (480),
    /// `NFKParameterFramesPerSecond` (``framesPerSecond``: 24 for Wan 2.2 TI2V-5B, 16 for Wan 2.1),
    /// `NFKParameterSeed` (0), `NFKParameterSteps` and `NFKParameterGuidanceScale` (the generator's ``steps``
    /// and ``guidance``). The frame count and sides round down as
    /// ``video(forPrompt:negativePrompt:frames:width:height:seed:)`` rounds them. A request's steps and guidance
    /// apply to that request only. Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: Self.modelName,
                                inputKeys: isImageToVideo ? Self.imageToVideoInputKeys : Self.backendInputKeys,
                                parameterKeys: hasExperts ? Self.expertParameterKeys : Self.backendParameterKeys,
                                outputKeys: [NFKOutputVideo]) { [self] request, job in
            let prompt = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            let negative = NFKMLXGenerationBackend.text(request, NFKInputNegativePrompt)
            func parameter(_ key: String) -> NSNumber? { NFKMLXGenerationBackend.number(request, key) }
            let image = isImageToVideo ? try NFKMLXGenerationBackend.requiredImage(request, NFKInputImage) : nil
            let lastImage = isImageToVideo ? try NFKMLXGenerationBackend.image(request, NFKMLXVideoInputKey.lastImage) : nil
            let (savedSteps, savedGuidance, savedLowNoise) = (steps, guidance, lowNoiseGuidance)
            defer { (steps, guidance, lowNoiseGuidance) = (savedSteps, savedGuidance, savedLowNoise) }
            steps = parameter(NFKParameterSteps)?.intValue ?? steps
            guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? guidance
            lowNoiseGuidance = parameter(NFKMLXGenerationParameterKey.lowNoiseGuidanceScale)?.floatValue ?? lowNoiseGuidance
            let framesPerSecond = parameter(NFKParameterFramesPerSecond)?.doubleValue ?? framesPerSecond
            job.reportProgress(0)
            let clip = try video(forPrompt: prompt, image: image, lastImage: lastImage, negativePrompt: negative,
                                 frames: parameter(NFKParameterFrameCount)?.intValue ?? 81,
                                 width: parameter(NFKParameterWidth)?.intValue ?? 832,
                                 height: parameter(NFKParameterHeight)?.intValue ?? 480,
                                 seed: NFKMLXGenerationBackend.seed(request))
            if job.status == .cancelled { return nil }
            let result = try NFKMLXGenerationBackend.videoResult(clip, framesPerSecond: framesPerSecond,
                                                                 identifier: Self.modelName)
            job.reportProgress(1)
            return result
        }
    }

    /// Assembles the model from a downloaded release directory, held as ``NFKMLXResidency/automatic``
    /// decides, and returns it behind the InferKit contract (``inferenceBackend()``). Introduced in InferKit
    /// 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL).inferenceBackend()
    }

    /// ``backend(directoryURL:)`` holding the stages as `residency` says. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL, residency: residency).inferenceBackend()
    }

    /// ``backend(directoryURL:residency:)`` with the transformers at `precision` and their block weights
    /// packed to `quantizationBits`, as ``generator(directoryURL:residency:precision:quantizationBits:)``
    /// reads them. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:precision:quantizationBits:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                               quantizationBits: Int) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL, residency: residency, precision: precision,
                      quantizationBits: quantizationBits).inferenceBackend()
    }

    /// Downloads a release (``releaseRepo`` when `repo` is nil, `Wan-AI/Wan2.2-TI2V-5B-Diffusers`,
    /// ``textToVideoExpertsRepo``, or ``imageToVideoExpertsRepo``) and returns the model behind the InferKit
    /// contract.
    ///
    /// @discussion Wan 2.1 T2V 1.3B is about 29 GB. A file already in the cache is not fetched again. The call
    /// blocks on the network, so run it off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func backend(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try generator(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                          residency: residency).inferenceBackend()
        }
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:residency:)``. The handler runs
    /// on a background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func backend(repo: String?, revision: String?, cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: residency)
        }
    }

    /// Registers the backend under ``modelName``. The registry's URL is the release directory; nil is
    /// refused. Introduced in InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXGenerationBackend.registerDirectoryModel(modelName) { try backend(directoryURL: $0) }
    }
}
