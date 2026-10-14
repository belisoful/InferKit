//
//  NFKMLXWanVideoBackend.swift
//  InferKitMLX
//
//  Wan 2.1 T2V and Wan 2.2 TI2V-5B text-to-video behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXWanVideoGenerator {

    /// The request inputs the backend reads.
    static let backendInputKeys: Set<String> = [NFKInputPrompt, NFKInputNegativePrompt]

    /// The request parameters the backend reads.
    static let backendParameterKeys: Set<String> = [
        NFKParameterFrameCount, NFKParameterWidth, NFKParameterHeight, NFKParameterFramesPerSecond,
        NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale,
    ]

    /// The generator behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt` (required) and `NFKInputNegativePrompt`, and returns an
    /// `NFKVideoAsset` (a QuickTime movie in the temporary directory) under `NFKOutputVideo`. Parameters:
    /// `NFKParameterFrameCount` (81), `NFKParameterWidth` (832), `NFKParameterHeight` (480),
    /// `NFKParameterFramesPerSecond` (``framesPerSecond``: 24 for Wan 2.2 TI2V-5B, 16 for Wan 2.1),
    /// `NFKParameterSeed` (0), `NFKParameterSteps` and `NFKParameterGuidanceScale` (the generator's ``steps``
    /// and ``guidance``). The frame count and sides round down as
    /// ``video(forPrompt:negativePrompt:frames:width:height:seed:)`` rounds them. A request's steps and guidance
    /// apply to that request only. Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: Self.modelName, inputKeys: Self.backendInputKeys,
                                parameterKeys: Self.backendParameterKeys) { [self] request, job in
            let prompt = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            let negative = NFKMLXGenerationBackend.text(request, NFKInputNegativePrompt)
            func parameter(_ key: String) -> NSNumber? { NFKMLXGenerationBackend.number(request, key) }
            let (savedSteps, savedGuidance) = (steps, guidance)
            defer { (steps, guidance) = (savedSteps, savedGuidance) }
            steps = parameter(NFKParameterSteps)?.intValue ?? steps
            guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? guidance
            let framesPerSecond = parameter(NFKParameterFramesPerSecond)?.doubleValue ?? framesPerSecond
            job.reportProgress(0)
            let clip = try video(forPrompt: prompt, negativePrompt: negative,
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

    /// Downloads a release (``releaseRepo`` when `repo` is nil, or `Wan-AI/Wan2.2-TI2V-5B-Diffusers`) and
    /// returns the model behind the InferKit contract.
    ///
    /// @discussion Wan 2.1 T2V 1.3B is about 29 GB. A file already in the cache is not fetched again. The call
    /// blocks on the network, so run it off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func backend(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try generator(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                      residency: residency).inferenceBackend()
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
