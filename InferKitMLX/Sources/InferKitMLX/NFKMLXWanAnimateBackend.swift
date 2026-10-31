//
//  NFKMLXWanAnimateBackend.swift
//  InferKitMLX
//
//  Wan 2.2 Animate 2 character animation behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit

extension NFKMLXGenerationParameterKey {
    /// The frames each segment of a segmented video generator denoises (`NSNumber` integer). Wan 2.2
    /// Animate reads it; its default is ``NFKMLXWanAnimateGenerator/segmentLength``. Introduced in InferKit 0.4.0.
    @objc public static let segmentLength = "NFKMLXParameterSegmentLength"
    /// The frames each segment after the first carries over from the one before it (`NSNumber` integer).
    /// Wan 2.2 Animate reads it; its default is ``NFKMLXWanAnimateGenerator/carriedFrames``. Introduced in
    /// InferKit 0.4.0.
    @objc public static let carriedFrames = "NFKMLXParameterCarriedFrames"
}

extension NFKMLXWanAnimateGenerator {

    /// The request inputs the backend reads.
    static let backendInputKeys: Set<String> = [NFKInputImage, NFKInputVideo, NFKInputPrompt, NFKInputNegativePrompt]

    /// The request parameters the backend reads.
    static let backendParameterKeys: Set<String> = [
        NFKParameterWidth, NFKParameterHeight, NFKParameterFramesPerSecond, NFKParameterSeed, NFKParameterSteps,
        NFKParameterGuidanceScale, NFKMLXGenerationParameterKey.segmentLength, NFKMLXGenerationParameterKey.carriedFrames,
    ]

    /// The generator behind the InferKit contract.
    ///
    /// @discussion The backend reads the reference image under `NFKInputImage` (required), the driving video
    /// under `NFKInputVideo` (an `NFKVideoAsset` with a file URL, required), `NFKInputPrompt` (a description of
    /// the character and the background; empty when absent), and `NFKInputNegativePrompt`, and returns an
    /// `NFKVideoAsset` (a QuickTime movie in the temporary directory) under `NFKOutputVideo`, one frame per
    /// driving frame after resampling. Parameters: `NFKParameterWidth` and `NFKParameterHeight` (the target
    /// area, 640 × 800), `NFKParameterFramesPerSecond` (the rate the driving video is resampled to and the
    /// clip plays at, ``framesPerSecond``), `NFKParameterSeed` (0), `NFKParameterSteps` and
    /// `NFKParameterGuidanceScale` (``steps`` and ``guidance``), and
    /// `NFKMLXGenerationParameterKey.segmentLength` and `.carriedFrames` (``segmentLength`` and
    /// ``carriedFrames``). A request's values apply to that request only. Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        let identifier = variant == .standard ? Self.modelName : Self.distilledModelName
        return NFKMLXGenerationBackend(identifier: identifier, inputKeys: Self.backendInputKeys,
                                       parameterKeys: Self.backendParameterKeys,
                                       outputKeys: [NFKOutputVideo]) { [self] request, job in
            let reference = try NFKMLXGenerationBackend.requiredImage(request, NFKInputImage)
            guard let asset = request.input(forKey: NFKInputVideo) as? NFKVideoAsset, let url = asset.fileURL else {
                throw NFKMLXError.unsupportedConfiguration("the driving video is an NFKVideoAsset with a file URL")
            }
            let driving = try NFKMLXVideoFile.read(url)
            func parameter(_ key: String) -> NSNumber? { NFKMLXGenerationBackend.number(request, key) }
            let saved = (steps, guidance, width, height, framesPerSecond, segmentLength, carriedFrames)
            defer {
                (steps, guidance, width, height, framesPerSecond, segmentLength, carriedFrames) = saved
            }
            steps = parameter(NFKParameterSteps)?.intValue ?? steps
            guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? guidance
            width = parameter(NFKParameterWidth)?.intValue ?? width
            height = parameter(NFKParameterHeight)?.intValue ?? height
            framesPerSecond = parameter(NFKParameterFramesPerSecond)?.doubleValue ?? framesPerSecond
            segmentLength = parameter(NFKMLXGenerationParameterKey.segmentLength)?.intValue ?? segmentLength
            carriedFrames = parameter(NFKMLXGenerationParameterKey.carriedFrames)?.intValue ?? carriedFrames
            job.reportProgress(0)
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let clip = try video(reference: NFKWanAnimateRGB(reference, colorSpace: colorSpace),
                                 driving: driving.frames.map { NFKWanAnimateRGB($0, colorSpace: colorSpace) },
                                 drivingRate: driving.framesPerSecond,
                                 prompt: NFKMLXGenerationBackend.text(request, NFKInputPrompt) ?? "",
                                 negativePrompt: NFKMLXGenerationBackend.text(request, NFKInputNegativePrompt),
                                 seed: NFKMLXGenerationBackend.seed(request),
                                 progress: { job.reportProgress(0.99 * $0) })
            if job.status == .cancelled { return nil }
            let result = try NFKMLXGenerationBackend.videoResult(clip, framesPerSecond: framesPerSecond,
                                                                 identifier: identifier)
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

    /// ``backend(directoryURL:residency:)`` with every stage at `precision` and the transformer's block weights
    /// packed to `quantizationBits` (0 for none), as
    /// ``generator(directoryURL:residency:precision:quantizationBits:)`` loads them. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:precision:quantizationBits:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency, precision: NFKMLXWeightPrecision,
                               quantizationBits: Int) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL, residency: residency, precision: precision,
                      quantizationBits: quantizationBits).inferenceBackend()
    }

    /// Downloads a release (`variant`'s repository when `repo` is nil) and returns the model behind the
    /// InferKit contract.
    ///
    /// @discussion Each release is about 50 GB. A file already in the cache is not fetched again. The call
    /// blocks on the network, so run it off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithVariant:repo:revision:cacheDirectoryURL:residency:error:)
    public static func backend(variant: NFKMLXWanAnimateVariant, repo: String?, revision: String?,
                               cacheDirectoryURL: URL?, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try generator(variant: variant, repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                          residency: residency).inferenceBackend()
        }
    }

    /// The asynchronous form of ``backend(variant:repo:revision:cacheDirectoryURL:residency:)``. The handler
    /// runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithVariant:repo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func backend(variant: NFKMLXWanAnimateVariant, repo: String?, revision: String?,
                               cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(variant: variant, repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                        residency: residency)
        }
    }

    /// Registers both releases, ``modelName`` and ``distilledModelName``. The registry's URL is the release
    /// directory; nil is refused. Introduced in InferKit 0.4.0.
    @objc public static func register() {
        for name in [modelName, distilledModelName] {
            NFKMLXGenerationBackend.registerDirectoryModel(name) { try backend(directoryURL: $0) }
        }
    }
}
