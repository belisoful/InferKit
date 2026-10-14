//
//  NFKMLXFluxBackend.swift
//  InferKitMLX
//
//  FLUX.1 text-to-image behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXFlux {

    /// The request inputs the backend reads.
    static let backendInputKeys: Set<String> = [NFKInputPrompt]

    /// The request parameters the backend reads.
    static let backendParameterKeys: Set<String> = [
        NFKParameterWidth, NFKParameterHeight, NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale,
    ]

    /// The model behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt` (required) and returns an RGB `CGImage` under
    /// `NFKOutputImage`. FLUX.1 takes no negative prompt. Parameters: `NFKParameterWidth` and
    /// `NFKParameterHeight` (1024, multiples of 16), `NFKParameterSeed` (0), `NFKParameterSteps` (the model's
    /// ``steps``), and `NFKParameterGuidanceScale` (the model's ``guidance``, which only a guidance-distilled
    /// release such as [dev] reads). A request's steps and guidance apply to that request only. Introduced in
    /// InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: Self.modelName, inputKeys: Self.backendInputKeys,
                                parameterKeys: Self.backendParameterKeys) { [self] request, job in
            let prompt = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            func parameter(_ key: String) -> NSNumber? { NFKMLXGenerationBackend.number(request, key) }
            let (savedSteps, savedGuidance) = (steps, guidance)
            defer { (steps, guidance) = (savedSteps, savedGuidance) }
            steps = parameter(NFKParameterSteps)?.intValue ?? steps
            guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? guidance
            job.reportProgress(0)
            let image = try cgImage(forPrompt: prompt,
                                    width: parameter(NFKParameterWidth)?.intValue ?? 1024,
                                    height: parameter(NFKParameterHeight)?.intValue ?? 1024,
                                    seed: NFKMLXGenerationBackend.seed(request))
            if job.status == .cancelled { return nil }
            job.reportProgress(1)
            return NFKMLXGenerationBackend.imageResult(image)
        }
    }

    /// Assembles the model from a downloaded diffusers FLUX release directory with the four-step schnell
    /// schedule, held as ``NFKMLXResidency/automatic`` decides, and returns it behind the InferKit contract
    /// (``inferenceBackend()``). Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try flux(directoryURL: directoryURL).inferenceBackend()
    }

    /// ``backend(directoryURL:)`` holding the stages as `residency` says. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try flux(directoryURL: directoryURL, residency: residency).inferenceBackend()
    }

    /// ``backend(directoryURL:residency:)`` over `schedule`: `.fluxSchnell` for the four-step distillation
    /// or `.flux` for the [dev] dynamic-shift schedule. Introduced in InferKit 0.4.0.
    public static func backend(directoryURL: URL, schedule: NFKMLXFlowMatchConfiguration,
                               residency: NFKMLXResidency = .automatic) throws -> any NFKInferenceBackend {
        try flux(directoryURL: directoryURL, schedule: schedule, residency: residency).inferenceBackend()
    }

    /// Downloads a diffusers FLUX.1 release and returns the model behind the InferKit contract, with the
    /// four-step schnell schedule.
    ///
    /// @discussion The download is about 34 GB for `black-forest-labs/FLUX.1-schnell`. Both FLUX.1 releases
    /// are gated, so the caller accepts the license on Hugging Face and sets `NFKHFHub.defaultAccessToken`
    /// before the first download. A file already in the cache is not fetched again. The call blocks on the
    /// network, so run it off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try flux(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL).inferenceBackend()
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``. The handler runs on a
    /// background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// ``backend(repo:revision:cacheDirectoryURL:)`` holding the stages as `residency` says. Introduced in
    /// InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try flux(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                 residency: residency).inferenceBackend()
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:residency:)``. The handler runs
    /// on a background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: residency)
        }
    }

    /// Registers the backend under ``modelName``, reading the release with the schnell schedule. The
    /// registry's URL is the release directory; nil is refused. Introduced in InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXGenerationBackend.registerDirectoryModel(modelName) { try backend(directoryURL: $0) }
    }
}
