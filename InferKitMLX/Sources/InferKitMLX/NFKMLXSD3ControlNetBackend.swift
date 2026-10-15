//
//  NFKMLXSD3ControlNetBackend.swift
//  InferKitMLX
//
//  SD3 ControlNet behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXSD3ControlNetGenerator {

    /// The request inputs the backend reads.
    static let backendInputKeys: Set<String> = [NFKInputPrompt, NFKInputNegativePrompt, NFKInputControl]

    /// The request parameters the backend reads.
    static let backendParameterKeys: Set<String> = [
        NFKParameterWidth, NFKParameterHeight, NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale,
        NFKMLXControlNetParameterKey.conditioningScale,
    ]

    /// The generator behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt` and the control image under `NFKInputControl` (a
    /// `CGImage` or texture, both required) and `NFKInputNegativePrompt`, and returns a `CGImage` under
    /// `NFKOutputImage`. Parameters: `NFKParameterWidth` and `NFKParameterHeight` (the control image's
    /// size rounded down to a multiple of 16; the control image is resized to the output size),
    /// `NFKParameterSeed` (0), and `NFKParameterSteps`, `NFKParameterGuidanceScale`, and
    /// ``NFKMLXControlNetParameterKey/conditioningScale`` (the generator's ``steps``, ``guidance``, and
    /// ``conditioningScale``). A request's settings apply to that request only. Introduced in InferKit
    /// 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: Self.modelName, inputKeys: Self.backendInputKeys,
                                parameterKeys: Self.backendParameterKeys,
                                outputKeys: [NFKOutputImage]) { [self] request, job in
            let prompt = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            let negative = NFKMLXGenerationBackend.text(request, NFKInputNegativePrompt)
            guard let control = request.input(forKey: NFKInputControl) else { throw NFKMLXError.unsupportedInput }
            func parameter(_ key: String) -> NSNumber? { NFKMLXGenerationBackend.number(request, key) }
            let (width, height) = try NFKMLXControlImage.size(request, control: control, multiple: 2 * pixelsPerLatent)
            let pixels = try NFKMLXControlImage.pixels(control, width: width, height: height)
            let (savedSteps, savedGuidance, savedScale) = (steps, guidance, conditioningScale)
            defer { (steps, guidance, conditioningScale) = (savedSteps, savedGuidance, savedScale) }
            steps = parameter(NFKParameterSteps)?.intValue ?? steps
            guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? guidance
            conditioningScale = parameter(NFKMLXControlNetParameterKey.conditioningScale)?.floatValue ?? conditioningScale
            job.reportProgress(0)
            let generated = try image(forPrompt: prompt, negativePrompt: negative, controlImage: pixels,
                                      width: width, height: height, seed: NFKMLXGenerationBackend.seed(request))
            if job.status == .cancelled { return nil }
            let result = NFKMLXGenerationBackend.imageResult(try NFKMLXGenerationBackend.cgImage(generated))
            job.reportProgress(1)
            return result
        }
    }

    /// Assembles the model from an SD3 release directory and a ControlNet release directory, held as
    /// ``NFKMLXResidency/automatic`` decides, and returns it behind the InferKit contract
    /// (``inferenceBackend()``). Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:controlNetDirectoryURL:error:)
    public static func backend(directoryURL: URL, controlNetDirectoryURL: URL) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL, controlNetDirectoryURL: controlNetDirectoryURL).inferenceBackend()
    }

    /// ``backend(directoryURL:controlNetDirectoryURL:)`` holding the stages as `residency` says.
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:controlNetDirectoryURL:residency:error:)
    public static func backend(directoryURL: URL, controlNetDirectoryURL: URL,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL, controlNetDirectoryURL: controlNetDirectoryURL,
                      residency: residency).inferenceBackend()
    }

    /// Downloads the base release and the ControlNet (``releaseRepo`` and ``controlNetReleaseRepo`` where
    /// nil) and returns the model behind the InferKit contract.
    ///
    /// @discussion `revision` names the base release's revision. The call blocks on the network, so run it
    /// off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:controlNetRepo:revision:cacheDirectoryURL:residency:error:)
    public static func backend(repo: String?, controlNetRepo: String?, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try generator(repo: repo, controlNetRepo: controlNetRepo, revision: revision,
                      cacheDirectoryURL: cacheDirectoryURL, residency: residency).inferenceBackend()
    }

    /// The asynchronous form of ``backend(repo:controlNetRepo:revision:cacheDirectoryURL:residency:)``. The
    /// handler runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:controlNetRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func backend(repo: String?, controlNetRepo: String?, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, controlNetRepo: controlNetRepo, revision: revision,
                        cacheDirectoryURL: cacheDirectoryURL, residency: residency)
        }
    }

    /// Registers the backend under ``modelName``. The registry's URL is a directory holding the base
    /// release under `base/` and the ControlNet release under `controlnet/`; nil is refused. Introduced in
    /// InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXGenerationBackend.registerDirectoryModel(modelName) {
            try backend(directoryURL: $0.appendingPathComponent("base"),
                        controlNetDirectoryURL: $0.appendingPathComponent("controlnet"))
        }
    }
}
