//
//  NFKMLXLTX2Backend.swift
//  InferKitMLX
//
//  LTX-2 behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

/// The LTX-2 request parameters that have no core `NFKParameter*` key.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXLTX2ParameterKey)
public final class NFKMLXLTX2ParameterKey: NSObject {
    /// The soundtrack's classifier-free guidance scale, as an `NSNumber`. See
    /// ``NFKMLXLTX2Generator/audioGuidance``, which a request without the key reads. Introduced in InferKit
    /// 0.4.0.
    @objc public static let audioGuidanceScale = "NFKMLXLTX2ParameterAudioGuidanceScale"
    /// The spatio-temporal guidance scale of both modalities, as an `NSNumber`; 0 skips the perturbed pass.
    /// See ``NFKMLXLTX2Generator/spatioTemporalGuidance``. Introduced in InferKit 0.4.0.
    @objc public static let spatioTemporalGuidanceScale = "NFKMLXLTX2ParameterSpatioTemporalGuidanceScale"
    /// The modality guidance scale of both modalities, as an `NSNumber`; 1 skips the isolated pass. See
    /// ``NFKMLXLTX2Generator/modalityGuidance``. Introduced in InferKit 0.4.0.
    @objc public static let modalityGuidanceScale = "NFKMLXLTX2ParameterModalityGuidanceScale"
    /// The guidance rescale of both modalities, as an `NSNumber`; 0 turns it off. See
    /// ``NFKMLXLTX2Generator/guidanceRescale``. Introduced in InferKit 0.4.0.
    @objc public static let guidanceRescale = "NFKMLXLTX2ParameterGuidanceRescale"
}

extension NFKMLXLTX2Generator {

    /// The request inputs the backend reads.
    static let backendInputKeys: Set<String> = [NFKInputPrompt, NFKInputNegativePrompt]

    /// The request parameters the backend reads.
    static let backendParameterKeys: Set<String> = Set([
        NFKParameterFrameCount, NFKParameterWidth, NFKParameterHeight, NFKParameterFramesPerSecond,
        NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale, NFKMLXLTX2ParameterKey.audioGuidanceScale,
        NFKMLXLTX2ParameterKey.spatioTemporalGuidanceScale, NFKMLXLTX2ParameterKey.modalityGuidanceScale,
        NFKMLXLTX2ParameterKey.guidanceRescale,
    ]).union(decodePathParameterKeys)

    /// The generator behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt` (required) and `NFKInputNegativePrompt`, and returns an
    /// `NFKVideoAsset` (a QuickTime movie in the temporary directory whose audio track is the generated
    /// soundtrack) under `NFKOutputVideo` and the soundtrack alone as an `NFKAudioAsset` (a stereo WAV at
    /// ``audioSampleRate``) under `NFKOutputAudio`. Parameters: `NFKParameterFrameCount` (121),
    /// `NFKParameterWidth` (768), `NFKParameterHeight` (512), `NFKParameterFramesPerSecond` (the generator's
    /// ``frameRate``), `NFKParameterSeed` (0), `NFKParameterSteps` and `NFKParameterGuidanceScale` (the
    /// generator's ``steps`` and ``guidance``), and the ``NFKMLXLTX2ParameterKey`` scales. A request's values
    /// apply to that request only. The job reports progress after each denoising step. Introduced in
    /// InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: modelName, inputKeys: Self.backendInputKeys,
                                parameterKeys: Self.backendParameterKeys,
                                outputKeys: [NFKOutputVideo, NFKOutputAudio]) { [self] request, job in
            let prompt = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            let negative = NFKMLXGenerationBackend.text(request, NFKInputNegativePrompt)
            func parameter(_ key: String) -> NSNumber? { NFKMLXGenerationBackend.number(request, key) }
            let saved = (steps, guidance, audioGuidance, spatioTemporalGuidance, modalityGuidance, guidanceRescale,
                         frameRate)
            defer {
                (steps, guidance, audioGuidance, spatioTemporalGuidance, modalityGuidance, guidanceRescale,
                 frameRate) = saved
            }
            steps = parameter(NFKParameterSteps)?.intValue ?? steps
            guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? guidance
            audioGuidance = parameter(NFKMLXLTX2ParameterKey.audioGuidanceScale)?.floatValue ?? audioGuidance
            spatioTemporalGuidance = parameter(NFKMLXLTX2ParameterKey.spatioTemporalGuidanceScale)?.floatValue
                ?? spatioTemporalGuidance
            modalityGuidance = parameter(NFKMLXLTX2ParameterKey.modalityGuidanceScale)?.floatValue ?? modalityGuidance
            guidanceRescale = parameter(NFKMLXLTX2ParameterKey.guidanceRescale)?.floatValue ?? guidanceRescale
            frameRate = parameter(NFKParameterFramesPerSecond)?.floatValue ?? frameRate
            let restoreDecodePaths = applyingDecodePathParameters(parameter)
            defer { restoreDecodePaths() }
            job.reportProgress(0)
            let total = reportedSteps
            let (video, audio) = try clip(forPrompt: prompt, negativePrompt: negative,
                                          frames: parameter(NFKParameterFrameCount)?.intValue ?? 121,
                                          width: parameter(NFKParameterWidth)?.intValue ?? 768,
                                          height: parameter(NFKParameterHeight)?.intValue ?? 512,
                                          seed: NFKMLXGenerationBackend.seed(request)) { step in
                job.reportProgress(Double(step + 1) / Double(total + 1))
            }
            if job.status == .cancelled { return nil }
            let result = try Self.result(video: video, audio: audio, framesPerSecond: Double(outputFrameRate),
                                         sampleRate: audioSampleRate, identifier: modelName)
            job.reportProgress(1)
            return result
        }
    }

    /// The movie with its soundtrack under `NFKOutputVideo` and the soundtrack alone under `NFKOutputAudio`.
    static func result(video: MLXArray, audio: MLXArray, framesPerSecond: Double, sampleRate: Int,
                       identifier: String) throws -> NFKInferenceResult {
        let frames = try (0 ..< video.dim(0)).map { try NFKMLXGenerationBackend.cgImage(video[$0]) }
        guard let first = frames.first else { throw NFKMLXError.noOutput }
        let samples = audio.asType(.float32).reshaped([-1]).asArray(Float.self)
        let channels = audio.dim(1)
        let soundtrack = NFKMLXVideoFile.Soundtrack(samples: samples, sampleRate: sampleRate, channels: channels)
        let stem = FileManager.default.temporaryDirectory.appendingPathComponent("\(identifier)-\(UUID().uuidString)")
        let movie = stem.appendingPathExtension("mov"), wave = stem.appendingPathExtension("wav")
        try NFKMLXVideoFile.write(frames, framesPerSecond: framesPerSecond, soundtrack: soundtrack, to: movie)
        try NFKMLXWaveFile.write(samples: samples, sampleRate: sampleRate, channels: channels, to: wave)
        let seconds = Double(samples.count / channels) / Double(sampleRate)
        return NFKInferenceResult(outputs: [
            NFKOutputVideo: NFKVideoAsset(fileURL: movie, durationSeconds: Double(frames.count) / framesPerSecond,
                                          framesPerSecond: framesPerSecond,
                                          dimensions: CGSize(width: first.width, height: first.height)),
            NFKOutputAudio: NFKAudioAsset(fileURL: wave, durationSeconds: seconds, sampleRate: Double(sampleRate),
                                          channelCount: channels),
        ])
    }

    /// Assembles the model from a downloaded release directory, held as ``NFKMLXResidency/automatic``
    /// decides, and returns it behind the InferKit contract (``inferenceBackend()``). The directory's
    /// transformer says which release it is. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL).inferenceBackend()
    }

    /// ``backend(directoryURL:)`` holding the stages as `residency` says. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL, residency: residency).inferenceBackend()
    }

    /// Downloads `variant`'s release (its ``NFKMLXLTX2Variant/releaseRepo`` when `repo` is nil) and returns
    /// the model behind the InferKit contract.
    ///
    /// @discussion LTX-2.3 is about 95 GB to download and LTX-2.5, whose repository is gated, about 70 GB. A file already in
    /// the cache is not fetched again. The call blocks on the network, so run it off the main and render
    /// threads. Introduced in InferKit 0.4.0.
    @objc(backendWithVariant:repo:revision:cacheDirectoryURL:residency:error:)
    public static func backend(variant: NFKMLXLTX2Variant, repo: String?, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try generator(variant: variant, repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                          residency: residency).inferenceBackend()
        }
    }

    /// The asynchronous form of ``backend(variant:repo:revision:cacheDirectoryURL:residency:)``. The handler
    /// runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithVariant:repo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func backend(variant: NFKMLXLTX2Variant, repo: String?, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(variant: variant, repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                        residency: residency)
        }
    }

    /// Registers both releases under their ``NFKMLXLTX2Variant/modelName``s. The registry's URL is the
    /// release directory; nil is refused, and so is a directory holding the other release. The load options
    /// the factory reads are the residency, the transformer's precision, and a runtime weight quantization
    /// of the transformer. Introduced in InferKit 0.4.0.
    @objc public static func register() {
        for variant in NFKMLXLTX2Variant.allCases {
            NFKMLXModelRegistry.register(name: variant.modelName,
                                         consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                                    NFKModelLoadOptionQuantizationBits,
                                                    NFKModelLoadOptionQuantizationGroupSize]) { location, options in
                guard let location else {
                    throw NFKMLXError.unsupportedConfiguration(
                        "\(variant.modelName) needs the release directory; there is no random-weights form")
                }
                _ = try options.cast(of: variant.modelName, offering: [:], reading: [NFKModelPrecisionFloat32, NFKModelPrecisionBFloat16])
                let generator = try generator(directoryURL: location, residency: options.residency, options: options)
                guard generator.variant == variant else {
                    throw NFKMLXError.unsupportedConfiguration(
                        "\(location.lastPathComponent) holds \(generator.variant.displayName), not \(variant.displayName)")
                }
                return generator.inferenceBackend()
            }
        }
    }
}
