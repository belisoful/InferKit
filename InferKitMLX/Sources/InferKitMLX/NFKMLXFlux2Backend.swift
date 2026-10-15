//
//  NFKMLXFlux2Backend.swift
//  InferKitMLX
//
//  FLUX.2 [klein] text-to-image, reference editing, and inpainting behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// The FLUX.2 request parameters that have no core `NFKParameter*` key.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXFlux2ParameterKey)
public final class NFKMLXFlux2ParameterKey: NSObject {
    /// Whether a request with references uses FLUX.2 [klein] 9B KV's reference cache, as an `NSNumber`
    /// boolean. See ``NFKMLXFlux2/cachesReferences``, which a request without the key reads. Introduced in
    /// InferKit 0.4.0.
    @objc public static let cachesReferences = "NFKMLXFlux2ParameterCachesReferences"
}

extension NFKMLXFlux2 {

    /// The request inputs the backend reads.
    static let backendInputKeys: Set<String> = [
        NFKInputPrompt, NFKInputNegativePrompt, NFKInputImages, NFKInputImage, NFKInputMask,
    ]

    /// The request parameters the backend reads.
    static let backendParameterKeys: Set<String> = [
        NFKParameterWidth, NFKParameterHeight, NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale,
        NFKParameterStrength, NFKMLXFlux2ParameterKey.cachesReferences,
    ]

    /// The model behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt` (required) and `NFKInputNegativePrompt`, and returns an
    /// RGB `CGImage` under `NFKOutputImage`. The release directory or repository selects the variant: klein
    /// 4B, klein 9B, their base releases, or klein 9B KV. Inputs:
    /// - `NFKInputImages` → the images the generation is conditioned on (an array of `CGImage` or
    ///   `MTLTexture`, or one), each resampled to a multiple of 16 on each side.
    /// - `NFKInputImage` with `NFKInputMask` → inpainting: the white part of the mask is repainted and the
    ///   output is the image's size rounded down to a multiple of 16. One without the other is refused.
    ///
    /// Parameters: `NFKParameterWidth` and `NFKParameterHeight` (1024, multiples of 16; inpainting reads the
    /// image's size), `NFKParameterSeed` (0), `NFKParameterStrength` (inpainting only, 0.8),
    /// `NFKParameterSteps` and `NFKParameterGuidanceScale` (the model's ``steps`` and ``guidance``; a distilled
    /// release ignores guidance and the negative prompt), and ``NFKMLXFlux2ParameterKey/cachesReferences``
    /// (the model's ``cachesReferences``). A request's steps, guidance, and reference caching apply to that
    /// request only. Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: Self.modelName, inputKeys: Self.backendInputKeys,
                                parameterKeys: Self.backendParameterKeys,
                                outputKeys: [NFKOutputImage]) { [self] request, job in
            let prompt = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            let negative = NFKMLXGenerationBackend.text(request, NFKInputNegativePrompt)
            func parameter(_ key: String) -> NSNumber? { NFKMLXGenerationBackend.number(request, key) }
            let source = request.input(forKey: NFKInputImage)
            let mask = request.input(forKey: NFKInputMask)
            guard (source == nil) == (mask == nil) else {
                throw NFKMLXError.unsupportedConfiguration(
                    "inpainting takes NFKInputImage and NFKInputMask together; references go under NFKInputImages")
            }
            let references = try NFKMLXGenerationBackend.images(request, NFKInputImages).map {
                try Self.backendPixels($0)
            }
            let (savedSteps, savedGuidance, savedCaching) = (steps, guidance, cachesReferences)
            defer { (steps, guidance, cachesReferences) = (savedSteps, savedGuidance, savedCaching) }
            steps = parameter(NFKParameterSteps)?.intValue ?? steps
            guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? guidance
            cachesReferences = parameter(NFKMLXFlux2ParameterKey.cachesReferences)?.boolValue ?? cachesReferences
            let seed = NFKMLXGenerationBackend.seed(request)
            job.reportProgress(0)
            let pixels: MLXArray
            if let source, let mask {
                let (sourcePixels, repaint) = try Self.backendInpaintingPixels(image: source, mask: mask)
                pixels = try inpaint(prompt: prompt, negativePrompt: negative, image: sourcePixels, mask: repaint,
                                     strength: parameter(NFKParameterStrength)?.doubleValue ?? 0.8, seed: seed,
                                     references: references)
            } else {
                pixels = try image(forPrompt: prompt, negativePrompt: negative,
                                   width: parameter(NFKParameterWidth)?.intValue ?? 1024,
                                   height: parameter(NFKParameterHeight)?.intValue ?? 1024,
                                   seed: seed, references: references)
            }
            if job.status == .cancelled { return nil }
            let result = NFKMLXGenerationBackend.imageResult(try NFKMLXGenerationBackend.cgImage(pixels[0]))
            job.reportProgress(1)
            return result
        }
    }

    /// An image (a `CGImage` or an `MTLTexture`) as `[1, H, W, 3]` in `[0, 1]`, resampled bilinearly to
    /// `height` × `width`, or to its own size rounded down to a multiple of 16 where those are nil.
    private static func backendPixels(_ image: Any, height: Int? = nil, width: Int? = nil) throws -> MLXArray {
        let rgb = try NFKMLXImageBridge.tensor(from: image, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
        let (sourceHeight, sourceWidth) = (rgb.dim(0), rgb.dim(1))
        let targetHeight = height ?? sourceHeight / 16 * 16
        let targetWidth = width ?? sourceWidth / 16 * 16
        let batched = rgb.reshaped([1, sourceHeight, sourceWidth, 3])
        guard targetHeight != sourceHeight || targetWidth != sourceWidth else {
            return batched
        }
        return NFKMLXResample.resizeBilinear(batched, height: targetHeight, width: targetWidth)
    }

    /// The image `[1, H, W, 3]` and the mask's luminance `[1, H, W, 1]` at the image's size rounded down to a
    /// multiple of 16, the mask read with the ITU-R 601-2 weights of PIL's `L` conversion as
    /// ``cgInpaint(image:mask:prompt:negativePrompt:strength:seed:)`` reads it.
    private static func backendInpaintingPixels(image: Any, mask: Any) throws -> (MLXArray, MLXArray) {
        let source = try NFKMLXGenerationBackend.cgImage(fromInput: image)
        let height = source.height / 16 * 16, width = source.width / 16 * 16
        guard height > 0, width > 0 else {
            throw NFKMLXError.unsupportedConfiguration("inpainting needs an image at least 16 pixels on each side")
        }
        let weights = MLXArray([Float(0.299), 0.587, 0.114]).reshaped([1, 1, 1, 3])
        let luminance = (try backendPixels(mask, height: height, width: width) * weights).sum(axis: -1, keepDims: true)
        return (try backendPixels(source, height: height, width: width), luminance)
    }

    /// Assembles the model from a downloaded diffusers FLUX.2 release directory, held as
    /// ``NFKMLXResidency/automatic`` decides, and returns it behind the InferKit contract
    /// (``inferenceBackend()``). The directory's own configuration selects the variant. Introduced in
    /// InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try flux2(directoryURL: directoryURL).inferenceBackend()
    }

    /// ``backend(directoryURL:)`` holding the stages as `residency` says. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try flux2(directoryURL: directoryURL, residency: residency).inferenceBackend()
    }

    /// Downloads a diffusers FLUX.2 [klein] release and returns the model behind the InferKit contract.
    ///
    /// @discussion `repo` selects the variant: `black-forest-labs/FLUX.2-klein-4B` and
    /// `FLUX.2-klein-base-4B` (public, about 16 GB), or `FLUX.2-klein-9B`, `FLUX.2-klein-base-9B`, and
    /// `FLUX.2-klein-9b-kv` (gated, about 35 GB; the caller accepts the license on Hugging Face and sets
    /// `NFKHFHub.defaultAccessToken` before the first download). A file already in the cache is not fetched
    /// again. The call blocks on the network, so run it off the main and render threads. Introduced in
    /// InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try flux2(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL).inferenceBackend()
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
        try flux2(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
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

    /// Registers the backend under ``modelName``. The registry's URL is the release directory, whose own
    /// configuration selects the variant; nil is refused. Introduced in InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXGenerationBackend.registerDirectoryModel(modelName) { try backend(directoryURL: $0) }
    }
}
