//
//  NFKMLXTextToImageDownload.swift
//  InferKitMLX
//
//  Stable Diffusion text-to-image and IP-Adapter: the download peers of the directory factories, and the
//  IP-Adapter embedding key where Objective-C reaches it.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The request-input keys the Stable Diffusion backends read beyond the core's, so an Objective-C caller
/// sets them the way it sets `NFKInputPrompt`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXTextToImageInputKey)
public final class NFKMLXTextToImageInputKey: NSObject {
    /// An IP-Adapter reference-image embedding: an `NSArray` of numbers, the CLIP image embedding a caller
    /// computed with the adapter's own image encoder. It is the string the Swift constant
    /// `NFKMLXInputImageEmbedding` holds. Introduced in InferKit 0.4.0.
    @objc public static let imageEmbedding = NFKMLXInputImageEmbedding
}

extension NFKMLXTextToImage {

    /// Downloads the named release and builds a text-to-image backend from it, as
    /// ``backend(model:directoryURL:)`` does.
    ///
    /// @discussion The download is the UNet, the autoencoder, each text tower, and each tokenizer's
    /// vocabulary, from `stable-diffusion-v1-5/stable-diffusion-v1-5`, `stabilityai/stable-diffusion-2-1-base`
    /// (gated: set `NFKHFHub.defaultAccessToken` after accepting its license), or `stabilityai/sdxl-turbo`.
    /// A file already in the cache is not fetched again. The call blocks on the network, so run it off
    /// the main and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithModel:cacheDirectoryURL:error:)
    public static func backend(model: NFKMLXStableDiffusionModel, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(model: model,
                    directoryURL: try NFKMLXStableDiffusionRelease(model).download(cacheDirectoryURL: cacheDirectoryURL))
    }

    /// The asynchronous form of ``backend(model:cacheDirectoryURL:)``. The handler runs on a background
    /// queue. Introduced in InferKit 0.4.0.
    @objc(backendWithModel:cacheDirectoryURL:completionHandler:)
    public static func backend(model: NFKMLXStableDiffusionModel, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(model: model, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Downloads the named release and an IP-Adapter checkpoint, then builds the IP-Adapter backend as
    /// ``imageAdapterBackend(model:directoryURL:adapterURL:scale:)`` does.
    ///
    /// @discussion `adapterRepo`, `adapterWeightsPath`, and `adapterRevision` name the adapter: the h94
    /// SD 1.5 adapter is `models/ip-adapter_sd15.safetensors` in `h94/IP-Adapter`. A request reads the
    /// reference under ``NFKMLXTextToImageInputKey/imageEmbedding``. A file already in the cache is not
    /// fetched again. The call blocks on the network, so run it off the main and render threads.
    /// Introduced in InferKit 0.4.0.
    @objc(imageAdapterBackendWithModel:adapterRepo:adapterWeightsPath:adapterRevision:scale:cacheDirectoryURL:error:)
    public static func imageAdapterBackend(model: NFKMLXStableDiffusionModel, adapterRepo: String,
                                           adapterWeightsPath: String, adapterRevision: String?, scale: Float,
                                           cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let directory = try NFKMLXStableDiffusionRelease(model).download(cacheDirectoryURL: cacheDirectoryURL)
        let adapterURL = try NFKMLXDownload.weightsURL(repo: adapterRepo, weightsPath: adapterWeightsPath,
                                                       revision: adapterRevision,
                                                       cacheDirectoryURL: cacheDirectoryURL)
        return try imageAdapterBackend(model: model, directoryURL: directory, adapterURL: adapterURL, scale: scale)
    }

    /// The asynchronous form of
    /// ``imageAdapterBackend(model:adapterRepo:adapterWeightsPath:adapterRevision:scale:cacheDirectoryURL:)``.
    /// The handler runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(imageAdapterBackendWithModel:adapterRepo:adapterWeightsPath:adapterRevision:scale:cacheDirectoryURL:completionHandler:)
    public static func imageAdapterBackend(model: NFKMLXStableDiffusionModel, adapterRepo: String,
                                           adapterWeightsPath: String, adapterRevision: String?, scale: Float,
                                           cacheDirectoryURL: URL?,
                                           completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try imageAdapterBackend(model: model, adapterRepo: adapterRepo, adapterWeightsPath: adapterWeightsPath,
                                    adapterRevision: adapterRevision, scale: scale,
                                    cacheDirectoryURL: cacheDirectoryURL)
        }
    }
}
