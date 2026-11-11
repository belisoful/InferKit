//
//  NFKMLXQwen3VLBackend.swift
//  InferKitMLX
//
//  Qwen3-VL behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXQwen3VL {

    /// The model behind the InferKit contract.
    ///
    /// @discussion The backend reads the image under `NFKInputImage` (a `CGImage` or an `MTLTexture`) and
    /// the question under `NFKInputPrompt`, both required, and returns the answer under `NFKOutputText`.
    /// `NFKParameterMaxTokens` bounds the answer (128). The model reads one user turn, so the backend
    /// takes no `NFKInputMessages`. A streamed decoder reports `streamed` under `NFKModelInfoResidency`,
    /// and a read of a streamed layer that fails fails the request. Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        let info: [String: Any]? = streamedLayerCount > 0 ? [NFKModelInfoResidency: NFKModelResidencyStreamed] : nil
        return NFKMLXGenerationBackend.imageQuestion(identifier: Self.modelName, modelInfo: info,
                                                     adapterStrengths: adapterStrengths) { [self] image, question, maxTokens in
            try answer(image: image, question: question, maxTokens: maxTokens)
        }
    }

    /// Holds released LoRA adapters beside the decoder's layers, unmerged, at their strengths
    /// (``NFKMLXDecoderAdapters``: PEFT's and mlx-lm's layouts), which ``inferenceBackend()`` runs and a
    /// request's `NFKParameterAdapterStrengths` changes. transformers' `model.language_model.` reads as the
    /// decoder; an adapter of the vision tower is refused.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` for a decoder that streams layers, and
    ///   `NFKMLXError.loRANotApplicable` for an adapter the decoder cannot hold, before any layer changes.
    func holdAdapters(_ adapters: [(url: URL, strength: Float)]) throws {
        guard !adapters.isEmpty else { return }
        try textDecoder.requireHeldLayers(for: "hold adapters")
        adapterStrengths = try NFKMLXDecoderAdapters.hold(adapters, in: textDecoder)
    }

    /// Builds the model from a downloaded release directory, as ``model(directoryURL:)`` does, and returns
    /// it behind the InferKit contract (``inferenceBackend()``). Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try model(directoryURL: directoryURL).inferenceBackend()
    }

    /// Builds the model as ``model(directoryURL:precision:residency:)`` does and returns it behind the
    /// InferKit contract. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:precision:residency:error:)
    public static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try model(directoryURL: directoryURL, precision: precision, residency: residency).inferenceBackend()
    }

    /// Downloads a release, as ``model(repo:revision:cacheDirectoryURL:)`` does, and returns the model
    /// behind the InferKit contract.
    ///
    /// @discussion A file already in the cache is not fetched again. The call blocks on the network, so
    /// run it off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try model(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL).inferenceBackend()
        }
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

    /// Registers the backend under ``modelName``. The registry's URL is the release directory; nil is
    /// refused. Introduced in InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXGenerationBackend.registerDirectoryModel(modelName) { try backend(directoryURL: $0) }
    }
}
