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
    /// takes no `NFKInputMessages`. Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend.imageQuestion(identifier: Self.modelName) { [self] image, question, maxTokens in
            answer(image: image, question: question, maxTokens: maxTokens)
        }
    }

    /// Builds the model from a downloaded release directory, as ``model(directoryURL:)`` does, and returns
    /// it behind the InferKit contract (``inferenceBackend()``). Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try model(directoryURL: directoryURL).inferenceBackend()
    }

    /// Downloads a release, as ``model(repo:revision:cacheDirectoryURL:)`` does, and returns the model
    /// behind the InferKit contract.
    ///
    /// @discussion A file already in the cache is not fetched again. The call blocks on the network, so
    /// run it off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try model(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL).inferenceBackend()
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
