//
//  NFKMLXModernBERTBackend.swift
//  InferKitMLX
//
//  The gte-reranker-modernbert-base cross-encoder behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXModernBERTReranker {

    /// The reranker behind the InferKit contract.
    ///
    /// @discussion The backend reads the query under `NFKInputPrompt` and the documents under
    /// `NFKInputDocuments` (`NSArray<NSString *>`), both required, and returns the ``NFKMLXRerankKey``
    /// fields under `NFKOutputStructured`: each document's unbounded relevance logit, in the documents'
    /// order, and the indices from most relevant to least. Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: Self.modelName, inputKeys: [NFKInputPrompt, NFKInputDocuments],
                                parameterKeys: [],
                                outputKeys: [NFKOutputStructured]) { [self] request, job in
            let query = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            let documents = try NFKMLXGenerationBackend.textCandidates(request)
            return NFKMLXGenerationBackend.rerank(documents, job: job) { score(query: query, document: $0) }
        }
    }

    /// Builds the reranker from a downloaded release directory, as ``reranker(directoryURL:)`` does, and
    /// returns it behind the InferKit contract (``inferenceBackend()``). Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try reranker(directoryURL: directoryURL).inferenceBackend()
    }

    /// Downloads the release, as ``reranker(repo:revision:cacheDirectoryURL:)`` does, and returns the
    /// reranker behind the InferKit contract.
    ///
    /// @discussion A file already in the cache is not fetched again. The call blocks on the network, so
    /// run it off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try reranker(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL).inferenceBackend()
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
