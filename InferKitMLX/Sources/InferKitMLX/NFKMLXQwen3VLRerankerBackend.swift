//
//  NFKMLXQwen3VLRerankerBackend.swift
//  InferKitMLX
//
//  Qwen3-VL-Reranker behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit

extension NFKMLXQwen3VLReranker {

    /// The reranker behind the InferKit contract.
    ///
    /// @discussion The backend reads the query as text under `NFKInputPrompt`, an image under
    /// `NFKInputImage`, or both; one of them is required. The documents under `NFKInputDocuments` are
    /// required, each an `NSString` or an image (`CGImage` or `MTLTexture`). `NFKMLXRerankKey.instruction`
    /// names the task (``defaultInstruction``). The result carries the ``NFKMLXRerankKey`` fields under
    /// `NFKOutputStructured`: each document's relevance between 0 and 1, in the documents' order, and the
    /// indices from most relevant to least. Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: Self.modelName,
                                inputKeys: [NFKInputPrompt, NFKInputImage, NFKInputDocuments],
                                parameterKeys: [NFKMLXRerankKey.instruction],
                                outputKeys: [NFKOutputStructured]) { [self] request, job in
            let query = NFKMLXGenerationBackend.text(request, NFKInputPrompt)
            let queryImage = try NFKMLXGenerationBackend.image(request, NFKInputImage)
            if query == nil && queryImage == nil {
                throw NFKMLXError.unsupportedInput
            }
            let instruction = NFKMLXGenerationBackend.parameterText(request, NFKMLXRerankKey.instruction)
            let documents = try NFKMLXGenerationBackend.rerankCandidates(request).map { candidate -> (String?, CGImage?) in
                if let text = candidate as? String {
                    return (text, nil)
                }
                return (nil, try NFKMLXGenerationBackend.cgImage(fromInput: candidate))
            }
            return NFKMLXGenerationBackend.rerank(documents, job: job) { document in
                score(query: query, queryImage: queryImage, document: document.0, documentImage: document.1,
                      instruction: instruction)
            }
        }
    }

    /// Builds the reranker from a downloaded release directory, as ``reranker(directoryURL:)`` does, and
    /// returns it behind the InferKit contract (``inferenceBackend()``). Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try reranker(directoryURL: directoryURL).inferenceBackend()
    }

    /// Downloads a release, as ``reranker(repo:revision:cacheDirectoryURL:)`` does, and returns the
    /// reranker behind the InferKit contract.
    ///
    /// @discussion A file already in the cache is not fetched again. The call blocks on the network, so
    /// run it off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try reranker(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL).inferenceBackend()
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
