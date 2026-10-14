//
//  NFKMLXRerankBackend.swift
//  InferKitMLX
//
//  The request and result shape the reranking backends share.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The names a reranking backend reads and writes beside the core keys.
///
/// @discussion A reranking backend reads the query under `NFKInputPrompt` and the candidates under
/// `NFKInputDocuments` (an `NSArray` of `NSString`; a multimodal reranker also takes a `CGImage` or an
/// `MTLTexture` as a candidate). It returns an `NSDictionary` under `NFKOutputStructured` holding
/// ``scores`` and ``rankedIndices``. Introduced in InferKit 0.4.0.
@objc(NFKMLXRerankKey)
public final class NFKMLXRerankKey: NSObject {
    /// The `NFKOutputStructured` field holding one relevance score per document, in the documents' order
    /// (`NSArray<NSNumber *>`). A higher score is more relevant. Introduced in InferKit 0.4.0.
    @objc public static let scores = "scores"
    /// The `NFKOutputStructured` field holding the documents' indices from most relevant to least
    /// (`NSArray<NSNumber *>`); equal scores keep the documents' order. Introduced in InferKit 0.4.0.
    @objc public static let rankedIndices = "rankedIndices"
    /// The request parameter naming the task relevance is judged for (`NSString`), for a reranker that
    /// reads one. Introduced in InferKit 0.4.0.
    @objc public static let instruction = "NFKMLXParameterRerankInstruction"
}

extension NFKMLXGenerationBackend {

    /// The candidates under `NFKInputDocuments`; a missing one is an unsupported input.
    static func rerankCandidates(_ request: NFKInferenceRequest) throws -> [Any] {
        guard let candidates = request.input(forKey: NFKInputDocuments) as? [Any] else {
            throw NFKMLXError.unsupportedInput
        }
        return candidates
    }

    /// The candidates under `NFKInputDocuments` as text; a candidate of another type is an unsupported
    /// input.
    static func textCandidates(_ request: NFKInferenceRequest) throws -> [String] {
        try rerankCandidates(request).map { candidate in
            guard let text = candidate as? String else { throw NFKMLXError.unsupportedInput }
            return text
        }
    }

    /// Scores each candidate with `score`, reporting progress per candidate; nil when `job` was
    /// cancelled.
    static func rerank<Candidate>(_ candidates: [Candidate], job: NFKInferenceJob,
                                  score: (Candidate) throws -> Double) rethrows -> NFKInferenceResult? {
        var scores = [Double]()
        for (index, candidate) in candidates.enumerated() {
            if job.status == .cancelled { return nil }
            job.reportProgress(Double(index) / Double(candidates.count))
            scores.append(try score(candidate))
        }
        job.reportProgress(1)
        return rerankResult(scores)
    }

    /// A result carrying `scores` and the ranking they give under `NFKOutputStructured`.
    static func rerankResult(_ scores: [Double]) -> NFKInferenceResult {
        let order = scores.indices.sorted { scores[$0] > scores[$1] || (scores[$0] == scores[$1] && $0 < $1) }
        let fields: [String: Any] = [
            NFKMLXRerankKey.scores: scores.map { NSNumber(value: $0) },
            NFKMLXRerankKey.rankedIndices: order.map { NSNumber(value: $0) },
        ]
        return NFKInferenceResult(outputs: [NFKOutputStructured: fields])
    }
}
