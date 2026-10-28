//
//  NFKMLXGenerationJob.swift
//  InferKitMLX
//
//  The job form of a text-generation backend: the run on a background task, the reply so far as the
//  tokens arrive, and a cancellation that ends generation at the next token.
//

import Foundation
import InferKit

/// A text-generation run submitted as a job: what it reports while it generates, and whether it was
/// cancelled.
///
/// @discussion The run calls ``proceed(produced:limit:partial:)`` after each token. Each call reports
/// the share of the token limit produced, and answers false once the job is cancelled, which the
/// generation loop takes as the signal to stop. The partial result decodes the whole reply so far, so
/// it is built at most once per `partialResultInterval`; the first token always reports one.
final class NFKMLXGenerationJob: @unchecked Sendable {

    private let job: NFKInferenceJob
    private let partialResultInterval: TimeInterval
    private var lastPartial: Date?

    private init(job: NFKInferenceJob, partialResultInterval: TimeInterval) {
        self.job = job
        self.partialResultInterval = partialResultInterval
    }

    /// Whether the job was cancelled.
    var isCancelled: Bool { job.status == .cancelled }

    /// Starts `run` on a background task and answers its job. The job finishes with the run's result or
    /// its error; a cancelled job finishes when it is cancelled, and the run's end is not reported.
    static func submit(partialResultInterval: TimeInterval,
                       run: @escaping @Sendable (NFKMLXGenerationJob) throws -> NFKInferenceResult) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let generation = NFKMLXGenerationJob(job: job, partialResultInterval: partialResultInterval)
        Task.detached(priority: .userInitiated) {
            guard !generation.isCancelled else { return }
            do {
                let result = try run(generation)
                if !generation.isCancelled { job.finish(with: result) }
            } catch {
                if !generation.isCancelled { job.finish(withError: error as NSError) }
            }
        }
        return job
    }

    /// Reports `produced` tokens of `limit`, with the partial result when one is due, and answers
    /// whether generation continues. Called from the generation loop only.
    func proceed(produced: Int, limit: Int, partial: () -> [String: Any]) -> Bool {
        guard !isCancelled else { return false }
        let now = Date()
        var result: NFKInferenceResult?
        if lastPartial.map({ now.timeIntervalSince($0) >= partialResultInterval }) ?? true {
            lastPartial = now
            result = NFKInferenceResult(outputs: partial())
        }
        job.reportProgress(min(Double(produced) / Double(max(limit, 1)), 0.99), partialResult: result)
        return !isCancelled
    }
}

/// The reply a running generation has produced so far, split as the finished reply is split.
enum NFKMLXPartialReply {

    /// The partial outputs for `text`: `NFKOutputText` holds the answer so far, and
    /// `NFKOutputReasoning` the chain so far when there is one.
    ///
    /// @discussion Each partial extends the one before it, so a caller can send the difference. Text
    /// that may still turn into a marker is held back: a trailing piece of a reasoning or call marker,
    /// and a trailing replacement character, which a multi-byte character split across tokens decodes
    /// to until its last byte arrives. A prompt that opens the chain leaves the model writing it
    /// straight away, so `opensChain` makes the text before a closing marker reasoning. The answer
    /// stops at the first call marker in `callMarkers`, since the calls are read from the finished
    /// reply.
    static func outputs(_ text: String, format: NFKMLXReasoningFormat?, opensChain: Bool,
                        callMarkers: [String]) -> [String: Any] {
        let markers = (format.map { [$0.opening, $0.closing] + $0.answerPrefixForms } ?? []) + callMarkers
        let settled = withoutPendingMarker(text, markers: markers.filter { !$0.isEmpty })
        var reasoning = ""
        var answer = settled
        if let format {
            if opensChain, !settled.contains(format.closing), !settled.contains(format.opening) {
                reasoning = settled.trimmingCharacters(in: .whitespacesAndNewlines)
                answer = ""
            } else {
                (reasoning, answer) = format.split(settled)
            }
        }
        if let call = callMarkers.compactMap({ answer.range(of: $0)?.lowerBound }).min() {
            answer = String(answer[..<call]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        var outputs: [String: Any] = [NFKOutputText: answer]
        if !reasoning.isEmpty {
            outputs[NFKOutputReasoning] = reasoning
        }
        return outputs
    }

    /// Whether the prompt's text ends by opening the chain, so the model's first tokens are reasoning.
    static func opensChain(_ prompt: [Int], format: NFKMLXReasoningFormat?, decode: ([Int]) -> String) -> Bool {
        guard let format, !format.opening.isEmpty else { return false }
        return decode(Array(prompt.suffix(16))).trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix(format.opening)
    }

    /// `text` without a trailing replacement character or a trailing start of one of `markers`.
    static func withoutPendingMarker(_ text: String, markers: [String]) -> String {
        var settled = text
        while settled.last == "\u{FFFD}" {
            settled.removeLast()
        }
        var cut = settled.endIndex
        for marker in markers {
            for length in stride(from: marker.count - 1, through: 1, by: -1)
            where settled.hasSuffix(marker.prefix(length)) {
                let start = settled.index(settled.endIndex, offsetBy: -length)
                if start < cut { cut = start }
                break
            }
        }
        return String(settled[..<cut])
    }
}
