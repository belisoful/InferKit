//
//  NFKSpeechAnalyzerLiveStream.swift
//  InferKitAppleSwift
//
//  Live transcription: one SpeechAnalyzer session over a stream's audio units.
//
//  Introduced in InferKit 0.4.0.
//

import AVFoundation
import CoreMedia
import Foundation
import InferKit
import Speech

extension NFKSpeechAnalyzerBackend {

    /// Opens a live transcription over audio units in any format. The stream holds one analyzer session
    /// fed from an input sequence: each unit's audio is converted to the analyzer's format and appended,
    /// timed from the stream's first sample. The analyzer answers asynchronously, so a unit's result holds
    /// what it reported since the unit before:
    ///
    /// - `NFKOutputSegments`: the spans it finalized, each with its transcript as the label and its time
    ///   range in seconds from the stream's start.
    /// - `NFKOutputText`: those spans' transcripts, joined by spaces, or an empty string.
    /// - `NFKOutputTentativeText`: the transcript of the audio after the last finalized span, which the
    ///   analyzer may still revise, or an empty string.
    ///
    /// The parameters choose the locale under `NFKParameterSourceLanguage`, as a request does; opening
    /// installs its assets first when the backend has not prepared it. `reset` ends the session and starts
    /// another, whose times begin again at zero, and `close` ends it. Introduced in InferKit 0.4.0.
    @objc(openLiveStreamWithParameters:error:)
    public func openLiveStream(parameters: [String: Any]?) throws -> any NFKLiveStream {
        let locale = try Self.locale(for: NFKInferenceRequest(inputs: [:], parameters: parameters ?? [:]),
                                     default: self.locale)
        try prepareIfNeeded(locale)
        let timeout = transcriptionTimeout
        var session = try NFKSpeechAnalyzerLiveSession.open(locale: locale, timeout: timeout)
        return NFKBlockLiveStream(parameters: parameters, unitHandler: { unit, errorPointer in
            do {
                try session.append(unit.input(forKey: NFKInputAudio))
                return session.drain()
            } catch {
                errorPointer?.pointee = error as NSError
                return nil
            }
        }, resetHandler: {
            session.finish()
            if let fresh = try? NFKSpeechAnalyzerLiveSession.open(locale: locale, timeout: timeout) {
                session = fresh
            }
        }, closeHandler: {
            session.finish()
        })
    }
}

/// One analyzer session behind a live stream: the input sequence's continuation, the analyzer, and the
/// results its transcriber has reported and the stream has not answered yet.
final class NFKSpeechAnalyzerLiveSession: @unchecked Sendable {

    private let continuation: AsyncStream<AnalyzerInput>.Continuation
    private let format: AVAudioFormat
    private let analyzer: SpeechAnalyzer
    private let collector: Task<Void, Never>
    private let results: NFKSpeechAnalyzerLiveResults
    private var converter: AVAudioConverter?
    private var appendedFrames: AVAudioFramePosition = 0
    private var finished = false

    private init(continuation: AsyncStream<AnalyzerInput>.Continuation, format: AVAudioFormat,
                 analyzer: SpeechAnalyzer, transcriber: SpeechTranscriber) {
        self.continuation = continuation
        self.format = format
        self.analyzer = analyzer
        let results = NFKSpeechAnalyzerLiveResults()
        collector = Task.detached(priority: .userInitiated) {
            do {
                for try await result in transcriber.results {
                    results.receive(text: String(result.text.characters), range: result.range, isFinal: result.isFinal)
                }
            } catch {
                results.fail(error)
            }
        }
        self.results = results
    }

    /// A session on `locale`, its analyzer started on an input sequence. The analyzer's format is chosen
    /// asynchronously, so the call waits for it up to `timeout` seconds.
    static func open(locale: Locale, timeout: TimeInterval) throws -> NFKSpeechAnalyzerLiveSession {
        let outcome = NFKAppleOutcome<NFKSpeechAnalyzerLiveSessionBox>()
        let semaphore = DispatchSemaphore(value: 0)
        let task = Task.detached(priority: .userInitiated) {
            do {
                guard let supported = await SpeechTranscriber.supportedLocale(equivalentTo: locale) else {
                    throw NFKSpeechAnalyzerLiveSession.error(.error_InferenceUnsupported,
                                                             "the speech analyzer does not transcribe \(locale.identifier)")
                }
                let transcriber = SpeechTranscriber(locale: supported, transcriptionOptions: [],
                                                    reportingOptions: [.volatileResults],
                                                    attributeOptions: [.audioTimeRange])
                guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) else {
                    throw NFKSpeechAnalyzerLiveSession.error(.error_InferenceUnsupported,
                                                             "the speech analyzer names no audio format")
                }
                let (inputs, continuation) = AsyncStream<AnalyzerInput>.makeStream()
                let analyzer = SpeechAnalyzer(modules: [transcriber])
                let session = NFKSpeechAnalyzerLiveSession(continuation: continuation, format: format,
                                                           analyzer: analyzer, transcriber: transcriber)
                try await analyzer.start(inputSequence: inputs)
                outcome.succeed(NFKSpeechAnalyzerLiveSessionBox(session: session))
            } catch {
                outcome.fail(error)
            }
            semaphore.signal()
        }
        try NFKAppleWait.wait(semaphore, seconds: timeout, cancelling: task, code: .error_InferenceNotReady,
                              reason: "the speech analyzer did not start")
        return try outcome.value().session
    }

    /// Converts a unit's audio to the analyzer's format and appends it, timed from the stream's start.
    func append(_ audio: Any?) throws {
        guard let buffer = try Self.buffer(from: audio) else {
            throw Self.error(.error_InferenceMissingInput, "a live unit carries its audio under NFKInputAudio")
        }
        if let failure = results.failure {
            throw failure
        }
        let converted = try convert(buffer)
        guard converted.frameLength > 0 else {
            return
        }
        let start = CMTime(value: appendedFrames, timescale: CMTimeScale(format.sampleRate))
        appendedFrames += AVAudioFramePosition(converted.frameLength)
        continuation.yield(AnalyzerInput(buffer: converted, bufferStartTime: start))
    }

    /// The result of the spans finalized and the tentative transcript since the last drain.
    func drain() -> NFKInferenceResult {
        let (spans, tentative) = results.drain()
        return NFKInferenceResult(outputs: [
            NFKOutputSegments: spans,
            NFKOutputText: spans.compactMap(\.label).joined(separator: " "),
            NFKOutputTentativeText: tentative,
        ])
    }

    /// Ends the input sequence and stops the analyzer and the collector.
    func finish() {
        guard !finished else {
            return
        }
        finished = true
        continuation.finish()
        let analyzer = analyzer
        Task.detached(priority: .userInitiated) {
            await analyzer.cancelAndFinishNow()
        }
        collector.cancel()
    }

    deinit {
        finish()
    }

    /// The unit's audio converted to the analyzer's format. One converter serves units of one format, so
    /// a resampler keeps its filter state across units; a unit in another format starts a new one.
    private func convert(_ buffer: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
        if buffer.format == format {
            return buffer
        }
        if converter?.inputFormat != buffer.format {
            converter = AVAudioConverter(from: buffer.format, to: format)
        }
        guard let converter else {
            throw Self.error(.error_InferenceUnsupported, "the unit's audio format cannot convert to the analyzer's")
        }
        let ratio = format.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount((Double(buffer.frameLength) * ratio).rounded(.up)) + 32
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
            throw Self.error(.error_InferenceBackendFailure, "no buffer for the converted audio")
        }
        var supplied = false
        var conversionError: NSError?
        converter.convert(to: output, error: &conversionError) { _, status in
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return buffer
        }
        if let conversionError {
            throw conversionError
        }
        return output
    }

    /// An `AVAudioPCMBuffer` as given, or one read from WAV bytes or an audio asset's file.
    private static func buffer(from audio: Any?) throws -> AVAudioPCMBuffer? {
        if let buffer = audio as? AVAudioPCMBuffer {
            return buffer
        }
        var url: URL?
        if let asset = audio as? NFKAudioAsset {
            url = asset.fileURL
        } else if let data = audio as? Data {
            let file = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("\(UUID().uuidString).wav")
            try data.write(to: file)
            url = file
        }
        guard let url else {
            return nil
        }
        let file = try AVAudioFile(forReading: url)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                            frameCapacity: AVAudioFrameCount(max(file.length, 1))) else {
            return nil
        }
        try file.read(into: buffer)
        return buffer
    }

    fileprivate static func error(_ code: NFKInferenceError, _ reason: String) -> NSError {
        NSError(domain: NFKInferenceErrorDomain, code: code.rawValue, userInfo: [NSLocalizedDescriptionKey: reason])
    }
}

/// Carries a session out of the task that opened it.
struct NFKSpeechAnalyzerLiveSessionBox: @unchecked Sendable {
    let session: NFKSpeechAnalyzerLiveSession
}

/// The transcriber's reports, collected as they arrive and taken by the stream's next unit.
final class NFKSpeechAnalyzerLiveResults: @unchecked Sendable {
    private let lock = NSLock()
    private var finalized: [NFKAudioSegment] = []
    private var tentative = ""
    private var storedFailure: Error?

    var failure: Error? {
        lock.withLock { storedFailure }
    }

    /// A final result joins the finalized spans and clears the tentative transcript; a volatile one
    /// replaces the tentative transcript.
    func receive(text: String, range: CMTimeRange, isFinal: Bool) {
        lock.withLock {
            if isFinal {
                finalized.append(NFKAudioSegment(startSeconds: range.start.seconds, endSeconds: range.end.seconds,
                                                 label: text, confidence: 1.0))
                tentative = ""
            } else {
                tentative = text
            }
        }
    }

    func fail(_ error: Error) {
        lock.withLock { storedFailure = error }
    }

    /// The spans finalized since the last drain, and the current tentative transcript.
    func drain() -> ([NFKAudioSegment], String) {
        lock.withLock {
            defer { finalized.removeAll() }
            return (finalized, tentative)
        }
    }
}
