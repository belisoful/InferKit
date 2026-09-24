//
//  NFKSpeechAnalyzerBackend.swift
//  InferKitAppleSwift
//

import AVFoundation
import Foundation
import InferKit
import Speech

/// Transcribes a recording with `SpeechAnalyzer`, Apple's speech stack from macOS 26.
///
/// The core already wraps `SFSpeechRecognizer` as `NFKSpeechRecognitionBackend`, which is Objective-C
/// and needs no companion. This one is here because `SpeechAnalyzer` is a Swift actor and its results
/// arrive as an `AsyncSequence`, neither of which a pure Objective-C target can reach. It transcribes
/// on device, needs no authorization prompt, and reads the locales the system has installed.
///
/// - Input: an `NFKAudioAsset`, an `NSURL`, or `NSData` under `NFKInputAudio`.
/// - Output: the transcription under `NFKOutputText`, and one `NFKAudioSegment` per reported range
///   under `NFKOutputSegments`.
///
/// `isReady` reports whether the chosen locale is installed. `prepare()` reserves it and asks the
/// system to install the assets, which is a download the first time. Introduced in InferKit 0.4.0.
@objc(NFKSpeechAnalyzerBackend)
public final class NFKSpeechAnalyzerBackend: NSObject, NFKInferenceBackend {

    /// The locale to transcribe. The user's current locale by default. Setting nil restores it.
    @objc public var locale: Locale {
        get { lock.withLock { explicitLocale } ?? Locale.current }
        set { lock.withLock { explicitLocale = newValue } }
    }

    /// The locales this machine can transcribe without a download. The system answers
    /// asynchronously, so the handler runs on an arbitrary thread.
    @objc public static func installedLocales(completionHandler handler: @escaping ([Locale]) -> Void) {
        Task.detached(priority: .userInitiated) {
            handler(await SpeechTranscriber.installedLocales)
        }
    }

    /// The locales the system can install, which is a superset of the installed ones.
    @objc public static func supportedLocales(completionHandler handler: @escaping ([Locale]) -> Void) {
        Task.detached(priority: .userInitiated) {
            handler(await SpeechTranscriber.supportedLocales)
        }
    }

    private let lock = NSLock()
    private var explicitLocale: Locale?
    private var preparedLocale: Locale?

    @objc public override init() {
        super.init()
    }

    /// A backend for one locale.
    @objc public convenience init(locale: Locale) {
        self.init()
        self.locale = locale
    }

    // MARK: NFKInferenceBackend

    /// YES once `prepare()` has installed the assets for the locale this backend is set to.
    ///
    /// The system reports availability and installed locales asynchronously, and this property
    /// cannot wait, so it answers from what `prepare()` found. A backend that has not been prepared
    /// reports NO even where the assets happen to be installed already; preparing is cheap in that
    /// case, because the install request is nil when nothing needs installing.
    @objc public var isReady: Bool {
        lock.withLock { preparedLocale != nil && preparedLocale == (explicitLocale ?? Locale.current) }
    }

    @objc public var backendIdentifier: String { "apple-speech-analyzer" }

    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }

    /// Reserves the locale and installs its assets, which is a download the first time. The call
    /// blocks until the install finishes, because a backend is prepared off the caller's thread.
    @objc(prepareWithError:)
    public func prepare() throws {
        let locale = self.locale
        let outcome = NFKAppleOutcome<Bool>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do {
                guard SpeechTranscriber.isAvailable else {
                    throw Self.error(.error_InferenceUnsupported, "this machine has no speech analyzer")
                }
                guard let supported = await SpeechTranscriber.supportedLocale(equivalentTo: locale) else {
                    throw Self.error(.error_InferenceUnsupported,
                                     "the speech analyzer does not transcribe \(locale.identifier)")
                }
                let transcriber = SpeechTranscriber(locale: supported,
                                                    preset: .timeIndexedTranscriptionWithAlternatives)
                _ = try await AssetInventory.reserve(locale: supported)
                // Nil means the assets are already installed, which is the second run.
                if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                    try await request.downloadAndInstall()
                }
                outcome.succeed(true)
            } catch {
                outcome.fail(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        _ = try outcome.value()
        lock.withLock { preparedLocale = locale }
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let url = Self.audioURL(for: request) else {
            throw Self.error(.error_InferenceMissingInput, "no audio file is set under NFKInputAudio")
        }
        let outcome = NFKAppleOutcome<NFKInferenceResult>()
        let semaphore = DispatchSemaphore(value: 0)
        let locale = self.locale
        Task.detached(priority: .userInitiated) {
            do {
                outcome.succeed(try await Self.transcribe(url: url, locale: locale))
            } catch {
                outcome.fail(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        return try outcome.value()
    }

    // MARK: Transcribing

    /// The analyzer reads the file while its module reports ranges, so the results are collected as
    /// they arrive and the read is awaited before the sequence is finished.
    private static func transcribe(url: URL, locale: Locale) async throws -> NFKInferenceResult {
        guard SpeechTranscriber.isAvailable else {
            throw error(.error_InferenceUnsupported, "this machine has no speech analyzer")
        }
        guard let supported = await SpeechTranscriber.supportedLocale(equivalentTo: locale) else {
            throw error(.error_InferenceUnsupported, "the speech analyzer does not transcribe \(locale.identifier)")
        }
        let transcriber = SpeechTranscriber(locale: supported, preset: .timeIndexedTranscriptionWithAlternatives)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let file = try AVAudioFile(forReading: url)

        let collector = Task { () -> ([String], [NFKAudioSegment]) in
            var texts: [String] = []
            var segments: [NFKAudioSegment] = []
            for try await result in transcriber.results {
                let text = String(result.text.characters)
                texts.append(text)
                segments.append(NFKAudioSegment(startSeconds: result.range.start.seconds,
                                                endSeconds: result.range.end.seconds,
                                                label: text,
                                                confidence: 1.0))
            }
            return (texts, segments)
        }

        try await analyzer.start(inputAudioFile: file, finishAfterFile: true)
        let (texts, segments) = try await collector.value
        return NFKInferenceResult(outputs: [NFKOutputText: texts.joined(separator: " "),
                                            NFKOutputSegments: segments])
    }

    private static func audioURL(for request: NFKInferenceRequest) -> URL? {
        let audio = request.input(forKey: NFKInputAudio)
        if let asset = audio as? NFKAudioAsset {
            return asset.fileURL
        }
        if let url = audio as? URL {
            return url
        }
        if let data = audio as? Data {
            let url = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("\(UUID().uuidString).wav")
            return (try? data.write(to: url)) == nil ? nil : url
        }
        return nil
    }

    private static func error(_ code: NFKInferenceError, _ reason: String) -> NSError {
        NSError(domain: NFKInferenceErrorDomain,
                code: code.rawValue,
                userInfo: [NSLocalizedDescriptionKey: reason])
    }
}

/// Carries one asynchronous result across a semaphore to a synchronous caller.
final class NFKAppleOutcome<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<Value, Error>?

    func succeed(_ value: Value) {
        lock.withLock { result = .success(value) }
    }

    func fail(_ error: Error) {
        lock.withLock { result = .failure(error) }
    }

    func value() throws -> Value {
        guard let result = lock.withLock({ result }) else {
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceBackendFailure.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "the run produced nothing"])
        }
        return try result.get()
    }
}
