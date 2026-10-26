//
//  NFKTranslationBackend.swift
//  InferKitAppleSwift
//

import Foundation
import InferKit
import NaturalLanguage
import Translation

/// Translates text with Apple's Translation framework, on device, with no key and no model to ship.
///
/// Translation's session is Swift-only, which is why this lives in the companion rather than the
/// core. The toolkit ships no translation model of its own, so this is the first engine to answer
/// `NFKCapabilityTranslation`; an MLX translation model registers ahead of it.
///
/// - Input: the text under `NFKInputPrompt`.
/// - Parameters: `NFKParameterTargetLanguage`, a BCP-47 tag, required. `NFKParameterSourceLanguage`,
///   also BCP-47, optional: without it ``detectedLanguage(of:)`` names the text's dominant language.
/// - Output: the translation under `NFKOutputText`. A text already in the target language comes back
///   unchanged.
///
/// A language pair needs its model installed. `prepare()` asks the system to fetch it, which shows
/// the person a prompt the first time, and `isReady` answers from what `prepare()` found because the
/// availability check is asynchronous. Introduced in InferKit 0.4.0.
@objc(NFKTranslationBackend)
public final class NFKTranslationBackend: NSObject, NFKInferenceBackend {

    /// The language to translate into, BCP-47. A request's `NFKParameterTargetLanguage` overrides it.
    @objc public var targetLanguage: String?

    /// The language to translate from, BCP-47. nil detects it from each request's text. A request's
    /// `NFKParameterSourceLanguage` overrides it.
    @objc public var sourceLanguage: String?

    /// The BCP-47 tag of the text's dominant language as NaturalLanguage recognizes it, such as "en" or
    /// "zh-Hans", or nil when it cannot tell. `languages`, when given, limits the answer to those tags.
    /// A translation session takes an installed source language and detects none itself, so a request
    /// without a source language is translated from this one. A short text can read as a language
    /// Apple does not translate ("Hi" reads as Catalan), so such a request detects again among the
    /// languages Apple translates. Introduced in InferKit 0.4.0.
    @objc(detectedLanguageOfText:amongLanguages:)
    public static func detectedLanguage(of text: String, among languages: [String]? = nil) -> String? {
        let recognizer = NLLanguageRecognizer()
        if let languages {
            recognizer.languageConstraints = languages.map { NLLanguage(rawValue: $0) }
        }
        recognizer.processString(text)
        guard let language = recognizer.dominantLanguage, language != .undetermined else {
            return nil
        }
        return language.rawValue
    }

    /// The languages Apple can translate on this machine.
    @objc public static func supportedLanguages(completionHandler handler: @escaping ([String]) -> Void) {
        Task.detached(priority: .userInitiated) {
            let languages = await LanguageAvailability().supportedLanguages
            handler(languages.map(\.maximalIdentifier))
        }
    }

    /// How long a call waits for the system to answer, in seconds. The Translation framework does not
    /// always answer at all, so every wait is bounded and reaching the bound is reported as
    /// `kNFKError_InferenceNotReady`. Default 60.
    @objc public var responseTimeout: TimeInterval = 60.0

    private let lock = NSLock()
    private var preparedPair: String?

    @objc public override init() {
        super.init()
    }

    /// A backend for one pair. A nil source detects it from each request's text.
    @objc public convenience init(sourceLanguage: String?, targetLanguage: String) {
        self.init()
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
    }

    // MARK: NFKInferenceBackend

    /// YES once `prepare()` has found the pair installed. The availability check is asynchronous and
    /// this property cannot wait, so an unprepared backend reports NO.
    @objc public var isReady: Bool {
        lock.withLock { preparedPair != nil && preparedPair == pairKey(source: sourceLanguage, target: targetLanguage) }
    }

    @objc public var backendIdentifier: String { "apple-translation" }

    @objc public var supportedInputKeys: Set<String> { [NFKInputPrompt] }

    @objc public var supportedOutputKeys: Set<String> { [NFKOutputText] }

    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterSourceLanguage, NFKParameterTargetLanguage]
    }

    @objc(prepareWithError:)
    public func prepare() throws {
        guard let target = targetLanguage else {
            throw Self.error(.error_InferenceMissingInput,
                             "set targetLanguage, or NFKParameterTargetLanguage on the request")
        }
        let source = sourceLanguage
        let outcome = NFKAppleOutcome<Bool>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do {
                try await Self.ensureInstalled(source: source, target: target)
                outcome.succeed(true)
            } catch {
                outcome.fail(error)
            }
            semaphore.signal()
        }
        try Self.wait(semaphore, seconds: responseTimeout)
        _ = try outcome.value()
        lock.withLock { preparedPair = pairKey(source: source, target: target) }
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let text = request.input(forKey: NFKInputPrompt) as? String, !text.isEmpty else {
            throw Self.error(.error_InferenceMissingInput, "no text is set under NFKInputPrompt")
        }
        let target = (request.parameter(forKey: NFKParameterTargetLanguage) as? String) ?? targetLanguage
        guard let target, !target.isEmpty else {
            throw Self.error(.error_InferenceMissingInput,
                             "NFKParameterTargetLanguage names the language to translate into")
        }
        let named = (request.parameter(forKey: NFKParameterSourceLanguage) as? String) ?? sourceLanguage
        guard let source = named ?? Self.detectedLanguage(of: text) else {
            throw Self.error(.error_InferenceMissingInput,
                             "the text's language could not be detected; set NFKParameterSourceLanguage")
        }
        if Self.isSameLanguage(source, target) {
            return NFKInferenceResult(outputs: [NFKOutputText: text])
        }

        let outcome = NFKAppleOutcome<String>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do {
                outcome.succeed(try await Self.translate(text, from: source, detected: named == nil, to: target))
            } catch {
                outcome.fail(error)
            }
            semaphore.signal()
        }
        try Self.wait(semaphore, seconds: responseTimeout)
        return NFKInferenceResult(outputs: [NFKOutputText: try outcome.value()])
    }

    // MARK: Translating

    private static func isSameLanguage(_ source: String, _ target: String) -> Bool {
        Locale.Language(identifier: source).isEquivalent(to: Locale.Language(identifier: target))
    }

    private static func translate(_ text: String, from named: String, detected: Bool, to target: String) async throws -> String {
        var source = named
        if detected {
            let known = await knownLanguageCodes()
            if !known.contains(Locale.Language(identifier: source).languageCode?.identifier ?? "") {
                // NaturalLanguage names Chinese by its script.
                let tags = known.contains("zh") ? known.union(["zh-Hans", "zh-Hant"]) : known
                guard let redetected = detectedLanguage(of: text, among: tags.sorted()) else {
                    throw error(.error_InferenceUnsupported,
                                "the text reads as \(source), which Apple does not translate; set NFKParameterSourceLanguage")
                }
                source = redetected
            }
            if isSameLanguage(source, target) {
                return text
            }
        }
        try await ensureInstalled(source: source, target: target)
        let session = TranslationSession(installedSource: Locale.Language(identifier: source),
                                         target: Locale.Language(identifier: target))
        return try await session.translate(text).targetText
    }

    private static func knownLanguageCodes() async -> Set<String> {
        Set(await LanguageAvailability().supportedLanguages.compactMap { $0.languageCode?.identifier })
    }

    /// Bounds every wait on the framework. `LanguageAvailability` does not always answer: asked about
    /// a language code it does not know, it returns nothing at all rather than reporting `unsupported`
    /// (measured on macOS 26.6.2). An unbounded wait there hangs the calling thread for the life of
    /// the process, so the wait has a deadline and reports reaching it.
    private static func wait(_ semaphore: DispatchSemaphore, seconds: TimeInterval) throws {
        if semaphore.wait(timeout: .now() + seconds) == .timedOut {
            throw error(.error_InferenceNotReady,
                        "the Translation framework did not answer within \(Int(seconds)) seconds")
        }
    }

    /// The system installs a pair on request, and refuses a pair it does not have at all, which is a
    /// different answer from one it has not downloaded yet.
    private static func ensureInstalled(source: String?, target: String) async throws {
        let availability = LanguageAvailability()
        let targetLanguage = Locale.Language(identifier: target)

        // Asking about a language the system does not know never returns, so the codes are checked
        // against the list it publishes before any pair is put to it.
        let known = await knownLanguageCodes()
        for identifier in [source, target].compactMap({ $0 }) {
            let code = Locale.Language(identifier: identifier).languageCode?.identifier
            guard let code, known.contains(code) else {
                throw error(.error_InferenceUnsupported, "Apple does not translate \(identifier)")
            }
        }

        guard let source else {
            return   // prepare() without a source; each request names or detects one.
        }
        let status = await availability.status(from: Locale.Language(identifier: source), to: targetLanguage)
        switch status {
        case .installed:
            return
        case .supported:
            throw error(.error_InferenceNotReady,
                        "the \(source) to \(target) model is supported but not installed yet")
        case .unsupported:
            throw error(.error_InferenceUnsupported, "Apple does not translate \(source) to \(target)")
        @unknown default:
            throw error(.error_InferenceUnsupported, "the \(source) to \(target) pair reported an unknown status")
        }
    }

    private func pairKey(source: String?, target: String?) -> String {
        "\(source ?? "auto")>\(target ?? "")"
    }

    private static func error(_ code: NFKInferenceError, _ reason: String) -> NSError {
        NSError(domain: NFKInferenceErrorDomain,
                code: code.rawValue,
                userInfo: [NSLocalizedDescriptionKey: reason])
    }
}

/// Activates ``NFKTranslationBackend`` through `NFKDynamicBackend` when this package is linked. The
/// core names it for `NFKCapabilityTranslation`; a registered model-backed provider wins over it.
@objc(NFKTranslationProvider)
public final class NFKTranslationProvider: NSObject, NFKDynamicBackendProvider {

    public static func makeInferenceBackend() -> (any NFKInferenceBackend)? {
        NFKTranslationBackend()
    }
}
