//
//  NFKMLXTranslationBackend.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import NaturalLanguage

/// The request keys the translation backends read beyond the core's.
///
/// @discussion The core keys (`NFKParameterSourceLanguage`, `NFKParameterTargetLanguage`,
/// `NFKParameterMaxTokens`) cover the contract; these tune the decode. An Objective-C caller sets
/// them per request the way it sets `NFKParameterTemperature`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXTranslationParameterKey)
public final class NFKMLXTranslationParameterKey: NSObject {
    /// Beams to search (NSNumber). 1 decodes greedily. The default is the model's own generation
    /// setting (Marian 4, M2M100 5, MADLAD 1).
    @objc public static let beamCount = "NFKMLXParameterBeamCount"
    /// Divides a finished hypothesis's log-probability by `length^penalty` (NSNumber). Default 1.
    @objc public static let lengthPenalty = "NFKMLXParameterLengthPenalty"
    /// Splits the input into sentences and translates each (NSNumber, boolean). Off by default: the
    /// whole input goes through the model as one sequence, paragraph by paragraph.
    @objc public static let splitsSentences = "NFKMLXParameterSplitsSentences"
}

/// A text-to-text translator the backend drives: a tokenizer-and-network pair with its language set.
///
/// Introduced in InferKit 0.4.0.
public protocol NFKMLXTranslator: AnyObject {
    /// The registered model name.
    var identifier: String { get }
    /// The source language a per-pair model is fixed to (BCP-47), or nil for a multilingual one.
    var fixedSourceLanguage: String? { get }
    /// The target language a per-pair model is fixed to (BCP-47), or nil.
    var fixedTargetLanguage: String? { get }
    /// Whether the model reads or produces `language` (a BCP-47 tag).
    func supports(language: String) -> Bool
    /// The decode a request starts from (the model's generation defaults).
    var defaultDecoding: NFKMLXSeq2SeqDecoding { get }
    /// Translates one segment.
    func translate(_ text: String, from source: String?, to target: String,
                   decoding: NFKMLXSeq2SeqDecoding) throws -> String
}

/// The `NFKInferenceBackend` face of an MLX translation model.
///
/// @discussion Reads the text under `NFKInputPrompt`, the languages under `NFKParameterSourceLanguage`
/// (optional; detected with `NLLanguageRecognizer` when absent and the model needs it) and
/// `NFKParameterTargetLanguage` (required), and returns the translation under `NFKOutputText`. The
/// input is translated paragraph by paragraph (split on line breaks), or sentence by sentence under
/// ``NFKMLXTranslationParameterKey/splitsSentences``. A request whose target the model does not produce,
/// or whose source a per-pair model does not read, fails with `kNFKError_InferenceUnsupported`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXTranslationBackend)
public final class NFKMLXTranslationBackend: NSObject, NFKInferenceBackend {
    public let translator: any NFKMLXTranslator

    public init(translator: any NFKMLXTranslator) {
        self.translator = translator
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { translator.identifier }

    /// The request parameters the backend reads.
    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterSourceLanguage, NFKParameterTargetLanguage, NFKParameterMaxTokens,
         NFKMLXTranslationParameterKey.beamCount, NFKMLXTranslationParameterKey.lengthPenalty,
         NFKMLXTranslationParameterKey.splitsSentences]
    }

    /// The request inputs the backend reads.
    @objc public var supportedInputKeys: Set<String> { [NFKInputPrompt] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let text = request.prompt else {
            throw NFKMLXError.unsupportedInput
        }
        guard let target = request.parameter(forKey: NFKParameterTargetLanguage) as? String, !target.isEmpty else {
            throw Self.error(.error_InferenceMissingInput, "a translation request names its target language under NFKParameterTargetLanguage")
        }
        let source = request.parameter(forKey: NFKParameterSourceLanguage) as? String
        var decoding = translator.defaultDecoding
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber {
            decoding.maxTokens = max(1, value.intValue)
        }
        if let value = request.parameter(forKey: NFKMLXTranslationParameterKey.beamCount) as? NSNumber {
            decoding.beams = max(1, value.intValue)
        }
        if let value = request.parameter(forKey: NFKMLXTranslationParameterKey.lengthPenalty) as? NSNumber {
            decoding.lengthPenalty = value.floatValue
        }
        let splitsSentences = (request.parameter(forKey: NFKMLXTranslationParameterKey.splitsSentences) as? NSNumber)?.boolValue ?? false
        let translated = try translate(text, from: source, to: target, decoding: decoding, splitsSentences: splitsSentences)
        return NFKInferenceResult(outputs: [NFKOutputText: translated])
    }

    /// Translates `text`, resolving the languages against the model and splitting paragraphs.
    public func translate(_ text: String, from source: String?, to target: String,
                          decoding: NFKMLXSeq2SeqDecoding? = nil, splitsSentences: Bool = false) throws -> String {
        let (resolvedSource, resolvedTarget) = try resolveLanguages(text: text, source: source, target: target)
        let decoding = decoding ?? translator.defaultDecoding
        var output = ""
        for segment in Self.segments(of: text, splitsSentences: splitsSentences) {
            switch segment {
            case .separator(let literal):
                output += literal
            case .text(let piece):
                output += try translator.translate(piece, from: resolvedSource, to: resolvedTarget, decoding: decoding)
            }
        }
        return output
    }

    private func resolveLanguages(text: String, source: String?, target: String) throws -> (String?, String) {
        let target = Self.canonical(target)
        if let fixed = translator.fixedTargetLanguage, Self.primary(fixed) != Self.primary(target) {
            throw Self.error(.error_InferenceUnsupported, "\(translator.identifier) translates into \(fixed), not \(target)")
        }
        guard translator.supports(language: target) else {
            throw Self.error(.error_InferenceUnsupported, "\(translator.identifier) does not translate into \(target)")
        }
        var source = source.map(Self.canonical)
        if let fixed = translator.fixedSourceLanguage {
            if let source, Self.primary(fixed) != Self.primary(source) {
                throw Self.error(.error_InferenceUnsupported, "\(translator.identifier) translates from \(fixed), not \(source)")
            }
            return (fixed, target)
        }
        if source == nil {
            source = Self.detectLanguage(of: text)
        }
        if let source {
            guard translator.supports(language: source) else {
                throw Self.error(.error_InferenceUnsupported, "\(translator.identifier) does not translate from \(source)")
            }
        }
        return (source, target)
    }

    /// The dominant language of `text` as a BCP-47 tag, or nil when unrecognizable.
    ///
    /// The recognizer loads its assets on first use and has blocked a test process for good under
    /// load, so the call runs off the caller's thread with a deadline; reaching it answers nil, the
    /// same answer as text no language matches.
    public static func detectLanguage(of text: String) -> String? {
        let done = DispatchSemaphore(value: 0)
        var detected: String?
        DispatchQueue.global(qos: .userInitiated).async {
            let recognizer = NLLanguageRecognizer()
            recognizer.processString(text)
            detected = recognizer.dominantLanguage?.rawValue
            done.signal()
        }
        guard done.wait(timeout: .now() + languageDetectionDeadline) == .success else { return nil }
        return detected
    }

    /// How long `detectLanguage(of:)` waits for the recognizer before answering nil.
    public static var languageDetectionDeadline: TimeInterval = 10

    /// A BCP-47 tag with its primary subtag lowercased and a script or region kept as written.
    public static func canonical(_ tag: String) -> String {
        let parts = tag.replacingOccurrences(of: "_", with: "-").split(separator: "-").map(String.init)
        guard let first = parts.first else { return tag }
        return ([first.lowercased()] + parts.dropFirst()).joined(separator: "-")
    }

    /// The primary language subtag of a BCP-47 tag.
    public static func primary(_ tag: String) -> String {
        canonical(tag).split(separator: "-").first.map(String.init) ?? tag
    }

    /// The script subtag of a BCP-47 tag (`Hant`, `Latn`), or nil.
    public static func script(_ tag: String) -> String? {
        canonical(tag).split(separator: "-").dropFirst().map(String.init).first { $0.count == 4 }
    }

    enum Segment {
        case text(String)
        case separator(String)
    }

    /// Paragraphs (line-break-separated runs) as translatable text with their separators kept, each
    /// paragraph optionally split further into sentences.
    static func segments(of text: String, splitsSentences: Bool) -> [Segment] {
        var result = [Segment]()
        var current = ""
        func flush() {
            guard !current.isEmpty else { return }
            let leading = current.prefix { $0 == " " || $0 == "\t" }
            let trailing = current.reversed().prefix { $0 == " " || $0 == "\t" }
            let body = String(current.dropFirst(leading.count).dropLast(trailing.count))
            if !leading.isEmpty { result.append(.separator(String(leading))) }
            if !body.isEmpty {
                if splitsSentences {
                    result.append(contentsOf: sentences(of: body))
                } else {
                    result.append(.text(body))
                }
            }
            if !trailing.isEmpty { result.append(.separator(String(trailing.reversed()))) }
            current = ""
        }
        for character in text {
            if character.isNewline {
                flush()
                result.append(.separator(String(character)))
            } else {
                current.append(character)
            }
        }
        flush()
        return result
    }

    private static func sentences(of paragraph: String) -> [Segment] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = paragraph
        var result = [Segment]()
        tokenizer.enumerateTokens(in: paragraph.startIndex ..< paragraph.endIndex) { range, _ in
            let sentence = paragraph[range]
            let trimmed = sentence.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { result.append(.text(trimmed)) }
            let trailing = sentence.reversed().prefix { $0 == " " || $0 == "\t" }
            if !trailing.isEmpty { result.append(.separator(String(trailing.reversed()))) }
            return true
        }
        return result
    }

    static func error(_ code: NFKInferenceError, _ message: String) -> NSError {
        NSError(domain: NFKInferenceErrorDomain, code: code.rawValue,
                userInfo: [NSLocalizedDescriptionKey: message])
    }
}

/// Language-code tables the translators share.
enum NFKMLXLanguageCodes {
    /// ISO 639-1 → ISO 639-3, for the models that name languages by three-letter codes.
    static let iso639_3: [String: String] = [
        "af": "afr", "am": "amh", "ar": "ara", "az": "aze", "be": "bel", "bg": "bul", "bn": "ben", "bs": "bos",
        "ca": "cat", "cs": "ces", "cy": "cym", "da": "dan", "de": "deu", "el": "ell", "en": "eng", "eo": "epo",
        "es": "spa", "et": "est", "eu": "eus", "fa": "fas", "fi": "fin", "fr": "fra", "ga": "gle", "gl": "glg",
        "gu": "guj", "he": "heb", "hi": "hin", "hr": "hrv", "ht": "hat", "hu": "hun", "hy": "hye", "id": "ind",
        "is": "isl", "it": "ita", "ja": "jpn", "jv": "jav", "ka": "kat", "kk": "kaz", "km": "khm", "kn": "kan",
        "ko": "kor", "ku": "kur", "ky": "kir", "la": "lat", "lb": "ltz", "lo": "lao", "lt": "lit", "lv": "lav",
        "mg": "mlg", "mk": "mkd", "ml": "mal", "mn": "mon", "mr": "mar", "ms": "msa", "mt": "mlt", "my": "mya",
        "nb": "nob", "ne": "nep", "nl": "nld", "nn": "nno", "no": "nor", "oc": "oci", "pa": "pan", "pl": "pol",
        "ps": "pus", "pt": "por", "ro": "ron", "ru": "rus", "sd": "snd", "si": "sin", "sk": "slk", "sl": "slv",
        "so": "som", "sq": "sqi", "sr": "srp", "sv": "swe", "sw": "swa", "ta": "tam", "te": "tel", "th": "tha",
        "tl": "tgl", "tr": "tur", "uk": "ukr", "ur": "urd", "uz": "uzb", "vi": "vie", "xh": "xho", "yi": "yid",
        "yo": "yor", "zh": "zho", "zu": "zul",
    ]
}
