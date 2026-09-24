//
//  NFKMLXM2M100.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX

// M2M-100, Meta's many-to-many translator over 100 languages (MIT weights): a pre-normalized
// encoder-decoder over a 128k SentencePiece BPE vocabulary, with a `__xx__` language token leading the
// source and forced as the decoder's first token. SMaLL-100 keeps the network and vocabulary and moves
// the target marker onto the source side.

/// The released M2M-100 sizes and the SMaLL-100 distillation.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXM2M100Variant)
public enum NFKMLXM2M100Variant: Int {
    /// `facebook/m2m100_418M`: 12 + 12 layers at 1024.
    case m418M
    /// `facebook/m2m100_1.2B`: 24 + 24 layers at 1024.
    case m1_2B
    /// `alirezamsh/small100`: the 418M geometry with a 12-layer encoder and 3-layer decoder, the
    /// target marker on the source.
    case small100

    var repo: String {
        switch self {
        case .m418M: return "facebook/m2m100_418M"
        case .m1_2B: return "facebook/m2m100_1.2B"
        case .small100: return "alirezamsh/small100"
        }
    }
}

/// A loaded M2M-100 release.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXM2M100Translator: NFKMLXTranslator {
    public let net: NFKMLXSeq2SeqNet
    public let tokenizer: NFKMLXSentencePieceTokenizer
    /// The `__xx__` marker ids by fairseq code.
    public let languageIds: [String: Int]
    /// SMaLL-100 marks the target on the source and starts the decoder plain.
    public let targetOnSource: Bool
    public let identifier: String
    public var defaultDecoding: NFKMLXSeq2SeqDecoding

    /// The 100 fairseq language codes, in the order that numbers their markers after the vocabulary.
    public static let languages = [
        "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de",
        "el", "en", "es", "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr",
        "ht", "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lg",
        "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns", "oc", "or",
        "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw",
        "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu",
    ]

    init(net: NFKMLXSeq2SeqNet, tokenizer: NFKMLXSentencePieceTokenizer, vocabularyCount: Int,
         targetOnSource: Bool, identifier: String, beams: Int, maxTokens: Int) {
        self.net = net
        self.tokenizer = tokenizer
        self.targetOnSource = targetOnSource
        self.identifier = identifier
        languageIds = Dictionary(uniqueKeysWithValues: Self.languages.enumerated().map { ($1, vocabularyCount + $0) })
        let c = net.configuration
        defaultDecoding = NFKMLXSeq2SeqDecoding(beams: beams, maxTokens: maxTokens, earlyStopping: true,
                                                startToken: c.decoderStartTokenId, endToken: c.eosTokenId)
    }

    public var fixedSourceLanguage: String? { nil }
    public var fixedTargetLanguage: String? { nil }

    /// The fairseq code for a BCP-47 tag, or nil when the model has no such language.
    public func code(for language: String) -> String? {
        let primary = NFKMLXTranslationBackend.primary(language)
        let aliases = ["nb": "no", "nn": "no", "fil": "tl", "iw": "he", "jw": "jv"]
        let code = aliases[primary] ?? primary
        return languageIds[code] != nil ? code : nil
    }

    public func supports(language: String) -> Bool { code(for: language) != nil }

    /// The source ids: the language marker, the pieces, and the end token.
    public func sourceIds(for text: String, source: String, target: String) -> [Int] {
        let marker = targetOnSource ? target : source
        var ids = [languageIds[code(for: marker) ?? marker] ?? net.configuration.eosTokenId]
        ids += tokenizer.encode(text, dummyPrefix: nil)
        ids.append(net.configuration.eosTokenId)
        return ids
    }

    public func translate(_ text: String, from source: String?, to target: String,
                          decoding: NFKMLXSeq2SeqDecoding) throws -> String {
        guard let source, let sourceCode = code(for: source) else {
            throw NFKMLXTranslationBackend.error(.error_InferenceMissingInput,
                                                 "\(identifier) needs the source language, which was neither given nor detected")
        }
        guard let targetCode = code(for: target) else {
            throw NFKMLXTranslationBackend.error(.error_InferenceUnsupported, "\(identifier) does not translate into \(target)")
        }
        var decoding = decoding
        if !targetOnSource {
            decoding.forcedFirstToken = languageIds[targetCode]
        }
        let ids = sourceIds(for: text, source: sourceCode, target: targetCode)
        let generated = NFKMLXSeq2SeqDecoder.generate(net, source: ids, decoding: decoding)
        return tokenizer.decode(ids: generated)
    }
}

/// Registration, download, and construction of M2M-100 translation backends.
///
/// @discussion A release directory holds `config.json`, `vocab.json`, `sentencepiece.bpe.model`, and
/// `pytorch_model.bin` or `model.safetensors`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXM2M100)
public final class NFKMLXM2M100: NSObject {
    /// The registered name.
    @objc public static let modelName = "m2m100"
    static let requiredFiles = ["config.json", "vocab.json", "sentencepiece.bpe.model"]
    static let optionalFiles = ["generation_config.json", "tokenizer_config.json"]
    static let weightFiles = ["model.safetensors", "pytorch_model.bin"]

    /// Loads a release directory as a translator.
    public static func translator(directoryURL directory: URL, variant: NFKMLXM2M100Variant = .m418M) throws -> NFKMLXM2M100Translator {
        let configuration = try NFKMLXSeq2SeqConfiguration(huggingFaceConfigURL: directory.appendingPathComponent("config.json"))
        let net = try network(directoryURL: directory, configuration: configuration)
        return try translator(net: net, directoryURL: directory, variant: variant)
    }

    /// Wraps a network with the release's tokenizer.
    public static func translator(net: NFKMLXSeq2SeqNet, directoryURL directory: URL,
                                  variant: NFKMLXM2M100Variant = .m418M) throws -> NFKMLXM2M100Translator {
        let vocabularyURL = directory.appendingPathComponent("vocab.json")
        let tokenizer = try NFKMLXSentencePieceTokenizer(modelURL: directory.appendingPathComponent("sentencepiece.bpe.model"),
                                                         vocabularyURL: vocabularyURL, eosTokenId: net.configuration.eosTokenId,
                                                         bosTokenId: 0)
        // The markers follow the vocabulary table, then 8 fairseq "madeupword" fillers close the
        // configured vocab_size. The base comes from that size rather than from counting the table:
        // Swift keys a dictionary by canonical equivalence, so the table's NFD and NFC spellings of one
        // piece collapse into one entry and the count runs short.
        let vocabularyCount = net.configuration.vocabularySize - NFKMLXM2M100Translator.languages.count - 8
        let generation = (try? JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("generation_config.json")))) as? [String: Any]
        let config = (try? JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("config.json")))) as? [String: Any]
        let beams = (generation?["num_beams"] as? NSNumber)?.intValue ?? (config?["num_beams"] as? NSNumber)?.intValue ?? 5
        let maxTokens = (generation?["max_length"] as? NSNumber)?.intValue ?? (config?["max_length"] as? NSNumber)?.intValue ?? 200
        let name: String
        switch variant {
        case .m418M: name = "m2m100-418m"
        case .m1_2B: name = "m2m100-1.2b"
        case .small100: name = "small100"
        }
        return NFKMLXM2M100Translator(net: net, tokenizer: tokenizer, vocabularyCount: vocabularyCount,
                                      targetOnSource: variant == .small100, identifier: name, beams: beams, maxTokens: maxTokens)
    }

    /// Builds the network alone, ready to adapt.
    public static func network(directoryURL directory: URL?,
                               configuration: NFKMLXSeq2SeqConfiguration = .tinyM2M100) throws -> NFKMLXSeq2SeqNet {
        let net = NFKMLXSeq2SeqNet(configuration)
        if let directory {
            try net.loadWeights(fromDirectory: directory)
        }
        return net
    }

    /// Builds the 418M backend from a release directory.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(variant: .m418M, directoryURL: directoryURL)
    }

    /// Builds a variant's backend from a release directory.
    @objc(backendWithVariant:directoryURL:error:)
    public static func backend(variant: NFKMLXM2M100Variant, directoryURL: URL) throws -> any NFKInferenceBackend {
        NFKMLXTranslationBackend(translator: try translator(directoryURL: directoryURL, variant: variant))
    }

    /// Downloads a variant's release and builds the backend. Blocking on the network.
    @objc(backendWithVariant:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXM2M100Variant, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: variant, directoryURL: try NFKMLXReleaseDownload.directory(
            repo: variant.repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``backend(variant:revision:cacheDirectoryURL:)``.
    @objc(backendWithVariant:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXM2M100Variant, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) { try backend(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL) }
    }

    /// Registers `m2m100` (the 418M release) and `small100` with `NFKMLXModelRegistry`; the registry's
    /// URL is the release directory.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { url in
            guard let url else { throw NFKMLXError.unsupportedConfiguration("m2m100 builds from a release directory, not without weights") }
            return try backend(variant: .m418M, directoryURL: url)
        }
        NFKMLXModelRegistry.register(name: "small100") { url in
            guard let url else { throw NFKMLXError.unsupportedConfiguration("small100 builds from a release directory, not without weights") }
            return try backend(variant: .small100, directoryURL: url)
        }
    }
}
