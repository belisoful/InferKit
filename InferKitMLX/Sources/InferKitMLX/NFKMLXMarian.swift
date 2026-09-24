//
//  NFKMLXMarian.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX

// Marian NMT as the Helsinki-NLP OPUS-MT releases ship it: one small encoder-decoder per language
// pair (or per group of target languages), a source and a target SentencePiece model, and a shared
// `vocab.json` numbering both. A group model names its target with a `>>xxx<<` token at the front of
// the source, which is how one checkpoint serves several languages.

/// A loaded OPUS-MT release: the network, its two tokenizers, and its language pair.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXMarianTranslator: NFKMLXTranslator {
    public let net: NFKMLXSeq2SeqNet
    public let sourceTokenizer: NFKMLXSentencePieceTokenizer
    public let targetTokenizer: NFKMLXSentencePieceTokenizer
    /// The release's source language as its `tokenizer_config.json` names it.
    public let sourceLanguage: String
    /// The release's target language or target group (`de`, `mul`, `ROMANCE`).
    public let targetLanguage: String
    /// The `>>xxx<<` target markers of a group release, by the code inside the brackets.
    public let targetCodes: [String: Int]
    public let identifier: String
    public var defaultDecoding: NFKMLXSeq2SeqDecoding

    init(net: NFKMLXSeq2SeqNet, sourceTokenizer: NFKMLXSentencePieceTokenizer,
         targetTokenizer: NFKMLXSentencePieceTokenizer, sourceLanguage: String, targetLanguage: String,
         targetCodes: [String: Int], beams: Int) {
        self.net = net
        self.sourceTokenizer = sourceTokenizer
        self.targetTokenizer = targetTokenizer
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
        self.targetCodes = targetCodes
        identifier = "opus-mt-\(sourceLanguage)-\(targetLanguage)"
        let c = net.configuration
        defaultDecoding = NFKMLXSeq2SeqDecoding(beams: beams, maxTokens: 512, startToken: c.decoderStartTokenId,
                                                endToken: c.eosTokenId, suppressedTokens: [c.padTokenId],
                                                renormalizes: true)
    }

    /// Whether the release names one target language (no `>>xxx<<` markers).
    public var isPerPair: Bool { targetCodes.isEmpty }

    public var fixedSourceLanguage: String? { Self.isLanguageTag(sourceLanguage) ? sourceLanguage : nil }
    public var fixedTargetLanguage: String? { isPerPair && Self.isLanguageTag(targetLanguage) ? targetLanguage : nil }

    public func supports(language: String) -> Bool {
        let primary = NFKMLXTranslationBackend.primary(language)
        if primary == NFKMLXTranslationBackend.primary(sourceLanguage) { return true }
        if isPerPair { return primary == NFKMLXTranslationBackend.primary(targetLanguage) }
        return targetCode(for: language) != nil
    }

    /// The `>>xxx<<` marker id for a BCP-47 tag, or nil when the group has no such target.
    public func targetCode(for language: String) -> Int? {
        let primary = NFKMLXTranslationBackend.primary(language)
        let script = NFKMLXTranslationBackend.script(language)
        var candidates = [String]()
        if let three = NFKMLXLanguageCodes.iso639_3[primary] {
            if let script { candidates.append("\(three)_\(script)") }
            candidates.append(three)
        }
        if let script { candidates.append("\(primary)_\(script)") }
        candidates.append(primary)
        return candidates.lazy.compactMap { self.targetCodes[$0] }.first
    }

    /// The source ids the model reads for `text`: an optional target marker, the pieces, and the end token.
    public func sourceIds(for text: String, target: String?) -> [Int] {
        var ids = [Int]()
        if !isPerPair, let target, let code = targetCode(for: target) {
            ids.append(code)
        }
        ids += sourceTokenizer.encode(text, dummyPrefix: nil)
        ids.append(net.configuration.eosTokenId)
        return ids
    }

    public func translate(_ text: String, from source: String?, to target: String,
                          decoding: NFKMLXSeq2SeqDecoding) throws -> String {
        let generated = NFKMLXSeq2SeqDecoder.generate(net, source: sourceIds(for: text, target: target), decoding: decoding)
        return targetTokenizer.decode(ids: generated)
    }

    static func isLanguageTag(_ name: String) -> Bool {
        let primary = NFKMLXTranslationBackend.primary(name)
        return (2 ... 3).contains(primary.count) && primary == primary.lowercased()
    }
}

/// Registration, download, and construction of OPUS-MT translation backends.
///
/// @discussion A release directory holds `config.json`, `tokenizer_config.json`, `vocab.json`,
/// `source.spm`, `target.spm`, and `pytorch_model.bin` or `model.safetensors`, which is the layout of
/// every `Helsinki-NLP/opus-mt-*` repo. The pair factory names the repo from two language tags.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXMarian)
public final class NFKMLXMarian: NSObject {
    /// The registered name.
    @objc public static let modelName = "opus-mt"
    static let requiredFiles = ["config.json", "tokenizer_config.json", "vocab.json", "source.spm", "target.spm"]
    static let optionalFiles = ["generation_config.json"]
    static let weightFiles = ["model.safetensors", "pytorch_model.bin"]

    /// Loads a release directory as a translator.
    public static func translator(directoryURL directory: URL) throws -> NFKMLXMarianTranslator {
        let configuration = try NFKMLXSeq2SeqConfiguration(huggingFaceConfigURL: directory.appendingPathComponent("config.json"))
        let net = try network(directoryURL: directory, configuration: configuration)
        return try translator(net: net, directoryURL: directory)
    }

    /// Wraps a network (adapted or freshly loaded) with the release's tokenizers.
    public static func translator(net: NFKMLXSeq2SeqNet, directoryURL directory: URL) throws -> NFKMLXMarianTranslator {
        let vocabularyURL = directory.appendingPathComponent("vocab.json")
        let source = try NFKMLXSentencePieceTokenizer(modelURL: directory.appendingPathComponent("source.spm"),
                                                      vocabularyURL: vocabularyURL, eosTokenId: net.configuration.eosTokenId)
        let target = try NFKMLXSentencePieceTokenizer(modelURL: directory.appendingPathComponent("target.spm"),
                                                      vocabularyURL: vocabularyURL, eosTokenId: net.configuration.eosTokenId)
        let tokenizerConfig = (try? JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")))) as? [String: Any]
        let sourceLanguage = (tokenizerConfig?["source_lang"] as? String) ?? "und"
        let targetLanguage = (tokenizerConfig?["target_lang"] as? String) ?? "und"
        let vocabulary = (try JSONSerialization.jsonObject(with: Data(contentsOf: vocabularyURL))) as? [String: Int] ?? [:]
        var codes = [String: Int]()
        for (piece, id) in vocabulary where piece.hasPrefix(">>") && piece.hasSuffix("<<") && piece.count > 4 {
            codes[String(piece.dropFirst(2).dropLast(2))] = id
        }
        let generation = (try? JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("generation_config.json")))) as? [String: Any]
        let beams = (generation?["num_beams"] as? NSNumber)?.intValue ?? 4
        return NFKMLXMarianTranslator(net: net, sourceTokenizer: source, targetTokenizer: target,
                                      sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
                                      targetCodes: codes, beams: beams)
    }

    /// Builds the network alone, ready to adapt, from a release directory (weights loaded) or a
    /// configuration (random weights).
    public static func network(directoryURL directory: URL?,
                               configuration: NFKMLXSeq2SeqConfiguration = .tinyMarian) throws -> NFKMLXSeq2SeqNet {
        let net = NFKMLXSeq2SeqNet(configuration)
        if let directory {
            try net.loadWeights(fromDirectory: directory)
        }
        return net
    }

    /// Builds the backend from a release directory.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        NFKMLXTranslationBackend(translator: try translator(directoryURL: directoryURL))
    }

    /// Downloads a release (`Helsinki-NLP/opus-mt-en-de`) and builds the backend. Blocking on the
    /// network; call off the render thread.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) { try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL) }
    }

    /// The OPUS-MT repo for a language pair: `Helsinki-NLP/opus-mt-<source>-<target>`.
    @objc public static func repo(sourceLanguage: String, targetLanguage: String) -> String {
        "Helsinki-NLP/opus-mt-\(NFKMLXTranslationBackend.primary(sourceLanguage))-\(NFKMLXTranslationBackend.primary(targetLanguage))"
    }

    /// Downloads the pair's release and builds the backend.
    @objc(backendWithSourceLanguage:targetLanguage:cacheDirectoryURL:error:)
    public static func backend(sourceLanguage: String, targetLanguage: String, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(repo: repo(sourceLanguage: sourceLanguage, targetLanguage: targetLanguage),
                    revision: nil, cacheDirectoryURL: cacheDirectoryURL)
    }

    /// The asynchronous form of ``backend(sourceLanguage:targetLanguage:cacheDirectoryURL:)``.
    @objc(backendWithSourceLanguage:targetLanguage:cacheDirectoryURL:completionHandler:)
    public static func backend(sourceLanguage: String, targetLanguage: String, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(sourceLanguage: sourceLanguage, targetLanguage: targetLanguage, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Registers `opus-mt` with `NFKMLXModelRegistry`; the registry's URL is the release directory.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { url in
            guard let url else { throw NFKMLXError.unsupportedConfiguration("opus-mt builds from a release directory, not without weights") }
            return try backend(directoryURL: url)
        }
    }
}

/// Fetches the files of a Hugging Face release into the hub cache and returns their directory.
enum NFKMLXReleaseDownload {
    private static let overrideLock = NSLock()
    nonisolated(unsafe) private static var overrideHub: NFKHFHub?

    /// Runs `body` with every release download going through `hub`, which is how a test serves a
    /// release from a local store and refuses the network. Downloads take the hub's own cache folder.
    static func using<T>(_ hub: NFKHFHub, _ body: () throws -> T) rethrows -> T {
        overrideLock.lock()
        overrideHub = hub
        overrideLock.unlock()
        defer {
            overrideLock.lock()
            overrideHub = nil
            overrideLock.unlock()
        }
        return try body()
    }

    /// The hub a download goes through: the test override, or one over `cacheDirectoryURL`.
    static func hub(cacheDirectoryURL: URL?) -> NFKHFHub {
        overrideLock.lock()
        defer { overrideLock.unlock() }
        return overrideHub ?? NFKHFHub(cacheDirectoryURL: cacheDirectoryURL ?? NFKHFHub.defaultCacheDirectoryURL())
    }

    /// Fetches a release's required files, whichever optional files it serves, and the first weights
    /// entry it serves (with the shards a `*.index.json` entry names), and returns the snapshot root:
    /// `<cache>/<repo>/<revision>`, whatever folders the files sit in.
    ///
    /// A weights candidate the repo does not serve moves on to the next; when none is served, the error
    /// names what each candidate's download reported, so a refused credential reads as one.
    static func directory(repo: String, revision: String?, cacheDirectoryURL: URL?,
                          required: [String], optional: [String], weights: [String]) throws -> URL {
        let hub = hub(cacheDirectoryURL: cacheDirectoryURL)
        var root: URL?
        func note(_ url: URL, _ path: String) {
            if root == nil {
                root = path.split(separator: "/").reduce(url) { folder, _ in folder.deletingLastPathComponent() }
            }
        }
        for file in required {
            note(try hub.downloadRepo(repo, revision: revision, path: file, sha256: nil), file)
        }
        for file in optional {
            if let url = try? hub.downloadRepo(repo, revision: revision, path: file, sha256: nil) {
                note(url, file)
            }
        }
        var failures = [String]()
        var weightsFound = weights.isEmpty
        for file in weights {
            let url: URL
            do {
                url = hub.isCachedRepo(repo, revision: revision, path: file)
                    ? try cachedURL(hub.localURL(forRepo: repo, revision: revision, path: file), file)
                    : try hub.downloadRepo(repo, revision: revision, path: file, sha256: nil)
            } catch {
                failures.append("\(file): \(error.localizedDescription)")
                continue
            }
            note(url, file)
            // A shard index names the files that hold the weights, relative to its own folder.
            if file.hasSuffix(".index.json") {
                guard let data = try? Data(contentsOf: url),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let weightMap = json["weight_map"] as? [String: String] else {
                    failures.append("\(file): not a shard index")
                    continue
                }
                let folder = (file as NSString).deletingLastPathComponent
                for shard in Set(weightMap.values).sorted() {
                    _ = try hub.downloadRepo(repo, revision: revision,
                                             path: folder.isEmpty ? shard : "\(folder)/\(shard)", sha256: nil)
                }
            }
            weightsFound = true
            break
        }
        guard weightsFound else {
            throw NFKMLXError.unsupportedConfiguration("\(repo) serves none of its weights: \(failures.joined(separator: "; "))")
        }
        guard let root else { throw NFKMLXError.unsupportedConfiguration("\(repo): the release names no files") }
        return root
    }

    /// A cached file's local URL, which exists whenever the hub reports the file cached.
    private static func cachedURL(_ url: URL?, _ path: String) throws -> URL {
        guard let url else { throw NFKMLXError.unsupportedConfiguration("\(path) is cached but has no local URL") }
        return url
    }

    static func async(_ completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void,
                      _ build: @escaping () throws -> any NFKInferenceBackend) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try build(), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }
}
