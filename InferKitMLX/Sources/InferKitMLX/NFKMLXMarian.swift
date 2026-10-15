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
        Self.markerCandidates(for: language).lazy.compactMap { self.targetCodes[$0] }.first
    }

    /// The OPUS codes a BCP-47 tag may be marked as, most specific first: the ISO 639-3 code with
    /// the tag's script, the ISO 639-3 code, the individual language of a macrolanguage with its
    /// script and alone (`zh` → `cmn_Hans`, `cmn`), then the primary subtag with its script and alone.
    /// Chinese without a script subtag takes the script its region implies (`zh-TW` → `cmn_Hant`).
    ///
    /// Introduced in InferKit 0.4.0.
    public static func markerCandidates(for language: String) -> [String] {
        let primary = NFKMLXTranslationBackend.primary(language)
        let script = NFKMLXTranslationBackend.impliedScript(language)
        var candidates = [String]()
        func add(_ code: String) {
            if let script { candidates.append("\(code)_\(script)") }
            candidates.append(code)
        }
        if let three = NFKMLXLanguageCodes.iso639_3[primary] { add(three) }
        if let individual = NFKMLXLanguageCodes.individualLanguage[primary] { add(individual) }
        add(primary)
        return candidates
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
        try translator(net: try network(directoryURL: directory), directoryURL: directory)
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

    /// Builds the network alone, ready to adapt. A release directory supplies its own geometry from its
    /// `config.json` and its weights. Without a directory the network takes `configuration` (the tiny
    /// test geometry when nil) and random weights. A `configuration` passed with a directory overrides
    /// the release's `config.json`.
    public static func network(directoryURL directory: URL?,
                               configuration: NFKMLXSeq2SeqConfiguration? = nil) throws -> NFKMLXSeq2SeqNet {
        guard let directory else {
            return NFKMLXSeq2SeqNet(configuration ?? .tinyMarian)
        }
        let net = NFKMLXSeq2SeqNet(try configuration ?? NFKMLXSeq2SeqConfiguration(
            huggingFaceConfigURL: directory.appendingPathComponent("config.json")))
        try net.loadWeights(fromDirectory: directory)
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
///
/// Every fetch reports to the core's current `NFKDownload` and stops when it is cancelled: the hub's
/// own downloads do, and this layer adds what only it knows. Under a current download, a release
/// with a file to fetch lists the repository once, so the files' sizes count toward the download's
/// expected bytes before they arrive; a shard index adds its shards when it is read. A completed
/// release is recorded in its snapshot, so `NFKHFHub.downloadStatus(repo:revision:paths: nil)`
/// reads it as complete and a removal of the snapshot knows its files.
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

    /// Throws the cancellation error when the current download is cancelled. A cancelled fetch of an
    /// optional file or a weights candidate reads as an absent file, so the loops check here.
    static func checkCancelled() throws {
        if NFKDownload.current?.isCancelled == true {
            throw NFKDownloadError(.cancelled)
        }
    }

    /// Whether a release's required files or every one of its weights entries is missing from the
    /// cache.
    static func needsFetch(hub: NFKHFHub, repo: String, revision: String?, required: [String],
                           weights: [String]) -> Bool {
        let cached = { (path: String) in hub.isCachedRepo(repo, revision: revision, path: path) }
        return !required.allSatisfy(cached)
            || (!weights.isEmpty && cachedWeights(hub: hub, repo: repo, revision: revision, candidates: weights) == nil)
    }

    /// The shard paths a shard index at `url` names, relative to the repository root (the index names
    /// them relative to its own folder); nil when the file is not a shard index.
    static func shards(ofIndexAt url: URL, path: String) -> [String]? {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = json["weight_map"] as? [String: String] else {
            return nil
        }
        let folder = (path as NSString).deletingLastPathComponent
        return Set(weightMap.values).sorted().map { folder.isEmpty ? $0 : "\(folder)/\($0)" }
    }

    /// The first weights candidate whole in the cache, in candidate order, with every path it holds: the
    /// file, and for a shard index every shard it names. nil when no candidate is whole in the cache.
    static func cachedWeights(hub: NFKHFHub, repo: String, revision: String?,
                              candidates: [String]) -> (file: String, paths: [String])? {
        for candidate in candidates where hub.isCachedRepo(repo, revision: revision, path: candidate) {
            guard candidate.hasSuffix(".index.json") else { return (candidate, [candidate]) }
            guard let index = hub.localURL(forRepo: repo, revision: revision, path: candidate),
                  let shards = shards(ofIndexAt: index, path: candidate),
                  shards.allSatisfy({ hub.isCachedRepo(repo, revision: revision, path: $0) }) else {
                continue
            }
            return (candidate, [candidate] + shards)
        }
        return nil
    }

    /// Whether a fetch failed because the server lacks the file (HTTP 404), which a completed release
    /// records so a cached build does not ask again. A network failure is not an absence.
    static func isAbsent(_ error: Error) -> Bool {
        let error = error as NSError
        return error.domain == NFKDownloadErrorDomain && error.code == NFKDownloadError.notFound.rawValue
    }

    /// Fetches the optional files the release serves and returns the ones the server refused. A file the
    /// snapshot's completion record lists as refused, and that is not cached, is not asked for again.
    static func fetchOptional(_ optional: [String], hub: NFKHFHub, repo: String, revision: String?,
                              note: (URL, String) -> Void) throws -> [String] {
        let refused = Set(hub.recordedAbsentPaths(ofRepo: repo, revision: revision))
        var absent = [String]()
        for file in optional {
            if refused.contains(file) && !hub.isCachedRepo(repo, revision: revision, path: file) {
                absent.append(file)
                continue
            }
            do {
                note(try hub.downloadRepo(repo, revision: revision, path: file, sha256: nil), file)
            } catch {
                try checkCancelled()
                if isAbsent(error) {
                    absent.append(file)
                }
            }
        }
        return absent
    }

    /// Fetches a release's weights: the first candidate whole in the cache, with no request; otherwise
    /// the first candidate the repository serves, in order, with the shards a shard index names. Throws
    /// when no candidate is served, naming what each candidate's download reported.
    static func fetchWeights(_ candidates: [String], hub: NFKHFHub, repo: String, revision: String?,
                             listing: NFKDownloadPlan?, note: (URL, String) -> Void) throws {
        guard !candidates.isEmpty else { return }
        if let cached = cachedWeights(hub: hub, repo: repo, revision: revision, candidates: candidates) {
            for path in cached.paths {
                note(try hub.downloadRepo(repo, revision: revision, path: path, sha256: nil), path)
            }
            return
        }
        var failures = [String]()
        for file in candidates {
            let url: URL
            do {
                url = try hub.downloadRepo(repo, revision: revision, path: file, sha256: nil)
            } catch {
                try checkCancelled()
                failures.append("\(file): \(error.localizedDescription)")
                continue
            }
            note(url, file)
            if file.hasSuffix(".index.json") {
                guard let shards = shards(ofIndexAt: url, path: file) else {
                    failures.append("\(file): not a shard index")
                    continue
                }
                expect(shards, from: listing)
                for shard in shards {
                    note(try hub.downloadRepo(repo, revision: revision, path: shard, sha256: nil), shard)
                }
            }
            return
        }
        throw NFKMLXError.unsupportedConfiguration("\(repo) serves none of its weights: \(failures.joined(separator: "; "))")
    }

    /// The first of `candidates` (single files) whole in the cache, else the first the repository
    /// serves. Throws when none is served.
    static func firstServed(_ candidates: [String], hub: NFKHFHub, repo: String, revision: String?) throws -> URL {
        var served: URL?
        try fetchWeights(candidates, hub: hub, repo: repo, revision: revision, listing: nil) { url, _ in
            served = served ?? url
        }
        guard let served else {
            throw NFKMLXError.unsupportedConfiguration("\(repo) serves none of \(candidates.joined(separator: ", "))")
        }
        return served
    }

    /// The repository's listing, fetched once for a current download when the release has a file to
    /// fetch; nil without a current download, with nothing to fetch, or when the listing fails (the
    /// fetches then report each file's size as it arrives).
    static func listing(hub: NFKHFHub, repo: String, revision: String?, needsFetch: Bool) -> NFKDownloadPlan? {
        guard NFKDownload.current != nil, needsFetch else { return nil }
        return try? hub.planDownload(ofRepo: repo, revision: revision, paths: nil)
    }

    /// Declares the listed files among `paths` to the current download.
    static func expect(_ paths: [String], from listing: NFKDownloadPlan?) {
        guard let listing, let download = NFKDownload.current else { return }
        download.add(listing.planRestricted(toPaths: paths))
    }

    /// The release's files that the listing holds: the required, the optional it serves, and the
    /// first weights entry it serves.
    static func expectedPaths(in listing: NFKDownloadPlan?, required: [String], optional: [String],
                              weights: [String]) -> [String] {
        guard let listing else { return [] }
        let served = { (path: String) in listing.file(atPath: path) != nil }
        return required + optional.filter(served) + (weights.first(where: served).map { [$0] } ?? [])
    }

    /// Fetches a release's required files, whichever optional files it serves, and the first weights
    /// entry it serves (with the shards a `*.index.json` entry names), and returns the snapshot root:
    /// `<cache>/<repo>/<revision>`, whatever folders the files sit in.
    ///
    /// A weights candidate the repo does not serve moves on to the next; when none is served, the error
    /// names what each candidate's download reported, so a refused credential reads as one. A cached
    /// release makes no request: a weights candidate whole in the cache wins over earlier candidates,
    /// and an optional file the completion record lists as refused is not asked for again.
    static func directory(repo: String, revision: String?, cacheDirectoryURL: URL?,
                          required: [String], optional: [String], weights: [String]) throws -> URL {
        let hub = hub(cacheDirectoryURL: cacheDirectoryURL)
        let repoListing = listing(hub: hub, repo: repo, revision: revision,
                              needsFetch: needsFetch(hub: hub, repo: repo, revision: revision,
                                                     required: required, weights: weights))
        expect(expectedPaths(in: repoListing, required: required, optional: optional, weights: weights), from: repoListing)
        var root: URL?
        var fetched = [String]()
        func note(_ url: URL, _ path: String) {
            fetched.append(path)
            if root == nil {
                root = path.split(separator: "/").reduce(url) { folder, _ in folder.deletingLastPathComponent() }
            }
        }
        for file in required {
            note(try hub.downloadRepo(repo, revision: revision, path: file, sha256: nil), file)
        }
        let absent = try fetchOptional(optional, hub: hub, repo: repo, revision: revision, note: note)
        try fetchWeights(weights, hub: hub, repo: repo, revision: revision, listing: repoListing, note: note)
        guard let root else { throw NFKMLXError.unsupportedConfiguration("\(repo): the release names no files") }
        try? hub.recordCompleteDownload(ofRepo: repo, revision: revision, paths: fetched, absentPaths: absent)
        return root
    }

    /// Runs `build`, a release download and the build over it, through ``asyncValue(_:_:)`` and delivers the
    /// backend or the error to `completionHandler`. The backend can unload and load again through `build`
    /// (`NFKMLXReload.reloadable(_:)`), which finds the release in the cache.
    static func async(_ completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void,
                      _ build: @escaping () throws -> any NFKInferenceBackend) {
        asyncValue(completionHandler) { try NFKMLXReload.reloadable(build) }
    }

    /// Runs `build` on a background queue and delivers its result to `completionHandler` there. The
    /// download current on the calling thread is carried to the queue and made current around
    /// `build`, with one activity spanning it, so the download observes the fetches and finishes when
    /// `build` returns.
    static func asyncValue<T>(_ completionHandler: @escaping (T?, Error?) -> Void, _ build: @escaping () throws -> T) {
        let download = NFKDownload.current
        download?.beginActivity()
        DispatchQueue.global(qos: .userInitiated).async {
            download?.becomeCurrent()
            let result = Result { try build() }
            download?.resignCurrent()
            switch result {
            case .success(let value):
                download?.endActivity(error: nil)
                completionHandler(value, nil)
            case .failure(let error):
                download?.endActivity(error: error)
                completionHandler(nil, error)
            }
        }
    }
}
