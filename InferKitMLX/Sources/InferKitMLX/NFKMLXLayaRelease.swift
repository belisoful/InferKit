//
//  NFKMLXLayaRelease.swift
//  InferKitMLX
//
//  Downloading a Laya release. The Hugging Face repository `convaiinnovations/laya` holds three
//  variants, each a folder of five files; these factories fetch one variant through the core's
//  `NFKHFHub` cache and build the model from the folder it lands in.
//

import Foundation
import InferKit

/// Which of the three Laya releases a model loads.
@objc(NFKMLXLayaVariant)
public enum NFKMLXLayaVariant: Int, Sendable, CaseIterable {
    /// The repository root: ModernBERT-large, English, the general decision model.
    case root = 0
    /// `typed-decisions/`: the root's geometry fine-tuned on the typed-decisions benchmark.
    case typedDecisions = 1
    /// `multilingual/`: mmBERT-base with Gemma's tokenizer, 100-plus languages.
    case multilingual = 2

    /// The variant's folder inside the repository, empty for the root.
    public var folder: String {
        switch self {
        case .root: return ""
        case .typedDecisions: return "typed-decisions"
        case .multilingual: return "multilingual"
        }
    }
}

extension NFKMLXLaya {

    /// The Hugging Face repository the releases are published in.
    @objc public static let repository = "convaiinnovations/laya"

    /// The repository commit the reference-parity measurements were taken at. Passing it as the
    /// revision pins a download to the measured weights; nil follows `main`.
    @objc public static let measuredRevision = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"

    /// The files a variant needs, relative to the repository root.
    @objc(releaseFilesForVariant:)
    public static func releaseFiles(for variant: NFKMLXLayaVariant) -> [String] {
        let names = ["rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer_config.json",
                     "tokenizer/tokenizer.json", "model.safetensors"]
        return variant.folder.isEmpty ? names : names.map { "\(variant.folder)/\($0)" }
    }

    /// Downloads a variant into the hub cache and returns its folder, ready for
    /// ``laya(directoryURL:)``. A cached file is not fetched again.
    ///
    /// The cache is `cacheDirectoryURL`, or `NFKHFHub.defaultCacheDirectoryURL()` when nil, laid out
    /// as `<cache>/convaiinnovations/laya/<revision>/<folder>`. Blocks on the network; run it off the
    /// render thread, or use the completion-handler form.
    @objc(downloadVariant:revision:cacheDirectoryURL:error:)
    public static func download(variant: NFKMLXLayaVariant, revision: String?,
                                cacheDirectoryURL: URL?) throws -> URL {
        try download(variant: variant, revision: revision,
                     hub: NFKHFHub(cacheDirectoryURL: cacheDirectoryURL ?? NFKHFHub.defaultCacheDirectoryURL()))
    }

    /// The download over a given hub, which is the seam a test replaces the transport through.
    static func download(variant: NFKMLXLayaVariant, revision: String?, hub: NFKHFHub) throws -> URL {
        var folder: URL?
        for path in releaseFiles(for: variant) {
            let local = try hub.downloadRepo(repository, revision: revision, path: path, sha256: nil)
            if path.hasSuffix("rl_agent_config.json") {
                folder = local.deletingLastPathComponent()
            }
        }
        guard let folder else { throw NFKMLXError.weightsMismatch("the release names no configuration file") }
        return folder
    }

    /// The asynchronous form of ``download(variant:revision:cacheDirectoryURL:)``. The handler runs on
    /// a background queue with the variant's folder or the error.
    @objc(downloadVariant:revision:cacheDirectoryURL:completionHandler:)
    public static func download(variant: NFKMLXLayaVariant, revision: String?, cacheDirectoryURL: URL?,
                                completionHandler: @escaping (URL?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try download(variant: variant, revision: revision,
                                               cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// Downloads a variant (or reads it from the cache) and builds the model from it.
    @objc(layaWithVariant:revision:cacheDirectoryURL:error:)
    public static func laya(variant: NFKMLXLayaVariant, revision: String?,
                            cacheDirectoryURL: URL?) throws -> NFKMLXLaya {
        try laya(directoryURL: download(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL))
    }

    /// The asynchronous form of ``laya(variant:revision:cacheDirectoryURL:)``: downloads and builds on
    /// a background queue and hands the model or the error to the handler there.
    @objc(layaWithVariant:revision:cacheDirectoryURL:completionHandler:)
    public static func laya(variant: NFKMLXLayaVariant, revision: String?, cacheDirectoryURL: URL?,
                            completionHandler: @escaping (NFKMLXLaya?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try laya(variant: variant, revision: revision,
                                           cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// Downloads a variant (or reads it from the cache) and builds the inference backend.
    @objc(backendWithVariant:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXLayaVariant, revision: String?,
                               cacheDirectoryURL: URL?) throws -> NFKMLXLayaBackend {
        try laya(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL).makeBackend()
    }

    /// The asynchronous form of ``backend(variant:revision:cacheDirectoryURL:)``.
    @objc(backendWithVariant:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXLayaVariant, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXLayaBackend?, Error?) -> Void) {
        laya(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL) { laya, error in
            completionHandler(laya?.makeBackend(), error)
        }
    }
}
