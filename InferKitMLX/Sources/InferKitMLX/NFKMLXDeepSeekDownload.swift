//
//  NFKMLXDeepSeekDownload.swift
//  InferKitMLX
//
//  The DeepSeek V4 / V4.1 download factories: each directory factory's repo and asynchronous peers.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXDeepSeek {

    /// The files a release's configuration and tokenizer come from.
    static let deepSeekRequiredFiles = ["config.json", "tokenizer.json"]
    /// The chat template's two spellings and the generation defaults, fetched where the repo serves them.
    static let deepSeekOptionalFiles = ["tokenizer_config.json", "chat_template.jinja", "generation_config.json"]
    /// The weights: the shard index and the shards it names, or a single file.
    static let deepSeekWeightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// Fetches a release into the hub cache and returns the directory the directory factories read.
    static func deepSeekReleaseDirectory(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> URL {
        try NFKMLXReleaseDownload.directory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                            required: deepSeekRequiredFiles, optional: deepSeekOptionalFiles,
                                            weights: deepSeekWeightFiles)
    }
}

public extension NFKMLXDeepSeek {

    /// Downloads a release and builds its text-generation backend, held as ``NFKMLXResidency/automatic``
    /// decides.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer.json`, the `tokenizer_config.json`,
    /// `chat_template.jinja`, and `generation_config.json` the repo serves, and the weights the shard
    /// index names. A file already in the cache is not fetched again. The releases are
    /// `deepseek-ai/DeepSeek-V4.1-Flash`, `deepseek-ai/DeepSeek-V4-Flash-0731`, and
    /// `deepseek-ai/DeepSeek-V4-Pro-0813`; V4.1 Flash stores 510 GB. The call blocks on the network, so
    /// run it off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(deepSeekBackendWithRepo:revision:cacheDirectoryURL:error:)
    static func deepSeekBackend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try deepSeekBackend(directoryURL: try deepSeekReleaseDirectory(repo: repo, revision: revision,
                                                                       cacheDirectoryURL: cacheDirectoryURL))
    }

    /// The asynchronous form of ``deepSeekBackend(repo:revision:cacheDirectoryURL:)``. The handler runs on
    /// a background queue. Introduced in InferKit 0.4.0.
    @objc(deepSeekBackendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    static func deepSeekBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try deepSeekBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// ``deepSeekBackend(repo:revision:cacheDirectoryURL:)`` holding the release as `residency` says.
    /// Introduced in InferKit 0.4.0.
    @objc(deepSeekBackendWithRepo:revision:cacheDirectoryURL:residency:error:)
    static func deepSeekBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try deepSeekBackend(directoryURL: try deepSeekReleaseDirectory(repo: repo, revision: revision,
                                                                       cacheDirectoryURL: cacheDirectoryURL),
                            residency: residency)
    }

    /// The asynchronous form of ``deepSeekBackend(repo:revision:cacheDirectoryURL:residency:)``. The
    /// handler runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(deepSeekBackendWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    static func deepSeekBackend(repo: String, revision: String?, cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                                completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try deepSeekBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                residency: residency)
        }
    }

    /// ``deepSeekBackend(repo:revision:cacheDirectoryURL:)`` loaded as `options` says: the residency or
    /// paging preset, speculation, activation rounding, and float32 compute. Introduced in InferKit 0.4.0.
    @objc(deepSeekBackendWithRepo:revision:cacheDirectoryURL:options:error:)
    static func deepSeekBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                options: NFKMLXDeepSeekLoadOptions) throws -> any NFKInferenceBackend {
        try deepSeekBackend(directoryURL: try deepSeekReleaseDirectory(repo: repo, revision: revision,
                                                                       cacheDirectoryURL: cacheDirectoryURL),
                            options: options)
    }

    /// The asynchronous form of ``deepSeekBackend(repo:revision:cacheDirectoryURL:options:)``. The handler
    /// runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(deepSeekBackendWithRepo:revision:cacheDirectoryURL:options:completionHandler:)
    static func deepSeekBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                options: NFKMLXDeepSeekLoadOptions,
                                completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try deepSeekBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, options: options)
        }
    }

    /// Downloads a release whose decoded weights exceed the machine and builds it paged, as
    /// ``deepSeekPagedBackend(directoryURL:expertCacheBytes:)`` does. Introduced in InferKit 0.4.0.
    @objc(deepSeekPagedBackendWithRepo:revision:cacheDirectoryURL:expertCacheBytes:error:)
    static func deepSeekPagedBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                     expertCacheBytes: Int) throws -> any NFKInferenceBackend {
        try deepSeekPagedBackend(directoryURL: try deepSeekReleaseDirectory(repo: repo, revision: revision,
                                                                            cacheDirectoryURL: cacheDirectoryURL),
                                 expertCacheBytes: expertCacheBytes)
    }

    /// The asynchronous form of ``deepSeekPagedBackend(repo:revision:cacheDirectoryURL:expertCacheBytes:)``.
    /// The handler runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(deepSeekPagedBackendWithRepo:revision:cacheDirectoryURL:expertCacheBytes:completionHandler:)
    static func deepSeekPagedBackend(repo: String, revision: String?, cacheDirectoryURL: URL?, expertCacheBytes: Int,
                                     completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try deepSeekPagedBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                     expertCacheBytes: expertCacheBytes)
        }
    }
}
