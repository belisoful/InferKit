//
//  NFKMLXReleaseDownloadSupport.swift
//  InferKitMLXTests
//
//  Serves a release from the local validation store as if it had been downloaded, with the network
//  refused. Only the files a factory's lists name are placed in the cache, so a list that misses a
//  file the model reads fails the build rather than passing on a file that happened to be present.
//

import XCTest
import InferKit
@testable import InferKitMLX

/// A hub whose transport refuses every fetch, so a download succeeds only from what is cached.
final class NFKOfflineHub: NFKHFHub {
    private(set) var refused = [String]()

    override func fetch(_ remoteURL: URL, toFileURL destinationURL: URL) throws {
        refused.append(remoteURL.path)
        throw NSError(domain: NSURLErrorDomain, code: NSURLErrorNotConnectedToInternet,
                      userInfo: [NSLocalizedDescriptionKey: "offline: \(remoteURL.lastPathComponent) is not in the store"])
    }
}

enum NFKMLXReleaseStore {

    /// The files a download with these lists would place, read from the store: every required and
    /// optional file the store holds, the first weights entry it holds, and the shards that entry's
    /// index names.
    static func files(in store: URL, required: [String], optional: [String], weights: [String]) -> [String] {
        let exists = { (path: String) in FileManager.default.fileExists(atPath: store.appendingPathComponent(path).path) }
        var files = required + optional.filter(exists)
        if let first = weights.first(where: exists) {
            files.append(first)
            if first.hasSuffix(".index.json"),
               let data = try? Data(contentsOf: store.appendingPathComponent(first)),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let map = json["weight_map"] as? [String: String] {
                let prefix = (first as NSString).deletingLastPathComponent
                files += Set(map.values).sorted().map { prefix.isEmpty ? $0 : "\(prefix)/\($0)" }
            }
        }
        return files
    }

    /// Links `files` from `store` into `cache` at the layout the hub reads, `<repo>/<revision>/<path>`.
    static func seed(_ files: [String], from store: URL, repo: String, revision: String?, into cache: URL) throws {
        let snapshot = cache.appendingPathComponent(repo).appendingPathComponent(revision ?? "main")
        for path in files {
            let source = store.appendingPathComponent(path).resolvingSymlinksInPath()
            guard FileManager.default.fileExists(atPath: source.path) else {
                throw NSError(domain: "NFKMLXReleaseStore", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "the store has no \(path)"])
            }
            let link = snapshot.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source)
        }
    }

    /// Seeds a fresh cache from the store with the files the lists name, then runs `body` with every
    /// release download going through an offline hub over that cache. Returns what `body` returns.
    static func offline<T>(store: URL, repo: String, revision: String? = nil,
                           required: [String], optional: [String], weights: [String],
                           _ body: () throws -> T) throws -> T {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("release-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        try seed(files(in: store, required: required, optional: optional, weights: weights),
                 from: store, repo: repo, revision: revision, into: cache)
        let hub = NFKOfflineHub()
        hub.cacheDirectoryURL = cache
        return try NFKMLXReleaseDownload.using(hub, body)
    }
}
