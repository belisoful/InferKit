//
//  NFKMLXReleaseComponents.swift
//  InferKitMLX
//
//  The download of a release laid out as component folders (`transformer/`, `vae/`, `text_encoder/`),
//  each fetched into the same snapshot.
//

import Foundation
import InferKit

/// The files one component folder of a release contributes to a download. Paths are relative to the
/// repo root and name the folder (`transformer/config.json`).
struct NFKMLXReleaseComponent {
    /// Files the download fails without.
    let required: [String]
    /// Files fetched when the repo serves them.
    let optional: [String]
    /// The entries the component's weights may be stored as, in preference order. The first the repo
    /// serves is fetched; empty for a component without weights.
    let weights: [String]

    init(required: [String], optional: [String] = [], weights: [String] = []) {
        self.required = required
        self.optional = optional
        self.weights = weights
    }
}

extension NFKMLXReleaseDownload {
    /// Fetches every component of a release into one snapshot and returns the snapshot root.
    ///
    /// @discussion A shard index names its shards relative to its own folder, so a component's
    /// shards are fetched beside its index. A cached release makes no request, as
    /// `directory(repo:revision:cacheDirectoryURL:required:optional:weights:)` describes. Under a current
    /// `NFKDownload` the files report to it as `directory(repo:revision:cacheDirectoryURL:required:optional:weights:)`
    /// describes, and the completed release is recorded in its snapshot.
    static func directory(repo: String, revision: String?, cacheDirectoryURL: URL?,
                          components: [NFKMLXReleaseComponent]) throws -> URL {
        let hub = hub(cacheDirectoryURL: cacheDirectoryURL)
        let missing = components.contains {
            needsFetch(hub: hub, repo: repo, revision: revision, required: $0.required, weights: $0.weights)
        }
        let repoListing = listing(hub: hub, repo: repo, revision: revision, needsFetch: missing)
        expect(components.flatMap { expectedPaths(in: repoListing, required: $0.required, optional: $0.optional,
                                                  weights: $0.weights) }, from: repoListing)
        var root: URL?
        var fetched = [String]()
        func note(_ url: URL, _ path: String) {
            fetched.append(path)
            root = root ?? path.split(separator: "/").reduce(url) { folder, _ in folder.deletingLastPathComponent() }
        }
        var absent = [String]()
        for component in components {
            for file in component.required {
                note(try hub.fetchFile(repo: repo, revision: revision, path: file), file)
            }
            absent += try fetchOptional(component.optional, hub: hub, repo: repo, revision: revision, note: note)
            try fetchWeights(component.weights, hub: hub, repo: repo, revision: revision, listing: repoListing, note: note)
        }
        guard let root else {
            throw NFKMLXError.unsupportedConfiguration("the download of \(repo) names no files")
        }
        recordComplete(hub: hub, repo: repo, revision: revision, paths: fetched, absent: absent)
        return root
    }
}
