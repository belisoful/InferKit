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
    /// shards are fetched beside its index. A cached file is not fetched again.
    static func directory(repo: String, revision: String?, cacheDirectoryURL: URL?,
                          components: [NFKMLXReleaseComponent]) throws -> URL {
        let hub = hub(cacheDirectoryURL: cacheDirectoryURL)
        var root: URL?
        func fetch(_ path: String) throws -> URL {
            let url = try hub.downloadRepo(repo, revision: revision, path: path, sha256: nil)
            root = root ?? path.split(separator: "/").reduce(url) { folder, _ in folder.deletingLastPathComponent() }
            return url
        }
        for component in components {
            for file in component.required {
                _ = try fetch(file)
            }
            for file in component.optional {
                _ = try? fetch(file)
            }
            guard !component.weights.isEmpty else { continue }
            guard let (file, url) = component.weights.lazy
                .compactMap({ file in (try? fetch(file)).map { (file, $0) } }).first else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(repo) serves none of \(component.weights.joined(separator: ", "))")
            }
            guard file.hasSuffix(".index.json") else { continue }
            guard let data = try? Data(contentsOf: url),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weightMap = json["weight_map"] as? [String: String] else {
                throw NFKMLXError.unsupportedConfiguration("\(file) in \(repo) carries no weight_map")
            }
            let folder = (file as NSString).deletingLastPathComponent
            for shard in Set(weightMap.values).sorted() {
                _ = try fetch(folder.isEmpty ? shard : "\(folder)/\(shard)")
            }
        }
        guard let root else {
            throw NFKMLXError.unsupportedConfiguration("the download of \(repo) names no files")
        }
        return root
    }
}
