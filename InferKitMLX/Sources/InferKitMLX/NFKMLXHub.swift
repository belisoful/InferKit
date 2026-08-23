//
//  NFKMLXHub.swift
//  InferKitMLX
//

import Foundation
import InferKit

/// Combines the core's Hugging Face download layer (`NFKHFHub`) with the MLX model registry
/// (`NFKMLXModelRegistry`): it downloads a model's weights from Hugging Face and builds the
/// registered MLX backend around them, in one call.
///
/// `NFKHFHub` lives in the dependency-free core and has no MLX; the registry is in this companion.
/// This helper is where they meet, so an Objective-C consumer gets a running MLX model from a
/// Hugging Face repo without writing Swift:
///
/// ```objc
/// // A Swift model was registered under "corridor-key" (its factory loads the weights URL).
/// NSError *error = nil;
/// id<NFKInferenceBackend> keyer =
///     [NFKMLXHub backendNamed:@"corridor-key"
///                        repo:@"org/corridor-key"
///                 weightsPath:@"model.safetensors"
///                    revision:nil
///           cacheDirectoryURL:nil
///                       error:&error];
/// ```
///
/// The `…error:` factory blocks on the network, so call it off the render thread; the `…completionHandler:`
/// peer runs the download on a background queue and delivers the backend (or an error) to the handler.
@objc(NFKMLXHub)
public final class NFKMLXHub: NSObject {

    /// Downloads `weightsPath` from the Hugging Face repo `repo` (caching under `cacheDirectoryURL`,
    /// or the default cache when nil) and builds the MLX backend registered under `modelName`, passing
    /// the local weights URL to its factory. Fails fast when no model is registered under the name, so
    /// a misconfiguration does not first spend a download.
    @objc(backendNamed:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(named modelName: String,
                               repo: String,
                               weightsPath: String,
                               revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        guard NFKMLXModelRegistry.isModelRegistered(modelName) else {
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceNotReady.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "no MLX model is registered under '\(modelName)'"])
        }
        let weightsURL = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath,
                                                       revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try NFKMLXModelRegistry.backend(named: modelName, weightsURL: weightsURL)
    }

    /// The asynchronous form of ``backend(named:repo:weightsPath:revision:cacheDirectoryURL:)``: it runs
    /// the download on a background queue and delivers the built backend (or an error) to
    /// `completionHandler`, so the caller does not hand-thread the blocking download off the render
    /// thread. The handler runs on the download's background queue; a caller that needs the main thread
    /// hops there itself.
    @objc(backendNamed:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(named modelName: String,
                               repo: String,
                               weightsPath: String,
                               revision: String?,
                               cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        guard NFKMLXModelRegistry.isModelRegistered(modelName) else {
            completionHandler(nil, NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceNotReady.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "no MLX model is registered under '\(modelName)'"]))
            return
        }
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try NFKMLXModelRegistry.backend(named: modelName, weightsURL: $0) },
                               completionHandler: completionHandler)
    }
}

/// Shared "download then build" plumbing over the core `NFKHFHub`, used by `NFKMLXHub` and by the
/// per-model `backendWithRepo:…` / `backendWithVariant:repo:…` factories so the download logic lives in
/// one place.
enum NFKMLXDownload {
    /// Downloads `weightsPath` from the Hugging Face `repo` (caching under `cacheDirectoryURL`, or the
    /// default cache when nil) and returns the local file URL. Blocking on the network; call off the
    /// render thread.
    static func weightsURL(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> URL {
        let hub = NFKHFHub(cacheDirectoryURL: cacheDirectoryURL ?? NFKHFHub.defaultCacheDirectoryURL())
        return try hub.downloadRepo(repo, revision: revision, path: weightsPath, sha256: nil)
    }

    /// Asynchronous "download then build": fetches `weightsPath` on a background queue over the core's
    /// async `NFKHFHub` download, then calls `build` with the local URL and delivers the backend (or the
    /// error) to `completionHandler` on that queue. Shared by `NFKMLXHub` and the per-model
    /// `backendWithRepo:…completionHandler:` / `backendWithVariant:repo:…completionHandler:` factories so
    /// the async plumbing lives in one place.
    static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                        build: @escaping (URL) throws -> any NFKInferenceBackend,
                        completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        let hub = NFKHFHub(cacheDirectoryURL: cacheDirectoryURL ?? NFKHFHub.defaultCacheDirectoryURL())
        hub.downloadRepo(repo, revision: revision, path: weightsPath, sha256: nil) { url, error in
            if let error {
                completionHandler(nil, error)
                return
            }
            guard let url else {
                completionHandler(nil, NSError(domain: NFKInferenceErrorDomain,
                    code: NFKInferenceError.error_InferenceNotReady.rawValue,
                    userInfo: [NSLocalizedDescriptionKey: "the download returned no file URL"]))
                return
            }
            do {
                completionHandler(try build(url), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }
}
