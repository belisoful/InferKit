//
//  NFKMLXTranslationProvider.swift
//  InferKitMLX
//

import Foundation
import InferKit

// The bridge that lets the core's `NFKDynamicBackend` activate an MLX translator for its `translation`
// capability ahead of Apple's. The core's default chain names only InferKitAppleSwift's provider; this
// one registers itself under the capability through `NFKMLXReferenceModels.registerAll()`, and a
// registered provider is tried before the defaults. It builds M2M-100 418M from the hub cache when
// that release is already downloaded, and declines otherwise, so a machine without the weights falls
// through to Apple's translator rather than failing the lookup.

/// Supplies the bundled M2M-100 translator to `NFKDynamicBackend` for `NFKCapabilityTranslation`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXTranslationProvider)
public final class NFKMLXTranslationProvider: NSObject, NFKDynamicBackendProvider {
    /// The release the provider serves.
    @objc public static let repo = "facebook/m2m100_418M"

    /// The M2M-100 418M backend when its release is in the default hub cache, else nil.
    public static func makeInferenceBackend() -> (any NFKInferenceBackend)? {
        let hub = NFKHFHub(cacheDirectoryURL: NFKHFHub.defaultCacheDirectoryURL())
        let cached = NFKMLXM2M100.requiredFiles.allSatisfy { hub.isCachedRepo(repo, revision: nil, path: $0) }
            && NFKMLXM2M100.weightFiles.contains { hub.isCachedRepo(repo, revision: nil, path: $0) }
        guard cached, let anchor = hub.localURL(forRepo: repo, revision: nil, path: "config.json") else {
            return nil
        }
        return try? NFKMLXM2M100.backend(variant: .m418M, directoryURL: anchor.deletingLastPathComponent())
    }

    /// Registers the provider ahead of the core's defaults for `NFKCapabilityTranslation`.
    @objc public static func register() {
        NFKDynamicBackend.registerProviderClassName("NFKMLXTranslationProvider", forCapability: NFKCapabilityTranslation)
    }
}
