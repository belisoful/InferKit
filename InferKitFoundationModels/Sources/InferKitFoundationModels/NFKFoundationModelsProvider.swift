//
//  NFKFoundationModelsProvider.swift
//  InferKitFoundationModels
//

import Foundation
import InferKit

// The bridge that lets InferKit's core activate the on-device system language model without depending
// on InferKitFoundationModels. The core's `NFKDynamicBackend` tries a provider class named exactly
// "NFKFoundationModelsProvider" for its built-in `text-generation` capability; this class carries that
// name and conforms to `NFKDynamicBackendProvider`. When a consumer links InferKitFoundationModels (on
// macOS 26 / iOS 26 with Apple Intelligence available), the class is present and the core discovers it
// by name, so `NFKDynamicBackend.backendForCapability("text-generation")` returns a working LLM
// backend. When the package is not linked, the class is absent and the capability is unavailable.
//
// This mirrors `NFKStableDiffusionProvider` in InferKitMLX: link the companion, get the capability.
@objc(NFKFoundationModelsProvider)
public final class NFKFoundationModelsProvider: NSObject, NFKDynamicBackendProvider {

    /// Builds the on-device Foundation Models text-generation backend.
    public static func makeInferenceBackend() -> (any NFKInferenceBackend)? {
        NFKFoundationModelsBackend()
    }
}
