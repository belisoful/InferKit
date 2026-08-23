//
//  NFKStableDiffusionProvider.swift
//  InferKitMLX
//

import Foundation
import InferKit

// The bridge that lets InferKit's core activate the bundled Stable Diffusion backend without depending
// on InferKitMLX. The core's `NFKDynamicBackend` tries a provider class named exactly
// "NFKStableDiffusionProvider" for its built-in `stable-diffusion` capability; this class carries that
// name and conforms to `NFKDynamicBackendProvider`. When a consumer links InferKitMLX, the class is
// present and the core discovers it by name (`NSClassFromString`), so
// `NFKDynamicBackend.stableDiffusionBackend()` returns a working Stable Diffusion backend. When
// InferKitMLX is not linked, the class is absent and the capability is simply unavailable.
//
// The backend is constructed lazily: building `NFKMLXBackend` only stores the model choice; weights
// download and the pipeline initialize on first `prepare()` / inference, off the render thread.
@objc(NFKStableDiffusionProvider)
public final class NFKStableDiffusionProvider: NSObject, NFKDynamicBackendProvider {

    /// Builds the bundled Stable Diffusion backend (SD 1.5).
    ///
    /// Stable Diffusion 1.5 is the release a consumer gets without naming one: it is ungated, so the
    /// capability activates with no credential, and it is the smallest of the three. A consumer
    /// wanting another builds ``NFKMLXBackend`` directly with that model.
    public static func makeInferenceBackend() -> (any NFKInferenceBackend)? {
        NFKMLXBackend(model: .stableDiffusion15)
    }
}
