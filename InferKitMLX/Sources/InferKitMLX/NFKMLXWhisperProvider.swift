//
//  NFKMLXWhisperProvider.swift
//  InferKitMLX
//

import Foundation
import InferKit

// The bridge that lets InferKit's core activate the bundled Whisper speech-to-text backend without
// depending on InferKitMLX. The core's `NFKDynamicBackend` tries a provider class named exactly
// "NFKMLXWhisperProvider" for its built-in `transcription` capability; this class carries that name and
// conforms to `NFKDynamicBackendProvider`. When a consumer links InferKitMLX, the class is present and
// the core discovers it by name, so `NFKDynamicBackend.backendForCapability("transcription")` returns a
// working transcription backend. A consumer with a native engine (whisper.cpp) registers their own
// provider under the same capability to override this default.
//
// The backend is built with random weights (the pipeline proves out); a real transcription downloads a
// checkpoint. Construction is lazy and does not evaluate MLX.
@objc(NFKMLXWhisperProvider)
public final class NFKMLXWhisperProvider: NSObject, NFKDynamicBackendProvider {

    /// Builds the bundled Whisper (tiny) transcription backend.
    public static func makeInferenceBackend() -> (any NFKInferenceBackend)? {
        try? NFKMLXWhisper.backend(weightsURL: nil)
    }
}
