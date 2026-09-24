//
//  NFKSpeechAnalyzerProvider.swift
//  InferKitAppleSwift
//

import Foundation
import InferKit

/// Activates `NFKSpeechAnalyzerBackend` through `NFKDynamicBackend` when this package is linked.
///
/// The core names this class for `NFKCapabilityTranscription` between `NFKMLXWhisperProvider` and its
/// own `NFKSpeechRecognitionProvider`: a consumer who linked InferKitMLX keeps Whisper, a consumer
/// who linked this package gets Apple's newer analyzer, and a consumer who linked neither gets the
/// core's `SFSpeechRecognizer` wrapper. Introduced in InferKit 0.4.0.
@objc(NFKSpeechAnalyzerProvider)
public final class NFKSpeechAnalyzerProvider: NSObject, NFKDynamicBackendProvider {

    public static func makeInferenceBackend() -> (any NFKInferenceBackend)? {
        NFKSpeechAnalyzerBackend()
    }
}
