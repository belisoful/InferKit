# ``InferKitAppleSwift``

Apple's Swift-only inference APIs, behind the InferKit contract, reachable from Objective-C.

@Metadata {
    @DisplayName("InferKitAppleSwift")
}

## Overview

The core wraps the Apple frameworks an Objective-C target can call. Three APIs it cannot:
`SpeechAnalyzer` is an actor whose results arrive as an `AsyncSequence`, and Vision's
`RecognizeDocumentsRequest` and `DetectLensSmudgeRequest` ship in `Vision.swiftmodule` with no `VN*`
header. This companion hosts them, and every type it adds is `@objc`.

```swift
let read = try NFKVisionDocumentBackend()
    .runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: page]))
read.structured?["tables"]        // rows of cells
```

- **A document's structure** — ``NFKVisionDocumentBackend`` returns the transcript under
  `NFKOutputText` and paragraphs, lists, and tables under `NFKOutputStructured`.
- **A dirty lens** — ``NFKVisionSmudgeBackend`` returns one classification labeled `smudge`.
- **Transcription** — ``NFKSpeechAnalyzerBackend`` transcribes a recording on Apple's newer speech
  stack, with per-range segments.
- **Translation** — ``NFKTranslationBackend`` translates on device, reading
  `NFKParameterTargetLanguage` and returning `NFKOutputText`.
- **Discovery** — linking this package puts ``NFKSpeechAnalyzerProvider`` ahead of the core's
  recognizer for `NFKCapabilityTranscription`, and ``NFKTranslationProvider`` behind any MLX
  translator for `NFKCapabilityTranslation`.

The floor is macOS 26 / iOS 26, where all four APIs begin. The core's floor is unchanged.

### Readiness is what prepare found

The system answers availability and installed locales asynchronously, and `isReady` cannot wait, so
it reports what `prepare()` found. An unprepared backend reports NO even where the assets happen to
be installed.

### Every wait on a framework is bounded

The Translation framework does not always answer: asked about a language outside a SwiftUI
presentation, it returns neither a result nor a refusal. A synchronous contract turns that into a
blocked thread, so ``NFKTranslationBackend/responseTimeout`` bounds every wait and reaching it is
reported as `kNFKError_InferenceNotReady`.

## Topics

### Backends

- ``NFKVisionDocumentBackend``
- ``NFKVisionSmudgeBackend``
- ``NFKSpeechAnalyzerBackend``
- ``NFKTranslationBackend``

### Dynamic discovery

- ``NFKSpeechAnalyzerProvider``
- ``NFKTranslationProvider``
