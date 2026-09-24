# InferKitAppleSwift

Apple's inference APIs that ship only in Swift, wrapped as InferKit backends so an Objective-C app
can reach them.

The core wraps every Apple framework it can: Vision, VideoToolbox, Speech, and Core ML all have
backends there. Three APIs cannot live in the core, because the core is a pure Objective-C target and
these are Swift-only:

| API | Why it cannot live in the core |
|---|---|
| `SpeechAnalyzer`, `SpeechTranscriber` | an actor, whose results arrive as an `AsyncSequence` |
| `RecognizeDocumentsRequest` | ships in `Vision.swiftmodule` with no `VN*` header |
| `DetectLensSmudgeRequest` | the same |

This package is that host. It needs macOS 26 / iOS 26, which is where all three begin, and it does
not raise the core's floor.

## The backends

```swift
// A photographed page becomes a transcript and a structure.
let document = NFKVisionDocumentBackend()
let read = try document.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: page]))
read.text                                  // the whole transcript
read.structured?["tables"]                 // rows of cells, as strings

// Was the lens dirty?
let smudge = try NFKVisionSmudgeBackend()
    .runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: frame]))
smudge.classifications?.first?.confidence  // labeled "smudge"

// Transcription on Apple's newer stack.
let speech = NFKSpeechAnalyzerBackend(locale: Locale(identifier: "en-US"))
try speech.prepare()                       // reserves the locale and installs its assets
let words = try speech.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
words.text                                 // the transcription
words.segments                             // one entry per reported range
```

Every one of them is `@objc`, which is the point of the package. The Objective-C half of
`Docs/examples.md` shows the same three from there.

## Readiness is what prepare found

The system reports availability and installed locales asynchronously, and `isReady` cannot wait. It
answers from what `prepare()` found, so a backend that has not been prepared reports NO even where
the assets are already installed. Preparing costs little in that case: the install request is nil
when nothing needs installing.

## Discovery

Linking this package names `NFKSpeechAnalyzerProvider` for `NFKCapabilityTranscription`, between
`NFKMLXWhisperProvider` and the core's own `NFKSpeechRecognitionProvider`. A consumer who brought
Whisper keeps it; a consumer who linked this package gets the analyzer; a consumer who linked
neither gets the core's `SFSpeechRecognizer` wrapper.

## Build & test

```bash
cd InferKitAppleSwift
swift build
swift test    # the transcription tests state what they need and skip without it
```
