<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Apple-framework backends in the core

The engines that wrap Apple's own inference frameworks: Vision, VideoToolbox, and Speech. They ship
in the core because they depend on Apple frameworks only, which is the core's rule. They are
alternatives to the MLX models they overlap, never replacements: `Docs/agent-reference/xcode27-foundation-models-and-neural-accelerators.md`
records the overlap survey that produced them.

These are fixed system models with no weights and no training path, so the MLX parity checklist and
the "customization is part of parity" rule do not apply to them. They carry no entry in
`Docs/model-index.md` or `Docs/model-parity.md`.

## The Swift host (InferKitAppleSwift)

Three Apple APIs cannot live in the core, which is a pure Objective-C target: `SpeechAnalyzer` (an
actor whose results are an `AsyncSequence`) and Vision's `RecognizeDocumentsRequest` and
`DetectLensSmudgeRequest` (in `Vision.swiftmodule`, no `VN*` header). `InferKitAppleSwift/` is the
host, at macOS 26 / iOS 26, and every type it adds is `@objc`.

- **The analyzer's readiness is not a question a property can answer.** `SpeechTranscriber.isAvailable`,
  `.supportedLocales`, and `.installedLocales` are all `async`, so `isReady` reports what `prepare()`
  found rather than probing. The locale lists are exposed as completion-handler class methods, which
  is also their Objective-C shape. Preparing twice is cheap: `assetInstallationRequest` is nil when
  nothing needs installing.
- The Swift Vision API performs with `try await request.perform(on: cgImage)`, so each backend hops a
  detached task and waits on a semaphore, the same shape `NFKFoundationModelsBackend` uses for its
  own asynchronous reads.
- A smudge verdict is a classification labeled `smudge` rather than a new output key: a confidence is
  exactly what the request returns.

## Discovery

`NFKAppleProviders.h` declares one provider per engine, and `NFKDynamicBackend` names them last for
their capabilities (2026-09-22). The order is deliberate: a companion's model wins where the consumer
linked one, and the core answers where nothing is linked. The VideoToolbox providers return nil when
their processor is absent, which discovery treats as a pass rather than a failure. The segmentation
provider hands back the subject mask where the OS has it and the saliency mask below that, so the
capability resolves at the core's floor.

## Geometry: the flip that nothing in the type system catches

`NFKDetection.boundingBox` and `NFKKeypoint.position` are normalized 0...1 with the origin at the
**top left**. Vision normalizes with the origin at the **lower left**. Every box and every point
crossing the boundary is flipped in `NFKVisionSupport`: `contractRect:` and `contractPoint:`. A
missing flip produces plausible-looking output, so `testARecognizedLineIsBoxedInTheContractsGeometry`
reads text drawn near the top and near the bottom of one picture and checks both, which a missing
flip and a doubled flip each fail.

A face landmark is normalized inside the face's own box, not the image, so `NFKVisionFaceBackend`
maps each point through the box before flipping it.

## Vision

- One class per capability, which is the core's convention and keeps `supportedInputKeys` meaningful:
  `NFKVisionTextBackend`, `NFKVisionSegmentationBackend`, `NFKVisionPoseBackend`,
  `NFKVisionFaceBackend`, `NFKVisionFeaturePrintBackend`. Shared plumbing is the private
  `NFKVisionSupport`, beside `NFKRemoteMediaSupport`.
- Floors, checked in the headers: text recognition, saliency, and feature print are macOS 10.15;
  face landmarks 10.13; body and hand pose 11.0. All clear the core's floor. The two that do not are
  person segmentation (macOS 12, iOS 15, tvOS 15) and the subject mask (macOS 14, iOS 17, tvOS 17);
  `isReady` is NO below them and a run reports `kNFKError_InferenceUnsupported`.
- A subject mask comes from the observation and the handler that produced it
  (`generateScaledMaskForImageForInstances:fromRequestHandler:error:`), so the run keeps the handler
  alive. Person and saliency masks carry their buffer on the observation.
- Vision hands joints back in an unordered dictionary, so the pose backend sorts by joint name for a
  stable `index`. Several subjects concatenate, and `index` restarting at zero is where the next one
  begins.
- Four corners are no longer a reason to leave a request out: `NFKQuadrilateral` (2026-09-22) holds
  them and `NFKDetection.quadrilateral` carries one, so barcodes, rectangles, and document
  segmentation ship. A barcode's payload is the detection's label; the symbology is an input, chosen
  through the backend's `symbologies`, because one label cannot hold both.
- The readings that are numbers rather than geometry go under `NFKOutputStructured`, which is what
  made them shippable: aesthetics (`overallScore`, `isUtility`), the horizon (`angleRadians`,
  `transform`), contours, and registration. Their keys are the observation's own names.
- `NFKVisionTrackingBackend` is the one backend in the toolkit that holds state, because Vision's
  sequence requests need a handler that lives across frames. `startTrackingBoundingBox:` names the
  region, each run advances it, and `reset` ends the sequence. Trajectory detection needs real
  timestamps, so the backend builds `CMSampleBuffer`s with synthetic ones at `framesPerSecond`.
- Registration answers a sparse or repeating image confidently and wrongly: a periodic test pattern
  produced a confident reading with the wrong sign. A test that measures alignment gives Vision a
  texture that does not repeat and keeps the shift on canvas. The measured truth for the API's
  direction: perform on the reference image and target the moved frame.
- `NFKVisionCoreMLBackend` dispatches by observation type, so one class covers whatever the model
  emits. It is the alternative to `NFKCoreMLBackend`, not a replacement: Vision resizes and crops the
  image the model's description asks for, where `NFKCoreMLBackend` takes tensors the caller built.
- Left out on purpose: 3D body pose, because `NFKKeypoint.position` is a `CGPoint` and a third axis
  would change the value type for every engine.
  `RecognizeDocumentsRequest` and `DetectLensSmudgeRequest` ship only in Vision.swiftmodule with no
  `VN*` header, so they need a Swift host, as `SpeechAnalyzer` does.

## VideoToolbox

Measured on an M1 Max running macOS 26.6.2 with a probe program, because none of this is in the
headers:

- `VTSuperResolutionScalerConfiguration.supportedScaleFactors` is **[4]** on this machine. A
  configuration built with any other factor returns nil and logs "unsupported scaleFactor". The
  backend's `scaleFactor` therefore defaults to 0, meaning the smallest factor the machine offers,
  and an unsupported explicit factor is refused with the supported list in the message.
- Pixel formats are not negotiable and are not BGRA. Interpolation reads and writes `420v`
  (`kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange`); optical flow reads `RGhA`
  (`kCVPixelFormatType_64RGBAHalf`). A BGRA frame handed straight to a processor fails with
  `VTFrameProcessorProcessingError` (-19740), which names nothing. Every buffer crossing the boundary
  goes through `VTPixelTransferSession` into the shape the configuration publishes, and the result is
  transferred back to BGRA.
- `kCVPixelBufferPixelFormatTypeKey` in a configuration's attributes may hold an **array** of formats
  rather than one number. Reading it as an `NSNumber` throws `unrecognized selector`.
- The flow field is smaller than the frame, in `kCVPixelFormatType_TwoComponent16Half`. Allocating
  it at frame size and reading it at frame size is a segmentation fault. Destination buffers are
  allocated from the configuration's `destinationPixelBufferAttributes`, which carry the size as
  well as the format, and that is the reliable rule. One measurement, 80 by 60 for a 320 by 240
  input, is a quarter per axis; whether that ratio holds at other frame sizes or on other silicon is
  unverified, so do not compute the size from it.
- Interpolation rejects two frames that share a presentation timestamp, so the pair is stamped 0 and
  600 in a 600 timescale with the interpolated frame at 300.
- The packed flow map follows `NFKMLXRAFT`: red and green carry `0.5 + component / (2 * flowScale)`
  with `flowScale` 32 by default, blue is zero. A consumer decodes with
  `(value - 0.5) * 2 * flowScale` and reads either engine the same way. Changing the default is a
  silent numeric change for every consumer, not a compile error.
- Availability is per processor and per platform: upscaling and flow are unavailable on tvOS
  entirely, interpolation exists there at tvOS 26. `VTFrameProcessor` itself is macOS 15.4 / iOS 26 /
  tvOS 26. The implementation compiles the unavailable paths out with `TARGET_OS_TV` and checks the
  rest with `@available`.
- Upscaling runs a model the system downloads once. `configurationModelStatus` reports it, and
  `prepare` starts the download and fails with `kNFKError_InferenceNotReady` while it runs, so a
  pending model is a backend that is not ready rather than a request that fails.

## Speech

- `SFSpeechRecognizer` is Objective-C and fits the core; `SpeechAnalyzer` and its modules
  (macOS 26) are a Swift actor and cannot live in the pure Objective-C core target. Hosting them
  needs a Swift package, which is a separate decision.
- Recognition needs the user's consent and an `NSSpeechRecognitionUsageDescription` in the app's
  Info.plist. A test bundle has neither, so the tests cover the contract and the refusals:
  `isReady` is NO and a run reports `kNFKError_InferenceNotReady`.
- The recognizer reads a file, so audio held as `NSData` is written to a temporary one.
  `requiresOnDeviceRecognition` defaults to YES here, which is not Apple's default; the toolkit's
  other on-device engines keep the audio on the machine and this one matches them.
- Unavailable on tvOS. The class is still there and reports `kNFKError_InferenceUnsupported`, so the
  API surface does not change shape per platform.

## Speech synthesis

`AVSpeechSynthesizer` writes samples through `writeUtterance:toBufferCallback:`. Three measured
conditions each produce the same nameless failure: the callback is never called, no error is
reported, and a blocking wait reaches its deadline. All three were found by probe, not by reading
the documentation, and each one alone is enough to yield silence.

- **The buffers arrive on the main run loop, and only there.** A secondary thread running its own
  run loop receives nothing, with or without a Mach port attached to keep the loop alive (measured
  on macOS 26.6.2: 83 callbacks on the main run loop, 0 on a secondary one). So the utterance starts
  on the main thread either way. A caller already on the main thread pumps the loop itself, which
  lets that thread's own timers and sources run during the write. A caller off the main thread, which
  is what the synchronous contract asks for, dispatches to the main queue and waits, which needs the
  program's main run loop to be running. A program that blocks its main thread instead reaches the
  deadline.
- **The synthesizer has to outlive the call that issues the write.** It holds no reference to itself
  for the duration. Built as a local inside the block that starts the utterance, ARC releases it on
  return and delivery stops.
- **An utterance with no voice is speakable but not writable.** `speakUtterance:` with
  `utterance.voice` nil plays through the system default. The same utterance through
  `writeUtterance:` produces no buffers. The backend resolves a voice for every request:
  `voiceWithIdentifier:`, then `voiceWithLanguage:`, then
  `voiceWithLanguage:AVSpeechSynthesisVoice.currentLanguageCode`, then the first of `speechVoices`.
  A named voice or language that does not resolve fails with `kNFKError_InferenceUnsupported`
  instead of writing an empty file.

`NFKAppleEngineTests` covers this live: the three conditions together took the suite from 12 tests
with 8 failures in 361 seconds, all of them deadlines, to 12 green in 2.1 seconds.

## Translation (InferKitAppleSwift)

`NFKTranslationBackend` wraps the Translation framework, which is Swift-only and therefore lives in
the companion. Two measured facts shape it.

- **The framework answers nothing in a test bundle.** `LanguageAvailability.status(from:to:)` and
  `supportedLanguages` both return neither a value nor an error, and `TranslationSession` behaves the
  same way. The framework is built around a SwiftUI presentation, and outside one it stays silent.
  The synchronous contract turns an async call into a blocking wait, so the wait needs a deadline:
  without one the calling thread is gone for the life of the process. `responseTimeout` is public,
  defaults to 60 seconds, and reaching it reports `kNFKError_InferenceNotReady`.
- **An unknown language code never returns either.** Asked about a code it does not know, the
  framework does not report `unsupported`; it reports nothing. The backend checks both codes against
  the published `supportedLanguages` before it puts a pair to the system, so an unknown code is
  refused with `kNFKError_InferenceUnsupported` rather than waiting.

The three outcomes stay distinct: a pair the system will not translate is `unsupported`, a pair whose
model is not installed is `notReady`, and silence is also `notReady`. The tests reflect what this
environment can prove: the contract and the refusals run everywhere, and a test that needs the
framework to answer skips when it does not.

`NFKDynamicBackend` names `NFKMLXTranslationProvider` ahead of `NFKTranslationProvider` for
`NFKCapabilityTranslation`, so an MLX translator answers where one is linked and Apple's framework
answers otherwise.

## SoundAnalysis and NaturalLanguage

- `NFKSoundClassificationBackend` runs `SNClassifySoundRequest` over Apple's built-in taxonomy. Each
  analysis window becomes an `NFKAudioSegment` under `NFKOutputSegments`, and the clip's own best
  guesses arrive under `NFKOutputClassifications`. `minimumConfidence` is 0.3 by default, which a
  short clip of a single sound does not always clear; the examples set it lower and say why.
- `NFKTextEmbeddingBackend` reads `NLEmbedding.sentenceEmbeddingForLanguage:`, and only that. The
  language is the one set on the backend, or English when none is set, and the text's own language is
  detected where the caller asks for it. `+availableLanguages` probes a candidate list rather than
  naming what the OS ought to have, because which languages carry a sentence embedding is the
  installed system's answer.
