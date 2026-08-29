# ``NFKMLXMusicBackend``

## Overview

MiniMax Music 3 behind the InferKit contract. A music description under `NFKInputPrompt` and lyrics
under `NFKInputLyrics` become a stereo 44.1 kHz `NFKAudioAsset` under `NFKOutputAudio`. Build it with
``NFKMLXMusic3/backend(directoryURL:)`` from a downloaded release directory.

```swift
import InferKit
import InferKitMLX

let backend = try NFKMLXMusic3.backend(directoryURL: releaseDirectory)
let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "driving synthwave, night highway",
                                           NFKInputLyrics: "[chorus]\nwe run until the dawn"])
let song = try backend.runInference(for: request).outputForKey(NFKOutputAudio)
```

### Honored parameters

- `NFKParameterDurationSeconds` — an upper bound; the model may stop earlier, and the value is capped
  at six minutes.
- `NFKParameterSeed` — repeatable generation.
- `NFKParameterSteps` — flow-matching steps per window.
- `NFKParameterGuidanceScale` — the DiT's, default 1.7.

The generation is multi-second and GPU-scale, so a caller runs it off the render thread and prefers
`submitInferenceJob(for:)` for progress and cancellation. Cancellation is honored between stages and
per flow step.

### Residency

Whether the stages stay loaded between runs is decided from the weights, not assumed. When the stack's
weight bytes plus a reserve for activations and the CFG pair's key-value cache fit the machine's
working set — which a release quantized by
``NFKMLXMusic3/quantizeRelease(at:to:bits:transformerBits:groupSize:)`` does — every stage loads once and later runs
skip straight to generation. Otherwise the stages load from disk per run and each is freed when its
part is done. The full-precision language model (16 GiB at bf16) and float32 DiT (9.7 GB) together
exceed a 32 GB machine's working set, so staging is what makes the model runnable there at all.

`isHoldingStagesResident` reports whether the backend currently holds its stages, which a host reads
when deciding whether to construct a second heavy backend.

Build this backend with ``NFKMLXMusic3/backend(directoryURL:)``.

## Topics

### Related

- <doc:ModelGallery>
