# ``NFKMLXMusic3``

## Overview

MiniMax Music 3 — the text-and-lyrics-to-music model, ported stage by stage in `MLXNN` and validated
at measured reference parity against diffusers' own implementation. `NFKMLXMusic3` is the namespace
that builds ``NFKMLXMusicBackend`` from a downloaded release directory and prepares a quantized copy of
that release.

The full model is a hybrid: a Qwen3-8B autoregressive stage over eight RVQ codebooks, a condition
encoder blending the per-codebook hidden states, a 36-layer flow-matching DiT over 8-second latent
windows, and a DAC-style Snake vocoder. The stages are strictly sequential, so a backend loads and
frees them in turn unless the whole stack fits the working set.

### Building the backend

`directoryURL` is the release tree as `MiniMaxAI/MiniMax-Music3` publishes it (`language_model/`,
`rvq_depth_decoder/`, `condition_encoder/`, `transformer/`, `vocoder/`, and the byte-level tokenizer
files under `qwen_7B/qwen3-8B-tokenizer-music/`). `isReady` reports whether the language model is
present, so a consumer can construct first and download later.

```swift
import InferKit
import InferKitMLX

let backend = try NFKMLXMusic3.backend(directoryURL: releaseDirectory)
let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "warm lo-fi piano, rain",
                                           NFKInputLyrics: "[verse]\nsomething about the morning"])
let song = try backend.runInference(for: request).outputForKey(NFKOutputAudio)
```

### Quantizing a release

The full-precision stack is 27 GB. ``quantizeRelease(at:to:bits:transformerBits:groupSize:)`` writes a
quantized copy in the release's own layout, so ``backend(directoryURL:)`` takes the result unchanged.
The default split is measured, not assumed: the language model and depth decoder pack to 4-bit while
the DiT stays at 8-bit, because the flow field is the quantization-sensitive stage. The stack falls to
about 8.9 GiB, which fits the working set with room to stay resident.

```swift
try NFKMLXMusic3.quantizeRelease(at: releaseDirectory, to: quantizedDirectory)
let backend = try NFKMLXMusic3.backend(directoryURL: quantizedDirectory)
```

The tokenizer files and the LICENSE copy through: the license travels with the weights. The MiniMax
Music 3 weights are not permissively licensed — see <doc:ModelGallery> and `Docs/companions.md`.

## Topics

### Backend

- ``NFKMLXMusicBackend``

### Related

- <doc:ModelGallery>
- <doc:WeightsAndConversion>
