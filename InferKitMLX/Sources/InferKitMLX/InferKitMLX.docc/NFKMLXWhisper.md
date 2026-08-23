# ``NFKMLXWhisper``

## Overview

On-device speech-to-text — the Whisper encoder-decoder transformer in `MLXNN` (log-mel via `MLXFFT`
`rfft` → audio encoder → greedy text decoder). ``NFKMLXWhisperBackend`` reads `NFKInputAudio` (an
`NFKAudioAsset` WAV, or `NSData`) and returns `NFKOutputText`.

```swift
import InferKit
import InferKitMLX

let backend = try NFKMLXWhisper.backend(weightsURL: weightsURL)
let request = NFKInferenceRequest(inputs: [NFKInputAudio: audioAsset])
let text = try backend.runInference(for: request).text
```

An optional `NFKTokenizer` detokenizes to words; without one, the result carries token ids. Module names
follow OpenAI Whisper, so a checkpoint converted with `Tools/whisper-to-safetensors/convert.py` loads
with the conv transpose.

### Segment times

`emitsTimestamps` asks the decoder for the spans as well as the words. The result then carries
`NSArray<NFKAudioSegment *>` under `NFKOutputSegments`, each span labelled with the text inside it,
beside the whole transcript under `NFKOutputText`.

```swift
let backend = try NFKMLXWhisper.backend(weightsURL: weightsURL, tokenizer: tokenizer,
                                        timestamps: true)

for span in try backend.runInference(for: request).segments ?? [] {
    print("\(span.startSeconds)-\(span.endSeconds): \(span.label ?? "")")
}
```

This is a different decode rather than a different reading of one: the prompt drops
`<|notimestamps|>` and the timestamp range stays open, so the model may choose different words than
the plain path does. That is why it is off by default.

### Activates the core's transcription capability

Linking InferKitMLX ships ``NFKMLXWhisperProvider``, named the default the core tries for its
`transcription` capability. So transcription lights up through `NFKDynamicBackend` with no registration;
a consumer's native engine (whisper.cpp) registers to override it.

```swift
if NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityTranscription) {
    let asr = try NFKDynamicBackend.backend(forCapability: NFKCapabilityTranscription)
}
```

## Topics

### Backend & discovery

- ``NFKMLXWhisperBackend``
- ``NFKMLXWhisperProvider``

### Related

- <doc:ModelGallery>
