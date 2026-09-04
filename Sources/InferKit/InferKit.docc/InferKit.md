# ``InferKit``

A small, cross-platform inference toolkit for Objective-C: a swappable backend protocol, immutable
request and result value types, an async job handle, shipped zero-dependency backends, and a Hugging
Face model-download layer.

@Metadata {
    @DisplayName("InferKit")
}

## Overview

InferKit turns "run a model" into one uniform contract. A caller builds an ``NFKInferenceRequest``
(inputs plus parameters), hands it to any object conforming to ``NFKInferenceBackend``, and reads an
``NFKInferenceResult`` — or submits an ``NFKInferenceJob`` for progress, cancellation, and streaming.

![The inference contract: a request flows into a backend and returns a result, synchronously or through an async job.](contract-flow)

Backends are the extension seam. The core ships only backends with no third-party dependencies; a
heavier engine (Stable Diffusion, an MLX model, a C/Rust runtime) is brought by the consumer and either
adopts ``NFKInferenceBackend`` directly or is discovered at runtime through ``NFKDynamicBackend``.

![InferKit layers: a dependency-free core, optional companion packages, and the consumer's engine, joined by runtime discovery.](architecture)

- **No host-framework dependency** — any Metal/Apple app (macOS 11, iOS 14, tvOS 14) can use it.
- **Source-distributed** through Swift Package Manager and CocoaPods.
- **Objective-C**, class prefix `NFK`; the companions add Swift.

### A first run

```objc
NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a lighthouse at dusk" }];

id<NFKInferenceBackend> backend = /* a shipped, companion, or discovered backend */;
NSError *error = nil;
NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
NSString *text = result.text;   // or result.embedding / result.detections / outputForKey:
```

Inference is synchronous and often multi-second; a caller runs it off the main/render thread, and
prefers ``NFKInferenceJob`` for anything interactive. See <doc:TheInferenceContract>.

## Topics

### Essentials

- <doc:TheInferenceContract>
- ``NFKInferenceBackend``
- ``NFKInferenceRequest``
- ``NFKInferenceResult``
- ``NFKInferenceJob``

### Concepts

- <doc:Architecture>
- <doc:Backends>
- <doc:DynamicDiscovery>
- <doc:DownloadingModels>

### Shipped backends

- ``NFKPassthroughBackend``
- ``NFKCoreMLBackend``
- ``NFKCoreMLLanguageBackend``
- ``NFKRemoteBackend``
- ``NFKRemoteEmbeddingBackend``
- ``NFKRemoteSpeechBackend``
- ``NFKRemoteImageBackend``
- ``NFKRemoteTranscriptionBackend``
- ``NFKAsyncGenerationBackend``

### Remote providers

- ``NFKRemoteProvider``
- ``NFKRemoteModelCatalog``
- ``NFKRemoteModel``
- ``NFKRemoteTransport``

### Local runners

- ``NFKLocalModelRunner``
- ``NFKOllamaRunner``
- ``NFKLMStudioRunner``

### More remote services

- ``NFKRemoteVideoBackend``
- ``NFKRemoteModerationBackend``
- ``NFKRemoteReranker``

### Media coding

- ``NFKImageCoding``
- ``NFKVideoSampling``

### Dynamic discovery

- ``NFKDynamicBackend``
- ``NFKDynamicBackendProvider``

### Downloading models

- ``NFKHFHub``

### Detected & predicted value types

- ``NFKDetection``
- ``NFKKeypoint``
- ``NFKClassification``
- ``NFKAudioSegment``

### Media value types

- ``NFKVideoAsset``
- ``NFKAudioAsset``

### Tokenizers & tensors

- ``NFKTokenizer``

### Modality

- ``NFKModality``

### Errors

- ``NFKInferenceError``
