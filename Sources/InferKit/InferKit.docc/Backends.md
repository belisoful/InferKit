# Backends

One protocol, many engines — shipped, companion, and consumer-brought.

## Overview

A backend is any object conforming to ``NFKInferenceBackend``: `isReady`, a synchronous
`runInferenceForRequest:error:`, and an asynchronous `submitInferenceJobForRequest:`. The protocol is
the seam where an engine plugs in.

![One protocol, with shipped zero-dependency backends, companion engines, and consumer-brought engines.](backend-taxonomy)

### Shipped, zero-dependency

Pure Apple frameworks, always available:

- ``NFKPassthroughBackend`` — returns its inputs, so an effect renders unchanged with no model present;
  keeps builds and tests green without weights.
- ``NFKCoreMLBackend`` — in-process Core ML with image and tensor I/O.
- ``NFKCoreMLLanguageBackend`` — a Core ML causal-LM runner.
- ``NFKRemoteBackend`` and ``NFKAnthropicBackend`` — the chat clients: one OpenAI-compatible, one for
  Anthropic's Messages API, which differs in its headers, its required `max_tokens`, and its top-level
  system prompt. ``NFKRemoteProvider`` names the services
  it is pointed at (hosted APIs and the local runners Ollama, LM Studio, llama.cpp, and vLLM), and
  ``NFKRemoteModelCatalog`` lists the models a provider serves so a caller chooses one from the
  server's own list. A local runner's native API — what is installed and loaded, and Ollama's
  download and delete — is reached through ``NFKLocalModelRunner``. Both chat clients stream through
  the job form, cancel the request when the job is cancelled, take tools and a JSON Schema, and retry
  a rate limit through ``NFKRemoteTransport``.
- ``NFKRemoteEmbeddingBackend`` — an OpenAI-compatible embeddings client; the vector comes back under
  the same key the on-device embedders use.
- ``NFKRemoteSpeechBackend`` — an OpenAI-compatible text-to-speech client, answering with an
  `NFKAudioAsset` the way the on-device speech backend does.
- ``NFKRemoteImageBackend`` — OpenAI-compatible image generation, edits, and inpainting, chosen from the
  request the way the Stable Diffusion backend chooses. An image beside a prompt through the chat
  backends is a vision question; ``NFKImageCoding`` is the codec under all of it. Audio, PDFs, and a
  clip's sampled frames (``NFKVideoSampling``) ride beside the prompt the same way, and a chat model can
  speak its reply.
- ``NFKRemoteVideoBackend`` — video generation as a job (OpenAI's videos API), the first shipped
  ``NFKAsyncGenerationBackend``.
- ``NFKRemoteModerationBackend`` — per-category moderation scores and a verdict for text or an image.
- ``NFKRemoteReranker`` — query-and-documents relevance scores, the shape of the on-device reranker.
- ``NFKRemoteTranscriptionBackend`` — an OpenAI-compatible audio-to-text client.
- ``NFKAsyncGenerationBackend`` — a subclassable submit → poll → fetch base for job-style generation
  services. Map the service's JSON through its template methods; the base owns the loop and the
  ``NFKInferenceJob``.

### Companion (link to enable)

Bringing a heavier runtime is a matter of linking a package: InferKitMLX (60-plus MLX models across
image, video, audio, and language, plus the bundled Stable Diffusion `NFKMLXBackend`) and
InferKitFoundationModels (Apple's on-device LLM).

### Brought by you

Adopt ``NFKInferenceBackend`` directly, or expose an engine through ``NFKDynamicBackendProvider`` so
the core discovers it by name at runtime — see <doc:DynamicDiscovery>.

### Writing a subclassed async backend

```objc
@interface MyGenerationBackend : NFKAsyncGenerationBackend @end
@implementation MyGenerationBackend
- (NSDictionary<NSString *, id> *)submitBodyForRequest:(NFKInferenceRequest *)request {
    return @{ @"prompt": request.prompt ?: @"", @"model": self.modelName ?: @"" };
}
- (nullable NSString *)jobIdentifierFromResponse:(NSDictionary *)response { return response[@"id"]; }
- (BOOL)isSucceededStatusResponse:(NSDictionary *)response { return [response[@"status"] isEqual:@"succeeded"]; }
- (nullable NFKInferenceResult *)resultFromStatusResponse:(NSDictionary *)response error:(NSError **)error {
    return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: response[@"output"] ?: @"" }];
}
@end
```

## Topics

### Shipped backends

- ``NFKPassthroughBackend``
- ``NFKCoreMLBackend``
- ``NFKCoreMLLanguageBackend``
- ``NFKRemoteBackend``
- ``NFKAnthropicBackend``
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
