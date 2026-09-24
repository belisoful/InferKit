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
- ``NFKVisionTextBackend``, ``NFKVisionSegmentationBackend``, ``NFKVisionPoseBackend``,
  ``NFKVisionFaceBackend``, and ``NFKVisionFeaturePrintBackend`` — Apple's Vision framework behind the
  contract: text recognition, masks, pose, faces, and an image embedding, with no weights to ship.
- ``NFKVisionClassificationBackend``, ``NFKVisionAnimalBackend``, ``NFKVisionRectangleBackend``,
  ``NFKVisionContourBackend``, ``NFKVisionMeasurementBackend``, and ``NFKVisionRegistrationBackend`` —
  the rest of what Vision reports: a scene taxonomy, cats and dogs, four-cornered shapes and barcodes,
  traced outlines, named readings, and the alignment between two frames.
- ``NFKVisionTrackingBackend`` — a region followed across frames. It holds state, which no other
  backend does.
- ``NFKVisionCoreMLBackend`` — a Core ML model run through Vision, which resizes and crops the image
  the model's description asks for.
- ``NFKVideoToolboxBackend`` — Apple's neural video processors: upscaling, frame interpolation, and
  optical flow, on Apple silicon.
- ``NFKSpeechRecognitionBackend`` — Apple's speech recognizer, transcribing a recording into text and
  per-word segments once the user allows it.
- ``NFKSoundClassificationBackend`` — Apple's everyday-sound taxonomy, a segment per analysis window.
- ``NFKSpeechSynthesisBackend`` — the system voices, answering the contract the remote and MLX voices
  answer.
- ``NFKTextEmbeddingBackend`` — NaturalLanguage's sentence vectors, with no model to ship.
- ``NFKRemoteBackend`` and ``NFKAnthropicBackend`` — the chat clients: one OpenAI-compatible, one for
  Anthropic's Messages API, which differs in its headers, its required `max_tokens`, and its top-level
  system prompt. ``NFKRemoteProvider`` names the services
  it is pointed at (hosted APIs and the local runners Ollama, LM Studio, llama.cpp, and vLLM), and
  ``NFKRemoteModelCatalog`` lists the models a provider serves so a caller chooses one from the
  server's own list. Discovery probes the local ports and answers which runner is up, so calling code
  names none. A local runner's native API — what is installed and loaded, and Ollama's
  download and delete — is reached through ``NFKLocalModelRunner``. Both chat clients stream through
  the job form, cancel the request when the job is cancelled, take tools and a JSON Schema, and retry
  a rate limit through ``NFKRemoteTransport``, which reports a failing status under the core code an
  app acts on: rate limited (with the reset date), refused, or a backend failure.
- ``NFKRemoteEmbeddingBackend`` — an OpenAI-compatible embeddings client; the vector comes back under
  the same key the on-device embedders use.
- ``NFKRemoteSpeechBackend`` — an OpenAI-compatible text-to-speech client, answering with an
  `NFKAudioAsset` the way the on-device speech backend does.
- ``NFKRemoteImageBackend`` — OpenAI-compatible image generation, edits, and inpainting, chosen from the
  request the way the Stable Diffusion backend chooses. An image beside a prompt through the chat
  backends is a vision question; ``NFKImageCoding`` is the codec under all of it. Audio, PDFs, and a
  clip's sampled frames (``NFKVideoSampling``) ride beside the prompt the same way, and a chat model can
  speak its reply.
- ``NFKRemoteVideoBackend`` — video generation as a job (Gemini Veo, xAI, Together, OpenRouter, OpenAI), the first shipped
  ``NFKAsyncGenerationBackend``.
- ``NFKRemoteModerationBackend`` — per-category moderation scores and a verdict for text or an image.
- ``NFKTypeSafeBackend`` — TypeSafe AI's System One API, which serves Jev: typed answers to
  ``NFKDecisionQuestion``s about a state, rather than text.
- ``NFKRemoteReranker`` — query-and-documents relevance scores, the shape of the on-device reranker.
- ``NFKRemoteResponsesBackend`` — the Responses API: service-run tools, reasoning summaries, background
  jobs, and continuation by response id.
- ``NFKGeminiInteractionsBackend`` — Gemini's native Interactions API: text, images, speech, music,
  transcription, and video from one endpoint.
- ``NFKRemoteCompletionBackend`` — raw continuation and fill-in-the-middle for code.
- ``NFKRemoteOCRBackend`` — a document or image read into markdown, with a schema filled from it.
- ``NFKRemoteClassifierBackend`` — a hosted classifier's scores as classifications.
- ``NFKRemoteTokenCounter`` — what a request costs in the hosted model's own tokens.
- ``NFKRealtimeSession`` — a live WebSocket session: spoken conversation, streaming transcription and
  speech, live translation, and live music, one set of calls across the providers.
- ``NFKRemoteFileStore`` — a hosted Files API: upload once, then name the file in requests.
- ``NFKRemoteRetrievalStore`` — hosted retrieval stores: create, fill, search, and delete.
- ``NFKRemoteUsageReporter`` — usage, spend, and balance, read with an administrative key.
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
- ``NFKVisionTextBackend``
- ``NFKVisionSegmentationBackend``
- ``NFKVisionPoseBackend``
- ``NFKVisionFaceBackend``
- ``NFKVisionFeaturePrintBackend``
- ``NFKVisionClassificationBackend``
- ``NFKVisionAnimalBackend``
- ``NFKVisionRectangleBackend``
- ``NFKVisionContourBackend``
- ``NFKVisionMeasurementBackend``
- ``NFKVisionRegistrationBackend``
- ``NFKVisionTrackingBackend``
- ``NFKVisionCoreMLBackend``
- ``NFKVideoToolboxBackend``
- ``NFKSpeechRecognitionBackend``
- ``NFKSoundClassificationBackend``
- ``NFKSpeechSynthesisBackend``
- ``NFKTextEmbeddingBackend``
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
- ``NFKRemoteResponsesBackend``
- ``NFKGeminiInteractionsBackend``
- ``NFKRemoteCompletionBackend``
- ``NFKRemoteOCRBackend``
- ``NFKRemoteClassifierBackend``
- ``NFKRemoteTokenCounter``
- ``NFKRealtimeSession``
- ``NFKRealtimeWebSocket``
- ``NFKRemoteFileStore``
- ``NFKRemoteFile``
- ``NFKRemoteRetrievalStore``
- ``NFKRemoteUsageReporter``
- ``NFKTypeSafeBackend``

### Media coding

- ``NFKImageCoding``
- ``NFKVideoSampling``
