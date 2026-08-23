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
- ``NFKRemoteBackend`` — an OpenAI-compatible chat client.
- ``NFKRemoteTranscriptionBackend`` — an OpenAI-compatible audio-to-text client.
- ``NFKAsyncGenerationBackend`` — a subclassable submit → poll → fetch base for job-style generation
  services. Map the service's JSON through its template methods; the base owns the loop and the
  ``NFKInferenceJob``.

### Companion (link to enable)

Bringing a heavier runtime is a matter of linking a package: InferKitMLX (31 MLX models, plus the
bundled Stable Diffusion `NFKMLXBackend`) and InferKitFoundationModels (on-device LLM).

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
- ``NFKRemoteTranscriptionBackend``
- ``NFKAsyncGenerationBackend``
