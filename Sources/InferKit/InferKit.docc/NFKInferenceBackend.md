# ``InferKit/NFKInferenceBackend``

## Overview

A backend is the swappable engine behind the contract. It reports readiness, runs a request
synchronously, and submits a request as an observable ``NFKInferenceJob``. Everything else in InferKit —
the shipped Core ML and remote backends, the MLX companion, a consumer's whisper.cpp — is just an object
adopting this protocol.

![One protocol, with shipped, companion, and consumer-brought engines.](backend-taxonomy)

```objc
id<NFKInferenceBackend> backend = /* shipped, companion, or discovered */;
if (backend.isReady) {
    NSError *error = nil;
    NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
}
```

- Inference is synchronous and often multi-second — run `runInferenceForRequest:error:` off the
  main/render thread, or prefer `submitInferenceJobForRequest:` for progress and cancellation.
- `isReady` is `NO` until a backend's weights or endpoint are configured; a lazy backend loads on
  first use.
- `supportedParameterKeys` and `supportedInputKeys` state the keys a backend acts on, for a caller
  that has to know in advance: a router picking the engine a request needs, or a bridge presenting
  the backend to another framework. Both are optional, and a backend that declares nothing is used
  the way it always was.
- `NFKInferencePrepare(backend, &error)` prepares a backend whether or not it implements
  `prepareWithError:`, the way `NFKInferenceSubmit` submits whether or not it implements
  `submitInferenceJobForRequest:`.

See <doc:Backends> for the shipped implementations and <doc:DynamicDiscovery> for bringing your own.
