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
  the backend to another framework. `supportedOutputKeys` states the result keys it writes. All three
  are optional, and a backend that declares nothing is used the way it always was.
- A failure is an `NSError` in `NFKInferenceErrorDomain` whose code says what to do:
  `kNFKError_InferenceRefused` means the engine declined the content and retrying will not help,
  `kNFKError_InferenceRateLimited` means back off, and `kNFKError_RemoteUnreachable` means nothing
  answered. An engine wrapping another framework keeps the original under `NSUnderlyingErrorKey`.
- `NFKInferencePrepare(backend, &error)` prepares a backend whether or not it implements
  `prepareWithError:`, the way `NFKInferenceSubmit` submits whether or not it implements
  `submitInferenceJobForRequest:`.

## Freeing a model's memory

A backend that holds a model in memory can free it and keep its configuration. Four optional members
state the contract, and `NFKInferenceUnload`, `NFKInferenceIsLoaded`, `NFKInferenceResidentByteCount`,
and `NFKInferencePeakByteCount` reach them on any backend.

- `unload` frees the weights and caches. The next run or `prepareWithError:` loads them again from the
  same source, and the identifier, the declared keys, and `isReady` keep their answers meanwhile.
- `isLoaded` is `NO` from the unload until the next load. A backend that cannot load again, such as one
  built from objects in memory, makes `unload` a no-op and stays loaded.
- `residentByteCount` is the rise in the runtime's allocated memory over the last load, and 0 while
  the model is unloaded.
- `peakByteCount` is the most the last load needed at once, kept after an unload.

All four are safe from any thread. An unload never cuts a run:

- no run in flight → the model is freed before `unload` returns;
- runs in flight → `unload` returns at once, and the last run to end frees the model. `isLoaded` stays
  `YES` until then, and a run's completion handler may call `unload`.

The readings cover the whole process, so work that overlaps a load falls into its figures. `modelInfo`
carries them under `NFKModelInfoResidentBytes` and `NFKModelInfoPeakBytes`, and ``NFKInferenceServer``
reports them for each hosted model.

See <doc:Backends> for the shipped implementations and <doc:DynamicDiscovery> for bringing your own.
