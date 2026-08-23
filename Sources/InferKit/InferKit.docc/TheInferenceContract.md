# The inference contract

How a request becomes a result — synchronously, or through an observable job.

## Overview

Every model in InferKit runs the same way. A caller assembles an ``NFKInferenceRequest``, hands it to
an ``NFKInferenceBackend``, and reads an ``NFKInferenceResult``. That is the whole contract; backends
differ, the shape does not.

![A request flows into a backend and returns a result, synchronously or through an async job.](contract-flow)

### The request

``NFKInferenceRequest`` is immutable. It carries `inputs` (keyed media and text), `parameters`
(keyed knobs like temperature or strength), and an `outputModality`. Typed convenience accessors read
the single-natural-type inputs:

```objc
NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a lighthouse",
                                              NFKInputNegativePrompt: @"blurry" }];
request.prompt;          // @"a lighthouse"
request.negativePrompt;  // @"blurry"
```

Inputs and parameters are shared vocabulary — `NFKInput…` / `NFKParameter…` constants — so a request
built for one backend is legible to another.

### The backend

``NFKInferenceBackend`` is a protocol: `isReady`, a synchronous `runInferenceForRequest:error:`, and
an asynchronous `submitInferenceJobForRequest:`. Any object can adopt it — see <doc:Backends>.

### The result

``NFKInferenceResult`` is the immutable output: a dictionary of named outputs, plus type-checked
convenience getters for the keys with a single natural type.

![Result output keys, their typed values, and the convenience accessors.](outputs)

Image, mask, and video outputs stay on `outputForKey:` because their representation is chosen by the
backend (a `CVPixelBuffer`, a texture, a `CGImage`).

### The job

Inference is often multi-second. ``NFKInferenceJob`` is the async handle: submit, observe progress,
receive streamed partials, and cancel.

![The job lifecycle: submitted becomes running, which resolves to succeeded, failed, or cancelled.](job-lifecycle)

```objc
NFKInferenceJob *job = [backend submitInferenceJobForRequest:request];
job.progressHandler = ^(double fraction) { /* update UI */ };
job.completionHandler = ^(NFKInferenceJob *finished) {
    if (finished.result) { /* use finished.result */ }
};
// [job cancel]; when the user backs out.
```

## Topics

### The types

- ``NFKInferenceRequest``
- ``NFKInferenceBackend``
- ``NFKInferenceResult``
- ``NFKInferenceJob``
- ``NFKInferenceError``
