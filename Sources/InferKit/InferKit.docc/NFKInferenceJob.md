# ``InferKit/NFKInferenceJob``

## Overview

A thread-safe handle to one asynchronous run. A backend returns it from
`submitInferenceJobForRequest:`; the caller observes progress, receives streamed partials, reads the
final result or error, and cancels.

![The job lifecycle: submitted becomes running, which resolves to succeeded, failed, or cancelled.](job-lifecycle)

```objc
NFKInferenceJob *job = [backend submitInferenceJobForRequest:request];
job.progressHandler   = ^(double fraction) { /* 0…1, update UI */ };
job.partialResultHandler = ^(NFKInferenceResult *partial) { /* stream tokens */ };
job.completionHandler = ^(NFKInferenceJob *finished) {
    if (finished.result) { /* success */ }
    else if (finished.error) { /* failed */ }
};
// [job cancel]; on user back-out — transitions to cancelled.
```

Prefer the job over the blocking `runInferenceForRequest:error:` for anything interactive: it keeps the
render thread free and lets the user cancel a long generation.
