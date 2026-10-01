# ``InferKit/NFKInferenceJob``

## Overview

A thread-safe handle to one asynchronous run. A backend returns it from
`submitInferenceJobForRequest:`; the caller observes progress, receives streamed partials, reads the
final result or error, and cancels.

![The job lifecycle: submitted becomes running, which resolves to succeeded, failed, or cancelled.](job-lifecycle)

```objc
NFKInferenceJob *job = [backend submitInferenceJobForRequest:request];
job.progressHandler   = ^(NFKInferenceJob *running) {
    /* running.progress is 0…1; running.partialResult carries the tokens so far */
};
job.completionHandler = ^(NFKInferenceJob *finished) {
    if (finished.result) { /* success */ }
    else if (finished.error) { /* failed */ }
};
// [job cancel]; on user back-out — transitions to cancelled.
```

Prefer the job over the blocking `runInferenceForRequest:error:` for anything interactive: it keeps the
render thread free and lets the user cancel a long generation.
