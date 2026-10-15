# The C interface

Build a backend from the model catalog, run it, follow its progress, and cancel it from C, C++, Rust,
or any language with a C foreign-function interface.

## Overview

`NFKInferKitC.h` is a pure C99 header in the `InferKitC` target. A program builds a backend by its
catalog identifier and a JSON object of options, and runs requests of tagged JSON with buffers beside
it. The buffers' memory is wrapped without a copy. A result carries every value with bytes as a buffer,
and InferKit's value classes, such as detections and keypoints, as JSON objects tagged by class.

```c
nfk_error *error = NULL;
nfk_backend *backend = nfk_backend_create("natural-language-embedding", NULL, &error);
nfk_payload request = { sizeof(nfk_payload), 0, "{\"inputs\": {\"prompt\": \"a lighthouse\"}}", NULL };
nfk_result *result = nfk_backend_run(backend, &request, &error);
printf("%s\n", nfk_result_json(result));
nfk_result_free(result);
nfk_backend_free(backend);
```

The rules every function follows:

- Everything InferKit returns is freed with its own `*_free` function, and each accepts NULL.
- A request's memory stays valid until ``nfk_backend_run`` returns, or until the completion of
  ``nfk_backend_submit`` fires.
- A job's callbacks run on a serial queue of its own at user-initiated quality of service. The
  completion fires exactly once, unless ``nfk_job_free`` comes first, and no callback runs after
  ``nfk_job_free`` returns.
- No Objective-C exception crosses into C. An exception becomes an error.

`Docs/c-api.md` in the repository gives the linking recipes for C, C++, and Rust, the request and
result forms, and the error codes.

## Topics

### The library

- ``nfk_api_version``
- ``nfk_release_version``
- ``nfk_catalog_copy_json``
- ``nfk_plugins_add_directory``

### Backends

- ``nfk_backend_create``
- ``nfk_backend_copy_description``
- ``nfk_backend_prepare``
- ``nfk_backend_unload``
- ``nfk_backend_free``

### Runs

- ``nfk_backend_run``
- ``nfk_backend_submit``
- ``nfk_job_cancel``
- ``nfk_job_free``
- ``nfk_progress_callback``
- ``nfk_completion_callback``

### Requests and results

- ``nfk_payload``
- ``nfk_buffer``
- ``nfk_buffer_kind``
- ``nfk_result_json``
- ``nfk_result_buffer_count``
- ``nfk_result_buffer``
- ``nfk_result_free``

### Errors and strings

- ``nfk_error_domain``
- ``nfk_error_code``
- ``nfk_error_message``
- ``nfk_error_copy_details_json``
- ``nfk_error_free``
- ``nfk_string_free``
