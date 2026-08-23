# Downloading models

Weights are fetched at runtime and cached on device — never bundled at build time.

## Overview

Model weights are large, often licensed or gated, and updated independently of the app. So InferKit
downloads them at runtime and caches them on device, rather than embedding them in the binary.
``NFKHFHub`` is the access layer: it turns a repo, revision, and path into a Hugging Face URL, fetches
the file, verifies it, and caches it — with no inference knowledge of its own.

### Where files live

A file caches at `<cacheDirectoryURL>/<repo>/<revision>/<path>`. The cache folder is host-supplied so a
sandboxed app can point at a security-scoped, user-controlled location and keep multi-gigabyte
checkpoints off the sandbox container. When a host has no preference, `defaultCacheDirectoryURL` gives a
ready location under Application Support:

```objc
NFKHFHub *hub = [NFKHFHub hubWithCacheDirectoryURL:NFKHFHub.defaultCacheDirectoryURL];
NSError *error = nil;
NSURL *local = [hub downloadRepo:@"org/model"
                        revision:nil            // defaults to "main"
                            path:@"model.safetensors"
                          sha256:nil            // optional integrity check
                           error:&error];       // blocking — call off the main thread
```

### Off the main thread

The download blocks (a file must arrive before a model can load), so the caller runs it off the
main/render thread — or uses the asynchronous form, which imports into Swift as `try await`:

```objc
[hub downloadRepo:@"org/model" revision:nil path:@"model.safetensors" sha256:nil
completionHandler:^(NSURL *url, NSError *asyncError) { /* ready on a background queue */ }];
```

### First run, cached after

A download is skipped when the file is already cached (and its checksum matches). An optional SHA-256
forces a re-fetch on mismatch. The companion factories (`+backendWith…repo:…`) call this for you and
substitute `defaultCacheDirectoryURL` when passed `nil`.

## Topics

### The type

- ``NFKHFHub``
