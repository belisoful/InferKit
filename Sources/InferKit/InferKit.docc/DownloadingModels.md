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

### Size and backup

`cacheSizeLimit` caps the bytes the cache holds; `NFKHFHubUnlimitedCacheSize` (-1) is no cap. Over
the limit, a download evicts whole `<repo>/<revision>` snapshots, least recently used first. The
snapshot just requested stays even when it alone exceeds the limit.

Only a snapshot the hub owns is eligible, so a shared folder keeps everything else. The hub owns a
snapshot it downloads into. A snapshot cached before InferKit 0.4.0 becomes owned on its next download
or cache hit, or at once through `-adoptCachedRepo:revision:error:`. `-pinCachedRepo:revision:error:`
keeps a snapshot through every eviction, and can pin a model before its first download.

`excludesCacheFromBackup` defaults to `YES`. The first download excludes the cache folder from Time
Machine on macOS and from iCloud backup on iOS, since every file in it can be downloaded again.
`+setExcludedFromBackup:forURL:error:` sets or clears the exclusion on any folder.

Both settings start from process-wide class defaults, `defaultCacheSizeLimit` and
`defaultExcludesCacheFromBackup`, which also reach the hubs the companion factories create:

```objc
NFKHFHub.defaultCacheSizeLimit = 20LL * 1024 * 1024 * 1024;
```

## Topics

### The type

- ``NFKHFHub``
