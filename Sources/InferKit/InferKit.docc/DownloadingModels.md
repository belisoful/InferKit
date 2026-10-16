# Downloading models

Weights are fetched at runtime and cached on device, never bundled at build time. A download is
planned, observed, cancelled, and removed through the same hub.

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
snapshot just requested stays even when it alone exceeds the limit, and so does every snapshot a
running ``NFKDownload`` writes into.

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

### Plan, progress, and cancellation

A download is planned before it starts. `-planDownloadOfRepo:revision:paths:error:` lists what a
download would fetch from the Hugging Face API: each file's path, size, and SHA-256 where the
repository records one, whether the cache holds it whole, and the bytes still to fetch. A path may be
a glob. `-planDownloadOfURL:sha256:error:` sizes a plain URL with a HEAD request, and
`-planDownloadOfModelSource:error:` plans a catalog variant's source. A gated repository without an
accepted access token fails with `NFKDownloadErrorAuthenticationRequired`.

`-downloadWithPlan:progressHandler:completionHandler:` runs a plan as an ``NFKDownload``, which reports
bytes received and expected per file and overall, the current file, the transfer rate, and its state.
Its `progress` is the same as an `NSProgress`. `cancel` stops the transfer in flight and removes the
partial file: a file streams into a temporary `.download` name and moves into place only whole. A
transfer that drops resumes with an HTTP Range request. A plan larger than `cacheSizeLimit` or than
the volume's free space is refused before any transfer. A file whose size differs from its listed
size fails with `NFKDownloadErrorSizeMismatch`, and one whose contents differ from its listed SHA-256
with `NFKDownloadErrorChecksumMismatch`; neither stays in the cache.

Every repository, revision, path, and file name names a place inside the cache. One that is empty or
absolute, holds an empty, `.`, or `..` component, or holds a backslash or a NUL character is refused
with `NFKDownloadErrorUnsafePath` before anything is read, written, or removed, and so is a repository
listing that names such a file.

```objc
NFKDownloadPlan *plan = [hub planDownloadOfRepo:@"Qwen/Qwen3-0.6B" revision:nil
                                          paths:@[ @"config.json", @"*.safetensors" ] error:&error];
NFKDownload *download = [hub downloadWithPlan:plan
                              progressHandler:^(NFKDownloadProgress *progress) { /* fractionCompleted */ }
                            completionHandler:^(NSURL *localURL, NSError *downloadError) { }];
[download cancel];
```

A download also collects work that knows nothing of it. While it is current on a thread
(`-performAsCurrent:`), every hub download on that thread reports to it and stops when it is
cancelled, and the companion packages' asynchronous factories carry it to their background queue.

### Catalog models

`-downloadModelSource:progressHandler:completionHandler:` plans and downloads a catalog variant's
source, so an app fetches every catalog model the same way. A source that needs a converter fails
with `NFKDownloadErrorRequiresConversion`. A file that names its own repository, revision, or URL,
such as an adapter published apart from its base model, is fetched from there and caches in that
repository's snapshot, where the base model's own download finds it. The plan, the status, and the
removal cover every repository a source's files come from.

### Downloaded or not, and removal

`-downloadStatusForModelSource:` reads the cache alone and answers complete, partial, or absent, with
the bytes on disk. `-removalPlanForModelSource:error:` lists the files a removal deletes and the files
it keeps, each with its reason: a file another catalog model whose download is complete uses, or the
target of a symbolic link that something else still needs, such as a file of another snapshot. A file
that a link elsewhere in the cache resolves to stays too. `-removeFilesInPlan:error:` deletes them,
all or none, and returns the bytes freed. It refuses with `NFKDownloadErrorInUse` while a download in
the process writes into the same snapshot.

## Topics

### The type

- ``NFKHFHub``

### Downloads

- ``NFKDownload``
- ``NFKDownloadProgress``
- ``NFKDownloadFileProgress``
- ``NFKDownloadPlan``
- ``NFKDownloadPlanFile``

### Status and removal

- ``NFKDownloadStatus``
- ``NFKRemovalPlan``
- ``NFKCachedFile``
