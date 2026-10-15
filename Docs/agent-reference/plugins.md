<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Plugins

The plugin system: what a plugin is, the package layout that carries it, the protocol a host and a
plugin process speak, the attachments that carry large values, the C interface, and what was measured
before any of it was built. The consumer guide is `Docs/plugins.md`, and the C API's is `Docs/c-api.md`.

## Shape

- One plugin runs in one process of its own by default. Listing a plugin reads its manifest and loads
  no code. The process starts on first use, exits when idle, and is replaced by a new version while the
  app runs.
- A plugin the app trusts runs in the app's process instead. It then stays loaded until the app quits,
  a crash in it ends the app, and it has every access the app has.
- Three kinds, named by the manifest's `entry.kind`:
  - `objc` → an Objective-C or Swift bundle whose principal class adopts `NFKPlugin`.
  - `c-abi` → a library that exports `nfk_plugin_entry` from `NFKPluginABI.h`.
  - `executable` → a program that speaks the protocol below itself.
- The plugin host, `inferkit-plugin-host`, serves an `objc` or `c-abi` plugin in its own process. Its
  whole program is `NFKPluginHostMain`.
- Every plugin source file is wrapped in `#if TARGET_OS_OSX`, and every public plugin class except
  `NFKPluginManifest` is `API_UNAVAILABLE(ios, tvos, watchos)`. iOS and tvOS build the headers and none
  of the code.

The public classes and where their behavior lives:

| class | role |
| --- | --- |
| `NFKPlugin`, `NFKPluginContext` | the protocol a plugin's principal class adopts, and what the host gives it |
| `NFKPluginManifest` | the manifest, read without loading code; available on every platform |
| `NFKPluginSignaturePolicy` | which code signatures a host accepts |
| `NFKPluginLoader`, `NFKLoadedPlugin` | loads a plugin's code into the calling process |
| `NFKPluginProcess` | supervises one plugin's process on the host side |
| `NFKPluginBackend` | one plugin model as an `NFKInferenceBackend` |
| `NFKPluginService`, `NFKPluginHostMain` | serves a plugin on the plugin side of the channel |
| `NFKPluginRegistry` | the installed plugins, listed in the model catalog |

The private files: `NFKPluginChannel` (framing, descriptor passing, the hand-off, the peer's identity),
`NFKPluginSharedMemory` (regions), and `NFKPluginWire` (message names, limits, the attachment writer and
reader).

## Package layout

| target | path | holds |
| --- | --- | --- |
| `InferKitC` | `Sources/InferKitC` | `NFKAPIVersion.h`, `NFKPluginABI.h`, `NFKInferKitC.h` under `include/InferKitC/`, and `NFKPluginABILayout.c` |
| `InferKit` | `Sources/InferKit` | the core, which depends on `InferKitC` |
| `inferkit-plugin-host` | `Sources/inferkit-plugin-host` | `main.m`, and `Entitlements/`, excluded from the build |

- SwiftPM rejects a second directory beside the umbrella in `include/`, so the C headers are a target of
  their own. The C headers are pure C99 with `extern "C"` guards and quoted includes, and the
  Objective-C umbrella does not import them.
- `NFKPluginABILayout.c` is the target's one source. It asserts every structure offset and size under
  `__LP64__`, so a field that moves fails the build. Its one symbol, `NFKPluginABILayoutVersion`, is
  hidden; libtool warns "has no symbols" about an object without one (measured with `libtool -static`).
- `NFK_API_VERSION_MAJOR`, `NFK_API_VERSION_MINOR`, and the packed `NFK_API_VERSION` live in
  `NFKAPIVersion.h`. `NFKInferKit.h` imports it by three routes, first that resolves:
  1. `<InferKitC/NFKAPIVersion.h>` → SwiftPM, through the `InferKitC` dependency.
  2. `<InferKit/NFKAPIVersion.h>` → CocoaPods and the built frameworks, which place the header beside
     the core's headers. `build.sh` and `build-mlx.sh` copy it there.
  3. `"../../../InferKitC/include/InferKitC/NFKAPIVersion.h"` → a tool that reads the source tree with
     only `Sources/InferKit/include` on its path, such as `Tools/doc-snippets/check-objc.py`.
- `Tools/xcframework/inferkit-version.sh` and `Tools/abi/abi-surface.py` read the macros from
  `NFKAPIVersion.h`, and from `NFKInferKit.h` in a tree that predates it. The core DocC build passes
  `NFKAPIVersion.h` to `clang -extract-api` with the core's headers.
- The podspec compiles `Sources/InferKitC/**/*.{h,c}` with the core and lists its headers as public.
  CocoaPods flattens them into `InferKit/`, which route 2 reaches.

### Signing the host

SwiftPM cannot sign with entitlements, so an app that embeds the host signs it:

```
codesign --force --timestamp --options runtime \
    --entitlements Sources/inferkit-plugin-host/Entitlements/inferkit-plugin-host.entitlements \
    --sign "Developer ID Application: <Team> (<TEAMID>)" \
    MyApp.app/Contents/Helpers/inferkit-plugin-host
```

- `inferkit-plugin-host.entitlements` → `com.apple.security.cs.disable-library-validation`, so the
  hardened host loads plugin code another team signed.
- `inferkit-plugin-host-sandboxed.entitlements` → `com.apple.security.app-sandbox` and
  `com.apple.security.inherit`, for a sandboxed app. A helper that inherits the sandbox carries those two
  entitlements only, so the hardened runtime's library validation then admits only code the app's own
  team signed. Open question: a sandboxed app that serves other teams' `objc` or `c-abi` plugins needs
  them signed by its team, or the `executable` kind.
- Sign the helper before the app, and the app last, so the app's seal covers the signed helper.

An app that loads a trusted `objc` plugin into its own process keeps the core's symbols exported:
`STRIP_STYLE = non-global`, or the InferKit dynamic framework. The plugin binds to them at load time.

## Measured before building (2026-10-15)

1. **The host exports the core.** `nm -gU` on `inferkit-plugin-host` lists `_OBJC_CLASS_$_NFK…` for all
   206 core classes, and the C functions and constants, in debug and in release (SwiftPM links release
   with `-dead_strip`).
2. **A `-undefined dynamic_lookup` bundle binds to the core inside xctest.** A bundle that references
   `NFKInferenceResult` and `NFKInferenceErrorDomain`, built with `-bundle -undefined dynamic_lookup`,
   loaded through `NSBundle` in the real `xctest` runner after it loaded `InferKitTests.xctest` (selected
   with a test name that matches nothing, so no test ran; the probe ran from an `atexit` handler injected
   with `DYLD_INSERT_LIBRARIES`). The class it bound is the test bundle's own: the same address. The
   fallback, a helper process for in-process tests, is not needed.
3. **The header route under CocoaPods.** `pod lib lint --quick` passes. A full lint stops at Xcode 27's
   refusal of macOS 11 as a deployment target, before any compile; the lint's generated workspace then
   built at `MACOSX_DEPLOYMENT_TARGET=12.0` (framework mode, Release): `BUILD SUCCEEDED`, with
   `NFKAPIVersion.h` reached through route 2. Every warning in that build predates the plugin work.
4. **A socket pair's peer is its creator.** `LOCAL_PEERTOKEN` and `LOCAL_PEERPID` on an end inherited by
   a spawned child report the parent, the process that called `socketpair`. A socket the child creates
   and sends back reports the child. This is why the protocol begins with a hand-off.
5. **Descriptors stay with their frame.** On a Unix stream socket, one `recvmsg` returns at most one
   `sendmsg`'s control data, starting at that message's first byte; three back-to-back messages of 64
   descriptors each arrive in three reads. Data without control data after it coalesces into the read.
6. **Shared memory.** A POSIX shared-memory object refuses a second `ftruncate` (`EINVAL`), so a peer
   cannot shrink it under a mapping. `fstat` reports no file type for it, while the temporary-file
   fallback reports a regular file. `MAP_PRIVATE` on it fails with `EINVAL`, so receivers map it shared.
   A name longer than 31 characters fails with `ENAMETOOLONG`.
7. **CoreVideo release callbacks.** `CVPixelBufferCreateWithBytes` and
   `CVPixelBufferCreateWithPlanarBytes` call the release callback only for a buffer they made; a failed
   creation calls nothing. `CVPixelBufferCreateWithBytes` accepts a row stride shorter than the width
   needs, so the decoder checks the geometry itself. The planar form accepts a NULL plane descriptor
   block for a biplanar format.
8. **QoS of a spawned child.** `posix_spawnattr_set_qos_class_np` with `QOS_CLASS_BACKGROUND` starts the
   child at base priority 4 (`PROC_PIDTASKINFO`'s `pti_priority`); a child without a class starts at 31.

## Transport

- The host creates a Unix stream socket pair and launches the plugin's process with one end as
  descriptor 3. `INFERKIT_PLUGIN_CHANNEL_FD=3` names it in the child's environment. Standard output and
  standard error stay logs.
- `NFKSubprocessPrivate.h` carries the launch hooks: `ownerInheritedDescriptors` (child number → this
  process's descriptor, numbers 3 and up) and `ownerQualityOfService`. The subprocess duplicates each
  source above every target number before the spawn, so no `dup2` in the child overwrites a source
  another still reads, and the standard streams go first. `POSIX_SPAWN_CLOEXEC_DEFAULT` still closes
  every descriptor not named.
- **Hand-off.** The plugin's first act: create a socket pair of its own, send one end to the host on
  descriptor 3 as the frame `{"type": "channel"}` with one descriptor, close descriptor 3, and speak on
  the other end. The host reads that one frame (up to its handshake timeout), closes descriptor 3, and
  speaks on the received socket, whose `LOCAL_PEERTOKEN` names the plugin's process. The host checks
  that audit token against the signature policy. An executable plugin performs the same hand-off.
- Both ends set `SO_NOSIGPIPE`, close-on-exec, and 1 MiB socket buffers. The channel reads on a queue of
  its own and delivers on a serial handler queue, so a handler blocked in a send does not stop the reads
  that would unblock it.

## Framing

A frame is a header and a JSON object:

| bytes | field | value |
| --- | --- | --- |
| 0–3 | magic | `0x504B464E`, "NFKP", little-endian |
| 4–7 | JSON length | at most 16 MiB (16777216) |
| 8–9 | descriptor count | at most 64 |
| 10–11 | flags | written 0, ignored on reading |
| 12– | JSON | UTF-8, one object with a string `"type"` |

- The header and the JSON go out in one `sendmsg` whose `SCM_RIGHTS` carries the descriptors. A partial
  send continues without control data; the descriptors rode the first byte.
- The receiver queues every descriptor it receives in order and gives each frame as many as its header
  counts, from the front.
- These close the channel with `NFKPluginErrorProtocolViolation`: a wrong magic number; a length or
  count over its limit; a complete frame whose descriptors are not queued (they arrive with the frame's
  first byte, so they never come later); `MSG_CTRUNC`; JSON that is not an object with a string type.
- End of file closes the channel with no error. A send on a closed channel fails with
  `NFKPluginErrorStopped`. A send that the peer does not drain within `sendTimeout` (30 s) fails.
- The sender keeps its own descriptors open; the receiver owns the duplicates it receives
  (`NFKPluginDescriptor` closes one on release unless taken).

## Messages

Every message has a `"type"`. A host request carries an integer `"id"`, and its reply is
`{"type": "reply", "id": n}` with exactly one of `"result"`, `"error"`, or `"cancelled": true`. Unknown
fields are ignored. An unknown type with an id is answered with an error of
`NFKPluginErrorProtocolUnsupported`; an unknown type without one is ignored.

| type | from | fields | result |
| --- | --- | --- | --- |
| `channel` | plugin | one descriptor, the plugin's socket | (none; the first frame only) |
| `hello` | host | `protocol {major, minor}`, `inferkitAPI`, `pluginABI`, `capabilities`, `options`, `limits {inlineBytes}` | `protocol`, `plugin {id, version}`, `inferkitAPI`, `capabilities`, `pid` |
| `describe` | host | | `models`, `tools`, `options` |
| `build` | host | `model`, `options` | `backend` (handle), `ready`, `inputs`, `parameters`, `outputs`, `modelInfo`, `loaded` |
| `prepare` | host | `backend` | `loaded`, `residentBytes`, `peakBytes` |
| `run` | host | `backend`, `request`, `stream`, `outputDirectory` | the result; `progress` messages first |
| `progress` | plugin | `id`, `progress` (−1 unknown), `partial` (optional) | (none) |
| `cancel` | host | `id` of the run | (none; the run replies `cancelled`) |
| `unload` | host | `backend` | (empty) |
| `release` | host | `backend` | (empty) |
| `status` | host | `backend` | `backendStatus`, `loaded`, `residentBytes`, `peakBytes` |
| `callTool` | host | `tool`, `arguments` | `toolResult` |
| `ping` | host | | (empty) |
| `shutdown` | host | | (empty); the plugin then exits |
| `event` | plugin | `event` (`loaded`, `unloaded`), `backend`, `residentBytes` | (none) |
| `log` | plugin | `level` (0–3), `message` | (none) |

- `request`, `partial`, and the run's result are the wire coding's request and result objects in the
  attachment mode.
- `outputDirectory` is a directory the host makes for the run. The plugin writes result files there,
  and the host deletes it once the result is consumed.
- A backend handle carries a generation. After a restart a stale `NFKPluginBackend` builds again from its
  stored model identifier and options before its next run.
- On first launch `describe` lists every model the manifest lists; a missing one is
  `NFKPluginErrorUnknownModel`. Differences in ports or options are logged.

## Versioning

- The protocol is versioned apart from the API and the plugin ABI: 1.0.
- `hello` carries the host's version. A different major version fails the handshake
  (`NFKPluginErrorProtocolUnsupported`); the two sides use the lower minor version.
- The C plugin ABI is 1.0 (`NFK_PLUGIN_ABI_VERSION_MAJOR`, `_MINOR`). A minor version only appends
  structure fields, and a reader uses a field only when `offsetof + sizeof <= struct_size`.

## Errors

- An `"error"` is the inner object of `+[NFKInferenceWireCoding JSONObjectForError:]`: `message`, `type`,
  `code`, `inferkit_domain`, `inferkit_code`, `inferkit_user_info`. The domain and code cross intact.
- A process that exits fails its runs in flight with `NFKPluginErrorProcessExited`, carrying the exit
  status or signal and the end of its standard error.
- A C plugin's error is `{"message", "domain"?, "code"?}`.
- Until a stage implements a stub, it answers `NFKPluginErrorNotImplemented` (1000), a private code
  outside the public `NFKPluginError` range. No stub remains once Phase 0 is complete.

## Attachments

`NFKInferenceWireCoding` has an attachment mode, on while `attachmentSink` (encoding) or
`attachmentSource` (decoding) is set. The HTTP route sets neither, so its JSON is unchanged:
`NFKWireAttachmentTests` pins the HTTP form of data, float arrays, a padded pixel buffer, and a file
against literal dictionaries, and the existing server tests keep their expectations.

- A value whose bytes exceed `inlineByteLimit` (64 KiB; `limits.inlineBytes` in `hello`) leaves the
  JSON: `{"$nfk": "buffer", "index": i, "as": kind, "length": n, …}`.

| `as` | extra fields | decodes to |
| --- | --- | --- |
| `data` | | `NSData` over the region, no copy |
| `float32Array`, `float64Array` | `count` | an `NSArray` of `NSNumber` (copied) |
| `multiArray` | `dataType`, `shape` | `MLMultiArray` over the region, no copy |
| `pixelBuffer` | `format`, `width`, `height`, `planes [{offset, bytesPerRow, rows}]` | `CVPixelBuffer` over the region, no copy |
| `pcm` | the stream description, `frameLength`, `buffers [{offset, length}]` | `AVAudioPCMBuffer` over the region, no copy |
| `cgImage`, `image` | | PNG bytes; a `CGImage`, or a 32BGRA pixel buffer for a texture |

- A pixel buffer keeps each plane's row padding; planes and PCM buffers start at 64-byte offsets. An
  IOSurface-backed buffer is copied into shared memory like any other; passing the surface itself is
  reserved for a later capability, `mach-surfaces`.
- The decoder checks every geometry against the region's length before it wraps the bytes: plane
  offsets and strides against the format's description, a multi-array's shape against its length, each
  PCM buffer against the frame count.
- A file, audio, or video value crosses by reference in this mode: its tag with `"path"` and `"name"` in
  place of `"base64"`, and `"index"` when the writer attaches an open descriptor
  (`passesFilesAsDescriptors`, for a sandboxed peer). The reader copies an indexed file into its
  temporary directory, and refuses a path outside `allowedFileDirectoryURL` when one is set. Without a
  source, a reference by path is refused as before ("carries no bytes"), so an HTTP client cannot name a
  path on the server.
- `NFKPluginAttachmentWriter` makes one region per attachment and lists their descriptors in index
  order for the frame. `NFKPluginAttachmentReader` maps each region on first use.

## Shared memory

- A region is `shm_open` with a random name, `O_EXCL`, `0600`, sized once with `ftruncate`, and
  unlinked at once; only descriptors reach it. Names hold at most 31 characters. A sandboxed process
  passes its app group prefix (`"<group>/"`), which leaves fewer random characters; at fewer than 10 the
  region falls back.
- The fallback is an unlinked temporary file. App → plugin, the plugin maps it as is. Plugin → app, the
  host copies it (`copiesFileRegions`), since a plugin that truncates its file would fault the host's
  mapping.
- A received region maps shared and writable. A descriptor that maps only for reading is copied, so
  every region is writable. A descriptor that is neither shared memory nor a regular file is refused.

## In process

- An `objc` plugin links InferKit's interface and does not embed its own copy: `-undefined
  dynamic_lookup`, or the framework the app ships. After loading, `objc_copyClassNamesForImage` refuses
  a bundle that defines any `NFK` class (`NFKPluginErrorDuplicateRuntime`).
- A `c-abi` plugin opens with `dlopen(RTLD_LOCAL)`. A private adapter turns its function table into
  `NFKInferenceBackend`s and implements the host table.
- In-process values pass directly: an `IOSurfaceRef`, a pointer, or a path.
- An in-process plugin is never closed with `dlclose`.

## Staging

- Stage 0: the layout, the C headers, every public header with a stub, the channel, shared memory, the
  attachment mode, the subprocess hooks, this note, and `NFKPluginChannelTests`,
  `NFKWireAttachmentTests`, `NFKSubprocessDescriptorTests`.
- Stage 1: the host side (`NFKPluginProcess`, `NFKPluginBackend`, a private tool proxy, and a launcher
  seam for an in-test peer), the loaders and serving (`NFKPluginLoader`, the C adapter,
  `NFKPluginService`, `NFKPluginHostMain`, `NFKPluginManifest`, `NFKPluginSignaturePolicy`), and the
  fixtures with the C header checks.
- Stage 2, after the catalog's build entry merges: `NFKInferKitC.m`, `NFKPluginRegistry` with the
  `plugin` engine builder, the completeness test, the consumer docs, the examples, and the ABI tool's
  coverage of both C headers.
