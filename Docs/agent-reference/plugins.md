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
  `NFKAPIVersion.h` and `NFKInferKitC.h` to `clang -extract-api` with the core's headers.
- `build.sh` and `build-mlx.sh` copy all three C headers beside the core's in every slice. The
  `InferKit` module maps `exclude header` `NFKInferKitC.h` and `NFKPluginABI.h`, which the umbrella does
  not import, so a module build reports no incomplete umbrella and a C program includes them as text.
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
  team signed. A sandboxed app that serves other teams' `objc` or `c-abi` plugins needs them signed by
  its team, or the `executable` kind (`Docs/plugins.md`, "Sandboxed and unsandboxed apps").
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
9. **An in-flight socket needs its sender's copy.** A socket sent with `SCM_RIGHTS` whose sender then
   closes every descriptor of it can be collected by the kernel's Unix-domain garbage collector before
   the receiver installs it; the receiver then gets a socket that is already hung up (`POLLHUP`, reads
   return 0) while its partner is still open. Measured 2026-10-15 in plain C over 300 trials each, with
   the handed socket and the transport closed before the receive: 18 hung up with no delay before the
   receive, 300 with 2 ms. Closing only the handed socket: 65 of 300 at 2 ms. Keeping the handed socket
   open until the receiver has it: 0 of 300. The plugin side therefore holds it until the host closes
   the transport.

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
  descriptor 3 as the frame `{"type": "channel"}` with one descriptor, and speak on the other end. The
  host reads that one frame (up to its handshake timeout), closes its end of the transport, and speaks
  on the received socket, whose `LOCAL_PEERTOKEN` names the plugin's process. The host checks that audit
  token against the signature policy.
- The plugin keeps its copy of the sent end, and descriptor 3, open until descriptor 3 reads end of
  file, which is the host closing its end after receiving the frame; it closes both then, or after 300
  seconds for a host that never reads (measurement 9). `channelByHandingOffDescriptor:error:` does this
  on a queue of its own and returns at once. An executable plugin performs the same hand-off and the
  same hold; the executable fixture holds them on a thread.
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
| `hello` | host | `protocol {major, minor}`, `inferkitAPI`, `pluginABI`, `capabilities`, `options`, `limits {inlineBytes}` | `protocol`, `plugin {id, version}`, `inferkitAPI`, `capabilities`, `pid`, optional `limits {inlineBytes}` |
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
| `event` | plugin | `event` (`loaded`, `unloaded`), `backend`, `residentBytes`, optional `peakBytes`; or `event` `verified` alone | (none) |
| `log` | plugin | `level` (0–3), `message` | (none) |

- `request`, `partial`, and the run's result are the wire coding's request and result objects in the
  attachment mode.
- InferKit's value classes (detections, keypoints, classifications, segments, MIDI, beats,
  quadrilaterals, decision answers) cross as JSON objects tagged by class, never as keyed archives, in
  both directions: the host's requests (`NFKPluginBackend`), the service's results, partials, and tool
  results (`NFKPluginService`), and the C adapter's requests to a C plugin. `NFKWireValueObjects.h` lists
  the fields; `Docs/c-api.md` gives them for a plugin author. A reader accepts an archive as well, so an
  Objective-C plugin decodes either form. The protocol stays 1.0, since nothing has shipped.
- `outputDirectory` is a directory the host makes for the run. The plugin writes result files there,
  and the host deletes it once the result is consumed.
- A backend handle is an integer, unique within one process. The host pairs it with the process's
  generation: after a restart a stale `NFKPluginBackend` builds again from its stored model identifier
  and options before its next run.
- The hello reply's `inferkitAPI` is the plugin side's API version as a string ("1.0"), and its
  `capabilities` is `["attachments"]`, with `"file-descriptors"` added when the host offered it.
- A `progress` message carries the run's latest partial result whenever there is one; `stream` does not
  suppress it.
- `cancel` gets no reply of its own. The run it names replies once, with `"cancelled": true`.
- `toolResult` is the wire coding's tagged tool result:
  `{"$nfk": "toolResult", "text", "isError", "images", "outputs"}`.
- The attachment mode decides per value: an input large enough to be attached is answered with an
  attached output, and an inline input inline.
- On first launch `describe` lists every model the manifest lists; a missing one is
  `NFKPluginErrorUnknownModel`. Differences in ports or options are logged.

## Versioning

- The protocol is versioned apart from the API and the plugin ABI: 1.0.
- `hello` carries the host's version. A different major version fails the handshake
  (`NFKPluginErrorProtocolUnsupported`); the two sides use the lower minor version.
- The C plugin ABI is 1.0 (`NFK_PLUGIN_ABI_VERSION_MAJOR`, `_MINOR`). A minor version only appends
  structure fields, and a reader uses a field only when `offsetof + sizeof <= struct_size`.
- An array of buffers strides by its first element's `struct_size`, so a reader of an earlier minor
  version steps over fields it does not know. Every element of one array has the same `struct_size`.
  `NFKPluginABILayout.c` asserts that `sizeof(nfk_plugin_buffer)` is a multiple of its alignment, so
  the stride keeps every element aligned.

## Errors

- An `"error"` is the inner object of `+[NFKInferenceWireCoding JSONObjectForError:]`: `message`, `type`,
  `code`, `inferkit_domain`, `inferkit_code`, `inferkit_user_info`. The domain and code cross intact.
- A process that exits fails its runs in flight with `NFKPluginErrorProcessExited`, carrying the exit
  status or signal and the end of its standard error.
- A C plugin's error is `{"message", "domain"?, "code"?}`.
- Every host deadline fails with `NFKPluginErrorTimedOut`: verification (`verificationTimeout`), hello
  and describe (`handshakeTimeout`), build and prepare (`buildTimeout`), and a run that a restart
  cancelled and that has not ended 5 seconds later. The description names the step and the limit.
  `NFKPluginErrorProtocolViolation` is for malformed traffic only. A hand-off that does not arrive within
  the handshake timeout reports `NFKPluginErrorTimedOut`; a malformed one reports
  `NFKPluginErrorHandshakeFailed`.

## Debug diagnostics

DEBUG builds check descriptor ownership (`NFKPluginAdoptDescriptor`, `NFKPluginDisownDescriptor`,
`NFKPluginCloseDescriptor` in `NFKPluginWire`): a number adopted while another owner holds it, a close of
a descriptor that is no longer open, or a close by no owner stops the process with the number's recent
events. `INFERKIT_PLUGIN_TRACE=1` adds a line on standard error for every channel's adoption, `close()`
call, and end, with its socket, the real reason, whether its descriptor still names the adopted socket,
and each event's callers, plus both sides of each hand-off. Off by default, because symbolicating the
callers slows descriptor-heavy tests.
- The plugin host traces its shutdown under `INFERKIT_PLUGIN_TRACE=1`: the service answered shutdown and
  closes its channel, the channel ended, the runs drained, the service returned, and main returns with its
  status. `NFKPluginServiceTests/testThePluginHostServesABundleItVerifies` failed 1 run in 7 on
  2026-10-15 with a host still running 10 s after it answered shutdown. Reading the path (reply, close,
  the read queue's end, the close handler's semaphore, a drain bounded by `drainTimeout`, `willUnload`,
  return from main) found no wait without a bound, and `NFKSubprocess` reports an exit within
  `NFKSubprocessDrainGrace` (0.5 s) of the reap, so the host had not exited. On that failure the test now
  samples every thread of the host for a second and reports it with `ps` and the end of standard error.
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

- `-[AVAudioPCMBuffer initWithPCMFormat:bufferListNoCopy:deallocator:]` copies the `AudioBufferList` it
  is given, and its deallocator receives that internal copy (measured 2026-10-15: three distinct
  addresses). The decoder keeps its list in an `NSMutableData` the deallocator block holds and frees nothing
  by hand. Every PCM field is checked before AVFoundation sees a pointer: frame count, bytes per frame,
  the buffer count for the layout, each buffer's length and alignment within the sliced attachment.

## Descriptor ownership

- Every descriptor a channel, an `NFKPluginDescriptor`, a region, the hand-off, or the attachment reader
  holds goes through `NFKPluginAdoptDescriptor`, `NFKPluginDisownDescriptor`, and `NFKPluginCloseDescriptor`
  (`NFKPluginWire.h`). Each has one owner, closes once, and its field becomes -1. In a DEBUG build the
  process aborts, naming the owner, when a number is adopted while another owner holds it, when a close
  finds the descriptor already closed, or when an unadopted descriptor is closed: a stray close of a
  reused number otherwise kills a later socket with no error of its own.
- `NFKSubprocess`'s inherited-descriptor hook duplicates each descriptor when it is set and owns the
  copies, closing them when the property is set again or the subprocess is released, so a caller closing
  its own numbers never leaves a later launch with a reused one.
- A writer whose descriptors a send carries is declared `NS_VALID_UNTIL_END_OF_SCOPE`, since an optimized
  build may otherwise release it, and close its regions, between reading `descriptors` and the send.

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
- macOS rounds a shared-memory object up to whole pages whatever `ftruncate` asks for: 16 KiB on Apple
  silicon (measured 2026-10-15: requests of 1, 4096, 4097, 16383, and 16384 bytes report `st_size`
  16384; 16385 reports 32768). A region therefore has two lengths: `length`, the bytes it carries, and
  `mappedLength`, the whole pages mapped. The rounding happens once, in `NFKSharedMemoryRoundedLength()`,
  when a region is created.
- An attachment entry's `"length"` is required and is the only length a reader uses. The decoder refuses
  an entry longer than its region (a shorter one is the page rounding) and slices exactly that many
  bytes, without copying, before any reader sees them. Pixel-buffer geometry is checked as whole numbers
  and every plane must fit inside that slice before CoreVideo receives a pointer.

## Host side

`NFKPluginProcess` supervises one plugin's process, `NFKPluginBackend` is one of its models, and the
private `NFKPluginToolProxy` is one of its tools. `NFKPluginProcessPrivate.h` holds what they share: the
connection to a launched process (`NFKPluginConnection`), the launcher seam, and the backend's private
initializers.

### States

| from | event | to |
| --- | --- | --- |
| NotRunning | a start, or the first request | Starting |
| Starting | hello and describe pass | Ready |
| Starting | any check below fails | Failed |
| Ready | the channel closes without the supervisor closing it | NotRunning |
| Ready | the `crashLimit`-th such close within `crashLimitInterval` | Failed |
| Ready | idle exit, or stop | NotRunning |
| any | restart | Restarting, then Ready or Failed |

- A start, in order: the code checks → the launch → the hand-off within `handshakeTimeout` → the peer
  check → the plugin host's verification within `verificationTimeout` (`objc`, `c-abi`) → hello within
  `handshakeTimeout` → describe. Each failure leaves Failed with `failureError`:
  - a refused signature → the policy's error, before any launch;
  - the launch → `NFKPluginErrorLaunchFailed`, the launcher's error under `NSUnderlyingErrorKey`;
  - no hand-off, a hello error, another plugin identifier or version → `NFKPluginErrorHandshakeFailed`;
  - a verification, hello, or describe that misses its deadline → `NFKPluginErrorTimedOut`, naming the
    step and the limit;
  - another major protocol version → `NFKPluginErrorProtocolUnsupported`;
  - a manifest model `describe` does not list → `NFKPluginErrorUnknownModel`.
- A Failed process refuses every request with `failureError` and launches nothing. `startWithError:`
  clears the failure and the crash count. A restart launches regardless.
- `stateHandler` and `logHandler` run on the process's serial event queue, in the order of the changes.

### Code checks

- Before a launch: the bundle under `signaturePolicy`, and for `objc` and `c-abi` the host executable
  under the host policy. The host policy is `signaturePolicy` without its team and requirement, which
  name the plugin's signer; the host is the app's helper, signed by the app's team.
- After the hand-off: the peer's `LOCAL_PEERPID` equals the launched process's identifier, and its audit
  token passes the policy (`executable`) or the host policy (`objc`, `c-abi`).

### Launching

- The default launcher runs `NFKSubprocess` with `INFERKIT_PLUGIN_CHANNEL_FD=3`, the child end of the
  socket pair as `ownerInheritedDescriptors[3]`, `QOS_CLASS_USER_INITIATED`, and restart policy Never.
  The supervisor relaunches on the next request instead.
- `objc`, `c-abi` → `<host> --bundle <path>`, plus every field of the policy: `--team-id <team>`,
  `--requirement <requirement>`, `--allow-ad-hoc`, and `--allow-unsigned` when `requiresValidSignature`
  is NO. The host enforces exactly the app's policy. It honors `--allow-unsigned` only beside
  `--allow-ad-hoc` (the development policy, for local development) and otherwise exits 64 with the reason.
- The plugin host sends its channel frame first, then checks the bundle's ownership and verifies its
  signature, then sends `{"type": "event", "event": "verified"}`, then reads hello. The supervisor sends
  hello at once and waits `verificationTimeout` (120 s) for that event, then `handshakeTimeout` for the
  reply. A reply that comes first ends both waits. An `executable` plugin verifies nothing and has
  `handshakeTimeout` alone.
- Verification hashes every file the bundle's seal covers, on every launch. A plugin keeps large
  weights outside its bundle, in a download cache.
- `executable` → `Contents/MacOS/<entry.executable>`, which must stay inside the bundle, or the bundle's
  own executable, with `entry.arguments`.
- `launcherFactory` (private) makes the launcher per launch. A test returns an in-test peer that adopts
  `NFKPluginLauncher`, performs the hand-off on the descriptor it receives, and answers the protocol:
  `Tests/InferKitTests/NFKPluginTestPeer`. A launcher that implements
  `validatePeerWithAuditToken:process:error:` replaces the peer check.

### Requests and generations

- A synchronous request goes through `performWithReservation:block:error:` on the caller's thread: it
  waits its turn behind a restart, starts the process when none runs, and runs the block with the live
  connection. The block sends and returns; the reply arrives on the channel's handler queue.
- A submitted job goes through `performAsynchronouslyWithBlock:completion:`. During a restart it takes
  its place in order at the call and holds no thread while it waits. Outside a restart its block runs on
  the process's private serial submission queue. Neither path waits for a thread from the shared pool,
  which a loaded test run can exhaust.
- A build and a preparation each wait for their reply up to `buildTimeout` (300 s, since a plugin may load
  large weights in either step). One that does not answer in time fails the request that needed it with
  `NFKPluginErrorTimedOut`, naming the plugin and the model. The request stops counting as in flight, so
  it holds neither an idle exit nor a restart's drain. A late build has its backend released; a late
  preparation's load is still recorded.
- Runs and tool calls have no deadline, since a model may run for minutes. Cancelling a submitted job
  sends `cancel`, `stopWithCompletionHandler:` fails them with `NFKPluginErrorStopped`, and the process's
  exit fails them with `NFKPluginErrorProcessExited`.
- Each launched process is a connection with a generation, counted from 1 per supervisor. A backend
  records the generation of its build and builds again on the next connection before its next request,
  from its stored model identifier and options.
- A connection that closes fails its requests in flight: the supervisor's own close with the error it
  names (`NFKPluginErrorStopped`), any other close with `NFKPluginErrorProcessExited`. That error's
  description names the exit status or the signal, and `NSLocalizedFailureReasonErrorKey` holds the
  last 2048 characters of standard error.

### Restart

1. The call starts holding requests before it returns, so a request made after it waits.
2. Restarting. The old connection takes no new requests. Its requests in flight get `restartDrainTimeout`
   to finish; then each run receives `cancel` and the rest get 5 seconds. A run that answers the cancel
   fails with `NSUserCancelledError`; one still in flight after the 5 seconds fails with
   `NFKPluginErrorTimedOut`.
3. `shutdown` (its reply awaited up to 2 seconds), the channel closes, and the launcher stops.
4. The new manifest replaces the old. `idleExitInterval` follows it unless the app changed it. The crash
   count clears. The new version launches.
5. The held requests are admitted one at a time, in order. A held asynchronous request (a submitted
   job) runs in its turn on the restart's queue. A held synchronous caller runs on its own thread, and
   the restart waits for it to send before it admits the next. Each builds its backend, within
   `buildTimeout`, and sends its run before the next is admitted, so a build that never answers fails its
   request and frees the queue for the next request, a later restart, or a stop. A launch failure fails
   each held request with its reason.

Restarts are sequential. Requests held while a second restart waits go to the last one.

### Idle exit

- A process is idle with no caller in a request's turn, no request in flight, and no live backend
  handle. A handle lives from its build until the backend is released.
- The effective interval is the server's formula over `idleExitInterval`: below the onset (0.6) the
  interval itself; from the onset to full pressure, linear down to the shortest interval (10 s). An
  infinite interval starts from 600 s under pressure. The reading is
  `-[NFKServedHostStatus memoryPressureFraction]`; `memoryPressureProvider`, `memoryPressureOnset`, and
  `shortestIdleExitInterval` are private settings a test sets.
- A timer checks at a fifth of the shorter of the interval and the shortest interval, between 0.05 and
  1 second. An idle exit sends `shutdown`, closes the channel, stops the launcher, and leaves NotRunning.

### Stop

`stopWithCompletionHandler:` fails the requests in flight with `NFKPluginErrorStopped` at once, then sends
`shutdown`, closes the channel, and stops the launcher with NFKSubprocess's escalation. A Failed process
stays Failed. Stops and restarts run in order on one serial queue.

### Backend

- Keys come from the manifest's descriptor: inputs and outputs by port key, parameters by the keys of
  request-stage options. `modelInfo` is the family as `NFKModelInfoArchitecture`, the default variant's
  byte counts, the build's `modelInfo`, and the measured bytes once known.
- `prepareWithError:` builds and sends `prepare`, both within `buildTimeout`; its result sets
  `isLoaded`, `residentByteCount`, and `peakByteCount`.
- A run encodes its request with an `NFKPluginAttachmentWriter` at the connection's inline limit (the
  smaller of 64 KiB and a positive `limits.inlineBytes` in the hello reply). Its `outputDirectory` is
  `NSTemporaryDirectory()/InferKit/plugin-runs/<UUID>`, symbolic links resolved, mode 0700. The reply
  decodes through an `NFKPluginAttachmentReader` that copies file regions and accepts file references
  inside that directory only. Each result and partial decoded from the run holds the directory, which is
  removed when the last of them is released.
- `submitInferenceJobForRequest:` takes its place behind a restart when called, and otherwise runs on
  the process's submission queue. The job relays each
  `progress` message with its partial result. Cancelling the job sends `cancel` with the run's
  identifier, or skips the run when it has not been sent.
- An error reply keeps its domain, code, and string and number user info. A `{message, domain, code}`
  object, as a C plugin reports, keeps its domain and code too. `"cancelled": true` is
  `NSCocoaErrorDomain` `NSUserCancelledError`.
- `unload` sends `unload` to a Ready process that holds this backend's build and starts nothing.
  `isLoaded` follows the build, prepare, and status results and the `loaded` and `unloaded` events, and
  is NO once the process that built the backend is gone. `peakByteCount` is the most the backend has
  held: each build or prepare result, status, and event raises it to its `peakBytes` and to the resident
  bytes, and nothing lowers it, so a message that omits `peakBytes` or reports less keeps it. The
  executable fixture once sent `loaded` without `peakBytes`, and the host read only `loaded` from a build
  result, so that backend reported a peak of 0 (`NFKPluginCompletenessTests`, 4 of 4).
- `backendStatus` answers the last status at once and asks a Ready process for a new one at most once
  a second.
- Releasing the backend sends `release` for its handle.
- The in-process form (`initWithPluginIdentifier:descriptor:options:pluginBackend:`, private, for the
  registry) forwards runs, jobs, preparation, and residency to the plugin's own backend and takes its keys
  from the descriptor.

### Tools

`tools` makes an `NFKPluginToolProxy` per manifest declaration, so listing starts nothing. A call sends
`callTool` through the same turn as a run. The reply's `toolResult` is the wire coding's tagged tool
result, a `{text, isError}` object, or a string. An error reply keeps its domain and code.

## Loaders

### Manifest

- `+manifestWithBundleURL:error:` reads `Contents/Resources/inferkit-plugin.json` and
  `Contents/Info.plist` and loads no code. `CFBundleIdentifier` must equal `id` and
  `CFBundleShortVersionString` must equal `version`; the executable (`entry.executable`, else
  `CFBundleExecutable`) must be a file in `Contents/MacOS`. Each disagreement →
  `NFKPluginErrorManifestMismatch`.
- `schemaVersion` above 1 → `NFKPluginErrorUnsupportedManifestVersion`. An entry kind other than `objc`,
  `c-abi`, `executable` → `NFKPluginErrorIncompatible`. Every other problem →
  `NFKPluginErrorMalformedManifest`, naming the field by its path ("compatibility.inferkitAPI").
- Required: `schemaVersion`, `kind` ("inferkit.plugin"), `id` (letters, digits, ".", "-", "_"),
  `version`, `compatibility.inferkitAPI` ("major.minor"), `entry.kind`, and `compatibility.pluginABI`
  for `c-abi`. Defaults: `displayName` → `id`; `protocol` → "1.0"; `idleExitSeconds` → 300;
  `allowsInProcess` → NO; `maximumConcurrentRuns` → 1; a tool's `description` → "" and `parameters` →
  an empty object schema.
- A model's `engine` defaults to "plugin", and any other engine is refused. A model identifier holds no
  "/", since the catalog joins `<pluginId>/<modelId>`. Models, tools, and options are unique by
  identifier, name, and key.
- `entry.executable` is relative and stays inside `Contents/MacOS`: no leading "/", no "." or "..".
- `JSONDataWithError:` writes sorted, pretty-printed keys, every default written out, empty lists left
  out. A manifest equals another when the two dictionaries and bundle URLs are equal.
- `extensions` is the manifest's `"extensions"` object, written back as it came. Other unknown fields
  are ignored.
- The private `-checkHostCompatibilityWithError:` (`NFKPluginManifestPrivate.h`) is what a loader runs:
  the API version (same major, minor at least the manifest's), the protocol major, the plugin ABI major
  for `c-abi`, `minimumOS.macOS`, and the architecture when `architectures` is not empty. The same
  header gives `executableURL`, `bundleExecutableName`, and `modelWithIdentifier:`.

### Signature policy

- Order: signed at all → valid → ad hoc allowed → team → requirement.
  - unsigned or invalid → `NFKPluginErrorSignatureInvalid`, unless `requiresValidSignature` is NO and
    the policy names no team and no requirement, which then accepts it;
  - ad hoc (`kSecCodeSignatureAdhoc`) under a policy that refuses it → `NFKPluginErrorSignatureInvalid`;
  - a `teamIdentifier` that differs from the signature's → `NFKPluginErrorTeamMismatch`;
  - the requirement: `anchor apple generic and certificate leaf[subject.OU] = "<team>"` when a team is
    pinned, since the team identifier in signing information is the signer's own claim, joined with
    `(<requirementString>)`. Failing it → `NFKPluginErrorTeamMismatch`; a requirement that does not
    compile → `NFKPluginErrorSignatureInvalid`.
- Code on disk: `SecStaticCodeCheckValidityWithErrors` with `kSecCSCheckAllArchitectures |
  kSecCSStrictValidate | kSecCSCheckNestedCode`. A resource edited after signing fails with -67054
  (`errSecCSBadResource`), measured on an ad hoc bundle.
- The cache holds successes only, keyed by cdhash, path, policy, and a fingerprint of every file under
  the path: its size, inode, mode, and modification and status-change times. Reading the cdhash hashes
  only the code directory, and the fingerprint costs a `stat` per file, so a cached validation of a
  bundle with gigabytes of weights is cheap. A later edit changes the status-change time, which a
  process cannot set back, so it misses the cache and validates again.
- A running process: `SecCodeCopyGuestWithAttributes` with `kSecGuestAttributeAudit`, then
  `SecCodeCheckValidityWithErrors`, through the same order. No cache.
- Measured: the `xctest` runner under Xcode 27 is signed ad hoc (`flags=0x2`), so the test of the
  running process uses the development policy.

### Loader

`+loadPluginAtURL:signaturePolicy:error:` runs, in order:

1. The manifest from the bundle.
2. `executable` → `NFKPluginErrorNotTrusted`.
3. `-checkHostCompatibilityWithError:`.
4. Ownership (`NFKPluginCheckBundleOwnership`): the directory that holds the bundle, the bundle, and
   every item inside it, each by `lstat`, must be owned by this user or root and writable by neither
   group nor others. A failure → `NFKPluginErrorLoadFailed` naming the path and the reason ("is writable
   by its group"). Symbolic links are checked for their owner only.
5. The registry, by plugin identifier: the same resolved bundle path and version → the same
   `NFKLoadedPlugin`, after the signature passes the caller's policy, so a caller with a stricter policy
   never receives a loaded plugin it would refuse; another version or another copy →
   `NFKPluginErrorLoadFailed`. A load that failed after code entered the process answers the same error
   again.
6. The signature under the policy, immediately before the code loads and never earlier.
7. The code:
   - `objc` → `NSBundle` `loadAndReturnError:`. `entry.executable`, when set, must be the bundle's
     executable. The image is found among dyld's images by resolved path, and
     `objc_copyClassNamesForImage` refuses any class whose name begins with "NFK"
     (`NFKPluginErrorDuplicateRuntime`, naming them). The principal class (`entry.principalClass`,
     else `NSPrincipalClass`) must come from that image and adopt `NFKPlugin`; `initWithContext:` runs
     inside `@try`.
   - `c-abi` → `dlopen(RTLD_NOW | RTLD_LOCAL)`, the same duplicate check on the image, then
     `dlsym(entry.symbol ?: "nfk_plugin_entry")` and the C adapter.

- The window between check and load. The kernel does not check a library's signature at load in a
  process with library validation off, which the host's entitlement turns off, so a file replaced after
  step 6 and before step 7 would load. The ownership rule closes that window to every account but this
  user and root: neither can be another app's sandboxed process, and either could change the app
  itself.
- The context (`NFKPluginHostContext`, `NFKPluginLoaderPrivate.h`): a temporary directory
  `$TMPDIR/InferKit/Plugins/<id>.<UUID>` (0700), `hostAPIVersion` = `NFKInferKit.APIVersion`, the
  options, `runsInHostProcess`, and a log handler, else the unified log (subsystem "com.inferkit",
  category "plugin").
- The public entry loads with empty options and `runsInHostProcess` YES. The private
  `+loadPluginAtURL:signaturePolicy:options:runsInHostProcess:logHandler:error:` is the plugin host's.
- `NFKLoadedPlugin` refuses a model the manifest does not list (`NFKPluginErrorUnknownModel`).
- Nothing is ever closed with `dlclose`, and a refused bundle's code stays in the process.
- In-process tests load the fixtures at 1.0.0 only; 2.0.0 is the second-version refusal. A test that
  needs another identity loads a re-signed copy under an identifier of its own. `exitAtStep` runs only
  out of process. The duplicate-runtime fixture loads only in a child process, the plugin host
  (`NFKPluginServiceTests`): loading it in the test process leaves two `NFKInferenceRequest` classes,
  and the runtime keeps an arbitrary one for every later test.

### C adapter

`NFKPluginCAdapter` (private) is a C plugin as an `NFKPlugin`, and `NFKPluginCBackend` one of its
backends.

- The entry receives `NFK_PLUGIN_ABI_VERSION` and the host table, which is an instance variable of the
  adapter, so its address holds for the plugin's life. `options_json` and `temporary_directory` point
  into data the adapter keeps.
- The table is refused, in order: entry status `INCOMPATIBLE` → `NFKPluginErrorIncompatible`; another
  status or no table → `NFKPluginErrorLoadFailed`; another major `abi_version`, read before the size since
  another major has another layout → `NFKPluginErrorIncompatible`; `struct_size` below ABI 1.0's →
  `NFKPluginErrorIncompatible`; a required function missing → `NFKPluginErrorLoadFailed`, after calling
  `shutdown` when it exists. `describe` runs once, and its models default to the "plugin" engine.
- Ownership, by allocation:
  - a request → the adapter encodes it in the attachment mode with a sink of `NSMutableData` regions,
    one `NFK_PLUGIN_BUFFER_BYTES` buffer each, and value classes as JSON; files pass by path. The JSON, the buffers, and the array
    stay alive until `run` returns.
  - a result → wrapped without a copy. Each `NSData` over a buffer retains an owner object, which calls
    `free_payload` when the last value over the payload is released. An error path calls
    `free_payload` at once on any payload the plugin returned.
  - a partial in `report_progress` → copied during the call; the plugin keeps ownership.
  - every string (`describe`, errors, `describe_backend`, `call_tool`) → copied, then `free_string`.
  - a backend → `destroy_backend` when the `NFKPluginCBackend` is released; runs in flight retain it.
- Buffers stride by the first buffer's `struct_size`, so a host reads a later minor's longer buffers;
  a buffer shorter than ABI 1.0's is refused. `SURFACE` buffers are locked read-only for the life of
  the wrapping data; `FILE` buffers are read from the path or the descriptor.
- `run` is limited to `maximumConcurrentRuns` per backend by `NFKPluginRunSlots` (`NFKPluginWire.h`):
  a run waiting for a slot is a block in a FIFO and holds no thread, and the run that ends starts the
  next. A synchronous `runInferenceForRequest:error:` blocks only its caller while it waits.
  `is_cancelled` reads the run's flag. `cancel`
  runs under the run's lock and only while the run is inside `run`, so the plugin never sees a run that
  returned. `output_directory` makes the run's directory on first call: the host's per-run directory
  under the service, else `<temporary directory>/runs/<UUID>`, which stays with the result's files.
- Status mapping when the error JSON names no domain and code: `CANCELLED` → `NSUserCancelledError`,
  `UNKNOWN_MODEL`, `UNKNOWN_TOOL`, `INCOMPATIBLE` → the plugin codes, `MISSING_INPUT` →
  `kNFKError_InferenceMissingInput`, `UNSUPPORTED` and `INVALID_ARGUMENT` →
  `kNFKError_InferenceUnsupported`, `NOT_READY` → `kNFKError_InferenceNotReady`, anything else →
  `kNFKError_InferenceBackendFailure`.
- A tool result JSON is the tagged form or `{text, isError, outputs, images}` without the tag; a JSON
  string or other text is the result's text.
- `willUnload` calls `shutdown` once. After it no backend calls the plugin, and the host table's
  `log` drops messages.

## Plugin side

### Service

`NFKPluginService` answers the protocol for one `id<NFKPlugin>`.

- `runWithChannelDescriptor:error:` performs the hand-off on the descriptor it is given, serves on its
  own socket, and blocks until `shutdown` or end of file (YES), or a protocol violation (NO with it).
  At the end it cancels the runs in flight, waits up to `drainTimeout` (5 s, private) for them, and
  releases every backend.
- Before hello, only `hello`, `ping`, and `shutdown` are answered; anything else →
  `NFKPluginErrorProtocolViolation`. A second hello → `NFKPluginErrorProtocolViolation`.
- Handler queue (fast answers): `hello`, `status`, `release`, `ping`, `shutdown`, `cancel`, and the
  checks of every request. Work queue (concurrent, user-initiated): `describe`, `build`, `prepare`,
  `run`, `unload`, `callTool`.
- `build` refuses a model the manifest does not list (`NFKPluginErrorUnknownModel`). Its reply carries
  `inputs`, `parameters`, and `outputs` sorted, each only when the backend implements the key set, and
  `modelInfo` and `backendStatus` keep only what JSON carries.
- `run` decodes its request with an `NFKPluginAttachmentReader` that maps regions as they are and copies
  descriptor-passed files into `outputDirectory`. A backend holds `maximumConcurrentRuns` slots in an
  `NFKPluginRunSlots`: a run waiting for one holds no thread, and 64 runs queued on one slot keep the
  process within 16 threads of where it started (`NFKPluginServiceTests`, `NFKPluginLoaderTests`). A
  backend with a job relays progress and ends the run in the job's completion handler, so a running job
  holds no thread either; a C backend receives the run's `outputDirectory`; a synchronous backend runs
  on the work queue and cannot be interrupted, so its `cancel` is answered at once and its result
  dropped.
- A result file outside `outputDirectory` is copied into it (the file system clones it on APFS), so a
  host that reads only that directory receives it. Audio and video assets keep their metadata.
- A load event goes out whenever `isLoaded` differs from what the host was last told: after `build`,
  `prepare`, a run, and `unload`.
- A reply too large for a frame is answered with that error instead.
- Plugin log messages travel as `log`; with no channel they go to standard error.
- The private `-initWithManifest:loader:` (`NFKPluginServicePrivate.h`) loads the plugin when hello
  arrives, with hello's `options`, and a load failure answers hello with its error, so
  `NFKPluginErrorDuplicateRuntime` or a signature refusal reaches the host with its domain and code.
  `plugin` is nil until then.
- hello's optional `sharedMemoryPrefix` names a sandboxed host's app group for result regions.

### `inferkit-plugin-host`

`NFKPluginHostMain(argc, argv)`:

| outcome | exit status |
| --- | --- |
| arguments other than `--bundle <path> [--team-id <team>] [--requirement <requirement>] [--allow-ad-hoc [--allow-unsigned]]` | 64 (`EX_USAGE`) |
| `--allow-unsigned` without `--allow-ad-hoc` | 64 (`EX_USAGE`), with the reason |
| no open descriptor `INFERKIT_PLUGIN_CHANNEL_FD` (default 3) | 71 (`EX_OSERR`) |
| an unreadable manifest | 65 (`EX_DATAERR`) |
| serving ended by a protocol violation | 76 (`EX_PROTOCOL`) |
| `shutdown` or end of file | 0 |

- Order: read the manifest → send the channel frame → check ownership and verify the signature (the
  service's private `verifier` hook) → send the `verified` event → read hello → load. A refused check
  answers hello with its error.
- The load goes through the private loader entry with `runsInHostProcess` NO and the policy the flags
  name. The loader checks ownership and the signature again immediately before the code loads; the
  verifier's success is cached, so the second check hashes nothing. A manifest that changed since the
  host read it → `NFKPluginErrorManifestMismatch`.
- On exit it calls `willUnload` (a C plugin's `shutdown`) and removes the context's temporary
  directory.
- Measured end to end (2026-10-15) with a Python host driving the built program over a socket pair: the
  `objc` and `c-abi` fixtures answer every message type; a 100 KiB attachment returns inverted through
  shared memory; an ad hoc bundle without `--allow-ad-hoc` answers hello with
  `NFKPluginErrorSignatureInvalid`, `--team-id` with `NFKPluginErrorTeamMismatch`, an edited manifest
  with -67054, and a bundle defining `NFKInferenceRequest` with `NFKPluginErrorDuplicateRuntime`. After
  the verification moved ahead of hello: a group-writable `Info.plist` answers hello with
  `NFKPluginErrorLoadFailed` naming the file, and `--allow-unsigned` alone exits 64.

## Registry

`NFKPluginRegistry` lists plugins in one `NFKModelCatalog` and makes their backends. Its state (entries
by plugin identifier, catalog identifier → plugin identifier, the trusted set) lives under one lock;
process starts, restarts, and in-process loads run outside it.

### Listing

`addPluginAtURL:error:` runs, in order:

1. `+[NFKPluginManifest manifestWithBundleURL:error:]`.
2. `NFKPluginCheckBundleOwnership`: the folder that holds the bundle, the bundle, and every item in it.
   For `addPluginsInDirectoryURL:errors:` that folder is the plugins folder, so the rule is checked up
   to it and no further. A failure → `NFKPluginErrorLoadFailed`, as the loader reports it.
3. `-checkHostCompatibilityWithError:` → `NFKPluginErrorIncompatible`.
4. The catalog descriptors. A catalog identifier the catalog already holds and this plugin did not add →
   `NFKPluginErrorIdentifierInUse`, and none of the plugin's models are listed. The same plugin
   identifier from another resolved bundle path → `NFKPluginErrorIdentifierInUse`; from the same path →
   a reload.

No signature is checked and no code loads. `addPluginsInDirectoryURL:errors:` takes the
`.inferkitplugin` entries directly inside the folder in name order, hidden files skipped.

Each descriptor is the manifest model's `dictionaryRepresentation` with `id` → `<pluginId>/<modelId>`,
`registryName` → the model identifier, `engine` → `plugin`, a top-level `plugin` key (which the reader
keeps as `extensions.plugin`) → `{id, version, kind}`, and `providerClass` removed, so the catalog never
looks up a class a plugin's manifest names.

### The engine builder

- The first registry made registers one `NFKPluginEngineBuilder` (private) for engine `plugin`. The
  builder registry is global, so the builder finds the registry by the build request's catalog among the
  live registries (a weak hash table) and asks the one that lists the identifier.
- It passes `request.loadOptions` to the plugin's build: every load option, defaults filled, the variant
  under `variant` when the model has several, and never `NFKModelLoadOptionIgnoreMemoryBudget`. Request
  options reach the plugin as the `NFKDefaultedBackend`'s defaults.
- A `plugin` descriptor no registry lists, such as one from `descriptorWithIdentifier:derivedFromBackend:`,
  builds from its `providerClass` through `NFKDynamicBackend` as the catalog's last resort does, else
  fails with `NFKModelCatalogErrorNoBuilder` and the catalog's own message.

### Backends, trust, and tools

- An untrusted plugin's backend comes from its `NFKPluginProcess`, made on first need with the registry's
  `signaturePolicy` and `hostExecutableURL`. Making it starts nothing. Setting `signaturePolicy` sets it
  on every existing process; `hostExecutableURL` applies to processes made afterward.
- A trusted plugin loads through the loader's private entry (`options` empty, `runsInHostProcess` YES)
  under `signaturePolicy`, and its backend is wrapped in the in-process `NFKPluginBackend`. A manifest
  without `allowsInProcess`, which includes every `executable` manifest → `NFKPluginErrorNotTrusted`
  before the loader runs.
- Trust is an in-memory set read at each backend request, which is "the next load". Withdrawing it sends
  the next backend to a process; the loaded code stays.
- `tools`: a trusted in-process plugin's tools are handler tools from the manifest declarations that load
  the plugin on their first call and run its tool of the same name; every other plugin's are its process's
  `NFKPluginToolProxy`s.

### Autorelease pools

A plugin backend's handle lives until the `NFKPluginBackend` is freed, and its process idles only once no
handle lives, so an autoreleased backend keeps a process running. Three paths autoreleased one into a pool
that drains only when its thread exits, and `NFKPluginRegistryTests` (the server idle test) failed 3/3 on
the first:

- `-[NFKInferenceServer unloadIdleModels]` took the backend from inside `@synchronized`, which returns it
  autoreleased into the caller's pool: the test's main thread, or the residency queue's thread. The take
  now returns outside the lock, and one `@autoreleasepool` spans the take and the release.
- A provider build and a run on a global queue: each block now drains its own pool.
  `-[NFKPluginProcess backendForModelIdentifier:options:error:]` returns its new backend from inside a
  `for` loop, so the return autoreleases it.
- The channel's handler queue: `routeEvent:` reads the backend from the connection's weak map table, and
  a weak read autoreleases. The channel's and the supervisor's queues use
  `DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM`.

The test now also waits for the backend itself to be freed after the unload, which separates a held
backend from an idle timer that does not fire.

### Tool names

`checkToolNamesOfManifest:error:` runs on every add and reload, under the lock: a tool name another
listed plugin declares → `NFKPluginErrorIdentifierInUse`, naming the tool and the holder. A plugin's
own entry is skipped, so a reload never conflicts with itself, and a removal frees the names. The
fixtures all declare `echo`, so a test that lists two of them drops one's tools from a copy's manifest.

### Plugin option values

- `setOptionValues:forPluginIdentifier:error:` resolves the values through a carrier descriptor whose
  options are the manifest's top-level options, with `resolvedOptions:stage:nil error:`. The error is the
  catalog's (`NFKModelCatalogErrorDomain`, `NFKModelCatalogErrorInvalidOptions`, `NFKModelOptionProblemsKey`),
  so a user interface shows plugin and model option problems alike. The stored dictionary holds the
  given keys only, converted; the plugin fills its own defaults.
- The values live in a dictionary by plugin identifier beside the trusted set, kept through a removal.
  Setting needs a listed plugin, since the manifest declares the options.
- A process takes them as `pluginOptions` (hello's `options`) when the registry makes it, and on each set.
  A set then hands the process its own manifest through `handManifest:toProcess:completion:`: a stopped
  process changes nothing else, and a running one restarts.
- An in-process load passes them to the loader's private entry. A plugin already loaded keeps its first
  context, so a later set does not reach it until the app restarts.
- `optionValues:keptByManifest:` runs on every add of a listed identifier and every reload, before the
  lock: each stored key resolves alone against the new manifest's options, and a key that fails (an
  unknown key, or a value the new declaration refuses) is dropped. The kept values replace the stored
  ones and become the process's `pluginOptions` before the manifest reaches it, so the restart's hello
  carries them. A plugin with no stored values keeps none.

### Shared-memory prefix

- `NFKPluginProcess.sharedMemoryPrefix` (public) set → hello's `capabilities` is `["file-descriptors"]`
  and hello carries `sharedMemoryPrefix`; nil → `capabilities` is empty and the key is absent, as before.
  `NFKPluginCapabilityFileDescriptors` moved from the service into `NFKPluginWire`.
- The backend's request writer takes the prefix as `namePrefix`, and passes files as descriptors when
  the hello reply's `capabilities` lists `file-descriptors`. The service already honored both.
- A POSIX shared-memory name holds at most 31 characters and a region falls back below 10 random
  characters, so an app group identifier of 21 characters or more sends every region through the
  temporary-file fallback. `Docs/plugins.md` tells apps to keep it short.
- The registry's prefix is set on each process it makes and on every existing one when it changes.

### Reload and removal

- `reloadPluginWithIdentifier:error:` reads the bundle at the listed URL again through the same checks.
  Another plugin identifier there → `NFKPluginErrorManifestMismatch`. A plugin loaded in process at
  another version → `NFKPluginErrorLoadFailed`, since loaded code never unloads. A failure changes
  nothing.
- The new manifest goes to the process through `-replaceManifestWhileStopped:` (private,
  `NFKPluginProcessPrivate.h`) when no process runs, no restart is pending, and the start lock is free
  (`tryLock`, since a start holds it for the whole launch). A Failed process returns to NotRunning. Any
  other state restarts with `restartWithManifest:completionHandler:`, which returns at once; the
  synchronous reload does not wait for the restart. `reloadPluginWithIdentifier:completionHandler:`
  passes the restart's completion through, or calls it at once for a stopped process, always on a global
  user-initiated queue; a reload that fails before the hand-off calls it with that error.
- `removePluginWithIdentifier:` removes the descriptors and drops the registry's process. Backends and
  tool proxies hold the process strongly, and its `dealloc` closes the channel and stops the launcher, so
  the process ends once the last of them is released.

### Host search

`hostExecutableURL` unset searches on each read: `Contents/Helpers/inferkit-plugin-host` in the main
bundle, `-[NSBundle URLForAuxiliaryExecutable:]`, beside the main executable, then beside the image
that contains InferKit (`dladdr`). Under `swift test` none of these is the host: the image is the test
bundle's executable inside `.xctest/Contents/MacOS`, and the host is beside the `.xctest`. Tests set the
URL from `NFKPluginFixtures.hostExecutableURL`.

### Tests

`NFKPluginRegistryTests` runs over the fixtures with the development policy. The in-process test loads
the cached 1.0.0 fixtures, as the loader tests do. The reload test copies the 1.0.0 C fixture into a
folder of its own and replaces it with 2.0.0 there while a `slow` run is in flight. The server test
hosts the C fixture through `addBackendProvider:forModelName:` with `idleUnloadInterval` 0.2 s and the
process's `idleExitInterval` 0.5 s, unloads it with `unloadIdleModels`, and waits for the process to
exit. The options test in process loads a re-signed copy of the C fixture under an identifier of its own,
since the cached fixture may already be loaded with other options. The hello test drives
`NFKPluginProcess` with the in-test peer. Tests that launch skip when the host is not built beside the
tests.

## The C API

`NFKInferKitC.m` (in the core) implements `NFKInferKitC.h`. Consumer guide: `Docs/c-api.md`.

- Every handle is a bridged retained Objective-C object of a private class: `NFKCBackend`, `NFKCJob`,
  `NFKCResult`, `NFKCError`. Each reader and `*_free` checks the class first, so NULL or a handle of
  another type answers NULL or 0 and frees nothing.
- Every entry point runs inside `@autoreleasepool` and `@try`, since a C caller's thread may have no
  pool. An exception becomes `NFKInferenceErrorDomain` `kNFKError_InferenceBackendFailure`, "InferKit
  raised <name>: <reason>".
- `nfk_backend_create` → `-[NFKModelCatalog backendForModelIdentifier:options:error:]` on the shared
  catalog, its error passed through. NULL or empty options → nil; JSON that is not an object →
  `NFKModelCatalogErrorInvalidOptions`; a NULL identifier → `NFKModelCatalogErrorUnknownModel`.
- A request decodes through `NFKInferenceWireCoding` with an attachment source over the caller's
  `nfk_payload` (`NFKCPayloadSource`). The payload, its JSON, and its buffer array are read during the
  call; bytes and surfaces are wrapped with `initWithBytesNoCopy:deallocator:`. Each wrapper enters a
  dispatch group and leaves it in its deallocator, after a surface's read-only unlock.
- Runs: `nfk_backend_run` calls `runInferenceForRequest:error:` on the caller's thread.
  `nfk_backend_submit` goes through `NFKInferenceSubmit` on the user-initiated global queue.
- Callbacks run on the job's serial queue (`QOS_CLASS_USER_INITIATED`). A job with a result completes
  at once. A job without one, cancelled or failed, completes from `dispatch_group_notify` on the
  request's group, so a cancelled run's completion waits until the backend has released every wrapper
  over the caller's memory. A result waits for nothing, since a backend that echoes its input keeps a
  wrapper alive inside the result.
- `NFKCJob` keeps `freed`, `delivering`, and `completed` under an `NSCondition`. A callback starts only
  while neither freed nor completed. `nfk_job_free` sets `freed`, waits while a callback runs unless it
  is called on the job's own queue (`dispatch_get_specific`), cancels a run that has not completed, and
  releases the handle. The handlers on the `NFKInferenceJob` hold the `NFKCJob` weakly, so a freed job
  deallocates while its run continues.
- A result and a partial encode with `encodesValueObjectsAsJSON` YES, so the value classes are tagged
  JSON objects (`NFKWireValueObjects.h` lists the fields) and never keyed archives. The plugin channel
  sets it too; the HTTP route leaves it off and keeps its bytes. The decoder reads the JSON form in every
  mode.
  Surveyed 2026-10-15 over the core and the three companions: every output value is JSON, bytes, a
  pixel buffer, an image, a texture, a multi-array, a file or asset, a value class, or a tool invocation,
  except Core ML's `MLFeatureValue` (`NFKVisionCoreMLBackend`) and a classifier's number-keyed dictionary
  (`NFKCoreMLBackend`), and the half and float textures `NFKCoreMLBackend` wraps over an image output's
  IOSurface, which PNG cannot hold. The wire coding now writes the first as the value it holds, the
  second as `numberKeyedDictionary`, and the third as a `pixelBuffer` of its format; before, each failed
  the encode on every route. Landmarks, contours, and
  transforms are `NFKKeypoint` arrays or numbers under `structured`, so nothing in an output is an
  `NSValue`.
- `nfk_backend_prepare` → `NFKInferencePrepare`. `nfk_error_copy_details_json` converts the error's user
  info: an `NSError` → `{domain, code, message, details}`, an `NFKModelOptionProblem` → `{key, code,
  message, value}`, an object with `dictionaryRepresentation` (the memory estimate) → it, JSON values as
  themselves, anything else left out; `NSLocalizedDescriptionKey` is left out because the message holds it.
- A result encodes with `inlineByteLimit` 0, so every value with bytes is an attachment, into an
  `NSMutableData` region per attachment. The `nfk_buffer` table, the NUL-terminated JSON, the
  Objective-C result (whose associated objects may own files), and the run's file directory live as long
  as the `NFKCResult`. JSON is written with sorted keys and unescaped slashes.
- A partial result is encoded once per distinct object the job reports and passed as a borrowed
  `NFKCResult` valid during the callback.
- Files decoded from a request (inline base64, or a descriptor-only file buffer) land in
  `$TMPDIR/InferKit/c-api/<UUID>`, removed when the last of the request, the job, and the result
  releases it. A file reference by path is taken as it is.
- `nfk_plugins_add_directory` → `-[NFKPluginRegistry addPluginsInDirectoryURL:errors:]` on the shared
  registry; the count added, and the first error. On iOS and tvOS → -1 with
  `kNFKError_InferenceUnsupported`, the registry call compiled out under `TARGET_OS_OSX`.
- The completion receives its handles through `NFKCDeliverCompletion` (`NFKInferKitCDelivery.m`), whose
  parameters are `CF_CONSUMED`. The static analyzer treats a +1 reference passed through a C function
  pointer as still owned and reports a leak; the consuming declaration records the transfer. The
  function sits in a file of its own because the analyzer inlines a body in the same file and reports
  the leak again (measured 2026-10-15 on single-file probes).
- `nfk_release_version` duplicates `NFKInferKit.releaseVersion` once into static storage.
- Images reach a C caller as a `pixelBuffer` buffer, or as PNG bytes for a `CGImage` and an 8-bit
  Metal texture. Raw pixels for those two is a possible later addition (decided 2026-10-15: not now).
- `NFKInferKitCTests` builds `Tests/PluginFixtures/headers/inferkit_c_consumer.c` (C99) and
  `inferkit_cpp_consumer.cpp` (C++11) under `-pedantic -Werror` as bundles with
  `-undefined dynamic_lookup`, loads them with `dlopen`, and calls their run functions. Their `nfk_`
  references bind to the test bundle, which exports all 23 functions (`nm -gU`, 2026-10-15).

## DocC

`Tools/docc/build.sh` passes `NFKPluginABI.h` to `clang -extract-api` with `NFKInferKitC.h`, and the article
`CPluginInterface` (`Sources/InferKit/InferKit.docc/`) groups its symbols: the entry, the versions, the two
tables, payloads, and status and logging. The opaque `nfk_plugin_backend` and `nfk_plugin_run` have no page.

## Building plugins

A plugin is a `.inferkitplugin` bundle. These steps build one of each kind from the command line; the
sample plugins in `Tests/PluginFixtures/` follow them, and `NFKPluginFixtures` runs them at test time.

### The bundle

```
MyPlugin.inferkitplugin/
  Contents/
    Info.plist
    MacOS/MyPlugin                   the bundle, library, or program
    Resources/inferkit-plugin.json   the manifest
```

- `Info.plist` names the plugin: `CFBundleIdentifier` equals the manifest's `id`,
  `CFBundleShortVersionString` equals its `version`, and `CFBundleExecutable` names the file in
  `Contents/MacOS`. `CFBundlePackageType` is `BNDL`. An `objc` plugin also names its `NSPrincipalClass`.
- The manifest's `entry.executable` names the same file. `entry.principalClass` (`objc`) and
  `entry.symbol` (`c-abi`, default `nfk_plugin_entry`) are optional.
- The models in the manifest are model-catalog descriptors with engine `"plugin"`
  (`Docs/model-catalog.md`). The plugin's own `describe` answer lists the same model identifiers.

### Compiling

Each command runs from the InferKit source root. `-arch` names the machine's architecture, which the
manifest's `compatibility.architectures` lists, and `-mmacosx-version-min` matches its
`compatibility.minimumOS`.

- `objc`: a bundle that binds to the InferKit already in the loading process.

  ```
  xcrun clang -bundle -fobjc-arc -undefined dynamic_lookup \
      -I Sources/InferKit/include -I Sources/InferKitC/include \
      -arch arm64 -mmacosx-version-min=11.0 -Wall -Wextra -Wno-unused-parameter -Werror \
      -framework Foundation -framework CoreVideo \
      -o MyPlugin.inferkitplugin/Contents/MacOS/MyPlugin MyPlugin.m
  ```

  `-undefined dynamic_lookup` leaves every InferKit symbol for the loader to bind, so the bundle
  embeds no copy of the core. Xcode 27's linker builds it without a warning at macOS 11.0 and at the
  SDK's default target, so `-Wl,-no_fixup_chains` is not needed. No class in the bundle carries the
  `NFK` prefix.
- `c-abi`: a library that includes `NFKPluginABI.h` and links nothing from InferKit.

  ```
  xcrun clang -dynamiclib -std=c99 -pedantic -fvisibility=hidden \
      -Wall -Wextra -Wno-unused-parameter -Werror -I Sources/InferKitC/include \
      -arch arm64 -mmacosx-version-min=11.0 -install_name @rpath/MyPlugin \
      -o MyPlugin.inferkitplugin/Contents/MacOS/MyPlugin my_plugin.c
  ```

  `-fvisibility=hidden` exports only what `NFK_PLUGIN_EXPORT` marks, which is `nfk_plugin_entry`.
- `executable`: any program that performs the hand-off and speaks the frames. The sample uses
  Foundation alone.

  ```
  xcrun clang -fobjc-arc -arch arm64 -mmacosx-version-min=11.0 \
      -Wall -Wextra -Wno-unused-parameter -Werror -framework Foundation \
      -o MyPlugin.inferkitplugin/Contents/MacOS/MyPlugin my_plugin.m
  ```

### Signing

Sign the bundle after the manifest and `Info.plist` are in place. The signature seals
`Contents/Resources`, so an edit to the manifest after signing fails verification.

- Development: `codesign --force --sign - MyPlugin.inferkitplugin`. A host accepts this ad hoc
  signature only under `+[NFKPluginSignaturePolicy developmentPolicy]`.
- Distribution: `codesign --force --timestamp --options runtime --sign "Developer ID Application:
  <Team> (<TEAMID>)" MyPlugin.inferkitplugin`.
- Check: `codesign --verify --strict MyPlugin.inferkitplugin`.

### The fixtures

`Tests/PluginFixtures/` holds one plugin of each kind and the templates they are built from:

| path | holds |
| --- | --- |
| `objc/SamplePlugin.m`, `objc/Info.plist.in` | the `objc` plugin; principal class `SamplePlugin` |
| `objc/DuplicateRuntimePlugin.m` | an `objc` plugin that defines its own `NFKInferenceRequest`, refused with `NFKPluginErrorDuplicateRuntime` |
| `c/sample_plugin.c`, `c/Info.plist.in` | the `c-abi` plugin |
| `executable/sample_executable.m`, `executable/Info.plist.in` | the `executable` plugin, Foundation only; it frames the protocol by hand |
| `manifests/{objc,c-abi,executable}.json.in` | the manifests |
| `headers/inferkit_c_usage.c` | a program using every declaration of `NFKInferKitC.h`, for `NFKCHeaderTests` |
| `headers/inferkit_c_consumer.c`, `headers/inferkit_cpp_consumer.cpp` | a C99 and a C++11 program that build the echo model of `NFKInferKitCTests`, run it, and submit a run, for that class |

- The templates take `@IDENTIFIER@`, `@VERSION@`, `@EXECUTABLE@`, `@PRINCIPAL_CLASS@`, and `@ARCH@`.
  The sources take `-DSAMPLE_PLUGIN_VERSION="<version>"`, so a test builds version 1.0.0 and 2.0.0 of
  the same plugin.
- Identifiers: `com.example.inferkit.sample-objc`, `com.example.inferkit.sample-c`,
  `com.example.inferkit.sample-executable`, and `com.example.inferkit.sample-duplicate-runtime`.
- Every kind serves the same models, tool, and option, and every result carries `pluginVersion`:
  - `invert` → a 32BGRA pixel buffer under `image` comes back with each color byte replaced by 255
    minus itself and alpha kept; a float32 or float64 array under `values` comes back with each element
    replaced by 1 minus itself. The C and executable plugins answer an attached input with an attached
    output and an inline input inline. The Objective-C plugin returns objects, which its host encodes. A
    detection under `region` comes back as a one-element `detections` array, its box mirrored left to
    right (x becomes 1 − x − width), its confidence replaced by 1 minus itself, its label and class index
    kept: the Objective-C plugin receives an `NFKDetection`, and the C and executable plugins the tagged
    JSON `{"$nfk": "detection", …}`, which they write back in the same form.
  - `slow` → counts `steps` steps (default 10) of `stepMilliseconds` each (default 20), reports
    progress and a partial result `{"step": n}` after each, and ends cancelled once the run is
    cancelled. `exitAtStep` ends the process with status 70 at that step, which only an out-of-process
    plugin survives.
  - `echo` → a tool result whose text is the `echoPrefix` option followed by the `text` argument.
  - Each backend holds 4096 bytes of weights once loaded: `residentBytes` 4096 while loaded and 0 after
    `unload`, `peakBytes` 4096.
- The C plugin counts its allocations. `sample_plugin_live_allocations()`, exported beside the entry,
  returns the count of allocations not yet freed.
- `NFKPluginFixtures` (`Tests/InferKitTests/`) finds the source root from its own `__FILE__`, builds a
  fixture with `xcrun clang` through `NFKSubprocess`, writes the `Info.plist` and the manifest, signs
  ad hoc, and caches the bundle in `$TMPDIR/inferkit-plugin-fixtures-<hash>`. The hash covers the
  sources, the templates, the headers the build reads, the flags, the version, and the architecture.
  Each build happens in a staging directory renamed into place, so concurrent test processes share one
  complete build. `build-commands.txt` beside the bundle records the commands.
- `+hostExecutableURL` is `NFK_PLUGIN_HOST_PATH` when set, else `inferkit-plugin-host` beside the
  test bundle, where `swift build --build-tests` and Xcode both place it.
- `+tamperedCopyOfBundleAtURL:error:` copies a bundle and edits its manifest's summary after signing.
- Builds within one test process run one at a time (`@synchronized` on the class), and a second caller
  finds the finished cache. Builds in two processes use separate staging directories, named by UUID, and
  the first `rename` wins. No two builds share a path.
- `+runProgram:arguments:error:` closes the tool's standard input, sets `CODESIGN_ALLOCATE` to the
  toolchain's `codesign_allocate` (found once with `xcrun --find`), and on its 300-second timeout lists
  the processes below the tool in the error. On 2026-10-15 a full core run timed out in `codesign --force
  --sign -` of the duplicate-runtime fixture after it printed "replacing existing signature", the point
  where codesign runs `codesign_allocate`; through `/usr/bin/codesign_allocate` that is an xcrun shim
  resolving the toolchain on each call. The same test passed alone. The cause was not observed, and the
  listing records it if it recurs.

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
- Stage 2 registry: `NFKPluginRegistry`, the `plugin` engine builder, `NFKPluginRegistryTests`,
  `Docs/plugins.md`, the DocC article `Plugins`, and the plugin examples.
- Item 13: `NFKPluginCompletenessTests`, the fixtures' `region` detection, the DocC article
  `CPluginInterface` over `NFKPluginABI.h`, and option values re-checked on a reload.

## Completeness test

`NFKPluginCompletenessTests` is the one place that walks every plugin form end to end; the other plugin
test classes each prove one layer, and `NFKPluginRegistryTests` the registry's rules on the C fixture.
Each test loops over the three kinds with a registry of its own catalog per kind, since every fixture
declares the `echo` tool and a tool name is unique within a registry.

- The listing test copies all three into one folder and drops the tools from two copies' manifests, which
  breaks their signatures; listing checks none.
- The attachment test sizes its values on both sides of the 64 KiB inline limit: 64 × 64 × 4 and
  4096 × 4 bytes cross inline, and 256 × 256 × 4 and 65536 × 4 bytes as attachments.
- The lifecycle test sets the process's `idleExitInterval` to 0.5 s before the first launch and keeps
  every backend and job inside an `@autoreleasepool`, so the idle exit follows the last release.
- The C API test configures `NFKPluginRegistry.sharedRegistry` (the development policy, the host built
  beside the tests), adds a copy of the C fixture with `nfk_plugins_add_directory`, and restores the
  policy and removes the plugin in a teardown block.
- The in-process test loads the cached 1.0.0 `objc` and `c-abi` fixtures only, as every in-process test
  does.
