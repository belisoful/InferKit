# Tone Filter: a C plugin

A plugin (`entry.kind` `c-abi`) written in C99 against `NFKPluginABI.h` alone. It links nothing from
InferKit. It serves two image filters in the family `tone-filter`:

- `sepia` tones an image in sepia.
- `grayscale` turns an image to shades of gray.

Each reads a 32BGRA image under `image`, blends the filtered color into the original by its option
`strength` (0 keeps the image, 1 filters it fully), keeps alpha, and returns the image under `image`.

| file | holds |
| --- | --- |
| `tone_filter.c` | the library: `nfk_plugin_entry`, the function table, a small JSON reader, and the filters |
| `inferkit-plugin.json` | the manifest, with `compatibility.pluginABI` `"1.0"` |
| `Info.plist.in` | the bundle's `Info.plist`, which `build.sh` fills from the manifest |
| `build.sh` | builds, assembles, and signs `build/ToneFilter.inferkitplugin` |

## What the code shows

**The entry.** `nfk_plugin_entry` checks the host's major ABI version, keeps the host's table for the
plugin's life, and returns its own table. It fills the required functions (`describe`,
`create_backend`, `run`, `destroy_backend`, `free_payload`, `free_string`, `shutdown`) and leaves the
optional ones NULL, which tells the host the plugin does not support them.

**The request.** `run` receives a payload: a JSON object in InferKit's tagged wire form and the buffers
its entries index. The image arrives in one of two forms:

- an image over the inline limit (64 KiB) as a buffer entry, its bytes in `buffers[index]`:

  ```json
  {"inputs": {"image": {"$nfk": "buffer", "as": "pixelBuffer", "index": 0, "length": 1048576,
   "format": 1111970369, "width": 512, "height": 512, "planes": [{"offset": 0, "bytesPerRow": 2048, "rows": 512}]}},
   "parameters": {"strength": 0.5}}
  ```

- a smaller image inline, as `{"$nfk": "pixelBuffer", …, "planes": [{"bytesPerRow", "rows", "base64"}]}`.

The buffer array strides by its first element's `struct_size`, since a later minor ABI version
lengthens the structure. `tone_request_buffer` reads buffer `i` at
`(const uint8_t *)buffers + i * buffers[0].struct_size`. The request and its buffers stay valid until
`run` returns, so the filter reads them in place.

**The result.** The plugin allocates the filtered pixels and returns them as the result's one buffer,
whatever the image's size. The result's JSON names the buffer with the same geometry:

```json
{"outputs": {"image": {"$nfk": "buffer", "as": "pixelBuffer", "index": 0, "length": 1048576,
 "format": 1111970369, "width": 512, "height": 512, "planes": [{"offset": 0, "bytesPerRow": 2048, "rows": 512}]}}}
```

**Progress and cancellation.** The filter works in sixteen bands of rows. Before each band it polls
`host->is_cancelled` and reports `host->report_progress` with the fraction done. A cancelled run frees
what it allocated and returns `NFK_PLUGIN_STATUS_CANCELLED`. Polling is enough to honor a cancellation,
so the table leaves `cancel` NULL. An app sees the progress on the job:

```objc
NFKInferenceJob *job = [filter submitInferenceJobForRequest:request];
job.progressHandler = ^(NFKInferenceJob *job) { NSLog(@"%.0f%%", job.progress * 100); };
[job cancel];   // the next band's poll ends the run
```

**Ownership.** Whoever allocates frees:

| what | allocated by | freed by |
| --- | --- | --- |
| the request payload and its buffers | the host | the host, after `run` returns |
| the result payload, its JSON, and its pixels | the plugin, in one `tone_result` | the plugin's `free_payload`, which the host calls when the last value over the pixels is released |
| the `describe` JSON and every error JSON | the plugin | the plugin's `free_string`, which the host calls after copying it |
| a backend | the plugin's `create_backend` | the plugin's `destroy_backend` |
| the plugin itself | the entry | `shutdown` |

## Build and sign

```
PluginExamples/Filter-C/build.sh
```

The script compiles a universal library with `-std=c99 -pedantic -fvisibility=hidden`, so it exports
only `nfk_plugin_entry`, which `NFK_PLUGIN_EXPORT` marks. It includes only
`Sources/InferKitC/include/InferKitC/NFKPluginABI.h`. It then writes the bundle's `Info.plist` and
manifest and signs it ad hoc.

## Install

```
ditto build/ToneFilter.inferkitplugin "$HOME/Library/Application Support/MyApp/Plugins/ToneFilter.inferkitplugin"
```

```objc
NSArray<NSError *> *errors = nil;
[NFKPluginRegistry.sharedRegistry addPluginsInDirectoryURL:pluginsFolder errors:&errors];
```

## What the app sees

The catalog lists `com.example.inferkit.tone-filter/sepia` and `com.example.inferkit.tone-filter/grayscale`,
each with the number option `strength`, from 0 to 1 in steps of 0.05. To the app they are
`NFKInferenceBackend`s like any other: a request carries a `CVPixelBuffer` under `NFKInputImage`, and the
result carries one under `NFKOutputImage`.

```objc
NSError *error = nil;
id<NFKInferenceBackend> filter =
	[NFKModelCatalog.sharedCatalog backendForModelIdentifier:@"com.example.inferkit.tone-filter/sepia"
													 options:@{ @"strength": @0.5 }
													   error:&error];
NFKInferenceResult *result = [filter runInferenceForRequest:request error:&error];
CVPixelBufferRef toned = (__bridge CVPixelBufferRef)[result outputForKey:NFKOutputImage];
```

## The same plugin in C++ or Rust

- **C++.** The library includes the same header, which carries `extern "C"` guards, so
  `nfk_plugin_entry` and the table's functions have C linkage. Build with `clang++ -dynamiclib
  -fvisibility=hidden`. No C++ exception crosses into the host: the entry and every function in the
  table catch what they throw and return a status. A `new` result is freed with the matching `delete`
  in `free_payload`.
- **Rust.** A `cdylib` crate exports `#[no_mangle] pub extern "C" fn nfk_plugin_entry(host_abi_version:
  u32, host: *const NfkPluginHost, out_plugin: *mut *const NfkPlugin) -> i32`. The structures are
  `#[repr(C)]` copies of the header's, which `bindgen` generates from `NFKPluginABI.h`. A panic does not
  unwind into the host: wrap each function in `std::panic::catch_unwind`, or build with
  `panic = "abort"`. Memory handed to the host returns through `free_payload` and `free_string`, which
  rebuild the `Box` or `CString` with `from_raw` and drop it. The built `.dylib` goes in
  `Contents/MacOS` under the name `entry.executable` gives.
