# C interface

`NFKInferKitC.h` is InferKit's interface for programs written in C, C++, Rust, or any language with a
C foreign-function interface. A program builds a backend from the [model catalog](model-catalog.md)
by its identifier and a JSON object of options, runs requests made of tagged JSON and buffers, follows
a run's progress, and cancels it. Every model the catalog lists is reachable this way, the models a
plugin adds included.

- The header is pure C99 with `extern "C"` guards. It includes `NFKAPIVersion.h` and nothing else
  from InferKit.
- The implementation is part of the InferKit library. A program links InferKit, or a framework that
  carries it, and calls the functions directly.
- Platforms: macOS 11, iOS 14, tvOS 14. `nfk_plugins_add_directory` adds plugins on macOS only; on iOS
  and tvOS it returns -1 with `kNFKError_InferenceUnsupported`.
- The [API version](installation.md#api-version) covers the interface. `nfk_api_version()` reports
  the library's, and `NFK_API_VERSION` the one the program compiled against. A program accepts a
  library with the same major version and a minor version at least its own.

## Linking

### Swift Package Manager

A C target in a package depends on the `InferKit` product and includes the header through the
`InferKitC` module path:

```swift
.target(name: "Infer", dependencies: [.product(name: "InferKit", package: "InferKit")])
```

```c
#include <InferKitC/NFKInferKitC.h>
```

SwiftPM links the core and the system frameworks it names.

### The XCFrameworks

`Tools/xcframework/build.sh` and `build-mlx.sh` place `NFKInferKitC.h`, `NFKPluginABI.h`, and
`NFKAPIVersion.h` beside the core's headers in every slice.

- `InferKit.xcframework`, the core alone, static: the headers sit directly in `<slice>/Headers`.

  ```bash
  clang -std=c99 -I InferKit.xcframework/<slice>/Headers main.c \
        InferKit.xcframework/<slice>/libInferKit.a -lobjc \
        -framework Foundation -framework CoreML -framework Vision -framework VideoToolbox \
        -framework Speech -framework SoundAnalysis -framework NaturalLanguage -framework CoreVideo \
        -framework CoreGraphics -framework ImageIO -framework AVFoundation -framework Metal \
        -framework IOSurface -framework Network -framework Security -framework IOKit \
        -o infer
  ```

  ```c
  #include "NFKInferKitC.h"
  ```

  `IOKit` is macOS only. The library holds the core as one object, so a reference to any `nfk_`
  function links all of it, the catalog's providers included. The core's objects name `AVFAudio`,
  `CoreMedia`, and `CoreFoundation` themselves, and the linker adds them.
- `InferKitMLXDynamic.xcframework`, the core with InferKitMLX, dynamic: the core's headers sit in
  `InferKitMLX.framework/Headers/InferKit`.

  ```bash
  clang -std=c99 -F InferKitMLXDynamic.xcframework/macos-arm64 \
        -I InferKitMLXDynamic.xcframework/macos-arm64/InferKitMLX.framework/Headers \
        main.c -framework InferKitMLX -Wl,-rpath,@executable_path/../Frameworks -o infer
  ```

  ```c
  #include <InferKit/NFKInferKitC.h>
  ```

  The framework ships beside the program at the run path. The framework carries InferKitMLX's
  descriptor provider, so the shared catalog lists the MLX models.
- `InferKitMLX.xcframework`, static: as the core's static library, with the library and the Metal
  shader bundle the [installation guide](installation.md#binary-distribution-xcframework) describes,
  and `Accelerate` and `libc++` added to the link.

### C++

The header declares everything inside `extern "C"`, so a C++ program includes it as it is. A
`std::unique_ptr` with a deleter owns each handle:

```cpp
#include <InferKit/NFKInferKitC.h>
#include <memory>

struct backend_deleter { void operator()(nfk_backend *backend) const { nfk_backend_free(backend); } };
struct result_deleter { void operator()(nfk_result *result) const { nfk_result_free(result); } };
struct error_deleter { void operator()(nfk_error *error) const { nfk_error_free(error); } };

using backend_ptr = std::unique_ptr<nfk_backend, backend_deleter>;
using result_ptr = std::unique_ptr<nfk_result, result_deleter>;
using error_ptr = std::unique_ptr<nfk_error, error_deleter>;

nfk_error *raw_error = nullptr;
backend_ptr backend(nfk_backend_create("natural-language-embedding", nullptr, &raw_error));
error_ptr error(raw_error);
```

### Rust

The Rust example below is not compiled by the repository's checks, which have no Rust toolchain. Its
declarations follow the header field by field.

A Rust crate declares the functions in an `extern "C"` block, or generates the block with `bindgen`
from `NFKInferKitC.h`, and links the framework from its build script. The handles are opaque types:

```rust
use std::ffi::{c_char, c_void, CStr, CString};

#[repr(C)] pub struct NfkBackend { _private: [u8; 0] }
#[repr(C)] pub struct NfkResult { _private: [u8; 0] }
#[repr(C)] pub struct NfkError { _private: [u8; 0] }

#[repr(C)]
pub struct NfkBuffer {
    pub struct_size: u32,
    pub kind: i32,
    pub bytes: *const c_void,
    pub length: u64,
    pub surface: *mut c_void,
    pub path: *const c_char,
    pub descriptor: i32,
    pub reserved: u32,
}

#[repr(C)]
pub struct NfkPayload {
    pub struct_size: u32,
    pub buffer_count: u32,
    pub json: *const c_char,
    pub buffers: *const NfkBuffer,
}

extern "C" {
    fn nfk_backend_create(id: *const c_char, options: *const c_char, error: *mut *mut NfkError) -> *mut NfkBackend;
    fn nfk_backend_run(backend: *mut NfkBackend, request: *const NfkPayload, error: *mut *mut NfkError) -> *mut NfkResult;
    fn nfk_result_json(result: *const NfkResult) -> *const c_char;
    fn nfk_result_free(result: *mut NfkResult);
    fn nfk_backend_free(backend: *mut NfkBackend);
    fn nfk_error_message(error: *const NfkError) -> *const c_char;
    fn nfk_error_free(error: *mut NfkError);
}

fn embed(text: &str) -> Result<String, String> {
    let id = CString::new("natural-language-embedding").unwrap();
    let json = CString::new(format!(r#"{{"inputs": {{"prompt": {:?}}}}}"#, text)).unwrap();
    unsafe {
        let mut error: *mut NfkError = std::ptr::null_mut();
        let backend = nfk_backend_create(id.as_ptr(), std::ptr::null(), &mut error);
        if backend.is_null() {
            let message = CStr::from_ptr(nfk_error_message(error)).to_string_lossy().into_owned();
            nfk_error_free(error);
            return Err(message);
        }
        let request = NfkPayload {
            struct_size: std::mem::size_of::<NfkPayload>() as u32,
            buffer_count: 0,
            json: json.as_ptr(),
            buffers: std::ptr::null(),
        };
        let result = nfk_backend_run(backend, &request, &mut error);
        let outcome = if result.is_null() {
            let message = CStr::from_ptr(nfk_error_message(error)).to_string_lossy().into_owned();
            nfk_error_free(error);
            Err(message)
        } else {
            let text = CStr::from_ptr(nfk_result_json(result)).to_string_lossy().into_owned();
            nfk_result_free(result);
            Ok(text)
        };
        nfk_backend_free(backend);
        outcome
    }
}
```

```rust
// build.rs
fn main() {
    println!("cargo:rustc-link-search=framework=/path/to/InferKitMLXDynamic.xcframework/macos-arm64");
    println!("cargo:rustc-link-lib=framework=InferKitMLX");
}
```

`NfkBuffer` and `NfkPayload` keep the C layout: 48 and 24 bytes on a 64-bit target.

## Ownership

| What | Owner | Freed by |
| --- | --- | --- |
| a backend from `nfk_backend_create` | the caller | `nfk_backend_free` |
| a job from `nfk_backend_submit` | the caller | `nfk_job_free` |
| a result from `nfk_backend_run` or a completion | the caller | `nfk_result_free` |
| an error from an `out_error` or a completion | the caller | `nfk_error_free` |
| a string from `nfk_catalog_copy_json` or `nfk_backend_copy_description` | the caller | `nfk_string_free` |
| a partial result in a progress callback | InferKit | nothing; valid during the call |
| `nfk_release_version()` | InferKit | nothing; static storage |
| the JSON, buffers, and buffer array of a result | the result | `nfk_result_free` |
| the domain and message of an error | the error | `nfk_error_free` |
| a request's `nfk_payload`, its JSON, and its `buffers` array | the caller | the caller; read during the call |
| the memory a request's buffer points to | the caller | the caller, after `nfk_backend_run` returns, the completion fires, or `nfk_job_free` returns |

- Every `*_free` function accepts NULL. A reader given NULL answers NULL or 0.
- A handle passed to a function of another type, such as an error to `nfk_result_free`, is ignored.
- A function with an `out_error` sets `*out_error` to NULL when it succeeds and to an error when it
  fails. `out_error` may be NULL.
- InferKit wraps a request's bytes and surfaces without copying them. A cancelled run's completion
  fires once the backend no longer holds those wrappers, so the caller frees the memory in the
  completion or after it.
- `nfk_job_free` before the completion returns once the run no longer reads the request's buffers, so
  the caller frees the memory after it returns:

  ```c
  nfk_job_free(job);   /* cancels the run and waits for it to let go of the buffers */
  free(pixels);        /* safe: nothing reads the buffer any more */
  ```
- A job holds its backend. `nfk_backend_free` on a backend with a job in flight frees it when the job's
  run ends.

## Threading

- Every function is safe to call from any thread.
- `nfk_backend_run` blocks its thread for the whole run, which can take seconds. A program calls it off
  its main or render thread, or uses `nfk_backend_submit`.
- A job's callbacks run on a serial queue of its own, at user-initiated quality of service, one at a
  time and in the order the run reported them. `nfk_backend_submit` never calls them itself; the
  first may start on that queue before `nfk_backend_submit` returns to the caller.
- The completion fires exactly once, after every progress callback. `nfk_job_free` before the
  completion cancels the run, and the completion then never fires.
- After `nfk_job_free` returns, no callback runs: it waits for a callback in progress on another thread.
- `nfk_job_free` on a run that has not completed also waits until the run no longer reads the request's
  buffers:
  - a run that already returned → it returns at once;
  - a run the backend stops at the cancellation → it returns when the backend stops;
  - a backend that does not check for cancellation → it returns when the run ends, and InferKit logs a
    fault every 10 seconds of the wait.
- Called from inside the job's progress callback, `nfk_job_free` waits the same way, and that callback
  is the last. Called from inside its completion, it returns at once.
- A program does not call `nfk_job_free` on a thread a callback or the backend's run waits for.
- `nfk_job_cancel` never calls the completion itself.

## Requests

A request is an `nfk_payload`: a UTF-8 JSON object and the buffers it indexes.

```json
{"inputs": {"prompt": "a lighthouse"}, "parameters": {"temperature": 0.2}, "outputModality": 0}
```

- `inputs` is required, and holds the input keys a model's descriptor lists as its ports (`image`,
  `prompt`, `audio`, …).
- `parameters` holds request parameters and the request-stage options of the descriptor.
- `outputModality` is optional: 0 text, 1 image (the default), 2 video, 3 audio.

JSON strings, numbers, booleans, null, arrays, and objects pass as themselves. A value with bytes is
tagged under `"$nfk"`, inline or as a buffer.

### Buffers

A buffer entry names its index in `buffers`, what its bytes are, and their length:

```json
{"$nfk": "buffer", "index": 0, "as": "float32Array", "length": 4096, "count": 1024}
```

| `as` | Extra fields | The backend receives |
| --- | --- | --- |
| `data` | | the bytes, unwrapped, not copied |
| `float32Array`, `float64Array` | `count` | an array of numbers, copied from little-endian words |
| `pixelBuffer` | `format`, `width`, `height`, `planes [{offset, bytesPerRow, rows}]` | a pixel buffer over the bytes, not copied |
| `multiArray` | `dataType`, `shape` | a multi-array over the bytes in row-major order, not copied |
| `pcm` | `sampleRate`, `formatID`, `formatFlags`, `bytesPerPacket`, `framesPerPacket`, `bytesPerFrame`, `channelsPerFrame`, `bitsPerChannel`, `frameLength`, `buffers [{offset, length}]` | an audio buffer over the bytes, not copied |
| `cgImage`, `image` | | an image decoded from PNG bytes |

- `length` is required, and is at most the buffer's length.
- `format` is the Core Video pixel format as a number: 1111970369 for 32BGRA (`'BGRA'`), 1380401729
  for 32RGBA, 1278226534 for a one-channel float (`'L00f'`).
- `dataType` is Core ML's `MLMultiArrayDataType`: 65568 float32, 65600 float64, 65552 float16, 131104
  int32.
- InferKit checks every geometry against `length` before it wraps the bytes.

An `nfk_buffer` holds its bytes in one of three ways:

| `kind` | Fields | Read |
| --- | --- | --- |
| `NFK_BUFFER_BYTES` | `bytes`, `length` | wrapped without a copy |
| `NFK_BUFFER_SURFACE` | `surface` (an `IOSurfaceRef`), `length` (0 for the whole surface) | locked read-only and wrapped without a copy |
| `NFK_BUFFER_FILE` | `path`, or `descriptor` with `path` NULL | the path mapped; the descriptor's file read into memory |

`struct_size` is `sizeof(nfk_buffer)` and `descriptor` is -1 when unused. An array of buffers strides
by its first element's `struct_size`. InferKit accepts any `struct_size` at least the 1.0 size of the
structure (48 bytes for `nfk_buffer`, 24 for `nfk_payload` on a 64-bit target), so a program built
against a later minor version runs on this library. A field a later minor version appends is read only
when the writer's `struct_size` covers it.

### Inline values

A small value travels in the JSON, base64-encoded:

| Tag | Fields |
| --- | --- |
| `data` | `base64` |
| `float32Array`, `float64Array` | `base64`, little-endian words |
| `pixelBuffer` | `format`, `width`, `height`, `planes [{bytesPerRow, rows, base64}]` |
| `file`, `audio`, `video` | `name`, and `base64`, or `path` to an absolute path in the program's file system |

`audio` adds `durationSeconds`, `sampleRate`, `channelCount`; `video` adds `durationSeconds`,
`framesPerSecond`, `width`, `height`. A file named by `path` stays where it is; the program keeps it
until the run ends.

## Results

A result is `{"outputs": {…}}` and its buffers. Every value with bytes comes back as a buffer entry,
whatever its size, so a program decodes no base64 for them. Strings, numbers, and booleans are
inline. Each buffer is `NFK_BUFFER_BYTES`, valid until `nfk_result_free`.

```json
{"outputs": {"embedding": {"$nfk": "buffer", "index": 0, "as": "float32Array", "length": 2048, "count": 512}}}
```

- A pixel buffer keeps each plane's row padding: read row `y` of plane `p` at
  `bytes + planes[p].offset + y * planes[p].bytesPerRow`.
- A file, audio, or video output is `{"$nfk": "file", "path", "name"}` (or `audio`, `video` with their
  fields). The file lives at least as long as the result.
- InferKit's value classes arrive as JSON objects tagged by class. A point is `{"x", "y"}` and a
  rectangle `{"x", "y", "width", "height"}`, in the coordinates the backend reports (Vision's are
  normalized, origin bottom-left). A figure that is not finite is `null`, and a string field that is
  absent is left out.

| Tag | Fields |
| --- | --- |
| `detection` | `label`, `classIndex`, `confidence`, `boundingBox`, `quadrilateral` |
| `keypoint` | `name`, `index`, `position`, `confidence` |
| `classification` | `label`, `classIndex`, `confidence` |
| `audioSegment` | `startSeconds`, `endSeconds`, `label`, `confidence`, `speaker` |
| `midiSequence` | `notes` (of `midiNote`), `tempoBPM`, `beatsPerBar`, `beatUnit`, `ticksPerQuarterNote`, `durationSeconds` |
| `midiNote` | `pitch`, `startSeconds`, `endSeconds`, `velocity`, `program`, `percussion`, `pitchBend` |
| `musicBeat` | `timeSeconds`, `positionInBar`, `downbeat` |
| `quadrilateral` | `topLeft`, `topRight`, `bottomLeft`, `bottomRight`, `boundingBox` |
| `decisionAnswer` | `type` (`choice`, `score`, `noul`), `choice`, `score`, `probability`, `confidence`, `probabilities`, `legend`, `raw`, `refused` |

```json
{"outputs": {"detections": [{"$nfk": "detection", "label": "cat", "classIndex": 3, "confidence": 0.87,
                             "boundingBox": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}}]}}
```

- A dictionary keyed by numbers, such as a Core ML classifier's probabilities by class index, is
  `{"$nfk": "numberKeyedDictionary", "value": {"3": 0.75, "7": 0.25}}`, each key as its decimal text.
- An image is a `pixelBuffer` buffer, or `cgImage` and `image` buffers of PNG bytes for a CGImage and
  an 8-bit Metal texture. A half or float texture is a `pixelBuffer` of its own format, such as
  1380410945 (`'RGfA'`) for four float channels.
- A Core ML feature value comes back as the value it holds. A feature of Core ML's state type has no
  form, and a run that returns one fails with `kNFKError_InferenceUnsupported`, as does a dictionary
  that mixes string and number keys.

A request takes the same forms, so a program passes a region to a tracker as a `detection`. A plugin
receives and returns value classes in the same forms. On
reading, `durationSeconds`, `downbeat`, and a quadrilateral's `boundingBox` are ignored, since each
follows from the other fields.

## Errors

An error carries a domain, a code, and a message. `nfk_error_copy_details_json` returns its details, a
JSON object of the error's user info under the same keys, `{}` when it has none:

| Key | Value |
| --- | --- |
| `NFKModelOptionProblems` | `[{"key", "code", "message", "value"}]`, each option a build refused; the codes are the [option problems](model-catalog.md#option-values) |
| `NFKModelMemoryEstimate` | the estimate a `DoesNotFit` refusal was checked against |
| `NFKModelMemoryFittingOptions` | the options nearest the requested ones that fit, where some do |
| `NFKModelMemoryFittingEstimate` | their estimate |
| `NSUnderlyingError` | the cause, `{"domain", "code", "message", "details"}`, as a plugin's process failure carries it |
| any other key | a string, a number, or a JSON value as itself |

A value of any other type is left out, and the message is not repeated.

```c
nfk_backend *backend = nfk_backend_create("qwen3-0.6b", "{\"quantizationBits\": 3}", &error);
if (backend == NULL) {
	char *details = nfk_error_copy_details_json(error);
	/* {"NFKModelOptionProblems": [{"code": "not-a-choice", "key": "quantizationBits", "message": …, "value": 3}]} */
	nfk_string_free(details);
	nfk_error_free(error);
}
```

| Domain | Code | When |
| --- | --- | --- |
| `NFKModelCatalogErrorDomain` | 5 `UnknownModel` | `nfk_backend_create` names no model the catalog lists, or NULL |
| `NFKModelCatalogErrorDomain` | 4 `InvalidOptions` | an option the model does not take or cannot use, or options that are not a JSON object |
| `NFKModelCatalogErrorDomain` | 7 `Unavailable` | the machine does not meet the model's requirements |
| `NFKModelCatalogErrorDomain` | 8 `DoesNotFit` | the model's memory estimate exceeds the machine's budget; the message names options that fit |
| `NFKModelCatalogErrorDomain` | 6 `NoBuilder` | nothing builds the model's engine in this process |
| `NFKInferenceErrorDomain` | 2 `MissingInput` | a request that is not JSON, not an object, lacks `inputs`, or names a buffer it does not pass |
| `NFKInferenceErrorDomain` | 4 `Unsupported` | a value InferKit cannot decode, such as a malformed buffer entry |
| `NFKInferenceErrorDomain` | 1, 3, 5, 6, 7 | the backend's own failure |
| `NSCocoaErrorDomain` | 3072 | a cancelled run |

## Examples

### Build a backend and run JSON

```c
#include <InferKit/NFKInferKitC.h>
#include <stdio.h>

int main(void)
{
	nfk_error *error = NULL;
	nfk_backend *backend = nfk_backend_create("natural-language-embedding", NULL, &error);
	if (backend == NULL) {
		fprintf(stderr, "%s (%s %lld)\n", nfk_error_message(error), nfk_error_domain(error),
				(long long)nfk_error_code(error));
		nfk_error_free(error);
		return 1;
	}
	nfk_payload request = { sizeof(nfk_payload), 0, "{\"inputs\": {\"prompt\": \"a lighthouse\"}}", NULL };
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	if (result != NULL) {
		printf("%s\n", nfk_result_json(result));
		nfk_result_free(result);
	} else {
		fprintf(stderr, "%s\n", nfk_error_message(error));
		nfk_error_free(error);
	}
	nfk_backend_free(backend);
	return 0;
}
```

### Loading ahead of the first run

```c
if (!nfk_backend_prepare(backend, &error)) {
	fprintf(stderr, "%s\n", nfk_error_message(error));
	nfk_error_free(error);
}
```

`nfk_backend_prepare` blocks until the model is in memory, so the first run starts at once. A backend
with nothing to load returns 1. `nfk_backend_unload` frees the model again; the next run or
preparation loads it.

### Options

```c
nfk_backend *backend = nfk_backend_create("qwen3-0.6b", "{\"quantizationBits\": 4, \"temperature\": 0.2}", &error);
```

Load options build the backend, and request options become the defaults of every request. The model
catalog's `options` list each key, its kind, and its stage.

### An image buffer

```c
uint8_t pixels[64 * 256];   /* 64 x 64, 32BGRA, 256 bytes a row */
nfk_buffer image = { sizeof(nfk_buffer), NFK_BUFFER_BYTES, pixels, sizeof pixels, NULL, NULL, -1, 0 };
nfk_payload request = { sizeof(nfk_payload), 1,
	"{\"inputs\": {\"image\": {\"$nfk\": \"buffer\", \"index\": 0, \"as\": \"pixelBuffer\", \"length\": 16384, "
	"\"format\": 1111970369, \"width\": 64, \"height\": 64, "
	"\"planes\": [{\"offset\": 0, \"bytesPerRow\": 256, \"rows\": 64}]}}}",
	&image };
nfk_result *result = nfk_backend_run(backend, &request, &error);
```

### A submitted run

```c
static void on_progress(void *context, double fraction, const nfk_result *partial)
{
	if (partial != NULL) {
		printf("%.0f%% %s\n", fraction * 100, nfk_result_json(partial));
	}
}

static void on_completion(void *context, nfk_result *result, nfk_error *error)
{
	if (error != NULL) {
		fprintf(stderr, "%s\n", nfk_error_message(error));
	}
	nfk_result_free(result);
	nfk_error_free(error);
	signal_done(context);   /* the program's own wake-up */
}

nfk_job *job = nfk_backend_submit(backend, &request, on_progress, on_completion, state);
/* … later, to stop it: nfk_job_cancel(job); the completion then reports 3072 */
wait_until_done(state);
nfk_job_free(job);
```

### Plugins

```c
nfk_error *error = NULL;
int32_t added = nfk_plugins_add_directory("/Library/Application Support/MyApp/Plugins", &error);
if (error != NULL) {
	fprintf(stderr, "a plugin was skipped: %s\n", nfk_error_message(error));
	nfk_error_free(error);
}
nfk_backend *backend = nfk_backend_create("com.example.plugin/model", NULL, &error);
```

A plugin's models join the catalog as `<plugin identifier>/<model identifier>` and build like any
other model.

### The catalog

```c
char *catalog = nfk_catalog_copy_json(&error);
/* {"kind": "inferkit.model-catalog", "models": [ … ], "schemaVersion": 1} */
nfk_string_free(catalog);
```

The [model catalog reference](model-catalog.md) documents every field.
