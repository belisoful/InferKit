# Tools

Developer and validation tooling. **Nothing here ships in any distribution.** The SwiftPM targets name
`Sources/InferKit` and `InferKitMLX/Sources` as their paths, the CocoaPods spec globs only
`Sources/InferKit/**/*.{h,m}`, and the XCFramework release assets are built binaries — none of them
compile or carry anything under `Tools/`. A consumer resolving the Swift package clones the whole
repository, so these files land on disk, but they are never built into the products or the release.

Consuming the library needs **no Python**: `NFKMLXTorchFormat` (InferKitMLX) reads a raw PyTorch
checkpoint (`.pth` / `.pt` / `.ckpt` / `.th` / HF `.bin`, including the TorchScript and `.nemo` and
live-module-tree forms) natively, so every model loads a raw checkpoint directly.

## What lives here

### Checkpoint converters — `*-to-safetensors/`

46 offline scripts that turn a released checkpoint into a safetensors file. Since the native
reader loads raw checkpoints, these are **optional**. They stay for two non-consumer reasons:

- **The byte oracle.** `NFKMLXTorchParityTests` proves the native reader correct by comparing a raw
  `.pth` against the converter's own safetensors, tensor for tensor.
- **The offline path** to a portable, pickle-free safetensors — smaller, faster to load, no
  arbitrary-code surface — which some CI and consumers prefer.

CLIP (TorchScript), YOLO (a live `ultralytics` module tree), and VAD (a `.nemo` tar) are no longer
special: the native reader handles them too, so their converters are optional like the rest.

### Reference-parity and validation — `reference-parity/`, `validation-assets/`

The measurement discipline the whole package rests on. `reference-parity/run_reference.py` runs a
model's (or a training objective's) real reference implementation — torch, `transformers`, `diffusers`
— and records input + output for numeric comparison; `validation-assets/{manifest.json,fetch.py}`
fetches every real checkpoint the parity suites load into the validation root (`IK_VALIDATION_ROOT`, else
`/Volumes/InferKit Models/inferkit-validation` while that volume is mounted, else
`~/.inferkit-validation`), and
`validation-assets/shapes.py` fetches a release's config and every tensor's shape by HTTP range request
over its safetensors headers, for the structural checks against releases too large to run. This ground truth
is **irreducibly Python**: a Swift port cannot be validated against another Swift port.

### Build and packaging — `xcframework/`, `docc/`, `ane-placement/`, `espeak/`, `build-all.sh`, `mlx-metallib.sh`

XCFramework packaging (`xcframework/build*.sh`, `verify-mlx.sh`, and `release.sh`, which writes the
release zips and `release.json`), the DocC catalog builder
(`docc/build.sh`), the Core ML ANE-placement measurement (`ane-placement/`), the optional system
espeak-ng installer (`espeak/install.sh`, GPLv3, not bundled), the build/test driver for the core, InferKitMLX,
InferKitFoundationModels, and InferKitAppleSwift (`build-all.sh`), and `mlx-metallib.sh`, which compiles mlx-swift's Metal kernels with `xcrun metal`
and places the library beside the InferKitMLX SwiftPM test binary, so `swift test` runs the
MLX-dependent tests SwiftPM's own build cannot (it compiles no shaders). Also `inferkit-convert/`,
the offline HF-causal-LM → Core ML model-directory exporter.

### Binary interface — `abi/`

`check-abi.sh` holds a built `InferKitMLX.framework` to a baseline release's binary interface:
`abi-surface.py` reads both sets of public headers through clang, applies the API version rule to
the differences, and a client compiled against the baseline headers loads the framework and finds
every class, selector, conformance, constant, and function. The baseline is a framework, a git ref,
or `current`. `Docs/agent-reference/distribution-and-packaging.md` ("ABI baseline check") has the
details.

### Large Model Coordination — `lmc/`

`lmc.py` is the lock every agent session takes before a large-model test run, so one run at a time
holds the machine's memory. A session requests the lock for a named test (`full-check`, or a model's
released-weight suite), waits until it is granted, runs, and releases as the first thing it does
when the run ends; `lmc.py run --test <name> -- <command>` does all four. Requests for the same
test set from the same working tree combine into one run whose outcome satisfies every requester;
`full-check` always queues last; a holder whose process is gone is reaped, and the sessions riding
on a stopped run are told to re-request. State lives in `~/.claude/inferkit-lmc/`, and the legacy
`~/.claude/inferkit-test-slot-mlx` file is written and honored alongside it. `test_lmc.py` runs
the tool's own tests. The rule that binds sessions to it is in
`Docs/agent-reference/build-and-verification.md`.

### Documentation snippets — `doc-snippets/`

`check-objc.py` type-checks every Objective-C code block in `README.md`, `Docs/examples.md`, and
`Docs/inference-guide.md` with `clang -fsyntax-only` against the core's public headers and the
Objective-C headers the three companions generate. `snippet-context.h` declares the variables the
blocks take from their prose. A block that differs uses an `<!-- objc-check: … -->` directive above
its fence (`given`, `continues`, or `skip`); the script's own help lists them. The check is step 7 of
the Full Check in `Docs/agent-reference/build-and-verification.md`.

### Model catalog measurement — `catalog/`

`measure.py` sizes every variant source the MLX catalog lists without downloading a weight file: the
Hugging Face file tree (sizes, Git LFS digests, the commit a revision resolves to), each safetensors
header by HTTP range request through `validation-assets/shapes.py` (tensor bytes by dtype, element
counts), and a HEAD request per URL file. It writes the generated
`InferKitMLX/Sources/InferKitMLX/NFKMLXCatalogMeasurements.swift`. Run it from the repository root with
`python3 -I Tools/catalog/measure.py`; the notes are in "Model catalog" in
`Docs/agent-reference/mlx-companion.md`.

### Swift isolation probe — `swift-isolation-probe/`

`run.sh` builds the core, compiles `main.swift` as Swift 6 with `-default-isolation MainActor`, and runs
it. The probe passes closures written in main-actor code to blocks InferKit calls on its own queues
(the server's provider, builder, delegate, app-route handler, load handler, and pressure provider, a
job's handlers, a tool's handler). Exit 0 means each imports as Sendable; an isolation trap exits with SIGTRAP (133). The
reasons are in `Docs/agent-reference/remote-providers.md` ("Sendable blocks").

## Requirements

The converters need `torch` and `safetensors`; the reference oracles additionally need the model's own
reference package (`transformers`, `diffusers`, `openai-whisper`, `demucs`, …), some in their own
interpreter recorded in `validation-assets/manifest.json`'s `oracle_environments`. None of this is a
requirement to build or use the shipped library.
