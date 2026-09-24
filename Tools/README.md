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

~30 offline scripts that turn a released PyTorch checkpoint into a safetensors file. Since the native
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
fetches every real checkpoint the parity suites load into `~/.inferkit-validation`, and
`validation-assets/shapes.py` fetches a release's config and every tensor's shape by HTTP range request
over its safetensors headers, for the structural checks against releases too large to run. This ground truth
is **irreducibly Python**: a Swift port cannot be validated against another Swift port.

### Build and packaging — `xcframework/`, `docc/`, `ane-placement/`, `espeak/`, `build-all.sh`, `mlx-metallib.sh`

XCFramework packaging (`xcframework/build*.sh`, `verify-mlx.sh`), the DocC catalog builder
(`docc/build.sh`), the Core ML ANE-placement measurement (`ane-placement/`), the optional system
espeak-ng installer (`espeak/install.sh`, GPLv3, not bundled), the all-packages build/test driver
(`build-all.sh`), and `mlx-metallib.sh`, which compiles mlx-swift's Metal kernels with `xcrun metal`
and places the library beside the InferKitMLX SwiftPM test binary, so `swift test` runs the
MLX-dependent tests SwiftPM's own build cannot (it compiles no shaders). Also `inferkit-convert/`,
the offline HF-causal-LM → Core ML model-directory exporter.

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
its fence (`given`, `continues`, or `skip`); the script's own help lists them. The check is step 5 of
the Full Check in `Docs/agent-reference/build-and-verification.md`.

## Requirements

The converters need `torch` and `safetensors`; the reference oracles additionally need the model's own
reference package (`transformers`, `diffusers`, `openai-whisper`, `demucs`, …), some in their own
interpreter recorded in `validation-assets/manifest.json`'s `oracle_environments`. None of this is a
requirement to build or use the shipped library.
