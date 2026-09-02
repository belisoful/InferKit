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
fetches every real checkpoint the parity suites load into `~/.inferkit-validation`. This ground truth
is **irreducibly Python**: a Swift port cannot be validated against another Swift port.

### Build and packaging — `xcframework/`, `docc/`, `ane-placement/`, `espeak/`, `build-all.sh`

XCFramework packaging (`xcframework/build*.sh`, `verify-mlx.sh`), the DocC catalog builder
(`docc/build.sh`), the Core ML ANE-placement measurement (`ane-placement/`), the optional system
espeak-ng installer (`espeak/install.sh`, GPLv3, not bundled), and the all-packages build/test driver
(`build-all.sh`). Also `inferkit-convert/`, the offline HF-causal-LM → Core ML model-directory
exporter.

## Requirements

The converters need `torch` and `safetensors`; the reference oracles additionally need the model's own
reference package (`transformers`, `diffusers`, `openai-whisper`, `demucs`, …), some in their own
interpreter recorded in `validation-assets/manifest.json`'s `oracle_environments`. None of this is a
requirement to build or use the shipped library.
