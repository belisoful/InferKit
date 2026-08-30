# Weights & conversion

Where weights come from, how they load, and how a PyTorch checkpoint becomes a safetensors file the
models read directly.

## Overview

Weights are downloaded at runtime, not bundled at build time — for size, redistribution-licensing, and
update reasons. A model builds with `nil` weights (random init, useful for wiring and tests) or from a
local safetensors URL, and a download companion fetches the checkpoint first.

![From a Hugging Face repo to a loaded module: download and cache, then load with the conv transpose.](weights-pipeline)

### Downloading and caching

The core's `NFKHFHub` resolves `<endpoint>/<repo>/resolve/<revision>/<path>`, fetches with
`NSURLSession`, and caches at `<cacheDirectoryURL>/<repo>/<revision>/<path>`. The download blocks, so run
it off the render thread. ``NFKMLXHub`` combines that download with the registry:

```swift
NFKMLXReferenceModels.registerAll()
let backend = try NFKMLXHub.backend(
    named: "real-esrgan-x4",
    repo: "your-org/real-esrgan",
    weightsPath: "RealESRGAN_x4.safetensors",
    revision: nil,
    cacheDirectoryURL: nil   // → Application Support/InferKit/models
)
```

`cacheDirectoryURL` is host-supplied (security-scoped for sandboxed apps); passing `nil` to the MLX
factories substitutes `NFKHFHub.defaultCacheDirectoryURL`. The core hub stays strict — an explicit `nil`
cache there fails.

Every download factory has an asynchronous peer that runs the fetch on a background queue and delivers
the built backend (or an error) to a completion handler, so the caller does not hand-thread the blocking
download. It is the `…completionHandler:` selector alongside the blocking `…error:` one, on
``NFKMLXHub`` and on each model's direct factory:

```swift
NFKMLXRealESRGAN.backend(variant: .x4, repo: "your-org/real-esrgan",
                         weightsPath: "RealESRGAN_x4.safetensors",
                         revision: nil, cacheDirectoryURL: nil) { backend, error in
    // runs on the download's background queue; hop to the main thread if the UI needs it
}
```

### Loading safetensors, with the conv transpose

`loadWeights(into:from:)` reads a safetensors checkpoint (`loadArrays` → `update(parameters:)`),
transposing convolution weights from PyTorch layout to MLX layout:

- 4-D conv `[out, in, kH, kW]` → `[out, kH, kW, in]`
- 3-D Conv1d `[out, in, k]` → `[out, k, in]`
- transposed conv swaps the in/out axes

The module structure and parameter names mirror the reference PyTorch model, so a converted checkpoint
loads without a per-key remap where the names already match.

### Reading a PyTorch checkpoint natively

A raw PyTorch checkpoint (`.pth`, `.pt`, `.ckpt`, `.th`, HF `.bin`) loads with no Python toolchain.
The checkpoint loader sniffs a file's leading bytes, so every `weightsURL:` factory accepts a raw
file wherever it accepts a converted safetensors: both the modern ZIP container and the pre-1.6
stream parse, the file stays memory-mapped, and no pickle code ever executes — globals resolve
against a fixed dense-tensor table and everything else becomes inert data. Training wrappers
(`state_dict`, `params_ema`, …) unwrap, non-tensor sidecars drop, and tensors stored as strided
views gather to row-major. ``NFKMLXTorchCheckpoint`` is the consumer API over the same reader:
inspect a state dict's names and shapes, read a tensor's bytes, or convert to safetensors on device
with `writeSafetensors(to:)`. A TorchScript archive, a checkpoint whose pickle wraps its weights in
a framework class, and a `.nemo` tar are refused with an error naming the offline converter to use.

### The offline converters

Each model has an offline converter under `Tools/<model>-to-safetensors/` that turns a released `.pth` /
`.pt` / `.ckpt` into safetensors. Where a checkpoint's key layout differs from the module's, the
converter renames keys (for example `rebnconvN` → `enc`/`dec` for U²-Net); several print the checkpoint's
keys with `--list-keys` so the remap can be worked out. The intricate ones (Depth Anything, the
colorizer) are self-validating: they match every key against the module's expected layout and report
mismatches.

## Topics

### Download

- ``NFKMLXHub``

### Registry

- ``NFKMLXModelRegistry``
- ``NFKMLXReferenceModels``

### PyTorch checkpoints

- ``NFKMLXTorchCheckpoint``
- ``NFKMLXTorchTensorInfo``
- ``NFKMLXTorchScalarType``
