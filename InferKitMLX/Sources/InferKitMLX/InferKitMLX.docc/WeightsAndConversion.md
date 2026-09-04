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
views gather to row-major. A checkpoint that pickled a live `nn.Module`
tree (YOLO's ultralytics DetectionModel), a TorchScript archive (CLIP, whose scripted module stores
its weights under attribute-keyed state), and a `.nemo` tar all load — no class is constructed and no
serialized code is interpreted, only the pickle's tensor records and module attributes are read.
``NFKMLXTorchCheckpoint`` is the consumer API over the same reader: inspect a state dict's names and
shapes, read a tensor's bytes, or convert to safetensors on device with `writeSafetensors(to:)`.

### Reading a GGUF file natively

``NFKMLXGGUF`` reads the container most quantized language models are distributed in: the typed
metadata, the tensor table, and the `F32` / `F16` / `Q4_0` / `Q5_0` / `Q8_0` / `Q4_K` / `Q6_K`
dequantizers, bit-exact against the reference `gguf` package. A type it does not implement leaves the
tensor listed and refuses only the read. `NFKMLXLanguage.backend(ggufURL:)` builds a text-generation
backend straight from a dense `llama` / `qwen2` / `qwen3` file, undoing llama.cpp's rotary permutation
and rebuilding the embedded tokenizer.

### Whole releases: sharded, at a precision, quantized

A language or diffusion release is a directory rather than one file — `config.json`, the tokenizer,
and weights split across shards named by `model.safetensors.index.json`. The `backend(directoryURL:)`
factories read the whole tree, and ``NFKMLXWeightPrecision`` chooses whether a bf16 release loads at
the precision it ships in or at float32; the parity records were measured at float32, which costs
twice the memory. Runtime MLX quantization packs a model's linear layers to 4 or 8 bits, and a saved
quantized checkpoint records its bit width in metadata so the loaders rebuild the packed structure
before applying it. `NFKMLXMusic3.quantizeRelease(at:to:)` writes a quantized copy of a release in
the release's own layout.

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

### GGUF

- ``NFKMLXGGUF``
- ``NFKMLXGGUFTensorInfo``
- ``NFKMLXGGMLType``

### Precision

- ``NFKMLXWeightPrecision``

### PyTorch checkpoints

- ``NFKMLXTorchCheckpoint``
- ``NFKMLXTorchTensorInfo``
- ``NFKMLXTorchScalarType``
