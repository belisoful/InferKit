# ``NFKMLXRealESRGAN``

## Overview

Real-ESRGAN super-resolution — the RRDBNet generator implemented in `MLXNN` (norm-free `Conv2d` +
`leakyRelu`), run through ``NFKMLXModuleBackend``. Three variants ship through
``NFKMLXRealESRGANVariant``: `.x4` (23 blocks), `.anime` (6 blocks, lighter), and `.x2` (pixel-unshuffle
front-end). The result is written under `NFKOutputImage`.

### From local weights

`weightsURL` points at a safetensors checkpoint; `nil` builds with random weights (`isReady` true),
useful for wiring and tests.

```swift
import InferKit
import InferKitMLX

let backend = try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: weightsURL)
let request = NFKInferenceRequest(inputs: [NFKInputImage: cgImage])
let upscaled = try backend.runInference(for: request).outputForKey(NFKOutputImage)
```

### Downloading the checkpoint

The download companion fetches from Hugging Face first, then builds. It blocks on the network — run it
off the render thread.

```swift
let backend = try NFKMLXRealESRGAN.backend(
    variant: .x4,
    repo: "your-org/real-esrgan",
    weightsPath: "RealESRGAN_x4.safetensors",
    revision: nil,
    cacheDirectoryURL: nil   // → Application Support/InferKit/models
)
```

For a fetch that does not block the caller, the download factory has an asynchronous peer that runs on a
background queue and delivers the backend (or an error) to a completion handler:

```swift
NFKMLXRealESRGAN.backend(variant: .x4, repo: "your-org/real-esrgan",
                         weightsPath: "RealESRGAN_x4.safetensors",
                         revision: nil, cacheDirectoryURL: nil) { backend, error in
    // the handler runs on the download's background queue
}
```

A `.pth` release converts to safetensors first with `Tools/realesrgan-to-safetensors/convert.py`, which
transposes the 4-D conv weights to MLX layout. See <doc:WeightsAndConversion>.

## Topics

### Variant

- ``NFKMLXRealESRGANVariant``

### Related

- <doc:ModelGallery>
- ``NFKMLXModuleBackend``
