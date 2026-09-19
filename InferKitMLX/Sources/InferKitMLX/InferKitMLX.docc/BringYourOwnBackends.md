# Bring-your-own MLX backends

Supply an MLX forward closure; a base backend handles the InferKit contract and the image or audio
bridge.

## Overview

When a model is not in the gallery, you provide only its MLX forward and pick the base backend that
matches the model's shape. Each base reads the InferKit request, bridges pixels or samples to and from
`MLXArray`, runs your closure, and writes the result under the right key.

![The base backends and the closure each one drives.](backend-families)

| Backend | Closure | Shape |
| --- | --- | --- |
| ``NFKMLXModuleBackend`` | `(MLXArray) -> MLXArray` | image → image (RGB/RGBA bridge) |
| ``NFKMLXMattingBackend`` | `(plate, hint) -> [H,W,4]` | keyer / background remover |
| ``NFKMLXTensorBackend`` | `[String: MLXArray] -> [String: MLXArray]` | multi-input / multi-output |
| ``NFKMLXSpeechBackend`` | `(String) -> MLXArray` | text → mono waveform |
| ``NFKMLXDiffusionBackend`` | encode / denoise / decode + scheduler | iterative sampler |

```swift
import MLX
import InferKitMLX

let backend = NFKMLXModuleBackend(identifier: "invert", isReady: true) { image in
    1.0 - image   // your MLX forward
}
let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: cgImage]))
```

### Declaring the keys a closure reads

A backend declares the request keys it acts on through the protocol's `supportedParameterKeys` and
`supportedInputKeys`, which a caller reads to configure the engine and a bridge reads to report what
the engine can do. A base backend knows its own keys and not the closure's, so a closure that reads
more names them:

```swift
let styled = NFKMLXModuleBackend(identifier: "style",
                                 forwardInputKeys: [NFKInputControl],
                                 forwardParameterKeys: [NFKParameterStrength]) { image, request in
    stylize(image, style: request.input(forKey: NFKInputControl))
}
styled.supportedInputKeys      // [NFKInputImage, NFKInputControl]
```

``NFKMLXMattingBackend`` takes the same two, ``NFKMLXDiffusionBackend`` takes `encodedInputKeys` and
`encodedParameterKeys` for what its `encode` closure reads, and ``NFKMLXTensorBackend`` derives its
inputs from the configured ports.

### Registering a closure by name

A model author registers a factory by name from Swift; an Objective-C consumer then builds it through
``NFKMLXModelRegistry`` without writing Swift. ``NFKMLXReferenceModels`` shows the pattern —
`registerGreenScreenKeyer` and `registerToneSpeech` are shipped references, and `registerAll()`
registers every gallery model plus these stand-ins.

The `NFKMLXDiffusionBackend` closures (`encode` / `denoise` / `decode`) drive an iterative sampler; that
loop and its scheduler seam are covered in <doc:DiffusionAndSchedulers>.

## Topics

### Backends

- ``NFKMLXModuleBackend``
- ``NFKMLXMattingBackend``
- ``NFKMLXTensorBackend``
- ``NFKMLXSpeechBackend``

### Registry

- ``NFKMLXModelRegistry``
- ``NFKMLXReferenceModels``
