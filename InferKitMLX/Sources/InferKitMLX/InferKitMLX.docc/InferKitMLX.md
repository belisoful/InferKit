# ``InferKitMLX``

MLX-backed inference on Apple Silicon — a gallery of ready-to-run models across image, video, and audio,
plus bring-your-own-model backends, all behind the InferKit contract.

@Metadata {
    @DisplayName("InferKitMLX")
}

## Overview

This companion adds [MLX](https://github.com/ml-explore/mlx-swift)-backed inference on top of the
InferKit core. It keeps MLX out of the core, so the core stays cross-platform and dependency-free while
this package targets Apple Silicon (macOS 14 / iOS 17). Every model adopts the same `NFKInferenceBackend`
protocol, so an InferKit consumer runs an MLX model exactly like any other backend.

Two ways to use it:

- **Shipped models** — thirty-plus real single-forward and recurrent models, each with a public `@objc`
  factory. See <doc:ModelGallery>.
- **Bring your own** — supply an MLX forward closure and let a base backend handle the InferKit contract
  and the image/audio bridge. See <doc:BringYourOwnBackends>.

```swift
import InferKit
import InferKitMLX

// A shipped model through its @objc factory (nil weights → random init, useful for wiring/tests):
let upscaler = try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: weightsURL)
let result = try upscaler.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: cgImage]))
```

Weights are downloaded at runtime, not bundled — for size, licensing, and update reasons. See
<doc:WeightsAndConversion>.

![The InferKitMLX model gallery grouped by modality: image, video, and audio tasks.](model-gallery)

### Activating core capabilities

Linking this package ships two dynamic-backend providers, so core capabilities light up with no
registration:

- ``NFKStableDiffusionProvider`` → the core's `stable-diffusion` capability (``NFKMLXBackend``, SD 1.5:
  the ungated release, so it activates with no credential).
- ``NFKMLXWhisperProvider`` → the core's `transcription` capability (`NFKMLXWhisper`).

## Topics

### Concepts

- <doc:ModelGallery>
- <doc:BringYourOwnBackends>
- <doc:DiffusionAndSchedulers>
- <doc:WeightsAndConversion>

### Image → image

- ``NFKMLXRealESRGAN``
- ``NFKMLXSwinIR``
- ``NFKMLXNAFNet``
- ``NFKMLXZeroDCE``
- ``NFKMLXStyleTransfer``
- ``NFKMLXColorizer``
- ``NFKMLXLaMa``
- ``NFKMLXStableDiffusionInpaint``

### Image → map (depth & segmentation)

- ``NFKMLXDepthAnything``
- ``NFKMLXMarigold``
- ``NFKMLXSegFormer``
- ``NFKMLXDeepLab``
- ``NFKMLXBiSeNet``

### Matting & segmentation

- ``NFKMLXU2Net``
- ``NFKMLXRVM``
- ``NFKMLXMODNet``
- ``NFKMLXSAM``
- ``NFKMLXSAM2``
- ``NFKMLXCodeFormer``

### Detection & pose

- ``NFKMLXYOLO``
- ``NFKMLXPose``

### Embeddings

- ``NFKMLXCLIP``

### Video

- ``NFKMLXRIFE``
- ``NFKMLXRAFT``
- ``NFKMLXVideoSR``
- ``NFKMLXSDUpscaler``
- ``NFKMLXSDPipeline``

### Audio → audio

- ``NFKMLXDemucs``
- ``NFKMLXHTDemucs``
- ``NFKMLXConvTasNet``
- ``NFKMLXDenoiser``

### Audio → text & labels

- ``NFKMLXWhisper``
- ``NFKMLXVAD``
- ``NFKMLXAudioTagger``

### Text → speech

- ``NFKMLXTTS``

### Text → music

- ``NFKMLXMusic3``
- ``NFKMLXMusicBackend``

### Text → image

- ``NFKMLXBackend``
- ``NFKMLXTextToImage``
- ``NFKMLXSDTextEncoder``
- ``NFKMLXSDPromptTokenizer``

### Bring-your-own backends

- ``NFKMLXModuleBackend``
- ``NFKMLXMattingBackend``
- ``NFKMLXTensorBackend``
- ``NFKMLXSpeechBackend``
- ``NFKMLXDiffusionBackend``

### Registry, download & discovery

- ``NFKMLXReferenceModels``
- ``NFKMLXModelRegistry``
- ``NFKMLXHub``
- ``NFKStableDiffusionProvider``
- ``NFKMLXWhisperProvider``

### Customizing a model

- ``NFKMLXTrainer``
- ``NFKMLXTrainingCheckpoint``
- ``NFKMLXTrainingStep``
- ``NFKMLXTrainingData``
- ``NFKMLXBatchSampler``
- ``NFKMLXLoRA``
- ``NFKMLXLoRALinear``
- ``NFKMLXZeroDCENet``
- ``NFKMLXZeroDCEObjective``
- ``NFKMLXZeroDCELoss``
- ``NFKMLXSegFormerNet``
- ``NFKMLXSegFormerObjective``
- ``NFKMLXSegFormerTrainable``
- ``NFKMLXCLIPNet``
- ``NFKMLXCLIPProbe``
- ``NFKMLXCLIPProbeBackend``
- ``NFKMLXWhisperNet``
- ``NFKMLXWhisperObjective``

### Runtime

- ``NFKMLXRandom``
- ``NFKMLXGPU``
- ``NFKMLXDevice``
- ``NFKMLXDeviceType``
