# ``InferKitMLX``

MLX-backed inference on Apple Silicon — a gallery of ready-to-run models across image, video, audio,
language, and generative diffusion, plus bring-your-own-model backends, all behind the InferKit contract.

@Metadata {
    @DisplayName("InferKitMLX")
}

## Overview

This companion adds [MLX](https://github.com/ml-explore/mlx-swift)-backed inference on top of the
InferKit core. It keeps MLX out of the core, so the core stays cross-platform and dependency-free while
this package targets Apple Silicon (macOS 14 / iOS 17). Every model adopts the same `NFKInferenceBackend`
protocol, so an InferKit consumer runs an MLX model exactly like any other backend.

Three ways to use it:

- **Shipped models** — sixty-plus real models, each implemented in `MLXNN` and measured against its
  reference implementation on the released weights, each with a public `@objc` factory. See
  <doc:ModelGallery> for what each does and <doc:ModelIndex> for the class, configuration, and a
  construction line to copy.
- **Generative pipelines** — Stable Diffusion, Z-Image, and SANA text-to-image; LTX-Video and Wan
  text-to-video; on-device language models (Qwen3, Qwen3.5, Gemma 4, DeepSeek V4, any dense GGUF); and
  MiniMax Music 3 text-to-music. See <doc:DiffusionAndSchedulers> and ``NFKMLXLanguageBackend``.
- **Bring your own** — supply an MLX forward closure and let a base backend handle the InferKit contract
  and the image/audio bridge. See <doc:BringYourOwnBackends>.

```swift
import InferKit
import InferKitMLX

// A shipped model through its @objc factory (nil weights → random init, useful for wiring/tests):
let upscaler = try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: weightsURL)
let result = try upscaler.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: cgImage]))

// A language model from a downloaded Hugging Face release directory:
let llm = try NFKMLXLanguage.backend(directoryURL: releaseDirectory)
let reply = try llm.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "Explain diffraction."])).text
```

Weights are downloaded at runtime, not bundled — for size, licensing, and update reasons. See
<doc:WeightsAndConversion>.

![The InferKitMLX model gallery grouped by modality: image, video, audio, generation, and language.](model-gallery)

### Activating core capabilities

Linking this package ships two dynamic-backend providers, so core capabilities light up with no
registration:

- ``NFKStableDiffusionProvider`` → the core's `stable-diffusion` capability (``NFKMLXBackend``, SD 1.5:
  the ungated release, so it activates with no credential).
- ``NFKMLXWhisperProvider`` → the core's `transcription` capability (`NFKMLXWhisper`).

### Sizing a model against the machine

``NFKMLXModelSizing`` counts a decoder's parameters from its geometry before anything is allocated,
measures the machine's memory bandwidth, and answers whether a release fits the working set — and at
what context window. ``NFKMLXGPU`` reports what the machine has and sets standing cache and memory
limits.

## Topics

### Concepts

- <doc:ModelGallery>
- <doc:ModelIndex>
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
- ``NFKMLXSiggraphColorizer``
- ``NFKMLXLaMa``
- ``NFKMLXStableDiffusionInpaint``
- ``NFKMLXTAESD``

### Image → map (depth & segmentation)

- ``NFKMLXDepthAnything``
- ``NFKMLXDepthAnything3``
- ``NFKMLXMarigold``
- ``NFKMLXSegFormer``
- ``NFKMLXDeepLab``
- ``NFKMLXBiSeNet``
- ``NFKMLXBiSeNetV2``

### Matting, segmentation & faces

- ``NFKMLXU2Net``
- ``NFKMLXRVM``
- ``NFKMLXMODNet``
- ``NFKMLXSAM``
- ``NFKMLXSAM2``
- ``NFKMLXCodeFormer``
- ``NFKMLXPhotoFaceBackend``
- ``NFKMLXFaceAlignment``
- ``NFKMLXRetinaFace``
- ``NFKMLXRetinaFaceDetector``
- ``NFKMLXVisionFaceDetector``

### Detection & pose

- ``NFKMLXYOLO``
- ``NFKMLXRTDetr``
- ``NFKMLXPose``

### Embeddings & reranking

- ``NFKMLXCLIP``
- ``NFKMLXSigLIP2``
- ``NFKMLXQwen3Embedding``
- ``NFKMLXEmbeddingGemma``
- ``NFKMLXTextEmbeddingBackend``
- ``NFKMLXModernBERTReranker``

### Language models

- ``NFKMLXLanguage``
- ``NFKMLXLanguageBackend``
- ``NFKMLXGenerationOptions``
- ``NFKMLXGenerationParameterKey``
- ``NFKMLXKeyValueCache``
- ``NFKMLXPromptCache``
- ``NFKMLXSpeculativeReport``
- ``NFKMLXJSONConstraint``
- ``NFKMLXChoiceConstraint``
- ``NFKMLXVocabulary``
- ``NFKMLXHybridLanguage``
- ``NFKMLXGemmaLanguage``
- ``NFKMLXGemmaBackend``
- ``NFKMLXDeepSeek``
- ``NFKMLXGemma2Net``
- ``NFKMLXT5Encoder``
- ``NFKMLXRoPEScaling``
- ``NFKMLXModelSizing``
- ``NFKMLXModelFit``

### Vision-language

- ``NFKMLXSmolVLM``
- ``NFKMLXSigLIPNet``
- ``NFKMLXSmolVLMConnector``
- ``NFKMLXQwen3VL``
- ``NFKMLXQwen3VLVisionNet``
- ``NFKMLXGemma4VisionNet``
- ``NFKMLXGemma4UnifiedNet``
- ``NFKMLXGemma4AudioNet``
- ``NFKMLXGemma4ImageProcessor``
- ``NFKMLXGemma4AudioFeatureExtractor``
- ``NFKMLXGemma4MultimodalEmbedder``
- ``NFKMLXGemma4Fusion``
- ``NFKMLXGemma4ConditionalGeneration``

### Video

- ``NFKMLXRIFE``
- ``NFKMLXRIFEv4``
- ``NFKMLXRAFT``
- ``NFKMLXVideoSR``
- ``NFKMLXSDUpscaler``
- ``NFKMLXVideoBackend``
- ``NFKMLXVideoFile``

### Audio → audio

- ``NFKMLXDemucs``
- ``NFKMLXHTDemucs``
- ``NFKMLXConvTasNet``
- ``NFKMLXDenoiser``
- ``NFKMLXDAC``
- ``NFKMLXSNAC``

### Audio → text & labels

- ``NFKMLXWhisper``
- ``NFKMLXParakeet``
- ``NFKMLXVAD``
- ``NFKMLXSileroVAD``
- ``NFKMLXAudioTagger``

### Text → speech

- ``NFKMLXTTS``
- ``NFKMLXVoice``
- ``NFKMLXFastSpeech2``
- ``NFKMLXHiFiGAN``
- ``NFKMLXKokoro``
- ``NFKMLXChatterbox``
- ``NFKMLXChatterboxTTS``
- ``NFKMLXPhonemizer``
- ``NFKMLXNeuralG2P``
- ``NFKMLXEspeakPhonemizer``

### Text → music

- ``NFKMLXMusic3``
- ``NFKMLXMusicBackend``

### Text → image

- ``NFKMLXBackend``
- ``NFKMLXTextToImage``
- ``NFKMLXSDPipeline``
- ``NFKMLXStableDiffusionModels``
- ``NFKMLXSDTextEncoder``
- ``NFKMLXSDPromptTokenizer``
- ``NFKMLXZImagePipeline``
- ``NFKMLXZImageTransformerNet``
- ``NFKMLXSANAPipeline``
- ``NFKMLXSANATransformerNet``
- ``NFKMLXDCAutoencoderNet``
- ``NFKMLXIPAdapterImageProjection``
- ``NFKMLXIPAdapterAttention``

### Text → video

- ``NFKMLXLTXPipeline``
- ``NFKMLXLTXVideoVAE``
- ``NFKMLXLTXTransformer``
- ``NFKMLXWanPipeline``
- ``NFKMLXWanTransformerNet``
- ``NFKMLXWanVideoVAENet``

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

### Chat templates

- ``NFKMLXChatTemplateRenderer``
- ``NFKMLXChatTemplate``

### Weights & formats

- ``NFKMLXTorchCheckpoint``
- ``NFKMLXGGUF``
- ``NFKMLXGGUFTensorInfo``
- ``NFKMLXWeightPrecision``

### Runtime

- ``NFKMLXRandom``
- ``NFKMLXGPU``
- ``NFKMLXDevice``
- ``NFKMLXDeviceType``
