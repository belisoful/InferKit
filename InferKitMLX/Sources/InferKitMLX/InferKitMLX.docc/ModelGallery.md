# Model gallery

Every shipped MLX model, grouped by task, with its registered name and the value type it returns.

## Overview

Each model is a real MLX implementation (in `MLXNN`), not a stand-in, measured against its reference
implementation on the released weights and run through one of the base backends. Two access paths reach
the same model:

- **Direct factory** — `SomeModel.backend(…weightsURL:)` builds it from local weights (`nil` → random
  init). A download companion, `SomeModel.backend(…repo:weightsPath:revision:cacheDirectoryURL:)`,
  fetches the checkpoint from Hugging Face first. Variant models take an `@objc` enum
  (``NFKMLXRealESRGANVariant``, ``NFKMLXDepthVariant``, ``NFKMLXU2NetVariant``, ``NFKMLXYOLOVariant``,
  ``NFKMLXNAFNetVariant``, ``NFKMLXSwinIRVariant``, ``NFKMLXSAMVariant``, ``NFKMLXWhisperVariant``).
  A model whose geometry lives in a release's `config.json` takes `backend(directoryURL:)` instead
  and reads the whole downloaded release.
- **By name** — ``NFKMLXReferenceModels/registerAll()`` registers every model, then
  ``NFKMLXModelRegistry/backend(named:weightsURL:)`` (local) or
  ``NFKMLXHub/backend(named:repo:weightsPath:revision:cacheDirectoryURL:)`` (download) builds by its
  registered name.

```swift
// By name, after registerAll():
NFKMLXReferenceModels.registerAll()
let depth = try NFKMLXModelRegistry.backend(named: "depth-anything-v2-small", weightsURL: url)
```

Every backend runs `runInference(for:)` synchronously and multi-second — call it off the render thread,
or submit a job for progress and cancellation. <doc:ModelIndex> lists, per model, the network class, the
configuration preset behind each registered name, and a construction line to copy.

### Image → image

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXRealESRGAN`` | `real-esrgan-x4` · `-anime` · `-x2` | ×4 / ×2 super-resolution |
| ``NFKMLXSwinIR`` | `swinir-x4` | transformer super-resolution (×2 / ×3 / ×4 / ×8, lightweight) |
| ``NFKMLXNAFNet`` | `nafnet` | denoise / deblur (SIDD, GoPro, REDS) |
| ``NFKMLXZeroDCE`` | `zero-dce` | low-light enhancement |
| ``NFKMLXStyleTransfer`` | `fast-style-transfer` | one baked style per checkpoint |
| ``NFKMLXColorizer`` | `colorizer-eccv16` | grayscale → color |
| ``NFKMLXSiggraphColorizer`` | `colorizer-siggraph17` | colorization with optional user hints |
| ``NFKMLXLaMa`` | `lama-inpaint` | mask-guided inpainting |
| ``NFKMLXStableDiffusionInpaint`` | `sd-inpaint` | latent-diffusion inpainting |
| ``NFKMLXCodeFormer`` | `codeformer` | face restoration; ``NFKMLXPhotoFaceBackend`` restores every face in a photograph |
| ``NFKMLXTAESD`` | `taesd` | tiny autoencoder: fast latent preview encode / decode |
| ``NFKMLXSDUNet`` / ``NFKMLXSDAutoencoder`` | — | the shared Stable Diffusion networks |

Each writes its result under `NFKOutputImage`.

### Image → map

`NFKMLXDepthAnything` and `NFKMLXMarigold` emit a grayscale depth map; the segmenters
(`NFKMLXSegFormer`, `NFKMLXDeepLab`, `NFKMLXBiSeNet`, `NFKMLXBiSeNetV2`) emit a grayscale class-label
map under `NFKOutputImage` — recover the class index as `round(gray · (classCount − 1))`.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXDepthAnything`` | `depth-anything-v2-small` · `-base` · `-large` | monocular depth |
| ``NFKMLXMarigold`` | `marigold-depth` | diffusion depth |
| ``NFKMLXSegFormer`` | `segformer-b0` | transformer segmentation |
| ``NFKMLXDeepLab`` | `deeplabv3` | CNN segmentation |
| ``NFKMLXBiSeNet`` | `bisenet` | real-time segmentation (ResNet-18 context path) |
| ``NFKMLXBiSeNetV2`` | `bisenet-v2` | real-time segmentation (detail + semantic branches) |

### Matting, segmentation & faces

`NFKMLXU2Net`, `NFKMLXRVM`, and `NFKMLXMODNet` produce a straight foreground plus an alpha matte under
`NFKOutputMask`; `NFKMLXSAM` segments from a point prompt; `NFKMLXRetinaFace` returns
`[NFKDetection]` with five-point landmarks through ``NFKMLXRetinaFaceDetector``.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXU2Net`` | `u2net` · `u2netp` | salient-object matting |
| ``NFKMLXRVM`` | `robust-video-matting` | recurrent video matting |
| ``NFKMLXMODNet`` | `modnet` | trimap-free portrait matting |
| ``NFKMLXSAM`` | `sam` | promptable segmentation |
| ``NFKMLXSAM2`` | — | SAM 2: Hiera encoder, prompt encoder, mask decoder, and the video memory path |
| ``NFKMLXRetinaFace`` | `retinaface-mobile025` | face detection with landmarks |
| ``NFKMLXFaceAlignment`` | — | five-point alignment to the CodeFormer template (RetinaFace or Vision) |

### Detection & pose

`NFKMLXYOLO` and `NFKMLXRTDetr` return `[NFKDetection]` under `NFKOutputDetections`; `NFKMLXPose`
returns `[NFKKeypoint]` under `NFKOutputPose`.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXYOLO`` | `yolo` | object detection (YOLOv8 n / s / m / l / x) |
| ``NFKMLXRTDetr`` | `rtdetr` | object detection, Apache-2.0 (RT-DETR r50vd; no NMS) |
| ``NFKMLXPose`` | `pose-simplebaseline` | top-down pose |

### Embeddings & reranking

The embedders return an L2-normalized vector under `NFKOutputEmbedding` — the key the core's
`NFKRemoteEmbeddingBackend` answers with too, so search code is engine-agnostic.
``NFKMLXModernBERTReranker`` scores a query against a list of documents and is an object rather than
a backend.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXCLIP`` | `clip-vit-b-32` | image + text embeddings (CLIP) |
| ``NFKMLXSigLIP2`` | `siglip2-base-patch16-224` | image + text embeddings (SigLIP 2, multilingual) |
| ``NFKMLXQwen3Embedding`` | — | text embeddings (Qwen3-Embedding-0.6B, Matryoshka) |
| ``NFKMLXEmbeddingGemma`` | — | text embeddings (EmbeddingGemma-300M, bidirectional) |
| ``NFKMLXModernBERTReranker`` | — | cross-encoder reranking (gte-reranker-modernbert-base) |

### Language models

``NFKMLXLanguageBackend`` generates text from a downloaded release directory
(``NFKMLXLanguage/backend(directoryURL:)``) or a GGUF file (``NFKMLXLanguage/backend(ggufURL:)``):
`NFKInputPrompt` or `NFKInputMessages` in, `NFKOutputText` out, with the release's own chat template
rendered by ``NFKMLXChatTemplateRenderer``. It streams through the job, and takes a context window,
key-value cache quantization, chunked prefill, a prompt cache, a draft model for speculative decoding,
and JSON or fixed-choice constrained decoding — each also settable from Objective-C through
``NFKMLXGenerationParameterKey``.

| Model | Factory | Architecture |
| --- | --- | --- |
| ``NFKMLXLanguage`` | `backend(directoryURL:)` | dense decoders (Qwen3, Qwen2, Llama) and the Qwen3-MoE / Mixtral mixtures |
| ``NFKMLXLanguage`` | `backend(ggufURL:)` | any dense `llama` / `qwen2` / `qwen3` GGUF (Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K) |
| ``NFKMLXHybridLanguage`` | — | Qwen3.5 / 3.6 / 3.8: gated delta-rule recurrence with full attention every fourth layer |
| ``NFKMLXGemmaLanguage`` | `backend(directoryURL:)` | Gemma 4 (E-series, 26B-A4B mixture, 12B unified) through ``NFKMLXGemmaBackend`` |
| ``NFKMLXDeepSeek`` | — | DeepSeek V4: multi-head latent attention over a mixture of experts (the arithmetic is measured; the released weights exceed a workstation) |
| ``NFKMLXGemma2Net`` | — | Gemma 2, the SANA text encoder |
| ``NFKMLXT5Encoder`` | `encoder(configuration:directory:)` | T5 v1.1 and umT5 encoders, the LTX and Wan text conditioning |

``NFKMLXModelSizing`` answers whether a release fits the machine before any weight loads.

### Vision-language

| Model | Factory | Task |
| --- | --- | --- |
| ``NFKMLXSmolVLM`` | `smolVLM(directoryURL:)` | an image and a question → an answer (SmolVLM2-500M) |
| ``NFKMLXQwen3VL`` | — | the Qwen3-VL-2B vision tower (2-D rotary ViT, deepstack) |
| ``NFKMLXGemma4ConditionalGeneration`` | — | the tri-modal Gemma 4 chain: image and audio towers fused into the decoder |

### Video

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXRIFE`` | `rife` | frame interpolation (HDv3, midpoint) |
| ``NFKMLXRIFEv4`` | `rife-v4` | frame interpolation (v4, any timestep) |
| ``NFKMLXRAFT`` | `raft` | optical flow |
| ``NFKMLXVideoSR`` | `video-super-resolution` | recurrent video super-resolution (BasicVSR ×4) |
| ``NFKMLXSDUpscaler`` | `sd-x4-upscaler` | diffusion ×4 upscaler |
| ``NFKMLXVideoBackend`` | — | the clip backend: an `NFKVideoAsset` in, every frame through a sequence transform, a new clip out |

The recurrent models (`NFKMLXRVM`, `NFKMLXVideoSR`) thread a hidden state across frames through their
`*Net.forward` / `upscaleSequence` entry points; `NFKMLXRIFE.clipBackend` and
`NFKMLXVideoSR.clipBackend` run a whole clip through ``NFKMLXVideoBackend``.

### Audio

The separators read `NFKInputAudio` and return one or more `NFKAudioAsset`s; the codecs return the
reconstruction under `NFKOutputAudio` and expose their tokens through `encode` / `decode`;
`NFKMLXWhisper` returns text (and `[NFKAudioSegment]` under `NFKOutputSegments` when timestamps are
on); the voice-activity detectors return `[NFKAudioSegment]`; `NFKMLXAudioTagger` returns
`[NFKClassification]` under `NFKOutputClassifications`.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXDemucs`` | `demucs` | 4-stem music separation (Demucs v2) |
| ``NFKMLXHTDemucs`` | `htdemucs` | 4-stem music separation (Hybrid Transformer Demucs v4) |
| ``NFKMLXConvTasNet`` | `conv-tasnet` | speech separation |
| ``NFKMLXDenoiser`` | `denoiser` | speech noise suppression |
| ``NFKMLXDAC`` | `dac` | neural audio codec (Descript, 44.1 / 24 / 16 kHz) |
| ``NFKMLXSNAC`` | `snac` | multi-scale neural audio codec (24 kHz speech) |
| ``NFKMLXWhisper`` | `whisper-tiny` | speech → text (tiny / small / medium / large-v3, timestamps) |
| ``NFKMLXParakeet`` | `parakeet-tdt` | speech → text (Parakeet-TDT 0.6B v2, FastConformer + token-and-duration transducer; per-token timestamps) |
| ``NFKMLXVAD`` | `vad-marblenet` | voice-activity detection (MarbleNet) |
| ``NFKMLXSileroVAD`` | `silero-vad` | voice-activity detection (Silero v6, streaming) |
| ``NFKMLXAudioTagger`` | `audio-tagger-panns` | audio tagging |

### Text → speech & music

| Model | Factory | Task |
| --- | --- | --- |
| ``NFKMLXVoice`` | `makeSpeechBackend(phonemize:)` | FastSpeech2 conformer + the paired HiFi-GAN (LJSpeech) |
| ``NFKMLXKokoro`` | `backend(directoryURL:voiceName:)` | Kokoro-82M (StyleTTS2 / iSTFTNet), a phoneme string in |
| ``NFKMLXChatterbox`` | `chatterbox` | text → speech in a cloned voice (VoiceEncoder + S3 tokenizer + T3 Llama + S3Gen flow matching + HiFT, 24 kHz) |
| ``NFKMLXTTS`` | — | a phonemizer, an acoustic model, and a vocoder chained by hand |
| ``NFKMLXMusicBackend`` | `NFKMLXMusic3.backend(directoryURL:)` | MiniMax Music 3: description + lyrics → stereo 44.1 kHz music (`minimax-music3`; separately licensed weights, ~27 GB or 7.7 GiB quantized) |

Every voice writes a WAV `NFKAudioAsset` under `NFKOutputAudio`, the container the core's
`NFKRemoteSpeechBackend` writes too.

### Text → image & video

The diffusion pipelines are covered in <doc:DiffusionAndSchedulers>.

| Pipeline | Networks | Task |
| --- | --- | --- |
| ``NFKMLXBackend`` / ``NFKMLXTextToImage`` | SD UNet + VAE + CLIP towers | Stable Diffusion 1.5, 2.1, SDXL-Turbo |
| ``NFKMLXZImagePipeline`` | ``NFKMLXZImageTransformerNet`` + the Flux VAE + Qwen3 | Z-Image text-to-image and image-to-image |
| ``NFKMLXSANAPipeline`` | ``NFKMLXSANATransformerNet`` + ``NFKMLXDCAutoencoderNet`` + Gemma 2 | SANA text-to-image |
| ``NFKMLXLTXPipeline`` | ``NFKMLXLTXTransformer`` + ``NFKMLXLTXVideoVAE`` + T5-XXL | LTX-Video text-to-video |
| ``NFKMLXWanPipeline`` | ``NFKMLXWanTransformerNet`` + ``NFKMLXWanVideoVAENet`` + umT5 | Wan text-to-video |

## Topics

### Related

- <doc:ModelIndex>
- <doc:BringYourOwnBackends>
- <doc:DiffusionAndSchedulers>
- <doc:WeightsAndConversion>
