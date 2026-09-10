# Agent reference

Auxiliary maintainer notes behind `AGENTS.md` / `CLAUDE.md` (one document, two names). Those two files
carry only the rules and the entry points; everything else that agents accumulated while building the
toolkit lives here, one file per subject. A new note goes into the file for its subject, or into a new
file listed here, never back into `AGENTS.md` / `CLAUDE.md`. The Documentation Style rules in
`CLAUDE.md` apply to these files.

## Repository

- [build-and-verification.md](build-and-verification.md) — every build and test command, and the full
  pre-commit check with the reasons behind each step (tvOS through the SDK, the analyzer's fresh
  derived data, the MLX Metal library under `swift test`, one testable per scheme).
- [project-structure.md](project-structure.md) — the full repository tree with per-directory notes,
  including the `Tools/*-to-safetensors` converters and the workspace rules.
- [distribution-and-packaging.md](distribution-and-packaging.md) — SwiftPM and CocoaPods distribution,
  the version's three locations, and the core and MLX XCFramework builds.
- [documentation-docc.md](documentation-docc.md) — the DocC catalogs and why the core needs a
  `clang -extract-api` symbol graph.
- [mlx-parity-checklist.md](mlx-parity-checklist.md) — every listing to update when an InferKitMLX
  model reaches reference parity, and the house style each listing keeps.

## Core

- [core-runtime-notes.md](core-runtime-notes.md) — value-type accessors, the tokenizer class cluster,
  grammar-constrained sampling, and dynamic backend discovery.
- [remote-providers.md](remote-providers.md) — the provider presets, the shared transport, model
  catalogs, local runners, streaming, tools, structured output, media in and out, and the probes that
  verified each endpoint.
- [coreml-compute-plan.md](coreml-compute-plan.md) — `NFKComputePlan` and what was measured about
  where Core ML places a language model.
- [hardware-and-model-sizing.md](hardware-and-model-sizing.md) — `NFKHardwareProfile`, the memory
  ceilings, and the measured-bandwidth model sizing in the MLX companion.

## InferKitMLX

- [mlx-companion.md](mlx-companion.md) — the package, the bring-your-own backend seams, the `@objc`
  factories, registry, hub, and runtime wrappers.
- [mlx-runtime-gotchas.md](mlx-runtime-gotchas.md) — the mlx-swift hazards measured here (the Metal
  library, `Pool` padding, numeric keys, duplicate keys, subnormals, cache clearing, shape checks).
- [mlx-weights-and-formats.md](mlx-weights-and-formats.md) — runtime quantization, the release
  reader, and the native GGUF and PyTorch checkpoint readers.
- [mlx-training.md](mlx-training.md) — the trainer, training data, LoRA, and the fine-tuning recipes.

### Model classes

- [mlx-models-diffusion.md](mlx-models-diffusion.md) — the diffusion backend and schedulers, Stable
  Diffusion networks and pipelines, Marigold, the ×4 upscaler, IP-Adapter, TAESD.
- [mlx-models-dit-generation.md](mlx-models-dit-generation.md) — LTX-Video, Z-Image, SANA, Wan, SD3,
  FLUX, their samplers and text encoders, and the SD3 and FLUX ControlNets.
- [mlx-models-image-restoration.md](mlx-models-image-restoration.md) — Real-ESRGAN, NAFNet, LaMa,
  SwinIR, style transfer, Zero-DCE, the colorizers, CodeFormer.
- [mlx-models-depth-segmentation-matting.md](mlx-models-depth-segmentation-matting.md) — Depth
  Anything V2 and 3, U²-Net, SAM and SAM 2, MODNet, RVM, SegFormer, DeepLab, BiSeNet, the ResNet
  backbone.
- [mlx-models-detection-pose.md](mlx-models-detection-pose.md) — YOLOv8, RT-DETR, RF-DETR, RetinaFace,
  SimpleBaseline pose.
- [mlx-models-video.md](mlx-models-video.md) — RIFE, RAFT, the video backend and file layer, BasicVSR.
- [mlx-models-embeddings-retrieval.md](mlx-models-embeddings-retrieval.md) — CLIP, SigLIP 2,
  Qwen3-Embedding, EmbeddingGemma, the ModernBERT reranker.
- [mlx-models-language.md](mlx-models-language.md) — the dense decoder and its generation runtime
  (cache, quantized cache, speculative decoding, mixtures of experts, constrained decoding), the hybrid
  decoder, DeepSeek V4, rotary scaling, the Jinja chat-template renderer.
- [mlx-models-gemma.md](mlx-models-gemma.md) — Gemma 2, Gemma 3, Gemma 3n, and the Gemma 4 text,
  vision, audio, and fusion stack.
- [mlx-models-vision-language.md](mlx-models-vision-language.md) — SmolVLM2 and Qwen3-VL.
- [mlx-models-speech-recognition.md](mlx-models-speech-recognition.md) — Whisper, Parakeet, MarbleNet
  and Silero VAD, PANNs audio tagging.
- [mlx-models-text-to-speech.md](mlx-models-text-to-speech.md) — Chatterbox, the phonemizers,
  FastSpeech2 + HiFi-GAN, Kokoro.
- [mlx-models-source-separation.md](mlx-models-source-separation.md) — Demucs v2 and v4, Conv-TasNet,
  the Demucs denoiser.
- [mlx-models-speech-restoration.md](mlx-models-speech-restoration.md) — the shared STFT primitives and
  the fifteen restoration models from MP-SENet to Apollo.
- [mlx-models-audio-codecs-music.md](mlx-models-audio-codecs-music.md) — DAC, SNAC, MiniMax Music 3.

## InferKitFoundationModels

- [foundation-models-companion.md](foundation-models-companion.md) — the Foundation Models backend,
  tools, structured output, and the provider bridge.
