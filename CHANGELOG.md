# Changelog

All notable changes to InferKit are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0.0, a minor bump may change public API. SwiftPM treats a 0.x minor as
breaking, so `from: "0.1.0"` resolves 0.1.x only and a consumer opts into each minor deliberately.

## [0.1.0] — unreleased

First public release.

### Core (`InferKit`)

- `NFKInferenceBackend`, `NFKInferenceRequest`, `NFKInferenceResult`, and the thread-safe
  `NFKInferenceJob` handle, with a shared input/parameter/output key vocabulary.
- Backends with no third-party dependency: the passthrough mock, in-process Core ML (image and tensor),
  an on-device Core ML language model, OpenAI-compatible chat and transcription clients, and a
  submit-poll-fetch base for generation services.
- `NFKDynamicBackend` — runtime discovery, so an optional engine activates only when its classes are
  linked, with no build dependency on it.
- `NFKRemoteProvider` — named presets for the services a consumer is likely to call: OpenAI, xAI Grok,
  Google Gemini, Groq, Mistral, DeepSeek, Together, OpenRouter, and the local servers Ollama, LM
  Studio, llama.cpp, and vLLM, all OpenAI-compatible. `NFKAnthropicBackend` covers Anthropic's Messages
  API, whose authentication, required fields, and response envelope differ.
- Value types for detections, pose keypoints, classifications, audio segments, video, and audio.
- Tokenizers as a class cluster: byte-level BPE, CLIP, WordPiece, and Unigram, built from tokenizer
  files.
- `NFKHFHub` — Hugging Face resolve, download, checksum, and cache, with bearer-token support for a
  gated repository through `accessToken` or `HF_TOKEN`.
- RGBA-interleaved ↔ planar tensor conversion and an `MLMultiArray` bridge.
- A DocC catalog, and examples compiled in both Objective-C and Swift so a documented snippet cannot
  drift from the API.

### InferKitMLX (companion)

- Around thirty-five models implemented in `MLXNN` and validated at measured reference parity against
  each model's own reference implementation, spanning image, video, audio, and text.
- Stable Diffusion text-to-image built entirely from this package's own components — CLIP text tower,
  UNet, DDIM sampler, autoencoder — at end-to-end parity for SD 1.5, SD 2.1, and SDXL-Turbo.
- On-device text generation: `NFKMLXLanguage`, a dense decoder (grouped-query attention, rotary
  embeddings, SwiGLU) with a key-value cache and sampling, at measured parity against Qwen3 0.6B, 1.7B,
  and 4B, including sharded checkpoints.
- `NFKMLXHybridLanguage` — the Qwen3.5 / 3.6 / 3.8 hybrid decoder (gated delta-rule recurrence with
  full attention every fourth layer). Built and verified structurally against the released Qwen3.8-27B
  checkpoint; its numerics are unmeasured, because the smallest release does not fit on the machine
  that built it.
- `NFKMLXDeepSeek` — the DeepSeek V4 decoder (multi-head latent attention over a mixture of experts),
  with `NFKMLXDeepSeekQuantization` decoding the block-scaled fp8 and 4-bit formats the release is
  stored in.
  Built and verified structurally against the released Flash checkpoint; unmeasured numerically, and
  verified more weakly than the hybrid because that release is quantized.
- `NFKMLXGemmaLanguage` — the Gemma 4 text decoder (per-layer input embeddings, sliding and full
  attention at different head widths, soft-capped logits). Built and verified structurally against the
  released E2B checkpoint; unmeasured numerically, for want of an oracle on this machine.
- DeepSeek V4 arithmetic measured against transformers' own implementation at a tiny all-sliding
  configuration — every layer exact, logits 0.9999999999 — which found and fixed the head collapse
  being built as the wrong class. The release's fp8/fp4 storage decodes exactly; DeepSeek V4 Pro is
  accounted structurally.
- A complete trained text-to-speech voice: the espnet FastSpeech2 conformer acoustic model at
  reference parity on its released LJSpeech weights (durations exact, mel 0.9999999999), chained
  through its paired HiFi-GAN and the release's own phoneme vocabulary by `NFKMLXVoice`. The
  end-to-end test synthesizes "hello world" and the package's own Whisper transcribes it back.
- HiFi-GAN vocoder at reference parity on the released UNIVERSAL_V1 weights (0.9999999999); the
  converter fuses the release's weight normalization and reads the espnet-paired layout too.
- Whisper reads transformers-layout checkpoints (`model.encoder.layers.N.self_attn.q_proj` …) as
  well as OpenAI-layout ones.
- Whisper timestamped decoding — `transcribeWithTimestamps`, and `emitsTimestamps` on the backend,
  which adds `NSArray<NFKAudioSegment *>` under `NFKOutputSegments`. Off by default: it is a
  different decode, not a different reading of one.
- Additional sizes in ported families, each at measured parity: Whisper small / medium / large-v3,
  YOLOv8 l / x, SwinIR ×8 and the lightweight ×2 (a different upsampler), SAM 2 base_plus and large,
  and Gemma 4 E4B (measured at the bf16 the release ships in). DeepSeek V4 Pro is verified
  structurally — its 149782-tensor index accounted for exactly, never run.
- Bring-your-own-model backends: module, matting, tensor, diffusion, speech, and video.
- Video as a produced modality: `NFKMLXVideoBackend` reads and writes `NFKVideoAsset`, with clip-level
  frame interpolation (RIFE) and super-resolution (BasicVSR) over whole sequences.
- Face detection and alignment, so `NFKMLXCodeFormer` restores a photograph rather than only a
  pre-aligned crop. Two detectors: `NFKMLXRetinaFace` (the reference pipeline's own, at measured
  parity, 1.7 MB) and `NFKMLXVisionFaceDetector` (no weights, no download).
- On-device fine-tuning: `NFKMLXTrainer`, LoRA, training data helpers, and recipes for Zero-DCE,
  SegFormer, CLIP probes, and Whisper.
- `NFKMLXModelRegistry`, `NFKMLXHub`, and direct `@objc` factories, so an Objective-C consumer builds
  and runs a model without writing Swift.
- `NFKMLXRandom`, `NFKMLXGPU`, and `NFKMLXDevice` expose MLX's global runtime knobs to Objective-C.
- Every model's weight loader reads through `NFKMLXWeights.loadCheckpoint`, so a checkpoint written by
  fine-tuning reloads through the model's existing factory without being transposed twice.

### InferKitFoundationModels (companion)

- `NFKFoundationModelsBackend` over Apple's on-device system language model, with streaming, multi-turn
  transcripts, runtime-schema tool calling, and structured output.

### Distribution

- Source through Swift Package Manager and CocoaPods, from the same files.
- Prebuilt XCFrameworks for the MLX companion in static and dynamic variants, arm64, with macOS, iOS
  device, and iOS simulator slices. The static variant carries the Metal library each slice's consumer
  ships. Linking recipes per consumer shape are in `Docs/installation.md`.
