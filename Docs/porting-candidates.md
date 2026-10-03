# Open-model porting candidates

A survey of open-weight models from large vendors (NVIDIA, Meta, Microsoft, Intel, Mistral, IBM,
Stability AI, Black Forest Labs, Tencent, ByteDance, Kyutai, Apple, Amazon, Google) that InferKit does
not yet support, scored as candidates for future MLX ports. This is preliminary research. Nothing here
is committed, and no port is scheduled by its presence on this list.

Surveyed on 2026-09-05. Every row was checked against the vendor page, the Hugging Face card, or the
official repository for license and reference-implementation availability at that date. The model
landscape moves faster than a release, so re-verify a candidate's license and reference before starting
it. This document complements the [Roadmap](inference-guide.md#roadmap), which tracks the standout pick
per modality that is already scoped; the entries here are the wider field the roadmap draws from.

## Contents

- [How to read this](#how-to-read-this)
- [The architectural lens](#the-architectural-lens)
- [Tier 1 — clean near-term wins](#tier-1--clean-near-term-wins)
- [Tier 2 — strong, mostly permissive](#tier-2--strong-mostly-permissive)
- [Tier 3 — strategic frontier, high cost](#tier-3--strategic-frontier-high-cost)
- [Updates to shipped families (Tiers A–C)](#updates-to-shipped-families-tiers-ac)
- [Text-to-speech candidates](#text-to-speech-candidates)
- [3D asset generation candidates](#3d-asset-generation-candidates)
- [Novel but license-blocked](#novel-but-license-blocked)
- [Vendor verdicts](#vendor-verdicts)
- [Connections to existing work](#connections-to-existing-work)
- [Recommendation](#recommendation)

## How to read this

A candidate is scored on four axes:

- **Novelty** — distance from an architecture InferKit already ports. A model that reduces to a covered
  decoder or encoder scores low, whatever its headline capability.
- **License** — whether the weights are redistributable and commercially usable. Code license and weight
  license differ often. The toolkit is license agnostic: it downloads nothing it redistributes, and
  compliance with a weight license is the consumer's. A restrictive license lowers a candidate's
  priority and never blocks its port; the entry records the license so the consumer can read it.
- **Reference** — whether a third-party implementation exists to measure parity against. The toolkit
  ports at exact reference parity, so a runnable oracle (HF `transformers`, HF `diffusers`, or the
  official repository) is a prerequisite, not a nicety.
- **Effort** — rough port difficulty, including new operations, model size against a 32 GB machine, and
  reference availability on the CPU rather than CUDA only.

Priority favors a candidate that is novel, permissively licensed, backed by a runnable reference, and
tractable in size, in that combination.

## The architectural lens

Most "new" LLMs and VLMs from these vendors reduce to a decoder or encoder InferKit already runs, so
vendor is the wrong axis. The useful question is which architecture *families* the toolkit lacks. The
territory that is genuinely new:

- **State-space models (Mamba-2 / selective scan).** No state-space layer exists in the toolkit. One
  selective-scan implementation unlocks a whole class of recent LLMs.
- **Time-series forecasting.** A new modality, built on the T5 encoder-decoder already at parity.
- **MMDiT (multimodal joint-stream diffusion).** Flux and Stable Diffusion 3 run text and image tokens
  through joint attention, distinct from the covered SD UNet and the LTX/Wan/SANA/Z-Image single-stream
  DiTs.
- **Generative detection and OCR through location tokens.** Florence-2 emits coordinates and polygons as
  text tokens from a seq2seq decoder.
- **Transformer-in-codec.** Mimi interleaves transformer layers into a convolutional codec, distinct
  from the convolutional DAC and SNAC.
- **JEPA video encoders, point tracking, and 3D-asset diffusion.** Tasks with no covered analog.

## Tier 1 — clean near-term wins

Novel enough to be worth porting, permissively licensed, backed by a runnable reference, and tractable
in size.

| Model | Vendor | Task | Architecture | License | Reference | Effort |
|-------|--------|------|--------------|---------|-----------|--------|
| **BigVGAN v2** — SHIPPED (`NFKMLXBigVGAN`) | NVIDIA | Vocoder | GAN generator with Snake anti-aliased periodic activations and low-pass filtered resampling | MIT | `NVIDIA/BigVGAN` + HF `transformers` | Low–Med |
| **Mimi** — SHIPPED (`NFKMLXMimi`) | Kyutai | Neural codec | SEANet conv encoder/decoder with interleaved transformer layers and split RVQ (one semantic codebook plus acoustic) | Code MIT, weights CC-BY-4.0 | `kyutai-labs/moshi` (PyTorch and MLX) + HF `MimiModel` | Low–Med |
| **Chronos-Bolt** — SHIPPED (`NFKMLXChronos`, Swift API) | Amazon | Time-series forecasting | T5 encoder-decoder over patched observations, direct multi-step quantile forecast | Apache-2.0 | `amazon-science/chronos-forecasting` + HF | Low–Med |
| **Pixtral vision tower** — SHIPPED (`NFKMLXPixtral`) | Mistral | Vision-language | From-scratch variable-resolution ViT with 2D RoPE, feeding a Mistral decoder | Apache-2.0 | HF `transformers` (`PixtralVisionModel`) | Med |
| **Flux.1 [schnell]** — SHIPPED (`NFKMLXFluxPipeline`) | Black Forest Labs | Text-to-image | Hybrid MMDiT, roughly 19 double-stream plus 38 single-stream blocks, rectified flow | Apache-2.0 | HF `diffusers` (`FluxPipeline`) | High |

Notes:

- **BigVGAN v2** adds a real vocoder architecture over the HiFi-GAN and iSTFTNet vocoders already
  shipped, and loads through the existing weight-norm-fusion pattern. An optional fused CUDA kernel
  exists, and the default reference is pure PyTorch.
- **Mimi** carries a transformer core and semantic vector quantization that DAC and SNAC do not, and it
  is the one codec here with an official MLX reference (`moshi-swift`) to cross-check against. The RVQ
  machinery from the DAC and SNAC ports transfers.
- **Chronos-Bolt** opens a new modality at low architectural cost, because it is a patched T5 and the
  toolkit already ships a parity-verified T5 and umT5.
- **Pixtral** contributes a novel vision tower; the decoder half reuses the covered dense Mistral stack.
- **Flux.1 [schnell]** completes a generation family InferKit half-owns: the Flux VAE is already ported,
  and schnell is the Apache-2.0 member of the Flux family.

## Tier 2 — strong, mostly permissive

| Model | Vendor | Task | Architecture | License | Reference | Effort |
|-------|--------|------|--------------|---------|-----------|--------|
| **Florence-2** — SHIPPED (`NFKMLXFlorence2`) | Microsoft | Unified vision: caption, detect, ground, segment, OCR | DaViT dual (spatial + channel) attention encoder → BART-style seq2seq emitting location tokens | MIT | HF `transformers` (`Florence2`) | High |
| **Voxtral-Mini 3B** — SHIPPED (`NFKMLXVoxtral`) | Mistral | ASR and speech translation | Whisper large-v3 encoder (reused) → 2-linear projector → Llama decoder (reused); tekken tokenizer | Apache-2.0 | HF `transformers` (`VoxtralForConditionalGeneration`) | Med |
| **Granite Speech 3.3-2B** — SHIPPED (`NFKMLXGraniteSpeech`) | IBM | ASR and translation | Conformer encoder and window Q-former projector → dense Granite decoder, audio LoRA folded | Apache-2.0 | HF `transformers` (`GraniteSpeechForConditionalGeneration`) | Med |
| **Canary-1B-v2** — SHIPPED (`NFKMLXCanary`) | NVIDIA | ASR and speech translation | biased FastConformer encoder (reused from Parakeet) → Transformer attention encoder-decoder; Metaspace BPE tokenizer | CC-BY-4.0 | NeMo (`EncDecMultiTaskModel`) | Med |
| **V-JEPA 2** — SHIPPED (`NFKMLXVJEPA2`) | Meta | Video and image encoder | ViT trained by joint-embedding predictive masked-latent prediction; inference is a ViT forward | MIT weights | HF `transformers` + `facebookresearch/vjepa2` | Med |
| **Wav2Vec2 / Wav2Vec2-BERT / HuBERT** — SHIPPED (`NFKMLXWav2Vec2`, `NFKMLXWav2Vec2Bert`) | Meta | Speech SSL and CTC ASR | Conv feature extractor + transformer encoder + CTC head | Apache-2.0 (base checkpoints) | HF `transformers` | Low–Med |
| **TrOCR** — SHIPPED (`NFKMLXTrOCR`) | Microsoft | OCR | ViT/BEiT encoder + RoBERTa decoder, standard seq2seq | MIT | HF `transformers` (`VisionEncoderDecoder`) | Low–Med |
| **Table Transformer (TATR)** — SHIPPED (`NFKMLXTableTransformer`) | Microsoft | Table detection and structure | Vanilla DETR, ResNet-18 backbone, no deformable attention | MIT | HF `transformers` + official repo | Low–Med |
| **TimesFM 2.5** — SHIPPED (`NFKMLXTimesFM`) | Google | Time-series forecasting | Decoder-only over non-overlapping patches, single-pass horizon, 9-quantile head | Apache-2.0 (2.5 only) | HF + `google-research/timesfm` | Med |
| **Cosmos Tokenizer** — SHIPPED (`NFKMLXCosmosTokenizer`) | NVIDIA | Image and video tokenizer | Haar wavelet patcher → 2D (image) or factorized causal 3D (video) autoencoder; continuous latents or FSQ tokens; all ten 0.1 releases | Code Apache-2.0, weights NVIDIA Open Model License | `nvidia-cosmos/cosmos-predict1` tokenizer modules + HF | Med |
| **Sa2VA** — SHIPPED (`NFKMLXSa2VA`) | ByteDance | Segmentation VLM | SAM 2 and a LLaVA-style VLM fused in a shared token space | Apache-2.0 | `bytedance/Sa2VA` + HF | Med |
| **SD 3.5 Large/Medium** — medium at release parity; large and turbo measured on cuts | Stability AI | Text-to-image | MMDiT-X with QK-norm and dual attention, three text encoders (2×CLIP + T5) | Stability Community (free under $1M revenue) | HF `diffusers` (`SD3Transformer2DModel`) | Med–High |
| **Stable Audio Open 1.0** | Stability AI | Text-to-audio | Latent audio DiT over an Oobleck autoencoder, T5 conditioning | Stability Community | HF `diffusers` (`StableAudioPipeline`) | Med–High |

Notes:

- **Florence-2 is now shipped at reference parity** (`NFKMLXFlorence2`, base and large). It performs
  captioning, detection, and grounding through one seq2seq decoder that emits `<loc_0..999>` location
  tokens. The DaViT vision tower pairs windowed spatial attention with grouped channel attention in
  every block, a dual-attention form new to the toolkit.
- **Granite Speech 3.3-2B is now shipped at reference parity** (`NFKMLXGraniteSpeech`): a Conformer
  acoustic encoder and a BLIP-2 Q-former projector feed a dense Granite decoder, with the released audio
  LoRA adapter folded in; the released 2b matches by shape across all 937 base tensors, the encoder /
  projector / logit cosines are ~1.0, and the backend transcribes the validation clip exactly. Its
  decoder is the dense `granite` (Llama + Granite's four multipliers), not the 4.0-H hybrid.
  **Voxtral-Mini** decomposes into a Whisper-family encoder (shipped) and a Mistral decoder the toolkit
  already runs, so the work is mostly wiring. The 24B Voxtral and 8B Granite Speech are too large for the
  target machine; the 3B and 2B variants fit.
- **Canary-1B-v2** reuses the FastConformer encoder from the Parakeet port; the attention
  encoder-decoder decoder is the new work. Canary-Qwen uses a Qwen decoder already covered, so it is a
  lower priority.
- **TrOCR and Table Transformer are now shipped at reference parity** (`NFKMLXTrOCR`,
  `NFKMLXTableTransformer`). TrOCR reads a handwritten line through a ViT encoder and a BART-style
  decoder. Table Transformer is a vanilla DETR over a ResNet-18 backbone.
- **V-JEPA 2 and Sa2VA are now shipped at reference parity** (`NFKMLXVJEPA2`, `NFKMLXSa2VA`). V-JEPA 2
  is a video ViT-L with a 3D tubelet embedding and 3D rotary position. Sa2VA-4B joins an InternViT
  encoder and a Qwen2.5-3B decoder to the shipped SAM 2 network, which draws a mask for each `[SEG]`
  token the decoder emits.
- **TimesFM 2.5** is a second time-series family. Stay at 2.5 or earlier; TimesFM 3.0 carries a
  non-commercial license.
- **Cosmos Tokenizer is now shipped at reference parity** (`NFKMLXCosmosTokenizer`): all ten released
  image and video tokenizers load from their own TorchScript `autoencoder.jit`, and the post-training
  objective ships with them. It shares no code with the LTX and Wan VAEs: its causal convolutions are
  factorized into spatial and temporal kernels and its resampling blocks add a strided convolution to
  an average pool. The code is Apache-2.0 and the weights use the commercially usable NVIDIA Open
  Model License.
- **SD 3.5** and **Stable Audio Open** carry the Stability Community License, which is free below $1M
  annual revenue and not fully permissive above it. Flag the revenue gate before committing.

## Tier 3 — strategic frontier, high cost

| Model | Vendor | Task | Architecture | License | Reference | Effort |
|-------|--------|------|--------------|---------|-----------|--------|
| **Codestral-Mamba 7B** — SHIPPED (`NFKMLXMamba`) | Mistral | Code LLM | Pure Mamba-2 SSM, linear-time, 256k context | Apache-2.0 | transformers `Mamba2ForCausalLM` (CPU) | High |
| **Granite 4.0-H** — SHIPPED (`NFKMLXGraniteHybrid`) | IBM | LLM | Hybrid Mamba-2 SSM, sparse attention, MoE in H-Small | Apache-2.0 | HF `transformers` (`GraniteMoeHybrid`) | High |
| **Nemotron Nano 2 (9B/12B v2)** — SHIPPED (`NFKMLXNemotronH`) | NVIDIA | LLM | Nemotron-H hybrid: Mamba-2 layers, ReLU-squared MLP, few NoPE attention layers | NVIDIA Open Model License | HF `transformers` (`NemotronH`) | High |
| **BAGEL-7B-MoT** — PRIORITY 1 (scheduled) | ByteDance | Any-to-any understand and generate | Mixture-of-Transformer-Experts, dual VAE + ViT encoders | Apache-2.0 | `ByteDance-Seed/BAGEL` (official repo) | High |
| **Phi-4-multimodal** — SHIPPED (`NFKMLXPhi4MM`) | Microsoft | Text + vision + audio | Phi-4-mini decoder + SigLIP-style vision + Conformer audio + mixture-of-LoRAs | MIT | the release's remote code (transformers 4.46.1) | High |
| **HunyuanVideo / Hunyuan3D** | Tencent | Text-to-video / image-to-3D | MMDiT + 3D causal VAE (video); flow-based shape DiT + PBR paint (3D) | Tencent Hunyuan Community | HF `diffusers` (video) / official repo (3D) | High |

Notes:

- The **Mamba / SSM track** is the highest-leverage strategic bet. It introduces the toolkit's first
  state-space layer, and one selective-scan (SSD) implementation serves Codestral-Mamba, Granite 4.0-H,
  and Nemotron Nano 2. The cost is real: Mamba-2's selective scan has no fused MLX kernel and the
  official references are CUDA-only. **Codestral-Mamba is now shipped at reference parity** (`NFKMLXMamba`, measured against transformers'
  `Mamba2ForCausalLM` on the CPU rather than the CUDA references — a float/bf16 path needs no fused
  kernel), which delivers the selective-scan primitive. **Granite 4.0-H is now shipped at reference
  parity** (`NFKMLXGraniteHybrid`, reusing that mixer verbatim for its Mamba layers and adding NoPE
  grouped-query attention, the routed mixture of experts, and Granite's scalar multipliers), the first
  hybrid Mamba-attention decoder and the first on-device language-decoder fine-tune. **Nemotron Nano 2 is
  now shipped at reference parity** (`NFKMLXNemotronH`, reusing that mixer verbatim for its Mamba layers
  and adding NoPE grouped-query attention and a ReLU-squared feed-forward, one mixer per block from
  `hybrid_override_pattern`, no scalar multipliers, its Mamba gated norm grouped by `n_groups`): tiny
  logit cosine 1.0, structural parity across all 341 tensors of Nemotron-Nano-9B-v2, and the same
  on-device LoRA fine-tune. This closes the SSM/hybrid track — one selective-scan mixer now serves all
  three (Codestral-Mamba, Granite 4.0-H, Nemotron Nano 2).
- **BAGEL** is the scheduled priority-1 port (permissive Apache-2.0, runnable official reference). Its
  weights download to the model store (`/Volumes/InferKit Models`) through the IO Manager, as for every
  scheduled port.
- **BAGEL** and **Phi-4-multimodal** are large multi-component integrations whose novelty concentrates
  in one part (the Mixture-of-Transformer routing; the audio Conformer and mixture-of-LoRAs).
  **Phi-4-multimodal is now shipped at reference parity** (`NFKMLXPhi4MM`) in all four of its modes
  (text, speech, vision, and vision with speech), measured against the release's own remote code on the
  released weights with token-exact answers from raw inputs. Its decoder reuses `NFKMLXLanguageNet`,
  which gained partial rotary and LongRoPE; the new work is the Conformer speech tower, the NaViT SigLIP
  embedding and Phi-3.5's HD layout, the runtime mixture-of-LoRAs layer, and both preprocessors.
- **Hunyuan3D** is a 3D-asset model with no covered analog. The Tencent Community License excludes the
  EU, UK, and South Korea and bans training competitors, so it is not truly open. The permissive
  alternatives are surveyed under [3D asset generation candidates](#3d-asset-generation-candidates).

## Updates to shipped families (Tiers A–C)

Tiers 1–3 survey families new to the toolkit. Tiers A–C come from a second survey, taken on
2026-09-21 against [the model index](model-index.md), of upstream releases that update a family InferKit
already ships. Each release is sorted by the work it needs. Every item was re-verified against the
Hugging Face detail endpoint before work started, and that check corrected several of the survey's
claims; the tables below carry the verified facts.

### Tier A — successor weights on a shipped architecture

| Update | Vendor | License | Status |
|--------|--------|---------|--------|
| **Parakeet-TDT 0.6B v3** — SHIPPED (`NFKMLXParakeet`) | NVIDIA | CC-BY-4.0 | Measured on the released model; its 8192-piece vocabulary sizes the net, and it transcribes 25 European languages |
| **Chatterbox Multilingual v3** — SHIPPED (`NFKMLXChatterbox`) | Resemble AI | MIT | The multilingual text layer and T3 are measured on the released weights; the backend prefers the multilingual release when a directory holds one |
| **Gemma 4 31B** — SHIPPED (`NFKMLXGemmaLanguage`) | Google | Apache-2.0 | Structural across all 832 tensors, plus a numeric probe of the first 6 released layers and bf16 records on cuts. The whole 31B exceeds a 32 GB machine |
| **RF-DETR segmentation** — SHIPPED (`NFKMLXRFDetrSegmentationNet`) | Roboflow | Apache-2.0 | Every released `rf-detr-seg-*` size, nano through xxlarge |
| **Wan 2.7** | Alibaba | — | Not released as open weights. The newest open Wan is 2.2 (checked 2026-09-26) |

Notes:

- The survey expected RF-DETR detection sizes above large and a keypoint head. Neither is published on
  Hugging Face; the xlarge and xxlarge releases are segmentation checkpoints, and those ship.
- Wan 2.7 was announced as API-only. Its audio output would be a new stage, so it moves to Tier B if
  its weights are released.

### Tier B — same vendor, new architecture

| Model | Vendor | License | Status |
|-------|--------|---------|--------|
| **DeepSeek V4.1 Flash** — SHIPPED (`NFKMLXDeepSeek`) | DeepSeek | MIT | The decoder, image tower, aligner, and DSpark draft stack match the release's own `inference/model.py`, bit-exact in bf16. Experts and n-gram tables page. The full released weights (510.3 GB) are not in the model store, and the release is held structurally, not run |
| **Qwen3.8-Flash-Next** — SHIPPED (`NFKMLXQwen4Exp`) | Alibaba Qwen | other | The `qwen4_exp` architecture at reference parity on a tiny oracle; the 180B release is held structurally |
| **FLUX.2 [dev]** — PARTIAL (`NFKMLXFlux2`) | Black Forest Labs | FLUX Non-Commercial, gated | [klein] 4B runs end to end at released-weight parity. [dev] needs Mistral-Small 3 as its text encoder and a 60 GB bf16 transformer, which exceeds a 32 GB machine |
| **LTX-2.5** — PARTIAL (`NFKMLXLTX2TransformerNet`) | Lightricks | other, gated | The audio-video transformer is at reference parity, with a structural check on the ungated LTX-2.3. The video and audio autoencoders, the Gemma 4 text front end, the vocoder, and the pipeline are not built |

Notes:

- The survey described DeepSeek V4.1 Flash as a 552B causal encoder-decoder. The release is a 763B
  decoder that extends V4 with n-gram memory, a vision tower, and DSpark speculation. V4 Flash and V4
  Pro, including the Pro 0813 draft stack, were brought to the same release-code parity alongside it.
- FLUX.2 [dev] and LTX-2.5 are gated with automatic approval. Accepting each license is a user action,
  and it releases the headers the structural checks still lack.

### Adjacent new families

The survey also named two families new to the toolkit that sit next to shipped surfaces:

| Model | Vendor | License | Status |
|-------|--------|---------|--------|
| **Qwen3-VL-Embedding / Reranker 2B** — SHIPPED (`NFKMLXQwen3VLEmbedder`, `NFKMLXQwen3VLReranker`) | Alibaba Qwen | Apache-2.0 | Text and image retrieval at reference parity, on the embeddings and rerank surfaces |
| **Qwen-Image 2.1** — SHIPPED (`NFKMLXQwenImagePipeline`) | Alibaba Qwen | Qwen Research License (non-commercial) | End to end: the transformer, the Wan 2.2-derived autoencoder, and the Qwen3-VL 8B text encoder |

### Tier C — no action

Checked on 2026-09-21 and left alone:

- **Whisper.** No new open checkpoint. The 2026 OpenAI transcription models are API-only.
- **gpt-oss.** No weights since 120b and 20b.
- **Kokoro.** No new architecture release.
- **BiRefNet.** The "2026.2" listing is a third-party host's packaging, not an upstream release.
- **Wan 2.6.** Closed.
- **SAM 3 / 3.1.** Public weights under Meta's custom SAM License. The recorded decision to skip on
  license terms stands.

## Text-to-speech candidates

Surveyed on 2026-09-26 against the Hugging Face detail endpoint (`/api/models/<repo>?blobs=true`).
The goal is long-form narration above the shipped Chatterbox and Kokoro. Download is the size of the
weights a port needs: duplicate formats, optimizer states, and training shards are excluded. Several
releases postdate the survey author's knowledge, so audition samples before committing to a port.

Priority rules:

- Priority 1 is scheduled. Its weights are downloaded to the model store (`/Volumes/InferKit Models`)
  through the IO Manager, at the pinned revisions below.
- Priority 2 is permissive and unscheduled.
- Priority 3 holds every model whose weight license is non-commercial, research-only, or undeclared. A
  restrictive license keeps a model at priority 3 unless the user asks for that model by name. A model
  the user asks for is ported in full, with the fixes it surfaces along the way.

### Priority 1 — scheduled (permissive)

| Model | Vendor | Download | License | Reuses or needs | Revision |
|-------|--------|----------|---------|-----------------|----------|
| **Chatterbox-Turbo** | Resemble AI | 4.0 GB repo (1.9 GB T3 + 1.06 GB S3Gen or its meanflow variant) | MIT | The shipped Chatterbox voice encoder, tokenizer, and S3Gen; the new work is the turbo T3 and the meanflow S3Gen | `749d1c1a` |
| **Chatterbox-Nano** | Resemble AI | 3.0 GB repo (0.87 GB T3 + the same S3Gen pair) | MIT | As Turbo | `71ccd1d0` |
| **VibeVoice-1.5B** | Microsoft | 5.4 GB | MIT | The Qwen2.5 decoder and tokenizer; the acoustic and semantic tokenizers and the diffusion head are new. Built for long-form, multi-speaker audio. Microsoft withdrew the reference code in 2025; confirm a runnable reference before starting | `c00898d2` |
| **VibeVoice-Realtime-0.5B** | Microsoft | 2.0 GB | MIT | The streaming member of the same family | `6bce5f06` |
| **Qwen3-TTS 12Hz 1.7B** (Base, CustomVoice, VoiceDesign) | Alibaba Qwen | 4.5 GB each | Apache-2.0 | The Qwen3 decoder; each repo carries its own 0.68 GB 12 Hz speech tokenizer | `fd4b2543`, `0c0e3051`, `5ecdb673` |
| **Qwen3-TTS 12Hz 0.6B** (Base, CustomVoice) | Alibaba Qwen | 2.5 GB each | Apache-2.0 | As 1.7B | `5d839924`, `85e237c1` |

### Priority 2 — potential (permissive)

| Model | Vendor | Download | License | Reuses or needs |
|-------|--------|----------|---------|-----------------|
| **Orpheus 3B** | Canopy Labs | 15.1 GB float32 (~7.5 GB bf16); gated=auto | Apache-2.0 | Llama decoder and the shipped SNAC codec |
| **Dia-1.6B** | Nari Labs | 6.4 GB | Apache-2.0 | The shipped DAC codec |
| **CSM-1B** | Sesame | 6.2 GB; gated=auto | Apache-2.0 | Llama decoder; needs Mimi (Tier 1) |
| **Kyutai TTS 1.6B (en/fr)** | Kyutai | 4.1 GB | CC-BY-4.0 | Needs Mimi (Tier 1) |
| **Fun-CosyVoice3 0.5B** | Alibaba FunAudioLLM | ~5.4 GB | Apache-2.0 | Qwen decoder; the speech tokenizer ships as ONNX only |
| **CosyVoice2 0.5B** | Alibaba FunAudioLLM | 4.9 GB | Apache-2.0 | As CosyVoice3 |
| **VoxCPM2** | OpenBMB | 5.0 GB | Apache-2.0 | — |
| **Zonos v0.1 (transformer)** | Zyphra | 3.3 GB | Apache-2.0 | — |
| **Maya1** | Maya Research | 6.6 GB | Apache-2.0 | Llama decoder |
| **NeuTTS Air** | Neuphonic | 3.0 GB; gated=auto | Apache-2.0 | — |
| **Confucius4-TTS** | NetEase Youdao | 3.1 GB | Apache-2.0 | — |
| **AuK, AuK-Flash** | Tencent | 6.8 GB each | MIT | — |
| **KugelAudio-0** | KugelAudio | 18.7 GB | MIT | — |
| **Supertonic-3** | Supertone | 0.4 GB | OpenRAIL | Published as ONNX only; needs a conversion step |

### Priority 3 — restrictive license (lowest)

| Model | Vendor | Download | License |
|-------|--------|----------|---------|
| **Fish Audio S2 Pro** | Fish Audio | 11.0 GB | Fish Audio research license |
| **Fish Audio S1-mini** | Fish Audio | 3.6 GB; gated=auto | CC-BY-NC-SA-4.0 |
| **Higgs TTS 3 4B** | Boson AI | 9.3 GB | Boson research and non-commercial |
| **IndexTTS-2** | Bilibili | 5.9 GB, plus helper models fetched at run time | undeclared on the card |
| **IndexTTS-2.5** | Bilibili | 5.5 GB | Bilibili model license |
| **Breeze-TTS-2** | BreezeBlue | 7.7 GB | research and non-commercial |
| **F5-TTS v1** | SWivid | 1.35 GB | CC-BY-NC-4.0 |
| **Spark-TTS 0.5B** | SparkAudio | 3.9 GB | CC-BY-NC-SA-4.0 |
| **MaskGCT** | Amphion | 6.6 GB | CC-BY-NC-4.0 |
| **OmniVoice** | k2-fsa | 3.3 GB | undeclared on the card |

The Higgs Audio v2 generation repo (`bosonai/higgs-audio-v2-generation-3B-base`) returned an empty
response on 2026-09-26, and `microsoft/VibeVoice-Large` returned 401.

## 3D asset generation candidates

Surveyed on 2026-09-30 against the Hugging Face detail endpoint (`/api/models/<repo>?blobs=true`), with
a real file fetch on each priority 1 repo. The modality is image-to-3D and text-to-3D: one image or one
prompt in, a mesh or a set of 3D Gaussians out. InferKit ships no model of this kind. Download is the
size of the weights a port needs. Nothing in this section is scheduled.

Two architecture families cover the field:

- **Feed-forward triplane reconstruction.** An image encoder feeds a transformer that emits three
  feature planes, a small MLP reads density or signed distance from them, and an iso-surface extractor
  produces the mesh. One forward pass, no sampler. TripoSR, Stable Fast 3D, and SPAR3D.
- **Latent flow over a 3D representation.** A rectified-flow transformer denoises a 3D latent that a VAE
  decodes. The latent is a sparse voxel grid in the TRELLIS family (TRELLIS, TRELLIS.2, Pixal3D, Arbor,
  SAM 3D Objects, Hi3DGen) and an unordered vector set in the other (TripoSG, Hunyuan3D, Step1X-3D).

TRELLIS is the hub. Six of the releases below are TRELLIS checkpoints or fine-tunes that load into its
networks, and TRELLIS.2 loads TRELLIS's sparse-structure decoder unchanged.

### Priority 1 — recommended

| Model | Vendor | Download | License | Reuses or needs | Revision |
|-------|--------|----------|---------|-----------------|----------|
| **TripoSR** | Stability AI and Tripo | 1.68 GB, one checkpoint | MIT | DINO ViT-B/16 image tokenizer, a 16-layer transformer over 3×32×32 triplane tokens with cross-attention, a transposed-convolution upsampler, a 9-layer NeRF MLP, marching cubes. Vertex colors, no UV texture. The reference runs on the CPU | `5b521936` |
| **Stable Fast 3D** | Stability AI | 4.02 GB, one checkpoint; gated=auto | Stability Community (free under $1M revenue) | TripoSR's triplane lineage with a larger transformer, marching tetrahedra, material and illumination heads (the material head reads OpenCLIP features), a box-projection UV unwrapper, and a texture baker. The reference supports MPS with Metal baking kernels and runs in about 6 GB | `f0c9a8ff` |
| **SPAR3D** | Stability AI | 7.33 GB, one checkpoint; gated=auto | Stability Community | Stable Fast 3D conditioned on a point cloud that a point-diffusion stage samples from the image. The point cloud is editable before meshing | `5699918c` |
| **TRELLIS image-large** | Microsoft | 3.3 GB repo; 2.66 GB for the mesh path, plus DINOv2 ViT-L/14 with registers (1.22 GB, Apache-2.0) | MIT | Stage 1: a 24-block flow transformer over a dense 16³ latent, decoded by a Conv3d VAE to a 64³ occupancy grid. Stage 2: a 24-block flow transformer over the occupied voxels with sparse-convolution input and output blocks. Decoders: sparse Swin transformers to a mesh (FlexiCubes), to 3D Gaussians, and to a radiance field. 25 Euler steps a stage | `25e0d31f` |
| **TRELLIS text** (base, large, xlarge) | Microsoft | 0.70, 2.28, 4.13 GB | MIT | The image model's networks conditioned on the shipped CLIP ViT-L text encoder | `f8e8cf00`, `4aad9f4a`, `e0b00432` |
| **TRELLIS.2-4B** | Microsoft | 16.24 GB repo; 9.8 GB for the 512³ path, plus DINOv3 ViT-L/16 (1.21 GB; gated=manual, DINOv3 License) | MIT | Three 1.3B flow transformers (30 blocks, width 1536, rotary position): sparse structure, shape, and texture conditioned on shape. Sparse ConvNeXt VAE decoders emit a dual-grid mesh and per-voxel PBR attributes. Loads TRELLIS's sparse-structure decoder. 12 steps a stage. The release's background remover is RMBG-2.0 (CC-BY-NC); the shipped BiRefNet replaces it | `af44b45f` |
| **TripoSG** | VAST | 7.95 GB (5.76 GB transformer, 0.97 GB VAE, 1.22 GB DINOv2 ViT-L) | MIT | A 21-layer, width-2048 rectified-flow transformer over 2048 vector-set latents of 64 channels. The VAE decoder answers signed-distance queries by cross-attention and marching cubes extracts the surface. Geometry only. `diffusers` layout | `2c1c516d` |

### Priority 2 — potential

| Model | Vendor | Download | License | Reuses or needs |
|-------|--------|----------|---------|-----------------|
| **Pixal3D** | Tencent ARC | 46.05 GB repo; 5.4–5.5 GB a stage, single-view and multi-view sets | MIT | TRELLIS.2's networks with pixel-aligned projection conditioning and view-aligned latents; DINOv3 |
| **Arbor** | Stability AI | 0.63 GB denoiser, plus TRELLIS text weights and the TRELLIS mesh decoder | Stability Community | Text-to-3D under hull, avoidance, and touch constraint meshes. A TRELLIS text fine-tune with a constraint encoder |
| **SAM 3D Objects** | Meta | 13.17 GB; gated=manual | `other` on the card; the MLX conversion's card names the SAM License | TRELLIS's two stages conditioned on a MoGe point map; reconstructs an object from a masked photograph |
| **Hi3DGen** (`Stable-X/trellis-normal-v0-1`) | Stable-X | 2.65 GB | MIT | TRELLIS conditioned on a normal map |
| **Step1X-3D** | StepFun | 19.64 GB repo; 7.25 GB for geometry | Apache-2.0 | A vector-set VAE and a 1.3B flow transformer for geometry; an SDXL-based multi-view model for texture |
| **MIDI-3D** | VAST | 5.07 GB | Apache-2.0 | TripoSG extended to several objects in one scene |
| **Direct3D-S2** | DreamTech | 8.51 GB | MIT | A sparse signed-distance VAE with spatial sparse attention, up to 1024³ |
| **InstantMesh** | Tencent ARC | 7.27 GB repo | Apache-2.0 | A triplane reconstruction model over six generated views. The view generator is Zero123++ v1.2, whose card declares no license |
| **LGM** | ashawkey | 0.83 GB fp16 | MIT | A U-Net from four views to 3D Gaussians; needs a multi-view diffusion model in front |
| **Shap-E** | OpenAI | 4.9 GB repo; 1.33 GB fp16 | MIT | A 2023 latent diffusion over implicit-function weights, text- or image-conditioned through CLIP. The one candidate with an HF `diffusers` pipeline as its reference |
| **Cube 3D v0.5, CubePart** | Roblox | 8.28 GB, 9.90 GB | OpenRAIL on the card | A shape tokenizer and an autoregressive text-to-shape transformer |
| **UltraShape** | infinith | 8.02 GB | Apache-2.0 on the card | A geometry refiner. Its 7.37 GB checkpoint matches the Hunyuan3D-2.1 shape transformer's size, so confirm its provenance before treating the Apache-2.0 declaration as the whole license |

### Priority 3 — restrictive license (lowest)

| Model | Vendor | Download | License |
|-------|--------|----------|---------|
| **Hunyuan3D-2.1** | Tencent | 14.91 GB (8.0 GB shape, 6.9 GB PBR paint) | Tencent Hunyuan Community |
| **Hunyuan3D-2, 2mini, 2mv, Omni** | Tencent | 74.89, 25.26, 29.58, 25.73 GB repos | Tencent Hunyuan Community |
| **Hunyuan3D-Part** | Tencent | 9.5 GB | undeclared on the card |
| **Stable Zero123** | Stability AI | 17.17 GB | Stability non-commercial community |
| **SV3D** | Stability AI | 18.74 GB; gated=auto | Stability SV3D non-commercial community |
| **Stable Virtual Camera** | Stability AI | 10.11 GB; gated=auto | Stability non-commercial |
| **SHARP** | Apple | 2.81 GB | Apple ML research (`apple-amlr`) |
| **ShapeR** | Meta | 4.18 GB | CC-BY-NC-4.0 |
| **Large Sparse Reconstruction Model** | Meta | 14.41 GB; gated=manual | CC-BY-NC-4.0 |
| **VGGT-1B** | Meta | 10.05 GB | CC-BY-NC-4.0; the commercial copy is gated=manual under its own acceptable-use license |
| **OpenLRM 1.1** | 3DTopia | 2.08 GB (mix-base) | CC-BY-NC-4.0 |
| **PartPacker** | NVIDIA | 3.18 GB | NVIDIA non-commercial |
| **LLaMA-Mesh** | NVIDIA research | 16.07 GB | Llama 3.1 Community |
| **CraftsMan3D** | CraftsMan3D | 4.99 GB | CreativeML OpenRAIL-M |

Stability's SV4D and SV4D 2.0 (Stability Community) are multi-view video synthesis and emit no mesh.
`openai/point-e`, `craftsman3d/craftsman-DoraVAE`, `ILSparkle/Sparc3D`, and `VAST-AI/MV-Adapter`
returned 401 on 2026-09-30.

### Access on 2026-09-30

- TripoSR, TRELLIS, TRELLIS.2, TripoSG, Pixal3D, and Arbor serve their files without a token.
- Stable Fast 3D and SPAR3D return 403 to the project's account. Each gate is automatic and opens when
  the account holder accepts the terms on the model page.
- DINOv3 ViT-L/16 and SAM 3D Objects return 403. Each gate is manual and Meta reviews the request.
  TRELLIS.2 and Pixal3D cannot run without DINOv3.

### What the modality needs

No 3D output type exists in the core, and no shipped model extracts a surface. The new work that is
shared across candidates:

- **Core value types.** A mesh (positions, normals, triangle indices, vertex colors, texture
  coordinates, PBR texture images) and a Gaussian set, with request and result keys, as `NFKMIDISequence`
  preceded the transcription models. ModelIO writes USD, OBJ, PLY, and STL with no third-party
  dependency. GLB needs an in-house writer.
- **Iso-surface extraction.** Marching cubes (TripoSR, TripoSG), marching tetrahedra (Stable Fast 3D),
  FlexiCubes (TRELLIS), and the flexible dual grid (TRELLIS.2). Each runs on the CPU and is
  deterministic.
- **Sparse voxel tensors.** A coordinate list with features, a submanifold sparse convolution, and sparse
  down- and up-sampling. The convolution is a neighbor table built on the CPU and a gather, the form the
  shipped deformable convolution uses. Token count varies per sample.
- **UV unwrapping and texture baking.** The references call xatlas, nvdiffrast, and cumesh. The core
  takes no third-party dependency, so both steps are written in-house. Stable Fast 3D's box-projection
  unwrapper and its Metal baking kernels are the closest model. A first release can emit vertex colors.
- **Encoders.** DINO ViT-B/16 and DINOv3 ViT-L/16 are new. DINOv2 ships inside Depth Anything and
  RF-DETR; the with-registers ViT-L/14 checkpoint needs confirming against that implementation. CLIP
  ViT-L and BiRefNet ship.

Reference availability on this machine:

- TripoSR runs on the CPU. Stable Fast 3D and SPAR3D run on the CPU and on MPS.
- TRELLIS depends on spconv, flash-attn or xformers, kaolin, nvdiffrast, and diffoctreerast. TRELLIS.2
  depends on flash-attn, flex_gemm, o_voxel, nvdiffrast, and cumesh. All are CUDA builds. A community
  port (`shivampkumar/trellis-mac`) replaces the attention, the sparse convolution, and the hash map with
  plain PyTorch and runs TRELLIS.2 at 512³ on a 24 GB M4 Pro in about 3.5 minutes at an 18 GB peak, with
  vertex colors and no baked texture. A parity oracle for the networks needs the same substitutions.
- TripoSG documents CUDA only. Its `diffusers` layout and scikit-image dependency suggest a CPU path;
  confirm before starting.
- Parity is measured at the network seams (triplanes, latents, density or distance grids). The extracted
  mesh is compared after the seams match.

Customization: TRELLIS publishes its training code under MIT, so its fine-tune objective can be measured
against a reference. TripoSR, Stable Fast 3D, SPAR3D, and TripoSG publish inference code only, so each
entry either derives the objective from the paper or rules the path out in writing.

### Order

1. **TripoSR.** The smallest permissive model. It forces the core mesh type, marching cubes, and the
   writers, which every later port uses.
2. **Stable Fast 3D**, once the gate is accepted. It reuses TripoSR's lineage, adds UV-textured PBR
   output, and its reference already runs on Apple GPUs.
3. **TRELLIS image-large and text.** The sparse-voxel infrastructure and the first generative 3D model,
   with Gaussian and mesh outputs.
4. **TRELLIS.2-4B**, once DINOv3 access is granted. The strongest open model at the survey date, and the
   base of Pixal3D.
5. **TripoSG.** The vector-set representation, which Step1X-3D, MIDI-3D, and Hunyuan3D share.

## Novel but license-blocked

Architecturally attractive, but the weight license blocks a redistributable, commercial toolkit. Worth
porting only if research-only or attribution-bound use is acceptable, and only after confirming the
exact license on the specific checkpoint.

- **Apple Depth Pro, FastVLM, MobileCLIP2, AIMv2.** Apple ML research licenses. Reports on Depth Pro
  conflict: one reads its license as research-only, another as permitting most commercial use with
  attribution. The Apple license files differ across repositories (`ml-depth-pro` versus `ml-fastvlm`
  and `ml-mobileclip`), so Depth Pro may be more usable than the others. Confirm the exact `LICENSE` on
  the checkpoint before committing. Depth Pro reuses the DINOv2 encoder already ported for Depth
  Anything.
- **Meta MusicGen, AudioGen, CoTracker3, Sapiens, SeamlessM4T, and EnCodec weights.** CC-BY-NC. EnCodec's
  code is MIT but its weights are non-commercial, so it is not the permissive near-freebie it appears to
  be next to DAC and SNAC. MusicGen adds the codebook delay-pattern interleaving the Music 3 notes
  observe is absent there.
- **NVIDIA Sortformer diarization.** CC-BY-NC. It is the runnable open alternative to the gated pyannote
  diarizer the roadmap is blocked on.
- **Flux [dev], Kontext [dev], Flux.2 [dev].** FLUX Non-Commercial. Only Flux.1 [schnell] and
  Flux.2 [klein] 4B are Apache-2.0.
- **Cohere Command-A, Aya-Vision.** CC-BY-NC, and both reduce to a covered decoder plus a SigLIP tower.

## Vendor verdicts

- **Intel: not worth pursuing.** Its open catalog is fine-tunes, quantization recipes, and the OpenVINO
  runtime. Its only distinct-looking models are LDM3D (a 6-channel `conv_in`/`conv_out` change on Stable
  Diffusion, already ported) and Intel DPT (the MiDaS DPT head, already covered by Depth Anything). No
  Intel port would add a genuinely new architecture.
- **NVIDIA, Meta, Microsoft** each contribute strong candidates across audio, vision, and unified
  models. Meta's most compelling picks below V-JEPA 2 and the Wav2Vec2 family carry non-commercial
  licenses.
- **Mistral, IBM, Stability, Black Forest Labs, Kyutai, Amazon, ByteDance** contribute the SSM track,
  the MMDiT track, the transformer-in-codec, and the time-series modality.

## Connections to existing work

Several candidates extend a family the toolkit already has or fill a gap the roadmap names:

- **Mimi and EnCodec** extend the DAC and SNAC codec family.
- **BigVGAN v2** joins the HiFi-GAN and iSTFTNet vocoder family.
- **Canary-1B-v2** reuses the Parakeet FastConformer encoder.
- **Sortformer diarization** and **Microsoft WavLM** both address the blocked pyannote diarization gap.
  WavLM (MIT) is the backbone pyannote uses and is a possible permissive route into it.
- **Flux.1 [schnell]** and the **SD 3.5 MMDiT** build on the already-ported Flux and SD VAEs.
- **Chronos-Bolt** and **TimesFM 2.5** reuse the T5 encoder-decoder.

## Recommendation

Three candidates are the cleanest near-term wins:

- **BigVGAN v2** — a self-contained MIT vocoder that slots into the existing loader.
- **Chronos-Bolt** — a new modality at near-zero architectural cost over the covered T5.
- **Mimi** — a novel codec with an official MLX reference for parity.

**Flux.1 [schnell]** is the flagship MMDiT port that completes a generation family the toolkit
half-owns. The **Mamba-2 SSM track** is the high-leverage strategic bet, with Codestral-Mamba first as
the pure-SSM testbed, then Granite 4.0-H and Nemotron Nano 2. Its one cost is a selective-scan
implementation in MLX.
