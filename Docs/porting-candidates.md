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
- [Novel but license-blocked](#novel-but-license-blocked)
- [Vendor verdicts](#vendor-verdicts)
- [Connections to existing work](#connections-to-existing-work)
- [Recommendation](#recommendation)

## How to read this

A candidate is scored on four axes:

- **Novelty** — distance from an architecture InferKit already ports. A model that reduces to a covered
  decoder or encoder scores low, whatever its headline capability.
- **License** — whether the weights are redistributable and commercially usable. Code license and weight
  license differ often, and the weight license is the one that gates a redistributable toolkit.
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
| **BigVGAN v2** | NVIDIA | Vocoder | GAN generator with Snake anti-aliased periodic activations and low-pass filtered resampling | MIT | `NVIDIA/BigVGAN` + HF `transformers` | Low–Med |
| **Mimi** | Kyutai | Neural codec | SEANet conv encoder/decoder with interleaved transformer layers and split RVQ (one semantic codebook plus acoustic) | Code MIT, weights CC-BY-4.0 | `kyutai-labs/moshi` (PyTorch and MLX) + HF `MimiModel` | Low–Med |
| **Chronos-Bolt** | Amazon | Time-series forecasting | T5 encoder-decoder over patched observations, direct multi-step quantile forecast | Apache-2.0 | `amazon-science/chronos-forecasting` + HF | Low–Med |
| **Pixtral vision tower** | Mistral | Vision-language | From-scratch variable-resolution ViT with 2D RoPE, feeding a Mistral decoder | Apache-2.0 | HF `transformers` (`PixtralVisionModel`) | Med |
| **Flux.1 [schnell]** | Black Forest Labs | Text-to-image | Hybrid MMDiT, roughly 19 double-stream plus 38 single-stream blocks, rectified flow | Apache-2.0 | HF `diffusers` (`FluxPipeline`) | High |

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
| **Florence-2** | Microsoft | Unified vision: caption, detect, ground, segment, OCR | DaViT dual (spatial + channel) attention encoder → BART-style seq2seq emitting location tokens | MIT | HF `transformers` (`Florence2`) | High |
| **Voxtral-Mini 3B** | Mistral | ASR and speech translation | Whisper-family audio encoder → Mistral dense decoder | Apache-2.0 | HF `transformers` (`VoxtralForConditionalGeneration`) | Med |
| **Granite Speech 3.3-2B** | IBM | ASR and translation | Conformer encoder and window Q-former projector → Granite decoder | Apache-2.0 | HF `transformers` (`GraniteSpeechForConditionalGeneration`) | Med |
| **Canary-1B-v2** | NVIDIA | ASR and speech translation | FastConformer encoder → transformer attention encoder-decoder decoder | CC-BY-4.0 | NeMo (`EncDecMultiTaskModel`) | Med |
| **V-JEPA 2** | Meta | Video and image encoder | ViT trained by joint-embedding predictive masked-latent prediction; inference is a ViT forward | MIT weights | HF `transformers` + `facebookresearch/vjepa2` | Med |
| **Wav2Vec2 / Wav2Vec2-BERT / HuBERT** | Meta | Speech SSL and CTC ASR | Conv feature extractor + transformer encoder + CTC head | Apache-2.0 (base checkpoints) | HF `transformers` | Low–Med |
| **TrOCR** | Microsoft | OCR | ViT/BEiT encoder + RoBERTa decoder, standard seq2seq | MIT | HF `transformers` (`VisionEncoderDecoder`) | Low–Med |
| **Table Transformer (TATR)** | Microsoft | Table detection and structure | Vanilla DETR, ResNet-18 backbone, no deformable attention | MIT | HF `transformers` + official repo | Low–Med |
| **TimesFM 2.5** | Google | Time-series forecasting | Decoder-only over non-overlapping patches, single-pass horizon, 9-quantile head | Apache-2.0 (2.5 only) | HF + `google-research/timesfm` | Med |
| **Cosmos Tokenizer** | NVIDIA | Image and video tokenizer | Causal spatiotemporal autoencoder, continuous latents or discrete codes | Code Apache-2.0, weights NVIDIA Open Model License | `NVIDIA/Cosmos-Tokenizer` + HF | Med |
| **Sa2VA** | ByteDance | Segmentation VLM | SAM 2 and a LLaVA-style VLM fused in a shared token space | MIT | `bytedance/Sa2VA` + HF | Med |
| **SD 3.5 Large/Medium** | Stability AI | Text-to-image | MMDiT-X with QK-norm and dual attention, three text encoders (2×CLIP + T5) | Stability Community (free under $1M revenue) | HF `diffusers` (`SD3Transformer2DModel`) | Med–High |
| **Stable Audio Open 1.0** | Stability AI | Text-to-audio | Latent audio DiT over an Oobleck autoencoder, T5 conditioning | Stability Community | HF `diffusers` (`StableAudioPipeline`) | Med–High |

Notes:

- **Florence-2** is the most novel Microsoft candidate. It performs detection, segmentation, grounding,
  and OCR through one seq2seq decoder that emits coordinate and polygon tokens, a paradigm the toolkit
  lacks.
- **Voxtral-Mini** and **Granite Speech-2B** decompose into a Whisper-family encoder and a decoder the
  toolkit already runs, so the work is mostly wiring. The 24B Voxtral and 8B Granite Speech are too
  large for the target machine; the 3B and 2B variants fit.
- **Canary-1B-v2** reuses the FastConformer encoder from the Parakeet port; the attention
  encoder-decoder decoder is the new work. Canary-Qwen uses a Qwen decoder already covered, so it is a
  lower priority.
- **TrOCR** and **Table Transformer** are cheap, permissive document wins. OCR has no path in the
  toolkit today, and TATR sits close to the covered DETR family.
- **TimesFM 2.5** is a second time-series family. Stay at 2.5 or earlier; TimesFM 3.0 carries a
  non-commercial license.
- **Cosmos Tokenizer** builds on the LTX and Wan 3D-causal-VAE machinery. The code is Apache-2.0 and the
  weights use the commercially usable NVIDIA Open Model License.
- **SD 3.5** and **Stable Audio Open** carry the Stability Community License, which is free below $1M
  annual revenue and not fully permissive above it. Flag the revenue gate before committing.

## Tier 3 — strategic frontier, high cost

| Model | Vendor | Task | Architecture | License | Reference | Effort |
|-------|--------|------|--------------|---------|-----------|--------|
| **Codestral-Mamba 7B** | Mistral | Code LLM | Pure Mamba-2 SSM, linear-time, 256k context | Apache-2.0 | `state-spaces/mamba` (CUDA), `mamba.py` (MLX) | High |
| **Granite 4.0-H** | IBM | LLM | Hybrid Mamba-2 SSM, sparse attention, MoE in H-Small | Apache-2.0 | HF `transformers` (`GraniteMoeHybrid`) | High |
| **Nemotron Nano 2 (9B/12B v2)** | NVIDIA | LLM | Nemotron-H hybrid: Mamba-2 layers, MLP, few attention layers | NVIDIA Open Model License | HF `transformers` (`NemotronH`) | High |
| **BAGEL-7B-MoT** | ByteDance | Any-to-any understand and generate | Mixture-of-Transformer-Experts, dual VAE + ViT encoders | Apache-2.0 | `ByteDance-Seed/BAGEL` (official repo) | High |
| **Phi-4-multimodal** | Microsoft | Text + vision + audio | Phi-4-mini decoder + SigLIP-style vision + Conformer audio + mixture-of-LoRAs | MIT | HF `transformers` | High |
| **HunyuanVideo / Hunyuan3D** | Tencent | Text-to-video / image-to-3D | MMDiT + 3D causal VAE (video); flow-based shape DiT + PBR paint (3D) | Tencent Hunyuan Community | HF `diffusers` (video) / official repo (3D) | High |

Notes:

- The **Mamba / SSM track** is the highest-leverage strategic bet. It introduces the toolkit's first
  state-space layer, and one selective-scan (SSD) implementation serves Codestral-Mamba, Granite 4.0-H,
  and Nemotron Nano 2. The cost is real: Mamba-2's selective scan has no fused MLX kernel and the
  official references are CUDA-only. A pure-MLX `mamba.py` reference exists to check parity against, and
  performance work is expected. Codestral-Mamba is the pure-SSM testbed; Granite and Nemotron add the
  hybrid attention and MoE on top.
- **BAGEL** and **Phi-4-multimodal** are large multi-component integrations whose novelty concentrates
  in one part (the Mixture-of-Transformer routing; the audio Conformer and mixture-of-LoRAs).
- **Hunyuan3D** is the one novel 3D-asset frontier with no covered analog. The Tencent Community License
  excludes the EU, UK, and South Korea and bans training competitors, so it is not truly open.

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
- **Cosmos Tokenizer** extends the LTX and Wan 3D-VAE stack.
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
