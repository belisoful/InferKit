# Removal candidates

A first pass, taken on 2026-09-26 from [the model index](model-index.md), [the parity record](model-parity.md),
and the residency notes in `Docs/agent-reference/mlx-companion.md`. It names the shipped models that are
candidates for InferKit deprecation and the reasoning behind each. It is not an audit: the per-candidate
check of shared code, tests, validation assets, and documentation (see "Open work") has not been done.

## Policy

These rules govern model development, and every decision below follows from them.

Sizes and precision:

- Every released size of a model is supported, tested or not.
- A size that cannot be tested on a 32 GB machine is listed as such (see "Sizes beyond a 32 GB machine").
  Its weights are not downloaded for testing. Every other size is downloaded and tested.
- A superseded release gets no weight download. Its successor's weights are fetched instead.
- Every floating-point precision a model supports is implemented: bfloat16 and float32 at least, and
  smaller formats (float16, 8-bit and 4-bit quantization, MXFP4, fp8) where the model can or should run
  in them.

Subsystems:

- Each model subsystem keeps at least one model that tests it. Where a subsystem needs several models to
  cover it, each of those models is kept. This rule outranks every deprecation rule below.
- The subsystem implements advanced model technology, in particular staging and paging, which run larger
  sizes on smaller machines.
- Customization, training, and fine-tuning ship wherever the model allows them.

Priority:

- The newest models are implemented first. Older models take a lower priority.

Deprecation:

- A retired, obsolete, deprecated, or superseded model is a candidate for InferKit deprecation, where it
  is no longer fully supported.
- Deprecation removes testing. It deletes no code, weights, or backups. A deprecated model stays in the
  source tree and on the Meta backup share.

## Deprecation levels

| Level | Name | What is tested | What remains |
| --- | --- | --- | --- |
| 0 | Supported | every size that fits a 32 GB machine, every precision | full support, new precisions and recipes |
| 1 | Large sizes untested | the sizes that fit comfortably; the larger sizes stop being tested | every size still loads; the larger sizes are structural-only |
| 2 | One size | one released size, at one precision | the other sizes still load; no numeric claim for them |
| 3 | Retro support | the implementation only: tiny-geometry and unit tests, no released weights | the code compiles and its tiny tests pass; no parity claim |

A subsystem exemplar (see "Models kept as subsystem exemplars") cannot go below the level that keeps its
subsystem tested on released weights.

## How a candidate is chosen

A model is a candidate when all of these hold:

1. A newer model for the same task already ships in InferKit, or the upstream release is withdrawn or
   marked superseded.
2. It is not the only released-weight test of a shared subsystem or building block.
3. No newer shipped model reuses its network as a component.

Candidates that meet all three are listed as strong. Candidates that fail one of them are listed as weak,
with the reason.

## Strong candidates

| Model | Superseded by (shipped) | Proposed level | Note |
| --- | --- | --- | --- |
| YOLOv8 (`NFKMLXYOLO`) | YOLOv9 through YOLO26 (`NFKMLXYOLOGenerations`) | 2 | v8 remains widely deployed; one size keeps it honest |
| RT-DETR v1 configurations (`NFKMLXRTDetr`) | RT-DETRv2 in the same class, RF-DETR | 3 | v1 and v2 share the class, so v2's tests cover the code |
| Depth Anything V2 (`NFKMLXDepthAnything`) | Depth Anything 3 | 2 | V2's ConvTranspose gating fix is shared with DA3; confirm before level 3 |
| BiSeNet V1 (`NFKMLXBiSeNet`) | BiSeNet V2 | 3 | |
| SAM, the original (`NFKMLXSAM`) | SAM 2 / 2.1, SAM 3 / 3.1 | 2 | confirm no newer model reuses its image encoder |
| U²-Net (`NFKMLXU2Net`) | IS-Net (DIS), BiRefNet | 3 | |
| Colorizer ECCV-16 (`NFKMLXColorizer`) | DDColor | 3 | |
| Colorizer SIGGRAPH-17 (`NFKMLXSiggraphColorizer`) | DDColor | 3 | |
| Zero-DCE (`NFKMLXZeroDCE`) | Zero-DCE++ | 2 | the first training-loss-parity recipe; keep one size until Zero-DCE++'s recipe carries the same checks |
| RIFE HDv3 (`NFKMLXRIFE`) | RIFE v4 (`NFKMLXRIFEv4`) | 3 | |
| SimpleBaseline pose (`NFKMLXPose`) | ViTPose | 2 | `NFKMLXResNetBackbone` may be shared; confirm before level 3 |
| DeepSeek V4 Flash / V4 Pro (`NFKMLXDeepSeek`) | DeepSeek V4.1 Flash, same class | 1 | V4.1 carries the paging test; V4 and V4 Pro keep their bit-exact records |
| Gemma 2 (`NFKMLXGemma2Net`) | Gemma 3, Gemma 4 | 3 | already has no entry class; 9B and 27B are structural-only |
| FastSpeech2 conformer + HiFi-GAN (`NFKMLXVoice`) | Kokoro, Chatterbox | 2 | HiFi-GAN is a vocoder building block; keep its tiny tests |

## Weak candidates

| Model | Successor | Why it is weak |
| --- | --- | --- |
| Demucs v2 (`NFKMLXDemucs`) | HT Demucs v4 | the speech denoiser runs on the same `NFKMLXDemucsNet`, so one of the two must keep a released-weight test |
| LTX-Video 0.9.0 | LTX-2 | LTX-2 has no entry class yet; LTX-Video is also one of the six staging adopters |
| SD ×4 upscaler (`NFKMLXSDUpscaler`) | Real-ESRGAN, HAT (different method) | no same-method successor; shares the SD UNet and autoencoder |
| BasicVSR (`NFKMLXVideoSR`) | none shipped | old, but the only video super-resolution model |
| Stable Diffusion 2.1 | SDXL-Turbo, SD3 / 3.5 | SD 1.5, 2.1, and SDXL-Turbo are one pipeline body; 2.1 can drop to level 2 without losing the code |
| Fast style transfer (`NFKMLXStyleTransfer`) | AdaIN (arbitrary style) | different use: one fixed style per checkpoint |
| Qwen2 / Qwen2-MoE / Mixtral | Qwen3, Qwen3-MoE, gpt-oss | expert-paging layouts; the tiny-geometry paging tests cover each layout, so level 3 keeps the paging code tested |
| Wan 2.1 T2V | Wan 2.2 | Wan 2.1's VAE and transformer code is shared by Wan 2.2 and Qwen-Image |
| Parakeet-TDT v2 | Parakeet-TDT v3 | same class; v3 is multilingual. Level 2 costs nothing |
| TrOCR | Florence-2, vision-language OCR | still the only handwriting-line reader |

## Models kept as subsystem exemplars

These are older but stay at level 0 or 1, because newer models or subsystems depend on them.

- **CLIP** (`NFKMLXCLIP`): the Stable Diffusion text encoder and the CLIP probe's backbone.
- **SAM 2 / 2.1**: Sa2VA reuses `NFKMLXSAM2TrackerNet` whole.
- **Gemma 3**: the base of TranslateGemma and EmbeddingGemma.
- **T5 v1.1 / umT5** (`NFKMLXT5Encoder`): the text encoder of LTX-Video, Wan, and SD3.
- **Codestral-Mamba** (`NFKMLXMamba`): the pure state-space exemplar; its mixer serves Granite 4.0-H and
  Nemotron-H.
- **gpt-oss-20b**: the released-weight exactness test of expert paging (max abs difference 0.0 against
  the resident model).
- **The staging adopters**: FLUX.1, FLUX.2, Qwen-Image 2.1, LTX-Video, Wan, and MiniMax Music 3. At least
  one must keep a released-weight staged test; FLUX.2 calls the stage-load check, so it is the natural
  keeper.

## Sizes beyond a 32 GB machine

The reference machine is an M1 Max with 32 GB. On a quiet machine, with no other programs holding
memory, it runs models up to about 30 GB of weights. That is the practical ceiling for testing.

`NFKMLXResidencyBudget` plans more conservatively: 0.85 of Metal's recommended 25 GB working set, less a
4 GiB reserve, about 17 GB. That budget assumes the machine is also serving other programs. A model
between 17 and 30 GB is therefore tested on a quiet machine, and its test documents that requirement.
A model over 30 GB can still be tested staged, paged (a mixture), or through a layer cut
(`Tools/validation-assets/truncate.py`).

First-pass list, from parameter counts at bfloat16 (2 bytes per parameter). Confirm each against
`NFKMLXModelSizing` before acting on it.

Testable on a quiet machine (17 to 30 GB):

| Size | bf16 weights |
| --- | --- |
| Gemma 2 9B, Nemotron-Nano-9B-v2, FLUX.2 klein 9B, Qwen3-VL-8B, Qwen3-Embedding-8B, SD3.5-large 8B | 16 to 18 GB |
| Gemma 3 12B, Gemma 4 12B, FLUX.1 schnell / dev (12B transformer, staged) | about 24 GB |
| Qwen3-14B, Wan 2.2 Animate 14B (staged) | about 28 GB |
| BAGEL-7B-MoT (14B total, one 29.2 GB file) | about 29 GB |

Beyond the machine (over 30 GB):

| Size | bf16 weights | Testable on 32 GB by |
| --- | --- | --- |
| Qwen3-32B, Qwen3-VL-32B, Gemma 3 27B, Gemma 2 27B, Gemma 4 31B, Qwen3.8-27B, Open-Jev-27B | about 54 to 66 GB | layer cut only |
| Mistral-Small 3.1 / 3.2 24B | about 48 GB | layer cut only |
| LTX-2.3 22B | about 44 GB | staging plus a layer cut |
| Qwen-Image 2.1 (20B transformer) | about 40 GB | staging plus a layer cut |
| Qwen3-30B-A3B, Qwen3-VL-30B-A3B, Gemma 4 26B-A4B, Qwen3.8-Flash-Next | 52 GB and up | expert paging |
| DeepSeek V4.1 Flash, V4 Flash, V4 Pro | hundreds of GB | expert paging (fully mapped) |

The dense rows over 30 GB are level 1 cases by definition: supported, structural-only, not downloaded
for testing. The mixtures stay testable through paging. The download rule applies to new downloads. The
Meta backup share keeps what it already holds.

A release stored at a wider precision than it runs at is sized at the precision it runs at.
HunyuanVideo-1.5 stores each transformer variant as 33.3 GB of float32, about 17 GB at bfloat16.

## Open work

- Per-candidate audit before any level changes: every other file that references the candidate's
  network, the tests and validation keys that read its weights, and its customization-ledger row.
- A precision audit: which families ship bfloat16 and float32 and which smaller formats they could take.
- Recording each model's level in the model index and the customization ledger once the levels are
  agreed.
