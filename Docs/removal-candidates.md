# Removal candidates

A first pass, taken on 2026-09-26 from [the model index](model-index.md), [the parity record](model-parity.md),
and the residency notes in `Docs/agent-reference/mlx-companion.md`. It names the shipped models that are
candidates for InferKit deprecation and the reasoning behind each. The per-candidate audit of 2026-11-26
checked each candidate's released-weight tests, shared code, and ledger row; "Applied levels" records
its result.

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
- Deprecation removes testing. It deletes no code or weights. A deprecated model stays in the source
  tree and in the model store (`/Volumes/InferKit Models`) until it is retired from support entirely.

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

Releases that share one pipeline body (one network class, one loop, configurations apart) are ranked by
one tie-breaker: the newer release keeps its level and the older release drops. Rules 2 and 3 outrank the
tie-breaker, so an older release that a shipped model reuses as a component, or that alone tests a
subsystem on released weights, keeps its level over a newer one that does not.

## Applied levels

Applied 2026-11-26 after the per-candidate audit. A retired release keeps its code, its download path,
and its tiny-geometry and unit tests. Its released-weight tests skip through
`NFKMLXSupportLevels.retired` (InferKitMLX tests), and `IK_TEST_RETIRED=1` runs them anyway.
`NFKMLXSupportLevelsTests` fails when that table disagrees with this one, or when the "Support levels"
tables in [the model index](model-index.md) and the
[customization ledger](agent-reference/mlx-customization-ledger.md) omit a retired release.

| Model | Retired release | Level | What stays tested on released weights |
| --- | --- | --- | --- |
| Colorizer ECCV-16 | `colorizer-eccv16` | 3 | nothing; DDColor tests `NFKLabColor`, and the pose loader test keeps the raw `.pth` ConvTranspose2d layout |
| Zero-DCE | `zero-dce` | 3 | nothing; its loss (`NFKMLXZeroDCELoss`) is tested through Zero-DCE++'s training parity |
| SAM, the original | `sam-vit-b` | 3 | nothing; SAM 2's decoder test covers the prompt encoder and two-way blocks SAM 2 and Sa2VA reuse |
| SAM, the original | `sam-vit-l` | 3 | |
| SAM, the original | `sam-vit-h` | 3 | |
| Hunyuan-MT 7B | `hunyuan-mt-7b` | 3 | nothing; HY-MT1.5 7B and Hy-MT2 7B run the same network at both precisions. Chimera 7B stays: it alone reads candidate translations |
| Whisper large-v2 | `whisper-large-v2` | 3 | nothing; large-v3 tests the large geometry and medium the 80-mel front end |
| YOLOv8 | `yolov8s` | 2 | nano, the only released-weight test of the pickled ultralytics model loader |
| YOLOv8 | `yolov8m` | 2 | |
| YOLOv8 | `yolov8l` | 2 | |
| YOLOv8 | `yolov8x` | 2 | |
| Depth Anything V2 | `depth-anything-v2-base` | 2 | small, which carries the only shipped depth fine-tune recipe and the live timing |
| Depth Anything V2 | `depth-anything-v2-large` | 2 | |
| Gemma 2 | `gemma-2-9b` | 2 | 2B, which is SANA's caption encoder |
| Gemma 2 | `gemma-2-27b` | 2 | |
| M2M-100 | `m2m100-1.2b` | 2 | 418M, with its probe pairs and the long-paragraph test |
| Gemma 3n | `gemma3n-e4b` | 2 | E2B at both precisions; the MatFormer extraction test still reads E4B |
| RT-DETR v1 | `rtdetr-r18vd` | 2 | r50vd; RT-DETRv2 tests all four geometries, and the r18vd live timing still runs |
| RT-DETR v1 | `rtdetr-r34vd` | 2 | |
| RT-DETR v1 | `rtdetr-r101vd` | 2 | |

Recorded at a level with no test change, because the level already describes their state:

- Level 3, no released-weight test exists: DeepSeek V4 Flash and V4 Pro (tiny oracles of the V4-only
  compressor, indexer, and draft stack stay), Qwen2 dense, and SD3 Medium.
- Level 2, one size at one precision already: Stable Diffusion 2.1 base, BiSeNet V1, SimpleBaseline pose,
  Colorizer SIGGRAPH-17, fast style transfer, Demucs v2, Parakeet-TDT v2, Qwen2-MoE, Mixtral, FastSpeech2
  with HiFi-GAN, TripoSR, LTX-2.3, and Chronos-Bolt.

Stable Diffusion 2.1 base. Stability AI withdrew the release, and the catalog downloads it from a pinned
mirror. No test loads the base release. The `IK_VAL_SD21*` keys hold the 768 v-prediction fine-tune
`friedrichor/stable-diffusion-2-1-realistic`, which is the only released-weight test of the OpenCLIP-H text
tower, the v-prediction sampler path, and `upcast_attention` in the shared SD pipeline. Those tests stay
until another shipped model covers the three.

Held for a prerequisite:

- LTX-Video 0.9.0 to level 3: its LoRA recipe ships at level 0 first, and the retirement then skips the
  recipe's released-weight step with the four parity tests.
- RT-DETR v1 to level 3: the `rtdetr-r18vd` live entry moves to `rtdetr-v2-r18vd`.
- U²-Net to level 3: the `u2netp` live entry moves to IS-Net or is dropped.
- RIFE HDv3 to level 3: RIFE v4 gains the clip backend `rife-clip` provides.
- Table Transformer structure v1 to level 3: its seam and backend tests move to v1.1-all.

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
| Gemma 2 (`NFKMLXGemma2Net`) | Gemma 3, Gemma 4 | 2 | SANA's caption encoder is `NFKMLXGemma2Net`, so rule 3 keeps 2B; 9B and 27B are tested through 4-layer cuts |
| FastSpeech2 conformer + HiFi-GAN (`NFKMLXVoice`) | Kokoro, Chatterbox | 2 | HiFi-GAN is a vocoder building block; keep its tiny tests |

## Weak candidates

| Model | Successor | Why it is weak |
| --- | --- | --- |
| Demucs v2 (`NFKMLXDemucs`) | HT Demucs v4 | the speech denoiser runs on the same `NFKMLXDemucsNet`, so one of the two must keep a released-weight test |
| LTX-Video 0.9.0 | LTX-2 | held for level 3 (below): `NFKMLXLTX2Generator` ships and stages, and LTX-Video's staging check reads shapes only |
| SD ×4 upscaler (`NFKMLXSDUpscaler`) | Real-ESRGAN, HAT (different method) | no same-method successor; shares the SD UNet and autoencoder |
| BasicVSR (`NFKMLXVideoSR`) | none shipped | old, but the only video super-resolution model |
| Stable Diffusion 2.1 | SDXL-Turbo, SD3 / 3.5 | SD 1.5, 2.1, SDXL base, SDXL-Turbo, and LCM Dreamshaper are one pipeline body. The tie-breaker keeps SDXL-Turbo and SDXL base as the newer SDXL releases, each with a released-weight end-to-end test (Turbo alone, base under the Lightning adapter). SD 1.5 is older than 2.1 but rule 3 holds it: SD inpainting, the IP-Adapter, the LCM-LoRA pairing, and the LoRA recipe run on its UNet, and the distillations are untrainable. Nothing shipped reuses 2.1, and Stability AI withdrew its repository, so 2.1 is at level 2 (see "Applied levels") |
| Fast style transfer (`NFKMLXStyleTransfer`) | AdaIN (arbitrary style) | different use: one fixed style per checkpoint |
| Qwen2 / Qwen2-MoE / Mixtral | Qwen3, Qwen3-MoE, gpt-oss | Qwen2-MoE's cut is the only released-weight test of the gated shared expert, and Mixtral's the only `.metaspace` tokenizer and its catalog pin's measurement, so both stay at level 2 |
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
| Mistral-Small 3 (2501) 24B | about 48 GB | layer cut only |
| LTX-2.3 22B | about 44 GB | staging plus a layer cut |
| Qwen-Image 2.1 (20B transformer) | about 40 GB | staging plus a layer cut |
| Qwen3-30B-A3B, Qwen3-VL-30B-A3B, Gemma 4 26B-A4B, Qwen3.8-Flash-Next | 52 GB and up | expert paging |
| DeepSeek V4.1 Flash, V4 Flash, V4 Pro | hundreds of GB | expert paging (fully mapped) |

The dense rows over 30 GB are level 1 cases by definition: supported, structural-only, not downloaded
for testing. The mixtures stay testable through paging. The download rule applies to new downloads. The
model store keeps what it already holds.

A release stored at a wider precision than it runs at is sized at the precision it runs at.
HunyuanVideo-1.5 stores each transformer variant as 33.3 GB of float32, about 17 GB at bfloat16.

## Open work

- The prerequisites under "Applied levels".
- A precision audit: which families ship bfloat16 and float32 and which smaller formats they could take.
- The validation store keeps the retired releases' weights. Removing them needs the
  approval of the validation store's maintainer.
