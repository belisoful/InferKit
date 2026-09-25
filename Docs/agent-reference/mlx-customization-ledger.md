<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX customization ledger

Every shipped MLX model, with the customization outcome the parity rule requires. The rule and the
minimum shipped set are in [mlx-training.md](mlx-training.md) ("Customization is part of parity").
This file is the index: one row per model entry, so a session can see what is settled, claim a model,
and know what it has to read before it writes the recipe. The prose that explains a ruling belongs in
the model's own `mlx-models-<class>.md` entry, not here.

**Triaged 2026-09-24** over the 164 model entries in the seventeen class files.

## How to use it

- Picking up a model → read its row, then its entry, then the reference file the row names.
- Shipping a recipe → follow the minimum shipped set in `mlx-training.md`, then update this row AND
  the model's entry AND the listings in [mlx-parity-checklist.md](mlx-parity-checklist.md).
- A row marked `uncertain` is not a gap in the model. It is a gap in what has been read. The row
  names the file that settles it. Resolve it to `trainable`, `offline`, or `untrainable` before
  writing any code, because the answer decides whether there is code to write.

## Legend

| Outcome | Meaning |
| --- | --- |
| `ships` | The customization path is shipped end to end and reachable without `@testable`. |
| `trainable` | A path can be implemented on an Apple-silicon device. The level names which one. |
| `offline` | The reference recipe needs what a device cannot hold. The row names what. |
| `untrainable` | No differentiable objective, or the reference publishes no training code. |
| `uncertain` | Neither the entry nor the source settles it. The row names what would. |

Levels are `probe`, `head-retarget`, `zero-reference`, `LoRA`, and `full`, as `mlx-training.md`
defines them.

**A public builder is not evidence of a training path.** Qwen4-Exp and Mamba-2 have fully public
builders and are offline on size. The dense Qwen, hybrid, and Gemma 3 decoders are LoRA-feasible at
4B and under and have no public builder at all. Feasibility and reachability are separate questions,
and a row answers the first. The `Reach` column answers the second.

## Summary

| Outcome | Rows |
| --- | --- |
| `ships` | 33 |
| `trainable`, no recipe yet | 54 |
| `offline` | 38 |
| `uncertain` | 15 |
| `untrainable` | 7 |
| Shared infrastructure, no objective of its own | 1 |
| Total rows | 148 |

The 164 model entries become 148 rows because a few entries take one ruling for several symbols: the
generation pipelines share a row, the schedulers share a row, and Gemma's parameter-free adapters
share a row.

Thirty-three recipes ship and 54 models are trainable with none written. That is the size of
the work the rule creates.

The largest single finding: **the detector losses are published and portable.** ultralytics ships
`v8DetectionLoss` over `TaskAlignedAssigner`, and transformers maps the RT-DETR and RF-DETR families
onto Hungarian matchers this package already reproduces in `NFKMLXHungarian`. Three of those were
read on disk rather than assumed. Detection was previously set aside as too expensive to port, and
that call was wrong.

The second: **eight audio models are trainable at `full` and all are small.** GTCRN is under one
megabyte, DeepFilterNet3 is eight, NU-Wave 2 and Conv-TasNet nineteen each. Their references publish
a plain reconstruction or score-matching objective with no adversary, verified in each repository's
own loss module.

## Depth, segmentation, matting

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXSAM2` | ships | head-retarget | public | The reference's `MultiStepMultiMasksAndIous`, trunk and memory path frozen. |
| `NFKMLXSAM3` | ships | head-retarget | public | `odinw_text_only_train.yaml`'s matcher and losses. A full fine-tune is offline at 3.44 GB. |
| `NFKMLXSegFormer` | ships | head-retarget | public | Shipped end to end: mmseg's cross entropy, measured by `run_reference.py segformer_loss`. |
| `NFKMLXSAM` | trainable | head-retarget | internal | No SAM 1 training source is vendored. The shipped SAM 2 objective covers the same two-way decoder. |
| `NFKMLXU2Net` | trainable | full | internal | 44M parameters, BCE over the side maps. `xuebinqin/U-2-Net`'s training script pins the optimizer. |
| `NFKMLXISNet` | trainable | full | internal | The vendored `isnet.py` itself ships `muti_loss_fusion` and its `ISNetGTEncoder` teacher. Read on disk. |
| `NFKMLXBiRefNet` | trainable | head-retarget | internal | The decoder trains with Swin-v1-L frozen. `ZhengPeng7/BiRefNet` pins the terms. |
| `NFKMLXRVM` | trainable | full | internal | An existing test already trains the tiny net to a falling loss with an ad-hoc alpha term. `PeterL1n/RobustVideoMatting` holds the real terms. |
| `NFKMLXMODNet` | trainable | full | internal | 6M parameters, three separately supervised branches. `ZHKKKe/MODNet` pins them. |
| `NFKMLXDeepLab` | trainable | head-retarget | internal | Per-pixel cross-entropy. torchvision's criterion is in `references/segmentation`, not the wheel. |
| `NFKMLXBiSeNet` | trainable | head-retarget | internal | The checkpoint's two auxiliary heads are training-only supervision. `CoinCheung/BiSeNet` pins the weights. |
| `NFKMLXBiSeNetV2` | trainable | head-retarget | internal | Same repository. Its four auxiliary heads the port neither builds nor loads. |
| `NFKMLXDepthAnything` | uncertain | — | internal | Read `DepthAnything/Depth-Anything-V2` for a training script. The relative release is a distillation product and its recipe may not be published. |
| `NFKMLXDepthAnything3` | uncertain | — | internal | Read the `depth_anything_3` GitHub repository. The 0.1.1 wheel was read and carries no loss module. |
| `NFKMLXResNetBackbone` | n/a | — | internal | A shared backbone with no objective of its own. |

## Detection and pose

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXTableTransformer` | ships | head-retarget | public | The vendored DETR `SetCriterion` behind a Hungarian matcher, measured term by term. |
| `NFKMLXYOLO` | ships | full | public | ultralytics' `v8DetectionLoss`, `optimizer=auto`, schedule, and `ModelEMA`, each matched; the class retarget transfers shapes alike. |
| `NFKMLXYOLOGenerations` | ships | full | public | YOLOv8's recipe, with `E2ELoss` for the v10 and YOLO26 end-to-end heads, matched on both. |
| `NFKMLXRTDetr` | ships | full | public | The original repository's criterion, denoising queries, and each release's configuration (freezing, AdamW groups, v2's warm-up), all eight releases matched; the class retarget transfers shapes alike. |
| `NFKMLXRFDetr` | trainable | head-retarget | partial | transformers `LwDetrForObjectDetectionLoss`. Read on disk. |
| `NFKMLXRFDetrSegmentationNet` | trainable | head-retarget | public | transformers `RfDetrForSegmentationLoss`. Its point-sampled mask terms reduce to gathers MLX has. |
| `NFKMLXPose` | trainable | head-retarget | internal | Heatmap MSE. The objective is in mmpose's own losses. |
| `NFKMLXVitPose` | trainable | head-retarget | internal | transformers states its loss is unsupported and points at `ViTAE-Transformer/ViTPose`. |
| `NFKMLXRetinaFace` | uncertain | — | internal | Read `biubug6/Pytorch_Retinaface`'s `multibox_loss.py`. facexlib here ships detection and no loss. |

## Image restoration

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXZeroDCE` | ships | zero-reference | public | Four `Myloss.py` losses measured by `run_reference.py zero_dce_losses`. |
| `NFKMLXNAFNet` | trainable | full | internal | Paired-data restoration. megvii-research's training option file supplies the loss. |
| `NFKMLXStyleTransfer` | trainable | full | internal | 1.7M parameters. The VGG-16 perceptual and Gram loss needs `NFKMLXVGG16Features`, which already ships publicly. |
| `NFKMLXAdaIN` | trainable | full (decoder) | internal | The encoder stays frozen and the loss network is the model's own ported VGG-19 encoder. |
| `NFKMLXZeroDCEPlus` | trainable | zero-reference | internal | The same four losses already ported. Only the shared curve and the `scaleFactor` 12 estimator differ. |
| `NFKMLXSwinIR` | trainable / offline | full | internal | Classical and lightweight train on L1. The two real-world releases are GANs whose discriminator is not ported. |
| `NFKMLXHAT` | trainable / offline | full | internal | `.base` and `.large` are reconstruction networks. `.realWorld` carries an adversarial recipe. |
| `NFKMLXRealESRGAN` | offline | — | internal | The generator alone is ported. The published recipe adds a discriminator no file here holds. |
| `NFKMLXLaMa` | offline | — | internal | big-lama pairs the FFC-ResNet generator with an adversarial discriminator and a perceptual network. |
| `NFKMLXCodeFormer` | offline | — | internal | The staged reference recipe needs a discriminator and a codebook-learning stage. |
| `NFKMLXColorizer` | uncertain | — | internal | Read richzhang/colorization's training directory for the class-rebalancing prior the 313-bin loss needs. |
| `NFKMLXDDColor` | uncertain | — | internal | Read piddnad/DDColor's `options/train/*.yml`. A `network_d` makes it offline; no training code makes it untrainable. |

## Video

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXVJEPA2` | ships | probe | public | `evals/video_classification_frozen` freezes the encoder and trains the attentive pooler and classifier. |
| `NFKMLXCosmosTokenizer` | ships | full | public | cosmos-predict1's post-training objective, with `.everything` and `.decoder` policies. |
| `NFKMLXRIFE` | trainable | full | internal | The reference distills through the `block_tea` teacher the entry records as dropped at load. |
| `NFKMLXRAFT` | trainable | full | internal | 5.3M parameters. The sequence loss needs ground-truth flow, and the correlation volume bounds the crop size. |
| `NFKMLXVideoSR` | trainable | full | internal | BasicVSR's mmediting configuration supplies the loss and the reduced SPyNet rate. Memory scales with clip length. |
| `NFKMLXWanAnimate` | offline | — | public | The smallest released form is 32.8 GB in bfloat16 against 32 GiB of unified memory. |
| `NFKMLXRIFEv4` | uncertain | — | internal | Read hzwer/Practical-RIFE for a v4 training script. v4's architecture ships inside the model zip. |
| `NFKMLXVideoBackend` | untrainable | — | n/a | The AVFoundation decode and encode layer holds no parameters. |

## Text to speech

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXChatterbox` | trainable (T3) / offline (HiFT) | LoRA | public nets | The 520M T3 decoder takes teacher-forced cross-entropy, the Whisper and translator pattern. The HiFT vocoder is the discriminator case. |
| `NFKMLXNeuralG2P` | trainable | full | internal | This package's own compact encoder-decoder. Sequence cross-entropy over a pronunciation lexicon is the whole recipe. |
| `NFKMLXTTS` | trainable (acoustic) / offline (vocoder) | full | internal | FastSpeech2 needs an external forced aligner for its duration, pitch, and energy targets. The vocoder's discriminators are in jik876's `models.py`. |
| `NFKMLXKokoro` | offline | — | public | The iSTFTNet decoder trains adversarially under StyleTTS2 and the vendored reference is inference-only. |

## Speech restoration

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXGTCRN` | ships | full | public | The repo's `HybridLoss`, matched exactly by `run_reference.py gtcrn_loss`, over 48.2K parameters. |
| `NFKMLXSGMSE` | trainable | full | internal | `sgmse/model.py` `_loss` is denoising score matching. `train.py` has no discriminator. |
| `NFKMLXStoRM` | trainable | full | internal | `_loss` is MSE for the discriminative predictor and for the score net. No adversary in either role. |
| `NFKMLXDeepFilterNet` | trainable | full | internal | The installed `df/train.py` assembles spectral, mask, SDR, and local-SNR terms with no discriminator. |
| `NFKMLXNUWave2` | ships | full | public | `NuWave2.common_step`'s noise-prediction L1, measured on the official checkpoint by `run_reference.py nuwave2_loss`. |
| `NFKMLXMPSENet` | offline | — | internal | `train.py` builds a `MetricDiscriminator` and the release ships only the generator half. |
| `NFKMLXResembleEnhance` | offline | — | internal | `enhancer/train.py` defines `load_D` over a discriminator the released shard does not carry. |
| `NFKMLXMetricGANPlus` | offline | — | internal | The learned metric is the whole training signal. |
| `NFKMLXCMGAN` | offline | — | internal | The discriminator is a training device the release omits. |
| `NFKMLXMossFormer2SRNet` | offline | — | internal | The reference's generator ships three discriminators and a feature loss. |
| `NFKMLXApollo` | offline | — | internal | `apollo.yaml` configures a frequency discriminator and a second optimizer. |
| `NFKMLXMossFormer2SENet` | uncertain | — | internal | Read the `train/` tree of modelscope/ClearerVoice-Studio. The local clone is inference only. |
| `NFKMLXFRCRN` | uncertain | — | internal | The same `train/` tree. |
| `NFKMLXVoiceRestore` | uncertain | — | internal | Read skirdey/voicerestore for a training script. The cloned class stores `sigma` and defines only `sample`. |

## Source separation

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXDemucs` | trainable | full | internal | demucs 4.0.1 `solver.py` trains with an L1 waveform loss and no adversary. |
| `NFKMLXHTDemucs` | trainable | full | internal | The same solver. 42M parameters, an 81 MB release. |
| `NFKMLXConvTasNet` | ships | full | public | asteroid v0.5.2's `PITLossWrapper(pairwise_neg_sisdr)`, matched by `run_reference.py convtasnet_loss`. 5M parameters. |
| `NFKMLXDenoiser` | uncertain | — | internal | Read `denoiser/solver.py` and `stft_loss.py`. The local clone holds the architecture only. |

## Audio codecs and music

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXDAC` | offline | — | internal | The installed package ships `discriminator.py` and a `GANLoss`. |
| `NFKMLXBigVGAN` | offline | — | public net | Multi-resolution and multi-period discriminators, with no inference weights shipped for them. |
| `NFKMLXMimi` | offline | — | public net | The codec trains adversarially against discriminators the release does not ship. |
| `NFKMLXMusic3` | offline | — | internal | The 16 GiB language model and the 9.7 GB transformer exceed a 32 GB working set, so the stack runs staged. |
| `NFKMLXSNAC` | uncertain | — | internal | Read hubertsiuzdak/snac for a training script. The installed package is inference-only. |

## Stable Diffusion and the diffusion seam

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXSDUNet` / `NFKMLXSDAutoencoder` | trainable | LoRA | public | The attention projections are `Linear`, and the epsilon objective runs over the parity-verified `addNoise`. |
| `NFKMLXSDPipeline` | trainable | LoRA | public | The round trip already exists: `loadWeights(from:)` reads the single file `NFKMLXWeights.save` writes. |
| `NFKMLXStableDiffusionInpaint` | trainable | LoRA | partial | The SD 1.5 UNet at nine input channels. The same objective with the mask channels held as input. |
| `NFKMLXTextToImage` | trainable | LoRA | partial | SD 1.5 and SD 2.1 hold with the backbone frozen. `.sdxlTurbo` is offline on SDXL's grounds. |
| `NFKMLXIPAdapter` | trainable | LoRA (adapter) | public | The entry already names the trained set: the projection and `to_k_ip` / `to_v_ip` over a frozen UNet. |
| `NFKMLXSDTextEncoderNet` | offline | — | public | Its reference objective is CLIP's contrastive loss, which needs the large batch of negatives. |
| `NFKMLXMarigold` / `NFKMLXSDUpscaler` | uncertain | — | internal | Read prs-eth/Marigold for its affine-invariant depth loss and the upscaler's noise-level schedule. |
| `NFKMLXTAESD` | uncertain | — | internal | Read madebyollin/taesd. A published distillation script makes it `full`; a discriminator makes it offline. |
| `NFKMLXSDPromptTokenizer` | untrainable | — | n/a | A tokenizer carries no parameters. |
| `NFKMLXDiffusionBackend` | untrainable | — | n/a | The seam holds no weights. The model arrives as the consumer's closures. |

## Transformer generation

Every transformer in this class is offline on weights except SANA, and the pattern is uniform: the
residual or the velocity objective is known, and the working set is what rules it out.

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXSANATransformerNet` | trainable | LoRA | partial | The measured `Sana_600M` release holds with the backbone frozen. NVlabs' own script is unread, which item 5 of the minimum set needs. |
| `NFKMLXLTXTransformer` | offline | — | internal | 2B at ~7.7 GB sharded, before a video-length activation stack. |
| `NFKMLXLTX2TransformerNet` | offline | — | public | No size of this 22B model fits a consumer machine. |
| `NFKMLXT5Encoder` | offline | — | internal | T5-XXL is ~19 GB at float32, the memory crux of the LTX pipeline. |
| `NFKMLXZImageTransformerNet` | offline | — | partial | 6B, past the working set at the released precision. |
| `NFKMLXWanTransformerNet` | offline | — | partial | The `.base` geometry is the 14B release. Wan 2.1 at 1.3B would hold, but its activation stack is unmeasured. |
| `NFKMLXQwenImageNet` | offline | — | public | 7.1B is 14 GB at bfloat16 before activations, and the backward pass holds thirty-two blocks of them. |
| `NFKMLXSD3TransformerNet` | offline | — | public | The presets are 2B, 2.5B, and 8B. |
| `NFKMLXFluxTransformerNet` | offline | — | public | ~24 GB resident for the 12B transformer. |
| `NFKMLXFlux2TransformerNet` | offline | — | public | The smallest release is 3.88B and `[dev]` is 32B. |
| `NFKMLXSD3ControlNetNet` | offline | — | public | Its zero-initialized residuals take their gradient through the frozen 2B to 8B base. |
| `NFKMLXFluxControlNetNet` | offline | — | public | The same residual gradient through the frozen 12B FLUX.1 transformer. |
| `NFKMLXLTXPipeline`, `NFKMLXSD3Pipeline`, `NFKMLXFluxPipeline`, `NFKMLXQwenImagePipeline`, `NFKMLXFlux` | offline | — | mixed | The glue holds no weights of its own and inherits its transformer's working set. |
| `NFKMLXQwenImageVAE` | uncertain | — | public | Read the Qwen-Image release or diffusers for autoencoder training code. The same question covers `NFKMLXWanVideoVAENet`, `NFKMLXDCAutoencoderNet`, and `NFKMLXFlux2LatentCodec`. |
| `NFKMLXLTXVideoVAE` | uncertain | — | internal | Read Lightricks/LTX-Video for autoencoder training code. Reconstruction makes it `full`; a discriminator makes it offline. |
| `NFKMLXFlowMatchScheduler`, `NFKMLXDPMSolverScheduler`, `NFKMLXUniPCScheduler` | untrainable | — | n/a | Value types with no parameters. |

## Language

Size is the deciding variable here, and the line falls at 4B, which the entries measure as the
largest size this machine holds at float32.

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXGraniteHybrid` | ships | LoRA | public | The reference's `labels=` loss within 1e-3. The 1B release fits float32. |
| `NFKMLXNemotronH` | ships | LoRA | public | Matches within 1e-3. The only release is 9B at ~17.8 GB bfloat16, and the trainer has no bfloat16 path. |
| `NFKMLXLanguage` (dense) | trainable to 4B, offline above | LoRA | **internal** | `q_proj` and `v_proj` are `Linear` under `@ModuleInfo`, which LoRA requires. The builder, the loader, and the initializer are all internal. |
| `NFKMLXHybridLanguage` | trainable at 2B and 4B, offline at 27B | LoRA | **internal** | Open-Jev already LoRA-trains this decoder at Qwen3.5-2B. Qwen3.8-27B is ~54 GB. |
| `NFKMLXGemma2Net` | trainable at 2B, offline at 9B and 27B | LoRA | public | The 2B release runs at float32 in the entry's own layer probe. |
| `NFKMLXLanguage` (mixture of experts) | offline | — | internal | LoRA adapts `Linear` only and never the expert switch layers. gpt-oss's experts stay MXFP4-packed at load. |
| `NFKMLXQwen4Exp` | offline | — | public | The smallest release is 180B. Its n-gram table alone is 51 billion parameters. |
| `NFKMLXDeepSeek` | offline | — | internal | Four-bit routed experts and fp8 attention. A gradient needs 510 GB dequantized first. |
| `NFKMLXMamba` | offline | — | public | The smallest release is Codestral-Mamba-7B, which does not fit float32 here. |
| `NFKMLXRoPEScaling`, `NFKMLXChatTemplateRenderer` | untrainable | — | n/a | No parameters. |

## Gemma

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXGemma3` | trainable at 270M, 1B, 4B; offline at 12B and 27B | LoRA | **internal** | `NFKMLXTranslateGemma.fineTune` already LoRA-adapts a Gemma 3 net from a float32 load. |
| `NFKMLXGemma4VisionNet` | trainable | probe | internal | Its projections are `NFKGemmaClippableLinear`, which LoRA cannot adapt, so a head over the pooled soft tokens is the level. |
| `NFKMLXGemma4AudioNet` | trainable | probe | internal | The same limit. Google publishes no training code, so the objective would be this package's, on the CLIP-probe precedent. |
| `NFKMLXGemma3n` | offline | — | internal | E2B is 10 GB and E4B is 16 GB, and the per-layer embedding table dominates the working set. |
| `NFKMLXGemmaLanguage` (Gemma 4) | offline | — | internal | The per-layer input embedding alone is 262144 by 8960 on the E-series. |
| `NFKMLXGemma4UnifiedNet` | offline | — | internal | One released size, the 12B decoder. |
| `NFKMLXGemma4Fusion`, `NFKMLXGemma4ConditionalGeneration` | offline | — | partial | Their only objective runs through the E-series decoder, which is offline at every released size. |
| `NFKMLXGemmaBackend`, `NFKMLXGemma4ImageProcessor`, `NFKMLXGemma4AudioFeatureExtractor` | untrainable | — | n/a | No parameters. |

## Vision language

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXFlorence2` | ships | LoRA | public | Microsoft publishes no script, so the objective is the release's own `labels=` loss. |
| `NFKMLXTrOCR` | ships | full | public | `microsoft/unilm/trocr`'s fairseq recipe, every weight trained, matched within 3e-7. |
| `NFKMLXSmolVLM` | trainable | LoRA | public | 256M, 500M, and 2.2B are all under the line, over the shared decoder LoRA already adapts. |
| `NFKMLXQwen3VL` | trainable at 2B and 4B; offline at 8B, 32B, 30B-A3B | LoRA | public | The retrieval recipe is already a public on-device path over the 2B decoder. |
| `NFKMLXPixtral` | offline | — | public | The only release is the 12B Mistral-Nemo decoder, ~24 GB resident for inference alone. |
| `NFKMLXPhi4MM` | offline | — | public | 5.6B, and its own fine-tune scripts train in bfloat16 on large-memory GPUs. |
| `NFKMLXSa2VA` | ships at 1B, 2B, 4B; offline at 7B and above | LoRA | public | bytedance/Sa2VA's own recipe: LoRA rank 128 on the language model plus the mask decoder, objective terms within 2e-7 and the mmengine schedule exact. 1B runs at float32; 4B float32 is ~16 GB of weights before optimizer state. The Qwen-VL and LLaVA overloads round-trip on Sa2VA-Qwen3-VL-2B (untied head, ~22 GB) and the LLaVA-1.5-7B cut. |

## Embeddings and retrieval

A contrastive model splits the same way every time: a probe or adapter over the frozen embedding is
trainable, and the full fine-tune is offline because the objective needs a large batch of in-batch
negatives.

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXQwen3VLEmbedder` / `NFKMLXQwen3VLReranker` | ships | probe | public | Both objectives at measured parity. The 2B full fine-tune is already named offline. |
| `NFKMLXLaya` | ships | head-retarget | public | `rl_common.proper_reward` at measured parity, with the episode-prefix path. |
| `NFKMLXOpenJevDeBERTa` | ships | head-retarget | public | The release's `decision_loss` with its two AdamW groups. |
| `NFKMLXOpenJev` | ships | LoRA | public | The loader's own loss, saved in the release's PEFT layout. Needs the base at float32: 2B fits on a device; 9B (about 36 GB) and 27B (about 108 GB) do not. |
| `NFKMLXCLIP` | ships | probe | public | The probe ships in `NFKMLXCLIPProbe.swift`. |
| `NFKMLXSigLIP2` | ships | probe | public | The shared `NFKMLXEmbeddingProbe` over the frozen pooled image embedding; a saved probe reloads through `probeBackendWithProbeURL:labels:error:`. |
| `NFKMLXTextEmbedder` | ships | probe | public | `NFKMLXTextEmbeddingBackend`'s identity-initialized adapter under `MultipleNegativesRankingLoss`, the objective measured for Qwen3-VL. |
| `NFKMLXEmbeddingGemma` | ships | probe | public | The same backend adapter over the Dense-projected embedding. The released Dense head stays untouched. |
| `NFKMLXModernBERTReranker` | trainable | head-retarget | internal | The mean-pool head and single-logit classifier over a frozen encoder, the same pair objective the Qwen3-VL reranker ports. |
| `NFKMLXChronos` | trainable | full | public | The entry already names a pinball-loss fine-tune as implementable. Only the objective, the data adapter, and the recipe are missing. |

## Speech recognition

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXWhisper` | ships | LoRA | public | LoRA on the decoder's attention with the encoder frozen. |
| `NFKMLXGraniteSpeech` | trainable | LoRA | public | transformers computes a `labels=` loss, and the 2B decoder is under the 4B line. |
| `NFKMLXVoxtral` | trainable | LoRA | public | The 3B decoder with the Whisper encoder frozen is the shipped Whisper recipe at one more billion parameters. |
| `NFKMLXCanary` | trainable | LoRA | public | NeMo publishes `transf_loss` with a prompt loss mask, and a 1B encoder-decoder is within budget. |
| `NFKMLXParakeet` | trainable | head-retarget | public | NeMo's `TDTLossPytorch` is pure PyTorch and portable. The prediction network and joint retrain over a frozen FastConformer. |
| `NFKMLXVAD` | ships | full | public | The release's own recipe: NeMo's masked cross-entropy, SGD, and `PolynomialHoldDecayAnnealing`, matched by `run_reference.py vad_training`. |
| `NFKMLXSileroVAD` | trainable | head-retarget | internal | snakers4 publishes `tuning/tune.py`, which freezes the transform and encoder and trains the decoder alone. That is exactly this port's split. |
| `NFKMLXAudioTagger` | trainable | head-retarget | internal | PANNs publishes `finetune_template.py` and a clip-level binary cross-entropy. |

## Translation

Every translator ships the same recipe: LoRA on the decoder's query and value projections with the
encoder frozen, over the reference's teacher-forced `labels=` loss.

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXMarian` | ships | LoRA | public | Measured to 3e-5. |
| `NFKMLXM2M100` | ships | LoRA | public | The round trip runs through `network(directoryURL:)` after the merge. |
| `NFKMLXMADLAD` | ships | LoRA | public | On the T5 decoder's projections. The 3B release is 11.8 GB at float32. |
| `NFKMLXTranslateGemma` | ships | LoRA | public | With the prompt positions masked, the reference's `labels=-100` rule. 12B and 27B stay offline. |

## Music transcription and structure

| Model | Outcome | Level | Reach | What decides it |
| --- | --- | --- | --- | --- |
| `NFKMLXBasicPitch` | trainable | full | public | The installed distribution ships `train.py` and three loss functions over a 35,736-parameter network. |
| `NFKMLXAllInOne` | ships | full | public | The authors' `compute_losses`, targets, and timm RAdam, each matched by `run_reference.py allin1_training`. A consumer's own annotated tracks replace the Harmonix audio. |
| `NFKMLXHFTTransformer` | trainable | full | public | sony/hFT-Transformer releases its training code. The blocker is the MAESTRO preparation pipeline, not the objective or the device. |
| `NFKMLXMuScriptor` | untrainable | — | public | The package holds no loss, optimizer, or backward pass, the three-stage pipeline is unreleased, and the weights are CC BY-NC 4.0. |

## Uncertain rows, collected

Fifteen rows turn on a file nobody here has read. Each is cheap to settle and blocks any decision
about whether there is code to write.

| Model | Read this |
| --- | --- |
| `NFKMLXDepthAnything` | `DepthAnything/Depth-Anything-V2`, for any training script. |
| `NFKMLXDepthAnything3` | The `depth_anything_3` GitHub repository, not the 0.1.1 wheel. |
| `NFKMLXRetinaFace` | `biubug6/Pytorch_Retinaface`, `layers/modules/multibox_loss.py`. |
| `NFKMLXColorizer` | richzhang/colorization's training directory and `prior_probs.npy`. |
| `NFKMLXDDColor` | piddnad/DDColor, `options/train/*.yml`. |
| `NFKMLXRIFEv4` | hzwer/Practical-RIFE, for a v4 training script. |
| `NFKMLXMossFormer2SENet`, `NFKMLXFRCRN` | The `train/` tree of modelscope/ClearerVoice-Studio. |
| `NFKMLXVoiceRestore` | skirdey/voicerestore, for a training script. |
| `NFKMLXDenoiser` | facebookresearch/denoiser, `solver.py` and `stft_loss.py`. |
| `NFKMLXSNAC` | hubertsiuzdak/snac, for a training script. |
| `NFKMLXMarigold`, `NFKMLXSDUpscaler` | prs-eth/Marigold's training script and the upscaler's noise-level schedule. |
| `NFKMLXTAESD` | madebyollin/taesd, for a training script and whether it carries a discriminator. |
| `NFKMLXLTXVideoVAE` | Lightricks/LTX-Video, for autoencoder training code. |
| `NFKMLXQwenImageVAE` | The Qwen-Image release or diffusers, for autoencoder training code. |

## Rows ruled on architecture, with the reference still unread

These are `trainable` on the architecture and the entry's own prose. The level is settled and the
reference's optimizer, rate, and loss weights are not, which item 3 and item 5 of the minimum shipped
set both need: `NFKMLXU2Net`, `NFKMLXBiRefNet`, `NFKMLXRVM`, `NFKMLXMODNet`, `NFKMLXBiSeNet`,
`NFKMLXBiSeNetV2`, `NFKMLXDeepLab`, `NFKMLXPose`, `NFKMLXVitPose`, `NFKMLXNAFNet`.

## Corrections this triage found

Ten entries stated something the code or the reference contradicts. Each was corrected in its class
file on 2026-09-24, and the rows above now agree with the entries.

- `NFKMLXBasicPitch` said the reference publishes no training code. The installed distribution holds
  `train.py` and three loss functions, so the entry now rules it trainable at `full`.
- `NFKMLXSegFormer`, `NFKMLXZeroDCE`, `NFKMLXWhisper`, and `NFKMLXCLIP` each ship a recipe their entry
  never mentioned. Each entry now names its builder, objective, reference optimizer, and round-trip
  test.
- `NFKMLXGraniteSpeech`, `NFKMLXVoxtral`, and `NFKMLXCanary` were ruled out because a labeled
  audio-and-transcript pipeline was beyond the reachable trainer. The shipped Whisper recipe takes
  exactly that input, so each entry now rules its model trainable at LoRA.
- `mlx-models-audio-codecs-music.md` called the BigVGAN and Mimi discriminator constraint
  "untrainable here." Both entries now say `offline`.

Twenty-one of the twenty-four entries in image restoration, video, and text to speech carry no
ruling at all, which is the failure the rule targets.

## Where the work is

Ordered by what a session gets per unit of effort, and grounded in what the triage read.

1. **The small full fine-tunes.** GTCRN, NU-Wave 2, All-In-One, Conv-TasNet, and MarbleNet ship.
   Basic Pitch at 35,736 parameters needs TensorFlow in its oracle environment
   first, because its losses are Keras functions, and its released graph folds the batch
   normalizations the reference trains. Each has a published objective with no adversary, and
   each fits a device with room to spare.
2. **The head retargets whose loss is already published and portable.** YOLO ships, every
   generation, and RT-DETR ships, every release of both versions. RF-DETR and its segmentation head,
   Silero VAD (whose reference tuning script freezes exactly what this port freezes), the PANNs
   tagger, and the segmenters remain.
3. **Reachability for the language decoders.** The dense Qwen, hybrid, and Gemma 3 decoders are
   LoRA-feasible at 4B and under and have no public builder. That is a visibility change plus a
   recipe, and it is the largest single piece of demand.
4. **Parakeet's transducer loss.** The only genuinely new numerical work in the whole ledger.
   `TDTLossPytorch` is pure PyTorch and portable, so it is a forward-probability recursion to write
   rather than a blocked path.
