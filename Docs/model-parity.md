# Implemented models and their reference parity

Every model InferKitMLX ships is measured against its reference implementation on the released weights.
The classes, configuration presets, and factories behind each row are in [model-index.md](model-index.md).
This document lists each model, the reference it is held to, and the number the measurement produced on
the run recorded below. The subsystems the models share are itemized at the end, with the models that
depend on each.

## How the numbers are produced

- **Oracle.** `Tools/reference-parity/run_reference.py <mode>` runs the model's own reference
  implementation (torch / transformers / diffusers / the authors' repository) on a fixed input and writes
  the input and every intermediate it exposes to a safetensors record. Five Python environments are
  needed, because the references pin incompatible versions; the manifest's `oracle_environments`
  records each one.
- **Measurement.** The Swift port runs the same input and compares. `NFKMLXReferenceParityTests`
  prints one `VALIDATION PARITY` line per measurement; the numbers below are copied from those lines.
  A cosine is over the flattened tensor. "Exact" means an integer result (tokens, labels, codes,
  argmax) matched element for element.
- **Assets.** `Tools/validation-assets/fetch.py` downloads every checkpoint and record into
  `~/.inferkit-validation` and writes their paths to `~/.inferkit-validation.json`. A test whose asset is
  absent skips rather than fails, so a green run on a machine without the assets proves nothing; the run
  below had every asset present except the one listed under "Skipped".

### The run this document records

| Suite | Command | Result |
| --- | --- | --- |
| Core | `swift test` | 337 tests, 8 skipped (live-network gates), 0 failures |
| InferKitFoundationModels | `swift test` | 24 tests, 0 failures |
| InferKitMLX | `xcodebuild test -scheme InferKitMLXTests -destination 'platform=macOS' -skipPackagePluginValidation` | 931 tests, 8 skipped (listed below), 741 s; 2 failures on the first pass, both test defects, corrected and rerun green |
| InferKitMLX examples | the `InferKitMLXExamples` and `InferKitMLXObjCExamples` schemes | 59 and 35 tests, 0 failures |

Measured on 2026-09-04 on an M1 Max with 32 GB, at float32 unless a row says otherwise. Two tests
failed on the first pass and were corrected before the reruns recorded above: a Gemma configuration
guard that still expected the 26B-A4B mixture to be refused (it is implemented, so the test now asserts
it is read), and an RVM fine-tuning test whose unseeded random initialization diverged under an
unclipped step (it now seeds and clips).

## Image → image

| Model | Class | Reference | Weights | Measured |
| --- | --- | --- | --- | --- |
| Real-ESRGAN ×4 | `NFKMLXRealESRGAN` | BasicSR `RRDBNet` | RealESRGAN_x4plus | cosine 0.9999947, mean abs 0.00146 |
| Real-ESRGAN anime | `NFKMLXRealESRGAN` (`.anime`) | BasicSR `RRDBNet`, 6 blocks | RealESRGAN_x4plus_anime_6B | cosine 0.9999956, mean abs 0.00133 |
| SwinIR classical ×4 | `NFKMLXSwinIR` | JingyunLiang `network_swinir.py` | 001_classicalSR_DIV2K_s48w8 x4 | cosine 0.99986, mean abs 0.0037 |
| SwinIR classical ×3 | `NFKMLXSwinIR` (`.x3`) | same | x3 (one ×3 shuffle stage) | cosine 0.99987, mean abs 0.0036 |
| SwinIR classical ×8 | `NFKMLXSwinIR` (`.x8`) | same | x8 (three ×2 stages) | cosine 0.99990, mean abs 0.0035 |
| SwinIR lightweight ×2 | `NFKMLXSwinIR` (`.lightX2`) | same, `pixelshuffledirect` | 002_lightweightSR x2 | cosine 0.99991, mean abs 0.00097 |
| NAFNet SIDD (denoise) | `NFKMLXNAFNet` | megvii-research NAFNet | NAFNet-SIDD-width32 | cosine 0.9999973, mean abs 0.00098 |
| NAFNet GoPro (deblur) | `NFKMLXNAFNet` (`.goPro`) | same | NAFNet-GoPro-width32 | cosine 0.9999973, mean abs 0.00098 |
| NAFNet REDS | `NFKMLXNAFNet` (`.reds`) | same | NAFNet-REDS-width64 | cosine 0.9999971, mean abs 0.00098 |
| Zero-DCE | `NFKMLXZeroDCE` | Li-Chongyi `model.py` | Epoch99 | cosine 0.9999971, mean abs 0.00134 |
| Fast style transfer | `NFKMLXStyleTransfer` | pytorch/examples `TransformerNet` | mosaic | **skipped** — the checkpoint is no longer served (last measured 0.9999926); a seeded stand-in measures only noise and was rejected |
| Colorizer ECCV-16 | `NFKMLXColorizer` | richzhang `eccv16.py` | colorization_release_v2 | ab cosine 0.9999999998, sRGB cosine 0.9999971 |
| Colorizer SIGGRAPH-17 | `NFKMLXSiggraphColorizer` | richzhang `siggraph17.py` | siggraph17 | ab cosine 0.99999999999956, sRGB cosine 0.99999999987 |
| LaMa | `NFKMLXLaMa` | advimman `FFCResNetGenerator` | big-lama | cosine 0.99999999999972, mean abs 5.9e-8 |
| CodeFormer | `NFKMLXCodeFormer` | sczhou CodeFormer (w 0.5, AdaIN on) | codeformer.pth | code logits 0.99999999999855, code agreement 1.0, restored face 0.99999999999868; public backend path 0.9999968 (8-bit image bridge) |
| TAESD | `NFKMLXTAESD` | madebyollin `taesd.py` | taesd encoder + decoder | latent 0.99999999999959, decode 0.99999999999978, mean abs 1.9e-7 |

The three mean-absolute figures near 0.001 (Real-ESRGAN, NAFNet, Zero-DCE, the RIFE and CodeFormer
backend paths) are the 8-bit quantization of the `CGImage` bridge, not the network: the same networks
measured on float tensors sit at 1e-7.

## Image → map (depth and segmentation)

| Model | Class | Reference | Weights | Measured |
| --- | --- | --- | --- | --- |
| Depth Anything V2 Small | `NFKMLXDepthAnything` | authors' `depth_anything_v2` package | vits | depth cosine 0.99817, mean abs 0.0286; encoder seam 0.9999924 over 525,696 values |
| Depth Anything V2 Base | `NFKMLXDepthAnything` (`.base`) | same | vitb | cosine 0.99807, mean abs 0.0249 |
| Depth Anything V2 Large | `NFKMLXDepthAnything` (`.large`) | same | vitl | cosine 0.99846, mean abs 0.0263 |
| Marigold depth | `NFKMLXMarigold` | diffusers `UNet2DConditionModel` | marigold-depth-v1-0 UNet | predicted noise cosine 0.99999999999043 |
| SegFormer B0 | `NFKMLXSegFormer` | transformers `SegformerForSemanticSegmentation` | nvidia/segformer-b0-finetuned-ade-512-512 | logit cosine 0.99999992, label agreement 0.99994 |
| DeepLabV3 | `NFKMLXDeepLab` | torchvision `deeplabv3_resnet50` | COCO release | logit cosine 0.99999999999976, label agreement 1.0 |
| BiSeNet V1 | `NFKMLXBiSeNet` | CoinCheung BiSeNetV1 | model_final_v1_city | logit cosine 0.99999999999886, label agreement 1.0 |
| BiSeNet V2 | `NFKMLXBiSeNetV2` | CoinCheung BiSeNetV2 (older pixel-shuffle head) | model_final_v2_city | logit cosine 0.99999999999922, label agreement 1.0 |

The Depth Anything map cosines near 0.998 are the recorded values for this family; its encoder seam at
0.9999924 shows the transformer is exact and the residual is the DPT head's bilinear resampling.

## Matting, segmentation, and faces

| Model | Class | Reference | Weights | Measured |
| --- | --- | --- | --- | --- |
| U²-Net | `NFKMLXU2Net` | xuebinqin `u2net.py` | u2net | cosine 0.99918, mean abs 0.00143 |
| U²-Net light | `NFKMLXU2Net` (`.light`) | `U2NETP` (a separate class) | u2netp | cosine 0.99981 |
| Robust Video Matting | `NFKMLXRVM` | PeterL1n `MattingNetwork` | rvm_mobilenetv3 | alpha 0.99999999999998, foreground 0.99999999999967, guided-filter refine at 0.5: 0.99999999999985; odd frame size 0.99999999999991 |
| MODNet | `NFKMLXMODNet` | ZHKKKe MODNet | photographic portrait | alpha cosine 0.99999999999939, mean abs 1.4e-8 |
| SAM ViT-B | `NFKMLXSAM` | facebookresearch `segment-anything` | sam_vit_b | encoder 0.9999986 over 1,048,576 values; selected mask 0.99995, binary agreement 0.99985 |
| SAM 2 (Hiera tiny) | `NFKMLXSAM2` | facebookresearch `sam2` sources (vendored) | sam2_hiera_tiny | FPN level 0 0.99999999999862, level 1 0.99999999999178, vision features 0.99999999999393 |
| SAM 2 (Hiera base_plus) | `NFKMLXSAM2` | same | sam2_hiera_base_plus | level 0 0.99999999999584, level 1 0.99999999998953 |
| SAM 2 (Hiera large) | `NFKMLXSAM2` | same | sam2_hiera_large | level 0 0.99999999999936, level 1 0.99999999999859 |
| SAM 2 prompt encoder + mask decoder | `NFKMLXSAM2` | same | tiny | sparse prompt 0.99999999999998, mask logits 0.99999999999498, object score 23.288193 vs 23.288197 |
| SAM 2 memory encoder + memory attention | `NFKMLXSAM2` | same | tiny | encoded memory 0.99999999999987, attention output 0.99999999999973 |
| RetinaFace mobile0.25 | `NFKMLXRetinaFace` | facexlib RetinaFace | detection_mobilenet0.25_Final | box 0.99999999999798, score 0.9999999999999969, landmark 0.99999999999804; end to end 1/1 face, box IoU 1.0 |

The two face detectors the CodeFormer photograph path can use disagree measurably (RetinaFace against
Vision on the validation portrait: box IoU 0.65, worst landmark 15.7 px over 960×1200), which is why
RetinaFace is the default and the choice is recorded rather than hidden. Feeding RetinaFace RGB instead
of the reference's BGR keeps the confidence (0.9971 vs 0.9975) while moving the box to IoU 0.96 and the
landmarks by up to 5.7 px; that is measured so the order stays pinned.

## Detection and pose

| Model | Class | Reference | Weights | Measured |
| --- | --- | --- | --- | --- |
| YOLOv8n | `NFKMLXYOLO` | ultralytics YOLOv8 | yolov8n | box 0.99999999999996, class 0.99999999999447; end to end on a 16:9 frame 9/9 detections, worst box IoU 0.9999984 |
| YOLOv8s | `NFKMLXYOLO` (`.small`) | same | yolov8s | box 0.99999999999997, class 0.99999999999966 |
| YOLOv8m | `NFKMLXYOLO` (`.medium`) | same | yolov8m | box 0.99999999999996, class 0.99999999999849 |
| YOLOv8l | `NFKMLXYOLO` (`.large`) | same | yolov8l | box 0.99999999999993, class 0.99999999999985 |
| YOLOv8x | `NFKMLXYOLO` (`.extraLarge`) | same | yolov8x | box 0.99999999999994, class 0.99999999999975 |
| RT-DETR (tiny config) | `NFKMLXRTDetr` | transformers `RTDetrForObjectDetection` | random tiny | backbone and encoder seams ≥ 0.99999999999999; query-selection scores 0.9999999999999925, boxes 0.9999999999999999; decoder over the reference's selection: logits 0.9999999999999958, boxes 0.9999999999999966; over its own selection 0.9999998 / 0.981 (a sub-ulp top-k tie swaps one query) |
| RT-DETR r50vd | `NFKMLXRTDetr` (`.r50vd`) | same | PekingU/rtdetr_r50vd | logits 0.99999999998875, boxes 0.99999999996419 |
| SimpleBaseline pose | `NFKMLXPose` | microsoft `pose_resnet.py` | mmpose ResNet-50 COCO | heatmap cosine 0.99999999999495, peak agreement 1.0 |

## Embeddings, reranking, and vision-language

| Model | Class | Reference | Weights | Measured |
| --- | --- | --- | --- | --- |
| CLIP ViT-B/32 image tower | `NFKMLXCLIP` | transformers `CLIPModel` | OpenAI ViT-B/32 | image cosine 0.9999965 |
| CLIP ViT-B/32 text tower | `NFKMLXCLIP` | same | same | text cosine 0.99999999999876 |
| SigLIP 2 base-patch16-224 | `NFKMLXSigLIP2` | transformers SigLIP 2 | google/siglip2-base-patch16-224 | image 0.99999999999892, worst text 0.99999999999861, max logit diff 9.5e-6 |
| Qwen3-Embedding-0.6B | `NFKMLXQwen3Embedding` | model-card transformers recipe | released 0.6B | query 0.99999999999492, document 0.99999999999162, retrieval score 0.7645564 vs 0.7645574 |
| EmbeddingGemma-300M | `NFKMLXEmbeddingGemma` | sentence-transformers over `Gemma3TextModel` | unsloth mirror | every layer exact; query 0.99999999999968, document 0.99999999999964, retrieval score 0.6092325 vs 0.6092324 |
| gte-reranker-modernbert-base | `NFKMLXModernBERTReranker` | transformers `ModernBertForSequenceClassification` | released | every layer exact; relevant 2.968215 vs 2.968218, irrelevant −2.403371 vs −2.403368 |
| SmolVLM2-500M | `NFKMLXSmolVLM` | transformers `SmolVLMForConditionalGeneration` | SmolVLM2-500M-Video-Instruct | SigLIP embeddings 0.9999999999992, layer 0 0.9999999999983, vision 0.99999999997, connector 0.99999999999; fused decoder argmax 1140/1140, last-position logit cosine 0.99999999992; greedy continuation token for token |
| Qwen3-VL-2B vision tower | `NFKMLXQwen3VL` | transformers Qwen3-VL vision model | Qwen3-VL-2B-Instruct | patch embed 0.99999999999991, position embed 0.99999999999998, merged output 0.99999999968, deepstack 0/1/2 0.99999999999 / 0.99999999994 / 0.99999999991 |
| Gemma 4 vision tower (tiny) | `NFKMLXGemma4VisionNet` | transformers Gemma 4 vision model | random tiny | encoder 1.0000000000000002, pooled 0.9999999999999997 |
| Gemma 4 vision tower (released) | `NFKMLXGemma4VisionNet` | same | E2B checkpoint | encoder 0.99999999998979, pooled 0.99999999999751, projected 0.99999999999480; the rope and the clamps are load-bearing (0.768 and 0.837 without them) |
| Gemma 4 audio tower | `NFKMLXGemma4AudioNet` | transformers `Gemma4AudioModel` | tiny; E2B | subsampler 0.9999999999999845; Conformer 0.9999999999999974 (attention seam 0.9999999999999688); full tower from mel 0.999999999999996; released: encoded 0.99999999999946, projected 0.99999999999939 |
| Gemma 4 mel front end | `NFKMLXGemma4AudioFeatureExtractor` | transformers `Gemma4AudioFeatureExtractor` | — | cosine 0.99999999999279 |
| Gemma 4 multimodal embedder | `NFKMLXGemma4MultimodalEmbedder` | transformers `Gemma4MultimodalEmbedder` | tiny | cosine 0.9999999999999927 |
| Gemma 4 conditional generation (image + text) | `NFKMLXGemma4ConditionalGeneration` | transformers `Gemma4ForConditionalGeneration` | E2B | logit cosine 0.99999999995857, argmax 8/8 |

## Language models

| Model | Class | Reference | Weights | Measured |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | `NFKMLXLanguage` | transformers `Qwen3ForCausalLM` | released | prefill logit cosine 0.99999999999433 (worst abs 8.9e-5), same argmax at every position; greedy continuation of 16 tokens exact |
| Qwen3-1.7B | `NFKMLXLanguage` | same | released | logit cosine 0.99999999999751 |
| Qwen3-4B | `NFKMLXLanguage` | same | released (16 GB at float32) | logit cosine 0.99999999998684 |
| Qwen3-MoE (tiny) | `NFKMLXLanguage` | transformers `Qwen3MoeForCausalLM` | random tiny | every hidden state exact layer by layer, logit cosine 0.99999999999999 |
| Mixtral (tiny) | `NFKMLXLanguage` | transformers `MixtralForCausalLM` | random tiny | every layer exact, logit cosine 0.99999999999999 |
| Qwen3-30B-A3B | `NFKMLXLanguage` | checkpoint headers | released (shapes only) | structural: 18,867 tensors consumed, 0 missing, 0 mismatched, 0 unaccounted |
| Qwen3.5-4B (hybrid) | `NFKMLXHybridLanguage` | transformers `Qwen3_5ForConditionalGeneration` | released | every one of 33 hidden states exact; logit cosine 0.99999999999620 |
| Qwen3.8-27B (hybrid) | `NFKMLXHybridLanguage` | checkpoint headers | released (shapes only) | structural: 851 decoder parameters, 0 missing, 0 mismatched; 333 vision + 15 multi-token-head tensors named as out of scope |
| Gemma 4 E2B | `NFKMLXGemmaLanguage` | transformers Gemma 4 | released | every one of 36 hidden states exact; logit cosine 0.99999999999935; 600 parameters structurally matched; backend generates " Paris." for "The capital of France is" |
| Gemma 4 E4B | `NFKMLXGemmaLanguage` | same | released, **bf16 both sides** | logit cosine 0.99983, argmax 5/5 |
| Gemma 4 26B-A4B mixture (tiny) | `NFKMLXGemmaLanguage` | transformers `Gemma4ForCausalLM` | random tiny | every layer exact, logit cosine 0.99999999999962 |
| Gemma 4 12B unified (tiny) | `NFKMLXGemma4UnifiedNet` | transformers `Gemma4UnifiedForCausalLM` | random tiny | every layer exact, logit cosine 0.99999999999952 |
| Gemma 2 | `NFKMLXGemma2Net` | transformers `Gemma2Model` | random tiny, small sliding window | last hidden cosine 0.99999999999987 |
| DeepSeek V4 (tiny, all-sliding) | `NFKMLXDeepSeek` | transformers DeepSeek V4 | random tiny | every layer ≥ 0.9999995, logit cosine 0.99999999991132 |
| DeepSeek V4 Flash | `NFKMLXDeepSeek` | checkpoint headers | released (shapes only) | 3,975 parameters across 5 captured layers, 0 mismatched; 34,223 declared + 33,389 block scales + 4,705 named (MTP / DSpark), 0 unaccounted of 72,317 |
| DeepSeek V4 Pro (0813) | `NFKMLXDeepSeek` | checkpoint headers | released (shapes only) | 3,540 parameters across 3 captured layers, 0 mismatched; 71,983 declared + 70,790 block scales + 7,009 named, 0 unaccounted |
| DeepSeek fp8 / fp4 storage | `NFKMLXDeepSeekQuantization` | torch 2.13 `float8_e4m3fn` + `ml_dtypes` `float4_e2m1fn` | real checkpoint bytes | both decodes exact (assertion) |
| SmolLM2-135M Q4_K_M (GGUF) | `NFKMLXLanguage` (`ggufURL:`) | transformers loading the same GGUF | unsloth Q4_K_M | dequantization bit-exact for F32 / Q8_0 / Q5_0 / Q4_K / Q6_K (worst diff 0.0 over 262,144 values each); logit cosine 0.99999999999708, same argmax at every prompt position, teacher-forced continuation 8/10 (two near-ties) |
| T5-XXL v1.1 encoder | `NFKMLXT5Encoder` | transformers `T5EncoderModel` | LTX-Video text_encoder (19 GB) | embed seam exact, block 0 0.99999999999963, text embedding 0.99999999997956 |
| umT5 encoder (tiny) | `NFKMLXT5Encoder` (`perLayerBias`) | transformers `UMT5EncoderModel` | random tiny | text embedding 0.9999999999999984 |
| RoPE scaling (`linear`, `yarn`) | `NFKMLXRoPEScaling` | transformers `ROPE_INIT_FUNCTIONS` | five configurations | frequencies within 1e-5 relative (assertion) |
| Chat templates | `NFKMLXChatTemplateRenderer` | transformers `apply_chat_template` | Qwen3, Llama-3, Gemma templates | six cases byte for byte (assertion) |
| Speculative decoding | `NFKMLXLanguage` | its own greedy run | Qwen3-1.7B ← 0.6B draft | token-identical; acceptance 0.735, 0.72× wall clock (launch-bound at float32) |
| Constrained decoding | `NFKMLXJSONConstraint` | — | Qwen3-0.6B | a valid JSON object under the grammar (assertion) |

## Video

| Model | Class | Reference | Weights | Measured |
| --- | --- | --- | --- | --- |
| RIFE HDv3 | `NFKMLXRIFE` | the release's own `IFNet_HDv3.py` | flownet.pkl | interpolated cosine 0.99999999999933, mean abs 2.7e-7; needs-padding frame 0.99999999999944; backend path 0.9999974 |
| RIFE v4.13 | `NFKMLXRIFEv4` | ComfyUI-Frame-Interpolation `rife_arch.py` (4.17) | rife-flownet-4.13.2 | interpolated cosine 0.99999999999947, mean abs 2.3e-7 |
| RAFT | `NFKMLXRAFT` | princeton-vl RAFT | raft-things | eighth-resolution flow 0.99999999999975 (mean abs 3.3e-7), full resolution 0.99999999999993 |
| BasicVSR | `NFKMLXVideoSR` | mmediting `BasicVSRNet` | REDS4 | three-frame clip cosine 0.99999999999979, mean abs 1.1e-7 |
| SD ×4 upscaler UNet | `NFKMLXSDUpscaler` | diffusers `UNet2DConditionModel` | stable-diffusion-x4-upscaler | predicted noise 0.99999999999730 |
| SD ×4 upscaler VAE | `NFKMLXSDUpscaler` | diffusers `AutoencoderKL` | same | latent 0.99999999998049, decoded 0.99999999991751 |

## Audio

| Model | Class | Reference | Weights | Measured |
| --- | --- | --- | --- | --- |
| Whisper tiny | `NFKMLXWhisper` | openai-whisper | tiny.pt | log-mel 0.99999999999401, first-step logits 0.99999999974, greedy tokens exact; the reference's suppression rules exact; timestamped decode exact with segment (0.0, 3.0) s |
| Whisper small / medium / large-v3 | `NFKMLXWhisper` (variants) | same | released | mel 0.99999999999401 / 0.99999999999401 / 0.99999999998511; tokens exact for all three |
| Demucs v2 | `NFKMLXDemucs` | facebookresearch demucs v2 | demucs-e07c671f | per-stem-channel cosine ≥ 0.99999999949 (worst), most ≥ 0.9999999998 |
| Hybrid Transformer Demucs (v4) | `NFKMLXHTDemucs` | `demucs` 4.0.1 `load_model` | htdemucs 955717e8 | spectrogram 0.99999999999995, bottleneck in/out 0.99999999996 / 0.99999999999, freq-out 0.99999999999601, time-out 0.99999999999975; separated stems 0.99999999999955, mean abs 7.7e-8 |
| Conv-TasNet | `NFKMLXConvTasNet` | asteroid `ConvTasNet` | Libri2Mix sep_clean 16k | per-speaker cosine 0.99999999949 / 0.99999999999975 |
| Speech denoiser (dns48) | `NFKMLXDenoiser` | facebookresearch/denoiser | dns48 | cosine 0.99999999999992; on real speech the correlation with the clean signal rises 0.9657 → 0.9970 |
| MarbleNet VAD | `NFKMLXVAD` | NeMo | Frame_VAD_Multilingual_MarbleNet_v2.0 | mel 0.99999999999974, probabilities 0.99999999999983 (largest abs 3.0e-7) |
| Silero VAD v6 | `NFKMLXSileroVAD` | snakers4 `silero_vad` 6.2.1 JIT | silero_vad.jit (16 kHz) | per-chunk cosine 0.99999999999979, max abs 6.9e-7, threshold agreement 32/32 |
| PANNs Cnn14 tagger | `NFKMLXAudioTagger` | `audioset_tagging_cnn` | Cnn14_mAP=0.431 | mel 0.99999999, embedding 0.99999994, tags 0.99999988, same top class (513) |
| Descript Audio Codec | `NFKMLXDAC` | `descript-audio-codec` | 44.1 kHz weights.pth | codes 783/783 exact, reconstruction 0.99999999999986 |
| SNAC 24 kHz | `NFKMLXSNAC` | `snac` package | snac_24khz | codes 42/42 exact, reconstruction 0.99999999999982 (decoder noise off) |
| HiFi-GAN universal | `NFKMLXHiFiGAN` | jik876 `models.py` | UNIVERSAL_V1 g_02500000 | cosine 0.99999999993412 |
| FastSpeech2 conformer | `NFKMLXFastSpeech2` | transformers `FastSpeech2ConformerModel` | espnet LJSpeech | encoder 0.99999999999983, durations exact ([3, 9, 15, 11, 9, 9, 11, 8, 14, 2]), pitch 0.99999999999884, energy 0.99999999999996, mel 0.99999999999924 |
| FastSpeech2 + paired HiFi-GAN voice | `NFKMLXVoice` | loop closure through Whisper | espnet `fastspeech2_conformer_with_hifigan` | "hello world" is transcribed by the package's own Whisper as " hello, world." |
| Kokoro-82M | `NFKMLXKokoro` | vendored `KModel` + `istftnet` (spaCy-free) | kokoro-v1_0.pth | PL-BERT 0.99999999999575, duration encoder 0.99999999999368, durations 0.99999999999998, F0 0.99999999999996, N 0.99999999999996, text encoder 0.99999999999994, asr 0.99999999999992, decoder encode 0.99999999999948, generator input 0.99999999999910; harmonic source 0.9999887 and conv_post 0.9999810 (float32 sine-phase limit); waveform 0.99559, full phoneme path 0.99696 |

## Text → music (MiniMax Music 3)

| Stage | Class | Reference | Measured |
| --- | --- | --- | --- |
| Vocoder | `NFKMusic3VocoderNet` | diffusers `MiniMaxMusic3Vocoder` | cosine 0.99999999999903, worst abs 8.3e-7 |
| RVQ depth decoder | `NFKMusic3DepthDecoderNet` | diffusers `MiniMaxMusic3RVQDepthDecoder` | hidden 0.99999999999674, heads 0.99999999999611, projection 0.99999999999948, embedding 1.0 |
| Condition encoder | `NFKMusic3ConditionEncoderNet` | diffusers `MiniMaxMusic3ConditionEncoder` | cosine 0.99999999999969 |
| Flow-matching DiT | `NFKMusic3DiTNet` | diffusers transformer | velocity at t0 0.99999999999893, mid 0.99999999999420, late 0.99999999999749, unconditional 0.99999999999410 |
| Autoregressive stage (bf16 both sides) | `NFKMusic3AutoregressiveStage` | the pipeline's own helpers, teacher-forced | prefill 0.99993, first logits 0.9999948, top-50 sets 51/52 shared with the same argmax, guided band 0.99998, frame hiddens 0.99994 |
| Prompt contract | `NFKMusic3Prompt` | the reference tokenizer and cleaners | token-exact over every case, both CFG rows (assertion) |
| Quantized release (4-bit LM incl. embedding, 8-bit DiT) | `NFKMLXMusic3.quantizeRelease` | the float32 records above | 7.74 GiB on disk; DiT velocity 0.99990, LM prefill 0.99047, LM first logits 0.99933; two consecutive 2 s generations 20.1 s and 18.3 s with the stack resident |

## Text → image and text → video

| Model | Class | Reference | Weights | Measured |
| --- | --- | --- | --- | --- |
| SD 1.5 UNet | `NFKMLXSDUNet` | diffusers `UNet2DConditionModel` | SD-1.5-inpainting UNet | predicted noise 0.99999999999925 |
| SD 1.5 VAE | `NFKMLXSDAutoencoder` | diffusers `AutoencoderKL` | SD 1.5 VAE | latent 0.99999999981842, decoded 0.99999999916914 |
| SD 2.1 UNet | `NFKMLXSDUNet` | same | SD 2.1 UNet | predicted noise 0.99999999931899 |
| SDXL UNet | `NFKMLXSDUNet` | same | sdxl-turbo UNet | predicted noise 0.99999999999555 |
| SD 1.5 text encoder | `NFKMLXSDTextEncoder` | transformers `CLIPTextModel` | SD 1.5 | last hidden 0.99999999999856, penultimate 0.99999999999962 |
| SD 2.1 text encoder | `NFKMLXSDTextEncoder` | same (OpenCLIP) | SD 2.1 | last hidden 0.99999999998408, penultimate 0.99999999999354 |
| SDXL text encoders 1 and 2 | `NFKMLXSDTextEncoder` | same | sdxl-turbo | tower 1 last hidden 0.99999999999857, penultimate 0.99999999999971; tower 2 penultimate 0.99999999989422, pooled 0.99999999998203 |
| SD prompt tokenizer | `NFKMLXSDPromptTokenizer` | transformers `CLIPTokenizer` | SD 1.5 tokenizer | token for token over five prompts (assertion) |
| DDIM scheduler | `NFKDDIMScheduler` | diffusers `DDIMScheduler` | — | worst per-step latent 0.99999999999999, add-noise 0.9999999999999983, visited schedule exact |
| SD 1.5 text-to-image | `NFKMLXTextToImage` | diffusers `StableDiffusionPipeline`, from its own initial latent | SD 1.5 | image cosine 0.99999895, mean abs 0.00064 |
| SD 2.1 text-to-image (v-prediction) | `NFKMLXTextToImage` | same | SD 2.1 | context 0.99999999998, first prediction 0.99999999659, per-step latents ≥ 0.99999997561, image 0.99999861, mean abs 0.00076 |
| SDXL-Turbo text-to-image | `NFKMLXTextToImage` | diffusers `StableDiffusionXLPipeline` (DDIM both sides) | sdxl-turbo | image 0.99999727 (one step); with guidance 0.99999898; with no negative prompt 0.99999876 |
| Z-Image S3-DiT (tiny) | `NFKMLXZImageTransformerNet` | diffusers `ZImageTransformer2DModel` | random tiny | t-embedder seam 0.99999999999511, velocity 0.9999999999999653 |
| Flux VAE (tiny, no quant conv) | `NFKMLXSDAutoencoder` (`.flux`) | diffusers `AutoencoderKL` | random tiny | latent mean 0.99999985919, decode 0.99999999481 |
| SANA linear-attention DiT (tiny) | `NFKMLXSANATransformerNet` | diffusers `SanaTransformer2DModel` | random tiny | embedded-timestep seam 0.99999999999999, velocity 0.9999999999999959 |
| DC-AE (tiny) | `NFKMLXDCAutoencoderNet` | diffusers `AutoencoderDC` | random tiny | latent 0.9999999999999344, decode 0.9999999999999848 |
| DC-AE (released) | `NFKMLXDCAutoencoderNet` | same | Sana_600M VAE (1.2 GB), 256×256 at 32× | latent 0.99999999999185, decode 0.99999999980079 |
| DPM-Solver++ (SANA) | `NFKMLXDPMSolverScheduler` | diffusers `DPMSolverMultistepScheduler` | — | sigmas within 6e-8, timesteps exact, trajectory worst abs 1.7e-6 |
| IP-Adapter | `NFKMLXIPAdapterImageProjection` / `…Attention` | diffusers `ImageProjection`, `IPAdapterAttnProcessor2_0` | random | projection 0.9999999999999964, decoupled attention 0.9999999999999997 |
| LTX-Video VAE | `NFKMLXLTXVideoVAE` | diffusers `AutoencoderKLLTXVideo` | Lightricks/LTX-Video vae | conv_in / down 0 / mid seams ≥ 0.99999999999936; latent 0.99999999999562, decode 0.99999999996192 |
| LTX-Video DiT (2B) | `NFKMLXLTXTransformer` | diffusers `LTXVideoTransformer3DModel` | released 7.7 GB shards | rope cos/sin 0.99999993 / 0.99999995, proj_in 0.99999999999999, block 0 0.99999999999864; 28-layer velocity 0.99999999999460 |
| Rectified-flow scheduler | `NFKMLXFlowMatchScheduler` | diffusers `FlowMatchEulerDiscreteScheduler` | — | sigmas to 1e-4, terminal sigma 0.1 (assertion, under `swift test`) |
| Wan DiT (tiny) | `NFKMLXWanTransformerNet` | diffusers `WanTransformer3DModel` | random tiny | velocity 0.9999999999999767 |
| Wan 2.2 VAE (tiny, residual) | `NFKMLXWanVideoVAENet` | diffusers `AutoencoderKLWan` | random tiny, 5 frames | six encoder / decoder seams ≥ 0.99999999999996; latent 0.9999999999999989, decode 0.99999999999989 |
| Wan 2.1 VAE (tiny, non-residual) | `NFKMLXWanVideoVAENet` (`.wan21`) | same | random tiny | latent 0.9999999999999978, decode 0.99999999999993 |
| UniPC (Wan) | `NFKMLXUniPCScheduler` | diffusers `UniPCMultistepScheduler` | — | sigmas within 6e-8, trajectory worst abs 1.4e-6 |

A sampled image or clip from the four DiT pipelines (`NFKMLXZImagePipeline`, `NFKMLXSANAPipeline`,
`NFKMLXLTXPipeline`, `NFKMLXWanPipeline`) is not compared bitwise; their noise streams differ from the
reference by construction. Each pipeline is validated by a weight-free glue test on matching tiny
configurations plus the per-stage parities above, the same treatment Music 3 gets.

## Training objectives and the checkpoint path

| Objective | Reference | Measured |
| --- | --- | --- |
| Zero-DCE losses (exposure, color constancy, illumination smoothness, spatial consistency) | Li-Chongyi `Myloss.py` | all four to float precision (assertion) |
| SegFormer decode-head loss | transformers `SegformerForSemanticSegmentation` | matches on identical logits (assertion) |
| Native `.pth` / `.pt` / `.ckpt` / `.th` / `.bin` / `.jit` / `.nemo` reader | the offline converters' output | tensor-for-tensor byte equality for every model with a raw checkpoint in the manifest (`NFKMLXTorchParityTests`, 24 tests) |
| Fine-tuned checkpoint reload | each model's own loader | save → reload reproduces the forward for every gated transpose (`NFKMLXCheckpointRoundTripTests`) |

## Skipped in this run

| Test | Reason |
| --- | --- |
| Fast style transfer parity and its real-weights run | the `mosaic.pth` checkpoint is no longer served anywhere; a seeded stand-in was tried and rejected because it measures noise |
| Music 3 listening clip, DiT bit-width sweep, embedding-quantization probe | opt-in probes (`IK_MUSIC3_LISTEN`, `IK_MUSIC3_DIT_SWEEP`, `IK_MUSIC3_EMB_PROBE`); the recorded sweep is in `Docs/companions.md` |
| Qwen3 tied-embedding quantization probe | opt-in (`IK_QWEN_EMB_PROBE`); recorded in `CLAUDE.md` |
| Live Hugging Face download | needs `INFERKIT_LIVE_MODEL` |
| Cross-thread evaluation probe | opt-in (`IK_PROBE_CROSS_THREAD`) |

## Shared subsystems

A subsystem verified once serves every model built on it. The table names where each is defined, which
models reuse it, and what pins it. A regression in a shared block therefore shows up in every dependent
model's parity number, which is what makes the per-model measurements above a check on the blocks too.

| Subsystem | Defined in | Reused by | Pinned by |
| --- | --- | --- | --- |
| `NFKMLXLanguageNet` (dense decoder: GQA + rotary + SwiGLU + RMSNorm) | `NFKMLXLanguageModel.swift` | Qwen3 / Qwen2 / Llama generation, Qwen3-MoE and Mixtral (the routed feed-forward swaps in), Qwen3-Embedding, the SmolVLM decoder, the Qwen3-VL decoder, the Music 3 autoregressive stage, the GGUF path | Qwen3 0.6B / 1.7B / 4B parity, the MoE and Mixtral tiny parities, `testACachedStepMatchesRecomputingThePrefix` |
| `NFKMLXKeyValueCache` / `NFKMLXPromptCache` (block-growing, windowed, quantizable, rollback) | `NFKMLXLanguageModel.swift`, `NFKMLXPromptCache.swift` | the language backend, speculative decoding, SmolVLM, Music 3 | the cache-exactness, chunked-prefill, window, prompt-cache, and speculative-equality tests |
| `NFKMLXRoPEScaling` (`linear`, `yarn`) | `NFKMLXRoPEScaling.swift` | the dense decoder, DeepSeek V4 Pro | `testTheScaledFrequenciesMatchTransformers` (five configurations) |
| `NFKMLXReleaseWeights` (single-file or sharded release reader) | `NFKMLXReleaseWeights.swift` | every `directoryURL:` loader: the dense, hybrid, Gemma, embedder, reranker, T5, LTX DiT, and Music 3 loaders | each model's parity (a mis-read shard is a wrong number) |
| `NFKMLXWeights.loadCheckpoint` (safetensors, native `.pth` / `.pt` / `.ckpt` / `.th` / `.bin`, layout metadata, quantized reload) | `NFKMLXWeights.swift`, `NFKMLXTorchFormat.swift` | every model's `weightsURL:` factory | `NFKMLXTorchParityTests` (raw against converted, tensor for tensor) and `NFKMLXCheckpointRoundTripTests` (save → reload through each loader) |
| `NFKMLXGGUFFormat` (container + Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K dequantizers) | `NFKMLXGGUFFormat.swift` | the GGUF language path | bit-exact dequantization against the `gguf` package, then the GGUF-LM logit parity |
| `NFKMLXChatTemplateRenderer` (native Jinja subset) | `NFKMLXChatTemplate*.swift` | the language backend and every instruct release | `testRenderedTemplatesMatchTheReference` (Qwen3, Llama-3, Gemma templates byte for byte) |
| MLX runtime quantization (`quantize`, `matchStructure`, checkpoint metadata) | `NFKMLXWeights.swift` | the language backend, the Music 3 quantized release | `testAQuantizedCheckpointRoundTripsThroughTheLoaders`, the Music 3 quantized measurements |
| `NFKGemmaAttention` / `NFKGemmaFeedForward` (learned QK norms, value norm, proportional rotary, per-layer head widths) | `NFKMLXGemmaModel.swift` | the Gemma 4 E-series, the 26B-A4B mixture, the 12B unified decoder, the Gemma 4 vision and audio towers' block structure | Gemma 4 E2B layer-by-layer parity, the unified and mixture tiny parities |
| `NFKMLXGemmaTokenizer` (byte-fallback BPE from `tokenizer.json`) | `NFKMLXEmbeddingGemma.swift` | EmbeddingGemma, the Gemma 4 backend | token-for-token agreement with the reference |
| `NFKMLXHybridLanguageNet` (gated delta-rule recurrence) | `NFKMLXHybridLanguageModel.swift` | Qwen3.5 / 3.6 / 3.8; DeepSeek reuses its norm and gate helpers | Qwen3.5-4B layer-by-layer parity, the 27B structural check |
| `NFKMLXT5Encoder` (T5 v1.1 and umT5) | `NFKMLXT5Encoder.swift` | LTX-Video (T5-XXL), Wan (umT5-XXL) | the T5-XXL and umT5 parities |
| `NFKZImageRope` (3-axis interleaved rotary) | `NFKMLXZImageTransformer.swift` | Z-Image, Wan | both DiT parities |
| `NFKMLXFlowMatchScheduler` (rectified flow), `NFKMLXDPMSolverScheduler`, `NFKMLXUniPCScheduler` | their own files | LTX, Z-Image (flow); SANA (DPM-Solver++); Wan (UniPC) | schedule and trajectory checks against diffusers |
| `NFKDDIMScheduler` / `NFKLCMScheduler` | `NFKMLXDiffusionScheduler.swift` | Stable Diffusion text-to-image, inpainting, Marigold, the ×4 upscaler, the reference diffusion pipelines | `sd-scheduler` parity (per-step latents and add-noise) |
| `NFKMLXSDUNet` / `NFKMLXSDAutoencoder` (diffusers `UNet2DConditionModel` / `AutoencoderKL`, one implementation over configurations) | `NFKMLXStableDiffusionModels.swift` | SD 1.5, SD 2.1, SDXL-Turbo, SD inpainting, Marigold, the ×4 upscaler, TAESD-style previews, the Flux VAE preset Z-Image decodes through | each configuration's UNet / VAE parity |
| `NFKMLXSDTextEncoderNet` (CLIP text tower over `NFKMLXCLIP`'s blocks) | `NFKMLXSDTextEncoder.swift` | SD 1.5, SD 2.1, SDXL (both towers), CLIP ViT-B/32 text | the four text-tower parities and `clip-text` |
| `NFKMLXDiffusionBackend` (the sampler loop, inpaint hold, windowed continuation, latent preview) | `NFKMLXDiffusionBackend.swift` | text-to-image, inpainting, Marigold, the upscaler, Music 3's windowed flow matcher | the end-to-end text-to-image parities, `NFKMLXDiffusionWindowedTests`, the preview correlation test |
| `NFKSigLIPEncoder` / `NFKSigLIPLayer` (bidirectional ViT with biases, gelu-tanh) | `NFKMLXSmolVLM.swift` | SmolVLM's vision tower, SigLIP 2's vision and text towers | SmolVLM encoder seams, SigLIP 2 parity |
| `NFKMLXCLIP` blocks (fused-projection attention, quick-GELU / erf-GELU) | `NFKMLXCLIP.swift` | CLIP ViT-B/32, the SD text encoders, the CLIP probe | `clip` and `clip-text` parities |
| `NFKMLXResNetBackbone` (bottleneck ResNet, stride-to-dilation) | `NFKMLXResNet.swift` | DeepLabV3, SimpleBaseline pose | `deeplab` and `pose` parities |
| `NFKMLXResample` (bilinear / nearest / bicubic, reflect padding, explicit-border pooling) | `NFKMLXDepthAnything.swift`, `NFKMLXHTDemucs.swift` | nearly every image model (thirty files); `maxPooled` by BiSeNet, RT-DETR, ResNet, SAM 2, YOLO; `reflectPadded` by LaMa and style transfer | `NFKMLXResampleTests`, and the style-transfer / LaMa parities that measured edge padding as a real defect |
| `NFKMLXPixelShuffle` / `NFKBiSeNetPixelShuffle` | `NFKMLXBiSeNet.swift` | Real-ESRGAN, RIFE, SwinIR, BasicVSR, DC-AE, BiSeNet V2 | the dependent parities, and the factor-eight order test |
| `NFKMLXImageBridge` (`CGImage` / `MTLTexture` ↔ `MLXArray`, alpha preserved) | `NFKMLXImageBridge.swift` | every image backend, the detection backends, training data | `NFKMLXImageBridgeTests` |
| `NFKMLXFaceAlignment` + `NFKMLXRetinaFaceDetector` / `NFKMLXVisionFaceDetector` | `NFKMLXFaceAlignment.swift`, `NFKMLXRetinaFace.swift` | CodeFormer's photograph path | `retinaface` parity, `testTheAlignedCropContainsTheFace` |
| `NFKSAMTwoWayBlock` / `NFKSAMAttention` / `NFKSAMMLP` (the two-way transformer) | `NFKMLXSAM.swift` | SAM, SAM 2's mask decoder | `sam-mask` and `sam2-decoder` parities |
| `NFKMLXDemucsNet` + `NFKDemucsConvT1d` / `NFKDemucsBLSTM` / fractional resample | `NFKMLXDemucs.swift` | Demucs v2, the speech denoiser; the ConvT1d by Conv-TasNet, HiFi-GAN, Music 3, SNAC | `demucs` and `denoiser` parities |
| `NFKMLXAudioRate.matched` (polyphase resampling, `julius.resample_frac`) | `NFKMLXDemucs.swift` | MarbleNet VAD, Silero VAD, the audio tagger, DAC, SNAC, HTDemucs | the dependent parities (a wrong rate puts every frequency in the wrong bin) |
| `NFKMusic3Snake` / `NFKMusic3ResidualUnit` / `NFKMusic3VocoderBlock`, weight-norm fusion | `NFKMLXMusic3.swift` | the Music 3 vocoder, DAC, SNAC (Snake), HiFi-GAN and Kokoro (fusion) | `music3-vocoder`, `dac`, `snac`, `hifigan-universal`, `kokoro` |
| `NFKWhisperBlock` (transformer block) and `NFKMLXWhisperNet` | `NFKMLXWhisper.swift` | Whisper (four sizes), the neural G2P, the acoustic model, Whisper LoRA training | the Whisper parities |
| `NFKMLXHiFiGANNet` | `NFKMLXHiFiGAN.swift` | the FastSpeech2 voice, the standalone vocoder | `hifigan-universal`, the voice loop-closure test |
| `NFKMLXWaveFile` / `NFKMLXVideoFile` (WAV and AVFoundation I/O) | their own files | every audio backend; the video backend | round-trip tests |
| `NFKMLXTrainer` / `NFKMLXLoRA` | their own files | Zero-DCE, SegFormer, Whisper, CLIP-probe recipes; RVM fine-tune test | the loss parities (`zero-dce-losses`, `segformer-loss`), the gradient-overflow and LoRA tests |
| `NFKMLXModuleBackend` / `NFKMLXMattingBackend` / `NFKMLXTensorBackend` / `NFKMLXSpeechBackend` / `NFKMLXVideoBackend` | their own files | the InferKit contract for every single-forward model | the backend round-trip tests, and each model's public-path parity where recorded (CodeFormer, RIFE, YOLO) |

## Reproducing a number

```bash
python3 Tools/validation-assets/fetch.py            # every checkpoint and record → ~/.inferkit-validation
cd InferKitMLX && xcodebuild test -scheme InferKitMLXTests -destination 'platform=macOS' \
    -skipPackagePluginValidation -only-testing:InferKitMLXTests/NFKMLXReferenceParityTests 2>&1 \
    | grep 'VALIDATION PARITY'
```

To regenerate a record rather than reuse one, run the matching mode of
`Tools/reference-parity/run_reference.py` under the interpreter the manifest names for it; the
`IK_PARITY_*` entry in `~/.inferkit-validation.json` points the test at the new file.
