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
  ``NFKMLXNAFNetVariant``, ``NFKMLXSwinIRVariant``, ``NFKMLXHATVariant``, ``NFKMLXSAMVariant``, ``NFKMLXWhisperVariant``).
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
| ``NFKMLXRealESRGAN`` | `real-esrgan-x4` · `-anime` · `-x2` · `real-esrgan-general-x4v3` · `real-esrgan-anime-video-x4v3` | ×4 / ×2 super-resolution, RRDBNet and the later compact generator |
| ``NFKMLXSwinIR`` | `swinir-x4` | transformer super-resolution — every released checkpoint: classical ×2 / ×3 / ×4 / ×8, lightweight ×2 / ×3 / ×4, real-world ×4 medium and large |
| ``NFKMLXHAT`` | `hat-x4` · `hat-l-x4` · `real-hat-gan-x4` | hybrid attention super-resolution: window attention plus a channel-attention branch and overlapping cross-attention |
| ``NFKMLXNAFNet`` | `nafnet` | denoise / deblur (SIDD and GoPro at widths 32 and 64, REDS) |
| ``NFKMLXZeroDCE`` | `zero-dce` | low-light enhancement |
| ``NFKMLXZeroDCEPlus`` | `zero-dce-plus` | low-light enhancement with depthwise-separable convolutions and one shared curve |
| ``NFKMLXStyleTransfer`` | `fast-style-transfer` | one baked style per checkpoint |
| ``NFKMLXAdaIN`` | `adain` | arbitrary style transfer: any style image, no per-style checkpoint |
| ``NFKMLXColorizer`` | `colorizer-eccv16` | grayscale → color |
| ``NFKMLXSiggraphColorizer`` | `colorizer-siggraph17` | colorization with optional user hints |
| ``NFKMLXDDColor`` | `ddcolor` · `-paper` · `-artistic` | modern automatic colorization (learned color queries) |
| ``NFKMLXLaMa`` | `lama-inpaint` | mask-guided inpainting |
| ``NFKMLXStableDiffusionInpaint`` | `sd-inpaint` | latent-diffusion inpainting |
| ``NFKMLXCodeFormer`` | `codeformer` | face restoration; ``NFKMLXPhotoFaceBackend`` restores every face in a photograph |
| ``NFKMLXTAESD`` | `taesd` | tiny autoencoder: fast latent preview encode / decode |
| ``NFKMLXSDUNet`` / ``NFKMLXSDAutoencoder`` | — | the shared Stable Diffusion networks |

Each writes its result under `NFKOutputImage`.

### Image → map

`NFKMLXDepthAnything`, `NFKMLXDepthAnything3`, and `NFKMLXMarigold` emit a grayscale depth map, and
`NFKMLXDepth3Estimator` reads Depth Anything 3's camera and ray map beside that depth; the segmenters
(`NFKMLXSegFormer`, `NFKMLXDeepLab`, `NFKMLXBiSeNet`, `NFKMLXBiSeNetV2`) emit a grayscale class-label
map under `NFKOutputImage` — recover the class index as `round(gray · (classCount − 1))`.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXDepthAnything`` | `depth-anything-v2-small` · `-base` · `-large` | monocular depth |
| ``NFKMLXDepthAnything3`` | `depth-anything-3-small` · `-base` · `-large` | monocular depth, rays, and camera (DA3) |
| ``NFKMLXMarigold`` | `marigold-depth` | diffusion depth |
| ``NFKMLXSegFormer`` | `segformer-b0` | transformer segmentation |
| ``NFKMLXDeepLab`` | `deeplabv3` | CNN segmentation |
| ``NFKMLXBiSeNet`` | `bisenet` | real-time segmentation (ResNet-18 context path) |
| ``NFKMLXBiSeNetV2`` | `bisenet-v2` | real-time segmentation (detail + semantic branches) |

### Matting, segmentation & faces

`NFKMLXU2Net`, `NFKMLXRVM`, `NFKMLXMODNet`, and `NFKMLXBiRefNet` produce a straight foreground plus an
alpha matte under `NFKOutputMask`; `NFKMLXSAM` segments from a point prompt; `NFKMLXRetinaFace` returns
`[NFKDetection]` with five-point landmarks through ``NFKMLXRetinaFaceDetector``.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXU2Net`` | `u2net` · `u2netp` | salient-object matting |
| ``NFKMLXISNet`` | `isnet` | dichotomous segmentation, U²-Net's successor |
| ``NFKMLXRVM`` | `robust-video-matting` | recurrent video matting (MobileNetV3, or ResNet-50 as `robust-video-matting-resnet50`) |
| ``NFKMLXMODNet`` | `modnet` | trimap-free portrait matting |
| ``NFKMLXBiRefNet`` | `birefnet` | high-resolution background removal, MIT (Swin-v1-L + ASPPDeformable) |
| ``NFKMLXSAM`` | `sam` | promptable segmentation (ViT-B, ViT-L, ViT-H) |
| ``NFKMLXSAM3`` | `makeImageModel(fromHuggingFace:)` | SAM 3 and SAM 3.1: every instance a worded prompt names, segmented — the rotary ViT with its FPN neck, the CLIP text tower, and the DETR detector (text prompts only; box prompts and video are not ported) |
| ``NFKMLXSAM2`` | `backend(variant:release:weightsURL:)` | SAM 2 and SAM 2.1: Hiera encoder (tiny, small, base_plus, large), prompt encoder, mask decoder, and the video memory path, assembled by ``NFKMLXSAM2TrackerNet`` |
| ``NFKMLXRetinaFace`` | `retinaface-mobile025` | face detection with landmarks |
| ``NFKMLXFaceAlignment`` | — | five-point alignment to the CodeFormer template (RetinaFace or Vision) |

### Detection & pose

`NFKMLXYOLO`, `NFKMLXRTDetr`, and `NFKMLXRFDetr` return `[NFKDetection]` under `NFKOutputDetections`;
`NFKMLXPose` and `NFKMLXVitPose` return `[NFKKeypoint]` under `NFKOutputPose`.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXYOLO`` | `yolo` | object detection (YOLOv8 n / s / m / l / x) |
| ``NFKMLXYOLOGenerations`` | `yolov9t` … `yolo26x` | YOLOv9, YOLOv10, YOLO11, YOLOv12 and YOLO26 — every released size; v10 and 26 need no suppression |
| ``NFKMLXRTDetr`` | `rtdetr` | object detection, Apache-2.0 (RT-DETR and RT-DETRv2, r18vd / r34vd / r50vd / r101vd; no NMS) |
| ``NFKMLXRFDetr`` | `rf-detr` | object detection, Apache-2.0 (RF-DETR nano / small / medium / base / large, Roboflow; no NMS) |
| ``NFKMLXTableTransformer`` | `backend(directoryURL:)` | table detection and table-structure recognition, MIT (Table Transformer, Microsoft; vanilla DETR + ResNet-18; no NMS; all five releases) |
| ``NFKMLXPose`` | `pose-simplebaseline` | top-down pose |
| ``NFKMLXVitPose`` | `vitpose-base-simple` | top-down pose, Apache-2.0 (ViTPose base with the simple decoder, or the classic decoder as `vitpose-base`; DARK-refined keypoints) |

### Embeddings & reranking

The embedders return an L2-normalized vector under `NFKOutputEmbedding` — the key the core's
`NFKRemoteEmbeddingBackend` answers with too, so search code is engine-agnostic.
``NFKMLXModernBERTReranker`` scores a query against a list of documents and is an object rather than
a backend.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXCLIP`` | `clip-vit-b-32` | image + text embeddings (CLIP ViT-B/32, B/16, L/14, L/14@336, and MetaCLIP weights) |
| ``NFKMLXSigLIP2`` | `siglip2-base-patch16-224` | image + text embeddings (SigLIP 2, multilingual; every fixed-resolution release from base to giant-opt) |
| ``NFKMLXQwen3Embedding`` | — | text embeddings (Qwen3-Embedding-0.6B, Matryoshka) |
| ``NFKMLXEmbeddingGemma`` | — | text embeddings (EmbeddingGemma-300M, bidirectional) |
| ``NFKMLXModernBERTReranker`` | — | cross-encoder reranking (gte-reranker-modernbert-base) |
| ``NFKMLXLaya`` | — | typed decisions (Laya, the open Jev: choice / score / noul about a state; root, typed-decisions, multilingual) |
| ``NFKMLXOpenJevDeBERTa`` | — | typed decisions (open-jev-deberta: one DeBERTa-v3-large pass over the state and every question) |
| ``NFKMLXOpenJev`` | — | typed decisions (Open-Jev 2B / 9B / 27B: a LoRA-adapted Qwen3.5 or Qwen3.8 scoring each candidate answer) |
| ``NFKMLXChronos`` | — | time-series forecasting (Chronos-Bolt, quantile forecasts; an object) |
| ``NFKMLXQwen3VLEmbedder`` | — | text + image embeddings in one space (Qwen3-VL-Embedding-2B, instruction-conditioned) |
| ``NFKMLXQwen3VLReranker`` | — | multimodal cross-encoder reranking (Qwen3-VL-Reranker-2B) |

### Language models

``NFKMLXLanguageBackend`` generates text from a downloaded release directory
(``NFKMLXLanguage/backend(directoryURL:)``) or a GGUF file (``NFKMLXLanguage/backend(ggufURL:)``):
`NFKInputPrompt` or `NFKInputMessages` in, `NFKOutputText` out, with the release's own chat template
rendered by ``NFKMLXChatTemplateRenderer``. It streams through the job, and takes a context window,
key-value cache quantization, chunked prefill, a prompt cache, a draft model for speculative decoding,
and JSON, JSON-Schema, or fixed-choice constrained decoding — each also settable from Objective-C through
``NFKMLXGenerationParameterKey``.

| Model | Factory | Architecture |
| --- | --- | --- |
| ``NFKMLXLanguage`` | `backend(directoryURL:)` | dense decoders (Qwen3, Qwen2, Llama) and the Qwen3-MoE / Mixtral mixtures |
| ``NFKMLXLanguage`` | `backend(ggufURL:)` | any dense `llama` / `qwen2` / `qwen3` GGUF (Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K) |
| ``NFKMLXQwen4Exp`` | — | Qwen3.8-Flash-Next: hyper-connections, hashed n-gram per-layer embeddings, a sparse-attention indexer, 512 experts |
| ``NFKMLXHybridLanguage`` | — | Qwen3.5 / 3.6 / 3.8: gated delta-rule recurrence with full attention every fourth layer |
| ``NFKMLXMamba`` | `backend(directoryURL:)` | Codestral-Mamba: a Mamba-2 selective-scan state-space decoder, the first SSM (released 7B at bf16 logit cosine 0.9999146, greedy 12/12) |
| ``NFKMLXGraniteHybrid`` | `backend(directoryURL:)` | Granite 4.0-H: a hybrid Mamba/attention decoder reusing the Mamba-2 mixer, dense and MoE, with on-device LoRA fine-tuning (released h-1b at float32 logit cosine 1.0, greedy 12/12) |
| ``NFKMLXNemotronH`` | `backend(directoryURL:)` | Nemotron Nano 2: a hybrid Mamba/MLP/attention decoder reusing the Mamba-2 mixer, with on-device LoRA fine-tuning (tiny logit cosine 1.0, Nemotron-Nano-9B-v2 structural 341/341) |
| ``NFKMLXGemma3`` | `backend(directoryURL:)` | Gemma 3 (270M, 1B, 4B): sliding/full attention, a hybrid key-value cache, the release's chat template, through ``NFKMLXGemma3Backend`` |
| ``NFKMLXGemma3n`` | `backend(directoryURL:)` | Gemma 3n (E2B, E4B): AltUp's four residual copies, LAuReL, per-layer embeddings, activation sparsity, key-value sharing, through ``NFKMLXGemma3nBackend`` |
| ``NFKMLXGemmaLanguage`` | `backend(directoryURL:)` | Gemma 4 (E-series, 26B-A4B mixture, 12B unified) through ``NFKMLXGemmaBackend``; a Gemma 3 release is routed to ``NFKMLXGemma3`` |
| ``NFKMLXDeepSeek`` | ``NFKMLXDeepSeekBackend`` | DeepSeek V4 and V4.1: multi-head latent attention over a mixture of experts, with V4.1's shared compressed cache, n-gram memory, image tower and ``NFKMLXDeepSeekDraftStack``. Generation is incremental through ``NFKMLXDeepSeekCache`` (the arithmetic is measured; the released weights exceed a workstation) |
| ``NFKMLXGemma2Net`` | — | Gemma 2, the SANA text encoder |
| ``NFKMLXT5Encoder`` | `encoder(configuration:directory:)` | T5 v1.1 and umT5 encoders, the LTX and Wan text conditioning |

``NFKMLXModelSizing`` answers whether a release fits the machine before any weight loads.

### Translation

Every translator reads `NFKInputPrompt` with `NFKParameterTargetLanguage` (BCP-47, required) and
`NFKParameterSourceLanguage` (optional) and returns `NFKOutputText`; the decode is tuned through
``NFKMLXTranslationParameterKey``.

| Class | Registered name | What it does |
| --- | --- | --- |
| ``NFKMLXMarian`` | `opus-mt` | OPUS-MT, one Helsinki-NLP release per language pair (or target group); built from a directory, a repo, or two language tags |
| ``NFKMLXM2M100`` | `m2m100`, `small100` | M2M-100 418M / 1.2B and SMaLL-100, 100 languages many-to-many; the source detected when omitted |
| ``NFKMLXMADLAD`` | `madlad400-3b-mt` | MADLAD-400 3B-MT, 400+ languages over T5, float32 or bfloat16 |
| ``NFKMLXTranslateGemma`` | `translategemma` | TranslateGemma 4B / 12B / 27B, Gemma 3 driven by its translation template; greedy |

### Vision-language

| Model | Factory | Task |
| --- | --- | --- |
| ``NFKMLXSmolVLM`` | `smolVLM(directoryURL:)` | an image and a question → an answer (SmolVLM2 256M, 500M, 2.2B) |
| ``NFKMLXGemma3`` | `load(directoryURL:)` | an image and a question → an answer (Gemma 3 4B: SigLIP so400m at 896, 256 soft tokens, bidirectional attention among them) |
| ``NFKMLXGemma3n`` | `load(directoryURL:)` | an image or a clip and a question → an answer (MobileNetV5-300M at 768 → 256 soft tokens; a USM Conformer → 188) |
| ``NFKMLXQwen3VL`` | — | the Qwen3-VL-2B vision tower (2-D rotary ViT, deepstack) |
| ``NFKMLXPixtral`` | — | Pixtral 12B (2-D rotary ViT + GELU connector + Mistral-Nemo decoder) |
| ``NFKMLXFlorence2`` | `backend(directoryURL:)` | Florence-2 (base, large, and their fine-tuned -ft releases): an image + a task token → text, plus boxes for the localization tasks (DaViT + BART) |
| ``NFKMLXTrOCR`` | `backend(directoryURL:)` | TrOCR: a handwriting-line image → its transcription (ViT encoder + trocr decoder) |
| ``NFKMLXSa2VA`` | `backend(directoryURL:)` | Sa2VA, every release (InternVL, Qwen3-VL, Qwen2.5-VL, and LLaVA-1.5 families; SAM 2 or SAM 3 grounding): an image + a referring prompt → text, plus a mask when the answer carries `[SEG]` |
| ``NFKMLXPhi4MM`` | `backend(directoryURL:precision:)` | Phi-4-multimodal: a prompt or conversation with any number of pictures and clips → text (SigLIP + Conformer + Phi-4-mini with a mixture of LoRAs); text logits 0.9999999999965636; speech encoder 0.9999999999982836, logits 0.9999999999994174; vision SigLIP 0.9999999999468102, logits 0.9999999999986776; vision with speech 0.9999999999993611; every answer exact |
| ``NFKMLXGemma4ConditionalGeneration`` | — | the tri-modal Gemma 4 chain: image and audio towers fused into the decoder |

### Video

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXRIFE`` | `rife` | frame interpolation (HDv3, midpoint) |
| ``NFKMLXRIFEv4`` | `rife-v4` | frame interpolation (v4, any timestep) |
| ``NFKMLXRAFT`` | `raft` | optical flow |
| ``NFKMLXVideoSR`` | `video-super-resolution` | recurrent video super-resolution (BasicVSR ×4) |
| ``NFKMLXVJEPA2`` | `backend(directoryURL:)` | self-supervised video features, and ranked classes from the Something-Something v2 and Diving48 releases, MIT (V-JEPA 2, Meta; 3D-RoPE ViT-L, ViT-H, or ViT-g) |
| ``NFKMLXCosmosTokenizer`` | `cosmos-tokenizer-ci8x8` … `cosmos-tokenizer-dv8x16x16` | image and causal video tokenizers: a continuous latent or discrete FSQ tokens, and the reconstruction (Cosmos Tokenizer, NVIDIA; ten releases) |
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
| ``NFKMLXHTDemucs`` | `htdemucs` | 4- or 6-stem music separation (Hybrid Transformer Demucs v4; `htdemucs-6s`, and the fine-tuned bag) |
| ``NFKMLXConvTasNet`` | `conv-tasnet` | speech separation |
| ``NFKMLXDenoiser`` | `denoiser` | speech noise suppression |
| ``NFKMLXMPSENet`` | `mpsenet` | speech enhancement (MP-SENet, magnitude + phase) |
| ``NFKMLXGTCRN`` | `gtcrn` | real-time speech enhancement (GTCRN, ~48K params) |
| ``NFKMLXSGMSE`` | `sgmse` | score-based generative dereverberation / enhancement (SGMSE+, NCSN++) |
| ``NFKMLXStoRM`` | `storm` | few-step stochastic regeneration (StoRM: predictor + conditioned score) |
| ``NFKMLXMossFormer2SENet`` | `mossformer2-se` | full-band 48 kHz speech enhancement (MossFormer2 SE) |
| ``NFKMLXMossFormer2SRNet`` | `mossformer2-sr` | 48 kHz speech super-resolution (MossFormer2 SR: mel-to-mel backbone + Snake HiFi-GAN + bandwidth substitution) |
| ``NFKMLXDeepFilterNet`` | `deepfilternet3` | real-time 48 kHz speech denoising (DeepFilterNet3, ~2.3M params) |
| ``NFKMLXVoiceRestore`` | `voicerestore` | flow-matching universal speech restoration (VoiceRestore + BigVGAN, ~301M) |
| ``NFKMLXResembleEnhance`` | `resemble-enhance` | five-network general speech restoration (STFT-mask denoiser + IRMAE/CFM latent flow matching + UnivNet LVC vocoder) |
| ``NFKMLXMetricGANPlus`` | `metricgan-plus` | speech enhancement (MetricGAN+, a two-layer BLSTM magnitude mask) |
| ``NFKMLXCMGAN`` | `cmgan` | speech enhancement (CMGAN, a conformer metric GAN with mask + complex decoders) |
| ``NFKMLXFRCRN`` | `frcrn` | speech enhancement (FRCRN, two complex UNets with frequency-recurrent FSMNs) |
| ``NFKMLXNUWave2`` | `nuwave2` | diffusion bandwidth extension to 48 kHz (NU-Wave 2, short-time Fourier convolutions, 8-step DDIM) |
| ``NFKMLXApollo`` | `apollo` | music codec-artifact restoration at 44.1 kHz (Apollo, 80-band Roformer; CC-by-SA weights) |
| ``NFKMLXDAC`` | `dac` | neural audio codec (Descript, 44.1 / 24 / 16 kHz) |
| ``NFKMLXSNAC`` | `snac` | multi-scale neural audio codec (24 kHz speech; 32 / 44.1 kHz music) |
| ``NFKMLXBigVGAN`` | `bigvgan-v2-24khz` | anti-aliased SnakeBeta vocoder (BigVGAN v2, mel → waveform copy-synthesis) |
| ``NFKMLXMimi`` | `mimi` | transformer-in-codec neural audio codec (Kyutai Mimi, 24 kHz → 12.5 Hz, split RVQ) |
| ``NFKMLXBasicPitch`` | `basic-pitch` | music transcription: audio to notes, as an `NFKMIDISequence` |
| ``NFKMLXAllInOne`` | `allin1` | music structure: beats, downbeats, tempo, and labeled sections |
| ``NFKMLXMuScriptor`` | `muscriptor` | multi-instrument transcription: audio to one MIDI track per instrument |
| ``NFKMLXHFTTransformer`` | `hft-transformer` | piano transcription: onset, offset, multi-pitch, and velocity |
| ``NFKMLXWhisper`` | `whisper-tiny` | speech → text (tiny / base / small / medium / large / large-v3 / large-v3-turbo, timestamps) |
| ``NFKMLXParakeet`` | `parakeet-tdt` | speech → text (Parakeet-TDT 0.6B v2, FastConformer + token-and-duration transducer; per-token timestamps) |
| ``NFKMLXGraniteSpeech`` | `backend(directoryURL:)` | speech → text (Granite Speech 3.3-2b: Conformer encoder + BLIP-2 Q-former + dense Granite decoder, audio LoRA folded; transcribes the validation clip exactly) |
| ``NFKMLXVoxtral`` | `backend(directoryURL:)` | speech → text (Voxtral-Mini 3B: Whisper encoder + 2-linear projector + Llama decoder, tekken tokenizer; transcribes the validation clip exactly) |
| ``NFKMLXCanary`` | `canary-1b-v2`; `backend(directoryURL:)` | speech → text (Canary-1B-v2: biased FastConformer encoder + Transformer attention encoder-decoder; multitask ASR/translation; transcribes the validation clip exactly) |
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
| ``NFKMLXZImageGenerator`` / ``NFKMLXZImagePipeline`` | ``NFKMLXZImageTransformerNet`` + the Flux VAE + Qwen3 | Z-Image text-to-image from a release, staged where it does not fit whole; image-to-image through the pipeline |
| ``NFKMLXSANAPipeline`` | ``NFKMLXSANATransformerNet`` + ``NFKMLXDCAutoencoderNet`` + Gemma 2 | SANA text-to-image |
| ``NFKMLXLTXPipeline`` | ``NFKMLXLTXTransformer`` + ``NFKMLXLTXVideoVAE`` + T5-XXL | LTX-Video text-to-video |
| ``NFKMLXLTX2TransformerNet`` | the LTX-2 audio-video transformer alone | one transformer denoising a video latent and an audio latent together |
| ``NFKMLXWanPipeline`` | ``NFKMLXWanTransformerNet`` + ``NFKMLXWanVideoVAENet`` + umT5 | Wan text-to-video |
| ``NFKMLXWanAnimate`` | ``NFKMLXWanAnimateNet`` | Wan 2.2 Animate 2: a reference character driven by a video's motion (the arithmetic is measured; the released weights exceed a workstation) |
| ``NFKMLXQwenImagePipeline`` | ``NFKMLXQwenImageNet`` + the Wan VAE at `.qwenImage21` + a Qwen3-VL 8B text encoder | Qwen-Image 2.1 text-to-image (Qwen Research License, non-commercial) |
| ``NFKMLXSD3Generator`` / ``NFKMLXSD3Pipeline`` | ``NFKMLXSD3TransformerNet`` + ``NFKMLXSDAutoencoder`` + CLIP/T5 | Stable Diffusion 3 / 3.5 text-to-image from a release, staged where it does not fit whole |
| ``NFKMLXFluxPipeline`` | ``NFKMLXFluxTransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux`) + CLIP-L/T5 | FLUX.1 text-to-image |
| ``NFKMLXSD3ControlNetPipeline`` | ``NFKMLXSD3ControlNetNet`` + ``NFKMLXSD3TransformerNet`` + ``NFKMLXSDAutoencoder`` | SD3 ControlNet: a spatial control image steers generation |
| ``NFKMLXFlux2`` | ``NFKMLXFlux2TransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux2`) + ``NFKMLXFlux2LatentCodec`` + a Qwen3 | FLUX.2 [klein] text-to-image, end to end |
| ``NFKMLXFluxControlNetPipeline`` | ``NFKMLXFluxControlNetNet`` + ``NFKMLXFluxTransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux`) | FLUX.1 ControlNet: a spatial control image steers generation |

## Topics

### Related

- <doc:ModelIndex>
- <doc:BringYourOwnBackends>
- <doc:DiffusionAndSchedulers>
- <doc:WeightsAndConversion>
