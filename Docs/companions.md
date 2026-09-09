# Optional companion packages

InferKit's core ships only backends built on Apple frameworks. Two companion packages add heavier
engines without raising the core's platform floor or adding dependencies to it. Each is a separate
SwiftPM package in a subdirectory of this repository; see the README's "Adding a companion" for how
to consume one.

## InferKitFoundationModels (optional companion)

`InferKitFoundationModels/` is a separate SwiftPM package (macOS 26 / iOS 26, Apple Intelligence
hardware) that bridges InferKit and Apple's **Foundation Models** framework:

- **`NFKFoundationModelsBackend`** — wraps the on-device system language model
  (`LanguageModelSession`) as an `NFKInferenceBackend`. The same request that runs against
  `NFKCoreMLLanguageBackend` or `NFKRemoteBackend` runs here: `NFKInputPrompt` or
  `NFKInputMessages` (a system message becomes the session's instructions), the standard text
  parameters, streamed partial text through the job. `isReady` reflects the model's availability.

The reverse direction — adopting Apple's provider protocols (`LanguageModel` /
`LanguageModelExecutor`, WWDC26) so InferKit's local and remote backends stand behind
`LanguageModelSession` — needs the macOS 27 / iOS 27 SDK and follows when that SDK is the build
baseline.

## InferKitMLX (optional companion)

`InferKitMLX/` is a separate SwiftPM package (Apple Silicon, macOS 14 / iOS 17) that keeps MLX out
of the core. It ships five bring-your-own-model backends and a gallery of sixty-plus real models —
upscaling, depth, matting, segmentation, detection, faces, pose, restoration, interpolation, optical
flow, colorization, embeddings, reranking, vision-language, on-device language models, text-to-image
and text-to-video diffusion, speech recognition and synthesis, audio codecs, and music — each
implemented in MLXNN and validated numerically against its reference implementation on the released
weights:

- **`NFKMLXBackend`** — a bundled Stable Diffusion release: SD 1.5, SD 2.1 base, or SDXL-Turbo. A
  request with no image runs text-to-image; a request with a `CGImage` under `NFKInputImage` runs
  image-to-image, with `NFKParameterStrength` controlling how much of the source survives. Every part
  of the run is this package's own — the CLIP text tower, the UNet, the DDIM sampler, and the
  autoencoder — each measured against its reference implementation, so text-to-image builds for iOS
  as the rest of the package does.
- **`NFKMLXModuleBackend`** — a bring-your-own MLX image model. Supply a `forward` closure over
  `MLXArray`; the backend handles the InferKit contract and the RGB `CGImage ↔ MLXArray` bridge.
- **`NFKMLXMattingBackend`** — a bring-your-own MLX image-matting model (a green/blue-screen keyer, a
  background remover). The plate under `NFKInputImage` and an optional hint (trimap or coarse alpha)
  under `NFKInputMask` become tensors, a `(plate, hint) -> [H, W, 4]` closure runs, and the straight
  foreground plus alpha matte returns as an RGBA `CGImage` under `NFKOutputImage`. `NFKMattingConfiguration`
  adds the matte on its own under `NFKOutputMask`, premultiplication, color space, tiled inference for
  large plates, and `MTLTexture` output.
- **`NFKMLXTensorBackend`** — a general bring-your-own MLX backend over named image tensors: several
  inputs in, several outputs out (a compositing model reading a foreground and a background, a model
  returning both an image and a mask). Each port binds an InferKit key to a tensor name.
- **`NFKMLXLanguageBackend`** — on-device text generation from a released Hugging Face directory.
  The dense decoder Qwen3 and Llama share, with the Qwen3-MoE, Qwen2-MoE (a shared expert beside the
  routed ones), Mixtral, and gpt-oss (sliding layers, attention sinks, clamped fused MXFP4 experts) mixtures
  of experts through a routed feed-forward; Gemma 4 (including its 26B-A4B mixture, a routed branch beside each layer's
  dense feed-forward) and the Qwen3.5 hybrid have their own classes. A key-value cache
  that bounds, quantizes, rolls back, and persists between turns; speculative decoding from a draft
  release; ChatML rendering, or the release's own Jinja `chat_template` through
  `NFKMLXChatTemplateRenderer`; and grammar-constrained output (JSON, a JSON Schema, or a fixed set of
  choices). Each
  family is measured against `transformers`' own implementation, and every option reaches
  Objective-C as a request parameter.
- **`NFKMLXGemmaBackend`** — text generation for the Gemma 4 decoders (`NFKMLXGemmaLanguage.backend(directoryURL:)`
  / `gemmaBackendWithDirectoryURL:error:`), dispatching on a release's config model type across the
  E-series, the 26B-A4B mixture, and the 12B unified decoder. Gemma runs prefill-only, so generation
  re-runs the growing sequence each step; the tokenizer is Gemma's byte-fallback BPE, and a message
  list is rendered into Gemma's turn format. Measured end to end on the released E2B. A Gemma 3
  release handed to the same factory is routed to `NFKMLXGemma3`.
- **`NFKMLXGemma3`** — the Gemma 3 line, end to end: the text decoder (`NFKMLXGemma3Net`: `(1 + w)`
  RMS norms, sandwich blocks, five sliding-window layers to one full layer at a local and a global
  rotary base, the larger sizes' 8× linear rotary scaling on the full layers, per-head QK norm, GeGLU,
  a tied head) for 270M, 1B, and 4B, generating through a hybrid key-value cache (unbounded for the
  full layers, bounded to the window for the sliding ones) with streaming and cancellation, the
  release's own Jinja chat template rendered by `NFKMLXChatTemplateRenderer`, and for the multimodal
  4B the SigLIP so400m vision tower at 896×896, the projector (a 4×4 average pool to 256 soft tokens,
  a Gemma norm, one matrix), the processor's `\n\n<start_of_image> … <end_of_image>\n\n` prompt
  expansion, and the bidirectional attention among an image's tokens. `NFKMLXGemma3Backend` takes
  `NFKInputImage` beside the text; `answer(image:question:)` / `answerForImage:question:error:` is
  the object path. Reference parity on the released weights against transformers' own Gemma 3 for
  every size and every stage (see [model parity](model-parity.md)). The gated `google/gemma-3-*`
  are mirrored ungated at `unsloth/gemma-3-*-it`. The same decoder blocks now serve EmbeddingGemma's
  encoder, whose bidirectional window bound is the reference's `span / 2 + 1`.
- **`NFKMLXGemma3n`** — Gemma 3n, tri-modal and end to end, and a distinct architecture rather than a
  Gemma 3 variant. The decoder (`NFKMLXGemma3nNet`) carries four mechanisms none of the other Gemmas
  has: **AltUp**, which makes the residual stream four parallel copies with a learned per-token map
  predicting them before each block and correcting them after; **LAuReL**, a rank-64 detour beside the
  attention residual; **per-layer embeddings**, a second wide table giving every layer its own slice;
  and **activation sparsity**, which holds the first ten layers' feed-forward gates at zero below a
  per-token Gaussian cutoff. Attention runs at scale 1.0, the values carry an unweighted normalization,
  and the last `num_kv_shared_layers` compute no keys or values at all, reusing the last non-shared
  layer of their own kind. The audio encoder (`NFKMLXGemma3nAudioNet`) is a Universal Speech Model
  Conformer over a cumulative group normalization; the vision tower (`NFKMLXGemma3nVisionNet`) is
  **MobileNetV5-300M**, a convolutional encoder with multi-query attention over the feature map, which
  the release reaches through `timm`. `NFKMLXGemma3nBackend` takes `NFKInputImage` and `NFKInputAudio`
  beside the text; `answer(image:question:)` / `answerForImage:question:error:` is the object path.
  Reference parity on the released E2B weights for every stage — the decoder layer by layer, both
  towers, the mel front end, and the whole fused chain (see [model parity](model-parity.md)). The gated
  `google/gemma-3n-*` are mirrored ungated at `unsloth/gemma-3n-*-it`.
- **`NFKMLXChatTemplateRenderer`** — renders the Jinja `chat_template` an instruct release ships, so the
  language backend reproduces the model's trained input rather than the ChatML approximation. A compact
  Jinja interpreter (for / if / set, `namespace`, slicing, the `loop` variable, `is` tests, string
  methods, `tojson`/`trim`, and the `trim_blocks`/`lstrip_blocks` whitespace model), pure Foundation so
  it needs no runtime. Reference parity against `transformers`' own `apply_chat_template` over the
  Qwen3, Llama-3, and Gemma templates.
- **`NFKMLXQwen3Embedding`** — on-device text embeddings: the Qwen3-0.6B dense decoder read one layer
  earlier (its post-final-norm hidden states), pooled at the last token over an appended
  `<|endoftext|>` and L2-normalized, so a dot product between two embeddings is their cosine
  similarity. A query carries a one-sentence task instruction and a document carries none.
  `NFKMLXTextEmbeddingBackend` returns the vector under `NFKOutputEmbedding` for semantic search,
  retrieval, clustering, and reranking; `backendWithDirectoryURL:outputDimensions:error:` truncates to
  a smaller Matryoshka width. Reference parity against the model card's own transformers recipe on the
  released 0.6B weights (query and document embedding cosine 0.99999999999, retrieval score to 1e-6).
- **`NFKMLXEmbeddingGemma`** — a second text embedder over a second architecture: the bidirectional
  Gemma 3 encoder (no causal mask, sandwich normalization, dual RoPE, QK-norm, GeGLU), mean-pooled over
  every token, through a Dense bottleneck (768 → 3072 → 768) and L2-normalized. The backbone is Gemma 3
  (`gemma3_text`), not the causal Gemma 4 the language model runs, so it is its own implementation. It
  reads Gemma's byte-fallback BPE `tokenizer.json` directly, so the text path needs no conversion.
  Reference parity against the sentence-transformers pipeline on the released 300M weights (every one of
  the 24 layers exact, query and document embedding cosine 0.99999999999), with the tokenizer reproducing
  the reference's ids token for token. The gated `google/embeddinggemma-300m` is mirrored ungated at
  `unsloth/embeddinggemma-300m`.
- **`NFKMLXModernBERTReranker`** — a cross-encoder reranker (`gte-reranker-modernbert-base`): a
  bidirectional ModernBERT encoder over a `[CLS] query [SEP] document [SEP]` pair, mean-pooled, through a
  prediction head and a single-logit classifier that gives a relevance score. Where an embedder scores a
  query and a document independently, a cross-encoder reads the pair together and is more accurate, which
  is what reorders an embedder's shortlist. `scoresForQuery:documents:` and `rankedIndicesForQuery:documents:`
  score and order candidates. ModernBERT is RoPE (a global base every third layer, a local base with a
  sliding window elsewhere), GeGLU, and LayerNorm without biases. Reference parity against transformers'
  own `ModernBertForSequenceClassification` (every one of the 22 layers exact, scores to within 5e-3);
  the byte-level BPE tokenizer is read from the release's `tokenizer.json`.
- **`NFKMLXSmolVLM`** — a vision-language model (SmolVLM2-500M): an image and a question in, an answer
  out. A SigLIP vision encoder turns each image tile into patch features, a pixel-shuffle connector
  projects them to the decoder width, and a Llama decoder (the dense stack the language model runs)
  reads the text with the projected vision tokens spliced in at the image-token positions.
  `answerForImage:question:` captions or answers about a `CGImage`. Reference parity against
  transformers' own SmolVLMForConditionalGeneration (the vision encoder, the connector, and the fused
  decoder logits exact, the greedy continuation token for token); the prompt expansion is token-exact
  and the image processor is CoreGraphics-based (a slight approximation of the reference's PIL resize).
- **`NFKMLXQwen3VLVisionNet`** — the vision tower of a second VLM, Qwen3-VL-2B: a 2D-rotary ViT (patches
  in 2×2 merge blocks, a bilinearly interpolated position embedding), a merger that folds each block to
  the decoder width, and a three-layer "deepstack" of feature maps. At reference parity against
  transformers' own Qwen3-VL vision model (patch embedding, position embedding, merged output, and every
  deepstack feature exact). The decoder is the Qwen3 dense stack; its Qwen3-VL-specific M-RoPE and
  deepstack injection are the remaining integration.
- **`NFKMLXGGUF`** — a native GGUF reader, the sequel to the native PyTorch checkpoint reader. GGUF is
  the format most quantized language models are distributed in. This reads the container's metadata and
  tensor table and dequantizes the block-quant formats a real model uses (`Q4_K`, `Q6_K`, `Q8_0`, `Q5_0`,
  `Q4_0`, `F16`, `F32`) into `MLXArray`s, with no Python and no llama.cpp. Bit-exact against the `gguf`
  package on a real Q4_K_M model (worst |difference| 0.0 across every dequantizer). A type it does not
  implement is refused per-tensor rather than failing the file. It is wired into the language-model
  loader, so a GGUF release generates text end to end through `NFKMLXLanguage.backend(ggufURL:)`
  (`backendWithGGUFURL:error:`): the metadata becomes a configuration, the llama.cpp tensor names are
  remapped and the query/key projections un-permuted for the decoder's rotary, and the embedded
  tokenizer is rebuilt. Reference parity against transformers loading the same GGUF (logit cosine
  0.9999999999).
Each model's classes, configuration preset, registered name, and factory are tabulated in
[model-index.md](model-index.md); the per-model parity numbers, the reference each is measured against,
and the subsystems the models share are in [model-parity.md](model-parity.md).

Every shipped real model has a direct Objective-C factory — `[Model backendWith[Variant:]weightsURL:error:]`
for local weights and `[Model backendWith[Variant:]repo:weightsPath:revision:cacheDirectoryURL:error:]`
to download from Hugging Face and build — so a consumer (e.g. MetalForge) constructs them without the
registry. The registry (`register()` / `registerAll()` / `NFKMLXHub`) remains for custom / bring-your-own
models.

- **`NFKMLXRealESRGAN`** — a real, shipped single-forward model: the Real-ESRGAN generator (RRDBNet)
  in MLXNN, run through `NFKMLXModuleBackend` for ×4 upscaling. Build directly from Objective-C via
  `backendWithVariant:weightsURL:error:`, or download and build via the `repo:` factory.
- **`NFKMLXDepthAnything`** — a real single-forward depth model: the Depth Anything V2 DINOv2 + DPT
  network in MLXNN, run through `NFKMLXModuleBackend` (image → grayscale depth). Register and build by
  name; a self-validating converter turns the release into a safetensors checkpoint.
- **`NFKMLXDepthAnything3`** — Depth Anything 3 monocular depth (DA3-SMALL, -BASE, and -LARGE): a DINOv2 ViT variant
  (2D rotary, query/key norm, a camera token, and `cat_token` local/global hooking from block 4) plus
  the DualDPT depth branch, in MLXNN, run through `NFKMLXModuleBackend` (image → grayscale depth). At
  reference parity against the authors' `depth_anything_3` package; the released safetensors loads
  directly (`depth-anything-3-small` / `-base` / `-large`, `NFKMLXDepth3Variant`).
- **`NFKMLXU2Net`** — a real single-forward background remover: the U²-Net nested-U saliency network
  in MLXNN, run through the matting backend (plate → foreground + alpha cutout). Full `u2net` + light `u2netp`.
- **`NFKMLXSAM`** — real promptable segmentation (Segment Anything): a ViT encoder, prompt encoder, and
  two-way-transformer mask decoder in MLXNN, run through the matting backend (plate + point → mask).
  Every released encoder: ViT-B, ViT-L, and ViT-H (`NFKMLXSAMVariant`).
- **`NFKMLXNAFNet`** — a real single-forward restoration network (denoise / deblur): a U-shaped stack
  of NAFBlocks in MLXNN, run through the module backend (degraded image → restored image). All five
  releases are presets (`NFKMLXNAFNetVariant`): SIDD and GoPro at width 32, REDS, and SIDD and GoPro at
  width 64.
- **`NFKMLXRIFE`** / **`NFKMLXRIFEv4`** — real frame interpolation: the RIFE HDv3 and v4 IFNets in
  MLXNN with a flow-based warp (v4 adds an arbitrary-timestep midpoint), run
  through the tensor backend (two frames → the interpolated middle frame) for slow-motion / retiming.
- **`NFKMLXRAFT`** — real optical flow: the RAFT correlation-and-ConvGRU pipeline in MLXNN, run through
  the tensor backend (two frames → a dense flow field) for motion vectors, warping, retiming.
- **`NFKMLXLaMa`** — a real single-forward inpainter: the LaMa FFC-ResNet generator in MLXNN, with an
  FFT spectral branch (via MLXFFT), run through the matting backend (plate + mask → inpainted image).
- **`NFKMLXTextToImage`** — Stable Diffusion text-to-image, on the diffusion backend: a CLIP text
  tower encodes the prompt, the UNet denoises over the DDIM loop with classifier-free guidance, the
  autoencoder decodes. Three releases ship as configurations — Stable Diffusion 1.5, Stable Diffusion
  2.1, and SDXL-Turbo — and `NFKMLXBackend` is that model behind a release name. Each is measured end
  to end against the diffusers pipeline it comes from, sampling with DDIM at the release's own
  timestep spacing on both sides.
- **`NFKMLXStableDiffusionInpaint`** — a latent-diffusion inpainter (VAE + UNet in MLXNN) on the
  diffusion backend: VAE-encode the plate, denoise a 9-channel input over the DDIM loop, VAE-decode.
- **`NFKMLXMarigold`** / **`NFKMLXSDUpscaler`** — image-conditioned latent-diffusion models on the
  diffusion backend: Marigold depth (image → depth) and the SD ×4 latent upscaler (image → ×4 image).
- **`NFKMLXStyleTransfer`** — real fast neural style transfer: Johnson et al.'s `TransformerNet` in
  MLXNN, run through the module backend (image → stylized image). The style is baked into the weights.
- **`NFKMLXCLIP`** — real image+text embeddings (CLIP ViT-B/32, B/16, L/14, L/14@336; `NFKMLXCLIPVariant`): a ViT image tower and a text
  transformer projected into a shared space. `NFKMLXCLIPBackend` returns an embedding under
  `NFKOutputEmbedding` for semantic search, tagging, and diffusion guidance.
- **`NFKMLXRVM`** — real video matting (Robust Video Matting, both released encoders — MobileNetV3 and ResNet-50): an encoder, LR-ASPP, and a **recurrent
  ConvGRU decoder** that carries state across frames, run through the matting backend (single frame) or
  `NFKMLXRVMNet.forward` (video, state threaded) for background removal without a green screen.
- **`NFKMLXCodeFormer`** — real face restoration: a VQGAN encoder/generator with a Transformer that
  predicts codebook indices, run through the module backend (aligned face → restored face).
- **`NFKMLXZeroDCE`** — real low-light enhancement: the Zero-DCE DCE-Net estimates pixel-wise tone
  curves and iteratively brightens, run through the module backend (dark image → brightened image).
- **`NFKMLXMODNet`** — real trimap-free portrait matting: the three-branch (semantic / detail /
  fusion) MODNet, run through the matting backend (portrait → foreground + alpha).
- **`NFKMLXYOLO`** — real object detection: an anchor-free YOLO with box decode and non-max
  suppression, returning `NFKDetection`s (label, confidence, normalized box) under
  `NFKOutputDetections`. Build with class names via the `backendWith…labels:` factory.
- **`NFKMLXSegFormer`** — real semantic segmentation: the SegFormer MiT transformer + all-MLP head,
  run through the module backend, emitting a grayscale class-label map under `NFKOutputImage`.
- **`NFKMLXSwinIR`** — real transformer super-resolution: SwinIR with true shifted-window attention
  and pixel-shuffle upsampling, run through the module backend (low-res → high-res image). Every
  released SR checkpoint has a variant: classical ×2/×3/×4/×8, lightweight ×2/×3/×4, and the two
  real-world ×4 models (nearest-neighbor tail; the large one with the three-convolution residual).
- **`NFKMLXColorizer`** / **`NFKMLXSiggraphColorizer`** — real colorization (ECCV-16 and
  SIGGRAPH-17): predict ab chroma from the L channel in CIELAB space and recombine with the original
  luminance, run through the module backend (grayscale photo → color photo); the SIGGRAPH model also
  takes user color hints. The converter loads the reference releases directly.
- **`NFKMLXPose`** — real top-down pose estimation (SimpleBaseline): a residual backbone and a
  deconvolution head produce joint heatmaps, decoded to `NFKKeypoint`s (name, normalized position,
  confidence) under `NFKOutputPose`. Build with joint names via the `backendWith…jointNames:` factory.
- **`NFKMLXDeepLab`** — real semantic segmentation (DeepLabV3): a residual backbone and an ASPP head,
  run through the module backend, emitting a grayscale class-label map (a CNN counterpart to SegFormer).
- **`NFKMLXConvTasNet`** — real time-domain speech separation: a convolutional encoder, a masking
  temporal-convolution network, and a shared decoder, returning one `NFKAudioAsset` per speaker.
- **`NFKMLXDenoiser`** — real speech noise suppression: the Demucs time-domain U-Net with a single
  output channel, returning one cleaned `NFKAudioAsset` under `NFKOutputAudio`.
- **`NFKMLXVAD`** — real voice activity detection (MarbleNet): a small conv net over a log-mel
  spectrogram marks speech spans, returning `NFKAudioSegment`s under `NFKOutputSegments`.
- **`NFKMLXAudioTagger`** — real audio tagging (PANNs): a conv net over the log-mel spectrogram
  predicts the sounds present, returning top-K `NFKClassification`s under `NFKOutputClassifications`.
- **`NFKMLXBiSeNet`** / **`NFKMLXBiSeNetV2`** — real real-time semantic segmentation: a two-path
  (spatial detail + context)
  network with attention refinement and feature fusion, emitting a grayscale class-label map.
- **`NFKMLXVideoSR`** — real recurrent video super-resolution: a ConvGRU propagates a hidden state
  along the clip; single frame through the module backend, or a sequence via `upscaleSequence`.
- **`NFKMLXSpeechBackend`** — a bring-your-own MLX text-to-speech backend: a `(String) -> MLXArray`
  waveform closure, written to a WAV file and returned as an `NFKAudioAsset` (text → audio).
- **`NFKMLXMusicBackend`** — real music generation (MiniMax Music 3): a music description under
  `NFKInputPrompt` and lyrics under `NFKInputLyrics` become a stereo 44.1 kHz `NFKAudioAsset`. A
  Qwen3-8B autoregressive stage samples RVQ codes frame by frame, a flow-matching transformer
  denoises audio latents conditioned on its hidden states over overlapping windows, and a Snake
  vocoder decodes them — every network measured against the official diffusers implementation on
  the released weights. Build with `NFKMLXMusic3.backend(directoryURL:)` from the downloaded
  release tree (~27 GB); the weights are separately licensed — see "Model weight licenses" below.
  `NFKMLXMusic3.quantizeRelease(at:to:)` writes a quantized copy (4-bit language model including its
  untied input embedding, 8-bit DiT — the split is measured: the flow field is the
  quantization-sensitive stage) that the same factory takes unchanged at **7.7 GiB**, small enough
  that the backend keeps every stage loaded between runs instead of staging them from disk per
  request. Fallback precision, if a smaller stack matters more than the last of the quality:
  `transformerBits: 6` trades the DiT down to a measured velocity cosine of 0.99844 (0.99990 at
  8-bit) for roughly 0.6 GB more — do a listening A/B first, because the DiT's error compounds over
  the sampling loop.
- **`NFKMLXHybridLanguage`** — the Qwen3.5 / 3.6 / 3.8 decoder: a gated delta-rule recurrence in three
  of every four layers (a fixed-size state instead of a growing cache) with gated full attention in the
  fourth. At reference parity on the released Qwen3.5-4B, layer by layer; the 27B is accounted for by
  shape against its checkpoint headers.
- **`NFKMLXDeepSeek`** — the DeepSeek V4 decoder: multi-head latent attention over a mixture of
  experts, hyper-connections, the compressor and sparse indexer, and the release's fp8 / fp4 block-scaled
  storage decoded exactly. The arithmetic is measured against transformers at a tiny configuration; the
  released weights exceed a workstation, so the checkpoint is verified structurally.
- **`NFKMLXLanguage.backend(ggufURL:)`** — text generation straight from a dense `llama` / `qwen2` /
  `qwen3` GGUF file through the native `NFKMLXGGUF` reader, undoing llama.cpp's rotary permutation and
  rebuilding the embedded tokenizer; at parity against transformers loading the same file.
- **Generation runtime** — a prompt cache kept between turns (`NFKMLXPromptCache`, persistable),
  speculative decoding with a draft model (`backend(directoryURL:draftDirectoryURL:)`, greedy-exact),
  key-value cache quantization and a bounded context window, chunked prefill, JSON, JSON-Schema, and
  fixed-choice constrained decoding (`NFKMLXJSONConstraint`, `NFKMLXJSONSchemaConstraint`,
  `NFKMLXChoiceConstraint`), Qwen3-MoE, Qwen2-MoE, Mixtral, and gpt-oss
  mixtures, runtime 4- / 8-bit quantization with a checkpoint contract that reloads packed weights onto
  matching structure, and `NFKMLXModelSizing`, which answers whether a release fits the machine — and
  at what context window — before any weight loads. Every option reaches Objective-C through
  `NFKMLXGenerationParameterKey`.
- **`NFKMLXT5Encoder`** / **`NFKMLXGemma2Net`** — the text encoders the diffusion pipelines condition on:
  T5 v1.1 and umT5 (`perLayerBias`) for LTX and Wan, Gemma 2 for SANA. Each at reference parity.
- **`NFKMLXSigLIP2`** — real image + text embeddings (SigLIP 2, base-patch16-224 at parity, and every other fixed-resolution release — base / large / so400m / giant-opt — held to its safetensors headers; `NFKMLXSigLIP2Variant`): the SigLIP encoder
  with an attention-pooling head and a 256k-vocabulary multilingual text tower, sigmoid similarity.
  `siglip2-base-patch16-224`; parity ~1e-12 on both towers.
- **`NFKMLXRTDetr`** — real object detection under Apache-2.0 (RT-DETR r18vd / r34vd / r50vd / r101vd, `NFKMLXRTDetrVariant`): a ResNet-D backbone (basic blocks below r50), a
  hybrid encoder, query selection, and a deformable-attention decoder with box refinement; no non-max
  suppression, since the one-to-one training makes the queries distinct. `rtdetr`; at parity on the
  released weights end to end.
- **`NFKMLXRFDetr`** — real object detection under Apache-2.0 (RF-DETR nano / small / medium / base / large, Roboflow; `NFKMLXRFDetrVariant`): a windowed
  DINOv2 backbone, a C2f / RepVGG projector, two-stage Group-DETR query selection, and an LW-DETR
  deformable decoder; no non-max suppression. `rf-detr`; at parity on the released weights end to end.
  `loadWeights` converts the original Roboflow naming on device (and splits the fused self-attention
  projection), so the released file loads directly.
- **`NFKMLXRetinaFace`** — real face detection with five-point landmarks (mobile0.25, the detector the
  CodeFormer reference pipeline uses); `retinaface-mobile025`. `NFKMLXPhotoFaceBackend` restores every
  face in a photograph through `NFKMLXFaceAlignment` (RetinaFace or a weight-free Vision detector) and
  CodeFormer, compositing each back with a feathered edge.
- **`NFKMLXTAESD`** — the tiny Stable Diffusion autoencoder (`taesd`), a fast preview encode / decode;
  the `[Module]`-array modeling loads the flat `nn.Sequential` release with no remap.
- **`NFKMLXIPAdapterImageProjection`** / **`NFKMLXIPAdapterAttention`** — IP-Adapter image conditioning
  for a diffusion model: a CLIP image embedding becomes a few tokens read through a second,
  image-conditioned cross-attention beside the text one. Both at parity against diffusers.
- **`NFKMLXZImagePipeline`** — Z-Image text-to-image and image-to-image: the single-stream S3-DiT
  (`NFKMLXZImageTransformerNet`, at parity), the Flux VAE (a preset of the shared `NFKMLXSDAutoencoder`),
  and a Qwen3-4B caption embedding read from the shipped decoder's penultimate layer, over the flow
  scheduler.
- **`NFKMLXSANAPipeline`** — SANA text-to-image: the ReLU linear-attention DiT (`NFKMLXSANATransformerNet`),
  the 32× Deep-Compression Autoencoder (`NFKMLXDCAutoencoderNet`, at parity on the released `Sana_600M`
  VAE), a Gemma 2 caption, and the released DPM-Solver++ sampler (`NFKMLXDPMSolverScheduler`).
- **`NFKMLXLTXPipeline`** — LTX-Video text-to-video: a causal 3-D VAE (`NFKMLXLTXVideoVAE`), the 2B
  DiT (`NFKMLXLTXTransformer`, 3-D rotary + adaLN + cross-attention), a T5-XXL prompt, and the
  rectified-flow sampler (`NFKMLXFlowMatchScheduler`, exact against diffusers). Every stage at parity;
  the caller stages the 19 GB encoder and the 7.7 GB DiT in turn.
- **`NFKMLXWanPipeline`** — Wan text-to-video: the Wan DiT (`NFKMLXWanTransformerNet`), the streaming
  3-D causal VAE with its per-convolution feature cache (`NFKMLXWanVideoVAENet`, the 2.1 and 2.2 paths),
  a umT5 prompt, and the released UniPC sampler (`NFKMLXUniPCScheduler`).
- **`NFKMLXSAM2`** — SAM 2's Hiera image encoder (tiny, small, base_plus, large), prompt encoder, mask decoder,
  and the video memory encoder and memory attention, each at parity against facebookresearch's sources.
- **`NFKMLXDemucs`** / **`NFKMLXHTDemucs`** — real four-stem music separation: the Demucs v2 time-domain
  U-Net (`demucs`) and the v4 Hybrid Transformer Demucs (`htdemucs`; the six-stem `htdemucs-6s`; the
  fine-tuned four-checkpoint bag through `backend(fineTunedWeightsURLs:)`), a spectrogram branch and a
  waveform branch joined by a cross-transformer; all at parity on the released weights.
- **`NFKMLXWhisper`** — real speech-to-text: the Whisper encoder-decoder at every released size (tiny, base, small, medium, large-v1/v2, large-v3, large-v3-turbo)
  with the reference's suppression rules and timestamped decoding (`emitsTimestamps` → segments);
  `whisper-tiny`; exact token matches against openai-whisper. Also the core's `transcription`
  capability through `NFKMLXWhisperProvider`.
- **`NFKMLXSileroVAD`** — real voice-activity detection (Silero VAD v6): a learned STFT, four
  convolutions, and an LSTM that streams chunk by chunk; `silero-vad`; threshold agreement 32/32 against
  the released JIT.
- **`NFKMLXMPSENet`** / **`NFKMLXGTCRN`** / **`NFKMLXSGMSE`** — speech restoration (denoising and
  dereverberation): MP-SENet (`mpsenet`), a time-frequency transformer that cleans magnitude and phase in
  parallel; GTCRN (`gtcrn`), a ~48K-parameter real-time enhancer; and SGMSE+ (`sgmse`), score-based
  generative dereverberation — a reverse-SDE predictor-corrector sampler over an NCSN++ score network.
  All three at reference parity on the released weights (SGMSE+'s net-seam cosine is 1.0 on both released
  backbone variants). The three share a complex-STFT front end and a PyTorch-`nn.GRU` weight fold.
- **`NFKMLXStoRM`** — StoRM (`storm`), a few-step follow-on on SGMSE+: a discriminative predictor produces
  an initial estimate, then a conditioned score network regenerates from it (the reverse SDE re-centered on
  the estimate), so the diffusion needs far fewer steps. Reuses SGMSE+'s NCSN++ backbone (generalized for
  the two roles); both networks at reference parity (cosine 1.0).
- **`NFKMLXMossFormer2SENet`** — MossFormer2 SE 48K (`mossformer2-se`, `alibabasglab/MossFormer2_SE_48K`,
  Apache-2.0), full-band speech enhancement: a mask-predicting MossFormer2 backbone (FLASH gated attention
  — quadratic ReLU-squared local plus a linear global path — interleaved with a `Gated_FSMN` depthwise
  memory, over a Kaldi-fbank + Δ + ΔΔ front end) produces a 961-bin magnitude mask applied to the STFT.
  At reference parity on the released weights (M1, float32): the fbank 1.0, the encoder and FLASH block 0
  0.99999994, FLASH block last and the mask 1.0, and the enhanced waveform 0.9999998.
- **`NFKMLXMossFormer2SRNet`** — MossFormer2 SR 48K (`mossformer2-sr`, `alibabasglab/MossFormer2_SR_48K`,
  Apache-2.0), speech super-resolution: the HiFi-GAN log-mel, the same MossFormer2 backbone as a
  mel-to-mel restorer, a Snake HiFi-GAN generator, and the reference decode path's bandwidth
  substitution (the input kept below its detected bandwidth through a Butterworth low-pass, the
  generated band added above it, a 100 ms crossfade). At reference parity on the released weights:
  mel 1.0, backbone 1.0, generator 0.9999999999987, substitution 1.0, end to end 0.9999999999992.
- **`NFKMLXDeepFilterNet`** — DeepFilterNet3 (`deepfilternet3`, Rikorose/DeepFilterNet, dual MIT/Apache-2.0),
  a ~2.3M-parameter real-time 48 kHz denoiser (the cheap counterpart to the Demucs speech denoiser): an
  ERB mask over the full spectrum plus a 5-tap causal complex deep filter on the lowest 96 bins, from a
  `SqueezedGRU_S` encoder / ERB decoder / DF decoder. Its STFT / ERB / normalization DSP (Rust `libdf` in
  the reference) is reproduced in MLX + Swift. At reference parity on the released weights: every net seam
  and the DSP features exact (1.0000000) and the enhanced waveform 0.9999999.
- **`NFKMLXVoiceRestore`** — VoiceRestore (`voicerestore`, skirdey/voicerestore, MIT), a ~301M-parameter
  flow-matching universal speech restorer (noise + reverb + clipping + band-limiting in one model, text-
  free). An E2-TTS transformer (a `SimpleGateLoopLayer` gated-linear recurrence + adaLN attention/FFN over
  32 register tokens) predicts the CFM velocity; a midpoint ODE sampler restores the mel; `NFKMLXBigVGAN`
  (BigVGAN v2, MIT, SnakeBeta + anti-aliased activations) vocodes it. At reference parity on the released
  weights, seam by seam and end to end: the transformer velocity 0.99999994, the BigVGAN waveform
  0.9999997, and the restored mel / waveform ~1.0.
- **`NFKMLXResembleEnhance`** — Resemble Enhance (`resemble-enhance`, resemble-ai, MIT), a five-network
  general speech restorer (noise + reverb + clipping + band-limiting). A stage-1 STFT-mask 2-D UNet
  denoiser, a Latent Conditional Flow Matching stage (an IRMAE autoencoder compresses the mel to a
  64-channel latent, a WaveNet CFM velocity net samples it through an exponential-decay midpoint ODE),
  and a UnivNet location-variable-convolution vocoder. At reference parity on the released
  enhancer_stage2 weights, seam by seam and end to end: mel 0.9999998, IRMAE encode 0.9999985 / decode
  1.0000002, CFM velocity / sample 1.0000001, UnivNet 0.9999365, denoiser 0.9999996, and the end-to-end
  restored waveform 0.9999971.
- **`NFKMLXDAC`** / **`NFKMLXSNAC`** — neural audio codecs, the classes a codec-token speech model
  generates into: the Descript Audio Codec (`dac`, 44.1 / 24 / 16 kHz, residual vector quantization) and
  SNAC (`snac`, 24 kHz speech; `snac-32khz` / `snac-44khz` music, four codebooks and bottleneck
  attention; multi-scale codebooks at different rates). `encode` returns the tokens,
  `decode` reconstructs; both match the reference's codes exactly.
- **`NFKMLXVoice`** / **`NFKMLXFastSpeech2`** / **`NFKMLXHiFiGAN`** — a complete text-to-speech voice:
  the espnet FastSpeech2 conformer on the released LJSpeech weights (durations exact frame for frame)
  with its paired HiFi-GAN vocoder, exposed through `makeSpeechBackend(phonemize:)`. The package's own
  Whisper transcribes its output as "hello, world." `NFKMLXTTS` chains a phonemizer
  (`NFKMLXNeuralG2P`, or a system espeak-ng), an acoustic model, and a vocoder by hand.
- **`NFKMLXKokoro`** — Kokoro-82M (StyleTTS2 / iSTFTNet), a second text-to-speech voice: a PL-BERT
  phoneme encoder, duration / F0 / energy predictors, and an iSTFTNet decoder with a harmonic sine
  source. `backend(directoryURL:voiceName:)` takes a phoneme string under `NFKInputPrompt`; every
  deterministic seam at parity, the waveform at 0.997.
- **`NFKMLXMetricGANPlus`** — MetricGAN+ (`metricgan-plus`, speechbrain/metricgan-plus-voicebank,
  Apache-2.0), the smallest restoration model: a two-layer bidirectional LSTM magnitude mask over
  `log1p(|X|)` frames of a 512-point zero-padded Hamming STFT, with a per-bin learnable sigmoid, the
  noisy phase, and a peak normalization. At reference parity on the released weights against
  speechbrain's own `SpectralMaskEnhancement`: features 1.0, mask 1.0000001, enhanced waveform
  1.0000001. The shared complex STFT gained speechbrain's zero padding and `torch.istft`'s `length` for it.
- **`NFKMLXCMGAN`** — CMGAN (`cmgan`, ruizhecao96/CMGAN, MIT), a 1.83M-parameter conformer metric GAN:
  a dense encoder, four two-stage (time, frequency) conformer blocks with Shaw's relative position
  embedding, and magnitude-mask + complex-residual decoders over a power-compressed spectrogram, reusing
  the MP-SENet dense and sub-pixel blocks. At reference parity on the released weights against the
  repository's own generator: every seam ≥ 0.9999998 and the enhanced waveform 0.99999994.
- **`NFKMLXFRCRN`** — FRCRN SE 16K (`frcrn`, alibabasglab/FRCRN_SE_16K, Apache-2.0), a
  frequency-recurrent complex CRN: two complex UNets over a conv-STFT with a frequency-recurrent FSMN
  memory before each encoder, a complex squeeze-excite after each, and a time FSMN at the bottleneck;
  the mask `tanh(unet2) + tanh(unet1)` is a complex product with the spectrum. At reference parity on
  the released weights against ClearerVoice's own DCCRN: every seam ≥ 0.99999994 and the enhanced
  waveform 1.0, through the reference decode path's own padding rule.
- **`NFKMLXNUWave2`** — NU-Wave 2 (`nuwave2`, maum-ai, BSD-3), diffusion bandwidth extension: a
  noise predictor of short-time Fourier convolutions (an STFT-domain 1×1 convolution modulated per bin
  by the input's bandwidth beside a local convolution branch), sampled by the released eight-step logSNR
  DDIM from seeded noise. At reference parity on the official checkpoint from the reference's own start
  noise: every seam and every step 1.0.
- **`NFKMLXApollo`** — Apollo (`apollo`, JusperLee, **CC-BY-SA-4.0** code and weights), music
  restoration of lossy-codec artifacts at 44.1 kHz: an 80-band split of a 20 ms STFT, six band-sequence
  layers (a rotary transformer across the bands, a depthwise convolutional block along time), and a
  gated head per band. At reference parity on the released weights against the repository's own model:
  band features 0.9999988, every band-sequence layer 1.0, the restored waveform 0.9999973.
- **`NFKMLXChatterbox`** — Chatterbox (Resemble AI, MIT), zero-shot voice cloning: a VoiceEncoder speaker
  embedding and the S3 speech tokenizer read a reference voice, a T3 Llama (llama3 rope scaling, a
  Perceiver-resampled prompt) samples speech codes for the text, and S3Gen renders them through a
  conditional flow-matching decoder and the HiFT vocoder at 24 kHz. `speechBackend(directoryURL:voiceURL:)`
  takes text under `NFKInputPrompt`; nil voice speaks the release's built-in `conds.pt`. Every stage at
  reference parity on the released weights; the synthesized validation sentence transcribes back
  through Parakeet exactly.
- **`NFKMLXParakeet`** — Parakeet-TDT 0.6B v2 (NVIDIA, CC-BY-4.0), a second speech recognizer beside
  Whisper: a FastConformer encoder and a token-and-duration transducer, greedy TDT decoding, a
  timestamp per token under `NFKOutputSegments`. `backend(directoryURL:)` reads an unpacked `.nemo`;
  at reference parity against NeMo (tokens and timestamps exact).
- **`NFKMLXVideoBackend`** — the first backend that produces video: an `NFKVideoAsset` in, every frame
  through a whole-sequence transform, a new clip out through `NFKMLXVideoFile` (AVFoundation).
  `NFKMLXRIFE.clipBackend` doubles a clip's frame rate and `NFKMLXVideoSR.clipBackend` upscales one.
- **Customizing a model on device** — `NFKMLXTrainer` runs supervised and zero-reference fine-tuning
  with clipping, checkpoints, and early stop; `NFKMLXLoRA` adapts attention blocks and merges the
  result back into plain weights; `NFKMLXCLIPProbe` trains a classifier over frozen CLIP embeddings;
  recipes ship for Zero-DCE, SegFormer's decode head, and Whisper, each with its loss at reference
  parity. A fine-tuned file loads through the model's ordinary `weightsURL:` factory.
- **`NFKMLXDiffusionBackend`** — a bring-your-own MLX diffusion model, for the iterative-sampler shape
  the single-forward backends cannot express. Supply `encode`, `denoise`, `decode`, and a scheduler;
  the backend runs the denoise loop with per-step progress and cancellation. No source latent runs
  text-to-image, a source latent runs image-to-image (`NFKParameterStrength`), and a source latent
  plus a mask runs inpainting. Reference pipelines register by name (upscale, depth, inpaint,
  **controlnet**); `NFKDDIMScheduler` and `NFKLCMScheduler` (few-step) ship, and `NFKDiffusionScheduler`
  is the seam for other samplers. **ControlNet and LCM need no full SD reimplementation** — LCM is a
  scheduler swap, ControlNet is a `denoise` closure over a control map (`NFKInputControl` →
  `conditioning["control"]`); the UNet is supplied by your `denoise` or a dynamically linked SD engine.

The image backends share `NFKMLXImageBridge`, which converts a **`CGImage` or an `MTLTexture`**
to and from `MLXArray` in either direction, preserving alpha — so a Metal render pipeline hands
textures straight in and gets textures back.

### Dynamic backend discovery (optional engines)

`NFKDynamicBackend` (core) activates a heavier engine **only when its classes are linked into your
build**, with no build dependency on it. Your engine's adapter conforms to `NFKDynamicBackendProvider`
and InferKit resolves it by name at runtime (`NSClassFromString`) — absent when unlinked, with no link
error. Built-in capabilities light up when you link a companion:

- **`stable-diffusion`** — linking **InferKitMLX** ships `NFKStableDiffusionProvider`, so
  `NFKDynamicBackend.stableDiffusionBackend()` returns the bundled SD backend.
- **`transcription`** — InferKitMLX also ships `NFKMLXWhisperProvider` (a native whisper.cpp can override).
- **`text-generation`** — linking **InferKitFoundationModels** ships `NFKFoundationModelsProvider` for
  on-device LLM.
- **`controlnet`** — no shipped default; bring a ControlNet engine and register its provider.

Without the companion, the capability is simply unavailable. Model **weights are downloaded at runtime**
(not bundled at build time) and cached under Application Support (`NFKHFHub.defaultCacheDirectoryURL`, or
a host-supplied security-scoped folder); the download blocks, so run it off the main thread or use the
async `downloadRepo:…completionHandler:` (`try await`).

### Model weight licenses

InferKit's code is MIT and the checkpoints the shipped models load are, with one exception, released
under MIT or Apache-2.0 terms. Weights are a separately licensed asset your app downloads at runtime;
the license travels with the checkpoint, not with InferKit.

The exception is **MiniMax Music 3** (`NFKMLXMusicBackend`). Its weights
(`MiniMaxAI/MiniMax-Music3` on Hugging Face) are under the **MiniMax-Music3 Community License**, which
is not a permissive license:

- A commercial product or service using the weights must prominently display "MiniMax-Music3" in its
  user interface (License §3.1).
- Aggregate yearly revenue above USD 20 million from such products requires separate prior written
  authorization from MiniMax (License §3.2, `api@minimax.io`).
- A product or hosted service that lets third parties generate outputs must implement and maintain
  safeguards against infringing uses and outputs (License §4).

The model's inference code carries no such terms (the architecture derives from Qwen3-8B, Apache-2.0;
Stable Audio tools, MIT; and the Descript Audio Codec, MIT). The obligations attach to the checkpoint.
An app that ships the MiniMax music backend accepts these terms on behalf of its own product; review
the LICENSE file in the weight repository before enabling it commercially.
