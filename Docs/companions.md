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
  parameters including top-k / top-p / seed, `NFKParameterJSONSchema` and `NFKParameterChoices` for
  structured output, `NFKParameterTools` with handlers registered as `NFKFoundationTool`s, and
  streamed partial text through the job. On macOS 27 / iOS 27 it also reads `NFKInputImage` and
  `NFKParameterReasoningEffort`, and answers with `NFKOutputReasoning` and `NFKOutputUsage`.
  `isReady` reflects the model's availability. `model`
  picks the on-device system model (specialized by `useCase` and `guardrails`) or Apple's larger
  model on Private Cloud Compute (macOS 27 / iOS 27), whose quota `privateCloudComputeQuota` reads.

The reverse direction ships too. `NFKInferKitLanguageModel` adopts Apple's provider protocols
(`LanguageModel` / `LanguageModelExecutor`), so an InferKit backend stands behind
`LanguageModelSession`: `LanguageModelSession(model: NFKInferKitLanguageModel(backend: backend))`
runs a remote endpoint or a converted Core ML model through Apple's session API. The session's
transcript becomes `NFKInputMessages`, its tool definitions `NFKParameterTools`, its schema
`NFKParameterJSONSchema`, its reasoning level `NFKParameterReasoningEffort`, and its generation
options the core's sampling keys; the backend's `NFKOutputReasoning` and `NFKOutputUsage` come back
on the session's own channels. The model reports what the backend declares through the protocol's
`supportedParameterKeys` and `supportedInputKeys`.
It needs macOS 27 / iOS 27 and a build with the macOS 27 SDK; the package floor stays at 26.

## InferKitAppleSwift (optional companion)

`InferKitAppleSwift/` is a separate SwiftPM package (macOS 26 / iOS 26) holding the Apple inference
APIs that ship in Swift alone. The core wraps every Apple framework an Objective-C target can call;
four APIs it cannot, because `SpeechAnalyzer` is an actor whose results arrive as an
`AsyncSequence`, Vision's `RecognizeDocumentsRequest` and `DetectLensSmudgeRequest` live in Vision's
Swift module with no `VN*` header, and the Translation framework is Swift-only throughout.

- `NFKVisionDocumentBackend` turns a photographed page into a transcript under `NFKOutputText` and
  its structure under `NFKOutputStructured`: paragraphs, lists, and tables as rows of cells.
- `NFKVisionSmudgeBackend` judges whether the lens was dirty, as one classification labeled `smudge`.
- `NFKSpeechAnalyzerBackend` transcribes on Apple's newer speech stack, with a segment per reported
  range. `prepare()` reserves the locale and installs its assets; `isReady` answers from what it
  found, because the system reports installed locales asynchronously.

- `NFKTranslationBackend` translates on device. Text arrives under `NFKInputPrompt`,
  `NFKParameterTargetLanguage` names the language to translate into, `NFKParameterSourceLanguage` is
  optional because Apple detects it, and the translation comes back under `NFKOutputText`. A pair the
  system will not translate reports `kNFKError_InferenceUnsupported`; a pair whose model is not
  installed reports `kNFKError_InferenceNotReady`. Every wait on the framework is bounded by
  `responseTimeout`, because it does not always answer.

Every type is `@objc`, which is the package's purpose. Linking it puts `NFKSpeechAnalyzerProvider`
ahead of the core's own recognizer for `NFKCapabilityTranscription`, and `NFKTranslationProvider`
behind any MLX translator for `NFKCapabilityTranslation`.

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
- **`NFKMLXMambaBackend`** — on-device text generation for Codestral-Mamba
  (`NFKMLXMamba.backend(directoryURL:)` / `mambaBackendWithDirectoryURL:error:`), the toolkit's first
  state-space model. Every layer is a Mamba-2 selective scan (SSD) instead of attention, carrying a
  fixed-size state rather than a growing key-value cache, and it runs prefill-only. The tokenizer is the
  Mistral byte-fallback BPE (`NFKMLXMistralTokenizer`) read from the release's `tokenizer.json`. At
  reference parity against transformers' `Mamba2ForCausalLM`: the released Codestral-Mamba-7B matches by
  shape across all 579 tensors and, at bfloat16, reproduces the prefill logits (cosine 0.9999146) and
  the greedy continuation token for token. One selective-scan mixer serves the SSM class, so the hybrid
  Mamba-attention decoders (Granite 4.0-H, Nemotron Nano 2) build on it.
- **`NFKMLXGraniteBackend`** — on-device text generation for Granite 4.0-H
  (`NFKMLXGraniteHybrid.backend(directoryURL:)` / `graniteBackendWithDirectoryURL:error:`), IBM's hybrid
  decoder. Most layers are the Mamba-2 selective scan reused from Codestral; the few its `layer_types` names
  are grouped-query attention with no positional embedding, and Granite's scalar multipliers scale the embedding, each
  residual, the attention logits, and the output logits. The dense sizes (h-350m, h-1b) carry a
  gated-linear shared MLP; the MoE sizes (h-tiny, h-small) add a routed mixture of experts beside it. The
  tokenizer is Granite's byte-level BPE, read from the release's `tokenizer.json`, and it runs
  prefill-only. At reference parity against transformers' `GraniteMoeHybridForCausalLM`: the released
  granite-4.0-h-1b matches by shape across all 466 tensors and reproduces the float32 prefill logits and
  greedy continuation, and the routed mixture is measured at a tiny configuration. Adapts to a
  consumer's own text with LoRA on the attention projections through `NFKMLXGraniteHybrid.fineTune`.
- **`NFKMLXNemotronBackend`** — on-device text generation for Nemotron Nano 2
  (`NFKMLXNemotronH.backend(directoryURL:)` / `nemotronBackendWithDirectoryURL:error:`), NVIDIA's hybrid
  decoder. Its 56 layers interleave three mixers from a `hybrid_override_pattern`: the Mamba-2 selective
  scan reused from Codestral, a ReLU-squared dense feed-forward, and grouped-query attention with no
  positional embedding at the standard scale. There are no scalar multipliers, each block carries a
  single pre-norm, and the output projection is untied. The Mamba mixer's gated norm is grouped by
  `n_groups`, the one numeric difference from the Codestral/Granite mixer. The tokenizer is Nemotron's
  byte-level BPE, read from the release's `tokenizer.json`, and it runs prefill-only. At reference parity
  against transformers' `NemotronHForCausalLM`: the tiny config reaches logit cosine 1.0, and the
  released Nemotron-Nano-9B-v2 matches by shape across all 341 tensors (its `backbone.*` naming remapped
  to `model.*`). Adapts to a consumer's own text with LoRA on the attention projections through
  `NFKMLXNemotronH.fineTune`.
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
- **`NFKMLXLaya`** — Laya (`convaiinnovations/laya`, Apache-2.0), the open reproduction of TypeSafe's
  Jev: a typed-decision model that answers a choice, a score, or a noul about a state in one
  bidirectional pass, without generating text. It takes the core's `NFKDecisionQuestion`s and returns
  `NFKDecisionAnswer`s, the objects `NFKTypeSafeBackend` returns, so a feature moves between the hosted
  model and the device by swapping the object; `NFKMLXLayaBackend` answers the same request through the
  contract. The network is the ModernBERT encoder plus a decision head (a type embedding, two pre-norm
  transformer layers, a marker scorer, an act-or-escalate head); every option is scored at its own mask
  token. Three variants: the root (ModernBERT-large, 421M), `typed-decisions` (fine-tuned on that
  benchmark), and `multilingual` (mmBERT-base, Gemma's tokenizer, 100-plus languages). Reference parity
  against the release's own inference code on all three, the prompt token for token. Customization is a
  head fine-tune (or the encoder too) with the reference's proper-scoring objective, reloaded through
  `layaWithDirectoryURL:weightsURL:error:`. `layaWithVariant:revision:cacheDirectoryURL:error:` (and its
  completion-handler form) downloads one variant's five files through the `NFKHFHub` cache and builds it;
  `NFKMLXLaya.measuredRevision` pins the download to the commit parity was measured at.
- **`NFKMLXOpenJevDeBERTa`** — open-jev-deberta-v3-large (`com-kotobalabs/open-jev-deberta-v3-large`,
  Apache-2.0), a community reproduction of Jev on DeBERTa-v3-large (`NFKMLXDeBERTaV2Net`, the package's
  disentangled-attention encoder): one pass reads `[CLS] [STATE] state ([Q] question ([OPT] option)*)*
  [SEP]`, and a head scores each option from the mean of its text, the mean of its question's text, and
  their product. Reference parity against the release's bundled `typed_decisions` code (every encoder
  layer, the logits, a padded batch, the SentencePiece tokenizer token for token). Customization is a head
  or full fine-tune on the release's cross-entropy-plus-Brier objective, measured against its
  `decision_loss`.
- **`NFKMLXOpenJev`** — Open-Jev 2B / 9B / 27B (`ZefanCai/Open-Jev-2B`, `-9B`, `-27B-v1.1`; weights
  Apache-2.0, loader MIT), a rank-8 LoRA adapter and a scalar head over a Qwen3.5-architecture text model
  (`NFKMLXHybridLanguageNet`: Qwen3.5-2B, Qwen3.5-9B, Qwen3.8-27B). Each
  candidate answer is its own chat-templated Yes/No prompt; the head reads the last token. The factory
  downloads the adapter and the exact base revision it names. Reference parity on the 2B release against
  the loader's own `DecisionModel` at float32 (every candidate's tokens, every hidden state, the logits);
  the 9B release agrees with the loader at bfloat16, both sides' own precision; the 27B release is
  checked structurally against its base's shard headers.
  Customization trains the adapter and head at the release's rates and saves in its own layout.
  `NFKMLXDecisionBackend` puts either behind the request `NFKTypeSafeBackend` reads.
- **`NFKMLXChronos`** — Chronos-Bolt (`amazon/chronos-bolt-base`, Amazon, Apache-2.0), a time-series
  forecaster: a patched T5 encoder-decoder that standardizes a numeric context window, patchifies it into
  16-sample patches, runs an encoder and a single-token decoder, and emits a quantile forecast over a
  horizon. Not a backend (a numeric series has no core key); `forecast(context:horizon:)` returns one row
  per quantile level (0.1 … 0.9) and `medianForecastForContext:horizon:` the point forecast. At reference
  parity against the `chronos` package's own `ChronosBoltPipeline`, every seam ~1.0 and all nine quantile
  rows matching.
- **`NFKMLXQwen3VLEmbedder`** and **`NFKMLXQwen3VLReranker`** — multimodal retrieval
  (`Qwen3-VL-Embedding-2B`, `Qwen3-VL-Reranker-2B`): a text, an image, or both embed into one space,
  and a reranker reads a query and a document together whichever of them carries the image. An
  instruction conditions both, so the same corpus is searched differently under a different task
  description. Both are the Qwen3-VL backbone the package already runs, pooled at the last position:
  the embedder normalizes it, and the reranker reads its preference for "yes" over "no" through a
  sigmoid. `embeddingForText:` / `embeddingForImage:text:instruction:` and
  `scoresForQuery:documents:` / `rankedIndicesForQuery:documents:` are the entry points. Reference
  parity against each release's own script on the released 2B weights: prompt ids exact, text
  embedding cosine 0.9999999999866735, image embedding 0.9999999999305262, and the reranker's scores
  to 2.4e-6. A consumer customizes either on their own corpus by training a small probe over the
  frozen backbone, against the objectives sentence-transformers trains these releases with.
- **`NFKMLXSmolVLM`** — a vision-language model (SmolVLM2-500M): an image and a question in, an answer
  out. A SigLIP vision encoder turns each image tile into patch features, a pixel-shuffle connector
  projects them to the decoder width, and a Llama decoder (the dense stack the language model runs)
  reads the text with the projected vision tokens spliced in at the image-token positions.
  `answerForImage:question:` captions or answers about a `CGImage`. The geometry comes from the
  release's own `config.json`, so all three sizes run the same code (256M, 500M, 2.2B). Reference parity
  against transformers' own SmolVLMForConditionalGeneration (the vision encoder, the connector, and the fused
  decoder logits exact, the greedy continuation token for token); the prompt expansion is token-exact
  and the image processor is CoreGraphics-based (a slight approximation of the reference's PIL resize).
- **`NFKMLXQwen3VLVisionNet`** — the vision tower of a second VLM, Qwen3-VL-2B: a 2D-rotary ViT (patches
  in 2×2 merge blocks, a bilinearly interpolated position embedding), a merger that folds each block to
  the decoder width, and a three-layer "deepstack" of feature maps. At reference parity against
  transformers' own Qwen3-VL vision model (patch embedding, position embedding, merged output, and every
  deepstack feature exact). The decoder is the Qwen3 dense stack; its Qwen3-VL-specific M-RoPE and
  deepstack injection are the remaining integration.
- **`NFKMLXPixtral`** — Pixtral 12B (`mistral-experimental/pixtral-12b`, Mistral, Apache-2.0), a third VLM
  and a third vision architecture: a from-scratch 2D-rotary vision tower (`NFKMLXPixtralVisionNet`) with
  native variable resolution, a two-layer GELU connector (`NFKMLXPixtralConnector`), and a Mistral-Nemo
  dense decoder the package already runs, the projected patch features splicing in at the `[IMG]`
  positions. At reference parity against transformers' own `LlavaForConditionalGeneration`: the vision
  tower and connector measured in float32 seam by seam, and the whole fused pipeline against a tiny
  float32 oracle with the reference's argmax at every position. The 12B decoder's released-weight fused
  pass needs ~24 GB resident, so it runs behind an opt-in. Customization is offline-only, the decoder
  being a language model above 4B.
- **`NFKMLXFlorence2`** — Florence-2 (`microsoft/Florence-2-base` and `-large` and their fine-tuned `-ft` releases, Microsoft, MIT; the geometry
  read from the release's `config.json`), a unified
  vision model that reads one image and a task token and writes text: a caption, detected objects, or
  grounded regions, with boxes carried as location tokens. The vision encoder is DaViT
  (`NFKMLXFlorence2VisionNet`), whose every block pairs a windowed spatial attention with a grouped
  channel attention; a projector (`NFKMLXFlorence2Projector`) turns the vision grid into tokens that
  concatenate before the prompt for a BART encoder-decoder (the shared `NFKMLXSeq2SeqTransformer`). At
  reference parity against the repo's own implementation seam by seam (the DaViT tower, the projector,
  the encoder, the first-step logits), and token for token in the release's own generation (three
  beams, no repeated 3-gram) on captioning, detailed captioning, OCR, and detection, on all four releases.
  The backend detects objects end to end. The oracle is the repo's `trust_remote_code` code, since the
  native transformers integration does not load the released checkpoint. Adapts to your own task on the
  device with LoRA on the BART decoder, scored by the release's own `labels=` loss:
  `NFKMLXFlorence2.network(directoryURL:)`, `fineTune`, `NFKMLXLoRA.merge(into:)`, then
  `save(_:toDirectoryURL:release:)`, which the same factory loads.
- **`NFKMLXTrOCR`** — TrOCR (`microsoft/trocr-base-handwritten`, Microsoft, MIT), a handwriting-line
  reader that turns one image into its transcription. A `VisionEncoderDecoder`: a plain `google/vit`
  image encoder (`NFKMLXTrOCRVisionNet`) whose patch tokens are the memory for a BART-style decoder (the
  shared `NFKMLXSeq2SeqTransformer` in its decoder-only shape, cross-attending the 768-wide features
  under a 1024-wide decoder). At reference parity against transformers' own `VisionEncoderDecoderModel`
  seam by seam (the embeddings, the first block, the encoder output, the first-step logits) and token
  for token in greedy generation; the backend reads a rendered line end to end. The decoder is entirely
  the shared seq2seq (which grew a cross-attention width and a decoder-only shape for it); the new work
  is the ViT encoder. Every release (small, base, and large; handwritten, printed, scene text, and the
  stage-1 pretrained ones) loads from its directory. Fine-tunes on your own lines on the device the way
  the authors fine-tuned it: `NFKMLXTrOCR.network(directoryURL:)`, `fineTune`, then
  `save(_:toDirectoryURL:release:)`, which the same factory loads.
- **`NFKMLXSa2VA`** — Sa2VA (ByteDance, Apache-2.0), a segmentation VLM: one image and a referring
  prompt in, an answer out, and a mask for each `[SEG]` the answer carries. One directory factory,
  `NFKMLXSa2VA.backend(directoryURL:)`, serves the four families the releases span: InternVL
  (`Sa2VA-1B`/`-4B`/`-8B`/`-26B`, `Sa2VA-InternVL3-2B`/`-8B`/`-14B`; InternViT-300M or -6B under a qwen2,
  phi3, or InternLM2 decoder), Qwen-VL (`Sa2VA-Qwen3-VL-2B`/`-4B`, `Sa2VA-Qwen2_5-VL-3B`/`-7B`), LLaVA-1.5
  (`Sa2VA-LLaVA-1.5-7B`), and SAM 3 grounding (`Sa2VA-Qwen3-VL-4B-SAM3`); the rest ground with SAM 2's
  tracker. Each release's own chat template, end token, and tokenizer (InternLM2's SentencePiece included)
  are read from its directory. At reference parity at float32 against the repos' own code, every seam
  within about 1e-6 of 1 and the mask at IoU 1.0, with token-exact generation and the backend returning
  the reference's decoded answer. Fine-tunes on the device with the authors' own recipe (LoRA on the
  language model, the mask decoder and `[SEG]` bridge trained, a language-plus-mask objective):
  `NFKMLXSa2VA.network(directoryURL:)` (or the Qwen-VL and LLaVA nets' `load(directoryURL:)`),
  `fineTune`, `NFKMLXLoRA.merge(into:)`, then `save(_:toDirectoryURL:release:)`, which the same factory
  loads.
- **`NFKMLXPhi4MM`** — Phi-4-multimodal (`microsoft/Phi-4-multimodal-instruct`, Microsoft, MIT), one model
  that reads text, images, and speech: a prompt or a multi-turn conversation with any number of pictures
  and clips in, an answer out, so it captions, answers questions about pictures, transcribes, and answers
  a spoken question about a picture. Greedy by default, sampled on request. A Phi-4-mini decoder (the shared `NFKMLXLanguageNet`, with partial rotary
  and LongRoPE) is fed by a SigLIP image tower laid out in Phi-3.5's dynamic-HD crops and by a
  Conformer speech tower, and the release's per-modality LoRAs ride on the decoder as a mixture, one
  adapter active per request. At reference parity against the release's own code in all four of its
  modes, with token-exact answers from raw inputs; both preprocessors match the reference processor
  (the image pixels exactly), including its handling of 44.1, 48, 8, and 11.025 kHz audio; a multi-turn
  conversation over two pictures and two clips, and a clip past 40 seconds, answer as the reference does.
  Loads from a release directory or the hub, at the released bfloat16 or float32; offline-only to
  customize (5.6 billion parameters).
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
  `backendWithVariant:weightsURL:error:`, or download and build via the `repo:` factory. The two later
  releases run the compact generator instead (`SRVGGNetCompact`: a flat body of convolutions with
  per-channel PReLU, one pixel shuffle, and a nearest-neighbor residual), reached through the same
  variant enum as `.generalX4V3` and `.animeVideoV3`.
- **`NFKMLXDepthAnything`** — a real single-forward depth model: the Depth Anything V2 DINOv2 + DPT
  network in MLXNN, run through `NFKMLXModuleBackend` (image → grayscale depth). Register and build by
  name; a self-validating converter turns the release into a safetensors checkpoint.
- **`NFKMLXDepthAnything3`** — Depth Anything 3 monocular depth and camera estimation (DA3-SMALL, -BASE,
  and -LARGE): a DINOv2 ViT variant (2D rotary, query/key norm, a camera token, and `cat_token`
  local/global hooking from block 4) plus both branches of the DualDPT head, the camera decoder, and the
  camera encoder, in MLXNN. The depth branch runs through `NFKMLXModuleBackend` (image → grayscale
  depth); `NFKMLXDepth3Estimator` returns the predicted camera as an `NFKMLXDepth3Camera` (translation,
  rotation, focal lengths, fields of view), takes a known camera through the camera encoder instead, and
  hands Swift callers the six-channel Plücker ray map beside its confidence. At reference parity against
  the authors' `depth_anything_3` package, with every released tensor loaded; the released safetensors
  loads directly (`depth-anything-3-small` / `-base` / `-large`, `NFKMLXDepth3Variant`).
- **`NFKMLXU2Net`** — a real single-forward background remover: the U²-Net nested-U saliency network
  in MLXNN, run through the matting backend (plate → foreground + alpha cutout). Full `u2net` + light `u2netp`.
- **`NFKMLXBiRefNet`** — high-resolution background removal (MIT): a Swin-v1-L backbone, a neck that
  concatenates a downscaled second view with a context stack, and a decoder whose `ASPPDeformable`
  blocks run a modulated deformable convolution. Run through the matting backend at 1024, at reference
  parity against the released weights on every seam.
- **`NFKMLXISNet`** — IS-Net, the dichotomous segmentation network the U²-Net authors published next:
  the same Residual U-blocks behind a stride-2 stem, wider stages, and six separate side maps with no
  fusion convolution, run through the matting backend (plate → foreground + alpha cutout) at
  reference parity against the DIS `isnet.py`.
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
- **`NFKMLXAdaIN`** — arbitrary style transfer: a normalized VGG-19 encodes the content and the style
  image, adaptive instance normalization moves the content features onto the style's per-channel
  statistics, and a mirrored decoder inverts the result. One pair of networks handles any style, where
  fast style transfer bakes one style per checkpoint. The content image is `NFKInputImage`, the style
  image is `NFKInputControl`, and `NFKParameterStrength` blends the two. At reference parity against
  naoto0804's `pytorch-AdaIN`.
- **`NFKMLXCLIP`** — real image+text embeddings (CLIP ViT-B/32, B/16, L/14, L/14@336; `NFKMLXCLIPVariant`): a ViT image tower and a text
  transformer projected into a shared space. `NFKMLXCLIPBackend` returns an embedding under
  `NFKOutputEmbedding` for semantic search, tagging, and diffusion guidance. MetaCLIP is the same
  architecture on a re-curated training set, and its released towers load through the same variants at
  reference parity.
- **`NFKMLXRVM`** — real video matting (Robust Video Matting, both released encoders — MobileNetV3 and ResNet-50): an encoder, LR-ASPP, and a **recurrent
  ConvGRU decoder** that carries state across frames, run through the matting backend (single frame) or
  `NFKMLXRVMNet.forward` (video, state threaded) for background removal without a green screen.
- **`NFKMLXCodeFormer`** — real face restoration: a VQGAN encoder/generator with a Transformer that
  predicts codebook indices, run through the module backend (aligned face → restored face).
- **`NFKMLXZeroDCE`** — real low-light enhancement: the Zero-DCE DCE-Net estimates pixel-wise tone
  curves and iteratively brightens, run through the module backend (dark image → brightened image).
- **`NFKMLXZeroDCEPlus`** — Zero-DCE++, the authors' own successor: depthwise-separable convolutions,
  one shared curve reused across all eight iterations, and an estimator that runs at a twelfth of the
  resolution and lifts its curve map back. At reference parity against the released weights.
- **`NFKMLXMODNet`** — real trimap-free portrait matting: the three-branch (semantic / detail /
  fusion) MODNet, run through the matting backend (portrait → foreground + alpha).
- **`NFKMLXYOLO`** — real object detection: an anchor-free YOLO with box decode and non-max
  suppression, returning `NFKDetection`s (label, confidence, normalized box) under
  `NFKOutputDetections`. Build with class names via the `backendWith…labels:` factory. This is
  YOLOv8.
- **`NFKMLXYOLOGenerations`** — the generations after v8, under the same detection contract: YOLOv9,
  YOLOv10, YOLO11, YOLOv12 and YOLO26, every released size of each. One graph interpreter builds them
  all from the reference's own layer rows, so a release is a `NFKMLXYOLORelease` case rather than a
  separate port. YOLOv10 and YOLO26 predict from a one-to-one branch and need no suppression. At
  reference parity against ultralytics on all twenty-six released checkpoints. The licence is
  ultralytics' AGPL-3.0; `NFKMLXRTDetr` and `NFKMLXRFDetr` are the licence-clean detectors.
- **`NFKMLXSegFormer`** — real semantic segmentation: the SegFormer MiT transformer + all-MLP head,
  run through the module backend, emitting a grayscale class-label map under `NFKOutputImage`.
- **`NFKMLXSwinIR`** — real transformer super-resolution: SwinIR with true shifted-window attention
  and pixel-shuffle upsampling, run through the module backend (low-res → high-res image). Every
  released SR checkpoint has a variant: classical ×2/×3/×4/×8, lightweight ×2/×3/×4, and the two
  real-world ×4 models (nearest-neighbor tail; the large one with the three-convolution residual).
- **`NFKMLXHAT`** — HAT, the Hybrid Attention Transformer, SwinIR's successor for super-resolution:
  the same shifted-window attention with a channel-attention convolution branch inside every block and
  an overlapping cross-attention block closing every group, whose keys and values reach a wider window
  than the queries. Run through the module backend (low-res → ×4 image). HAT-L and Real-HAT-GAN are
  presets (`NFKMLXHATVariant`), both at reference parity against XPixelGroup's own `hat_arch.py`.
- **`NFKMLXColorizer`** / **`NFKMLXSiggraphColorizer`** — real colorization (ECCV-16 and
  SIGGRAPH-17): predict ab chroma from the L channel in CIELAB space and recombine with the original
  luminance, run through the module backend (grayscale photo → color photo); the SIGGRAPH model also
  takes user color hints. The converter loads the reference releases directly.
- **`NFKMLXDDColor`** — modern automatic colorization (DDColor): a ConvNeXt-L encoder, a
  spectral-normalized U-Net decoder, and 100 learned color queries whose attention maps become the two
  chroma channels, run through the module backend (grayscale photo → color photo). The three released
  checkpoints each have a variant (`.modelscope`, `.paper`, `.artistic`), at reference parity against
  the authors' own DDColor. The caller's lightness is kept at full resolution, so luminance is
  preserved exactly.
- **`NFKMLXPose`** — real top-down pose estimation (SimpleBaseline): a residual backbone and a
  deconvolution head produce joint heatmaps, decoded to `NFKKeypoint`s (name, normalized position,
  confidence) under `NFKOutputPose`. Build with joint names via the `backendWith…jointNames:` factory.
- **`NFKMLXVitPose`** — real top-down pose estimation (ViTPose, Apache-2.0): a plain ViT backbone and a
  small decoding head produce joint heatmaps, refined by the DARK distribution-aware decode into
  `NFKKeypoint`s under `NFKOutputPose`. Both released decoders are built (`.baseSimple` upsamples and
  convolves once; `.base` carries SimpleBaseline's transposed-convolution head), at reference parity
  against transformers' `VitPoseForPoseEstimation`. Build with joint names via the
  `backendWith…jointNames:` factory, or from a release directory, which reads its own `config.json`.
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
- **`NFKMLXVJEPA2`** — V-JEPA 2 (Meta, MIT), a self-supervised video encoder: a video (`NFKInputVideo`)
  or an image (`NFKInputImage`) becomes a mean-pooled feature embedding under `NFKOutputEmbedding`, for
  retrieval or as a video encoder for a vision-language model, and a classification release also ranks
  its classes under `NFKOutputClassifications` (Something-Something v2 or Diving48). A ViT with a 3D
  tubelet patch embedding and 3D rotary attention, no class token and no learned position table; the
  classifiers add an attentive pooler. Build with `NFKMLXVJEPA2.backend(directoryURL:)` from any
  `facebook/vjepa2-*` release (ViT-L, ViT-H, or ViT-g), whose `config.json` supplies the geometry, or
  download with `backend(repo:revision:cacheDirectoryURL:)`; at reference parity against transformers'
  own `VJEPA2Model` and `VJEPA2ForVideoClassification`. A probe on your own classes trains on the device
  the way the authors evaluate the encoder: `network(directoryURL:labels:)`, `fineTune`, then
  `save(_:toDirectoryURL:)`, which the same factory loads.
- **`NFKMLXCosmosTokenizer`** — the Cosmos Tokenizer (NVIDIA, Open Model License): all ten released
  image and video tokenizers (`nvidia/Cosmos-0.1-Tokenizer-*`), continuous (a 16-channel latent) or
  discrete (FSQ tokens from a 64,000-entry codebook), at 8× or 16× spatial and 4× or 8× temporal
  compression. An image (`NFKInputImage`) or, for a video variant, a clip (`NFKInputVideo`) is encoded and
  reconstructed under `NFKOutputImage` / `NFKOutputVideo`; the latent or token grid itself comes from
  `NFKMLXCosmosTokenizer`'s `codeForImage:error:` / `codeForFrames:error:` and decodes back through
  `framesForCode:error:`. Build with `backendWithVariant:weightsURL:error:` over the release's
  `autoencoder.jit` (read directly, no conversion), or register every variant under
  `cosmos-tokenizer-<variant>`; at reference parity against NVIDIA's own modules on every variant, and
  fine-tunable on a consumer's own footage with the reference's post-training objective.
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
- **`NFKMLXQwen4Exp`** — the Qwen4-Exp decoder, which Qwen3.8-Flash-Next is the released 180B
  instance of: the hybrid family's recurrence and gated attention, carried over a residual stream
  held four times over by hyper-connections, with a per-layer embedding over hashed n-grams, a
  query-sparse-attention indexer choosing what each query may see, and 512 experts.
- **`NFKMLXHybridLanguage`** — the Qwen3.5 / 3.6 / 3.8 decoder: a gated delta-rule recurrence in three
  of every four layers (a fixed-size state instead of a growing cache) with gated full attention in the
  fourth. At reference parity on the released Qwen3.5-4B, layer by layer; the 27B is accounted for by
  shape against its checkpoint headers.
- **`NFKMLXDeepSeek`** — the DeepSeek V4 and V4.1 decoders: multi-head latent attention over a mixture of
  experts, hyper-connections, the compressor and sparse indexer, and the release's fp8 / fp4 block-scaled
  storage decoded exactly. V4.1 adds a shared compressed cache read by layers that own no compressor,
  candidate block selection, an n-gram memory of 384 million rows a layer, an image tower, and the
  DSpark draft stack with its speculative loop. `NFKMLXDeepSeek.backend(directoryURL:)` generates from
  a release directory, one token a step through `NFKMLXDeepSeekCache`, which carries the sliding
  window, the shared compressed cache, the index keys, a compressor's unfinished group and the n-gram
  id history by absolute position. Both versions are measured against their releases' own inference
  code at a tiny configuration, decode included; a load computes in bf16, the release's own dtype,
  and matches that code bit for bit: every decoder layer, every buffer a decode step carries, V4.1's
  draft stack and image tower, and V4's overlapping compressor and indexer.
  V4.1 Flash's weights decode to 1.39 TiB of bf16 parameters, so a plain load is refused with the
  shortfall and the checkpoint is verified structurally. Passing `paging: .all` holds the routed
  experts and the n-gram tables as the release stores them, decoding an expert as the router reaches
  it and a table row as it is looked up. Paging the experts alone takes the fit figure from 1423.0
  GiB to 679.4 GiB, paging both takes it to 502.0 GiB, and mapping them out of the release rather
  than reading them in takes the decoder to 17.7 GiB. The release's own draft stack also verifies at
  decode, so a greedy run keeps the proposals that match the decoder's own argmax and produces the
  same tokens. `quantizesActivations` adds the rounding the release's inference code applies, and
  `computesInFloat32` opts out of bf16 at twice the bytes a step reads. A long prompt prefills in
  chunks, and a release carrying an image tower accepts `NFKInputImage`.
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
  released weights end to end. The same port runs RT-DETRv2 (`rtdetr-v2-r18vd` / `-r34vd` / `-r50vd` /
  `-r101vd`), which changes only the decoder's deformable sampling: `decoderOffsetScale` scales the
  learned offsets and `NFKMLXRTDetrSamplingMethod` chooses bilinear or nearest-cell gathering. Every
  released v2 configuration repeats its RT-DETR namesake's geometry and its sampling settings, so the four
  v2 presets are their v1 counterparts and the releases differ only in their trained weights, each at
  reference parity against transformers' `RTDetrV2ForObjectDetection`.
- **`NFKMLXRFDetr`** — real object detection under Apache-2.0 (RF-DETR nano / small / medium / base / large, Roboflow; `NFKMLXRFDetrVariant`): a windowed
  DINOv2 backbone, a C2f / RepVGG projector, two-stage Group-DETR query selection, and an LW-DETR
  deformable decoder; no non-max suppression. `rf-detr`; at parity on the released weights end to end.
  `loadWeights` converts the original Roboflow naming on device (and splits the fused self-attention
  projection), so the released file loads directly.
- **`NFKMLXTableTransformer`** — table detection and table-structure recognition under MIT (Table
  Transformer, Microsoft), a vanilla DETR with a ResNet-18 backbone; no non-max suppression.
  `backend(directoryURL:)` reads the release's `config.json` for the geometry and the class names and
  its `preprocessor_config.json` for the input size, so one code path serves the detection release and
  the v1.0 and v1.1 structure releases. A table image in yields the table with its rows, columns, and
  headers under `NFKOutputDetections`. All five releases are at reference parity against transformers'
  `TableTransformerForObjectDetection` on the released weights. Retargets to your own document classes on
  the device with the authors' DETR objective: `network(directoryURL:labels:)`, `fineTune`, then
  `save(_:toDirectoryURL:release:)`, which the same factory loads.
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
- **`NFKMLXLTX2TransformerNet`** — the LTX-2 audio-video transformer: one 22B transformer denoises a
  video latent and an audio latent together, so a generated clip carries its own sound. Six attentions
  a block (video and audio self-attention, each stream over its own text, and the two cross-modal
  directions), an across-heads RMS query/key norm, per-head sigmoid gates, and a split rotary whose
  positions are each patch's midpoint in seconds and pixels. At reference parity against diffusers in
  BOTH released arrangements (LTX-2.3's and LTX-2.5's three switches), and held to the released
  LTX-2.3 headers by shape (4186 tensors, 0 missing / mismatched / unaccounted). The two VAEs, the
  Gemma-4 text front end, the vocoder and the pipeline are the remaining stages.
- **`NFKMLXQwenImagePipeline`** — Qwen-Image 2.1 text-to-image: a 7.1B block-causal DiT, a
  vision-language text encoder, and a single-frame autoencoder. The caption and the image latents share
  one sequence, read causally, while each image block stays bidirectional within itself, and the text
  half modulates from timestep zero so its keys and values are the same at every denoising step. The
  text encoder is the Qwen3-VL decoder this package already runs, read one layer before its final
  normalization; the autoencoder is the Wan 2.2 residual VAE specialized to one frame. Every stage is
  at reference parity against diffusers on its own, and the glue against diffusers' own pipeline. The
  weights are under the Qwen Research License, which is non-commercial.
- **`NFKMLXWanAnimate`** — the Wan 2.2 Animate 2 denoising transformer
  (`NFKMLXWanAnimateNet`): the Wan block with an image cross-attention branch, and an in-context
  reference mechanism built on `NFKMLXWanAnimateKVCache` — one pass stores the reference latents'
  keys and values, each denoising pass attends over them per frame. The released 14B weights exceed a
  workstation, so the arithmetic is measured at a tiny configuration and the release is held to the
  module by shape.
- **`NFKMLXWanPipeline`** — Wan text-to-video: the Wan DiT (`NFKMLXWanTransformerNet`), the streaming
  3-D causal VAE with its per-convolution feature cache (`NFKMLXWanVideoVAENet`, the 2.1 and 2.2 paths),
  a umT5 prompt, and the released UniPC sampler (`NFKMLXUniPCScheduler`).
- **`NFKMLXSD3Pipeline`** — Stable Diffusion 3 / 3.5 text-to-image: the MMDiT dual-stream joint-attention
  transformer (`NFKMLXSD3TransformerNet`, at reference parity against diffusers; the SD3.5 RMS q/k norm
  and MMDiT-X dual attention exercised), the SD3 autoencoder (`NFKMLXSDAutoencoder`, quant convolutions
  kept), a CLIP-L + CLIP-G + T5-XXL text context, and the rectified-flow sampler with classifier-free
  guidance. Presets `.sd3Medium` / `.sd35Medium` / `.sd35Large`; the released sizes held to the module
  by shape (SD3.5-large 1227, SD3.5-medium 909 tensors, 0 missing / mismatched / unaccounted).
- **`NFKMLXFluxPipeline`** — FLUX.1 text-to-image: the double- and single-stream transformer
  (`NFKMLXFluxTransformerNet`, at reference parity against diffusers; axial rotary, guidance embedding),
  the FLUX autoencoder (`NFKMLXSDAutoencoder`, `.flux`), a CLIP-L pooled + T5-XXL text context, and the
  rectified-flow sampler over a packed latent. Presets `.dev` (guidance-distilled) / `.schnell`
  (four-step); the released sizes held to the module by shape (FLUX.1 [schnell] 1156, FLUX.1 [dev] 1160
  tensors, 0 missing / mismatched / unaccounted). The released FLUX.1 [schnell] transformer is also at
  numeric parity against diffusers at the bfloat16 it ships in (velocity 0.99998).
- **`NFKMLXFlux2`** — FLUX.2 [klein] text-to-image end to end, a prompt string in and an image out,
  with editing against reference images and inpainting under a mask (`inpaint`, and
  `inpaintImage:mask:prompt:negativePrompt:strength:seed:error:` from Objective-C). FLUX.2 [klein] 9B KV
  caches its references' keys and values after the first step (`cachesReferences`). A release too
  large to hold whole is staged, the text encoder released before the transformer loads
  (`NFKMLXResidency`), which is how the 9B sizes run on a 32 GB machine.
  The transformer moves its modulation onto the model, uses a SwiGLU feed-forward behind one fused
  projection, fuses the single-stream block's attention and MLP into one projection each way, and
  rotates over four axes (velocity 0.9999999999999934). The autoencoder is the shared
  `NFKMLXSDAutoencoder` at 32 latent channels, and `NFKMLXFlux2LatentCodec` carries what FLUX.2 puts
  between it and the transformer: a 2×2 patch folded into the channel axis and a BatchNorm's running
  statistics in place of a scalar scale and shift (decode 0.9999999999998801). The conditioning is
  THREE intermediate layers of a Qwen3 concatenated per token, over a right-padded sequence whose
  attention mask is load-bearing (1.0000000000000013). The prompt runs through the release's own chat
  template, which appends an empty think block at `enable_thinking=False` (text and ids exact on three
  prompts). The sigma schedule uses FLUX.2's empirical shift, which depends on the step count as well
  as the sequence length. The 32B [dev] size is gated, so the end-to-end path is klein's. Its
  text front end ships: [dev] conditions on Mistral-Small 3, carried as
  `NFKMLXLanguageConfiguration.mistralSmall3` and measured at conditioning cosine 1.0. The 9B sizes
  read `NFKMLXFlux2Configuration.klein9B`, held to the released headers by shape.
- **`NFKMLXFlux`** — the end-to-end FLUX.1 text-to-image path, a prompt string in and an image out.
  It assembles the transformer, the autoencoder, and the two text encoders (`NFKMLXFluxTextEncoder`:
  CLIP-L for the pooled projection, T5-XXL for the sequence) from a diffusers release directory, and
  `image(forPrompt:)` runs the text encoding, the sampler, and the decode. The text front end is at
  reference parity against transformers' `CLIPTextModel` and `T5EncoderModel` (CLIP-L pooled 0.99997,
  T5-XXL sequence 0.9995). A full generation loads the 24 GB transformer beside the encoders, so it runs
  on a machine that can hold it; the text encoding alone fits more widely.
- **`NFKMLXSD3ControlNetPipeline`** — Stable Diffusion 3 ControlNet: a partial MMDiT
  (`NFKMLXSD3ControlNetNet`, at reference parity against diffusers' `SD3ControlNetModel`) that steers a
  generation with a spatial control image, emitting per-block residuals the base `NFKMLXSD3TransformerNet`
  injects. Both released shapes are configurable: the InstantX dual-stream ControlNets (Canny, pose,
  tile, presets `.instantXMedium`) and Stability's official SD3.5-large 8B single-stream ControlNets
  (Blur, Canny, Depth, `.stabilitySD35Large`). The pipeline runs the ControlNet once per classifier-free
  guidance branch.
- **`NFKMLXFluxControlNetPipeline`** — FLUX.1 ControlNet: a partial FLUX transformer
  (`NFKMLXFluxControlNetNet`, at reference parity against diffusers' `FluxControlNetModel`) emitting
  double- and single-block residuals the base `NFKMLXFluxTransformerNet` injects. The union control-type
  embedding (`.unionPro`) and the `input_hint_block` full-resolution-image pyramid are both built; a
  single-control ControlNet is `.single`.
- **`NFKMLXSAM3`** — SAM 3 and SAM 3.1: every instance a worded prompt names, segmented, at
  released-weight parity. `NFKMLXSAM3VisionNet` is a 32-layer 2-D-rotary ViT under a windowed
  schedule with an FPN neck reading its one output map at four scales; `NFKMLXSAM3TextNet` is a
  causal CLIP text tower and the projection that carries its tokens to the detector's width; and
  `NFKMLXSAM3DetectorNet` is the DETR encoder, the 200-query decoder with its presence token, the
  scoring head, and the mask decoder. `NFKMLXSAM3ImageModel.detect(image:tokens:valid:)` chains them
  and returns masks, boxes, per-query logits, and a presence logit. Box prompts and video tracking
  are not ported, so there is no backend yet.
- **`NFKMLXSAM2`** — SAM 2 and SAM 2.1 promptable segmentation and video tracking. The Hiera image
  encoder (tiny, small, base_plus, large), the prompt encoder and mask decoder, and the video memory
  path, assembled by `NFKMLXSAM2TrackerNet`: one released checkpoint loads whole, and
  `track(image:frameIndex:points:session:)` follows a clicked object across a clip through
  `NFKMLXSAM2TrackerSession`. Registered as `sam2`, which segments a single plate from a click under
  `NFKSAMPointKey`. `NFKMLXSAM2Release` selects 2.0 or 2.1; 2.1 adds an occlusion embedding and a
  temporal encoding on the object pointers.
- **`NFKMLXDemucs`** / **`NFKMLXHTDemucs`** — real four-stem music separation: the Demucs v2 time-domain
  U-Net (`demucs`) and the v4 Hybrid Transformer Demucs (`htdemucs`; the six-stem `htdemucs-6s`; the
  fine-tuned four-checkpoint bag through `backend(fineTunedWeightsURLs:)`), a spectrogram branch and a
  waveform branch joined by a cross-transformer; all at parity on the released weights.
- **`NFKMLXMarian`** — OPUS-MT translation, one Helsinki-NLP release per language pair (or per target
  group, named with a `>>xxx<<` marker): a 6 + 6 Marian transformer with a source and a target
  SentencePiece model, built from a release directory, a repo, or a pair of language tags
  (`backend(sourceLanguage:targetLanguage:cacheDirectoryURL:)` names `Helsinki-NLP/opus-mt-<s>-<t>`);
  `opus-mt`; at reference parity against transformers' `MarianMTModel` (tokens, greedy and 4-beam
  outputs exact on en-de).
- **`NFKMLXM2M100`** — M2M-100 many-to-many translation over 100 languages (`.m418M`, `.m1_2B`) and its
  SMaLL-100 distillation (`.small100`), the source language detected when a request omits it; `m2m100`
  and `small100`; at reference parity against transformers' `M2M100ForConditionalGeneration` (tokens,
  greedy and 5-beam outputs exact on 418M).
- **`NFKMLXMADLAD`** — MADLAD-400 3B-MT, Google's T5 translator over 400+ languages named by a `<2xx>`
  marker, loaded at float32 or bfloat16 (`half`); `madlad400-3b-mt`; at reference parity against
  transformers' `T5ForConditionalGeneration` (tokens, greedy and 4-beam outputs exact). All three read
  `NFKParameterSourceLanguage` / `NFKParameterTargetLanguage`, tune the decode through
  `NFKMLXTranslationParameterKey`, fine-tune with LoRA on the decoder, and `NFKMLXTranslationProvider`
  answers the core's `translation` capability with M2M-100 when its release is cached.
- **`NFKMLXTranslateGemma`** — TranslateGemma (4B, 12B, 27B; Gemma terms), Gemma 3 fine-tuned for
  translation and driven by the release's own chat template, rendered in Swift with the language table
  read from `chat_template.jinja`; the shipped Gemma 3 model underneath, greedy decoding, LoRA on the
  decoder; `translategemma`; at reference parity against transformers' `Gemma3ForConditionalGeneration`
  (template ids, logits, and the greedy translation exact on the 4B).
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
- **`NFKMLXBigVGAN`** — BigVGAN v2 (`bigvgan-v2-24khz`, nvidia/bigvgan_v2_24khz_100band_256x, MIT), an
  anti-aliased SnakeBeta vocoder (SnakeBeta periodic activations and a kaiser-sinc up/down filter around
  each one). `callAsFunction` is the generator (mel → waveform) a TTS or restoration chain calls;
  `NFKMLXBigVGANBackend` runs copy-synthesis (audio → the released mel front end → generator → waveform).
  At reference parity against BigVGAN's own generator, the mel → waveform cosine 0.9999997.
- **`NFKMLXMimi`** — Mimi (`mimi`, kyutai/mimi, Kyutai, CC-BY-4.0), a transformer-in-codec neural audio
  codec: a SEANet encoder/decoder with a RoPE Transformer on each side and a split residual vector
  quantizer (one semantic codebook beside 31 acoustic ones). 24 kHz audio in, 12.5 Hz discrete codes,
  audio back. `encode` returns the per-codebook token streams (codebook 0 semantic), `decode`
  reconstructs, and `NFKMLXMimiBackend` runs the round trip. At reference parity against transformers'
  own `MimiModel`, every seam 0.9999999–1.0 with the 32 codebook codes matching exactly.
- **`NFKMLXBasicPitch`** — music transcription (`basic-pitch`): a recording in, notes out. A nine-octave
  constant-Q front end, harmonic stacking, and three small convolutional heads score a pitch contour, a
  note activation, and an onset per frame; note creation turns them into notes with a pitch-bend curve
  each. The result is an `NFKMIDISequence` under `NFKOutputMIDI`, which writes a Standard MIDI File a DAW
  opens. 35,736 parameters, at reference parity against Spotify's own released graph.
- **`NFKMLXAllInOne`** — music structure analysis (`allin1`): a track in, its parts out. Eleven blocks
  of dilated neighborhood attention across time and across the four HT Demucs stems score, per frame, a
  beat, a downbeat, a section boundary, and which of ten functional labels is playing. The sections come
  back as `NFKAudioSegment`s under `NFKOutputSegments`, the beats as `NFKMusicBeat`s under
  `NFKOutputBeats` with their position in the bar, and the tempo under `NFKOutputTempo`.
  `NFKMLXBarTracker` decodes the beats with the bar-pointer model madmom uses. At reference parity on
  the released Harmonix weights, seam by seam and through the post-processing.
- **`NFKMLXHFTTransformer`** — piano transcription (`hft-transformer`): the accuracy counterpart to
  Basic Pitch. A convolutional stem, a transformer attending across frequency that turns mel bins into
  88 note queries, and a second attending across time. Onset, offset, multi-pitch, and velocity at two
  levels; 5.5M parameters, MIT, at reference parity on the released MAESTRO weights.
- **`NFKMLXMuScriptor`** — multi-instrument transcription (`muscriptor`): a mixture in, one MIDI track
  per instrument out. Five seconds of mel spectrogram condition a causal decoder that writes an MT3
  event stream, which becomes notes with their programs. Three released sizes, at reference parity on
  the released medium weights. The weights are CC BY-NC 4.0 behind a gated repository: accept the
  license on the model page and supply a token (`NFKHFHub.accessToken` or `HF_TOKEN`).
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
- **`NFKMLXApollo`** — Apollo (`apollo`, JusperLee, **CC-by-SA-4.0** code and weights), music
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
- **`NFKMLXParakeet`** — Parakeet-TDT 0.6B v2 (NVIDIA, CC-by-4.0), a second speech recognizer beside
  Whisper: a FastConformer encoder and a token-and-duration transducer, greedy TDT decoding, a
  timestamp per token under `NFKOutputSegments`. `backend(directoryURL:)` reads an unpacked `.nemo`;
  at reference parity against NeMo (tokens and timestamps exact).
- **`NFKMLXGraniteSpeech`** — Granite Speech 3.3-2b (IBM, Apache-2.0), the first speech language model:
  a Conformer acoustic encoder, a BLIP-2 Q-former projector, and a dense Granite decoder that generates
  the transcription with the audio embeddings scattered into the prompt. The released audio LoRA adapter
  is folded into the decoder on load. `NFKMLXGraniteSpeech.backend(directoryURL:)`
  (`graniteSpeechBackendWithDirectoryURL:error:`) reads `NFKInputAudio` and an instruction under
  `NFKInputPrompt`. At reference parity against transformers' `GraniteSpeechForConditionalGeneration`:
  the released 2b matches by shape across all 937 base tensors and reproduces the float32 logits and
  greedy continuation, and the backend transcribes the validation clip.
- **`NFKMLXVoxtral`** — Voxtral-Mini 3B (Mistral, Apache-2.0), a second speech language model: the
  Whisper large-v3 encoder (reused), a two-linear projector, and a Llama decoder (reused).
  `NFKMLXVoxtral.backend(directoryURL:)` (`voxtralBackendWithDirectoryURL:error:`) reads `NFKInputAudio`
  and a language code under `NFKInputPrompt`; the tokenizer is Mistral's tekken. At reference parity
  against transformers' `VoxtralForConditionalGeneration`: the released 3B matches by shape across all
  761 base tensors and reproduces the float32 logits and greedy continuation, and the backend
  transcribes the validation clip.
- **`NFKMLXCanary`** — Canary-1B-v2 (NVIDIA NeMo, CC-by-4.0), a multitask speech model that transcribes
  and translates: the biased FastConformer encoder (reused from Parakeet) and a Transformer attention
  encoder-decoder that generates from a task prompt of control tokens.
  `NFKMLXCanary.backend(directoryURL:)` (`backendWithDirectoryURL:error:`) reads `NFKInputAudio` and a
  language code or a `src>tgt` pair under `NFKInputPrompt`; the tokenizer is the release's Metaspace
  BPE. At reference parity against NeMo's `EncDecMultiTaskModel`: the released model matches by shape
  across all 1475 base tensors and reproduces the encoder and decoder seams, and the backend transcribes
  the validation clip exactly.
- **`NFKMLXVideoBackend`** — the first backend that produces video: an `NFKVideoAsset` in, every frame
  through a whole-sequence transform, a new clip out through `NFKMLXVideoFile` (AVFoundation).
  `NFKMLXRIFE.clipBackend` doubles a clip's frame rate and `NFKMLXVideoSR.clipBackend` upscales one.
- Customizing a model on device — `NFKMLXTrainer` runs supervised and zero-reference fine-tuning
  with clipping, checkpoints, and early stop; `NFKMLXLoRA` adapts attention blocks and merges the
  result back into plain weights; `NFKMLXCLIPProbe` trains a classifier over frozen CLIP embeddings;
  recipes ship for Zero-DCE, SegFormer's decode head, and Whisper. The Zero-DCE and SegFormer losses
  are at reference parity; the Whisper objective is not yet measured against a reference. A
  fine-tuned file loads through the model's ordinary `weightsURL:` factory.
- **`NFKMLXDiffusionBackend`** — a bring-your-own MLX diffusion model, for the iterative-sampler shape
  the single-forward backends cannot express. Supply `encode`, `denoise`, `decode`, and a scheduler;
  the backend runs the denoise loop with per-step progress and cancellation. No source latent runs
  text-to-image, a source latent runs image-to-image (`NFKParameterStrength`), and a source latent
  plus a mask runs inpainting. Reference pipelines register by name (upscale, depth, inpaint,
  **controlnet**); `NFKDDIMScheduler` and `NFKLCMScheduler` (few-step) ship, and `NFKDiffusionScheduler`
  is the seam for other samplers. ControlNet and LCM need no full SD reimplementation — LCM is a
  scheduler swap, ControlNet is a `denoise` closure over a control map (`NFKInputControl` →
  `conditioning["control"]`); the UNet is supplied by your `denoise` or a dynamically linked SD engine.

The image backends share `NFKMLXImageBridge`, which converts a `CGImage` or an `MTLTexture`
to and from `MLXArray` in either direction, preserving alpha — so a Metal render pipeline hands
textures straight in and gets textures back.

### Dynamic backend discovery (optional engines)

`NFKDynamicBackend` (core) activates a heavier engine only when its classes are linked into your
build, with no build dependency on it. Your engine's adapter conforms to `NFKDynamicBackendProvider`
and InferKit resolves it by name at runtime (`NSClassFromString`) — absent when unlinked, with no link
error. Built-in capabilities light up when you link a companion:

- **`stable-diffusion`** — linking **InferKitMLX** ships `NFKStableDiffusionProvider`, so
  `NFKDynamicBackend.stableDiffusionBackend()` returns the bundled SD backend.
- **`transcription`** — InferKitMLX also ships `NFKMLXWhisperProvider` (a native whisper.cpp can override).
- **`text-generation`** — linking **InferKitFoundationModels** ships `NFKFoundationModelsProvider` for
  on-device LLM.
- **`controlnet`** — no shipped default; bring a ControlNet engine and register its provider.

Without the companion, the capability is simply unavailable. Model weights are downloaded at runtime
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
